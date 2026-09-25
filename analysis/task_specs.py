"""Structural gap (-) vs missing value (NaN) for overview tables.

``-`` is an expected hole: the (family/group, task) combo is not applicable
(e.g. ``*:two_pole`` on a ``pair: false`` task, or SAE on a task that disables
it). ``NaN`` means the combo should have been computed but the dump has no
value.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

GAP_DISPLAY = "-"
MISSING_DISPLAY = "NaN"

_FAMILIES = (
    "gradiend",
    "actiend",
    "actiend_ridge",
    "sae",
    "caa",
    "cga",
    "caga",
    "agiend",
)
_Spec = Dict[str, Any]
_SpecKey = Tuple[str, str]


def _iter_results_json(runs_root: Path) -> Iterable[Path]:
    runs_root = Path(runs_root)
    seen: set[Path] = set()
    for pattern in ("*/*/results.json", "*/results.json"):
        for path in sorted(runs_root.glob(pattern)):
            parts = path.parts
            model_dir = parts[-3] if len(parts) >= 3 else ""
            if str(model_dir).endswith("_old"):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


@lru_cache(maxsize=None)
def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=None)
def _load_task_cfg(task_id: str) -> Dict[str, Any]:
    if not task_id:
        return {}
    try:
        from study.config import _load_task_yaml

        return dict(_load_task_yaml(str(task_id)))
    except Exception:
        return _load_yaml(CONFIGS / "tasks" / f"{task_id}.yaml")


@lru_cache(maxsize=None)
def _load_suite_cfg(suite_id: Optional[str]) -> Dict[str, Any]:
    if not suite_id:
        return {}
    return _load_yaml(CONFIGS / "suites" / f"{suite_id}.yaml")


def _sae_tags_from_suite(suite_cfg: Mapping[str, Any]) -> Optional[Set[str]]:
    methods = suite_cfg.get("methods") or {}
    sae = methods.get("sae")
    if sae is False:
        return set()
    if isinstance(sae, (list, tuple)):
        return {str(x) for x in sae}
    return None


# Families that are on unless a suite switches them off. CGA is the exception:
# it is an opt-in ablation arm, so an absent ``methods.cga`` key means off, not
# on (``core`` never lists it) -- defaulting it on would put empty CGA columns
# into every core task's tables.
_OPT_IN_FAMILIES = ("actiend_ridge", "cga", "caga", "agiend")


def _enabled_from_yaml(task_cfg: Mapping[str, Any], suite_cfg: Mapping[str, Any]) -> Set[str]:
    suite_methods = suite_cfg.get("methods") or {}
    enabled = {
        k
        for k in _FAMILIES
        if k not in _OPT_IN_FAMILIES and suite_methods.get(k, True) is not False
    }
    enabled |= {k for k in _OPT_IN_FAMILIES if suite_methods.get(k)}
    if not suite_cfg:
        enabled = {k for k in _FAMILIES if k not in _OPT_IN_FAMILIES}
    for key, val in (task_cfg.get("methods") or {}).items():
        name = str(key).lower()
        if name not in _FAMILIES:
            continue
        if val is False:
            enabled.discard(name)
        else:
            enabled.add(name)
    # Ridge is a causal-only decoder fitted over an ACTIEND checkpoint, enabled
    # by the suite (``methods.actiend_ridge``).  Mirror
    # ``study.deep_pipeline._default_enabled`` so missing ridge results render
    # as missing (NaN), not as a structural gap (-).
    if "actiend" not in enabled:
        enabled.discard("actiend_ridge")
    return enabled


def spec_for_task(
    task: str,
    *,
    suite: Optional[str] = None,
    model: str = "",
    results: Optional[Mapping[str, Any]] = None,
    results_path: str = "",
) -> _Spec:
    """Ablation / enabled-method spec for one (model, task)."""
    payload = dict(results) if results else {}
    cfg = dict(payload.get("config") or {})
    parts = Path(results_path).parts if results_path else ()
    model_out = str(
        payload.get("model") or cfg.get("model_key") or model or (parts[-3] if len(parts) >= 3 else "")
    )
    task_out = str(payload.get("task") or cfg.get("task_id") or task or (parts[-2] if len(parts) >= 2 else ""))
    suite_out = str(payload.get("suite") or cfg.get("suite_id") or suite or "core")

    task_cfg = _load_task_cfg(task_out)
    suite_cfg = _load_suite_cfg(suite_out)
    suite_abl = dict(suite_cfg.get("ablations") or {})
    task_abl = dict(task_cfg.get("ablations") or {})
    run_abl = dict(cfg.get("ablations") or {})
    abl = {**suite_abl, **task_abl, **run_abl}

    # Applicability comes from task + suite YAML, not from this dump's
    # ``enabled_methods`` (a SAE-only retry would otherwise mark CAA as a gap).
    enabled = _enabled_from_yaml(task_cfg, suite_cfg)

    sae_tags = _sae_tags_from_suite(suite_cfg)
    return {
        "model": model_out,
        "task": task_out,
        "suite": suite_out,
        "pair": bool(abl.get("pair", True)),
        "one_pole": bool(abl.get("one_pole", True)),
        "enabled_methods": enabled,
        "sae_tags": sae_tags,
    }


def spec_from_results(results: Mapping[str, Any], *, results_path: str = "") -> _Spec:
    return spec_for_task("", results=results, results_path=results_path)


def collect_task_specs(runs_root: Path) -> Dict[_SpecKey, _Spec]:
    out: Dict[_SpecKey, _Spec] = {}
    canonical: set[_SpecKey] = set()
    for path in _iter_results_json(Path(runs_root)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        spec = spec_from_results(payload, results_path=str(path))
        key = (str(spec.get("model") or ""), str(spec.get("task") or ""))
        if not key[0] or not key[1]:
            continue
        parts = path.parts
        path_is_canonical = len(parts) >= 3 and parts[-2] == key[1] and parts[-3] == key[0]
        if key in out and key in canonical and not path_is_canonical:
            continue
        out[key] = spec
        if path_is_canonical:
            canonical.add(key)
    return out


# pre_prediction is captured one token before the filled prediction span (see
# caa_eval.py), so on a one-pole-only task (``pair: false``) its rival class is
# the CF-duplicated row from ``expand_one_pole_cf_rows`` — same left context up
# to that point, hence an identical activation under causal attention. Any
# roc_auc_other / class_exclusivity computed there is tautological (chance,
# not signal; see study/stages/sae.py). Not a gap on multi-class tasks, whose
# rival classes are genuinely different text.
_SAE_PRE_TAUTOLOGICAL_ONE_POLE_METRICS = frozenset(
    {"roc_auc_other", "class_exclusivity", "detection_score"}
)


# Backends whose method ids share the GRADIEND shape ``{backend}[:{pair}]:{cls}``
# and are bucketed by training pole. CGA (Contrastive Gradient Addition) is in
# here because a CGA checkpoint *is* a GRADIEND checkpoint -- one direction in
# the same scoped weight space -- whose weights were computed in closed form
# rather than trained; each CGA variant carries its own backend id so the
# tensor-norm ablation never merges into the headline CGA column.
GRADIEND_LIKE_BACKENDS: frozenset = frozenset(
    {"gradiend", "actiend", "actiend_ridge", "actiend_pre", "cga", "cga_tensor_norm",
     "caga", "agiend"}
)

def group_is_applicable(group: str, spec: Mapping[str, Any], *, metric: str = "") -> bool:
    """Whether a headline method_group is in-scope for this task."""
    enabled = spec.get("enabled_methods") or set()
    backend, _, rest = str(group).partition(":")
    backend = backend.lower()

    if backend in GRADIEND_LIKE_BACKENDS:
        if backend == "actiend_pre":
            if enabled and "actiend_pre" not in enabled:
                return False
        elif backend.startswith("cga"):
            # Every CGA variant is enabled by the single ``cga`` family flag.
            if enabled and "cga" not in enabled:
                return False
        elif enabled and backend not in enabled:
            return False
        if rest == "two_pole":
            return bool(spec.get("pair", True))
        if rest == "one_pole":
            return bool(spec.get("one_pole", True))
        return True

    if backend in {"sae", "sae_pre"}:
        if enabled and "sae" not in enabled:
            return False
        if (
            backend == "sae_pre"
            and metric in _SAE_PRE_TAUTOLOGICAL_ONE_POLE_METRICS
            and not spec.get("pair", True)
        ):
            return False
        tags = spec.get("sae_tags")
        if rest == "joint":
            return bool(tags) and "joint" in tags
        if rest == "k1":
            return tags is None or "k1" in tags or "all_k1" in tags
        if rest == "kstar":
            return tags is None or "kstar" in tags
        return True

    if backend == "caa":
        if enabled and "caa" not in enabled:
            return False
        if rest == "two_pole":
            return bool(spec.get("pair", True))
        if rest == "one_pole":
            return bool(spec.get("one_pole", True))
        return True

    return True


def family_is_applicable(family: str, spec: Mapping[str, Any]) -> bool:
    enabled = spec.get("enabled_methods") or set()
    if not enabled:
        return True
    fam = str(family).lower()
    if fam == "sae_pre" and "sae" in {str(x).lower() for x in enabled}:
        return True
    # actiend_pre is opt-in — only expected when present in enabled_methods.
    return fam in {str(x).lower() for x in enabled}


def resolve_spec(
    specs: Optional[Mapping[_SpecKey, _Spec]],
    *,
    model: str,
    task: str,
) -> _Spec:
    if specs:
        hit = specs.get((str(model), str(task)))
        if hit:
            return hit
        # Model-agnostic fallback when the caller keyed by task only.
        for (mod, tsk), spec in specs.items():
            if tsk == str(task) and (not model or mod == str(model)):
                return spec
    return spec_for_task(str(task), model=str(model))
