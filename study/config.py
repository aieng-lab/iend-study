"""Config loader with override order: defaults → task → model → suite → CLI."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
            and not isinstance(value, (str, bytes))
        ):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


_TRAIN_ABLATION_ALIASES = {
    "pair": "pair",
    "pairs": "pair",
    "two_pole": "pair",
    "two-pole": "pair",
    "one_pole": "one_pole",
    "one-pole": "one_pole",
    "onepole": "one_pole",
}


def normalize_train_ablations(values: Any) -> Optional[frozenset]:
    """``--train-ablations`` value(s) -> ``{"pair","one_pole"}`` subset, or ``None`` (= all).

    Torch-free so the Slurm table builder and the train stage share one parser.
    Raises on an unknown name (a typo must not silently mean "everything").
    """
    if not values:
        return None
    items = [values] if isinstance(values, str) else list(values)
    out = set()
    for item in items:
        for part in str(item).replace(",", " ").split():
            key = part.strip().lower()
            if key not in _TRAIN_ABLATION_ALIASES:
                raise ValueError(
                    f"Unknown --train-ablations value {part!r}; choose from pair, one_pole"
                )
            out.add(_TRAIN_ABLATION_ALIASES[key])
    return frozenset(out) or None


def list_models() -> List[str]:
    return sorted(p.stem for p in (CONFIGS / "models").glob("*.yaml"))


def list_tasks(*, include_hidden: bool = False) -> List[str]:
    """Task ids from ``configs/tasks/*.yaml``.

    Tasks with ``study_hidden: true`` (e.g. deprecated ``gender_en_pre``) are
    omitted unless ``include_hidden=True``. Explicit ``--task gender_en_pre``
    still loads via ``load_study_config``.

    The canonical study order puts multi-class tasks first, then binary tasks,
    then tasks that enable only the one-pole ablation. Task ids are alphabetical
    within each group. Slurm launchers use this order when ``TASKS=all``.
    """
    entries: List[tuple[str, Dict[str, Any]]] = []
    for path in sorted((CONFIGS / "tasks").glob("*.yaml")):
        raw = _load_task_yaml(path.stem)
        if not include_hidden and bool(raw.get("study_hidden")):
            continue
        entries.append((path.stem, raw))

    def order_key(entry: tuple[str, Dict[str, Any]]) -> tuple[int, str]:
        task_id, raw = entry
        ablations = raw.get("ablations") or {}
        one_pole_only = bool(ablations.get("one_pole")) and ablations.get("pair") is False
        if one_pole_only:
            group = 2
        elif len(raw.get("classes") or []) > 2:
            group = 0
        else:
            group = 1
        return group, task_id

    return [task_id for task_id, _ in sorted(entries, key=order_key)]


def list_suites() -> List[str]:
    return sorted(p.stem for p in (CONFIGS / "suites").glob("*.yaml"))


def resolve_suite_default(model_key: str, model_cfg: Mapping[str, Any], suite_cfgs: Mapping[str, Any]) -> str:
    if model_cfg.get("suite_default"):
        return str(model_cfg["suite_default"])
    full = suite_cfgs.get("full") or {}
    full_models = set(full.get("models") or [])
    if model_key in full_models:
        return "full"
    return "core"


def _load_task_yaml(task: str) -> Dict[str, Any]:
    """Load a task YAML, resolving ``extends:`` chains (base ← overlay)."""
    seen: List[str] = []
    overlays: List[Dict[str, Any]] = []
    current = str(task)
    while True:
        if current in seen:
            raise ValueError(
                f"Task config extends cycle: {' -> '.join(seen + [current])}"
            )
        seen.append(current)
        path = CONFIGS / "tasks" / f"{current}.yaml"
        cfg = _load_yaml(path)
        overlays.append(cfg)
        parent = cfg.get("extends")
        if not parent:
            break
        current = str(parent).strip()
        if not current:
            break
    # overlays are [leaf, ..., base]; merge base first, then overlays.
    merged: Dict[str, Any] = {}
    for cfg in reversed(overlays):
        overlay = {k: v for k, v in cfg.items() if k != "extends"}
        merged = _deep_merge(merged, overlay)
    return merged


@dataclass
class StudyConfig:
    model_key: str
    task_id: str
    suite_id: str
    raw: Dict[str, Any] = field(default_factory=dict)
    cli_training_keys: frozenset = field(default_factory=frozenset)
    """Names of ``training`` keys that came from the command line.

    Needed to resolve learning rates correctly: an explicitly typed CLI value
    must not be silently discarded by a YAML default, but a *more specific* CLI
    flag (``--lr-actiend``) must still beat a less specific one (``--lr``).
    Both arrive as ordinary ``training`` keys, so provenance is the only thing
    that distinguishes them once merged.
    """

    @property
    def hf_model(self) -> str:
        return str(self.raw["model"]["hf_model"])

    @property
    def output_dir(self) -> Path:
        extra = str(self.raw.get("output_subdir") or "").strip().replace("\\", "/")
        base = ROOT / "runs" / self.model_key
        if extra:
            parts = [p for p in Path(extra).parts if p not in ("", ".", "..")]
            if parts:
                return base.joinpath(*parts) / self.task_id
        return base / self.task_id

    @property
    def training(self) -> Dict[str, Any]:
        return dict(self.raw.get("training") or {})

    @property
    def task(self) -> Dict[str, Any]:
        return dict(self.raw.get("task") or {})

    @property
    def suite(self) -> Dict[str, Any]:
        return dict(self.raw.get("suite") or {})

    @property
    def model(self) -> Dict[str, Any]:
        return dict(self.raw.get("model") or {})

    @property
    def suitability(self) -> Dict[str, Any]:
        return dict(self.raw.get("suitability") or {})

    def config_hash(self) -> str:
        # Resume / logging switches are not scientific protocol. Including
        # ``skip_existing`` here used to make a normal resume look like a suite
        # change (SKIP_EXISTING=0 vs 1), defeating config-aware cache policy.
        hashable = dict(self.raw)
        for key in (
            "skip_existing",
            "refresh_causal",
            "force_causal",
            "fail_fast",
            "rerun_collapsed",
            "rerun_orphaned_artifacts",
            "save_modified_models",
            "train_splits",
        ):
            hashable.pop(key, None)
        # ``run_deep_pipeline`` records operator switches under ``cli`` for
        # provenance. A targeted split refresh is an execution filter, not a
        # different scientific protocol, so it must not alter the config hash.
        cli = dict(hashable.get("cli") or {})
        cli.pop("train_splits", None)
        # Likewise a pair/one-pole slice is an execution filter (parallel GPUs),
        # not a different scientific protocol.
        cli.pop("train_ablations", None)
        # A causal-only id whitelist is likewise an execution filter.
        # (and the family filter / marker of that second pass with it)
        if cli.pop("causal_only", None):
            cli.pop("methods", None)
        cli.pop("_paired_pass", None)
        cli.pop("no_rerun_orphaned_artifacts", None)
        # Searching the LR is an execution mode: the chosen rate is recorded per artifact (done.json).
        cli.pop("tune_lr", None)
        if cli:
            hashable["cli"] = cli
        else:
            hashable.pop("cli", None)
        blob = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def learning_rate(
        self,
        cli_lr: Optional[float] = None,
        *,
        backend: Optional[str] = None,
    ) -> float:
        """Resolve training LR.

        Priority (specificity first, then provenance — the command line always
        beats a YAML default at the same level of specificity):
        1. CLI backend-specific ``training.learning_rate_{gradiend|actiend}``
        2. ``cli_lr`` / CLI shared ``training.learning_rate`` (global ``--lr``)
        3. YAML backend-specific ``training.learning_rate_{gradiend|actiend}``
        4. merged ``training.learning_rate`` (defaults → task → model default)
        5. ``1e-5``

        Ranks 1 and 3 used to be a single rule above ``cli_lr``, which silently
        discarded every ``--lr`` for gradiend and actiend, since
        ``configs/defaults.yaml`` sets a backend key for both.

        Note: ``model.learning_rate`` is injected into ``training`` only when the
        task did not set one (``setdefault`` in ``load_study_config``). Do not
        re-read ``model.learning_rate`` here — that previously ignored task LRs.
        """
        t = self.training or {}
        be = str(backend or "").strip().lower()
        lr_backend = "actiend" if be == "actiend_pre" else be
        backend_keys: list[str] = []
        if lr_backend in {"gradiend", "actiend", "agiend"}:
            backend_keys = [
                f"learning_rate_{lr_backend}",
                # Legacy alias used in some task YAMLs (e.g. race.yaml).
                f"{lr_backend}_learning_rate",
            ]
        for key in backend_keys:
            if key in self.cli_training_keys and t.get(key) is not None:
                return float(t[key])
        if cli_lr is not None:
            return float(cli_lr)
        if "learning_rate" in self.cli_training_keys and t.get("learning_rate") is not None:
            return float(t["learning_rate"])
        for key in backend_keys:
            if key in t and t[key] is not None:
                return float(t[key])
        return float(t.get("learning_rate", 1e-5))

    def learning_rate_decoder(
        self,
        cli_lr_decoder: Optional[float] = None,
        *,
        backend: Optional[str] = None,
    ) -> Optional[float]:
        """Resolve the optional decoupled decoder LR, mirroring ``learning_rate``.

        Returns ``None`` when nothing is configured, which is the package
        default and reproduces the historical single-LR optimizer exactly.

        Priority mirrors ``learning_rate`` exactly — specificity first, then
        provenance:
        1. CLI backend-specific ``training.learning_rate_decoder_{gradiend|actiend|agiend}``
        2. ``cli_lr_decoder`` / CLI shared ``training.learning_rate_decoder``
        3. YAML backend-specific ``training.learning_rate_decoder_{gradiend|actiend|agiend}``
        4. merged ``training.learning_rate_decoder``
        5. ``None``
        """
        t = self.training or {}
        be = str(backend or "").strip().lower()
        lr_backend = "actiend" if be == "actiend_pre" else be
        key = (
            f"learning_rate_decoder_{lr_backend}"
            if lr_backend in {"gradiend", "actiend", "agiend"}
            else None
        )
        # Keep the "auto" sentinel as a string (the package resolves the reachability
        # lift); only float a numeric value. float("auto") crashes — see CLAUDE.md
        # "ALWAYS FAIL FAST" / the decoder-LR ablation.
        def _lr(v: Any) -> Any:
            return v if isinstance(v, str) else float(v)

        if key and key in self.cli_training_keys and t.get(key) is not None:
            return _lr(t[key])
        if cli_lr_decoder is not None:
            return _lr(cli_lr_decoder)
        if (
            "learning_rate_decoder" in self.cli_training_keys
            and t.get("learning_rate_decoder") is not None
        ):
            return _lr(t["learning_rate_decoder"])
        if key and key in t and t[key] is not None:
            return _lr(t[key])
        shared = t.get("learning_rate_decoder")
        return _lr(shared) if shared is not None else None

    def max_steps(self, *, backend: Optional[str] = None) -> Optional[int]:
        """Per-backend training step budget.

        ACTIEND is schedule-robust (running_rms) and converges in far fewer
        steps than GRADIEND, which is schedule-sensitive (equal lr*steps does
        not preserve it). ``training.max_steps_{gradiend|actiend}`` overrides the
        shared ``training.max_steps`` for that backend. Priority mirrors
        ``learning_rate``: a CLI ``--max-steps`` (shared) still wins, so an
        explicit sweep is never silently clamped by the backend default.
        """
        t = self.training or {}
        be = str(backend or "").strip().lower()
        ms_backend = "actiend" if be == "actiend_pre" else be
        key = f"max_steps_{ms_backend}" if ms_backend in {"gradiend", "actiend"} else None
        if "max_steps" in self.cli_training_keys and t.get("max_steps") is not None:
            return int(t["max_steps"])
        if key and t.get(key) is not None:
            return int(t[key])
        ms = t.get("max_steps")
        return int(ms) if ms is not None else None

    def eval_steps(self, *, backend: Optional[str] = None) -> Optional[int]:
        """Resolve a backend-specific evaluation interval when configured.

        Evaluation advances RNG state in the trainer, so changing its interval
        can change an otherwise identical training trajectory.  Keep the same
        specificity rules as ``max_steps``: an explicit shared CLI value wins,
        then a backend-specific YAML value, then the shared YAML fallback.
        """
        t = self.training or {}
        be = str(backend or "").strip().lower()
        es_backend = "actiend" if be == "actiend_pre" else be
        key = f"eval_steps_{es_backend}" if es_backend in {"gradiend", "actiend"} else None
        if "eval_steps" in self.cli_training_keys and t.get("eval_steps") is not None:
            return int(t["eval_steps"])
        if key and t.get(key) is not None:
            return int(t[key])
        es = t.get("eval_steps")
        return int(es) if es is not None else None


def load_study_config(
    *,
    model: str,
    task: str,
    suite: Optional[str] = None,
    cli_overrides: Optional[Mapping[str, Any]] = None,
) -> StudyConfig:
    defaults = _load_yaml(CONFIGS / "defaults.yaml")
    model_cfg = _load_yaml(CONFIGS / "models" / f"{model}.yaml")
    task_cfg = _load_task_yaml(task)

    suite_cfgs = {s: _load_yaml(CONFIGS / "suites" / f"{s}.yaml") for s in list_suites()}
    suite_id = suite or resolve_suite_default(model, model_cfg, suite_cfgs)
    if suite_id not in suite_cfgs:
        raise ValueError(f"Unknown suite {suite_id!r}. Known: {sorted(suite_cfgs)}")
    suite_cfg = suite_cfgs[suite_id]

    # Order: defaults → task → model (LR etc.) → suite → CLI
    merged = _deep_merge(defaults, {"task": task_cfg})
    # Fold task training into top-level training
    if "training" in task_cfg:
        merged["training"] = _deep_merge(merged.get("training") or {}, task_cfg["training"])
    if "ablations" in task_cfg:
        merged["ablations"] = _deep_merge(merged.get("ablations") or {}, task_cfg["ablations"])
    if "causal" in task_cfg:
        merged["causal"] = _deep_merge(merged.get("causal") or {}, task_cfg["causal"])

    merged = _deep_merge(merged, {"model": model_cfg})
    # Model-specific backend training defaults (for example, smaller learning
    # rates on Pythia) override the shared/task defaults before suite and CLI.
    if "training" in model_cfg:
        merged["training"] = _deep_merge(
            merged.get("training") or {}, model_cfg["training"]
        )
    # Model learning_rate is only a default when the task did not set one.
    if "learning_rate" in model_cfg:
        training = merged.setdefault("training", {})
        training.setdefault("learning_rate", model_cfg["learning_rate"])

    merged = _deep_merge(merged, {"suite": suite_cfg})
    if "ablations" in suite_cfg:
        merged["ablations"] = _deep_merge(merged.get("ablations") or {}, suite_cfg["ablations"])
    if "causal" in suite_cfg:
        merged["causal"] = _deep_merge(merged.get("causal") or {}, suite_cfg["causal"])
    if "training" in suite_cfg:
        merged["training"] = _deep_merge(merged.get("training") or {}, suite_cfg["training"])
    # Task-specific ablation/causal/methods overrides must win over suite defaults
    # (e.g. IOI: pair=false, one_pole=true; race_one_pole: sae/caa off).
    if "ablations" in task_cfg:
        merged["ablations"] = _deep_merge(merged.get("ablations") or {}, task_cfg["ablations"])
    if "causal" in task_cfg:
        merged["causal"] = _deep_merge(merged.get("causal") or {}, task_cfg["causal"])
    if "methods" in task_cfg:
        suite_block = merged.setdefault("suite", {})
        suite_block["methods"] = _deep_merge(
            suite_block.get("methods") or {}, task_cfg["methods"]
        )

    cli_training_keys = frozenset(
        str(key) for key in ((cli_overrides or {}).get("training") or {})
    )
    if cli_overrides:
        merged = _deep_merge(merged, dict(cli_overrides))

    merged["model_key"] = model
    merged["task_id"] = task
    merged["suite_id"] = suite_id
    return StudyConfig(
        model_key=model,
        task_id=task,
        suite_id=suite_id,
        raw=merged,
        cli_training_keys=cli_training_keys,
    )


# Feature-learning / retrieval methods only. Causal + localization are pipeline
# stages (on by default; disable with --skip-causal / --skip-localization).
METHOD_FAMILIES = (
    "gradiend",
    "actiend",
    "actiend_ridge",
    "sae",
    "caa",
    # CGA — Contrastive Gradient Addition: the untrained mean-difference
    # estimator over GRADIEND's gradient signal (``cga_eval.py``).
    "cga",
    # CAGA — Contrastive Activation-Gradient Addition: the untrained mean-diff
    # over the dL/dh (activation-gradient) signal; steers activations
    # (``caga_eval.py``). Opt-in like cga (a suite must list ``methods.caga``).
    "caga",
    # AGIEND — the LEARNED encoder-decoder on the dL/dh signal (activation-gradient),
    # trained like ACTIEND; steers activations. Opt-in. Completes the 3x2
    # signal×estimator grid (act-grad × learned).
    "agiend",
)

PIPELINE_STAGES = (
    "causal",
    "localization",
)


def parse_methods_arg(raw: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Parse CLI ``--methods`` values (space- and/or comma-separated).

    Accepts only :data:`METHOD_FAMILIES`. ``causal`` / ``localization`` are stages,
    not methods — reject them with a pointer to ``--skip-*``.
    Flattens nested lists from argparse ``action=append`` + ``nargs=+``.
    """
    if not raw:
        return None
    flat: List[str] = []

    def _walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for part in item:
                _walk(part)
            return
        flat.append(str(item))

    _walk(raw)
    out: List[str] = []
    for item in flat:
        for part in str(item).split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name in PIPELINE_STAGES:
                raise ValueError(
                    f"{name!r} is a pipeline stage, not a method. "
                    f"It runs by default for selected methods; disable with "
                    f"--skip-{name}. Methods: {', '.join(METHOD_FAMILIES)}"
                )
            if name not in METHOD_FAMILIES:
                raise ValueError(
                    f"Unknown method {name!r}. Choose from: {', '.join(METHOD_FAMILIES)}. "
                    f"Stages (not methods): {', '.join(PIPELINE_STAGES)} "
                    f"(toggle with --skip-causal / --skip-localization)."
                )
            if name not in out:
                out.append(name)
    return out or None


_TORCH_DTYPES = ("float32", "float16", "bfloat16", "float64")


def resolve_torch_dtype(value: Any) -> Any:
    """Resolve a YAML dtype name to a ``torch.dtype``.

    Raises on an unknown name rather than falling back to float32: silently
    training an 8B model in float32 because ``bflaot16`` was misspelled would
    waste the run and look like an out-of-memory problem.
    """
    import torch

    if isinstance(value, torch.dtype):
        return value
    name = str(value).strip().replace("torch.", "")
    if name not in _TORCH_DTYPES:
        raise ValueError(
            f"Unknown training.torch_dtype {value!r}; expected one of "
            f"{', '.join(_TORCH_DTYPES)}"
        )
    return getattr(torch, name)


def training_kwargs_from_config(
    cfg: StudyConfig,
    *,
    cli_lr: Optional[float] = None,
    cli_lr_decoder: Optional[float] = None,
    backend: Optional[str] = None,
    skip_pre_prune: bool = False,
) -> Dict[str, Any]:
    """Map study training block onto GRADIEND TrainingArguments field names.

    Always materializes ``pre_prune_config`` / ``post_prune_config`` from
    ``training.pre_prune`` / ``training.post_prune`` (defaults: topk 0.1 / 0.01).
    Callers strip prune configs for ACTIEND (no pre-prune API).

    ``backend`` selects ``learning_rate_gradiend`` / ``learning_rate_actiend``
    when set; otherwise the shared ``learning_rate``.

    ``skip_pre_prune``: one-pole runs — package stratification needs ≥1 matching
    feature_class in the train frame; skip rather than crash.
    """
    t = cfg.training
    keys = [
        "source",
        "target",
        "train_batch_size",
        "eval_batch_size",
        "max_length",
        "max_seeds",
        "seed",
        "use_cache",
        "fail_on_non_convergence",
        "encoder_eval_max_size",
        "encoder_eval_train_max_size",
        "add_neutral_identity_transitions",
        "add_identity_for_other_classes",
        "activation_site",
        "target_activation_site",
        "actiend_source_site",
        "actiend_target_site",
        "prediction_objective",
        # Decoder-only label-token convention. Absent from every config, so
        # nothing is forwarded: new runs take the package default ("canonical")
        # and reloads follow each artifact's own stamp. Set
        # ``training.label_token_protocol: legacy`` only to reproduce an old run.
        "label_token_protocol",
        "signal",
        "signal_scope",
        "gradiend_exclude_embeddings",
        "convergent_metric",
        "selection_metric",
        "convergent_score_threshold",
        "convergent_mean_by_class_threshold",
        "prefer_convergent_checkpoint",
        # Optimizer choice. Absent from every current config, so this forwards
        # nothing and no existing train-artifact hash shifts; set
        # ``training.optim: sgd`` to run the non-Adam reachability arm
        # (IEND_THEORY_PLAN 2.7).
        "optim",
        "sgd_momentum",
        # Large-model loading. Absent from every current config, so this
        # forwards nothing and no existing artifact hash shifts. The package
        # defaults to float32 and no device_map, which is what every run so far
        # used; an 8B model needs bfloat16 to fit at all.
        "base_model_device_map",
    ]
    out: Dict[str, Any] = {k: t[k] for k in keys if k in t}
    # torch_dtype is a torch.dtype in TrainingArguments but a plain string in
    # YAML, so it needs resolving rather than a straight copy. Imported lazily:
    # this module is imported by tooling that must run without torch.
    dtype_name = t.get("torch_dtype")
    if dtype_name is not None:
        out["torch_dtype"] = resolve_torch_dtype(dtype_name)
    out["learning_rate"] = cfg.learning_rate(cli_lr, backend=backend)
    ms = cfg.max_steps(backend=backend)
    if ms is not None:
        out["max_steps"] = ms
    es = cfg.eval_steps(backend=backend)
    if es is not None:
        out["eval_steps"] = es
    # Only emitted when actually configured, so every existing config produces a
    # byte-identical kwargs dict and no train-artifact hash shifts.
    decoder_lr = cfg.learning_rate_decoder(cli_lr_decoder, backend=backend)
    if decoder_lr is not None:
        out["learning_rate_decoder"] = decoder_lr

    # Crucial study defaults (topk 0.1 / 0.01). Disable with ``pre_prune: false``.
    # Pre-pruning is defined over the alternative gradient even when the
    # training objective consumes both factual and alternative inputs.
    if t.get("pre_prune") is not False and not skip_pre_prune:
        from gradiend import PrePruneConfig

        pre = dict(t.get("pre_prune") or {})
        out["pre_prune_config"] = PrePruneConfig(
            n_samples=int(pre.get("n_samples", 16)),
            topk=pre.get("topk", 0.1),
            source="alternative",
        )
    if t.get("post_prune") is not False:
        from gradiend import PostPruneConfig

        post = dict(t.get("post_prune") or {})
        out["post_prune_config"] = PostPruneConfig(
            topk=post.get("topk", 0.01),
            part=str(post.get("part") or "decoder-weight"),
        )
    return out
