"""Lightweight causal-policy configuration helpers.

The causal runner historically evaluated every ACTIEND token/gate policy on
every invocation.  Suites now select the policies they actually need so a
large-model core run does not pay for small-model ablations.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple


ACTIEND_POLICY_GATED_ALL = "gated_all"
ACTIEND_POLICY_ALL = "all"
ACTIEND_POLICY_PREDICTION = "prediction"

# The resolved ACTIEND causal *default* intervention (token selector × gate). As of
# 2026-09-03 this is ungated all-token (``tok_all``): on matched gpt2-small cells it
# steers better than the gated policy (ACTIEND-CAA -0.035 gated vs +0.006 ungated),
# so it is the honest ACTIEND-at-its-best. The gated policy is retained as the
# ``gated_all`` ablation. ``gate=None`` means no direction gate. (Historically these
# lived in ``sae_eval.py``, which was just a grab-bag eval module and had nothing to
# do with ACTIEND; ``sae_eval`` now re-exports them from here.)
ACTIEND_TOK_DEFAULT = "all"
ACTIEND_GATE_DEFAULT = None

# Preserve the historical behavior for programmatic callers that do not pass
# a policy list. Suite YAMLs opt into their intended subset explicitly.
ACTIEND_CAUSAL_POLICY_DEFAULTS: Tuple[str, ...] = (
    ACTIEND_POLICY_GATED_ALL,
    ACTIEND_POLICY_ALL,
    ACTIEND_POLICY_PREDICTION,
)

_ALIASES = {
    "default": ACTIEND_POLICY_GATED_ALL,
    "gate": ACTIEND_POLICY_GATED_ALL,
    "gated": ACTIEND_POLICY_GATED_ALL,
    "gated_all": ACTIEND_POLICY_GATED_ALL,
    "tok_all_gate_encoder_direction": ACTIEND_POLICY_GATED_ALL,
    "all": ACTIEND_POLICY_ALL,
    "tok_all": ACTIEND_POLICY_ALL,
    "prediction": ACTIEND_POLICY_PREDICTION,
    "tok_prediction": ACTIEND_POLICY_PREDICTION,
}


def normalize_actiend_causal_policies(values: Any) -> Tuple[str, ...]:
    """Return canonical, de-duplicated ACTIEND policy names.

    ``None`` retains the historical three-policy behavior. An explicit empty
    list disables ACTIEND causal evaluation while leaving encoder evaluation
    untouched. Strings are accepted as a convenience for CLI/config overlays.
    Unknown names raise so a typo cannot silently launch the wrong expensive
    rollout.
    """

    if values is None:
        return ACTIEND_CAUSAL_POLICY_DEFAULTS
    raw_values: Iterable[Any]
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = values

    out = []
    for raw in raw_values:
        key = str(raw).strip().lower().replace("-", "_")
        if key not in _ALIASES:
            allowed = ", ".join(ACTIEND_CAUSAL_POLICY_DEFAULTS)
            raise ValueError(
                f"Unknown ACTIEND causal policy {raw!r}; expected one of: {allowed}"
            )
        canonical = _ALIASES[key]
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


def actiend_default_policy_enabled(policies: Any) -> bool:
    """The ACTIEND causal *default* is the ungated all-token policy (``tok_all``,
    gate=None). It is computed whenever ``all`` (ungated) is requested. This is the
    2026-09-03 swap: ungated all-token is now the default (it steers better than the
    gated policy on matched cells), and the gated policy is a named ablation."""
    return ACTIEND_POLICY_ALL in set(normalize_actiend_causal_policies(policies))


def actiend_ablation_policy_specs(policies: Any) -> Tuple[Tuple[str, Any], ...]:
    """``(token_selector, activation_gate)`` for the NON-default ACTIEND policies.

    The default is ungated all-token (handled separately). The remaining requested
    policies are ablations: ``gated_all`` -> ``("all", "encoder_direction")`` (the
    gated policy, id ``tok_all_gate_encoder_direction``) and ``prediction`` ->
    ``("prediction", None)``. Note ``all`` is NOT emitted here (it is the default),
    so there is no id collision. Id strings are unchanged from before the swap, so a
    run launched under the old default/ablation split remains valid.
    """
    selected = set(normalize_actiend_causal_policies(policies))
    out: List[Tuple[str, Any]] = []
    if ACTIEND_POLICY_GATED_ALL in selected:
        out.append(("all", "encoder_direction"))
    if ACTIEND_POLICY_PREDICTION in selected:
        out.append(("prediction", None))
    return tuple(out)


# Back-compat alias: callers that want the ungated-scope ablation specs.


# --- Causal-only id whitelist (execution filter, not part of the config hash) ---
#
# ``primary_only`` (core) keeps causal for ONE validation-selected representation
# per method/class.  The layer-selection appendix and the layer-wise figures need
# the aggregate AND every layer, in the same ids ``full_plus`` writes.  A causal
# rerun with ``primary_only: false`` would also recompute every ablation, so this
# whitelist narrows it to exactly the ids a preset names; everything else is
# treated as already done.
CAUSAL_ONLY_LAYERWISE = "layerwise"
# The comparison the layer-selection strip/paired plots read: for SAE k=1, CAA, CGA
# and CAGA, the all-layer estimate versus the validation-selected best single layer.
CAUSAL_ONLY_PAIRED = "paired"
PAIRED_FAMILIES = ("sae", "caa", "cga", "caga")

_LAYERWISE_PATTERNS: Tuple[str, ...] = (
    # SAE k=1: every physical layer and the all-layer aggregate (``k1`` is the
    # selected-site row that core already stores; a stored id is skipped anyway).
    r"^sae:[^:]+:(?:L\d+_k1|all_k1|k1)$",
    # CAA act_prediction, one-pole (caa:C:...) and pair (caa:A-B:C:...): every
    # layer, the concatenated aggregate and the all-layers aggregate.
    r"^caa:(?:[^:]+:){1,2}(?:L\d+_|all_)?act_prediction$",
)


def causal_only_patterns(
    specs: Any, *, results_path: Any = None
) -> List[str]:
    """Regexes (``re.search``) selecting the method ids a causal-only run computes.

    ``specs`` is one or more preset names or raw regexes.  The ``layerwise``
    preset also names, from the task's stored results, the validation-selected
    best physical layer of each CGA/CAGA cell (their per-layer sweeps are weight
    space grids, so all layers of them would be far too expensive).
    """
    if specs is None:
        return []
    if isinstance(specs, str):
        specs = [specs]
    out: List[str] = []
    for spec in specs:
        spec = str(spec).strip()
        if not spec:
            continue
        if spec == CAUSAL_ONLY_PAIRED or spec.startswith(CAUSAL_ONLY_PAIRED + "@"):
            import re

            ids = _paired_view_ids(results_path)
            if "@" in spec:
                # ``paired@i/n``: the i-th of n interleaved slices of the sorted id list,
                # so n jobs (each on its OWN output subdir, seeded from the same stored
                # results) partition the missing sweeps with no overlap.
                idx_s, _, n_s = spec.split("@", 1)[1].partition("/")
                idx, n = int(idx_s), int(n_s)
                if n < 1 or not 0 <= idx < n:
                    raise ValueError(f"bad paired slice {spec!r} (want paired@i/n, 0<=i<n)")
                ids = ids[idx::n]
            # No ids => every needed sweep is already stored: match nothing.
            out.append(
                "^(?:" + "|".join(re.escape(i) for i in ids) + ")$" if ids else r"(?!)"
            )
        elif spec.startswith(CAUSAL_ONLY_LAYERWISE + "@"):
            out.extend(_layer_slice_patterns(spec.split("@", 1)[1]))
        elif spec == CAUSAL_ONLY_LAYERWISE:
            out.extend(_LAYERWISE_PATTERNS)
            ids = _best_layer_weight_space_ids(results_path)
            if ids:
                import re

                out.append("^(?:" + "|".join(re.escape(i) for i in sorted(ids)) + ")$")
        else:
            out.append(spec)
    return out


def _layer_slice_patterns(bounds: str) -> List[str]:
    """``layerwise@A-B``: SAE k1 + CAA act_prediction for layers A..B only.

    The aggregates (``all_k1``/``k1``, ``act_prediction``/``all_act_prediction``)
    belong to the slice that contains layer 0, so a set of disjoint slices covering
    every layer computes each id exactly once.  Slices must write to DIFFERENT
    output subdirs (concurrent writers to one results.json can drop each other's
    causal entries); merge them afterwards.
    """
    lo_s, _, hi_s = bounds.partition("-")
    lo, hi = int(lo_s), int(hi_s or lo_s)
    if lo < 0 or hi < lo:
        raise ValueError(f"bad layer slice {bounds!r}")
    alt = "|".join(str(n) for n in range(lo, hi + 1))
    out = [
        rf"^sae:[^:]+:L(?:{alt})_k1$",
        rf"^caa:(?:[^:]+:){{1,2}}L(?:{alt})_act_prediction$",
    ]
    if lo == 0:
        out += [
            r"^sae:[^:]+:(?:all_k1|k1)$",
            r"^caa:(?:[^:]+:){1,2}(?:all_)?act_prediction$",
        ]
    return out


def _paired_view_ids(results_path: Any) -> List[str]:
    """Method ids the all-layer vs best-single-layer comparison reads, per stored results.

    Taken from the analysis's own selection (validation Detection), for both views,
    so the ids can never drift from what the plots consume.  Ids that already have a
    stored causal sweep are skipped later by the ordinary skip set.
    """
    if results_path is None:
        return []
    from pathlib import Path

    path = Path(results_path)
    if not path.is_file():
        return []
    from analysis.method_groups import _load_results, collect_group_rows_for_results

    payload = _load_results(path)
    ids: set = set()
    for view in ("all_layer", "best_single_layer"):
        for row in collect_group_rows_for_results(
            payload, results_path=str(path), representation_view=view
        ):
            if str(row.get("backend") or "") not in PAIRED_FAMILIES:
                continue
            for sid in str(row.get("source_methods") or "").split(","):
                if sid.strip():
                    ids.add(sid.strip())
    return sorted(ids)


def _best_layer_weight_space_ids(results_path: Any) -> List[str]:
    """CGA/CAGA ``...:L<n>`` ids the best-single-layer view reads, from stored results."""
    if results_path is None:
        return []
    import re
    from pathlib import Path

    path = Path(results_path)
    if not path.is_file():
        return []
    from analysis.method_groups import _load_results, collect_group_rows_for_results

    payload = _load_results(path)
    ids: set = set()
    for row in collect_group_rows_for_results(
        payload, results_path=str(path), representation_view="best_single_layer"
    ):
        if str(row.get("backend") or "") not in ("cga", "caga"):
            continue
        for sid in str(row.get("source_methods") or "").split(","):
            sid = sid.strip()
            if re.search(r":L\d+$", sid):
                ids.add(sid)
    return sorted(ids)


class SkipUnlessWanted(set):
    """A skip-set that also claims every id the whitelist does not name.

    Drop-in for the ``skip_causal_methods`` set: ``mid in skip`` is True for a
    stored-ok id (as before) OR for any id matching none of ``patterns``.  It is
    always truthy, so the ``if not skip`` short-circuits keep enumerating groups.
    """

    def __init__(self, stored: Iterable[str], patterns: Iterable[str]):
        import re

        super().__init__(str(x) for x in (stored or ()))
        self._wanted = [re.compile(p) for p in patterns]
        if not self._wanted:
            raise ValueError("SkipUnlessWanted needs at least one pattern")

    def __bool__(self) -> bool:
        return True

    def __contains__(self, item: object) -> bool:
        if set.__contains__(self, item):
            return True
        return not any(rx.search(str(item)) for rx in self._wanted)
