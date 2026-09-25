"""Shared activation-site protocol for ACTIEND / CAA / SAE.

PROTOCOL_VERSION bumps invalidate prior encode/causal comparisons that mixed
sites (SAE on unfilled last-token vs ACTIEND/CAA on filled prediction).

Sites
-----
``prediction``
    Mean residual over the filled prediction span (ACTIEND default).
    This is the residual **of** the target token (e.g. ``he``), not the
    residual used to *generate* it.
``pre_prediction`` (alias ``unfilled_prediction``)
    Residual at one token **before** the prediction span. On decoder-only
    causal LMs this is the last-context / generation residual: position
    t-1 cannot attend to the fill at t, so a filled forward and a
    prefix truncated before the fill yield the same H at this site.
    (Bidirectional models, or pooling that includes later tokens such as
    ``mean`` / ``last``, *can* leak the fill.)

One-pole limitation
-------------------
On circuit one-pole tasks (IOI, …) factual and CF share the same masked string.
A true unfilled forward makes H_factual = H_alternative, so
``source=diff`` / F-vs-A activation contrasts are impossible without filling.
``pre_prediction`` still distinguishes F vs A when the *left* context differs
(e.g. gendered names before the pronoun). If F and A share an identical
prefix, causal ``pre_prediction`` residuals collapse even on filled strings.

Rule
----
CAA and SAE encode use the configured ``activation_site``. ACTIEND **training**
may be mixed-site: encoder input = ``activation_site`` (or
``actiend_source_site``), decoder target = ``target_activation_site``.
``activation_site: pre_prediction`` without an explicit target site defaults to
mixed-site ACTIEND (source ``pre_prediction``, target filled ``prediction``).
The study can optionally train method family ``actiend_pre`` (forced mixed
sites, ids ``actiend_pre:…``) when ``training.actiend_pre: true`` — off by
default. Same-site ``pre_prediction`` target is not a pronoun-diff
objective on causal LMs (the fill is at t; t-1 never sees it). SAE headline ids
use backend ``sae`` / ``sae_pre`` for the two select sites. Never reintroduce
``collect_activations`` on raw ``[MASK]`` strings beside a filled CAA/ACTIEND path.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

# Bump when activation geometry or shared-H rules change (invalidates fair tables).
# v3: neutrals scored per non-pad token (not document-mean) for prediction/pre_prediction.
# v4: sampled train/val/test neutrals; drop excluded class tokens; Youden/causal
#     select on val, report on test; ACTIEND scores the same token H as CAA/SAE.
# Mixed-site ACTIEND is method family ``actiend_pre`` (ids ``actiend_pre:…``),
# opt-in via ``training.actiend_pre`` — not the deprecated ``gender_en_pre`` task.
ACTIVATION_PROTOCOL_VERSION = 4

ACTIVATION_SITES: Tuple[str, ...] = (
    "prediction",
    "pre_prediction",
    "mean",
    "last",
)

# Primary fair encode site for ACTIEND ↔ CAA ↔ SAE (config may override).
DEFAULT_ACTIVATION_SITE = "prediction"

# Site aliases accepted in YAML / CLI.
_SITE_ALIASES = {
    "unfilled_prediction": "pre_prediction",
    "pre-prediction": "pre_prediction",
    "pre_fill": "pre_prediction",
    "prefill": "pre_prediction",
    "all": "mean",
}


def normalize_activation_site(site: Optional[str]) -> str:
    raw = str(site or DEFAULT_ACTIVATION_SITE).strip().lower()
    raw = _SITE_ALIASES.get(raw, raw)
    if raw not in ACTIVATION_SITES:
        raise ValueError(
            f"Unknown activation_site={site!r}; expected one of {ACTIVATION_SITES} "
            f"(aliases: {sorted(_SITE_ALIASES)})"
        )
    return raw


def actiend_token_selector(site: Optional[str]) -> Any:
    """Value for ``Signal.activation(token_selector=…)``."""
    s = normalize_activation_site(site)
    if s == "prediction":
        return "prediction"
    if s == "pre_prediction":
        return "pre_prediction"
    if s == "mean":
        return "all"
    if s == "last":
        from caa_eval import last_token_selector

        return last_token_selector
    raise ValueError(s)


def resolve_actiend_train_sites(
    source_site: Optional[str] = None,
    target_site: Optional[str] = None,
) -> Tuple[str, str]:
    """ACTIEND encoder gather vs decoder-target gather.

    ``pre_prediction`` as the source without an explicit target is mixed-site:
    encoder sees H_{t-1}, decoder reconstructs the filled-token ``prediction``
    diff. Same-site last-context target is only used when requested explicitly.
    """
    src = normalize_activation_site(source_site)
    if target_site is None or str(target_site).strip() == "":
        tgt = "prediction" if src == "pre_prediction" else src
    else:
        tgt = normalize_activation_site(target_site)
    return src, tgt


def protocol_metadata(site: Optional[str] = None, *, target_site: Optional[str] = None) -> dict:
    s = normalize_activation_site(site)
    src, tgt = resolve_actiend_train_sites(s, target_site)
    return {
        "activation_protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "activation_site": s,
        "actiend_source_site": src,
        "actiend_target_site": tgt,
        "actiend_mixed_site": src != tgt,
        "shared_h_methods": ["actiend", "caa", "sae"],
        "one_pole_unfilled_note": (
            "True unfilled F vs A collapses on one-pole (identical masked inputs). "
            "Causal pre_prediction also collapses when F and A share the same "
            "left context; use filled prediction, differing prefixes, or pair data."
        ),
        "neutral_unit": (
            "token" if s in {"prediction", "pre_prediction"} else "text"
        ),
        "neutral_split": "config_train_val_test_counts",
        "youden_split": "validation",
        "causal_select_split": "validation",
        "causal_report_split": "test",
    }
