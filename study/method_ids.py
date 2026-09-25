"""Stable method-id helpers for the study deep pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def sorted_pair(a: str, b: str) -> Tuple[str, str]:
    left, right = sorted([str(a), str(b)])
    return left, right


def pair_key(a: str, b: str) -> str:
    left, right = sorted_pair(a, b)
    return f"{left}-{right}"


def public_split_tag(split_mode: str) -> Optional[str]:
    """Map training split mode → method-id suffix.

    Suite / training / on-disk use ``tensors``. Method ids use the same token
    (``actiend:positive:tensors``), not legacy ``:all`` — ``:all`` was confusing
    because none-split also trains over all sites. Readers still accept ``:all``
    as a legacy alias (see ``results_schema.none_vs_tensor_pairs``).
    """
    mode = str(split_mode or "none").strip().lower()
    if not mode or mode == "none":
        return None
    if mode in {"tensors", "by_tensor", "tensor", "all"}:
        return "tensors"
    return mode


# Canonical method-id suffix for GradiendSplit.by_tensor() aggregate rows.
TENSORS_SPLIT_TAG = "tensors"


def pair_trainer_id(backend: str, a: str, b: str, *, split_mode: str = "none") -> str:
    base = f"{backend}:{pair_key(a, b)}"
    tag = public_split_tag(split_mode)
    if tag:
        return f"{base}:{tag}"
    return base


def pair_feature_id(backend: str, a: str, b: str, cls: str, *, split_mode: str = "none") -> str:
    """``actiend:asian-white:asian`` (optional ``:tensors`` for by_tensor)."""
    key = pair_key(a, b)
    tag = public_split_tag(split_mode)
    if tag:
        return f"{backend}:{key}:{tag}:{cls}"
    return f"{backend}:{key}:{cls}"




def is_pair_key(token: str) -> bool:
    """True for unordered pair keys like ``asian-white`` (not ``L11`` / ``tok_all``)."""
    s = str(token)
    if "-" not in s:
        return False
    left, right = s.split("-", 1)
    return bool(left) and bool(right) and not left.startswith("L") and "tok_" not in s


def _tok_suffix(
    token_selector: Optional[str] = None,
    activation_gate: Optional[str] = None,
) -> Optional[str]:
    if not token_selector and not activation_gate:
        return None
    tok = token_selector or "all"
    if activation_gate:
        return f"tok_{tok}_gate_{activation_gate}"
    return f"tok_{tok}"


def feature_or_causal_id(
    backend: str,
    cls: str,
    raw: Optional[dict] = None,
    *,
    split_mode: str = "none",
    layer: Optional[str] = None,
    token_selector: Optional[str] = None,
    activation_gate: Optional[str] = None,
) -> str:
    """Encoder / causal method id for this trainer × class.

    Pair: ``gradiend:asian-white:asian``,
    ``actiend:asian-white:asian:tok_all_gate_encoder_direction``,
    ``actiend:asian-white:tensors:asian:tok_all_gate_encoder_direction``.
    One-pole keeps ``gradiend:{cls}`` / ``actiend:{cls}:tok_*``.
    """
    tok = _tok_suffix(token_selector, activation_gate)
    pair = raw.get("pair") if isinstance(raw, dict) else None
    is_pair = (
        str((raw or {}).get("ablation") or "") == "pair"
        and isinstance(pair, (list, tuple))
        and len(pair) == 2
    )
    if is_pair:
        a, b = pair[0], pair[1]
        if layer and str(layer) != "tensors":
            base = f"{backend}:{pair_key(a, b)}:{cls}:{layer}"
            return f"{base}_{tok}" if tok else base
        enc = pair_feature_id(
            backend,
            a,
            b,
            cls,
            split_mode="tensors" if str(layer) == "tensors" else split_mode,
        )
        return f"{enc}:{tok}" if tok else enc
    if str(layer) == "tensors" or str(split_mode) == "tensors":
        if tok:
            return f"{backend}:{cls}:tensors_{tok}"
        return f"{backend}:{cls}:tensors"
    if tok or layer:
        part = tok or ""
        if layer:
            part = f"{layer}_{tok}" if tok else str(layer)
        if part:
            return f"{backend}:{cls}:{part}"
    return f"{backend}:{cls}"


def onepole_feature_id(backend: str, cls: str, *, split_mode: str = "none") -> str:
    """``actiend:asian`` / ``actiend:asian:tensors`` (by_tensor aggregate)."""
    tag = public_split_tag(split_mode)
    if tag:
        return f"{backend}:{cls}:{tag}"
    return f"{backend}:{cls}"


def artifact_dirname(backend: str, *, kind: str, key: str, split_mode: str = "none") -> str:
    """Filesystem-safe artifact folder name (keeps training split token, e.g. ``tensors``)."""
    parts = [backend, kind, key]
    mode = str(split_mode or "none").strip().lower()
    if mode and mode != "none":
        # Prefer stable on-disk token ``tensors`` even when suite lists ``all``.
        if mode in {"all", "by_tensor", "tensor"}:
            mode = "tensors"
        parts.append(mode)
    return "__".join(parts)


def label_tokens_from_config(
    classes: Sequence[str],
    causal_cfg: dict,
) -> dict:
    """Map class → surface token for CAA / token scoring.

    Only emit entries with an explicit ``causal.target_tokens`` mapping.
    Circuit one-pole tasks (IOI / induction) fill per-row ``label`` names; defaulting
    to the class id (``IO`` / ``SUBJECT``) would poison prediction-fill CAA.
    """
    raw = causal_cfg.get("target_tokens") or {}
    out = {}
    for cls in classes:
        cls = str(cls)
        toks = raw.get(cls)
        if isinstance(toks, (list, tuple)) and toks:
            out[cls] = str(toks[0])
        elif isinstance(toks, str) and toks:
            out[cls] = toks
    return out


SAE_BACKEND_PRE = "sae_pre"


def normalize_sae_method_id(method_id: str) -> str:
    """Map legacy ``sae:{cls}:k1_pre`` ids to ``sae_pre:{cls}:k1`` for analysis."""
    mid = str(method_id or "")
    parts = mid.split(":")
    if len(parts) < 2 or parts[0] != "sae":
        return mid
    if parts[1] == "joint_pre":
        rest = parts[2:] if len(parts) > 2 else []
        return ":".join([SAE_BACKEND_PRE, "joint", *rest])
    if len(parts) == 2 and parts[1] == "joint_pre":
        return f"{SAE_BACKEND_PRE}:joint"
    if len(parts) >= 3:
        tail = parts[-1]
        if tail.endswith("_pre_tok_prediction"):
            base = tail[: -len("_pre_tok_prediction")]
            return f"{SAE_BACKEND_PRE}:{':'.join(parts[1:-1])}:{base}_tok_prediction"
        if tail.endswith("_pre"):
            base = tail[: -len("_pre")]
            return f"{SAE_BACKEND_PRE}:{':'.join(parts[1:-1])}:{base}"
    return mid


# Headline estimator plus its ablations. ``plain`` is the CAA-matched estimator
# (one global normalization). ``tensor_norm`` rescales each base-model tensor's
# block to unit norm before the global normalization, testing whether raw
# gradient-magnitude heterogeneity across tensors -- not the gradient signal
# itself -- is what limits the plain estimator.
CGA_VARIANTS: Tuple[str, ...] = ("plain", "tensor_norm")
CGA_METHOD_ABLATIONS: Tuple[str, ...] = ("layers",)

CGA_DEFAULT_VARIANT = "plain"


def resolve_cga_variants(requested: Optional[Sequence[str]]) -> List[str]:
    """Validate/normalize estimator entries in ``methods.cga``.

    Non-estimator method ablations (currently ``layers``) share this list but do
    not create another checkpoint/backend.  They are consumed by the encode and
    causal stages after the selected direction has been fitted or reloaded.
    """
    if not requested:
        return [CGA_DEFAULT_VARIANT]
    out: List[str] = []
    for item in requested:
        name = str(item).strip().lower()
        if not name:
            continue
        if name in CGA_METHOD_ABLATIONS:
            continue
        if name not in CGA_VARIANTS:
            raise ValueError(
                f"Unknown CGA variant {name!r}. Choose from: "
                f"{', '.join(CGA_VARIANTS + CGA_METHOD_ABLATIONS)}"
            )
        if name not in out:
            out.append(name)
    return out or [CGA_DEFAULT_VARIANT]


def cga_backend_id(variant: str) -> str:
    """Backend id (and therefore method-id prefix) for a CGA variant.

    Each variant gets its own backend id -- ``cga`` for the headline estimator,
    ``cga_tensor_norm`` for the ablation -- rather than a suffix inside the class
    token. Everything downstream (artifact dir, ``{backend}:{pair}:{cls}`` ids,
    analysis family, LaTeX column) then separates them automatically, and the
    ablation cannot be averaged into the headline CGA number by accident.
    """
    name = str(variant).strip().lower()
    if name not in CGA_VARIANTS:
        raise ValueError(
            f"Unknown CGA variant {name!r}. Choose from: {', '.join(CGA_VARIANTS)}"
        )
    if name == CGA_DEFAULT_VARIANT:
        return "cga"
    return f"cga_{name}"


def cga_backend_ids() -> Tuple[str, ...]:
    """Every backend id CGA can emit, headline first."""
    return tuple(cga_backend_id(v) for v in CGA_VARIANTS)


def cga_variant_from_backend(backend: str) -> str:
    """Inverse of :func:`cga_backend_id`."""
    name = str(backend).strip().lower()
    if name == "cga":
        return CGA_DEFAULT_VARIANT
    if name.startswith("cga_"):
        variant = name[len("cga_"):]
        if variant in CGA_VARIANTS:
            return variant
    raise ValueError(f"{backend!r} is not a CGA backend id")
