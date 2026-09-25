"""
Cross-method localization: do SAE / ACTIEND / GRADIEND hit the same neurons?

Index spaces differ, so raw index Jaccard is invalid. Comparable units:

* **Residual channels** at layer L (primary / neuron-level bridge)
  - ACTIEND: activation dims at ``activation:transformer.h.L``
  - SAE: top-|W_dec| residual dims of a selected feature at resid-post L
  - GRADIEND: write-out weights (``attn.c_proj`` / ``mlp.c_proj``) mapped to
    residual output coords (HF GPT-2 Conv1D: weight ``(nx, nf)``, channel = col)

* **Coarse bags** (secondary context only): layer / attn-vs-mlp mass profiles

GRADIEND mass in ``c_attn`` / ``c_fc`` is reported but not residual-channel-
comparable.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from sae_eval import sae_decoder_direction

DEFAULT_TOPK: Union[int, float] = 0.01
DEFAULT_RESID_M = 50
DEFAULT_LAYER_MASS_TAU = 0.05
_LAYER_PAT = re.compile(r"(?:^|[./])(?:h|layers)\.(\d+)(?:$|[./])")
_EMBED_PAT = re.compile(r"(?:wte|wpe|embed|embedding|embed_in)", re.IGNORECASE)


def _optional_int_layer(value: Any) -> Optional[int]:
    """Parse a residual layer index; virtual ``all`` is not a layer."""
    if value is None or str(value) == "all":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_ready(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def export_method_topk(
    trainer,
    *,
    topk: Union[int, float] = DEFAULT_TOPK,
    part: str = "decoder-weight",
) -> List[Dict[str, Any]]:
    """Top-k GRADIEND/ACTIEND input dims with decoded base-model metadata."""
    mwg = trainer.get_model()
    rows = mwg.get_topk_feature_metadata(part=part, topk=topk)
    return [dict(r) for r in rows]


def export_resid_channel_rows(
    trainer,
    *,
    part: str = "decoder-weight",
    write_out_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    All AE input dims that map to residual channels, with importance.

    ACTIEND: every ``activation:transformer.h.L`` channel.
    GRADIEND: every kept write-out (``attn/mlp.c_proj``) weight mapped to resid dim.

    Unlike global top-k, this keeps per-layer channel scores so neuron-level
    Jaccard at L6/L11 is not biased by mass elsewhere.
    """
    mwg = trainer.get_model()
    g = mwg.gradiend
    importance = g.get_weight_importance(part=part)
    base_map = g._get_base_global_index_map()
    n = int(importance.numel())
    # Heuristic: activation maps → keep all activation rows; else write-out only.
    if write_out_only is None:
        sample_names = []
        for i in range(min(8, n)):
            try:
                sample_names.append(
                    str(g.decode_base_global_index(int(base_map[i].item())).get("param_name") or "")
                )
            except Exception:
                pass
        write_out_only = not any(s.startswith("activation:") for s in sample_names)

    rows: List[Dict[str, Any]] = []
    for local_index in range(n):
        imp = float(importance[local_index].item())
        if imp <= 0.0:
            continue
        base_global_index = int(base_map[local_index].item())
        decoded = g.decode_base_global_index(base_global_index)
        name = str(decoded.get("param_name") or "")
        row = {
            "rank": len(rows) + 1,
            "part": part,
            "importance": imp,
            "local_input_index": int(local_index),
            "base_global_index": base_global_index,
            **decoded,
        }
        if name.startswith("activation:"):
            if actiend_resid_channel(row) is not None:
                rows.append(row)
            continue
        if write_out_only:
            if write_out_resid_channel(row) is not None:
                rows.append(row)
        else:
            rows.append(row)
    rows.sort(key=lambda r: -float(r["importance"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def parse_layer(param_name: str) -> Optional[str]:
    """Return ``L{n}``, ``embed``, or None."""
    name = str(param_name or "")
    if name.startswith("activation:"):
        name = name[len("activation:") :]
    m = _LAYER_PAT.search(name)
    if m:
        return f"L{int(m.group(1))}"
    if _EMBED_PAT.search(name):
        return "embed"
    return None


def parse_component(param_name: str) -> str:
    """Coarse bag: resid / attn / mlp / embed / other."""
    name = str(param_name or "")
    low = name.lower()
    if name.startswith("activation:") or low.startswith("activation:"):
        return "resid"
    if _EMBED_PAT.search(low):
        return "embed"
    if "attn" in low or "attention" in low:
        return "attn"
    if "mlp" in low or "c_fc" in low:
        return "mlp"
    if "ln_" in low or "norm" in low:
        return "other"
    return "other"


def is_write_out_param(param_name: str) -> bool:
    """True for residual write-out matrices (GPT-2 ``c_proj`` / GPT-NeoX dense)."""
    name = str(param_name or "")
    if name.startswith("activation:"):
        return False
    # GPT-2 Conv1D write-outs
    if re.search(r"(?:attn|mlp)\.c_proj(?:\.weight)?$", name, re.IGNORECASE):
        return True
    # GPT-NeoX / Pythia Linear write-outs
    if re.search(r"attention\.dense(?:\.weight)?$", name, re.IGNORECASE):
        return True
    if re.search(r"mlp\.dense_4h_to_h(?:\.weight)?$", name, re.IGNORECASE):
        return True
    return False


def write_out_resid_channel(row: Dict[str, Any]) -> Optional[int]:
    """
    Residual output channel for a GRADIEND write-out weight.

    HF GPT-2 ``Conv1D`` weight is ``(nx, nf)`` with forward ``x @ W``;
    unravel coords are ``(in_idx, out_idx)`` → residual channel = ``coords[1]``.

    GPT-NeoX ``nn.Linear`` weight is ``(out, in)`` with forward ``x @ W.T``;
    residual channel = ``coords[0]`` (out feature).
    """
    if not is_write_out_param(str(row.get("param_name") or "")):
        return None
    coords = row.get("coords")
    name = str(row.get("param_name") or "")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        if re.search(r"c_proj", name, re.IGNORECASE):
            return int(coords[1])
        return int(coords[0])
    if re.search(r"c_proj", name, re.IGNORECASE):
        col = row.get("param_col")
        if col is not None:
            return int(col)
    else:
        row_idx = row.get("param_row")
        if row_idx is not None:
            return int(row_idx)
    return None


def actiend_resid_channel(row: Dict[str, Any]) -> Optional[int]:
    """Channel index for an ACTIEND activation-site dim."""
    name = str(row.get("param_name") or "")
    if not name.startswith("activation:"):
        return None
    coords = row.get("coords")
    if isinstance(coords, (list, tuple)) and len(coords) >= 1:
        return int(coords[0])
    row_idx = row.get("param_row")
    if row_idx is not None:
        return int(row_idx)
    flat = row.get("flat_index_in_param")
    if flat is not None:
        return int(flat)
    return None


def set_jaccard(a: Iterable[int], b: Iterable[int]) -> Dict[str, float]:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return {"jaccard": 1.0, "overlap_frac": 1.0, "intersection": 0, "union": 0, "size_a": 0, "size_b": 0}
    inter = len(sa & sb)
    union = len(sa | sb)
    min_sz = min(len(sa), len(sb)) or 1
    return {
        "jaccard": float(inter / union) if union else 0.0,
        "overlap_frac": float(inter / min_sz),
        "intersection": float(inter),
        "union": float(union),
        "size_a": float(len(sa)),
        "size_b": float(len(sb)),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of average ranks (no scipy dependency)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom <= 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def _normalize_mass(d: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in d.values()))
    if total <= 0.0:
        return {str(k): 0.0 for k in d}
    return {str(k): float(v) / total for k, v in d.items()}


def bucket_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    resid_m: int = DEFAULT_RESID_M,
) -> Dict[str, Any]:
    """
    Aggregate top-k metadata into coarse bags + per-layer residual channel scores.

    Returns layer/component mass, tensor mass, and residual-channel maps suitable
    for neuron-level cross-method comparison.
    """
    layer_mass: Dict[str, float] = defaultdict(float)
    component_mass: Dict[str, float] = defaultdict(float)
    tensor_mass: Dict[str, float] = defaultdict(float)
    # layer -> channel -> importance (ACTIEND channels + GRADIEND write-out)
    resid_imp: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    resid_sources: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    write_out_mass = 0.0
    non_write_out_mass = 0.0
    n_write_out = 0
    n_actiend = 0

    for row in rows:
        imp = float(row.get("importance") or 0.0)
        if not math.isfinite(imp) or imp < 0.0:
            imp = abs(imp) if math.isfinite(imp) else 0.0
        name = str(row.get("param_name") or "")
        layer = parse_layer(name) or "other"
        comp = parse_component(name)
        layer_mass[layer] += imp
        component_mass[comp] += imp
        tensor_mass[name] += imp

        ch = actiend_resid_channel(row)
        if ch is not None and layer.startswith("L"):
            resid_imp[layer][ch] += imp
            resid_sources[layer]["actiend"] += imp
            n_actiend += 1
            continue

        ch_w = write_out_resid_channel(row)
        if ch_w is not None and layer.startswith("L"):
            resid_imp[layer][ch_w] += imp
            resid_sources[layer]["gradiend_write_out"] += imp
            write_out_mass += imp
            n_write_out += 1
        else:
            non_write_out_mass += imp

    # Per-layer top-m residual channel sets + dense sparse dicts
    resid_topk: Dict[str, List[int]] = {}
    resid_vectors: Dict[str, Dict[str, float]] = {}
    for layer, ch_map in resid_imp.items():
        ranked = sorted(ch_map.items(), key=lambda kv: (-kv[1], kv[0]))
        resid_topk[layer] = [int(c) for c, _ in ranked[: max(1, int(resid_m))]]
        resid_vectors[layer] = {str(int(c)): float(v) for c, v in ch_map.items()}

    total = float(sum(layer_mass.values())) or 1.0
    return {
        "n_rows": len(rows),
        "total_importance": float(sum(layer_mass.values())),
        "layer_mass": _normalize_mass(dict(layer_mass)),
        "layer_mass_raw": {k: float(v) for k, v in layer_mass.items()},
        "component_mass": _normalize_mass(dict(component_mass)),
        "tensor_mass": _normalize_mass(dict(tensor_mass)),
        "resid_channels_topk": resid_topk,
        "resid_channel_importance": resid_vectors,
        "resid_sources": {L: dict(v) for L, v in resid_sources.items()},
        "write_out_mass_frac": float(write_out_mass / total),
        "non_write_out_mass_frac": float(non_write_out_mass / total),
        "n_write_out_rows": int(n_write_out),
        "n_actiend_rows": int(n_actiend),
        "top_layers": [
            L
            for L, _ in sorted(layer_mass.items(), key=lambda kv: -kv[1])[:5]
        ],
    }


def sae_resid_channels_union(
    sae,
    feature_indices: Sequence[int],
    *,
    m: int = DEFAULT_RESID_M,
    layer: Optional[int] = None,
) -> Dict[str, Any]:
    """Union of per-feature top-m |W_dec| channels (not mean-then-topk)."""
    feats = [int(i) for i in feature_indices]
    if not feats:
        return sae_resid_channels(sae, feats, m=m, layer=layer)
    union: set = set()
    per_feat = []
    for fi in feats:
        spec = sae_resid_channels(sae, [fi], m=m, layer=layer)
        ch = [int(x) for x in (spec.get("resid_channels_topk") or [])]
        per_feat.append({"feature": fi, "resid_channels_topk": ch})
        union.update(ch)
    top = sorted(union)
    # Importance = count of features that include the channel in their top-m
    counts: Dict[int, int] = {}
    for row in per_feat:
        for c in row["resid_channels_topk"]:
            counts[c] = counts.get(c, 0) + 1
    imp = {str(c): float(counts[c]) for c in top}
    layer_key = f"L{int(layer)}" if layer is not None else None
    d_model = None
    if per_feat:
        # Recover d_model from a single-feature call
        probe = sae_resid_channels(sae, [feats[0]], m=m, layer=layer)
        d_model = probe.get("d_model")
    return {
        "feature_indices": feats,
        "layer": int(layer) if layer is not None else None,
        "layer_key": layer_key,
        "resid_channels_topk": top,
        "resid_channel_importance": imp,
        "per_feature_topk": per_feat,
        "union": True,
        "d_model": d_model,
    }


def sae_resid_channels(
    sae,
    feature_indices: Sequence[int],
    *,
    m: int = DEFAULT_RESID_M,
    layer: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Map SAE feature decoder directions to residual-channel sets / importance.

    Multiple features: average |W_dec| then take top-m (neuron-level readout bag).
    """
    feats = [int(i) for i in feature_indices]
    if not feats:
        return {
            "feature_indices": [],
            "layer": layer,
            "resid_channels_topk": [],
            "resid_channel_importance": {},
            "decoder_l2": None,
        }
    acc = None
    norms = []
    for fi in feats:
        d = sae_decoder_direction(sae, fi).detach().float().cpu().numpy().ravel()
        norms.append(float(np.linalg.norm(d)))
        mag = np.abs(d)
        acc = mag if acc is None else acc + mag
    assert acc is not None
    acc = acc / float(len(feats))
    order = np.argsort(-acc)
    top = [int(i) for i in order[: max(1, int(m))]]
    # Store sparse importance for all non-negligible dims (or top 4m for size)
    keep_n = max(int(m) * 4, int(m))
    keep = order[:keep_n]
    imp = {str(int(i)): float(acc[i]) for i in keep}
    layer_key = f"L{int(layer)}" if layer is not None else None
    return {
        "feature_indices": feats,
        "layer": int(layer) if layer is not None else None,
        "layer_key": layer_key,
        "resid_channels_topk": top,
        "resid_channel_importance": imp,
        "decoder_l2_mean": float(np.mean(norms)) if norms else None,
        "d_model": int(acc.size),
    }


def _profile_vector(mass: Dict[str, float], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([float(mass.get(k, 0.0)) for k in keys], dtype=np.float64)


def compare_mass_profiles(
    a: Dict[str, float],
    b: Dict[str, float],
    *,
    keys: Optional[Sequence[str]] = None,
    tau: float = DEFAULT_LAYER_MASS_TAU,
) -> Dict[str, Any]:
    """Cosine / Spearman of mass bags + Jaccard of layers above tau."""
    if keys is None:
        keys = sorted(set(a) | set(b))
    va, vb = _profile_vector(a, keys), _profile_vector(b, keys)
    set_a = {k for k in keys if float(a.get(k, 0.0)) >= tau}
    set_b = {k for k in keys if float(b.get(k, 0.0)) >= tau}
    # Encode keys as ints for set_jaccard helper via hash — use string sets directly
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    min_sz = min(len(set_a), len(set_b)) or 1
    return {
        "cosine": _cosine(va, vb),
        "spearman": _spearman(va, vb),
        "keys": list(keys),
        "top_layer_jaccard": float(inter / union) if union else 1.0,
        "top_layer_overlap_frac": float(inter / min_sz),
        "layers_a": sorted(set_a),
        "layers_b": sorted(set_b),
        "tau": float(tau),
    }


def compare_resid_channels(
    a_topk: Sequence[int],
    b_topk: Sequence[int],
    *,
    a_imp: Optional[Dict[str, float]] = None,
    b_imp: Optional[Dict[str, float]] = None,
    d_model: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Neuron-level agreement in residual-channel space.

    Always reports set Jaccard on top-m. When importance maps are given, also
    builds aligned dense vectors (union support or full d_model) for cosine /
    Spearman over channel scores.
    """
    out = set_jaccard(a_topk, b_topk)
    out["topk_a"] = [int(x) for x in a_topk]
    out["topk_b"] = [int(x) for x in b_topk]
    if a_imp is None and b_imp is None:
        return out
    a_imp = a_imp or {}
    b_imp = b_imp or {}
    if d_model is not None and int(d_model) > 0:
        n = int(d_model)
        va = np.zeros(n, dtype=np.float64)
        vb = np.zeros(n, dtype=np.float64)
        for k, v in a_imp.items():
            i = int(k)
            if 0 <= i < n:
                va[i] = float(v)
        for k, v in b_imp.items():
            i = int(k)
            if 0 <= i < n:
                vb[i] = float(v)
    else:
        keys = sorted(set(a_imp) | set(b_imp), key=lambda x: int(x))
        va = np.asarray([float(a_imp.get(k, 0.0)) for k in keys], dtype=np.float64)
        vb = np.asarray([float(b_imp.get(k, 0.0)) for k in keys], dtype=np.float64)
    out["channel_cosine"] = _cosine(va, vb)
    out["channel_spearman"] = _spearman(va, vb)
    out["support_size"] = int(va.size)
    return out


def _layers_of_interest(
    *profiles: Dict[str, Any],
    sae_layers: Optional[Sequence[int]] = None,
    max_layers: int = 6,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for prof in profiles:
        for L, m in (prof.get("layer_mass") or {}).items():
            if str(L).startswith("L"):
                scores[str(L)] += float(m)
        for L in (prof.get("resid_channels_topk") or {}):
            if str(L).startswith("L"):
                scores[str(L)] += 1e-6
    for L in sae_layers or []:
        scores[f"L{int(L)}"] += 1.0
    ranked = sorted(scores.keys(), key=lambda L: (-scores[L], L))
    return ranked[:max_layers] if ranked else [f"L{int(L)}" for L in (sae_layers or [])]


def compare_localization(
    *,
    gradiend_rows: Optional[Sequence[Dict[str, Any]]] = None,
    actiend_rows: Optional[Sequence[Dict[str, Any]]] = None,
    gradiend_resid_rows: Optional[Sequence[Dict[str, Any]]] = None,
    actiend_resid_rows: Optional[Sequence[Dict[str, Any]]] = None,
    sae_by_class: Optional[Dict[str, Dict[str, Any]]] = None,
    resid_m: int = DEFAULT_RESID_M,
    layer_mass_tau: float = DEFAULT_LAYER_MASS_TAU,
    d_model: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build localization profiles and pairwise neuron-level + coarse comparisons.

    ``*_rows`` (global top-k) → coarse layer/component bags.
    ``*_resid_rows`` (all resid-mappable dims) → neuron-level channel bridges.
    If resid rows are omitted, fall back to top-k rows for the neuron bridge.
    """
    grad = bucket_rows(gradiend_rows or [], resid_m=resid_m) if gradiend_rows is not None else None
    act = bucket_rows(actiend_rows or [], resid_m=resid_m) if actiend_rows is not None else None
    grad_n = (
        bucket_rows(gradiend_resid_rows, resid_m=resid_m)
        if gradiend_resid_rows is not None
        else grad
    )
    act_n = (
        bucket_rows(actiend_resid_rows, resid_m=resid_m)
        if actiend_resid_rows is not None
        else act
    )
    sae_by_class = sae_by_class or {}

    sae_layers = []
    for spec in sae_by_class.values():
        if spec.get("variant") == "per_feature_scan":
            continue
        layer_i = _optional_int_layer(spec.get("layer"))
        if layer_i is not None:
            sae_layers.append(layer_i)

    profiles = {
        "gradiend": grad,
        "actiend": act,
        "gradiend_neuron": grad_n,
        "actiend_neuron": act_n,
        "sae_by_class": sae_by_class,
    }

    # ----- coarse (secondary) -----
    coarse: Dict[str, Any] = {}
    if grad and act:
        keys = sorted(set(grad["layer_mass"]) | set(act["layer_mass"]))
        coarse["gradiend_vs_actiend_layer"] = compare_mass_profiles(
            grad["layer_mass"], act["layer_mass"], keys=keys, tau=layer_mass_tau
        )
        ckeys = sorted(set(grad["component_mass"]) | set(act["component_mass"]))
        coarse["gradiend_vs_actiend_component"] = compare_mass_profiles(
            grad["component_mass"], act["component_mass"], keys=ckeys, tau=layer_mass_tau
        )

    for cls, spec in sae_by_class.items():
        if spec.get("variant") == "per_feature_scan":
            continue
        lk = spec.get("layer_key")
        if not lk:
            continue
        sae_mass = {lk: 1.0}
        if grad:
            keys = sorted(set(grad["layer_mass"]) | {lk})
            coarse[f"gradiend_vs_sae:{cls}_layer"] = compare_mass_profiles(
                grad["layer_mass"], sae_mass, keys=keys, tau=layer_mass_tau
            )
        if act:
            keys = sorted(set(act["layer_mass"]) | {lk})
            coarse[f"actiend_vs_sae:{cls}_layer"] = compare_mass_profiles(
                act["layer_mass"], sae_mass, keys=keys, tau=layer_mass_tau
            )

    # ----- neuron-level residual bridges (primary) -----
    interest = _layers_of_interest(
        *(p for p in (grad_n, act_n) if p),
        sae_layers=sae_layers,
    )
    neuron: Dict[str, Any] = {"layers": interest, "by_layer": {}, "by_class": {}}

    for L in interest:
        layer_block: Dict[str, Any] = {}
        act_top = (act_n or {}).get("resid_channels_topk", {}).get(L) or []
        act_imp = (act_n or {}).get("resid_channel_importance", {}).get(L) or {}
        grad_top = (grad_n or {}).get("resid_channels_topk", {}).get(L) or []
        grad_imp = (grad_n or {}).get("resid_channel_importance", {}).get(L) or {}

        if act_top and grad_top:
            layer_block["actiend_vs_gradiend_writeout"] = compare_resid_channels(
                act_top, grad_top, a_imp=act_imp, b_imp=grad_imp, d_model=d_model
            )
            layer_block["actiend_vs_gradiend_writeout"]["note"] = (
                "GRADIEND side = residual channels implied by write-out "
                "(attn/mlp c_proj) weights; top-m within layer; c_attn/c_fc excluded"
            )

        for cls, spec in sae_by_class.items():
            if spec.get("variant") == "per_feature_scan":
                continue
            if spec.get("layer_key") != L:
                continue
            sae_top = spec.get("resid_channels_topk") or []
            sae_imp = spec.get("resid_channel_importance") or {}
            dm = d_model or spec.get("d_model")
            if act_top and sae_top:
                layer_block[f"actiend_vs_sae:{cls}"] = compare_resid_channels(
                    act_top, sae_top, a_imp=act_imp, b_imp=sae_imp, d_model=dm
                )
            if grad_top and sae_top:
                layer_block[f"gradiend_writeout_vs_sae:{cls}"] = compare_resid_channels(
                    grad_top, sae_top, a_imp=grad_imp, b_imp=sae_imp, d_model=dm
                )
                layer_block[f"gradiend_writeout_vs_sae:{cls}"]["note"] = (
                    "GRADIEND write-out residual channels vs SAE top-|W_dec| dims"
                )
        if layer_block:
            neuron["by_layer"][L] = layer_block

    for cls, spec in sae_by_class.items():
        if spec.get("variant") == "per_feature_scan":
            # Best single feature among bag vs ACTIEND (may beat k=1)
            L = spec.get("layer_key")
            if not L:
                continue
            act_top = (act_n or {}).get("resid_channels_topk", {}).get(L) or []
            act_imp = (act_n or {}).get("resid_channel_importance", {}).get(L) or {}
            dm = d_model
            best = None
            per_rows = []
            for pspec in spec.get("per_feature") or []:
                sae_top = pspec.get("resid_channels_topk") or []
                if not act_top or not sae_top:
                    continue
                m = compare_resid_channels(
                    act_top,
                    sae_top,
                    a_imp=act_imp,
                    b_imp=pspec.get("resid_channel_importance") or {},
                    d_model=dm or pspec.get("d_model"),
                )
                feat = (pspec.get("feature_indices") or [None])[0]
                row = {"feature": feat, **m}
                per_rows.append(row)
                if best is None or float(m.get("jaccard") or 0) > float(
                    best.get("jaccard") or 0
                ):
                    best = row
            base_cls = str(cls).split(":")[0]
            block = neuron["by_class"].setdefault(
                base_cls,
                {"sae_layer": L},
            )
            block["per_feature_vs_actiend"] = per_rows
            block["best_feature_vs_actiend"] = best
            continue

        L = spec.get("layer_key")
        if not L:
            continue
        class_block: Dict[str, Any] = {
            "sae_layer": L,
            "sae_features": spec.get("feature_indices"),
            "sae_topk": spec.get("resid_channels_topk"),
            "variant": spec.get("variant") or "k1",
        }
        sae_top = spec.get("resid_channels_topk") or []
        sae_imp = spec.get("resid_channel_importance") or {}
        dm = d_model or spec.get("d_model")
        act_top = (act_n or {}).get("resid_channels_topk", {}).get(L) or []
        act_imp = (act_n or {}).get("resid_channel_importance", {}).get(L) or {}
        grad_top = (grad_n or {}).get("resid_channels_topk", {}).get(L) or []
        grad_imp = (grad_n or {}).get("resid_channel_importance", {}).get(L) or {}
        if act_top:
            class_block["actiend_topk"] = list(act_top)
        if grad_top:
            class_block["gradiend_writeout_topk"] = list(grad_top)
        if act_top and sae_top:
            class_block["actiend_vs_sae"] = compare_resid_channels(
                act_top, sae_top, a_imp=act_imp, b_imp=sae_imp, d_model=dm
            )
        if grad_top and sae_top:
            class_block["gradiend_writeout_vs_sae"] = compare_resid_channels(
                grad_top, sae_top, a_imp=grad_imp, b_imp=sae_imp, d_model=dm
            )
        if act_top and grad_top:
            class_block["actiend_vs_gradiend_writeout"] = compare_resid_channels(
                act_top, grad_top, a_imp=act_imp, b_imp=grad_imp, d_model=dm
            )
        neuron["by_class"][str(cls)] = class_block

    return {
        "resid_m": int(resid_m),
        "layer_mass_tau": float(layer_mass_tau),
        "profiles": profiles,
        "coarse": coarse,
        "neuron": neuron,
        "notes": [
            "Primary metric = residual-channel Jaccard / cosine / Spearman "
            "(ACTIEND channels, SAE top-|W_dec|, GRADIEND c_proj->resid dim), "
            "top-m within each layer from all resid-mappable dims.",
            "SAE variants: k1 (headline), kstar_bag (mean |W_dec| over top-k*), "
            "kstar_union (union of per-feature top-m); best_feature_vs_actiend scans the k* bag.",
            "Coarse layer/component profiles are secondary context only "
            "(agreeing that L11 is heavy is not neuron identity).",
            "GRADIEND non-write-out parameters (c_attn, c_fc, ...) are not mapped to residual channels.",
        ],
    }


def run_localization_analysis(
    train_raw: Dict[str, Any],
    sae_raw: Optional[Dict[str, Any]],
    *,
    output_dir: Union[str, Path],
    topk: Union[int, float] = DEFAULT_TOPK,
    resid_m: int = DEFAULT_RESID_M,
    target_classes: Sequence[str] = ("M", "F"),
    part: str = "decoder-weight",
) -> Dict[str, Any]:
    """
    End-to-end localization from live trainers + SAE objects.

    Writes ``{output_dir}/localization/`` artifacts and returns a JSON-ready block.

    Neuron-level scores use **all** residual-mappable dims (ACTIEND sites /
    GRADIEND write-outs), then top-``resid_m`` channels **per layer**. Coarse
    bags still use global ``topk`` mass.
    """
    output_dir = Path(output_dir)
    loc_dir = output_dir / "localization"
    loc_dir.mkdir(parents=True, exist_ok=True)

    gradiend_topk = None
    actiend_topk = None
    gradiend_resid_rows = None
    actiend_resid_rows = None

    if "gradiend" in train_raw and train_raw["gradiend"].get("trainer") is not None:
        tr = train_raw["gradiend"]["trainer"]
        gradiend_topk = export_method_topk(tr, topk=topk, part=part)
        gradiend_resid_rows = export_resid_channel_rows(tr, part=part, write_out_only=True)
        (loc_dir / "gradiend_topk.json").write_text(
            json.dumps(_json_ready(gradiend_topk[:5000]), indent=2), encoding="utf-8"
        )
        (loc_dir / "gradiend_resid_channels.json").write_text(
            json.dumps(_json_ready(gradiend_resid_rows[:20000]), indent=2), encoding="utf-8"
        )
    if "actiend" in train_raw and train_raw["actiend"].get("trainer") is not None:
        tr = train_raw["actiend"]["trainer"]
        actiend_topk = export_method_topk(tr, topk=topk, part=part)
        actiend_resid_rows = export_resid_channel_rows(tr, part=part, write_out_only=False)
        (loc_dir / "actiend_topk.json").write_text(
            json.dumps(_json_ready(actiend_topk[:5000]), indent=2), encoding="utf-8"
        )
        (loc_dir / "actiend_resid_channels.json").write_text(
            json.dumps(_json_ready(actiend_resid_rows[:20000]), indent=2), encoding="utf-8"
        )

    sae_by_class: Dict[str, Dict[str, Any]] = {}
    d_model = None
    if sae_raw and not sae_raw.get("error"):
        selections = sae_raw.get("selections") or {}
        per_class = selections.get("per_class") or {}
        fbc = per_class.get("features_by_class") or {}
        selected_by_class = sae_raw.get("selected_layer_by_class") or {}
        sae_by_layer = sae_raw.get("_sae_by_layer") or {}
        class_readouts = (sae_raw.get("readouts") or {}).get("per_class_by_class") or {}
        for cls in target_classes:
            cls = str(cls)
            feats = list(fbc.get(cls) or [])
            if not feats:
                continue
            layer = _optional_int_layer(
                selected_by_class.get(cls) or sae_raw.get("selected_layer_global")
            )
            if layer is None:
                # E-lock picked the virtual all-layers site: no single SAE W_dec.
                sae_by_class[cls] = {
                    "layer": "all",
                    "layer_key": "all",
                    "variant": "k1",
                    "k": 1,
                    "feature_indices": feats[:1],
                    "resid_channels_topk": [],
                    "skipped": "all_layers_site",
                }
                continue
            sae = sae_by_layer.get(layer)
            if sae is None:
                try:
                    import torch
                    from sae_eval import active_sae_release, load_sae, resid_sites

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    _, sae_id = resid_sites(layer)
                    sae = load_sae(active_sae_release(), sae_id, device=device)
                    sae_by_layer[layer] = sae
                except Exception:
                    continue
            # Bag width = val-selected k* (same as encoder sae:{cls}), not a fixed k.
            k_star = int((class_readouts.get(cls) or {}).get("readout_k") or 1)
            k_bag = min(max(1, k_star), len(feats))

            # Headline key ``M`` = top-1 (matches causal sae:M:k1)
            spec_k1 = sae_resid_channels(sae, feats[:1], m=resid_m, layer=layer)
            sae_by_class[cls] = {**spec_k1, "variant": "k1", "k": 1}
            if d_model is None and spec_k1.get("d_model"):
                d_model = int(spec_k1["d_model"])

            # Mean-|W_dec| bag / union over top-k* features
            if k_bag > 1:
                bag = sae_resid_channels(sae, feats[:k_bag], m=resid_m, layer=layer)
                sae_by_class[f"{cls}:kstar_bag"] = {
                    **bag,
                    "variant": "kstar_bag",
                    "k": k_bag,
                }
                uni = sae_resid_channels_union(sae, feats[:k_bag], m=resid_m, layer=layer)
                sae_by_class[f"{cls}:kstar_union"] = {
                    **uni,
                    "variant": "kstar_union",
                    "k": k_bag,
                }

            # Per-feature top-m among k* bag — best Jaccard vs ACTIEND in compare
            per_feat_specs = []
            for fi in feats[:k_bag]:
                per_feat_specs.append(
                    sae_resid_channels(sae, [fi], m=resid_m, layer=layer)
                )
            sae_by_class[f"{cls}:_per_feature"] = {
                "layer": layer,
                "layer_key": f"L{int(layer)}",
                "feature_indices": feats[:k_bag],
                "per_feature": per_feat_specs,
                "variant": "per_feature_scan",
                "k": k_bag,
                "resid_channels_topk": [],  # placeholder; compare skips empty
            }

        (loc_dir / "sae_resid_channels.json").write_text(
            json.dumps(_json_ready(sae_by_class), indent=2), encoding="utf-8"
        )

    # Coarse from global top-k; neuron bridge from full residual-mappable rows.
    result = compare_localization(
        gradiend_rows=gradiend_topk,
        actiend_rows=actiend_topk,
        gradiend_resid_rows=gradiend_resid_rows,
        actiend_resid_rows=actiend_resid_rows,
        sae_by_class=sae_by_class,
        resid_m=resid_m,
        d_model=d_model,
    )
    result["config"] = {
        "topk": topk,
        "resid_m": resid_m,
        "part": part,
        "target_classes": list(target_classes),
        "neuron_source": "per_layer_top_m_from_all_resid_mappable_dims",
    }
    result["dir"] = str(loc_dir)
    ready = _json_ready(result)
    # Drop bulky raw channel maps from results.json blob (kept on disk).
    slim = dict(ready)
    profiles = dict(slim.get("profiles") or {})
    for key in ("gradiend", "actiend", "gradiend_neuron", "actiend_neuron"):
        prof = profiles.get(key)
        if isinstance(prof, dict) and "resid_channel_importance" in prof:
            prof = dict(prof)
            # Keep only top layers' channel counts in the slim JSON
            imp = prof.get("resid_channel_importance") or {}
            prof["resid_channel_importance_n"] = {
                L: len(ch) for L, ch in imp.items()
            }
            prof.pop("resid_channel_importance", None)
            profiles[key] = prof
    slim["profiles"] = profiles
    (loc_dir / "comparison.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    (loc_dir / "comparison_slim.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return slim


def localization_table(localization: Optional[Dict[str, Any]]) -> str:
    """
    Console table focused on **neuron-level** residual-channel agreement.

    Coarse layer Spearman is a one-line footnote, not the headline.
    """
    if not localization or localization.get("error"):
        err = (localization or {}).get("error")
        return f"\nLocalization: -{f' ({err})' if err else ''}"

    neuron = localization.get("neuron") or {}
    by_class = neuron.get("by_class") or {}
    resid_m = localization.get("resid_m") or localization.get("config", {}).get("resid_m")
    lines = [
        "",
        f"Localization (neuron-level residual channels, top-m={resid_m}):",
        "  Jaccard / overlap on channel *sets*; cosine/spearman on channel *scores*",
        "  SAE variants: k1 | k*_bag (mean|W_dec|) | k*_union | best feat in k* bag",
        f"{'pair':<42} {'layer':>5} {'jacc':>7} {'ovlp':>7} {'cos':>7} {'spr':>7}  intersect",
        "-" * 102,
    ]

    def _intersect_preview(metrics: Optional[Dict[str, Any]], limit: int = 8) -> str:
        if not metrics:
            return ""
        a = set(int(x) for x in (metrics.get("topk_a") or []))
        b = set(int(x) for x in (metrics.get("topk_b") or []))
        inter = sorted(a & b)
        if not inter:
            return "[]"
        if len(inter) > limit:
            return str(inter[:limit])[:-1] + ", ...]"
        return str(inter)

    def _row(pair: str, layer: str, metrics: Optional[Dict[str, Any]]) -> None:
        if not metrics:
            lines.append(
                f"{pair:<42} {layer:>5} {'-':>7} {'-':>7} {'-':>7} {'-':>7}  -"
            )
            return
        lines.append(
            f"{pair:<42} {layer:>5} "
            f"{_fmt(metrics.get('jaccard'))} "
            f"{_fmt(metrics.get('overlap_frac'))} "
            f"{_fmt(metrics.get('channel_cosine'))} "
            f"{_fmt(metrics.get('channel_spearman'))}  "
            f"{_intersect_preview(metrics)}"
        )

    if by_class:
        # Prefer stable order: class headlines first, then bag/union variants
        def _cls_sort(key: str) -> tuple:
            if ":" not in key:
                return (0, key, "")
            base, rest = key.split(":", 1)
            return (1, base, rest)

        for cls in sorted(by_class.keys(), key=_cls_sort):
            block = by_class[cls] or {}
            L = str(block.get("sae_layer") or "?")
            tag = block.get("variant") or ("k1" if ":" not in str(cls) else str(cls))
            label = str(cls) if ":" in str(cls) else f"{cls}:{tag}"
            _row(f"actiend vs sae:{label}", L, block.get("actiend_vs_sae"))
            _row(f"gradiend_wo vs sae:{label}", L, block.get("gradiend_writeout_vs_sae"))
            if ":" not in str(cls):
                _row(
                    f"actiend vs gradiend_wo ({cls})",
                    L,
                    block.get("actiend_vs_gradiend_writeout"),
                )
                best = block.get("best_feature_vs_actiend")
                if best:
                    feat = best.get("feature")
                    _row(
                        f"actiend vs sae:{cls}:best_feat({feat})",
                        L,
                        best,
                    )
    else:
        for L, block in (neuron.get("by_layer") or {}).items():
            for pair, metrics in (block or {}).items():
                if pair == "note" or not isinstance(metrics, dict):
                    continue
                _row(pair, str(L), metrics)

    coarse = localization.get("coarse") or {}
    ga = coarse.get("gradiend_vs_actiend_layer") or {}
    if ga:
        lines.append(
            f"  (coarse layer footnote: gradiend<->actiend spearman={_fmt(ga.get('spearman')).strip()} "
            f"cosine={_fmt(ga.get('cosine')).strip()})"
        )

    grad = (localization.get("profiles") or {}).get("gradiend") or {}
    if grad and grad.get("write_out_mass_frac") is not None:
        lines.append(
            f"  gradiend write-out mass frac={float(grad['write_out_mass_frac']):.3f} "
            f"(only this maps to residual channels)"
        )
    return "\n".join(lines)


def _fmt(value: Any, width: int = 7, precision: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):>{width}.{precision}f}"
    return f"{'-':>{width}}"
