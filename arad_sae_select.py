"""
Arad et al. (2025) SAE output-score feature selection (steering filter).

Adapted from https://github.com/technion-cs-nlp/saes-are-good-for-steering
(MIT License, Copyright (c) 2025 technion-cs-nlp).

Paper: "SAEs Are Good for Steering—If You Select the Right Features"
  — rank features by *output score* (logit-lens top tokens × post-amplify
    rank/prob), then steer with high-scoring features.

We port only the output-score ranking (input score needs Neuronpedia dumps).
Study ablation id: ``sae:{cls}:sel_arad_out``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from sae_eval import SAEFeatureSelection

# Match Arad defaults (src/output_score.py, src/utils.py).
ARAD_NEUTRAL_PROMPT = "From my experience,"
ARAD_AMP_FACTOR = 10.0
ARAD_LOGIT_LENS_K = 20

# Common HF paths for final RMS/LayerNorm + unembed.
_NORM_CANDIDATES = (
    "transformer.ln_f",  # GPT-2
    "model.norm",  # Llama / Gemma / Qwen2+
    "gpt_neox.final_layer_norm",  # Pythia
    "model.model.norm",
    "transformer.final_layer_norm",
)
_LM_HEAD_CANDIDATES = (
    "lm_head",
    "embed_out",  # GPT-NeoX / Pythia
)


def find_final_norm_and_lm_head(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """Resolve final pre-unembed norm and lm_head across study architectures."""
    named = dict(model.named_modules())
    norm = None
    for path in _NORM_CANDIDATES:
        if path in named:
            norm = named[path]
            break
    if norm is None:
        raise AttributeError(
            "Could not find final layer norm; tried: " + ", ".join(_NORM_CANDIDATES)
        )
    head = None
    for path in _LM_HEAD_CANDIDATES:
        if path in named:
            head = named[path]
            break
    if head is None and hasattr(model, "lm_head"):
        head = model.lm_head
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None:
        raise AttributeError(
            "Could not find lm_head; tried: " + ", ".join(_LM_HEAD_CANDIDATES)
        )
    return norm, head


def _w_dec_matrix(sae) -> torch.Tensor:
    """Return W_dec as [n_features, d_model]."""
    w = sae.W_dec
    if w.ndim != 2:
        raise AttributeError(f"Unexpected W_dec ndim={w.ndim}")
    if w.shape[0] > w.shape[1]:
        return w
    return w.T


def cache_logit_lens_topk(
    sae,
    final_norm: nn.Module,
    lm_head: nn.Module,
    k: int = ARAD_LOGIT_LENS_K,
) -> torch.Tensor:
    """
    Logit-lens top-k token indices per SAE feature (Arad ``cache_logit_lens``).

    Returns LongTensor [n_features, k].
    """
    device = next(final_norm.parameters()).device
    dtype = next(final_norm.parameters()).dtype
    dec = _w_dec_matrix(sae).detach().to(device=device, dtype=dtype)
    with torch.no_grad():
        norms = final_norm(dec)
        logits = lm_head(norms)
        conf = torch.softmax(logits, dim=-1)
        topk = torch.topk(conf, dim=-1, k=min(int(k), conf.shape[-1]))
    return topk.indices.detach().cpu()


@contextmanager
def _disable_sae_hooks(sae):
    """Blank SAELens hooks during clean encode/decode (Arad ``_disable_hooks``)."""
    hook_dict = getattr(sae, "hook_dict", None)
    if not hook_dict:
        yield
        return
    saved = {}
    try:
        for name in list(hook_dict):
            saved[name] = getattr(sae, name)
            setattr(sae, name, nn.Identity())
        yield
    finally:
        for name, mod in saved.items():
            setattr(sae, name, mod)


class AmplifySAEFeatureHook:
    """
    Amplify selected SAE feature(s) at the last sequence position.

    Faithful to Arad ``AmlifySAEHook``: encode → boost feature by
    ``amp_factor * max_act`` → decode + reconstruction error.
    """

    def __init__(
        self,
        sae,
        features: Sequence[int],
        amp_factor: float = ARAD_AMP_FACTOR,
    ) -> None:
        self.sae = sae
        self.features = [int(f) for f in features]
        self.amp_factor = float(amp_factor)

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        resid = output[0] if is_tuple else output
        # Match Arad: operate on the last token index of the padded sequence.
        x = resid
        dtype_in = x.dtype
        with torch.no_grad():
            # Encode → amplify → decode *before* the clean error path.
            # SAELens layer_norm / constant_norm_rescale store ln_std (or
            # x_norm_coeff) on encode and delete them on decode; doing clean
            # encode/decode first would wipe those attrs (TopKSAE crash).
            # Same order as SAELens SAE.forward(use_error_term=True).
            acts = self.sae.encode(x)
            max_act = torch.max(acts[:, -1, :]).item()
            for feat in self.features:
                acts[:, -1, feat] = acts[:, -1, feat] + max_act * self.amp_factor
            out = self.sae.decode(acts).to(torch.float64)

            with _disable_sae_hooks(self.sae):
                acts_clean = self.sae.encode(x)
                x_hat = self.sae.decode(acts_clean)
            err = x.to(torch.float64) - x_hat.to(torch.float64)
            hook_err = getattr(self.sae, "hook_sae_error", None)
            if callable(hook_err):
                err = hook_err(err)
            out = (out + err).to(dtype_in)
        if is_tuple:
            return (out,) + tuple(output[1:])
        return out


def output_score_for_feature(
    model: nn.Module,
    tokenizer,
    sae,
    module_path: str,
    feature_index: int,
    logit_lens_indices: Sequence[int],
    *,
    sentence: str = ARAD_NEUTRAL_PROMPT,
    amp_factor: float = ARAD_AMP_FACTOR,
    device: Optional[torch.device] = None,
) -> float:
    """
    Arad output score for one feature on a neutral prompt.

    ``rank_output_score * top_token_score`` where
      top_token_score = max softmax prob among logit-lens tokens
      rank_output_score = 1 - (min rank of those tokens) / vocab_size
    """
    device = device or next(model.parameters()).device
    named = dict(model.named_modules())
    if module_path not in named:
        raise KeyError(f"module_path {module_path!r} not in model")
    block = named[module_path]
    hook = AmplifySAEFeatureHook(sae, [int(feature_index)], amp_factor=amp_factor)
    handle = block.register_forward_hook(hook)
    try:
        enc = tokenizer(sentence, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        model.eval()
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits, dim=0).detach().cpu()
        vocab = int(probs.shape[0])
        ll = [int(i) for i in logit_lens_indices if 0 <= int(i) < vocab]
        if not ll:
            return 0.0
        order = torch.argsort(probs, descending=True)
        ranks = [(order == t).nonzero(as_tuple=True)[0].item() for t in ll]
        top_token_score = float(torch.max(probs[ll]).item())
        rank_output_score = 1.0 - (min(ranks) / float(vocab))
        return float(rank_output_score * top_token_score)
    finally:
        handle.remove()


def rerank_per_class_by_arad_output(
    model: nn.Module,
    tokenizer,
    sae,
    module_path: str,
    base_selection: SAEFeatureSelection,
    *,
    candidate_k: int = 16,
    target_classes: Optional[Sequence[str]] = None,
    sentence: str = ARAD_NEUTRAL_PROMPT,
    amp_factor: float = ARAD_AMP_FACTOR,
    logit_lens_k: int = ARAD_LOGIT_LENS_K,
    progress: Optional[Any] = None,
) -> SAEFeatureSelection:
    """
    Re-rank ``per_class`` candidates by Arad output score; keep exclusivity.

    Candidates = first ``candidate_k`` features already owned by each class
    under mean-diff (or whatever produced ``base_selection``). Layer / feature
    pick for the study ablation uses this score's top-1.
    """
    def _p(msg: str) -> None:
        if callable(progress):
            progress(msg)

    classes = [str(c) for c in (target_classes or list((base_selection.features_by_class or {}).keys()))]
    if not classes:
        classes = [str(base_selection.contrast_pair[0]), str(base_selection.contrast_pair[1])]

    final_norm, lm_head = find_final_norm_and_lm_head(model)
    device = next(model.parameters()).device
    # Keep SAE on same device as model for encode during hooks.
    try:
        sae = sae.to(device)
    except Exception:
        pass

    _p(f"  Arad output-score: logit-lens top-{logit_lens_k} …")
    ll_topk = cache_logit_lens_topk(sae, final_norm, lm_head, k=logit_lens_k)

    fbc_base = base_selection.features_by_class or {}
    scored: Dict[str, Dict[int, float]] = {c: {} for c in classes}
    k_cand = max(1, int(candidate_k))

    for c in classes:
        cands = [int(i) for i in (fbc_base.get(c) or [])][:k_cand]
        _p(f"  Arad output-score: class={c} n_cand={len(cands)}")
        for feat in cands:
            ll_idx = ll_topk[int(feat)].tolist() if int(feat) < ll_topk.shape[0] else []
            try:
                sc = output_score_for_feature(
                    model,
                    tokenizer,
                    sae,
                    module_path,
                    int(feat),
                    ll_idx,
                    sentence=sentence,
                    amp_factor=amp_factor,
                    device=device,
                )
            except Exception as exc:
                _p(f"  Arad score failed feat={feat}: {exc}")
                sc = float("-inf")
            scored[c][int(feat)] = float(sc)

    # Exclusive top-k assignment ordered by Arad score (mirror select_per_class).
    k_out = max(len(fbc_base.get(c) or []) for c in classes) if classes else 1
    k_out = max(1, min(k_out, k_cand))
    taken: set = set()
    features_by_class: Dict[str, List[int]] = {c: [] for c in classes}
    for _ in range(k_out):
        picks: Dict[str, int] = {}
        for c in classes:
            order = [
                i
                for i, _ in sorted(scored[c].items(), key=lambda kv: -kv[1])
                if i not in taken and scored[c][i] > float("-inf")
            ]
            if order:
                picks[c] = order[0]
        if not picks:
            break
        claimed: Dict[int, str] = {}
        for c, idx in sorted(picks.items(), key=lambda kv: -float(scored[kv[0]][kv[1]])):
            if idx in claimed or idx in taken:
                order = [
                    i
                    for i, _ in sorted(scored[c].items(), key=lambda kv: -kv[1])
                    if i not in taken and i not in claimed and scored[c][i] > float("-inf")
                ]
                if not order:
                    continue
                idx = order[0]
            claimed[idx] = c
        if not claimed:
            break
        for idx, c in claimed.items():
            features_by_class[c].append(idx)
            taken.add(idx)

    selected: List[int] = []
    scores: Dict[int, float] = {}
    for c in classes:
        for i in features_by_class[c]:
            selected.append(i)
            scores[i] = float(scored[c][i])

    return SAEFeatureSelection(
        feature_indices=selected,
        scores=scores,
        mean_by_class=dict(base_selection.mean_by_class or {}),
        contrast_pair=base_selection.contrast_pair,
        mode="per_class_arad_out",
        layer=int(base_selection.layer),
        sae_id=str(base_selection.sae_id or ""),
        selection_split=str(base_selection.selection_split or "validation"),
        used_neutral_specificity=bool(base_selection.used_neutral_specificity),
        neutral_means=dict(base_selection.neutral_means or {}),
        features_by_class=features_by_class,
        scores_by_class={
            c: {i: float(scored[c][i]) for i in feats}
            for c, feats in features_by_class.items()
        },
    )

