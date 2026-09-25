"""Make CLM class probabilities SUM space- and case-variant token ids.

Package ``_clm_first_continuation_token_ids`` encodes one contextual first
token and returns it. Vocab-normalized ids (``he`` / ``He`` / ``Ġhe`` / …)
are only used as a fallback when that encode fails. GRADIEND training fill
already sums those surface variants; this patch unions them so SAE/CAA/IEND
decoder scoring share the same SUM (not MAX) over matching ids.

Safe to apply twice. Dedup keeps a patched package copy from double-counting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_PATCH_ATTR = "_gradiend_sae_sum_surface_variants"


def apply_clm_surface_variant_sum_patch() -> bool:
    """Patch package CLM id resolution. Returns True if the helper is patched."""
    try:
        import gradiend.trainer.text.prediction.decoder_eval_utils as deu
    except ImportError:
        return False
    current = deu._clm_first_continuation_token_ids
    if getattr(current, _PATCH_ATTR, False):
        return True
    orig = current

    def wrapped(
        tokenizer,
        prefix_context: str,
        gap: str,
        target: str,
        vocab_norm_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[int]:
        ids = list(orig(tokenizer, prefix_context, gap, target, vocab_norm_map))
        if vocab_norm_map is not None and target is not None:
            ids.extend(
                deu._single_token_candidate_ids(
                    tokenizer, str(target).lstrip(), vocab_norm_map
                )
            )
        return deu._unique_token_ids(ids)

    wrapped.__name__ = orig.__name__
    wrapped.__doc__ = orig.__doc__
    setattr(wrapped, _PATCH_ATTR, True)
    deu._clm_first_continuation_token_ids = wrapped
    return True


def surface_variant_token_ids(tokenizer: Any, surface: str) -> List[int]:
    """Token ids whose vocab surface matches ``surface`` ignoring space and case."""
    from gradiend.trainer.text.prediction.decoder_eval_utils import (
        _build_vocab_norm_map,
        _single_token_candidate_ids,
        _unique_token_ids,
    )

    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    vocab_norm_map = _build_vocab_norm_map(tok)
    return _unique_token_ids(
        _single_token_candidate_ids(tok, str(surface).lstrip(), vocab_norm_map)
    )
