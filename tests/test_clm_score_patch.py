from __future__ import annotations

import torch

from causal_eval import token_pair_probs
from study.clm_score_patch import (
    apply_clm_surface_variant_sum_patch,
    surface_variant_token_ids,
)


class _VocabTok:
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self):
        self.vocab = {"he": 10, "He": 11, "Ġhe": 12, "she": 20, "Ġshe": 21}

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(str(token), self.unk_token_id)

    def tokenize(self, text):
        return [text] if text in self.vocab else []

    def __call__(self, text, add_special_tokens=False, padding=False):
        vid = self.vocab.get(str(text))
        if vid is not None:
            return {"input_ids": [vid]}
        return {"input_ids": [99]}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]


def test_surface_variant_token_ids_include_space_and_case():
    ids = surface_variant_token_ids(_VocabTok(), "he")
    assert set(ids) >= {10, 11, 12}


def test_token_pair_probs_sums_variants_not_max():
    tok = _VocabTok()
    logits = torch.full((30,), -20.0)
    logits[10] = 0.0
    logits[11] = 0.0
    logits[12] = 0.0
    logits[20] = 0.0
    p_he, p_she, _ = token_pair_probs(logits, tok, token_a="he", token_b="she")
    # Three he-variants vs two she-variants, each with equal pre-softmax logit.
    assert p_he > p_she
    one_he = float(torch.softmax(logits, dim=-1)[10].item())
    assert p_he == one_he * 3


def test_clm_patch_unions_vocab_ids(monkeypatch):
    import gradiend.trainer.text.prediction.decoder_eval_utils as deu

    def fake_orig(_tokenizer, _prefix, _gap, _target, vocab_norm_map=None):
        del vocab_norm_map
        return [12]

    monkeypatch.setattr(deu, "_clm_first_continuation_token_ids", fake_orig)
    assert apply_clm_surface_variant_sum_patch() is True
    ids = deu._clm_first_continuation_token_ids(
        _VocabTok(), "The", " ", "he", deu._build_vocab_norm_map(_VocabTok())
    )
    assert 12 in ids
    assert set(ids) >= {10, 11, 12}
