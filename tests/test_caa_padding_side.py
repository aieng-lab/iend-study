"""CAA's per-token selectors must give the same tokens under left and right padding.

Neutral CAA batches come from ``tokenizer(padding=True)``; Gemma pads LEFT, where the old
``attention_mask.sum() - 1`` read a different token for every row shorter than the batch.
"""

import pytest
import torch

LENGTHS = [3, 7, 1, 5, 7]


def _batch(side, lengths=LENGTHS, d=4):
    """activation[b, t, :] carries the token id, so a selector's output names the token it read."""
    width = max(lengths)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    act = torch.zeros(len(lengths), width, d)
    last = []
    for row, n in enumerate(lengths):
        ids = torch.arange(1, n + 1, dtype=torch.float) + 100 * (row + 1)
        sl = slice(width - n, width) if side == "left" else slice(0, n)
        mask[row, sl] = 1
        act[row, sl, :] = ids.unsqueeze(-1)
        last.append(float(ids[-1]))
    return act, {"attention_mask": mask}, last


@pytest.mark.parametrize("side", ["left", "right"])
def test_last_token_selector_reads_each_rows_last_real_token(side):
    from caa_eval import last_token_selector

    act, inputs, last = _batch(side)
    out = last_token_selector(act, inputs)
    assert out[:, 0].tolist() == last


def test_last_token_selector_is_padding_side_invariant():
    from caa_eval import last_token_selector

    left = last_token_selector(*_batch("left")[:2])
    right = last_token_selector(*_batch("right")[:2])
    assert torch.equal(left, right)


def test_nonpad_token_selector_keeps_exactly_the_real_tokens_for_both_sides():
    from caa_eval import nonpad_token_selector

    for side in ("left", "right"):
        act, inputs, _ = _batch(side)
        out = nonpad_token_selector(act, inputs)
        assert out.shape[0] == sum(LENGTHS)
        assert (out[:, 0] > 0).all(), "a padding position leaked into the selection"
