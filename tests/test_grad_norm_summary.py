"""Gradient-norm summary must reach done.json.

The per-step series live in training.json, which rsync_analysis deliberately
excludes (multi-MB). Without a compact summary in done.json the mechanism
experiment runs on the cluster and its result is never readable locally --
instrumentation that produces unreadable data is not instrumentation.
"""

from __future__ import annotations

import pytest

from study.stages.train import _grad_norm_summary


class TestAbsentSeries:
    def test_empty_payload_returns_nothing(self):
        assert _grad_norm_summary({}) == {}

    def test_older_package_output_is_byte_identical(self):
        """A package without the instrumentation must not change done.json."""
        assert _grad_norm_summary({"convergence_info": {"x": 1}}) == {}

    def test_empty_dicts_are_treated_as_absent(self):
        assert _grad_norm_summary({"encoder_grad_norms": {}, "decoder_grad_norms": {}}) == {}


class TestSummary:
    @staticmethod
    def _payload(enc, dec, ratio=None):
        p = {
            "encoder_grad_norms": {str(i + 1): v for i, v in enumerate(enc)},
            "decoder_grad_norms": {str(i + 1): v for i, v in enumerate(dec)},
        }
        if ratio is not None:
            p["decoder_over_encoder_grad_ratio"] = {
                str(i + 1): v for i, v in enumerate(ratio)
            }
        return p

    def test_median_not_mean(self):
        """First steps are transient; a mean would be dragged by them."""
        out = _grad_norm_summary(self._payload([100.0, 1.0, 1.0, 1.0], [1.0] * 4))
        assert out["encoder_grad_norm_median"] == pytest.approx(1.0)

    def test_first_and_last_are_ordered_by_step_not_dict_order(self):
        payload = {
            "encoder_grad_norms": {"10": 5.0, "2": 1.0, "100": 9.0},
            "decoder_grad_norms": {"2": 1.0},
        }
        out = _grad_norm_summary(payload)
        assert out["encoder_grad_norm_first"] == pytest.approx(1.0)
        assert out["encoder_grad_norm_last"] == pytest.approx(9.0)

    def test_reported_ratio_is_preferred_over_recomputation(self):
        out = _grad_norm_summary(self._payload([1.0, 1.0], [1.0, 1.0], ratio=[0.5, 0.5]))
        assert out["decoder_over_encoder_median"] == pytest.approx(0.5)

    def test_ratio_recomputed_per_step_when_absent(self):
        out = _grad_norm_summary(self._payload([1.0, 2.0], [0.001, 0.002]))
        assert out["decoder_over_encoder_median"] == pytest.approx(0.001)

    def test_a_starved_decoder_is_visible(self):
        """The hypothesis this exists to test."""
        out = _grad_norm_summary(self._payload([1.0] * 5, [1e-4] * 5))
        assert out["decoder_over_encoder_median"] < 1e-3

    def test_nan_and_non_numeric_are_dropped(self):
        payload = {
            "encoder_grad_norms": {"1": 1.0, "2": float("nan"), "3": "oops"},
            "decoder_grad_norms": {"1": 0.5},
        }
        out = _grad_norm_summary(payload)
        assert out["encoder_grad_norm_median"] == pytest.approx(1.0)

    def test_zero_encoder_gradient_does_not_divide_by_zero(self):
        out = _grad_norm_summary(self._payload([0.0, 0.0], [1.0, 1.0]))
        assert "decoder_over_encoder_median" not in out


class TestWiredIntoDoneJson:
    def test_summary_is_attached_under_grad_norms(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "study/stages/train.py").read_text(encoding="utf-8")
        assert 'out["grad_norms"] = grad' in text
        assert "_grad_norm_summary(payload)" in text


class TestNesting:
    """The series live under training_stats, not at the top level.

    Reading the top level returned None for every key, which is indistinguishable
    from "the instrumentation never ran" -- the same nesting the long-standing
    encoder_norms/decoder_norms use.
    """

    NESTED = {
        "training_stats": {
            "encoder_grad_norms": {"1": 1.0, "2": 2.0, "3": 3.0},
            "decoder_grad_norms": {"1": 0.001, "2": 0.002, "3": 0.003},
        }
    }

    def test_reads_the_nested_location(self):
        out = _grad_norm_summary(self.NESTED)
        assert out["n_steps"] == 3
        assert out["decoder_over_encoder_median"] == pytest.approx(0.001)

    def test_flat_payload_still_works(self):
        """Back-compat for any caller passing the stats dict directly."""
        out = _grad_norm_summary({"encoder_grad_norms": {"1": 1.0},
                                  "decoder_grad_norms": {"1": 0.5}})
        assert out["decoder_over_encoder_median"] == pytest.approx(0.5)

    def test_non_dict_training_stats_falls_back(self):
        assert _grad_norm_summary({"training_stats": "unexpected"}) == {}
