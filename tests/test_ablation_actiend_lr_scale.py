from __future__ import annotations

import json

from gradiend import Signal
import scripts.ablation_actiend_lr_scale as ablation

from scripts.ablation_actiend_lr_scale import (
    DEFAULT_LRS,
    DEFAULT_MODELS,
    _causal_requested_for_row,
    _cache_uses_current_one_pole_data_protocol,
    _effective_lrs,
    _load_seed_summary,
    _patched_training_arguments,
    _parser,
    _with_signal_scale,
    _write_report_shards,
)


class _Args:
    def __init__(self) -> None:
        self.signal = Signal.activation(
            token_selector="prediction",
            target_token_selector="pre_prediction",
            scale="running_rms",
            scale_reduce="per_site",
            scale_momentum=0.0,
        )
        self.signals = [self.signal]
        self.metadata = {
            "actiend_signal_scale": {
                "scale": "running_rms",
                "scale_reduce": "per_site",
            }
        }

    def _normalize_signal_arguments(self) -> None:
        self.signals = [self.signal]


def test_raw_arm_explicitly_removes_study_default_running_rms():
    args = _with_signal_scale(_Args(), None)

    assert "scale" not in args.signal.options
    assert args.signal.options["token_selector"] == "prediction"
    assert args.signal.options["target_token_selector"] == "pre_prediction"
    assert args.signals == [args.signal]
    assert args.metadata["actiend_signal_scale"] == {"scale": "raw"}


def test_running_rms_arm_rebuilds_signal_and_plural_axis():
    args = _with_signal_scale(_Args(), "running_rms")

    assert args.signal.options["scale"] == "running_rms"
    assert args.signal.options["scale_reduce"] == "per_site"
    assert args.signal.options["scale_momentum"] == 0.0
    assert args.signals == [args.signal]
    assert args.metadata["actiend_signal_scale"] == {
        "scale": "running_rms",
        "scale_reduce": "per_site",
        "scale_momentum": 0.0,
    }


def test_patch_overrides_current_study_running_rms_default_with_raw(tmp_path):
    from study import training_profiles as tp
    from study.config import load_study_config

    cfg = load_study_config(
        model="gpt2-small",
        task="language",
        suite="core",
        cli_overrides={"methods": ["actiend"]},
    )
    shared = tp.shared_training_kwargs(cfg, backend="actiend")
    shared["learning_rate"] = 3e-5

    with _patched_training_arguments(scale=None, lr=3e-5):
        args = tp.build_training_arguments(
            "actiend",
            experiment_dir=str(tmp_path),
            shared=shared,
            one_pole=True,
        )

    assert "scale" not in args.signal.options
    assert float(args.learning_rate) == 3e-5


def test_default_grid_covers_pythia_and_3e4_without_stress_lr():
    args = _parser().parse_args([])

    assert "pythia-70m-deduped" in DEFAULT_MODELS
    assert 3e-4 in DEFAULT_LRS
    assert 1e-3 not in DEFAULT_LRS
    assert args.output_subdir == "ablation_actiend_lr_scale_v2"
    assert _effective_lrs(DEFAULT_LRS, include_stress=False) == sorted(DEFAULT_LRS)
    assert _effective_lrs(DEFAULT_LRS, include_stress=True)[-1] == 1e-3


def test_seed_summary_reports_score_and_full_convergence_counts(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "seed_report.json").write_text(
        json.dumps(
            {
                "threshold": 0.9,
                "seeds_tried": [4, 5, 6],
                "convergent_count": 1,
                "best_seed": 5,
                "early_stop_reason": "max_seeds reached",
                "runs": [
                    {"seed": 4, "convergence_metric_value": 0.88, "converged": False},
                    {"seed": 5, "convergence_metric_value": 0.94, "converged": True},
                    {"seed": 6, "convergence_metric_value": 0.91, "converged": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = _load_seed_summary(tmp_path)

    assert summary["seed_count"] == 3
    assert summary["seed_score_pass_count"] == 2
    assert summary["seed_convergent_count"] == 1
    assert summary["seed_score_min"] == 0.88
    assert summary["seed_score_mean"] == 0.91
    assert summary["seed_score_max"] == 0.94
    assert summary["best_seed"] == 5


def test_causal_score_cutoff_includes_borderline_cells():
    assert _causal_requested_for_row({"score": 0.8}, 0.8) is True
    assert _causal_requested_for_row({"score": 0.7999}, 0.8) is False
    assert _causal_requested_for_row({"score": None}, 0.8) is False


def test_stale_one_pole_cache_is_not_reused(tmp_path):
    done = tmp_path / "done.json"
    done.write_text(json.dumps({"extras": {}}), encoding="utf-8")
    assert _cache_uses_current_one_pole_data_protocol(tmp_path) is False

    done.write_text(
        json.dumps(
            {
                "extras": {
                    "one_pole_data_protocol_version": ablation.ONE_POLE_DATA_PROTOCOL_VERSION
                }
            }
        ),
        encoding="utf-8",
    )
    assert _cache_uses_current_one_pole_data_protocol(tmp_path) is True


def test_report_shards_merge_split_model_case_jobs(tmp_path):
    protocol = {"lrs": [1e-5], "modes": ["raw"], "causal": False}
    first = {
        "model": "gpt2-small",
        "task": "ioi",
        "feature_class": "IO",
        "mode": "raw",
        "lr": 1e-5,
        "score": 0.8,
    }
    second = {
        "model": "pythia-70m-deduped",
        "task": "language",
        "feature_class": "fr",
        "mode": "raw",
        "lr": 1e-5,
        "score": 0.7,
    }

    _write_report_shards(tmp_path, [first], protocol=protocol)
    merged, table_path = _write_report_shards(tmp_path, [second], protocol=protocol)

    assert len(merged) == 2
    assert {row["model"] for row in merged} == {
        "gpt2-small",
        "pythia-70m-deduped",
    }
    assert table_path.is_file()


def test_causal_all_seeds_uses_selected_best_and_reloads_other_seeds(
    monkeypatch, tmp_path
):
    seen = []

    def fake_reload(**kwargs):
        return {"trainer": f"seed-{kwargs['seed']}"}

    def fake_run(raw, **_kwargs):
        seen.append(raw["trainer"])
        return [{"variant": "gated_all", "value": 0.5}]

    monkeypatch.setattr(ablation, "_reload_seed_for_causal", fake_reload)
    monkeypatch.setattr(ablation, "_run_causal", fake_run)

    rows = ablation._run_cell_causal(
        {"trainer": "selected-best"},
        {"seeds_tried": [2, 3], "best_seed": 3},
        cfg=object(),
        bundle=object(),
        feature_class="IO",
        scale="running_rms",
        lr=1e-4,
        smoke=False,
        experiment_dir=tmp_path,
        max_size=8,
        lrs=[1.0],
        all_seeds=True,
    )

    assert seen == ["seed-2", "selected-best"]
    assert [row["seed"] for row in rows] == [2, 3]
