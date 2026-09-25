"""scripts/failed_jobs.py: failed jobs -> the command that resubmits them (reads logs only)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("failed_jobs", Path(__file__).resolve().parents[1] / "scripts" / "failed_jobs.py")
fj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fj)


def _log(d, name, cmd, tail="", err=""):
    (d / f"{name}.out").write_text(f"header\nRunning: {cmd}\n{tail}\n", encoding="utf-8")
    (d / f"{name}.err").write_text(err, encoding="utf-8")


def test_parse_run_command_pieces():
    p = fj.parse_run_command(
        "python run_study.py --model qwen3.5-27b --task gender_en --suite core --skip-existing --refresh-causal "
        "--methods gradiend agiend --lr-gradiend 1e-6 --output-subdir lr_r1 --tune-lr --train-ablations pair --fail-fast"
    )
    assert p["model"] == "qwen3.5-27b" and p["task"] == "gender_en" and p["methods"] == ["gradiend", "agiend"]
    assert p["lr"] == {"LR_GRADIEND": "1e-6"} and p["output_subdir"] == "lr_r1"
    env = fj.resubmit_env(p)
    assert {"REFRESH_CAUSAL=1", "TUNE_LR=1", "OUTPUT_SUBDIR=lr_r1", "TRAIN_ABLATIONS=pair", "LR_GRADIEND=1e-6"} <= set(env)
    assert "SKIP_EXISTING=0" not in env


def test_a_run_without_skip_existing_is_resubmitted_without_it():
    p = fj.parse_run_command("python run_study.py --model m --task t --suite core --methods caa --fail-fast")
    assert "SKIP_EXISTING=0" in fj.resubmit_env(p)


def test_failures_are_found_grouped_and_successes_ignored(tmp_path):
    base = "python run_study.py --model qwen3.5-2b-base --task {t} --suite core --skip-existing --refresh-causal --methods gradiend agiend"
    _log(tmp_path, "slurm-1_0", base.format(t="race"), tail="FAILED qwen3.5-2b-base/race: boom")
    _log(tmp_path, "slurm-1_1", base.format(t="religion"), tail="FAILED qwen3.5-2b-base/religion: boom")
    _log(tmp_path, "slurm-1_2", base.format(t="language"), tail="all good")
    _log(tmp_path, "slurm-2_0", "python run_study.py --model gemma-3-27b-pt --task gender_en --suite core --methods caa",
         err="slurmstepd: error: Detected 1 oom_kill event")
    failures = fj.find_failures(tmp_path, since=None, job=None)
    assert {f["log"] for f in failures} == {"slurm-1_0.out", "slurm-1_1.out", "slurm-2_0.out"}
    groups = fj.group_commands(failures)
    assert len(groups) == 2
    tasks = sorted(sorted(g["tasks"]) for g in groups.values())
    assert tasks == [["gender_en"], ["race", "religion"]]


def test_job_filter_and_missing_running_line(tmp_path):
    _log(tmp_path, "slurm-5_0", "python run_study.py --model m --task a --suite core", tail="FAILED m/a: x")
    (tmp_path / "slurm-6_0.out").write_text("FAILED m/b: x\n", encoding="utf-8")  # no Running: line -> cannot resubmit
    assert [f["log"] for f in fj.find_failures(tmp_path, since=None, job="5")] == ["slurm-5_0.out"]
    assert fj.find_failures(tmp_path, since=None, job="6") == []
