"""Map SLURM_ARRAY_TASK_ID → job_table row and run ``run_study.py``.

Invoked by ``slurm/_array_task.sh`` (no bash heredoc — CRLF-safe).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def main() -> int:
    table = json.loads(
        Path(os.environ["SUBMIT_DIR"], "job_table.json").read_text(encoding="utf-8")
    )
    jobs = table["jobs"]
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if idx < 0 or idx >= len(jobs):
        print(f"array index {idx} out of range 0..{len(jobs)-1}", file=sys.stderr)
        return 1
    job = jobs[idx]
    model, task, suite = job["model"], job["task"], job.get("suite") or "core"
    fail_fast = _enabled(os.environ.get("FAIL_FAST"))
    subdir = str(job.get("output_subdir") or os.environ.get("OUTPUT_SUBDIR") or "").strip()
    subdir = subdir.replace("\\", "/")
    # REFRESH_CAUSAL=1: re-enter status=ok jobs but still --skip-existing (reload
    # train artifacts; do not retrain). Used after causal-protocol fixes.
    refresh_causal = os.environ.get("REFRESH_CAUSAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    refresh_encoder_eval = _enabled(os.environ.get("REFRESH_ENCODER_EVAL"))
    force_causal = _enabled(os.environ.get("FORCE_CAUSAL"))
    skip_existing = os.environ.get("SKIP_EXISTING_FLAG") == "1"
    # Do not make task-level skip decisions here. The runner compares the
    # requested resolved config to results.json; suite changes can add work.
    cmd = ["python", "run_study.py", "--model", model, "--task", task, "--suite", suite]
    scale = os.environ.get("SCALE") or job.get("scale")
    if scale:
        cmd.extend(["--scale", scale])
    # Reload train checkpoints whenever skipping full retrain or a refresh stage.
    if skip_existing or refresh_causal or force_causal or refresh_encoder_eval:
        cmd.append("--skip-existing")
    if force_causal:
        cmd.append("--force-causal")
    elif refresh_causal:
        # Critical: without this, run_study(--skip-existing) still exits on status=ok
        # in ~20s and never re-runs causal (REFRESH_CAUSAL looked like a no-op).
        cmd.append("--refresh-causal")
    if refresh_encoder_eval:
        cmd.append("--refresh-encoder-eval")
    methods = os.environ.get("METHODS") or job.get("methods")
    if methods:
        # One --methods flag: argparse nargs="+" last-wins if we pass
        # --methods gradiend --methods actiend (only actiend would run).
        parts = [m.strip() for m in str(methods).replace(",", " ").split() if m.strip()]
        if parts:
            cmd.append("--methods")
            cmd.extend(parts)
    train_splits = os.environ.get("TRAIN_SPLITS") or job.get("train_splits")
    if train_splits:
        parts = [
            item.strip()
            for item in str(train_splits).replace(",", " ").split()
            if item.strip()
        ]
        if parts:
            cmd.append("--train-splits")
            cmd.extend(parts)
    train_ablations = os.environ.get("TRAIN_ABLATIONS") or job.get("train_ablations")
    if train_ablations:
        parts = [
            item.strip()
            for item in str(train_ablations).replace(",", " ").split()
            if item.strip()
        ]
        if parts:
            cmd.append("--train-ablations")
            cmd.extend(parts)
    causal_only = os.environ.get("CAUSAL_ONLY") or job.get("causal_only")
    if causal_only:
        parts = [item.strip() for item in str(causal_only).replace(",", " ").split() if item.strip()]
        if parts:
            cmd.append("--causal-only")
            cmd.extend(parts)
    if _enabled(os.environ.get("NO_RERUN_ORPHANS")) or _enabled(job.get("no_rerun_orphans")):
        cmd.append("--no-rerun-orphans")
    if _enabled(os.environ.get("TUNE_LR")) or _enabled(job.get("tune_lr")):
        cmd.append("--tune-lr")
    lr_gradiend = os.environ.get("LR_GRADIEND") or job.get("lr_gradiend")
    if lr_gradiend:
        cmd.extend(["--lr-gradiend", str(lr_gradiend)])
    lr_actiend = os.environ.get("LR_ACTIEND") or job.get("lr_actiend")
    if lr_actiend:
        cmd.extend(["--lr-actiend", str(lr_actiend)])
    lr_agiend = os.environ.get("LR_AGIEND") or job.get("lr_agiend")
    if lr_agiend:
        cmd.extend(["--lr-agiend", str(lr_agiend)])
    lr_decoder_actiend = os.environ.get("LR_DECODER_ACTIEND") or job.get(
        "lr_decoder_actiend"
    )
    if lr_decoder_actiend:
        cmd.extend(["--lr-decoder-actiend", str(lr_decoder_actiend)])
    lr_decoder_gradiend = os.environ.get("LR_DECODER_GRADIEND") or job.get(
        "lr_decoder_gradiend"
    )
    if lr_decoder_gradiend:
        cmd.extend(["--lr-decoder-gradiend", str(lr_decoder_gradiend)])
    max_steps = os.environ.get("MAX_STEPS") or job.get("max_steps")
    if max_steps:
        cmd.extend(["--max-steps", str(max_steps)])
    eval_steps = os.environ.get("EVAL_STEPS") or job.get("eval_steps")
    if eval_steps:
        cmd.extend(["--eval-steps", str(eval_steps)])
    selection_metric = os.environ.get("IEND_SELECTION_METRIC") or job.get(
        "iend_selection_metric"
    )
    if selection_metric:
        cmd.extend(["--iend-selection-metric", str(selection_metric)])
    convergent_metric = os.environ.get("IEND_CONVERGENT_METRIC") or job.get(
        "iend_convergent_metric"
    )
    if convergent_metric:
        cmd.extend(["--iend-convergent-metric", str(convergent_metric)])
    train_cache_mode = os.environ.get("TRAIN_CACHE_MODE") or job.get("train_cache_mode")
    if train_cache_mode:
        cmd.extend(["--train-cache-mode", str(train_cache_mode)])
    if os.environ.get("SKIP_CAUSAL") == "1":
        cmd.append("--skip-causal")
    if os.environ.get("SKIP_LOCALIZATION") == "1":
        cmd.append("--skip-localization")
    if _enabled(os.environ.get("SKIP_REPORTS", "1")):
        cmd.append("--skip-reports")
    if subdir:
        cmd.extend(["--output-subdir", subdir])
    if _enabled(os.environ.get("SMOKE")) or _enabled(job.get("smoke")):
        cmd.append("--smoke")
    if fail_fast:
        cmd.append("--fail-fast")
    if force_causal:
        print(
            f"force-causal: re-enter {model}/{task} with --skip-existing (no retrain)",
            flush=True,
        )
    elif refresh_causal:
        print(
            f"refresh-causal: re-enter {model}/{task} with --skip-existing (no retrain)",
            flush=True,
        )
    elif refresh_encoder_eval:
        print(
            f"refresh-encoder-eval: re-enter {model}/{task} with --skip-existing (no retrain)",
            flush=True,
        )
    print("Running:", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
