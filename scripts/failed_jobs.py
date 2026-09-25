#!/usr/bin/env python3
"""List failed study jobs from ``slurm-logs/`` and print the command that resubmits each group.

Reads only the logs (no Slurm access needed, run it wherever ``slurm-logs/`` is synced). A job counts as
failed when its ``.out`` has a ``FAILED <model>/<task>`` line (the study's fail-fast marker) or its ``.err``
has a slurmstepd error (OOM kill, time limit). The exact ``run_study.py`` command of the job is read from its
``Running:`` line, so the resubmit command reproduces its model, task, methods and flags.

    python scripts/failed_jobs.py                 # failures from the last 24 h
    python scripts/failed_jobs.py --since-hours 6
    python scripts/failed_jobs.py --job 4286284   # every task of one array job

It never submits anything: it prints ``bash slurm/study_array.sh`` commands for you to run (after fixing the
cause -- the reason of each failure is shown so a resubmit that would fail again is not sent blindly).
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG_RE = re.compile(r"slurm-(?P<job>\d+)(?:_(?P<task>\d+))?\.out$")
RUNNING_RE = re.compile(r"Running: (python run_study\.py .*)$")
FAILED_RE = re.compile(r"^FAILED (?P<model>[^/\s]+)/(?P<task>\S+?):? (?P<msg>.*)$")
STEPD_RE = re.compile(r"slurmstepd: error: (?P<msg>.*)$")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_run_command(cmd: str) -> Dict[str, object]:
    """``run_study.py`` argv -> the pieces study_array.sh takes as environment."""
    toks = shlex.split(cmd)[2:]
    out: Dict[str, object] = {"methods": [], "flags": set(), "lr": {}, "output_subdir": "", "train_ablations": []}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("--model", "--task", "--suite", "--output-subdir", "--scale", "--max-steps", "--eval-steps"):
            out[t.lstrip("-").replace("-", "_")] = toks[i + 1]
            i += 2
        elif t in ("--methods", "--train-ablations"):
            key = t.lstrip("-").replace("-", "_")
            i += 1
            vals = []
            while i < len(toks) and not toks[i].startswith("--"):
                vals.append(toks[i])
                i += 1
            out[key] = vals
        elif t.startswith("--lr-") and i + 1 < len(toks):
            out["lr"][t[len("--"):].replace("-", "_").upper()] = toks[i + 1]
            i += 2
        else:
            out["flags"].add(t)
            i += 1
    return out


def resubmit_env(parsed: Dict[str, object]) -> List[str]:
    """Environment assignments that make study_array.sh reproduce the job's mode."""
    flags = parsed["flags"]
    env: List[str] = []
    if "--skip-existing" not in flags and "--refresh-causal" not in flags:
        env.append("SKIP_EXISTING=0")
    if "--refresh-causal" in flags:
        env.append("REFRESH_CAUSAL=1")
    if "--force-causal" in flags:
        env.append("FORCE_CAUSAL=1")
    if "--refresh-encoder-eval" in flags:
        env.append("REFRESH_ENCODER_EVAL=1")
    if "--skip-causal" in flags:
        env.append("SKIP_CAUSAL=1")
    if "--tune-lr" in flags:
        env.append("TUNE_LR=1")
    if "--no-rerun-orphans" in flags:
        env.append("NO_RERUN_ORPHANS=1")
    if parsed.get("output_subdir"):
        env.append(f"OUTPUT_SUBDIR={parsed['output_subdir']}")
    if parsed.get("scale"):
        env.append(f"SCALE={parsed['scale']}")
    if parsed.get("train_ablations"):
        env.append("TRAIN_ABLATIONS=" + ",".join(parsed["train_ablations"]))
    for key, val in sorted(parsed["lr"].items()):
        env.append(f"{key}={val}")
    for key in ("max_steps", "eval_steps"):
        if parsed.get(key):
            env.append(f"{key.upper()}={parsed[key]}")
    return env


def find_failures(log_dir: Path, *, since: Optional[float], job: Optional[str]) -> List[Dict[str, object]]:
    failures: List[Dict[str, object]] = []
    for out_path in sorted(log_dir.glob("slurm-*.out")):
        m = LOG_RE.search(out_path.name)
        if not m or (job and m.group("job") != job):
            continue
        if since is not None and out_path.stat().st_mtime < since:
            continue
        text = _read(out_path)
        err_text = _read(out_path.with_suffix(".err"))
        reason = None
        for line in text.splitlines():
            fm = FAILED_RE.match(line.strip())
            if fm:
                reason = fm.group("msg")[:200] or "failed"
        if reason is None:
            sm = [STEPD_RE.search(l) for l in err_text.splitlines()]
            sm = [x for x in sm if x]
            if sm:
                reason = "slurmstepd: " + sm[-1].group("msg")[:160]
        if reason is None:
            continue
        run = [RUNNING_RE.search(l) for l in text.splitlines()]
        run = [x for x in run if x]
        if not run:
            continue
        failures.append({"log": out_path.name, "reason": reason, "cmd": run[-1].group(1), "parsed": parse_run_command(run[-1].group(1))})
    return failures


def group_commands(failures: List[Dict[str, object]]) -> "OrderedDict[Tuple, Dict[str, object]]":
    groups: "OrderedDict[Tuple, Dict[str, object]]" = OrderedDict()
    for f in failures:
        p = f["parsed"]
        env = tuple(resubmit_env(p))
        key = (p.get("model"), p.get("suite", "core"), tuple(p.get("methods") or ()), env)
        g = groups.setdefault(key, {"tasks": [], "logs": []})
        if p.get("task") not in g["tasks"]:
            g["tasks"].append(p.get("task"))
        g["logs"].append((f["log"], f["reason"]))
    return groups


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="slurm-logs")
    ap.add_argument("--since-hours", type=float, default=24.0, help="only logs modified in the last N hours (0 = all)")
    ap.add_argument("--job", default=None, help="only this Slurm job id")
    args = ap.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"no such directory: {log_dir}", file=sys.stderr)
        return 2
    since = time.time() - args.since_hours * 3600 if args.since_hours > 0 else None
    failures = find_failures(log_dir, since=since, job=args.job)
    if not failures:
        print("no failed study jobs found")
        return 0
    groups = group_commands(failures)
    print(f"{len(failures)} failed job(s) in {len(groups)} group(s)\n")
    for (model, suite, methods, env), g in groups.items():
        for log, reason in g["logs"]:
            print(f"# {log}: {reason}")
        parts = list(env) + [f"MODELS={model}", f"TASKS={','.join(g['tasks'])}", f"SUITE={suite}"]
        if methods:
            parts.append(f"METHODS={','.join(methods)}")
        print(" ".join(parts) + " bash slurm/study_array.sh\n")
    print("Check the reasons above first: a resubmit only helps if the cause is fixed and synced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
