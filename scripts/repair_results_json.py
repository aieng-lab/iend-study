"""Repair results.json when the pipeline ran but final JSON write failed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.json_util import json_ready
from study.stages.sae import _load_sae_encode_snapshot
from study.config import load_study_config


def _load_causal_summary(run_dir: Path) -> list:
    path = run_dir / "causal" / "summary.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("results") or payload.get("methods") or [])


def _merge_causal(method_rows: list, causal_rows: list) -> list:
    by_id = {str(r.get("method")): r for r in method_rows if r.get("method")}
    for crow in causal_rows:
        mid = str(crow.get("method") or "")
        if not mid:
            continue
        if mid in by_id:
            row = by_id[mid]
            row.setdefault("metrics", {}).update(crow.get("metrics") or {})
            extras = row.setdefault("extras", {})
            for k, v in (crow.get("extras") or {}).items():
                extras[k] = v
        else:
            method_rows.append(crow)
            by_id[mid] = crow
    return method_rows


def repair_run(run_dir: Path, *, model: str, task: str, suite: str | None) -> bool:
    cfg = load_study_config(model=model, task=task, suite=suite)
    if cfg.output_dir.resolve() != run_dir.resolve():
        # Allow explicit run_dir override while keeping config metadata.
        pass
    sae_rows = _load_sae_encode_snapshot(cfg)
    if not sae_rows:
        print(f"{run_dir}: no artifacts/sae/encode_method_rows.json", file=sys.stderr)
        return False
    pre = [r for r in sae_rows if str(r.get("method", "")).startswith("sae_pre:")]
    print(f"{run_dir}: restoring {len(sae_rows)} SAE encode rows ({len(pre)} _pre)")
    method_rows = list(sae_rows)
    method_rows = _merge_causal(method_rows, _load_causal_summary(run_dir))
    results_path = run_dir / "results.json"
    previous = {}
    if results_path.is_file():
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    payload = dict(previous)
    payload.update(
        {
            "model": model,
            "task": task,
            "suite": suite or payload.get("suite") or cfg.suite_id,
            "config_hash": cfg.config_hash(),
            "status": "partial",
            "methods": method_rows,
            "repaired_from": ["artifacts/sae/encode_method_rows.json", "causal/summary.json"],
        }
    )
    results_path.write_text(
        json.dumps(json_ready(payload), indent=2), encoding="utf-8"
    )
    print(f"Wrote {results_path} methods={len(method_rows)}")
    try:
        from plot_study import write_all_study_plots
        from report_study import write_study_report

        plots = write_all_study_plots(payload, run_dir)
        write_study_report(payload, run_dir, plot_paths=plots)
        print(f"Regenerated REPORT.md under {run_dir}")
    except Exception as exc:
        print(f"Report/plot regen skipped: {exc}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="e.g. runs/gpt2-small/language")
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--suite", default=None)
    args = p.parse_args()
    ok = repair_run(args.run_dir, model=args.model, task=args.task, suite=args.suite)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
