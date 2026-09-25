#!/usr/bin/env python
"""Prime a runs tree for a SURGICAL one-pole rerun under source="both".

The one-pole source flip (alternative -> both, study/stages/train.py::
one_pole_train_source) is invisible to the *task* config_hash, so a completed
(status=ok) task would be skipped whole by --skip-existing, and its already-ok
one-pole causal rows would be kept as-is (stale, alternative-source). This
script fixes both, touching ONLY one-pole GRADIEND/ACTIEND rows:

  1. status "ok" -> "partial"        (forces --skip-existing to re-enter)
  2. drop one-pole ids from raw.causal.by_method / summaries
                                     (so the causal stage recomputes them;
                                      pairs stay ok -> skipped)
  3. clear the one-pole method rows' causal metrics
                                     (recompute overwrites; avoids stale display)

Pairs, tensors, SAE, CAA, localization rows are untouched -> reused by
hash_match (train) + --skip-existing (causal) + METHODS filter. One-pole train
artifacts retrain on their own because their artifact hash changed (source).

Idempotent. --dry-run to preview. Run on the tree the rerun reads (the cluster
checkout), e.g.:

    python scripts/reset_one_pole_for_rerun.py runs/gpt2-small/suite_s100 \
        runs/pythia-70m-deduped/suite_s100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# One-pole GRADIEND/ACTIEND ids: backend:<class>[:tok_*][|proxy=...]. The
# distinguishing feature vs a pair is the ABSENCE of a pair separator '-'
# (a pair id embeds 'A-B'); ACTIEND one-pole ids carry a :tok_* suffix, so a
# "no extra ':'" rule wrongly excludes them (actiend:F:tok_all). Class names use
# underscores, never '-', so '-' reliably marks a pair.
ONE_POLE_RE = re.compile(r"^(gradiend|actiend):(?!.*-).+$")


def _one_pole_matcher(backends):
    bk = backends or ["gradiend", "actiend"]
    alt = "|".join(re.escape(b) for b in bk)
    return re.compile(rf"^({alt}):(?!.*-).+$")

CAUSAL_METRIC_KEYS = (
    "causal_signed_effect",
    "causal_strength",
    "causal_lms",
    "causal_lms_ok",
    "causal_grid_floor",
    "causal_grid_ceiling",
    "causal_null_effect",
    "causal_random_signed_effect",
    "causal_lms_curve",
    "causal_error",
)


def _is_one_pole(mid: str, matcher: "re.Pattern" = ONE_POLE_RE) -> bool:
    return bool(matcher.match(str(mid)))


def _is_full_backend(mid: str, full_backends) -> bool:
    if not full_backends:
        return False
    core = str(mid)
    return any(core.startswith(f"{b}:") for b in full_backends)


def reset_payload(
    payload: dict,
    matcher: "re.Pattern" = ONE_POLE_RE,
    full_backends=(),
) -> dict:
    """Return {counts} describing the in-place edits made."""
    counts = {"status_reset": 0, "causal_dropped": 0, "rows_cleared": 0}

    if payload.get("status") == "ok":
        payload["status"] = "partial"
        counts["status_reset"] = 1

    raw = payload.get("raw")
    causal = (raw or {}).get("causal") if isinstance(raw, dict) else None
    if isinstance(causal, dict):
        by = causal.get("by_method")
        if isinstance(by, dict):
            for mid in [
                m for m in by
                if _is_one_pole(m, matcher) or _is_full_backend(m, full_backends)
            ]:
                del by[mid]
                counts["causal_dropped"] += 1
        summaries = causal.get("summaries")
        if isinstance(summaries, list):
            causal["summaries"] = [
                s
                for s in summaries
                if not (
                    isinstance(s, dict)
                    and (
                        _is_one_pole(s.get("method", ""), matcher)
                        or _is_full_backend(s.get("method", ""), full_backends)
                    )
                )
            ]

    for row in payload.get("methods") or []:
        mid = row.get("method", "")
        if not (_is_one_pole(mid, matcher) or _is_full_backend(mid, full_backends)):
            continue
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and any(k in metrics for k in CAUSAL_METRIC_KEYS):
            for k in CAUSAL_METRIC_KEYS:
                metrics.pop(k, None)
            counts["rows_cleared"] += 1
        extras = row.get("extras")
        if isinstance(extras, dict):
            extras.pop("causal", None)
            extras.pop("causal_lms_curve", None)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="Runs subtree(s) to prime.")
    ap.add_argument(
        "--backend",
        action="append",
        choices=["gradiend", "actiend"],
        help="Only invalidate this backend's one-pole rows (repeatable). "
        "Default: both. Use --backend actiend when rerunning ACTIEND only in a "
        "tree whose GRADIEND you are NOT redoing (and vice versa), so the other "
        "backend's one-pole causal is not wiped without being recomputed.",
    )
    ap.add_argument(
        "--all",
        dest="all_backends",
        action="append",
        choices=["gradiend", "actiend"],
        help="Drop ALL causal (pairs AND one-pole) for this backend, not just "
        "one-pole. Use when the backend's whole config changed (e.g. ACTIEND's "
        "step/LR/decoder-LR defaults), so its pairs must be recomputed too.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-hidden",
        action="store_true",
        help="Also reset tasks hidden from list_tasks() (e.g. race_one_pole). "
        "By default they are skipped, since the TASKS=all rerun never revisits "
        "them and resetting would strand them at status=partial.",
    )
    args = ap.parse_args()

    matcher = _one_pole_matcher(args.backend)

    listed = None
    if not args.include_hidden:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from study.config import list_tasks

        listed = set(list_tasks())

    paths = []
    for root in args.roots:
        for pth in sorted(Path(root).glob("*/results.json")):
            if listed is not None and pth.parent.name not in listed:
                print(f"{pth}: skipped (hidden/not in list_tasks())")
                continue
            paths.append(pth)
    if not paths:
        print("No results.json found under the given roots.", file=sys.stderr)
        return 1

    grand = {"status_reset": 0, "causal_dropped": 0, "rows_cleared": 0}
    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        counts = reset_payload(payload, matcher, tuple(args.all_backends or ()))
        touched = any(counts.values())
        if touched and not args.dry_run:
            p.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
        for k in grand:
            grand[k] += counts[k]
        flag = "" if touched else "  (nothing to do)"
        print(
            f"{p}: status_reset={counts['status_reset']} "
            f"causal_dropped={counts['causal_dropped']} "
            f"rows_cleared={counts['rows_cleared']}{flag}"
        )
    print(
        f"\nTotal: status_reset={grand['status_reset']} "
        f"causal_dropped={grand['causal_dropped']} rows_cleared={grand['rows_cleared']}"
        f"{'  (dry run, nothing written)' if args.dry_run else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
