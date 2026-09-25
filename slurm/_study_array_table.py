"""Build the job table used by ``study_array.sh``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.config import (
    METHOD_FAMILIES,
    list_tasks,
    load_study_config,
    normalize_train_ablations,
    parse_methods_arg,
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_slurm_profile_defaults(path: Path) -> dict:
    """Load scheduler-only model/method defaults (never study config)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_slurm_profile(
    model: str,
    method: str,
    profile_defaults: Mapping,
    *,
    override: Optional[str] = None,
) -> str:
    """Resolve explicit override, then model/method, then method default."""
    if override:
        return str(override)
    model_profiles = (profile_defaults.get("models") or {}).get(model) or {}
    profile = model_profiles.get(method)
    if profile is None:
        profile = (profile_defaults.get("defaults") or {}).get(method)
    if not profile:
        raise ValueError(f"No Slurm profile configured for model={model} method={method}")
    return str(profile)


def profile_groups(jobs: Sequence[Mapping]) -> list[tuple[str, str]]:
    """Return ``(profile, comma-separated array ids)`` preserving table order."""
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for job in jobs:
        profile = str(job.get("slurm_profile") or "").strip()
        if not profile:
            raise ValueError(f"Job {job.get('index')} has no slurm_profile")
        groups.setdefault(profile, []).append(str(int(job["index"])))
    return [(profile, ",".join(indices)) for profile, indices in groups.items()]


def build_jobs(
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    suite: str,
    output_subdir: str = "",
    array_by_method: bool = False,
    methods: Optional[Iterable[str]] = None,
    slurm_profile_defaults: Optional[Mapping] = None,
    slurm_profile_override: Optional[str] = None,
    train_ablations: Optional[Iterable[str]] = None,
) -> list[dict]:
    """Return model/task rows, optionally crossed with enabled suite methods.

    ``train_ablations`` (``pair`` / ``one_pole``) drops tasks that do not train that
    slice at all (e.g. ``pair`` for the one-pole-only circuit tasks), so no GPU job is
    scheduled just to train nothing.
    """
    requested = parse_methods_arg(list(methods) if methods is not None else None)
    requested_set = set(requested or [])
    slice_filter = normalize_train_ablations(
        list(train_ablations) if train_ablations is not None else None
    )
    rows: list[dict] = []
    for model in models:
        for task in tasks:
            if slice_filter is not None:
                abl = load_study_config(model=model, task=task, suite=suite).raw.get(
                    "ablations"
                ) or {}
                # same defaults as study.stages.train._ablations (both on unless disabled)
                enabled = {
                    name for name in ("pair", "one_pole") if bool(abl.get(name, True))
                }
                if not (enabled & slice_filter):
                    print(
                        f"skip {model}/{task}: no {sorted(slice_filter)} trainers "
                        f"(ablations={dict(abl)})",
                        flush=True,
                    )
                    continue
            if not array_by_method:
                rows.append(
                    {
                        "index": len(rows),
                        "model": model,
                        "task": task,
                        "suite": suite,
                        "output_subdir": output_subdir,
                    }
                )
                continue

            cfg = load_study_config(model=model, task=task, suite=suite)
            configured = cfg.suite.get("methods") or {}
            families = [
                family
                for family in METHOD_FAMILIES
                # actiend_ridge is produced by the ACTIEND cell; it is not an
                # independently trained suite family.
                if family != "actiend_ridge"
                and configured.get(family)
                and (not requested_set or family in requested_set)
            ]
            for family in families:
                row = {
                    "index": len(rows),
                    "model": model,
                    "task": task,
                    "suite": suite,
                    "methods": family,
                    "output_subdir": output_subdir,
                }
                if slurm_profile_defaults is not None:
                    row["slurm_profile"] = resolve_slurm_profile(
                        model,
                        family,
                        slurm_profile_defaults,
                        override=slurm_profile_override,
                    )
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--array-by-method", action="store_true")
    parser.add_argument("--methods", default="")
    parser.add_argument("--train-ablations", default="")
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--slurm-profile", default="")
    parser.add_argument("--profile-groups-output", type=Path)
    args = parser.parse_args()

    tasks = list_tasks() if args.tasks == "all" else _csv(args.tasks)
    method_filter = _csv(args.methods) if args.methods else None
    profile_defaults = None
    if args.array_by_method and args.profile_config:
        profile_defaults = load_slurm_profile_defaults(args.profile_config)
    jobs = build_jobs(
        models=_csv(args.models),
        tasks=tasks,
        suite=args.suite,
        output_subdir=args.output_subdir,
        array_by_method=args.array_by_method,
        methods=method_filter,
        slurm_profile_defaults=profile_defaults,
        slurm_profile_override=args.slurm_profile or None,
        train_ablations=_csv(args.train_ablations) if args.train_ablations else None,
    )
    args.output.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")
    if args.profile_groups_output:
        groups = profile_groups(jobs)
        args.profile_groups_output.write_text(
            "".join(f"{profile}\t{indices}\n" for profile, indices in groups),
            encoding="utf-8",
        )
    print(f"Wrote {args.output} with {len(jobs)} jobs")


if __name__ == "__main__":
    main()
