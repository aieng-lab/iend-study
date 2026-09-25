#!/usr/bin/env python
"""Study job entrypoint: one task, or all tasks for a model.

Examples:
  python run_study.py --model gpt2-small
  python run_study.py --model gpt2-small --suite full --skip-existing
  python run_study.py --model pythia-70m-deduped --task gender_en --suite full
  python run_study.py --model gpt2-small --task race --smoke
  python run_study.py --model gpt2-small --task race_one_pole --methods actiend
  # Stop a multi-task run at its first unhandled task error, with its traceback:
  python run_study.py --model gpt2-small --fail-fast
  # SAE only (still runs causal + localization unless --skip-*):
  python run_study.py --model gpt2-small --task key_value --methods sae
  # Encode without causal / localization:
  python run_study.py --model gpt2-small --methods actiend --skip-causal --skip-localization
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure repo root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from study.config import METHOD_FAMILIES, list_models, list_suites, list_tasks
from study.runner import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list_models())
    parser.add_argument(
        "--task",
        default=None,
        # Include study_hidden tasks (e.g. ioi) so they load when NAMED explicitly;
        # the TASKS=all default below still uses the visible-only list_tasks(), so a
        # hidden task never enters a bulk sweep. Without this, `--task ioi` fast-fails
        # with "invalid choice: 'ioi'" (jobs 4090138_84..).
        choices=list_tasks(include_hidden=True),
        help="Task id. If omitted, run all tasks for --model sequentially.",
    )
    parser.add_argument("--suite", default=None, choices=list_suites() or None)
    parser.add_argument("--lr", type=float, default=None, help="Override shared learning rate (both backends)")
    parser.add_argument(
        "--lr-gradiend",
        type=float,
        default=None,
        help="Override GRADIEND learning rate only",
    )
    parser.add_argument(
        "--lr-actiend",
        type=float,
        default=None,
        help="Override ACTIEND learning rate only",
    )
    parser.add_argument(
        "--lr-agiend",
        type=float,
        default=None,
        help="Override AGIEND learning rate only",
    )
    parser.add_argument(
        "--lr-decoder-actiend",
        type=float,
        default=None,
        help=(
            "Opt-in decoupled ACTIEND decoder learning rate. Unset (default) "
            "keeps encoder and decoder on one shared rate, exactly as before."
        ),
    )
    parser.add_argument(
        "--lr-decoder-gradiend",
        type=float,
        default=None,
        help="Opt-in decoupled GRADIEND decoder learning rate (default: shared).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Token window for training / ACTIEND filled templates (default from YAML, usually 128)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the training step budget from the study configuration",
    )
    parser.add_argument(
        "--pre-prune-topk",
        type=float,
        default=None,
        help=(
            "Override training.pre_prune.topk for this invocation only (CLI "
            "override merges last, per-run -- never touches the checked-in "
            "model YAML). GRADIEND-only lever for its dominant training-time "
            "memory cost (encoder/decoder + Adam state scale with the kept "
            "dim count). Deliberately does NOT affect CGA even when "
            "cga_pre_prune=true in the model config: CGA reuses this same "
            "field intentionally (to share GRADIEND's coordinate projection), "
            "so shrinking it here would silently shrink CGA's coordinate "
            "space too with no compensating post-prune step -- pass "
            "--methods gradiend (not cga) when using this flag."
        ),
    )
    parser.add_argument(
        "--post-prune-topk",
        type=float,
        default=None,
        help=(
            "Override training.post_prune.topk for this invocation only (see "
            "--pre-prune-topk). GRADIEND's post-prune runs AFTER training on "
            "the already pre-pruned dimension set, so raising this by the "
            "same factor --pre-prune-topk was lowered keeps the final "
            "non-zero decoder-weight count unchanged while training on fewer "
            "dims (e.g. pre=0.01->0.001, post=0.01->0.1)."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Override the checkpoint/evaluation interval from the study configuration",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop this worker at the first task, training, or causal-evaluation "
            "error and re-raise it with its original traceback. By default, "
            "failed method evaluations are recorded and the remaining work continues."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=["small", "large"],
        default=None,
        help="Dataset scale profile: 'small' for quick iteration, 'large' for full evaluation (default: use defaults.yaml values)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Resume: skip whole task if results.json status=ok; otherwise skip each "
            "completed GRADIEND/ACTIEND artifact (done.json or promoted checkpoint) "
            "and reload instead of retraining"
        ),
    )
    parser.add_argument(
        "--refresh-causal",
        action="store_true",
        help=(
            "Re-enter status=ok tasks: still reload train with --skip-existing "
            "(no retrain), but do not skip the whole run — re-run causal. "
            "Use after causal-protocol fixes (Slurm: REFRESH_CAUSAL=1)."
        ),
    )
    parser.add_argument(
        "--refresh-encoder-eval",
        action="store_true",
        help=(
            "Re-enter status=ok tasks and recompute encoder/detection metrics from "
            "saved train artifacts, without retraining. Combine with --skip-causal "
            "and --skip-localization for a detection-only refresh (Slurm: "
            "REFRESH_ENCODER_EVAL=1). Idempotent: only artifacts/SAE/CAA blocks not "
            "yet stamped with the current encoder_eval_rules_version (validation-"
            "frozen Spec_n/Excl) are recomputed, so a resumed refresh skips finished "
            "work. CGA/CAGA/CAA reuse their stored validation readouts (and CAA its "
            "stored vectors) and score the test split only."
        ),
    )
    parser.add_argument(
        "--train-ablations",
        nargs="+",
        default=None,
        metavar="ABLATION",
        help=(
            "Train/causal only this slice of the task's trainers: 'pair' and/or "
            "'one_pole' (space- or comma-separated). An execution filter, not part "
            "of the config hash: launch one job per slice on separate GPUs, "
            "results.json is merged under a file lock. Slurm: TRAIN_ABLATIONS=pair."
        ),
    )
    parser.add_argument(
        "--tune-lr",
        action="store_true",
        help=(
            "Before training GRADIEND/ACTIEND/AGIEND, search the learning rate with the package's LR "
            "finder (the yaml rate is the first guess; short probes are classified and bracketed, a "
            "converged probe is confirmed at the full step budget) and train at the chosen rate. "
            "Fails when no learning rate converges. The trace is written to <artifact>/lr_search.json "
            "and summarised in done.json. An execution switch, not part of the config hash. "
            "Slurm: TUNE_LR=1."
        ),
    )
    parser.add_argument(
        "--causal-only",
        nargs="+",
        default=None,
        metavar="PRESET_OR_REGEX",
        help=(
            "Compute ONLY the causal method ids named by these presets/regexes and keep every "
            "stored sweep. Implies --refresh-causal and disables core's primary_only budget for "
            "the named ids. Preset 'layerwise' = SAE k1 + CAA act_prediction at every layer and "
            "the all-layer aggregates (+ the best-layer CGA/CAGA ids). An execution filter, not "
            "part of the config hash. Slurm: CAUSAL_ONLY=layerwise."
        ),
    )
    parser.add_argument(
        "--no-rerun-orphans",
        action="store_true",
        help=(
            "With --skip-existing: never retrain a finished artifact just because no result row "
            "points at it. Re-entering an existing task always counts as 'config changed', which "
            "retrains such 'orphans'; use this to rebuild rows lost from results.json (e.g. by "
            "the pre-2026-09-24 slice-merge bug) from the intact done.json instead. Slurm: "
            "NO_RERUN_ORPHANS=1."
        ),
    )
    parser.add_argument(
        "--force-causal",
        action="store_true",
        help=(
            "Re-evaluate every completed causal sweep for the selected methods. "
            "Implies --refresh-causal and reloads fitted artifacts without retraining."
        ),
    )
    parser.add_argument(
        "--methods",
        action="append",
        nargs="+",
        default=None,
        metavar="METHOD",
        help=(
            "Restrict methods (space- or comma-separated). Repeatable. "
            f"Choices: {', '.join(METHOD_FAMILIES)}. "
            "Causal and localization are stages (not methods): they still run "
            "unless --skip-causal / --skip-localization. "
            "Example: --methods sae   or   --methods gradiend,actiend"
        ),
    )
    parser.add_argument(
        "--iend-selection-metric",
        choices=["default", "correlation", "min_auc_n_o", "encoding_e"],
        default=None,
        help=(
            "Checkpoint/seed selector for GRADIEND and ACTIEND. 'encoding_e' "
            "uses the same validation E as SAE/CAA; convergence remains governed "
            "by convergent_metric. 'default' preserves the configured behavior."
        ),
    )
    parser.add_argument(
        "--iend-convergent-metric",
        choices=["correlation", "roc_auc", "min_auc_n_o"],
        default=None,
        help="Override the IEND convergence metric independently of selection.",
    )
    parser.add_argument(
        "--train-cache-mode",
        choices=["always", "hash_match"],
        default=None,
        help=(
            "Per-run override of training.train_cache_mode. 'hash_match' retrains "
            "any train artifact whose config hash changed (e.g. the one-pole "
            "source flip) while reusing pairs/SAE/CAA; 'always' trusts existing "
            "checkpoints regardless of hash (the default)."
        ),
    )
    parser.add_argument(
        "--sae-disk-cache",
        choices=["on", "off"],
        default=None,
        help=(
            "Per-run override of training.sae_disk_cache. 'off' skips writing/"
            "reading sae_cache/ latent shards entirely (for storage-constrained "
            "clusters running each SAE task once); 'on' restores the default "
            "skip-on-recompute cache."
        ),
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        choices=["on", "off"],
        default=None,
        help=(
            "Per-run override of training.cleanup_checkpoints. 'on' deletes this "
            "cell's trained-weight files after it fully finishes (causal ran, no "
            "errors); for storage-constrained clusters. Env fallback: "
            "CLEANUP_CHECKPOINTS=on|off."
        ),
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        choices=["none", "tensors"],
        default=None,
        metavar="SPLIT",
        help=(
            "Restrict GRADIEND/ACTIEND training to selected splits. "
            "Causal evaluation remains enabled for the freshly trained splits. "
            "Example: --methods actiend --train-splits tensors"
        ),
    )
    parser.add_argument(
        "--skip-sae",
        action="store_true",
        help="Disable SAE when running the full default method set (ignored if --methods is set).",
    )
    parser.add_argument(
        "--skip-caa",
        action="store_true",
        help="Disable CAA when running the full default method set (ignored if --methods is set).",
    )
    parser.add_argument(
        "--skip-causal",
        action="store_true",
        help="Skip the causal intervention stage (applies to all selected methods).",
    )
    parser.add_argument(
        "--skip-localization",
        action="store_true",
        help="Skip the localization stage.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help=(
            "Skip plots/REPORT/TABLES generation after persisting results.json. "
            "GPU Slurm launchers enable this by default; regenerate reports in "
            "a separate CPU job."
        ),
    )
    parser.add_argument(
        "--rerun-collapsed",
        action="store_true",
        help=(
            "With --skip-existing: retrain GRADIEND/ACTIEND artifacts whose encoder "
            "was chance-level (status=collapsed / done.json collapsed_encoder). "
            "Distinct from hard train errors."
        ),
    )
    parser.add_argument(
        "--save-modified-models",
        action="store_true",
        help="Persist causal rewrites under modified_models/ (off by default)",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Write under runs/{model}/{subdir}/{task} instead of runs/{model}/{task}. "
            "Use to keep suite=full (or other ablations) out of an ongoing core tree. "
            "Slurm: OUTPUT_SUBDIR=suite_full."
        ),
    )
    args = parser.parse_args()
    if args.causal_only:
        args.refresh_causal = True

    tasks = [args.task] if args.task else list_tasks()
    cli_overrides: dict = {
        "methods": args.methods,
        "train_splits": args.train_splits,
        "no_rerun_orphaned_artifacts": bool(args.no_rerun_orphans),
        "tune_lr": bool(args.tune_lr),
        "train_ablations": (
            [p for a in args.train_ablations for p in str(a).replace(",", " ").split()]
            if args.train_ablations
            else None
        ),
        "causal_only": (
            [p for a in args.causal_only for p in str(a).replace(",", " ").split()]
            if args.causal_only
            else None
        ),
        "fail_fast": args.fail_fast,
        "skip_existing": (
            args.skip_existing
            or args.refresh_causal
            or args.force_causal
            or args.refresh_encoder_eval
        ),
        "refresh_causal": args.refresh_causal,
        "force_causal": args.force_causal,
        "refresh_encoder_eval": args.refresh_encoder_eval,
        "rerun_collapsed": args.rerun_collapsed,
        "skip_sae": args.skip_sae,
        "skip_caa": args.skip_caa,
        "skip_causal": args.skip_causal,
        "skip_localization": args.skip_localization,
        "skip_reports": args.skip_reports,
        "save_modified_models": args.save_modified_models,
    }
    if args.output_subdir:
        cli_overrides["output_subdir"] = str(args.output_subdir).strip()
    training_cli: dict = {}
    if args.lr_gradiend is not None:
        training_cli["learning_rate_gradiend"] = float(args.lr_gradiend)
    if args.lr_actiend is not None:
        training_cli["learning_rate_actiend"] = float(args.lr_actiend)
    if args.lr_agiend is not None:
        training_cli["learning_rate_agiend"] = float(args.lr_agiend)
    if args.lr_decoder_actiend is not None:
        training_cli["learning_rate_decoder_actiend"] = float(args.lr_decoder_actiend)
    if args.lr_decoder_gradiend is not None:
        training_cli["learning_rate_decoder_gradiend"] = float(args.lr_decoder_gradiend)
    if args.max_length is not None:
        training_cli["max_length"] = int(args.max_length)
    if args.pre_prune_topk is not None:
        training_cli["pre_prune"] = {"topk": float(args.pre_prune_topk)}
    if args.post_prune_topk is not None:
        training_cli["post_prune"] = {"topk": float(args.post_prune_topk)}
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be at least 1")
        training_cli["max_steps"] = int(args.max_steps)
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            parser.error("--eval-steps must be at least 1")
        training_cli["eval_steps"] = int(args.eval_steps)
    if args.iend_selection_metric and args.iend_selection_metric != "default":
        training_cli["selection_metric"] = args.iend_selection_metric
    if args.iend_convergent_metric:
        training_cli["convergent_metric"] = args.iend_convergent_metric
    if args.train_cache_mode:
        training_cli["train_cache_mode"] = args.train_cache_mode
    # CLI flag wins; otherwise the env var (set by slurm/gradiend_sae_lrz.env.sh),
    # so jobs whose TRAIN_CMD was built without the flag still get the setting.
    def _on_off(flag, env_name):
        value = flag or os.environ.get(env_name)
        return value if value in ("on", "off") else None

    sae_disk_cache = _on_off(args.sae_disk_cache, "SAE_DISK_CACHE")
    if sae_disk_cache:
        training_cli["sae_disk_cache"] = sae_disk_cache == "on"
    cleanup_checkpoints = _on_off(args.cleanup_checkpoints, "CLEANUP_CHECKPOINTS")
    if cleanup_checkpoints:
        training_cli["cleanup_checkpoints"] = cleanup_checkpoints == "on"
    if training_cli:
        cli_overrides["training"] = training_cli
    failures = []
    for task in tasks:
        print(f"\n=== model={args.model} task={task} ===")
        try:
            run_study(
                model=args.model,
                task=task,
                suite=args.suite,
                smoke=args.smoke,
                scale=args.scale,
                skip_existing=(
                    args.skip_existing
                    or args.refresh_causal
                    or args.force_causal
                    or args.refresh_encoder_eval
                ),
                refresh_causal=args.refresh_causal,
                force_causal=args.force_causal,
                lr=args.lr,
                cli_overrides=cli_overrides,
            )
        except Exception as exc:
            print(f"FAILED {args.model}/{task}: {exc}")
            failures.append((task, str(exc)))
            if args.fail_fast or args.task is not None:
                raise

    if failures:
        print("\nCompleted with failures:")
        for task, err in failures:
            print(f"  {task}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
