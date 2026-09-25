# Study Slurm launchers (bash-only, mirrors gradiend/slurm).

## Setup

1. Edit `gradiend_sae.env.sh` paths for your cluster (`USER_DIR`, `REPO_DIR`, conda, Apptainer).
2. Ensure `gradiend` is importable (bind `GRADIEND_REPO_DIR` or install into the conda env).

## Modes

```bash
# One (model, task) — default suite is core
SKIP_EXISTING=0 bash slurm/study_single.sh gpt2-small gender_en core

# All tasks for one model (sequential in one job)
SKIP_EXISTING=0 TASKS=gender_en,race,religion bash slurm/study_tasks.sh gpt2-small core

# Array over model×task (restartable). Default SUITE=core.
SKIP_EXISTING=0 MODELS=gpt2-small TASKS=all SUITE=core bash slurm/study_array.sh

  # Add one array cell per enabled suite method family.
  ARRAY_BY_METHOD=1 MODELS=gpt2-small TASKS=gender_en SUITE=core bash slurm/study_array.sh

  # Method arrays use configs/slurm_profiles.yaml when SLURM_PROFILE is omitted.
  # Slurm resources are array-wide, so rows are grouped into one submitted array
  # per resolved profile. An explicit profile remains a global override.
  SLURM_PROFILE=gpumem-141-1x ARRAY_BY_METHOD=1 MODELS=llama-3.1-8b TASKS=gender_en bash slurm/study_array.sh

# Full suite into a separate output tree (not merged with core runs/{model}/{task}/)
SKIP_EXISTING=0 MODELS=gpt2-small TASKS=all SUITE=full OUTPUT_SUBDIR=suite_full bash slurm/study_array.sh

# Rewrite every REPORT.md + family/method-group tables from results.json
# (CPU / partition=general; no train, no causal, no GPU)
bash slurm/regenerate_reports.sh
MODEL=gpt2-small bash slurm/regenerate_reports.sh

# Refresh the cluster's appendix evidence tables. CPU only; this does not gather the
# distributed k* data because SAE artifacts can reside on the cluster, a second cluster, and
# workstations.
bash slurm/regenerate_appendix_evidence.sh
# Pull compact SAE artifacts from the cluster.
bash scripts/rsync_analysis.sh --go
# Run this locally after the source syncs. It reads encode_method_rows.json,
# not raw activations or giant results.json files.
python analysis/appendix_evidence.py --scan-kstar

# IEND learning-rate screen on gender_en (how the per-model GRADIEND/ACTIEND rates in
# configs/models/*.yaml were chosen). Preview only by default; --pilot = 1 smoke
# task, --go = the array. Defaults to llama-3.1-8b + qwen3.5-9b-base; override MODELS.
bash slurm/iend_lr_cross_model.sh
MODELS=gemma-2-2b bash slurm/iend_lr_cross_model.sh --go

# ACTIEND one-pole LR x activation-scale validation. Submits one job per
# (model, case); defaults to both models, 3 seeds, 1e-3 stress LR, and causal
# evaluation across those seeds for score >= 0.8. Shards merge under
# runs/_ablation_*_v2/.
bash slurm/ablation_actiend_lr_scale.sh
# Resume completed cells / restrict the grid:
SKIP_EXISTING=1 bash slurm/ablation_actiend_lr_scale.sh
MODELS=gpt2-small CASES=language:fr bash slurm/ablation_actiend_lr_scale.sh
SMOKE=1 MODELS=gpt2-small CASES=ioi:IO MAX_SEEDS=1 bash slurm/ablation_actiend_lr_scale.sh

```

All study launchers default to `SKIP_EXISTING=1` (skips when the job's `results.json` has `"status": "ok"`). Set `SKIP_EXISTING=0` to force a re-run. With `OUTPUT_SUBDIR=suite_full` that file is `runs/{model}/suite_full/{task}/results.json`, so core dumps are left alone.

Use `TRAIN_SPLITS=tensors METHODS=actiend SKIP_EXISTING=0` for an in-place
ACTIEND tensor-only retrain. The tensor encode and matching causal rows are
regenerated; scalar ACTIEND and localization rows are preserved.

Backend LR overrides are available on every study launcher as
`LR_GRADIEND=<value>` and `LR_ACTIEND=<value>`. Keep them unset to use the
configured defaults (`1e-4` GRADIEND, `1e-5` ACTIEND in `defaults.yaml`).

## Priority policy (`--nice`)

the cluster uses multifactor priority with backfill rather than strict FIFO. Positive
nice values lower a job's priority, but a lower-priority job can still start
first if it fits a backfill window.

- `study_array.sh` and `iend_lr_cross_model.sh --go` default to
  `SBATCH_NICE=1000`, keeping large backlogs below later normal work.
- Single jobs submitted through `submit_train.sh` stay at normal priority
  unless `SBATCH_NICE` is explicitly set.
- Use `SBATCH_NICE=0` to submit an array at normal priority, or
  `SBATCH_NICE=5000` for deep-background work.
- `submit_train.sh` accepts `SBATCH_NICE` for every wrapper that uses it.

```bash
# Normal-priority array override
SBATCH_NICE=0 MODELS=gpt2-small TASKS=gender_en bash slurm/study_array.sh

# Deep-background array
SBATCH_NICE=5000 MODELS=gpt2-small TASKS=all bash slurm/study_array.sh

# Lower an already-pending job or array
scontrol update JobId=<jobid> Nice=1000
sprio -j <jobid>
```

Ordinary users can lower their own jobs with `nice`; `scontrol top` is not
enabled for users on the cluster. Inspect current priorities with `sprio -u "$USER"`
and configured weights with `sprio -w`.

Default GPU profile is `gpumem-24-1x`: `--gres=gpu:1` plus `--constraint=gpumem.24gib&nvidiagpu` (any NVIDIA GPU with ≥24 GiB VRAM; no specific GRES type). the cluster VRAM floors (pick the smallest that fits):

| Profile | Constraint | Typical nodes |
|---|---|---|
| `gpumem-11-1x` | `gpumem.11gib&nvidiagpu` | ≥11 GiB NVIDIA |
| `gpumem-16-1x` | `gpumem.16gib&nvidiagpu` | ≥16 GiB NVIDIA (excludes Chimaira AMD RX 9060 XT) |
| `gpumem-24-1x` | `gpumem.24gib&nvidiagpu` | ≥24 GiB (default) |
| `gpumem-48-1x` | `gpumem.48gib&nvidiagpu` | L40 / L40S+ |
| `gpumem-80-1x` | `gpumem.80gib&nvidiagpu` | A100 SXM+ |
| `gpumem-141-1x` | `gpumem.141gib&nvidiagpu` | H200 SXM |

Production `*-1x` profiles request **4 CPUs** and **64 G host RAM** (VRAM floor is the only thing that changes). Do not drop RAM to 32 G: other tasks have OOM’d there even when one array task’s MaxRSS was ~11 G. Each size also has a `-test-1x` variant (2 h, 16 G host RAM). Optional GRES pins (`l40s-1x`, `a100-1x`, …) remain under `slurm/profiles/`.

## Manual TRAIN_CMD

```bash
export TRAIN_CMD='python run_study.py --model pythia-70m-deduped --task ioi --suite core'
bash slurm/submit_train.sh gpumem-24-1x
```

## Actual usage vs requested

`sacct` MaxRSS / `TRESUsageInMax` are empty while the task is **RUNNING**. Use a completed array task (or wait), or the live snapshot (`sstat` + `nvidia-smi`).

```bash
bash slurm/job_usage.sh 3776425_18    # completed: peak VRAM/RAM/CPU + profile hint
bash slurm/job_usage.sh 3776425_17    # running: sstat + nvidia-smi snapshot
```

`seff` = CPU and host RAM. `gres/gpumem` in `TRESUsageInMax` = GPU VRAM peak. Pick the next the cluster floor above that VRAM (`11 / 16 / 24 / 48 / 80 / 141`). Keep `SBATCH_CPUS=4` and `SBATCH_MEM=64G` unless a completed job’s MaxRSS is well above 64 G.
