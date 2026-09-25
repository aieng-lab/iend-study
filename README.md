# gradiend-sae

Study code for **"Targeted Feature Learning in Language Models"**. The paper
compares six *targeted* feature-learning methods, obtained by crossing three
model-derived signals with two feature estimators, against a pretrained-SAE
reference:

| Signal | Contrastive mean | IEND (learned 1-D encoder-decoder) |
|---|---|---|
| Activation value | CAA | ACTIEND |
| Activation gradient | CAGA | AGIEND |
| Parameter gradient | CGA | GRADIEND |

Untargeted reference: **SAE** (pretrained SAELens dictionaries, features selected post hoc).

Every method is evaluated on **15 tasks** (semantic, grammatical, factual,
language-ID, algorithmic) for **3 models** (GPT-2 small, Pythia-70M-deduped,
Llama-3.1-8B), on two axes: **Detection** (does the feature score
separate held-out examples?) and **Intervention** (does steering with the
feature shift the model's next-token behavior under a preservation constraint?).

This repository is the *study* code (task definitions, CAA/CGA/CAGA/SAE
implementations, evaluation protocol, analysis). It depends on an extended
fork of the `gradiend` package, which
provides the signal extraction and the IEND trainers (GRADIEND/ACTIEND/AGIEND).

## Repository layout

```
run_study.py            CLI: one (model, task, suite) pipeline run
study/                  pipeline: config, train/SAE/CAA/causal stages, result merging
  data/                 dataset generators (synthetic, MIB/RAVEL, language-ID)
  tasks/  stages/       task loaders; pipeline stages
  signals/              signal extraction (used by CAGA) and the opt-in `actiend_ridge` variant
caa_eval.py cga_eval.py caga_eval.py sae_eval.py   the contrastive-mean and SAE methods
causal_eval.py causal_study.py                     LMS-gated intervention protocol
results_schema.py  suitability.py  ...             method ids, metrics, bookkeeping
configs/                models/ (4 paper models + others), tasks/ (15 paper tasks), suites/
data/                   generated/downloaded task CSVs (not tracked, see Datasets)
scripts/                data generation, sync, QA/repair helpers, LR/ablation dev scripts
slurm/                  Slurm launchers (study_array.sh is the entry point)
analysis/               tables, figures and statistics of the paper
tests/                  pytest suite
```

Material that did not make it into the paper (an IEND theory side-study, an
SAE-absorption study, an AxBench validation, cluster-specific infrastructure)
is deliberately **not** part of this repository.

---

# Reproducibility

## 1. Installation

Tested with Python 3.10/3.11 (cluster) and 3.13 (local), PyTorch >= 2.0.

```bash
git clone <this repository> gradiend-sae
# 1) the extended gradiend fork (editable) -- NOT the PyPI release
git clone <extended gradiend fork> gradiend && pip install -e ./gradiend
# 2) the study requirements (gradiend>=0.2.1 is then already satisfied)
cd gradiend-sae && pip install -r requirements.txt
```

`requirements.txt` includes everything the study needs beyond `gradiend`:
`torch`, `transformers`, `datasets`, `sae-lens` (pretrained SAEs for the SAE
reference), `scikit-learn`, `autorank` (Bayesian method comparison),
`matplotlib`, `adjustText`, `pypdf`, `filelock`, `PyYAML`, `pandas`.

Additional prerequisites:

- **Hugging Face access.** `meta-llama/Llama-3.1-8B` is
  gated: accept its license and run `huggingface-cli login`. Models, SAEs
  (`gpt2-small-resid-post-v5-32k`, `pythia-70m-deduped-res-sm`,
  `llama_scope_lxr_32x`) and the task
  datasets are downloaded on first use into the standard HF cache.
- **LaTeX** (`latexmk` or `pdflatex`) is only needed to compile the summary-table
  PDFs; the `.tex`/`.csv` sources and all figures are written without it.
- **Hardware.** GPT-2 small / Pythia-70M: one 24 GB GPU.
  Llama-3.1-8B: 80-141 GB for the parameter-gradient methods (bfloat16, see
  `configs/models/llama-3.1-8b.yaml`). Per-method Slurm resource tiers are in
  `configs/slurm_profiles.yaml`.

Smoke test (tiny data, no long training):

```bash
python run_study.py --model gpt2-small --task gender_en --suite core --smoke
pytest                      # unit tests, no GPU or downloads needed for most
```

## 2. Datasets

| Task(s) | Source | How it is obtained |
|---|---|---|
| `gender_en`, `emotion`, `race`, `religion`, `pronoun_number`, `pronoun_person` | Hugging Face datasets (IDs in `configs/tasks/*.yaml`) | downloaded automatically on first use |
| `key_value`, `induction`, `function_composition`, `repetition` | controlled local generators (`study/data/synthetic.py`) | generated offline into `data/synthetic/` |
| `ioi_mib`, `ravel_continent`, `ravel_country`, `ravel_language` | `mib-bench/ioi`, `hij/ravel` (`study/data/mib.py`) | generated from the source datasets (download) into `data/mib/` |
| `language` | OPUS-100 + MUSE lexica (`study/data/language_parallel.py`) | generated deterministically; needs network |
| neutral texts | a shared neutral-text corpus (`neutral_hf` in the task configs), task-specific neutral sets | downloaded (cached in `data/neutral`); `data/mib/*_neutral.csv` are generated |

No dataset is tracked in this repository (third-party licenses); `data/` is
git-ignored and every CSV is built or downloaded on first use.

**Local (re)generation**, from scratch or after changing a generator:

```bash
python scripts/generate_datasets.py                       # all locally built tasks (keeps valid caches)
python scripts/generate_datasets.py --only synthetic --force   # offline: key_value/induction/function_composition/repetition
python scripts/generate_datasets.py --only mib --force         # ioi_mib + 3 RAVEL tasks (downloads)
python scripts/generate_datasets.py --only language --force    # language-ID (downloads OPUS-100/MUSE)
```

A task whose CSV is missing or stale (older generator version) is regenerated
automatically by `run_study.py`.

## 3. Running the study

A run is one `(model, task, suite)` cell. It trains all methods of the suite,
evaluates Detection, runs the intervention sweep and writes
`runs/<model>[/<output-subdir>]/<task>/results.json` (plus `REPORT.md` and
`artifacts/`). Suites (`configs/suites/`) choose the method variants:

- `core`: headline methods (CAA, CGA/CAGA with layer scopes, GRADIEND, ACTIEND,
  AGIEND, SAE k=1 / k* / all-layer), pairwise and one-sided.
- `full_plus`: `core` plus the ablation variants (SAE variants, CAA/CGA/CAGA
  variants, by-tensor GRADIEND/ACTIEND).
- `runtime_benchmark`: one comparable instance per method (Appendix "Computational cost").

Learning rates, step budgets and pruning per model are fixed in
`configs/models/<model>.yaml` and `configs/defaults.yaml` (Appendix Table
"Training, selection, and evaluation hyperparameters"); the per-model
GRADIEND/ACTIEND rates come from a short `gender_en` screen
(`slurm/iend_lr_cross_model.sh`; development ablations behind the optimization
choices are `scripts/ablation_*.py`).

**Which suite/subdirectory holds which paper model** (this is also encoded in
`configs/report_model_sets.json`, which the analysis scripts read):

| Model | Suite | Output tree |
|---|---|---|
| `gpt2-small`, `pythia-70m-deduped` | `full_plus` | `runs/<model>/suite_full2/<task>/` |
| `llama-3.1-8b` | `core` | `runs/<model>/<task>/` |

### Python (single machine / one cell at a time)

```bash
# one cell
python run_study.py --model gpt2-small --task gender_en --suite full_plus --output-subdir suite_full2
python run_study.py --model llama-3.1-8b --task gender_en --suite core

# all 15 tasks for the two small models (resumable: finished tasks are skipped)
for task in $(python -c "from study.config import list_tasks; print(' '.join(list_tasks()))"); do
  for model in gpt2-small pythia-70m-deduped; do
    python run_study.py --model $model --task $task --suite full_plus \
        --output-subdir suite_full2 --skip-existing
  done
done
for task in $(python -c "from study.config import list_tasks; print(' '.join(list_tasks()))"); do
  python run_study.py --model llama-3.1-8b --task $task --suite core --skip-existing
done
```

`python run_study.py --help` lists all flags (`--methods`, `--skip-causal`,
`--refresh-causal`, per-backend learning-rate overrides, ...).

### Slurm (what was used for the paper)

`slurm/study_array.sh` builds a job table over `MODELS x TASKS` and submits one
Slurm array; resources come from `slurm/profiles/` and
`configs/slurm_profiles.yaml`. Adapt `slurm/gradiend_sae.env.sh` (paths, conda /
Apptainer) and `slurm/profiles/*.sh` (partitions, constraints) to your cluster first.

```bash
# GPT-2 small + Pythia (paper tree: runs/<model>/suite_full2/)
MODELS=gpt2-small,pythia-70m-deduped TASKS=all SUITE=full_plus \
  OUTPUT_SUBDIR=suite_full2 bash slurm/study_array.sh

# Llama-3.1-8B (paper tree: runs/<model>/); one job per method family
MODELS=llama-3.1-8b TASKS=all SUITE=core ARRAY_BY_METHOD=1 bash slurm/study_array.sh

# Runtime benchmark (Appendix): gender_en, one instance per method, FIXED hardware:
# SLURM_PROFILE pins the GPU type (l40s-1x / a100-1x); without it the array runs on any >=24 GiB GPU.
SLURM_PROFILE=l40s-1x MODELS=gpt2-small TASKS=gender_en SUITE=runtime_benchmark OUTPUT_SUBDIR=runtime_benchmark_l40s \
  bash slurm/study_array.sh
SLURM_PROFILE=a100-1x MODELS=llama-3.1-8b TASKS=gender_en SUITE=runtime_benchmark OUTPUT_SUBDIR=runtime_benchmark_a100 \
  bash slurm/study_array.sh
# SAE cost companion (one SAE instance: k=1, prediction site only; replaces the SAE bars)
SLURM_PROFILE=l40s-1x METHODS=sae SKIP_LOCALIZATION=1 MODELS=gpt2-small TASKS=gender_en SUITE=runtime_benchmark_sae_k1 OUTPUT_SUBDIR=runtime_benchmark_l40s_sae_k1_v2 \
  bash slurm/study_array.sh
SLURM_PROFILE=a100-1x METHODS=sae SKIP_LOCALIZATION=1 MODELS=llama-3.1-8b TASKS=gender_en SUITE=runtime_benchmark_sae_k1 OUTPUT_SUBDIR=runtime_benchmark_a100_sae_k1_v2 \
  bash slurm/study_array.sh
```

`SKIP_EXISTING=1` (default) skips cells whose `results.json` is `status: ok`.
Useful switches: `METHODS=gradiend,actiend,...` (subset), `REFRESH_CAUSAL=1`
(recompute only the intervention stage), `TRAIN_ABLATIONS=pair|one_pole`,
`SBATCH_NICE`. Details: [`slurm/README.md`](slurm/README.md).

If runs were produced on a cluster, pull the results (small files only; no
checkpoints) with `bash scripts/rsync_analysis.sh --go` (or
`scripts/rsync_analysis.ps1`); it snapshots and merges so a pull never
overwrites richer local results.

## 4. Evaluation and analysis

Detection (AUC_n, AUC_o, Spec_n, Excl, `Det.`) and Intervention
(`ΔP+`, `ΔP-`, `Int.`, LMS) are computed inside the pipeline with all
selection decisions (checkpoints, SAE latents/layers, thresholds, intervention
strengths) taken on validation data and test data used once. `analysis/`
only aggregates the stored `results.json` files (CPU, no model needed):

```bash
# every table, figure and statistic of the paper; reads runs/ as laid out in Sec. 3
REPORT_MODE=full INCLUDE_APPENDIX_ARTIFACTS=1 bash scripts/summary_tables_pdf.sh

# only the two small models / a quick headline refresh
MODELS=gpt2-small,pythia-70m-deduped bash scripts/summary_tables_pdf.sh

# runtime/memory figure (needs the two runtime_benchmark runs from Sec. 3)
python analysis/plot_runtime_benchmark.py runs/llama-3.1-8b/runtime_benchmark_a100_clean_v2/gender_en/results.json \
  --sae-from runs/llama-3.1-8b/runtime_benchmark_a100_sae_k1_v2/gender_en/results.json \
  --joint-with runs/gpt2-small/runtime_benchmark_l40s/gender_en/results.json \
  --joint-sae-from runs/gpt2-small/runtime_benchmark_l40s_sae_k1_v2/gender_en/results.json \
  --out analysis/plots/runtime_benchmark_llama_a100 \
  --joint-out analysis/plots/runtime_benchmark_gpt2_llama_joint
```

`scripts/summary_tables_pdf.sh` chains `analysis/summary_latex.py` (per-model and
across-model tables), `analysis/across_model_detail.py` (per-task heatmaps,
factor breakdowns), `analysis/headline_scatter.py`, `analysis/method_statistics.py`
(Autorank/Bayesian comparison), `analysis/layer_performance_plots.py`,
`analysis/layer_selection_appendix.py` and `analysis/appendix_evidence.py`.
Which models/subdirectories it reads is defined by `configs/report_model_sets.json`;
`MODELS=...` overrides it. QA helpers before trusting a table:
`python scripts/report_causal_gaps.py`, `python scripts/headline_completeness.py`.

## 5. Where to find the outputs

Raw results (per run):

| What | Path |
|---|---|
| all metrics of one (model, task) cell | `runs/<model>[/suite_full2]/<task>/results.json` |
| human-readable report of the cell | `.../<task>/REPORT.md` |
| trained IEND checkpoints, `done.json` (LR, convergence, seeds) | `.../<task>/artifacts/` |
| intervention sweeps / decoder grids | `.../<task>/causal/` |
| runtime-benchmark cost ledger | `runs/<model>/runtime_benchmark_*/gender_en/results.json` |

Generated (gitignored, regenerated by Sec. 4) paper artifacts:

| Paper item | File |
|---|---|
| Detection-vs-Intervention scatter (main figure), per-model scatter | `analysis/figures/headline_scatter_signal_mean.pdf`, `analysis/figures/headline_scatter_by_model.pdf` |
| Task x method Det./Int. heatmap; Bayesian comparison (2x2 summary, full 4x4) | `analysis/tables/latex_across_models/headline_method_task_det_int_heatmap_latex.pdf`, `autorank_posterior_decision_overview_2x2.pdf`, `autorank_posterior_overview_4x4.pdf` |
| Per-model task heatmaps (Appendix, all paper models) | `analysis/tables/latex_across_models/across_model_det_int_<model>_heatmap.pdf` |
| Task-family / pairwise-vs-one-sided breakdowns | `analysis/tables/latex_across_models/factor_breakdown_task_family.pdf`, `factor_breakdown_construction.pdf` |
| Task-averaged and per-task result tables | `analysis/tables/latex_across_models/*.tex`, `analysis/tables/latex_<model>[_suite_full2]/summary_*.tex` (+ compiled `summary_tables.pdf`) |
| Layer-wise Det./Int. profiles | `analysis/figures/layer_<model>[_suite_full2]/layer_det_int_<model>.pdf` |
| Single- vs all-layer analysis (Table + Figure) | `analysis/tables/layer_selection/layer_selection_paired_deltas.pdf` (+ csv) |
| SAE variants ($k=1$, $k^\star$, pre-target) | `analysis/tables/appendix_evidence/appendix_sae_variants.pdf` |
| Runtime / memory figure | `analysis/plots/runtime_benchmark_gpt2_llama_joint/runtime_benchmark_gpt2_llama_joint.pdf` |
| Ablation inventory (which variants exist per task) | `analysis/tables/appendix_ablations/ablation_inventory.{csv,tex}` |

Method ids in `results.json` follow `<method>:<pair>:<class>[:<variant>]`
(`gradiend:F-M:F` = pairwise, `gradiend:F` = one-sided); see `results_schema.py`
and `study/method_ids.py`.

## 6. Tests

```bash
pytest
```

`tests/conftest.py` skips a few torch-heavy files by default (set
`GRADIEND_RUN_HEAVY_TESTS=1` to include them). Some tests build real GPT-2
trainers only when `GRADIEND_RUN_CGA_E2E=1` is set.
