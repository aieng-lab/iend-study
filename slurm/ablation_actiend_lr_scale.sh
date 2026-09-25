#!/usr/bin/env bash
# Submit the ACTIEND activation-scale validation as one job per (model, case).
#
# Defaults run the comprehensive v2 protocol:
#   - gpt2-small + pythia-70m-deduped
#   - raw + running_rms
#   - default LR grid through 3e-4 plus isolated 1e-3 stress cells
#   - all three training seeds
#   - causal evaluation across all trained seeds for checkpoint scores >= 0.8
#
# Examples:
#   bash slurm/ablation_actiend_lr_scale.sh
#   MODELS=gpt2-small CASES=language:fr bash slurm/ablation_actiend_lr_scale.sh
#   SKIP_EXISTING=1 bash slurm/ablation_actiend_lr_scale.sh
#   SMOKE=1 MODELS=gpt2-small CASES=ioi:IO MAX_SEEDS=1 bash slurm/ablation_actiend_lr_scale.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_CSV="${MODELS:-gpt2-small,pythia-70m-deduped}"
CASES_CSV="${CASES:-emotion:negative,emotion:positive,induction:MATCH,ioi:IO,key_value:VALUE,language:fr,pronoun_person:1}"
SUITE="${SUITE:-core}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-ablation_actiend_lr_scale_v2}"
MAX_SEEDS="${MAX_SEEDS:-3}"
ALL_SEEDS="${ALL_SEEDS:-1}"
INCLUDE_STRESS_LR="${INCLUDE_STRESS_LR:-1}"
CAUSAL="${CAUSAL:-1}"
CAUSAL_ALL_SEEDS="${CAUSAL_ALL_SEEDS:-1}"
CAUSAL_MIN_SCORE="${CAUSAL_MIN_SCORE:-0.8}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LRS_CSV="${LRS:-}"
MODES_CSV="${MODES:-}"

enabled() {
  case "${1,,}" in
    1|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a models <<< "${MODELS_CSV}"
IFS=',' read -r -a cases <<< "${CASES_CSV}"

submitted=0
cd "${REPO_DIR}"
for model_raw in "${models[@]}"; do
  model="${model_raw//[[:space:]]/}"
  [[ -n "${model}" ]] || continue
  for case_raw in "${cases[@]}"; do
    case_id="${case_raw//[[:space:]]/}"
    [[ -n "${case_id}" ]] || continue
    cmd=(
      python -u scripts/ablation_actiend_lr_scale.py
      --models "${model}"
      --cases "${case_id}"
      --suite "${SUITE}"
      --output-subdir "${OUTPUT_SUBDIR}"
      --max-seeds "${MAX_SEEDS}"
    )
    if enabled "${ALL_SEEDS}"; then
      cmd+=(--all-seeds)
    fi
    if enabled "${INCLUDE_STRESS_LR}"; then
      cmd+=(--include-stress-lr)
    fi
    if enabled "${CAUSAL}"; then
      cmd+=(--causal --causal-min-score "${CAUSAL_MIN_SCORE}")
      if enabled "${CAUSAL_ALL_SEEDS}"; then
        cmd+=(--causal-all-seeds)
      fi
    fi
    if enabled "${SKIP_EXISTING}"; then
      cmd+=(--skip-existing)
    fi
    if enabled "${SMOKE}"; then
      cmd+=(--smoke)
    fi
    if [[ -n "${LRS_CSV}" ]]; then
      IFS=',' read -r -a requested_lrs <<< "${LRS_CSV}"
      cmd+=(--lrs "${requested_lrs[@]}")
    fi
    if [[ -n "${MODES_CSV}" ]]; then
      IFS=',' read -r -a requested_modes <<< "${MODES_CSV}"
      cmd+=(--modes "${requested_modes[@]}")
    fi

    printf -v train_cmd '%q ' "${cmd[@]}"
    job_case="${case_id//:/-}"
    job_name="a-lr-${model}-${job_case}"
    job_name="${job_name:0:80}"
    echo "Submitting ${model} ${case_id}: ${train_cmd}"
    if ! enabled "${DRY_RUN}"; then
      TRAIN_CMD="${train_cmd}" JOB_NAME="${job_name}" \
        bash "${SCRIPT_DIR}/submit_train.sh" "${SLURM_PROFILE:-gpumem-24-1x}"
    fi
    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} ACTIEND LR/scale ablation jobs."
