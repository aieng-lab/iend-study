#!/usr/bin/env bash
# Mode: tasks — one job looping all (or listed) tasks for a fixed model.
#
#   bash slurm/study_tasks.sh pythia-70m-deduped full
#   TASKS=gender_en,race,religion bash slurm/study_tasks.sh gpt2-small core

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-${MODEL:-pythia-70m-deduped}}"
SUITE="${2:-${SUITE:-core}}"
TASKS_CSV="${TASKS:-all}"
SKIP_EXISTING_FLAG="${SKIP_EXISTING:-1}"

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
export REPO_DIR="${REPO_DIR}"
export SLURM_DIR="${SCRIPT_DIR}"

export TRAIN_CMD="MODEL=${MODEL} SUITE=${SUITE} TASKS_CSV=${TASKS_CSV} SKIP_EXISTING_FLAG=${SKIP_EXISTING_FLAG} OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-} SCALE=${SCALE:-} METHODS=${METHODS:-} TRAIN_SPLITS=${TRAIN_SPLITS:-} LR_GRADIEND=${LR_GRADIEND:-} LR_ACTIEND=${LR_ACTIEND:-} LR_DECODER_GRADIEND=${LR_DECODER_GRADIEND:-} LR_DECODER_ACTIEND=${LR_DECODER_ACTIEND:-} MAX_STEPS=${MAX_STEPS:-} EVAL_STEPS=${EVAL_STEPS:-} IEND_SELECTION_METRIC=${IEND_SELECTION_METRIC:-} IEND_CONVERGENT_METRIC=${IEND_CONVERGENT_METRIC:-} SKIP_CAUSAL=${SKIP_CAUSAL:-} SKIP_LOCALIZATION=${SKIP_LOCALIZATION:-} REFRESH_CAUSAL=${REFRESH_CAUSAL:-} SMOKE=${SMOKE:-} FAIL_FAST=${FAIL_FAST:-1} bash /workspace/slurm/_tasks_loop.sh"
_M="${MODEL##*/}"
export JOB_NAME="${JOB_NAME:-s-${_M:0:3}-tasks}"

exec bash "${SCRIPT_DIR}/submit_train.sh" "${SLURM_PROFILE:-gpumem-24-1x}"
