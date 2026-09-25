#!/usr/bin/env bash
# Mode: single — one (model, task) job.
#
#   bash slurm/study_single.sh pythia-70m-deduped race core
#   MODEL=gpt2-small TASK=gender_en SUITE=core bash slurm/study_single.sh
# Third arg / SUITE defaults to core (gpt2-small.yaml suite_default is full if omitted).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-${MODEL:-pythia-70m-deduped}}"
TASK="${2:-${TASK:-gender_en}}"
SUITE="${3:-${SUITE:-core}}"

EXTRA=()
if [[ -n "${SUITE}" ]]; then
  EXTRA+=(--suite "${SUITE}")
fi
if [[ -n "${SCALE:-}" ]]; then
  EXTRA+=(--scale "${SCALE}")
fi
if [[ -n "${METHODS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(--methods ${METHODS//,/ })
fi
if [[ -n "${TRAIN_SPLITS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(--train-splits ${TRAIN_SPLITS//,/ })
fi
if [[ -n "${LR_GRADIEND:-}" ]]; then
  EXTRA+=(--lr-gradiend "${LR_GRADIEND}")
fi
if [[ -n "${LR_ACTIEND:-}" ]]; then
  EXTRA+=(--lr-actiend "${LR_ACTIEND}")
fi
if [[ -n "${LR_DECODER_GRADIEND:-}" ]]; then
  EXTRA+=(--lr-decoder-gradiend "${LR_DECODER_GRADIEND}")
fi
if [[ -n "${LR_DECODER_ACTIEND:-}" ]]; then
  EXTRA+=(--lr-decoder-actiend "${LR_DECODER_ACTIEND}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${EVAL_STEPS:-}" ]]; then
  EXTRA+=(--eval-steps "${EVAL_STEPS}")
fi
if [[ -n "${IEND_SELECTION_METRIC:-}" ]]; then
  EXTRA+=(--iend-selection-metric "${IEND_SELECTION_METRIC}")
fi
if [[ -n "${IEND_CONVERGENT_METRIC:-}" ]]; then
  EXTRA+=(--iend-convergent-metric "${IEND_CONVERGENT_METRIC}")
fi
if [[ "${SKIP_CAUSAL:-}" == "1" ]]; then
  EXTRA+=(--skip-causal)
fi
if [[ "${SKIP_LOCALIZATION:-}" == "1" ]]; then
  EXTRA+=(--skip-localization)
fi
if [[ "${SKIP_REPORTS:-1}" == "1" ]]; then
  EXTRA+=(--skip-reports)
fi
if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi
if [[ -n "${OUTPUT_SUBDIR:-}" ]]; then
  EXTRA+=(--output-subdir "${OUTPUT_SUBDIR}")
fi
if [[ "${SMOKE:-}" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ "${FAIL_FAST:-1}" == "1" ]]; then
  EXTRA+=(--fail-fast)
fi

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
export REPO_DIR="${REPO_DIR}"
export SLURM_DIR="${SCRIPT_DIR}"
export TRAIN_CMD="python run_study.py --model ${MODEL} --task ${TASK} ${EXTRA[*]-}"
# Compact squeue name: s-<3 model chars>-<task>  (gpt2-small → gpt).
_M="${MODEL##*/}"
export JOB_NAME="${JOB_NAME:-s-${_M:0:3}-${TASK}}"

exec bash "${SCRIPT_DIR}/submit_train.sh" "${SLURM_PROFILE:-gpumem-24-1x}"
