#!/usr/bin/env bash
# In-job sequential task loop (used by study_tasks.sh).
set -euo pipefail
cd /workspace
TASKS_CSV="${TASKS_CSV:-all}"
MODEL="${MODEL:?}"
SUITE="${SUITE:-core}"

if [[ "${TASKS_CSV}" == "all" ]]; then
  mapfile -t TASK_LIST < <(python -c "from study.config import list_tasks; print('\n'.join(list_tasks()))")
else
  IFS=, read -r -a TASK_LIST <<< "${TASKS_CSV}"
fi

EXTRA=()
if [[ "${SKIP_EXISTING_FLAG:-1}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi
if [[ -n "${OUTPUT_SUBDIR:-}" ]]; then
  EXTRA+=(--output-subdir "${OUTPUT_SUBDIR}")
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
if [[ "${REFRESH_CAUSAL:-}" == "1" ]]; then
  EXTRA+=(--skip-existing --refresh-causal)
fi
if [[ "${SMOKE:-}" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ "${FAIL_FAST:-1}" == "1" ]]; then
  EXTRA+=(--fail-fast)
fi

for t in "${TASK_LIST[@]}"; do
  t="$(echo "$t" | tr -d '[:space:]')"
  [[ -z "$t" ]] && continue
  echo "=== task=$t model=${MODEL} suite=${SUITE} output_subdir=${OUTPUT_SUBDIR:-} ==="
  python run_study.py --model "${MODEL}" --task "$t" --suite "${SUITE}" "${EXTRA[@]+"${EXTRA[@]}"}"
done
