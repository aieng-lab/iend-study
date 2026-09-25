#!/usr/bin/env bash
# Submit one Slurm job per task for a given model. Each job is named with the task.
#
#   bash slurm/study_all_tasks.sh gpt2-small core
#   TASKS=gender_en,race bash slurm/study_all_tasks.sh gpt2-small core

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-${MODEL:-gpt2-small}}"
SUITE="${2:-${SUITE:-core}}"
TASKS_CSV="${TASKS:-all}"

cd "${REPO_DIR}"
if [[ "${TASKS_CSV}" == "all" ]]; then
  TASK_LIST="$(python -c "from study.config import list_tasks; print(' '.join(list_tasks()))")"
else
  TASK_LIST="${TASKS_CSV//,/ }"
fi

echo "Submitting ${MODEL} suite=${SUITE} tasks: ${TASK_LIST}"

for task in ${TASK_LIST}; do
  _M="${MODEL##*/}"
  JOB_NAME="s-${_M:0:3}-${task}" \
    bash "${SCRIPT_DIR}/study_single.sh" "${MODEL}" "${task}" "${SUITE}"
done
