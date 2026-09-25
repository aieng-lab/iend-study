#!/usr/bin/env bash
# CPU Slurm job: regenerate every REPORT.md + cross-task tables from results.json.
# Does not re-run train or causal. No GPU.
#
#   bash slurm/regenerate_reports.sh
#   MODEL=gpt2-small bash slurm/regenerate_reports.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
export REPO_DIR="${REPO_DIR}"
export SLURM_DIR="${SCRIPT_DIR}"
export APPTAINER_NV="${APPTAINER_NV:-0}"
export MODEL="${MODEL:-}"
export RUNS="${RUNS:-runs}"
export LAYER_SELECTION_SOURCES="${LAYER_SELECTION_SOURCES:-}"
export APPENDIX_LAYER_SELECTION_ONLY="${APPENDIX_LAYER_SELECTION_ONLY:-0}"
export TRAIN_CMD="MODEL=${MODEL} RUNS=${RUNS} LAYER_SELECTION_SOURCES=${LAYER_SELECTION_SOURCES} APPENDIX_LAYER_SELECTION_ONLY=${APPENDIX_LAYER_SELECTION_ONLY} bash /workspace/slurm/_regen_reports_cmd.sh"
export JOB_NAME="${JOB_NAME:-s-regen-reports}"

PROFILE="${SLURM_PROFILE:-cpu-1x}"
PROFILE_FILE="${SLURM_DIR}/profiles/${PROFILE}.sh"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unknown profile: ${PROFILE}" >&2
  echo "Available: $(for f in "${SLURM_DIR}"/profiles/*.sh; do basename "$f" .sh; done | tr '\n' ' ')" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SLURM_DIR}/_source_profile.sh"
source_profile "${PROFILE_FILE}"
SBATCH_MEM="$(printf '%s' "${SBATCH_MEM:-}" | tr -d '\r' | xargs)"
SBATCH_CPUS="$(printf '%s' "${SBATCH_CPUS:-}" | tr -d '\r' | xargs)"
SBATCH_TIME="$(printf '%s' "${SBATCH_TIME:-}" | tr -d '\r' | xargs)"
SBATCH_PARTITION="$(printf '%s' "${SBATCH_PARTITION:-}" | tr -d '\r' | xargs)"
SBATCH_GRES="$(printf '%s' "${SBATCH_GRES:-}" | tr -d '\r' | xargs)"
SBATCH_CONSTRAINT="$(printf '%s' "${SBATCH_CONSTRAINT:-}" | tr -d '\r' | xargs)"
if [[ ! "${SBATCH_MEM}" =~ ^[0-9]+[KMG]?$ ]]; then
  echo "Invalid SBATCH_MEM=${SBATCH_MEM@Q} from profile ${PROFILE} (${PROFILE_FILE})" >&2
  exit 1
fi

GRES_ARGS=()
if [[ -n "${SBATCH_GRES}" ]]; then
  GRES_ARGS+=(--gres="${SBATCH_GRES}")
fi
CONSTRAINT_ARGS=()
if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
  CONSTRAINT_ARGS+=(--constraint="${SBATCH_CONSTRAINT}")
fi

mkdir -p "${SLURM_LOG_DIR}"

echo "Profile: ${PROFILE}"
echo "  partition=${SBATCH_PARTITION} gres=${SBATCH_GRES:-(none)} constraint=${SBATCH_CONSTRAINT:-(none)} cpus=${SBATCH_CPUS} mem=${SBATCH_MEM} time=${SBATCH_TIME}"
echo "  APPTAINER_NV=${APPTAINER_NV} MODEL=${MODEL:-(all)} RUNS=${RUNS} appendix_sources=${LAYER_SELECTION_SOURCES:-(none)} appendix_only=${APPENDIX_LAYER_SELECTION_ONLY}"
echo "Command: ${TRAIN_CMD}"

exec sbatch \
  --chdir="${USER_DIR}" \
  --job-name="${JOB_NAME}" \
  --output="${SLURM_LOG_DIR}/slurm-%j.out" \
  --error="${SLURM_LOG_DIR}/slurm-%j.err" \
  --partition="${SBATCH_PARTITION}" \
  "${GRES_ARGS[@]}" \
  "${CONSTRAINT_ARGS[@]}" \
  --cpus-per-task="${SBATCH_CPUS}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --export=ALL \
  "${SLURM_DIR}/analysis.sbatch"
