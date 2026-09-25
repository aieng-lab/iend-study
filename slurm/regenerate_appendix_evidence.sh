#!/usr/bin/env bash
# CPU Slurm job: generate the appendix evidence package on the cluster, where the
# final the cluster summaries are available.  This is not the distributed k* data
# gatherer: SAE records can live on the cluster, a second cluster, and workstations.
#
#   bash slurm/regenerate_appendix_evidence.sh
#
# It does not run training, inference, causal evaluation, or request a GPU.
# Run the k* scan locally after source-preserving compact-artifact syncs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
export REPO_DIR="${REPO_DIR}"
export SLURM_DIR="${SCRIPT_DIR}"
export APPTAINER_NV=0
export TRAIN_CMD="bash /workspace/slurm/_appendix_evidence_cmd.sh"
export JOB_NAME="${JOB_NAME:-s-appendix-evidence}"

PROFILE="${SLURM_PROFILE:-cpu-model-1x}"
PROFILE_FILE="${SLURM_DIR}/profiles/${PROFILE}.sh"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unknown profile: ${PROFILE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SLURM_DIR}/_source_profile.sh"
source_profile "${PROFILE_FILE}"

for required in SBATCH_MEM SBATCH_CPUS SBATCH_TIME SBATCH_PARTITION; do
  value="$(printf '%s' "${!required:-}" | tr -d '\r' | xargs)"
  printf -v "${required}" '%s' "${value}"
done

echo "Profile: ${PROFILE}"
echo "  partition=${SBATCH_PARTITION} cpus=${SBATCH_CPUS} mem=${SBATCH_MEM} time=${SBATCH_TIME} gpu=none"
echo "Command: ${TRAIN_CMD}"

exec sbatch \
  --chdir="${USER_DIR}" \
  --job-name="${JOB_NAME}" \
  --output="${SLURM_LOG_DIR}/slurm-%j.out" \
  --error="${SLURM_LOG_DIR}/slurm-%j.err" \
  --partition="${SBATCH_PARTITION}" \
  --cpus-per-task="${SBATCH_CPUS}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --export=ALL \
  "${SLURM_DIR}/analysis.sbatch"
