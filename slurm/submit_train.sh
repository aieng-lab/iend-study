#!/usr/bin/env bash
# Submit a study job with a GPU profile (mirrors gradiend/slurm/submit_train.sh).
#
# Usage:
#   export TRAIN_CMD='python run_study.py --model pythia-70m-deduped --task race --suite full'
#   bash slurm/submit_train.sh
#   bash slurm/submit_train.sh gpumem-24-test-1x

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"
: "${SLURM_DIR:=${SCRIPT_DIR}}"

PROFILE="${SLURM_PROFILE:-gpumem-24-1x}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  PROFILE="$1"
  shift
fi

PROFILE_FILE="${SLURM_DIR}/profiles/${PROFILE}.sh"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unknown profile: ${PROFILE}" >&2
  echo "Available: $(for f in "${SLURM_DIR}"/profiles/*.sh; do basename "$f" .sh; done | tr '\n' ' ')" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${SLURM_DIR}/_source_profile.sh"
source_profile "${PROFILE_FILE}"
# Strip CR (Windows sync) / whitespace so Slurm never sees ``--mem=$'64G\r'``.
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

# ``study.sbatch`` has embedded GPU directives. Supplying no --gres on the
# sbatch command line does not cancel those directives, so CPU profiles would
# still request a GPU and cannot run in the general partition. Use the matching
# CPU sbatch wrapper whenever the resolved profile intentionally has no GRES.
SBATCH_SCRIPT="${SLURM_DIR}/study.sbatch"
if [[ -z "${SBATCH_GRES}" ]]; then
  SBATCH_SCRIPT="${SLURM_DIR}/analysis.sbatch"
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "SBATCH script not found: ${SBATCH_SCRIPT}" >&2
  exit 1
fi
CHDIR="${USER_DIR}"

GRES_ARGS=()
if [[ -n "${SBATCH_GRES}" ]]; then
  GRES_ARGS+=(--gres="${SBATCH_GRES}")
fi
CONSTRAINT_ARGS=()
if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
  CONSTRAINT_ARGS+=(--constraint="${SBATCH_CONSTRAINT}")
fi
NICE_ARGS=()
# Default to 100 rather than 0 so a genuinely urgent job can still outrank the
# normal queue by passing SBATCH_NICE=0. With 0 as the default there was no
# priority left above the ordinary case, and a diagnostic needed within minutes
# queued behind hours of routine work.
: "${SBATCH_NICE:=100}"
if [[ -n "${SBATCH_NICE:-}" ]]; then
  if [[ ! "${SBATCH_NICE}" =~ ^[0-9]+$ ]]; then
    echo "Invalid SBATCH_NICE=${SBATCH_NICE@Q}; expected a non-negative integer." >&2
    exit 1
  fi
  NICE_ARGS+=(--nice="${SBATCH_NICE}")
fi

echo "Profile: ${PROFILE}"
echo "  partition=${SBATCH_PARTITION} gres=${SBATCH_GRES} constraint=${SBATCH_CONSTRAINT:-(none)} cpus=${SBATCH_CPUS} mem=${SBATCH_MEM} time=${SBATCH_TIME} nice=${SBATCH_NICE:-0}"
echo "Train command: ${TRAIN_CMD:-(not set)}"
if [[ -z "${TRAIN_CMD:-}" ]]; then
  echo "Set TRAIN_CMD before submitting." >&2
  exit 1
fi

exec sbatch \
  --chdir="${CHDIR}" \
  --job-name="${JOB_NAME:-s-study}" \
  --output="${SLURM_LOG_DIR}/slurm-%j.out" \
  --error="${SLURM_LOG_DIR}/slurm-%j.err" \
  --partition="${SBATCH_PARTITION}" \
  "${GRES_ARGS[@]}" \
  "${CONSTRAINT_ARGS[@]}" \
  "${NICE_ARGS[@]}" \
  --cpus-per-task="${SBATCH_CPUS}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --export=ALL \
  "$@" \
  "${SBATCH_SCRIPT}"
