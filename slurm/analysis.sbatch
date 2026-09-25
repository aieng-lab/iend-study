#!/usr/bin/env bash
# CPU analysis job (no GPU). Used by regenerate_reports.sh.
#SBATCH --job-name=sae-regen
#SBATCH --chdir=/path/to/workdir
#SBATCH --output=/path/to/workdir/slurm-logs/slurm-%j.out
#SBATCH --error=/path/to/workdir/slurm-logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --mail-user=you@example.org
#SBATCH --mail-type=FAIL

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-/path/to/workdir/gradiend-sae/slurm/gradiend_sae.env.sh}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

RUN_SCRIPT="${SLURM_DIR}/study_run.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Missing run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${RUN_SCRIPT}"
