#!/usr/bin/env bash
# Mode: array — one task per (model, task), optionally per method family too.
#
#   bash slurm/study_array.sh
#   MODELS=pythia-70m-deduped,gpt2-small TASKS=gender_en,race SUITE=core bash slurm/study_array.sh
#   ARRAY_BY_METHOD=0 MODELS=gpt2-small TASKS=gender_en bash slurm/study_array.sh   # one job per task instead
#
# Writes runs/_submit/<id>/job_table.json and submits an array job.
# Array tasks with existing results.json status=ok are skipped when SKIP_EXISTING=1.
# REFRESH_CAUSAL=1: still re-enter status=ok jobs, but pass --skip-existing so
# trainers reload from artifacts (no GRADIEND/ACTIEND retrain) and causal re-runs.
# Use after causal-protocol fixes without SKIP_EXISTING=0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_CSV="${MODELS:-pythia-70m-deduped}"
TASKS_CSV="${TASKS:-all}"
SUITE="${SUITE:-core}"
SKIP_EXISTING_FLAG="${SKIP_EXISTING:-1}"
# Defaults: one array cell per (model, task, method family) on its own resource tier, and no localization
# stage. Opt out with ARRAY_BY_METHOD=0 / SKIP_LOCALIZATION=0.
ARRAY_BY_METHOD_FLAG="${ARRAY_BY_METHOD:-1}"
SKIP_LOCALIZATION="${SKIP_LOCALIZATION:-1}"
SLURM_PROFILE_OVERRIDE="${SLURM_PROFILE:-}"
# Concurrency cap per array (sbatch "--array=<spec>%N", i.e. the array's ArrayTaskThrottle): at most
# ARRAY_THROTTLE tasks of one array run at once. Default 3; ARRAY_THROTTLE=0 removes the cap. A spec that
# already carries a "%" is left alone. Change a queued array with: scontrol update JobId=<id> ArrayTaskThrottle=N
ARRAY_THROTTLE="${ARRAY_THROTTLE:-3}"
[[ "${ARRAY_THROTTLE}" =~ ^[0-9]+$ ]] || { echo "Invalid ARRAY_THROTTLE=${ARRAY_THROTTLE@Q}; expected a non-negative integer (0 = no cap)." >&2; exit 1; }
with_array_throttle() {
  local spec="$1"
  if [[ "${ARRAY_THROTTLE}" =~ ^[1-9][0-9]*$ && "${spec}" != *%* ]]; then
    printf '%s%%%s' "${spec}" "${ARRAY_THROTTLE}"
  else
    printf '%s' "${spec}"
  fi
}

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
export REPO_DIR="${REPO_DIR}"
export SLURM_DIR="${SCRIPT_DIR}"

SUBMIT_ID="$(date +%Y%m%d_%H%M%S)_$$"
SUBMIT_DIR="${REPO_DIR}/runs/_submit/${SUBMIT_ID}"
mkdir -p "${SUBMIT_DIR}"

cd "${REPO_DIR}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
TABLE_ARGS=(
  --output "${SUBMIT_DIR}/job_table.json"
  --models "${MODELS_CSV}"
  --tasks "${TASKS_CSV}"
  --suite "${SUITE}"
  --output-subdir "${OUTPUT_SUBDIR}"
)
if [[ "${ARRAY_BY_METHOD_FLAG}" == "1" ]]; then
  TABLE_ARGS+=(
    --array-by-method
    --profile-config "${REPO_DIR}/configs/slurm_profiles.yaml"
    --profile-groups-output "${SUBMIT_DIR}/profile_groups.tsv"
  )
  if [[ -n "${SLURM_PROFILE_OVERRIDE}" ]]; then
    TABLE_ARGS+=(--slurm-profile "${SLURM_PROFILE_OVERRIDE}")
  fi
fi
if [[ -n "${METHODS:-}" ]]; then
  TABLE_ARGS+=(--methods "${METHODS}")
fi
if [[ -n "${TRAIN_ABLATIONS:-}" ]]; then
  TABLE_ARGS+=(--train-ablations "${TRAIN_ABLATIONS}")
fi
python slurm/_study_array_table.py "${TABLE_ARGS[@]}"

N_JOBS="$(python -c "import json; print(len(json.load(open(r'${SUBMIT_DIR}/job_table.json'))['jobs']))")"
if [[ "${N_JOBS}" -lt 1 ]]; then
  echo "No jobs in table" >&2
  exit 1
fi

METHODS_ENV="${METHODS:-}"
if [[ "${ARRAY_BY_METHOD_FLAG}" == "1" ]]; then
  # Each job-table row supplies its own method; a global value would override it.
  METHODS_ENV=""
fi
export TRAIN_CMD="SUBMIT_DIR=/workspace/runs/_submit/${SUBMIT_ID} SKIP_EXISTING_FLAG=${SKIP_EXISTING_FLAG} REFRESH_CAUSAL=${REFRESH_CAUSAL:-} FORCE_CAUSAL=${FORCE_CAUSAL:-} REFRESH_ENCODER_EVAL=${REFRESH_ENCODER_EVAL:-} SCALE=${SCALE:-} METHODS=${METHODS_ENV} TRAIN_SPLITS=${TRAIN_SPLITS:-} TRAIN_ABLATIONS=${TRAIN_ABLATIONS:-} CAUSAL_ONLY=${CAUSAL_ONLY:-} NO_RERUN_ORPHANS=${NO_RERUN_ORPHANS:-} TUNE_LR=${TUNE_LR:-} LR_GRADIEND=${LR_GRADIEND:-} LR_ACTIEND=${LR_ACTIEND:-} LR_AGIEND=${LR_AGIEND:-} LR_DECODER_GRADIEND=${LR_DECODER_GRADIEND:-} LR_DECODER_ACTIEND=${LR_DECODER_ACTIEND:-} MAX_STEPS=${MAX_STEPS:-} EVAL_STEPS=${EVAL_STEPS:-} IEND_SELECTION_METRIC=${IEND_SELECTION_METRIC:-} IEND_CONVERGENT_METRIC=${IEND_CONVERGENT_METRIC:-} TRAIN_CACHE_MODE=${TRAIN_CACHE_MODE:-} SKIP_CAUSAL=${SKIP_CAUSAL:-} SKIP_LOCALIZATION=${SKIP_LOCALIZATION:-} OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-} SMOKE=${SMOKE:-} FAIL_FAST=${FAIL_FAST:-1} bash /workspace/slurm/_array_task.sh"
export SLURM_ARRAY_COUNT="${N_JOBS}"

# shellcheck source=/dev/null
source "${SLURM_DIR}/_source_profile.sh"
echo "job_table: ${SUBMIT_DIR}/job_table.json"

submit_array_group() {
  local profile="$1"
  local array_spec
  array_spec="$(with_array_throttle "$2")"
  local profile_file="${SLURM_DIR}/profiles/${profile}.sh"
  if [[ ! -f "${profile_file}" ]]; then
    echo "Unknown profile: ${profile}" >&2
    echo "Available: $(for f in "${SLURM_DIR}"/profiles/*.sh; do basename "$f" .sh; done | tr '\n' ' ')" >&2
    return 1
  fi
  source_profile "${profile_file}"
  # Strip CR (Windows sync) / whitespace so Slurm never sees ``--mem=$'64G\r'``.
  SBATCH_MEM="$(printf '%s' "${SBATCH_MEM:-}" | tr -d '\r' | xargs)"
  SBATCH_CPUS="$(printf '%s' "${SBATCH_CPUS:-}" | tr -d '\r' | xargs)"
  SBATCH_TIME="$(printf '%s' "${SBATCH_TIME:-}" | tr -d '\r' | xargs)"
  SBATCH_PARTITION="$(printf '%s' "${SBATCH_PARTITION:-}" | tr -d '\r' | xargs)"
  SBATCH_GRES="$(printf '%s' "${SBATCH_GRES:-}" | tr -d '\r' | xargs)"
  SBATCH_CONSTRAINT="$(printf '%s' "${SBATCH_CONSTRAINT:-}" | tr -d '\r' | xargs)"
  SBATCH_NICE="${SBATCH_NICE:-1000}"
  if [[ ! "${SBATCH_MEM}" =~ ^[0-9]+[KMG]?$ ]]; then
    echo "Invalid SBATCH_MEM=${SBATCH_MEM@Q} from profile ${profile} (${profile_file})" >&2
    return 1
  fi
  if [[ ! "${SBATCH_NICE}" =~ ^[0-9]+$ ]]; then
    echo "Invalid SBATCH_NICE=${SBATCH_NICE@Q}; expected a non-negative integer." >&2
    return 1
  fi

  local -a constraint_args=()
  if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
    constraint_args+=(--constraint="${SBATCH_CONSTRAINT}")
  fi

  # SBATCH_DEPENDENCY="afterany:<jobid>[:<jobid>...]" holds this array until those jobs (whole
  # arrays included) have ended, whatever their state.
  local -a dependency_args=()
  if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    dependency_args+=(--dependency="${SBATCH_DEPENDENCY}")
    echo "  dependency=${SBATCH_DEPENDENCY}"
  fi

  echo "Profile: ${profile}"
  echo "  partition=${SBATCH_PARTITION} gres=${SBATCH_GRES} constraint=${SBATCH_CONSTRAINT:-(none)} cpus=${SBATCH_CPUS} mem=${SBATCH_MEM} time=${SBATCH_TIME} nice=${SBATCH_NICE}"
  echo "Submitting array ${array_spec} suite=${SUITE} by_method=${ARRAY_BY_METHOD_FLAG} skip_existing=${SKIP_EXISTING_FLAG} refresh_causal=${REFRESH_CAUSAL:-0} force_causal=${FORCE_CAUSAL:-0} fail_fast=${FAIL_FAST:-1} output_subdir=${OUTPUT_SUBDIR:-}"
  sbatch \
    --chdir="${USER_DIR}" \
    --job-name="${JOB_NAME:-s-arr}" \
    --array="${array_spec}" \
    --output="${SLURM_LOG_DIR}/slurm-%A_%a.out" \
    --error="${SLURM_LOG_DIR}/slurm-%A_%a.err" \
    --partition="${SBATCH_PARTITION}" \
    --gres="${SBATCH_GRES}" \
    "${constraint_args[@]}" \
    "${dependency_args[@]}" \
    --nice="${SBATCH_NICE}" \
    --cpus-per-task="${SBATCH_CPUS}" \
    --mem="${SBATCH_MEM}" \
    --time="${SBATCH_TIME}" \
    --export=ALL \
    "${SLURM_DIR}/study.sbatch"
}

if [[ "${ARRAY_BY_METHOD_FLAG}" == "1" ]]; then
  while IFS=$'\t' read -r profile array_spec; do
    [[ -n "${profile}" ]] || continue
    submit_array_group "${profile}" "${array_spec}"
  done < "${SUBMIT_DIR}/profile_groups.tsv"
else
  submit_array_group "${SLURM_PROFILE_OVERRIDE:-gpumem-24-1x}" "0-$((N_JOBS - 1))"
fi
