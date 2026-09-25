#!/usr/bin/env bash
# Fixed-budget cross-model IEND learning-rate screen on gender_en.
#
# Safety contract:
#   bash slurm/iend_lr_cross_model.sh          # preview only; submits nothing
#   bash slurm/iend_lr_cross_model.sh --pilot  # one isolated smoke array task
#   bash slurm/iend_lr_cross_model.sh --go     # one 24-task Slurm array
#
# Scientific factors:
#   models:  llama-3.1-8b, qwen3.5-9b-base
#   methods: GRADIEND and ACTIEND
#   splits:  none and tensors
#   steps:   100 for this compute-aware LR screen (override with MAX_STEPS)
#
# ACTIEND uses split-specific grids because split=none uses running_rms while
# split=tensors uses raw activations. GRADIEND uses one grid for both splits.
# Causal and localization are skipped by default during LR selection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_CSV="${MODELS:-llama-3.1-8b,qwen3.5-9b-base}"
FAMILIES_CSV="${FAMILIES:-gradiend,actiend}"
SPLITS_CSV="${SPLITS:-none,tensors}"
TASK="${TASK:-gender_en}"
SUITE="${SUITE:-full_plus}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-iend_lr_screen_steps${MAX_STEPS}}"

# Relative to the old 500-step proposal, rates are multiplied by five so the
# approximate Adam displacement budget (learning_rate * max_steps) is retained.
# The already-demonstrably-too-low ACTIEND tensor 1e-5 equivalent is omitted.
ACTIEND_NONE_LRS_CSV="${ACTIEND_NONE_LRS:-5e-5,1.5e-4,5e-4}"
ACTIEND_TENSOR_LRS_CSV="${ACTIEND_TENSOR_LRS:-5e-4,1.5e-3,5e-3}"
GRADIEND_LRS_CSV="${GRADIEND_LRS:-5e-4,1.5e-3,5e-3}"

SKIP_EXISTING_FLAG="${SKIP_EXISTING:-1}"
SKIP_CAUSAL_FLAG="${SKIP_CAUSAL:-1}"
SKIP_LOCALIZATION_FLAG="${SKIP_LOCALIZATION:-1}"
PROFILE="${SLURM_PROFILE:-gpumem-24-1x}"
PYTHON_BIN="${PYTHON:-python3}"

mode="preview"
case "${1:-}" in
  "") ;;
  --go)
    mode="full"
    shift
    ;;
  --pilot)
    mode="pilot"
    shift
    ;;
  --help|-h)
    sed -n '1,18p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown argument: ${1}" >&2
    echo "Use no argument (preview), --pilot, or --go." >&2
    exit 2
    ;;
esac
if [[ $# -gt 0 ]]; then
  echo "Unexpected extra arguments: $*" >&2
  exit 2
fi

# Full bulk arrays default to background priority; the one-cell smoke pilot is
# a quick job and stays normal priority. An explicit SBATCH_NICE overrides both.
if [[ -n "${SBATCH_NICE+x}" ]]; then
  ARRAY_NICE="${SBATCH_NICE}"
elif [[ "${mode}" == "pilot" ]]; then
  ARRAY_NICE=0
else
  ARRAY_NICE=1000
fi
if [[ ! "${ARRAY_NICE}" =~ ^[0-9]+$ ]]; then
  echo "Invalid SBATCH_NICE=${ARRAY_NICE@Q}; expected a non-negative integer." >&2
  exit 1
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid MAX_STEPS=${MAX_STEPS@Q}; expected a positive integer." >&2
  exit 1
fi
if [[ ! "${EVAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid EVAL_STEPS=${EVAL_STEPS@Q}; expected a positive integer." >&2
  exit 1
fi

export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/gradiend_sae.env.sh}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
export REPO_DIR="${REPO_DIR}"
export SLURM_DIR="${SCRIPT_DIR}"

SUBMIT_ID="$(date +%Y%m%d_%H%M%S)_$$"
SUBMIT_DIR="${REPO_DIR}/runs/_submit/${SUBMIT_ID}"
mkdir -p "${SUBMIT_DIR}"

export MODELS_CSV FAMILIES_CSV SPLITS_CSV TASK SUITE OUTPUT_ROOT
export ACTIEND_NONE_LRS_CSV ACTIEND_TENSOR_LRS_CSV GRADIEND_LRS_CSV
export MAX_STEPS EVAL_STEPS
export IEND_LR_LAUNCH_MODE="${mode}"
export IEND_LR_JOB_TABLE="${SUBMIT_DIR}/job_table.json"

cd "${REPO_DIR}"
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


def csv(name: str) -> list[str]:
    return [item.strip() for item in os.environ[name].split(",") if item.strip()]


models = csv("MODELS_CSV")
families = set(csv("FAMILIES_CSV"))
splits = set(csv("SPLITS_CSV"))
task = os.environ["TASK"]
suite = os.environ["SUITE"]
root = os.environ["OUTPUT_ROOT"].strip("/")
max_steps = int(os.environ["MAX_STEPS"])
eval_steps = int(os.environ["EVAL_STEPS"])
rows: list[dict[str, object]] = []


def output_subdir(method: str, split: str, lr: str) -> str:
    # Reuse protocol-equivalent existing 500-step ACTIEND tensor cells only.
    # The 100-step screen must remain isolated because train_cache_mode=always
    # intentionally trusts artifacts already present in an output directory.
    if max_steps == 500 and task == "gender_en" and method == "actiend" and split == "tensors":
        if lr in {"1e-5", "1e-05", "1.0e-5", "1.0e-05"}:
            return "actiend_tensor_gender_v2"
        if lr in {"1e-4", "1e-04", "1.0e-4", "1.0e-04"}:
            return "actiend_tensor_gender_lr1e4_v3"
    return f"{root}/{method}/{split}/lr_{lr}"


def add(model: str, method: str, split: str, lr: str) -> None:
    row: dict[str, object] = {
        "index": len(rows),
        "model": model,
        "task": task,
        "suite": suite,
        "methods": method,
        "train_splits": split,
        "output_subdir": output_subdir(method, split, lr),
        "max_steps": max_steps,
        "eval_steps": eval_steps,
    }
    row[f"lr_{method}"] = lr
    rows.append(row)


for model in models:
    if "actiend" in families:
        if "none" in splits:
            for lr in csv("ACTIEND_NONE_LRS_CSV"):
                add(model, "actiend", "none", lr)
        if "tensors" in splits:
            for lr in csv("ACTIEND_TENSOR_LRS_CSV"):
                add(model, "actiend", "tensors", lr)
    if "gradiend" in families:
        for split in ("none", "tensors"):
            if split not in splits:
                continue
            for lr in csv("GRADIEND_LRS_CSV"):
                add(model, "gradiend", split, lr)

if os.environ["IEND_LR_LAUNCH_MODE"] == "pilot":
    if not rows:
        raise SystemExit("No cells selected for pilot")
    pilot = dict(rows[0])
    pilot["index"] = 0
    pilot["smoke"] = True
    lr = pilot.get("lr_actiend") or pilot.get("lr_gradiend")
    pilot["output_subdir"] = (
        f"{root}/_pilot_smoke/{pilot['methods']}/{pilot['train_splits']}/lr_{lr}"
    )
    rows = [pilot]

if not rows:
    raise SystemExit("No LR-screen cells selected")
for index, row in enumerate(rows):
    row["index"] = index

out = Path(os.environ["IEND_LR_JOB_TABLE"])
out.write_text(json.dumps({"jobs": rows}, indent=2), encoding="utf-8")
print(f"Wrote {out} with {len(rows)} cell(s)")
for row in rows:
    lr = row.get("lr_actiend") or row.get("lr_gradiend")
    smoke = " smoke" if row.get("smoke") else ""
    print(
        f"  [{row['index']:02d}] {row['model']} {row['methods']} "
        f"{row['train_splits']} lr={lr} steps={row['max_steps']}"
        f" eval={row['eval_steps']}{smoke} -> {row['output_subdir']}"
    )
PY

N_JOBS="$("${PYTHON_BIN}" -c "import json; print(len(json.load(open(r'${SUBMIT_DIR}/job_table.json'))['jobs']))")"
ARRAY_SPEC="0-$((N_JOBS - 1))"
# concurrency cap per array (ArrayTaskThrottle); ARRAY_THROTTLE=0 removes it
ARRAY_THROTTLE="${ARRAY_THROTTLE:-3}"
[[ "${ARRAY_THROTTLE}" =~ ^[0-9]+$ ]] || { echo "Invalid ARRAY_THROTTLE=${ARRAY_THROTTLE@Q}; expected a non-negative integer (0 = no cap)." >&2; exit 1; }
if [[ "${ARRAY_THROTTLE}" =~ ^[1-9][0-9]*$ ]]; then ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_THROTTLE}"; fi

if [[ "${mode}" == "preview" ]]; then
  echo
  echo "Preview only: no jobs submitted."
  echo "Array nice value: ${ARRAY_NICE} (override with SBATCH_NICE=0 or another non-negative integer)."
  echo "One smoke pilot: bash slurm/iend_lr_cross_model.sh --pilot"
  echo "One ${N_JOBS}-task array: bash slurm/iend_lr_cross_model.sh --go"
  exit 0
fi

# Per-cell factors live in job_table.json. Clear their global counterparts so
# _array_task.py cannot accidentally override an individual row.
export TRAIN_CMD="FAIL_FAST=${FAIL_FAST:-1} SUBMIT_DIR=/workspace/runs/_submit/${SUBMIT_ID} SKIP_EXISTING_FLAG=${SKIP_EXISTING_FLAG} METHODS= TRAIN_SPLITS= LR_GRADIEND= LR_ACTIEND= MAX_STEPS=${MAX_STEPS} EVAL_STEPS=${EVAL_STEPS} SKIP_CAUSAL=${SKIP_CAUSAL_FLAG} SKIP_LOCALIZATION=${SKIP_LOCALIZATION_FLAG} bash /workspace/slurm/_array_task.sh"
export SLURM_ARRAY_COUNT="${N_JOBS}"
export SBATCH_NICE="${ARRAY_NICE}"
if [[ "${mode}" == "pilot" ]]; then
  export JOB_NAME="i-lr-pilot"
else
  export JOB_NAME="i-lr-grid"
fi

echo
echo "Submitting one Slurm array ${ARRAY_SPEC} (${mode}, nice=${SBATCH_NICE})"
echo "job_table: ${SUBMIT_DIR}/job_table.json"
exec bash "${SCRIPT_DIR}/submit_train.sh" "${PROFILE}" \
  --array="${ARRAY_SPEC}" \
  --output="${SLURM_LOG_DIR}/slurm-%A_%a.out" \
  --error="${SLURM_LOG_DIR}/slurm-%A_%a.err"
