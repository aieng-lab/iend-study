#!/usr/bin/env bash
# Show CPU / RAM / GPU actually used vs requested for a Slurm job.
#
#   bash slurm/job_usage.sh 1234567
#   bash slurm/job_usage.sh 3776425_18
#   bash slurm/job_usage.sh            # most recent job for $USER
#
# Finished jobs: seff (CPU/RAM) + sacct TRESUsageInMax (gres/gpumem, gres/gpuutil).
# Running jobs:  sstat snapshot + nvidia-smi on the allocated node (live, not peak).

set -euo pipefail

FLOORS=(11 16 24 48 80 141)

JOB="${1:-}"
if [[ -z "${JOB}" ]]; then
  JOB="$(sacct -X -n -P --user="${USER}" --format=JobID --starttime=now-21days | awk -F. 'NF==1{id=$1} END{print id}')"
  if [[ -z "${JOB}" ]]; then
    echo "No recent jobs found. Pass a JobID: bash slurm/job_usage.sh JOBID" >&2
    exit 1
  fi
  echo "Using latest job ${JOB}"
fi

tres_field() {
  local tres="$1" key="$2"
  printf '%s\n' "${tres}" | tr ',' '\n' | awk -F= -v k="${key}" '$1==k {print $2; exit}'
}

to_gib() {
  # 11.29G / 10880M / 10.63 -> GiB float (Slurm --units=G is 1024-based for mem).
  awk -v x="$1" 'BEGIN {
    if (x == "" || x == "0") { print ""; exit }
    u = tolower(x)
    if (u ~ /t$/) { gsub(/t$/, "", u); printf "%.2f", u * 1024; exit }
    if (u ~ /g$/) { gsub(/g$/, "", u); printf "%.2f", u + 0; exit }
    if (u ~ /m$/) { gsub(/m$/, "", u); printf "%.2f", u / 1024; exit }
    if (u ~ /k$/) { gsub(/k$/, "", u); printf "%.2f", u / 1024 / 1024; exit }
    printf "%.2f", u + 0
  }'
}

vram_floor() {
  awk -v used="$1" -v floors="11 16 24 48 80 141" 'BEGIN {
    n = split(floors, f, " ")
    u = used + 0
    for (i = 1; i <= n; i++) if (u < f[i] + 0) { print f[i]; exit }
    print f[n]
  }'
}

cpu_suggest() {
  # Study jobs keep 4 CPUs (dataloaders). Do not drop to 2 from a single seff %.
  echo 4
}

SEFF_OUT=""
echo "=== seff ${JOB} ==="
if command -v seff >/dev/null 2>&1; then
  SEFF_OUT="$(seff "${JOB}" 2>&1 || true)"
  printf '%s\n' "${SEFF_OUT}"
else
  echo "(seff not on PATH; using sacct only)"
fi

echo
echo "=== sacct ${JOB} (requested vs used) ==="
SACCT="$(sacct -j "${JOB}" --units=G \
  --format=JobID,JobName,State,Elapsed,ReqCPUS,AllocCPUS,ReqMem,MaxRSS,AveRSS,MaxVMSize,AllocTRES,TRESUsageInMax \
  -P)"
printf '%s\n' "${SACCT}"

STATE="$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR==2 {print $3; exit}')"
REQ_CPUS="$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR==2 {print $5; exit}')"
REQ_MEM="$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR==2 {print $7; exit}')"
ALLOC_TRES="$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR==2 {print $11; exit}')"

# Accounting (MaxRSS, TRESUsageInMax) lives on the .batch / .0 step, not the job line.
STEP_LINE="$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR>1 && $8 != "" {print; exit}')"
if [[ -z "${STEP_LINE}" ]]; then
  STEP_LINE="$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR>1 && $12 != "" {print; exit}')"
fi
MAX_RSS="$(printf '%s\n' "${STEP_LINE}" | awk -F'|' '{print $8}')"
TRES_MAX="$(printf '%s\n' "${STEP_LINE}" | awk -F'|' '{print $12}')"
GPU_MEM="$(tres_field "${TRES_MAX}" gres/gpumem)"
GPU_UTIL="$(tres_field "${TRES_MAX}" gres/gpuutil)"
TRES_HOST_MEM="$(tres_field "${TRES_MAX}" mem)"

RUNNING=0
if [[ "${STATE}" == RUNNING* ]] || squeue -h -j "${JOB}" >/dev/null 2>&1; then
  RUNNING=1
fi

if [[ "${RUNNING}" -eq 1 ]]; then
  echo
  echo "=== live snapshot (job still RUNNING; not a peak) ==="
  STEP="${JOB}"
  if [[ "${JOB}" != *.* ]]; then
    STEP="${JOB}.batch"
  fi
  if command -v sstat >/dev/null 2>&1; then
    sstat -j "${STEP}" --noconvert -P \
      --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU,NTasks 2>/dev/null \
      || sstat -j "${JOB}" --noconvert -P \
           --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU,NTasks 2>/dev/null \
      || echo "(sstat failed; wait until the job finishes for MaxRSS / gres/gpumem)"
  fi
  NODE="$(squeue -h -j "${JOB}" -o '%N' 2>/dev/null | head -n1 || true)"
  if [[ -n "${NODE}" ]]; then
    echo "Node: ${NODE}"
    echo "Trying nvidia-smi on the job (live VRAM, not high-water mark)..."
    if srun --jobid="${JOB}" --overlap --ntasks=1 --quiet \
         nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
         --format=csv 2>/dev/null; then
      :
    else
      echo "(could not srun --overlap nvidia-smi; from the login node try:)"
      echo "  srun --jobid=${JOB} --overlap nvidia-smi"
    fi
  fi
  echo
  echo "seff/sacct MaxRSS and TRESUsageInMax stay empty until the task completes."
  echo "Re-run: bash slurm/job_usage.sh ${JOB}"
fi

GPU_GIB="$(to_gib "${GPU_MEM}")"
RSS_GIB="$(to_gib "${MAX_RSS:-${TRES_HOST_MEM}}")"
REQ_MEM_GIB="$(to_gib "${REQ_MEM}")"
SEFF_CPU_PCT="$(printf '%s\n' "${SEFF_OUT}" | awk -F'[ %]' '/CPU Efficiency:/ {print $3; exit}')"
SEFF_MEM_PCT="$(printf '%s\n' "${SEFF_OUT}" | awk -F'[ %]' '/Memory Efficiency:/ {print $3; exit}')"

echo
echo "=== recommendation ==="
if [[ -z "${GPU_GIB}" && "${RUNNING}" -eq 1 ]]; then
  echo "No GPU accounting yet (RUNNING). Use the live nvidia-smi line above, or wait until COMPLETED."
elif [[ -z "${GPU_GIB}" ]]; then
  echo "No gres/gpumem in TRESUsageInMax. GPU accounting may be off; check runs/*/results.json peak_gpu_* instead."
else
  FLOOR="$(vram_floor "${GPU_GIB}")"
  echo "GPU VRAM peak:  ${GPU_GIB} GiB   (gpuutil=${GPU_UTIL:-?})"
  echo "  -> profile    gpumem-${FLOOR}-1x"
  if awk -v u="${GPU_GIB}" 'BEGIN { exit !(u >= 11) }'; then
    echo "  -> not 11:    ${GPU_GIB} GiB is at/above the 11 GiB floor"
  fi
fi
if [[ -n "${RSS_GIB}" ]]; then
  MEM_SUGGEST="$(awk -v u="${RSS_GIB}" 'BEGIN {
    need = u * 1.5
    n = 64
    if (need > 64) n = int(need + 8)
    printf "%dG", n
  }')"
  echo "Host RAM peak:  ${RSS_GIB} GiB of ${REQ_MEM:-?}  (seff ${SEFF_MEM_PCT:-?}%)"
  echo "  -> SBATCH_MEM ${MEM_SUGGEST}  (keep >=64G; 32G OOMed on other tasks)"
fi
if [[ -n "${SEFF_CPU_PCT}" && -n "${REQ_CPUS}" ]]; then
  CPUS="$(cpu_suggest "${REQ_CPUS}" "${SEFF_CPU_PCT}")"
  echo "CPU:            ${SEFF_CPU_PCT}% of ${REQ_CPUS} cores"
  echo "  -> SBATCH_CPUS ${CPUS}"
fi
if [[ "${ALLOC_TRES}" == *h200* || "${ALLOC_TRES}" == *H200* ]]; then
  echo "Allocated GRES: H200 — overkill if VRAM peak is well below 141 GiB."
fi
echo
echo "the cluster VRAM floors: ${FLOORS[*]} GiB (smallest strictly above peak)."
echo "This job: ${JOB}  state=${STATE}  elapsed=$(printf '%s\n' "${SACCT}" | awk -F'|' 'NR==2{print $4}')"
