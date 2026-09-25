#!/usr/bin/env bash
# Shared study runtime (Apptainer + conda + TRAIN_CMD). Sourced by study.sbatch.

set -euo pipefail

: "${USER_DIR:?USER_DIR must be set}"
: "${REPO_DIR:?REPO_DIR must be set}"
: "${CONDA_ENV_DIR:?CONDA_ENV_DIR must be set}"
: "${APPTAINER_IMAGE:?APPTAINER_IMAGE must be set}"
: "${TRAIN_CMD:?TRAIN_CMD must be set}"

mkdir -p "${SLURM_LOG_DIR}" "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${LOCAL_BASE}"

if [[ ! -d "${APPTAINER_IMAGE}" && ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
  exit 1
fi

if [[ ! -d "${CONDA_ENV_DIR}" ]]; then
  echo "Conda environment not found: ${CONDA_ENV_DIR}" >&2
  exit 1
fi

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Repo dir: ${REPO_DIR}"
echo "Train command: ${TRAIN_CMD}"

export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

BIND_ARGS=(--bind "${USER_DIR}:${USER_DIR}" --bind "${REPO_DIR}:/workspace")
if [[ -n "${GRADIEND_REPO_DIR:-}" && -d "${GRADIEND_REPO_DIR}" ]]; then
  BIND_ARGS+=(--bind "${GRADIEND_REPO_DIR}:${GRADIEND_REPO_DIR}")
fi
BIND_ARGS+=(--bind "${TMPDIR}:${TMPDIR}")

NV_ARGS=()
if [[ "${APPTAINER_NV:-1}" != "0" ]]; then
  NV_ARGS+=(--nv)
fi

# The study binds its shared storage explicitly and sets HOME inside the
# container.  Avoid Apptainer's redundant automatic host-home mount: on skadi
# that mount fails during container creation because /home/USER is an
# automount/symlink-backed path.
apptainer exec --no-home "${NV_ARGS[@]}" \
  "${BIND_ARGS[@]}" \
  --pwd /workspace \
  "${APPTAINER_IMAGE}" \
  bash -lc '
set -euo pipefail

export HOME="'"${CONDA_HOME}"'"
export CONDA_PKGS_DIRS="'"${CONDA_PKGS_DIR}"'"
export CONDA_ENVS_DIRS="$(dirname "'"${CONDA_ENV_DIR}"'")"
export HF_HOME="'"${HF_HOME}"'"
export HF_HUB_DISABLE_XET="'"${HF_HUB_DISABLE_XET:-1}"'"
export TORCH_HOME="'"${TORCH_HOME}"'"
export XDG_CACHE_HOME="'"${XDG_CACHE_HOME}"'"
export TMPDIR="'"${TMPDIR}"'"
export PYTHONPATH="'"${GRADIEND_REPO_DIR:-}"':${PYTHONPATH:-}"

unset TRANSFORMERS_CACHE SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE

set +u
source /opt/miniconda3/etc/profile.d/conda.sh
set -u
conda activate "'"${CONDA_ENV_DIR}"'"

# Triton needs a C compiler when it JIT-compiles GPU kernels. The AxBench
# installer supplies this compiler in its dedicated conda prefix; leave other
# environments unchanged and honor an explicitly supplied CC.
if [[ -z "${CC:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
fi

# Torch must work before anything else (sae_lens imports torch).
if ! python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)" ; then
  echo "ERROR: torch import failed in ${CONDA_ENV_DIR}." >&2
  echo "Typical cause: NCCL mismatch (undefined symbol ncclCommResume)." >&2
  echo "Fix the env once (do not pip-install torch from a Slurm job):" >&2
  echo "  python -m pip install --force-reinstall \"nvidia-nccl-cu12\"" >&2
  echo "  # or reinstall the cluster-matched torch CUDA wheel" >&2
  exit 1
fi

# Install sae-lens only if missing. --no-deps so pip cannot upgrade/break torch/NCCL.
if ! python -c "import sae_lens" >/dev/null 2>&1; then
  echo "sae-lens missing; installing with --no-deps (will not touch torch) …"
  python -m pip install --no-cache-dir --no-deps "sae-lens>=5.0" || true
fi
if python -c "import sae_lens; print(\"sae_lens\", getattr(sae_lens, \"__version__\", \"unknown\"))" ; then
  :
else
  echo "WARNING: sae_lens still not importable; SAE baselines will fail until the env is fixed." >&2
fi

SSL_CERT_FILE="$(python - <<PY
import certifi
print(certifi.where())
PY
)"
export SSL_CERT_FILE REQUESTS_CA_BUNDLE="${SSL_CERT_FILE}" CURL_CA_BUNDLE="${SSL_CERT_FILE}"

cd /workspace
echo "Running: '"${TRAIN_CMD}"'"
eval "'"${TRAIN_CMD}"'"
'

echo "Job finished: $(date)"
