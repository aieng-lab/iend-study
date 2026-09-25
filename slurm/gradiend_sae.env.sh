#!/usr/bin/env bash

# Study-repo Slurm env (mirrors gradiend/slurm/gradiend.env.sh).
# The study repo lives at ${USER_DIR}/gradiend-sae, so USER_DIR (the base dir for
# conda, caches, logs, ...) defaults to the parent of this checkout. No user
# name is hard-coded; export USER_DIR to override.
: "${USER_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${REPO_URL:=}"
: "${REPO_DIR:=${USER_DIR}/gradiend-sae}"
: "${GRADIEND_REPO_DIR:=${USER_DIR}/gradiend}"
: "${DATASET_DIR:=${USER_DIR}/datasets}"

: "${SIF_DIR:=${USER_DIR}/apptainer/images}"
: "${MINICONDA_IMAGE:=docker://anaconda/miniconda:26.3.2}"
: "${APPTAINER_IMAGE:=${SIF_DIR}/miniconda-26.3.2-sandbox}"

: "${CONDA_HOME:=${USER_DIR}/conda-home}"
: "${CONDA_ENV_DIR:=${USER_DIR}/conda-envs/gradiend}"
: "${CONDA_PKGS_DIR:=${USER_DIR}/conda-pkgs}"
: "${PYTHON_VERSION:=3.11}"

: "${HF_TOKEN_FILE:=${USER_DIR}/hf_token.txt}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
fi

: "${HF_HOME:=${USER_DIR}/hf-cache}"
: "${HF_HUB_DISABLE_XET:=1}"
: "${TORCH_HOME:=${USER_DIR}/torch-cache}"
: "${XDG_CACHE_HOME:=${USER_DIR}/cache}"
: "${TMPDIR:=${USER_DIR}/tmp}"

: "${SLURM_LOG_DIR:=${USER_DIR}/slurm-logs}"
: "${LOCAL_BASE:=${USER_DIR}/local}"
: "${SLURM_DIR:=${REPO_DIR}/slurm}"

: "${TRAIN_CMD:=python run_study.py --model pythia-70m-deduped --task gender_en}"

export USER_DIR REPO_URL REPO_DIR GRADIEND_REPO_DIR DATASET_DIR
export SIF_DIR MINICONDA_IMAGE APPTAINER_IMAGE
export CONDA_HOME CONDA_ENV_DIR CONDA_PKGS_DIR PYTHON_VERSION
export HF_TOKEN HF_TOKEN_FILE HF_HOME HF_HUB_DISABLE_XET TORCH_HOME XDG_CACHE_HOME TMPDIR
export SLURM_LOG_DIR LOCAL_BASE SLURM_DIR TRAIN_CMD
