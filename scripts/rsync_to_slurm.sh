#!/usr/bin/env bash
# Push project files to the Slurm host with ONE ssh connection (= one password).
# Never deletes remote files.
#
# Why tar-over-ssh instead of rsync: Windows OpenSSH has no ControlMaster, so
# every rsync/ssh invocation prompted for the password again (probe + repo +
# package = 3 prompts, the last one minutes in). One tar stream carries both
# trees over one connection. Files are overwritten, nothing is deleted.
#
# Sends: this repo -> $CLUSTER_USER_DIR/gradiend-sae, and (if present) the
# sibling ../gradiend/gradiend package -> $CLUSTER_USER_DIR/gradiend/gradiend.
#
#   bash scripts/rsync_to_slurm.sh           # dry-run: lists files, no ssh, no password
#   bash scripts/rsync_to_slurm.sh --go      # actually copy (one password prompt)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/_cluster_env.sh"

: "${REMOTE:=slurm}"
: "${SYNC_GRADIEND_PACKAGE:=1}"

PARENT="$(dirname "${ROOT}")"
REPO_NAME="$(basename "${ROOT}")"
PKG_REL="gradiend/gradiend"   # relative to both $PARENT (local) and $CLUSTER_USER_DIR (remote)
if [[ "${REPO_NAME}" != "gradiend-sae" ]]; then
  echo "error: local checkout dir must be named gradiend-sae (got ${REPO_NAME}) so paths map 1:1 to the cluster." >&2
  exit 1
fi

GO=0
if [[ "${1:-}" == "--go" || "${1:-}" == "-y" ]]; then GO=1; fi

MEMBERS=("${REPO_NAME}")
if [[ "${SYNC_GRADIEND_PACKAGE}" == "1" && -d "${PARENT}/${PKG_REL}" ]]; then
  MEMBERS+=("${PKG_REL}")
else
  echo "note: sibling core package not found at ${PARENT}/${PKG_REL}; syncing repo only"
fi

# Member names are relative to $PARENT, so patterns are prefixed with the repo dir.
R="${REPO_NAME}"
EXCLUDES=(
  --exclude="${R}/cluster.env"
  --exclude="${R}/data"
  --exclude="${R}/runs"
  --exclude="${R}/.git"
  --exclude="${R}/slurm-logs"
  --exclude="${R}/tmp"
  --exclude="${R}/.pytest_tmp_*"
  --exclude="${R}/pytest_tmp_*"
  --exclude="${R}/analysis/tables"
  --exclude="${R}/analysis/figures"
  --exclude="${R}/analysis/plots"
  --exclude="${R}/iclr2027"
  --exclude="${R}/.claude"
  --exclude="${R}/.cursor"
  --exclude="${R}/gradiend_requests"
  --exclude="${R}/error_report.jsonl*"
  --exclude="${R}/*.log"
  --exclude="__pycache__"
  --exclude=".pytest_cache"
  --exclude=".ruff_cache"
  --exclude=".venv"
  --exclude=".idea"
  --exclude="*.pyc"
)

echo "from: ${PARENT}  members: ${MEMBERS[*]}"
echo "to:   ${REMOTE}:${CLUSTER_USER_DIR}  (no delete)"

if [[ "${GO}" == "0" ]]; then
  echo "(dry-run: file list, no ssh; pass --go to copy)"
  tar -C "${PARENT}" "${EXCLUDES[@]}" -cf - "${MEMBERS[@]}" | tar -tf - | sed -n '1,400p'
  N="$(tar -C "${PARENT}" "${EXCLUDES[@]}" -cf - "${MEMBERS[@]}" | tar -tf - | grep -vc '/$' || true)"
  B="$(tar -C "${PARENT}" "${EXCLUDES[@]}" -cf - "${MEMBERS[@]}" | wc -c)"
  echo "... ${N} files, $((B / 1024 / 1024)) MiB uncompressed (first 400 paths shown)"
  exit 0
fi

# Prefer Windows' own ssh.exe: Git Bash's MSYS ssh cannot show a password prompt
# when stdin is a pipe outside mintty and just hangs silently. Override: SSH_BIN=...
SSH_BIN="${SSH_BIN:-}"
if [[ -z "${SSH_BIN}" ]]; then
  for c in /c/Windows/System32/OpenSSH/ssh.exe /c/WINDOWS/System32/OpenSSH/ssh.exe; do
    if [[ -x "$c" ]]; then SSH_BIN="$c"; break; fi
  done
  SSH_BIN="${SSH_BIN:-ssh}"
fi
echo "Sending over one ssh connection (${SSH_BIN}) -- enter the password once..."
tar -C "${PARENT}" "${EXCLUDES[@]}" -czf - "${MEMBERS[@]}" \
  | "${SSH_BIN}" -o ClearAllForwardings=yes -o ConnectTimeout=15 "${REMOTE}" \
      "mkdir -p '${CLUSTER_USER_DIR}' && tar -xzf - -C '${CLUSTER_USER_DIR}' --no-same-owner"
echo "Synced ${MEMBERS[*]} to ${REMOTE}:${CLUSTER_USER_DIR} (remote-only files untouched)."
