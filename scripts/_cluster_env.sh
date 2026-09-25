# Sourced by scripts/rsync_*.sh (expects ROOT to be set).
# Loads the gitignored ROOT/cluster.env and requires CLUSTER_USER_DIR, the only
# site-specific value; REMOTE_REPO / REMOTE_LOG_DIR / ... default from it.
if [[ -f "${ROOT}/cluster.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/cluster.env"
  set +a
fi
if [[ -z "${CLUSTER_USER_DIR:-}" ]]; then
  echo "error: CLUSTER_USER_DIR is not set." >&2
  echo "       copy cluster.env.example to cluster.env and fill it in." >&2
  exit 1
fi
