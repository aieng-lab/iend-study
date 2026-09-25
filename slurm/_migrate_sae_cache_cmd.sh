#!/usr/bin/env bash
# Inside the study container: shrink sae_cache/*/*.npy (dense) to *.npz (sparse).
# Pure numpy I/O, no torch/model/GPU — see scripts/migrate_sae_cache_to_sparse.py.
set -euo pipefail
cd /workspace

RUNS="${RUNS:-runs}"
DRY_RUN_FLAG=""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAG="--dry-run"
fi

echo "=== Migrating sae_cache under ${RUNS} (dry_run=${DRY_RUN:-0}) ==="
python scripts/migrate_sae_cache_to_sparse.py "${RUNS}" ${DRY_RUN_FLAG}
