#!/usr/bin/env bash
# Fast headline-completion status.  Uses the latest generated coverage CSV;
# it never rereads heavyweight results.json artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PYTHON:-python}"
command -v "${PY_BIN}" >/dev/null 2>&1 || PY_BIN="python3"
exec "${PY_BIN}" "${ROOT}/scripts/headline_completeness.py" "$@"
