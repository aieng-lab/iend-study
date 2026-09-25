#!/usr/bin/env bash
# Inside the study container: rewrite every REPORT from results.json, then overviews.
# No train / causal / GPU.
set -euo pipefail
cd /workspace

RUNS="${RUNS:-runs}"
if [[ "${APPENDIX_LAYER_SELECTION_ONLY:-0}" != "1" ]]; then
  echo "=== REPORT.md / TABLES.txt / metrics.csv under ${RUNS} ==="
  if [[ -n "${MODEL:-}" ]]; then
    python report_study.py --runs "${RUNS}" --model "${MODEL}"
  else
    python report_study.py --runs "${RUNS}"
  fi

  echo "=== Family overview + method-group pivots ==="
  if [[ -n "${MODEL:-}" ]]; then
    python analysis/summarize_runs.py --runs "${RUNS}" --model "${MODEL}"
  else
    python analysis/summarize_runs.py --runs "${RUNS}"
  fi

  echo "=== Global suitability matrix ==="
  python - <<'PY'
from suitability import refresh_global_suitability

result = refresh_global_suitability()
text = result.get("overall_text") or result.get("text") or ""
if text:
    print(text)
print("wrote", result.get("paths"))
PY
fi

if [[ -n "${LAYER_SELECTION_SOURCES:-}" ]]; then
  echo "=== Layer-selection appendix ==="
  IFS=',' read -r -a _layer_sources <<< "${LAYER_SELECTION_SOURCES}"
  _layer_args=()
  for _layer_source in "${_layer_sources[@]}"; do
    [[ -n "${_layer_source}" ]] && _layer_args+=(--source "${_layer_source}")
  done
  python analysis/layer_selection_appendix.py "${_layer_args[@]}"
fi

echo "Done regenerating reports + overviews."
