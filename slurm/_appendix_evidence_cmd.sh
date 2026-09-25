#!/usr/bin/env bash
# Inside the study container: refresh current summary CSVs, then build the
# appendix evidence package from the cluster's server-side results tree.  The
# distributed k* audit is intentionally left to the local merged workspace,
# where source-preserving SAE artifacts from every machine are available.
# No training, causal evaluation, model loading, or GPU is involved.
set -euo pipefail
cd /workspace

echo "=== Refresh final merged summaries used by appendix evidence ==="
# ``selected`` is summary_latex.py's historical default.  Do not pass
# --method-set here: the the cluster checkout can legitimately lag the local CLI
# addition, and the default gives the exact intended headline set on both.
# Models/subdirs come from configs/report_model_sets.json (single source of truth).
while IFS='=' read -r model subdir; do
  model="${model%$'\r'}"; subdir="${subdir%$'\r'}"
  [[ -z "${model}" ]] && continue
  if [[ -n "${subdir}" ]]; then
    python analysis/summary_latex.py --model "${model}" --subdir "${subdir}"
  else
    python analysis/summary_latex.py --model "${model}"
  fi
done < <(python analysis/report_model_policy.py --set paper_models --format sources)

echo "=== Build Det./Int. appendix evidence from final server artifacts ==="
python analysis/appendix_evidence.py

echo "Done regenerating appendix evidence."
