#!/usr/bin/env bash
# Run the entire post-hoc analysis pipeline for one or more models in one
# command: paper summary tables (PDF), layer-vs-performance plots, and the
# extra paper figures (suitability scatter, one-pole/two-pole chart,
# SAE-selection heatmap, compute summary) — instead of calling
# summary_tables_pdf.sh / layer_performance_plots.py / paper_figures.py
# separately.
#
# Usage:
#   MODELS=gpt2-small bash scripts/run_full_analysis.sh
#   MODELS=gpt2-small OUTPUT_SUBDIR=suite_full bash scripts/run_full_analysis.sh
#   MODELS=gpt2-small,pythia-70m-deduped OUTPUT_SUBDIR=suite_full bash scripts/run_full_analysis.sh
#
# OUTPUT_SUBDIR applies to the summary tables and the layer plots (both read
# runs/{model}/{OUTPUT_SUBDIR}/...). paper_figures.py's suitability scatter
# and one-pole/two-pole chart intentionally stay on the CORE tree unless you
# pass PAPER_FIGURES_SUBDIR explicitly — as of 2026-08-19 suite_full's causal
# stage is still mostly incomplete (see CLAUDE.md's "OUTPUT_SUBDIR" note), so
# pointing those two specifically at it makes them sparser, not more
# complete; the SAE feature-selection heatmap always needs SAE_SELECT_SUBDIR
# (default suite_full) since those ablations were never run under
# --suite core at all.
#
# Writes (auto-named the same way each underlying script names itself, so
# different MODELS/subdir combinations never collide):
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables.pdf         (tables only, fast)
#   analysis/figures/layer_<model>[_<subdir>]/layer_vs_*.pdf
#   analysis/figures/paper_<model>[_<subdir>]/*.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_with_figures_<model>.pdf
#     (2nd PDF, one per model: the SAME tables PDF above, followed by that
#     model's layer plots and paper figures, all pages concatenated via
#     analysis/merge_pdfs.py — set SKIP_COMBINED=1 to skip building this)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "${PYTHON}"
    return 0
  fi
  local c path
  for c in python python.exe py python3; do
    path="$(command -v "${c}" 2>/dev/null || true)"
    [[ -z "${path}" ]] && continue
    case "${path}" in
      *WindowsApps*) continue ;;
    esac
    if [[ "${c}" == py ]]; then
      if py -3 -c "import pandas" >/dev/null 2>&1; then
        printf '%s\n' 'py -3'
        return 0
      fi
      continue
    fi
    if "${c}" -c "import pandas" >/dev/null 2>&1; then
      printf '%s\n' "${c}"
      return 0
    fi
  done
  local extra
  for extra in \
    "${ROOT}/../miniconda/python.exe" \
    /c/Git/miniconda/python.exe \
    "${CONDA_PREFIX:-}/python.exe"
  do
    [[ -x "${extra}" ]] || [[ -f "${extra}" ]] || continue
    if "${extra}" -c "import pandas" >/dev/null 2>&1; then
      printf '%s\n' "${extra}"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(resolve_python)"; then
  echo "error: no python on PATH (set PYTHON=...)." >&2
  exit 1
fi
# shellcheck disable=SC2206
PYTHON=(${PYTHON_BIN})

# When this WSL launcher selects python.exe from the Windows PATH, convert
# absolute WSL paths before passing them to Python.  Without this, /mnt/c/...
# is interpreted by Windows pathlib as C:\\mnt\\c\\..., which creates a
# shadow output tree and leaves the real report files stale.
PYTHON_NEEDS_WINDOWS_PATHS=0
if [[ -n "${WSL_INTEROP:-}" || "$(uname -r 2>/dev/null || true)" == *[Mm]icrosoft* ]]; then
  PYTHON_EXE="$(command -v "${PYTHON[0]}" 2>/dev/null || true)"
  case "${PYTHON_EXE,,}" in
    *.exe) PYTHON_NEEDS_WINDOWS_PATHS=1 ;;
  esac
fi

python_path() {
  local path="$1"
  if [[ "${PYTHON_NEEDS_WINDOWS_PATHS}" -eq 1 ]]; then
    wslpath -w -- "${path}"
  else
    printf '%s\n' "${path}"
  fi
}

MODELS="${MODELS:-gpt2-small}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
PAPER_FIGURES_SUBDIR="${PAPER_FIGURES_SUBDIR:-}"
SAE_SELECT_SUBDIR="${SAE_SELECT_SUBDIR:-suite_full}"
SKIP_COMBINED="${SKIP_COMBINED:-}"

IFS=',' read -r -a RAW_MODEL_ARR <<< "${MODELS}"
MODEL_ARR=()
for model in "${RAW_MODEL_ARR[@]}"; do
  model="${model//[[:space:]]/}"
  [[ -n "${model}" ]] && MODEL_ARR+=("${model}")
done

echo "=== Summary tables (scripts/summary_tables_pdf.sh) — plain, fast ==="
TABLES_LOG="$(mktemp)"
trap 'rm -f "${TABLES_LOG}"' EXIT
MODELS="${MODELS}" OUTPUT_SUBDIR="${OUTPUT_SUBDIR}" PYTHON="${PYTHON_BIN}" bash scripts/summary_tables_pdf.sh | tee "${TABLES_LOG}"
mapfile -t TABLES_PDFS < <(grep '^Wrote .*summary_tables\.pdf$' "${TABLES_LOG}" | sed 's/^Wrote //')
if [[ "${#TABLES_PDFS[@]}" -ne "${#MODEL_ARR[@]}" ]]; then
  echo "error: expected ${#MODEL_ARR[@]} per-model table PDFs, found ${#TABLES_PDFS[@]}." >&2
  exit 1
fi

for model_index in "${!MODEL_ARR[@]}"; do
  model="${MODEL_ARR[$model_index]}"
  TABLES_PDF="${TABLES_PDFS[$model_index]}"

  echo ""
  echo "=== Layer-vs-performance plots: ${model} ==="
  layer_suffix="${model}"
  [[ -n "${OUTPUT_SUBDIR}" ]] && layer_suffix="${layer_suffix}_${OUTPUT_SUBDIR}"
  layer_args=(--model "${model}")
  [[ -n "${OUTPUT_SUBDIR}" ]] && layer_args+=(--subdir "${OUTPUT_SUBDIR}")
  "${PYTHON[@]}" analysis/layer_performance_plots.py "${layer_args[@]}"
  LAYER_DIR="analysis/figures/layer_${layer_suffix}"

  echo ""
  echo "=== Extra paper figures: ${model} ==="
  paper_suffix="${model}"
  [[ -n "${PAPER_FIGURES_SUBDIR}" ]] && paper_suffix="${paper_suffix}_${PAPER_FIGURES_SUBDIR}"
  paper_args=(--model "${model}" --sae-select-subdir "${SAE_SELECT_SUBDIR}")
  [[ -n "${PAPER_FIGURES_SUBDIR}" ]] && paper_args+=(--subdir "${PAPER_FIGURES_SUBDIR}")
  "${PYTHON[@]}" analysis/paper_figures.py "${paper_args[@]}"
  PAPER_DIR="analysis/figures/paper_${paper_suffix}"

  if [[ -z "${SKIP_COMBINED}" ]]; then
    echo ""
    echo "=== Combined tables+figures PDF: ${model} ==="
    COMBINED="$(dirname "${TABLES_PDF}")/summary_tables_with_figures_${model}.pdf"
    "${PYTHON[@]}" analysis/merge_pdfs.py \
      --tables "$(python_path "${TABLES_PDF}")" \
      --figures "$(python_path "${LAYER_DIR}")" "$(python_path "${PAPER_DIR}")" \
      --out "$(python_path "${COMBINED}")"
  fi
done

echo ""
echo "Done."
