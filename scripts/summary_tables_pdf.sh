#!/usr/bin/env bash
# Rebuild paper summary .tex snippets and compile one PDF per model.
#
# Usage (from repo root, Git Bash / WSL):
#   bash scripts/summary_tables_pdf.sh
#   MODELS=gpt2-small bash scripts/summary_tables_pdf.sh
#   MODELS=gpt2-small,pythia-70m-deduped bash scripts/summary_tables_pdf.sh
#
# OUTPUT_SUBDIR reads runs/{model}/{OUTPUT_SUBDIR}/{task}/results.json instead
# of runs/{model}/{task}/results.json — use this for a tree produced with
# `OUTPUT_SUBDIR=...` on slurm/study_array.sh (e.g. a `--suite full` job kept
# separate from the ongoing `--suite core` tree, see CLAUDE.md's
# "OUTPUT_SUBDIR" note):
#   MODELS=gpt2-small OUTPUT_SUBDIR=suite_full bash scripts/summary_tables_pdf.sh
#
# With OUTPUT_SUBDIR unset, the paper source map is used: GPT-2 and Pythia
# read suite_full2, while Llama and Qwen read their model roots. This supports
# mixed-source multi-model regeneration in one invocation.
#
# RUNS overrides the runs/ root entirely (default: repo runs/) — for pointing
# at a whole separate runs tree (e.g. an rsync'd copy), not for OUTPUT_SUBDIR
# (use the env var above for that).
# Set SKIP_FIGURES=1 to skip scatter/layerwise figure regeneration and PDF
# merging while still rebuilding the tables. Set INCLUDE_DIAGNOSTICS=1 to also
# compile the collapsed/convergence diagnostic report; it is off by default.
# Set STRICT_CAUSAL_PROVENANCE=1 only for an audit build that must suppress
# legacy causal results lacking later-added provenance metadata.
#
# By default every model is written to its own stable directory, even when
# MODELS contains several entries:
#   analysis/tables/latex_gpt2-small/summary_tables.pdf
#   analysis/tables/latex_gpt2-small_suite_full/summary_tables.pdf
#   analysis/tables/latex_pythia-70m-deduped_suite_full/summary_tables.pdf
# Set OUT=... explicitly to opt into one combined comparison book containing
# every model in MODELS (e.g. for a paper build).
#
# Writes (paths below assume the per-model default):
#   analysis/tables/latex_<model>[_<subdir>]/summary_matrix_*.tex
#   analysis/tables/latex_<model>[_<subdir>]/summary_methods_*.tex
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables.tex
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_pairwise.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_one_pole.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_median.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_pairwise_median.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_one_pole_median.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_detection.pdf
#   analysis/tables/latex_<model>[_<subdir>]/summary_tables_intervention.pdf
#     (concatenated raw-matrix tables; intervention excludes LMS metrics)
#   analysis/tables/latex_<model>[_<subdir>]/summary_diagnostics.pdf
#     (only when INCLUDE_DIAGNOSTICS=1)
#   analysis/tables/latex_across_models/summary_headline_tables.pdf
#   analysis/tables/latex_across_models/summary_headline_tables_median.pdf
#   analysis/tables/latex_across_models/summary_model_task_tables.pdf
#     (per-model Det./Int. percentage matrices, heatmaps, and auto-rank table)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

resolve_python() {
  local import_check="import pandas"
  # Prefer a fully featured interpreter for the normal report path.  Merely
  # installing pandas into WSL must not steal selection from the Windows
  # interpreter that has historically generated the figures.
  if [[ "${SKIP_FIGURES:-0}" != "1" ]]; then
    import_check="import pandas, matplotlib"
  fi
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
      if py -3 -c "${import_check}" >/dev/null 2>&1; then
        printf '%s\n' 'py -3'
        return 0
      fi
      continue
    fi
    if "${c}" -c "${import_check}" >/dev/null 2>&1; then
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
    if "${extra}" -c "${import_check}" >/dev/null 2>&1; then
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

# A WSL shell can discover a Windows Python through its inherited PATH.  WSL
# does not perform MSYS-style argument conversion for native Windows programs:
# passing /mnt/c/repo/... to python.exe makes pathlib write to C:\\mnt\\c\\repo
# instead of C:\\repo.  Keep shell paths in their native form (LaTeX and cd
# need that), but convert filesystem arguments at the Python boundary.
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

# The paper-model population lives in configs/report_model_sets.json.  An
# explicit MODELS=... remains a one-off override, but ordinary report refreshes
# use the completed paper set.  The >=90% companion is allowed to consider the
# wider ablation-candidate set; its separate source set is wired below.
if [[ -n "${MODELS:-}" ]]; then
  PAPER_MODELS="${MODELS}"
else
  PAPER_MODELS="$("${PYTHON[@]}" analysis/report_model_policy.py --set paper_models --format csv)"
fi
ABLATION_CANDIDATE_MODELS="$("${PYTHON[@]}" analysis/report_model_policy.py --set ablation_candidate_models --format csv)"
MODELS="${PAPER_MODELS}"

# ``essential`` is the normal paper-refresh path: one selected mean book per
# model plus the headline/taskwise reports and figures. ``full`` restores the
# historical exhaustive PDF matrix. FORCE=1 bypasses the result-summary cache.
REPORT_MODE="${REPORT_MODE:-essential}"
case "${REPORT_MODE}" in
  essential|full) ;;
  *) echo "error: REPORT_MODE must be essential or full (got ${REPORT_MODE})." >&2; exit 1 ;;
esac
FORCE="${FORCE:-0}"
if [[ "${REPORT_MODE}" == "essential" ]]; then
  DEFAULT_APPENDIX_ARTIFACTS=0
else
  DEFAULT_APPENDIX_ARTIFACTS=1
fi
echo "Report mode: ${REPORT_MODE} (FORCE=${FORCE})."
if [[ "${REPORT_MODE}" == "full" ]]; then
  echo "Full PDF matrix enabled by REPORT_MODE=full (complete90/median/full books are unrelated to appendix artifacts)."
fi

# The collapsed/non-convergence appendix is useful for debugging but is not a
# standard paper-table artifact. Keep it opt-in so ordinary table refreshes
# only generate the result-table PDFs.
INCLUDE_DIAGNOSTICS="${INCLUDE_DIAGNOSTICS:-0}"

# The paper headline has one canonical selected-method set. The expanded set
# belongs to appendix/diagnostic work and would reopen every local results.json
# for a second pass, so make it explicit rather than paying that cost by default.
INCLUDE_FULL_METHOD_SET="${INCLUDE_FULL_METHOD_SET:-0}"
METHOD_SETS=(selected)
if [[ "${INCLUDE_FULL_METHOD_SET}" == "1" ]]; then
  METHOD_SETS+=(full)
fi

# The detailed SAE appendix and the layer-selection appendix both use the paper
# set from configs/report_model_sets.json.
INCLUDE_APPENDIX_ARTIFACTS="${INCLUDE_APPENDIX_ARTIFACTS:-${DEFAULT_APPENDIX_ARTIFACTS}}"
# Layerwise books are appendix artifacts too.  Resolve their default only
# after INCLUDE_APPENDIX_ARTIFACTS is known so an explicit appendix request in
# `essential` mode enables them, while an explicit plot setting still wins.
if [[ -z "${INCLUDE_LAYER_PERFORMANCE_PLOTS+x}" ]]; then
  INCLUDE_LAYER_PERFORMANCE_PLOTS="${INCLUDE_APPENDIX_ARTIFACTS}"
fi
if [[ "${INCLUDE_APPENDIX_ARTIFACTS}" == "1" ]]; then
  echo "Including appendix figures/artifacts; report PDF scope remains ${REPORT_MODE}."
  echo "WARNING: SAE evidence and layer selection use all paper models; only the full-suite ablation inventory is GPT-2-small + Pythia-70M-deduped."
fi

if [[ "${SKIP_FIGURES:-0}" == "1" ]]; then
  INCLUDE_HEADLINE_SCATTER=0
  INCLUDE_LAYER_PERFORMANCE_PLOTS=0
elif ! "${PYTHON[@]}" -c "import matplotlib" >/dev/null 2>&1; then
  # Tables only need pandas.  A fresh WSL/base Python commonly has pandas but
  # not matplotlib; do not discard a long table regeneration for optional
  # figures in that case.
  echo "warning: matplotlib is unavailable in ${PYTHON_BIN}; generating tables without figures." >&2
  echo "         Install it in this environment to restore figures: ${PYTHON_BIN} -m pip install matplotlib" >&2
  INCLUDE_HEADLINE_SCATTER=0
  INCLUDE_LAYER_PERFORMANCE_PLOTS=0
fi

RUNS_ARGS=()
if [[ -n "${RUNS:-}" ]]; then
  RUNS_ARGS+=(--runs "$(python_path "${RUNS}")")
fi
CAUSAL_FALLBACK_ARGS=()
if [[ "${CAUSAL_VALIDATION_FALLBACK:-0}" == "1" ]]; then
  CAUSAL_FALLBACK_ARGS+=(--causal-validation-fallback)
fi

model_source_subdir() {
  local model="$1"
  if [[ -n "${OUTPUT_SUBDIR:-}" ]]; then
    printf '%s\n' "${OUTPUT_SUBDIR}"
    return 0
  fi
  # The subdir of every model comes from configs/report_model_sets.json (models
  # outside the policy read directly from runs/<model>).
  "${PYTHON[@]}" analysis/report_model_policy.py --subdir-of "${model}" | tr -d '\r'
}

IFS=',' read -r -a MODEL_ARR <<< "${MODELS}"

CLEAN_MODELS=()
for model in "${MODEL_ARR[@]}"; do
  model="${model//[[:space:]]/}"
  [[ -n "${model}" ]] && CLEAN_MODELS+=("${model}")
done
if [[ "${#CLEAN_MODELS[@]}" -eq 0 ]]; then
  echo "error: MODELS did not contain a model id." >&2
  exit 1
fi

for model in "${CLEAN_MODELS[@]}"; do
  model_subdir="$(model_source_subdir "${model}")"
  source_dir="${RUNS:-${ROOT}/runs}/${model}"
  if [[ -n "${model_subdir}" ]]; then
    source_dir="${source_dir}/${model_subdir}"
  fi
  if [[ ! -d "${source_dir}" ]]; then
    echo "error: no results directory for model '${model}' at ${source_dir}" >&2
    echo "       expected model id: qwen3.5-9b-base (not qwen3.5-9b)" >&2
    exit 1
  fi
done

# Default: one stable output directory per model. An explicit OUT is the
# deliberate escape hatch for callers that really want a combined book.
BOOK_MODELS=()
BOOK_OUTS=()
if [[ -n "${OUT:-}" ]]; then
  BOOK_MODELS+=("$(IFS=','; echo "${CLEAN_MODELS[*]}")")
  BOOK_OUTS+=("${OUT}")
else
  for model in "${CLEAN_MODELS[@]}"; do
    model_out="${ROOT}/analysis/tables/latex_${model}"
    model_subdir="$(model_source_subdir "${model}")"
    if [[ -n "${model_subdir}" ]]; then
      model_out="${model_out}_${model_subdir}"
    fi
    BOOK_MODELS+=("${model}")
    BOOK_OUTS+=("${model_out}")
  done
fi

find_engine() {
  # In Git Bash, an older TeX Live can appear first on PATH even when the
  # local MiKTeX installation is complete.  Prefer the known Windows install:
  # it is the one paired with the project’s Font Awesome 7 task-label setup.
  if command -v cygpath >/dev/null 2>&1 && [[ -n "${LOCALAPPDATA:-}" ]]; then
    local miktex_dir
    miktex_dir="$(cygpath -u "${LOCALAPPDATA}")/Programs/MiKTeX/miktex/bin/x64"
    # XeLaTeX consumes Font Awesome's installed Unicode fonts directly.  The
    # local pdfLaTeX tries to generate a bitmap FA glyph and fails on this
    # Windows installation.
    if [[ -x "${miktex_dir}/xelatex.exe" && -x "${miktex_dir}/kpsewhich.exe" ]]; then
      echo "${miktex_dir}/xelatex.exe"
      return 0
    fi
  fi
  if command -v latexmk >/dev/null 2>&1; then
    echo latexmk
    return 0
  fi
  if command -v pdflatex >/dev/null 2>&1; then
    echo pdflatex
    return 0
  fi
  if command -v pdflatex.exe >/dev/null 2>&1; then
    echo pdflatex.exe
    return 0
  fi
  if command -v wsl >/dev/null 2>&1 && wsl command -v pdflatex >/dev/null 2>&1; then
    echo wsl-pdflatex
    return 0
  fi
  return 1
}

if ! ENGINE="$(find_engine)"; then
  echo "error: no pdflatex/latexmk on PATH (tried native + wsl)." >&2
  echo "       install MiKTeX/TeX Live, or run this script inside WSL." >&2
  exit 1
fi

# Task labels use Font Awesome 7 icons (for example `person` and
# `hands-praying`), which are not available in the older Font Awesome 5
# package. Check before the expensive report/figure phases so a missing TeX
# package never wastes a full regeneration pass.
case "${ENGINE}" in
  wsl-pdflatex)
    FA7_AVAILABLE_CMD=(wsl kpsewhich fontawesome7.sty)
    ;;
  */pdflatex.exe|*/xelatex.exe)
    FA7_AVAILABLE_CMD=("$(dirname "${ENGINE}")/kpsewhich.exe" fontawesome7.sty)
    ;;
  *)
    FA7_AVAILABLE_CMD=(kpsewhich fontawesome7.sty)
    ;;
esac
if ! "${FA7_AVAILABLE_CMD[@]}" >/dev/null 2>&1; then
  echo "error: the selected TeX installation does not provide fontawesome7.sty." >&2
  echo "       These summary tables render Font Awesome 7 task icons and cannot fall back to blank labels." >&2
  echo "       Install it once with: tlmgr install --usermode fontawesome7" >&2
  echo "       If this TeX Live release is too old for that package, install the current fontawesome7 CTAN package into ~/texmf, then rerun this command." >&2
  exit 1
fi

if [[ "${INCLUDE_LAYER_PERFORMANCE_PLOTS:-1}" == "1" ]]; then
  echo "Regenerating layerwise-vs-all performance figures ..."
  for model in "${CLEAN_MODELS[@]}"; do
    model_subdir="$(model_source_subdir "${model}")"
    layer_plot_args=(--model "${model}" --out "$(python_path "${ROOT}/analysis/figures")")
    if [[ -n "${RUNS:-}" ]]; then
      layer_plot_args+=(--runs "$(python_path "${RUNS}")")
    fi
    if [[ -n "${model_subdir}" ]]; then
      layer_plot_args+=(--subdir "${model_subdir}")
    fi
    "${PYTHON[@]}" "$(python_path "${ROOT}/analysis/layer_performance_plots.py")" "${layer_plot_args[@]}"

    layer_suffix="${model}"
    if [[ -n "${model_subdir}" ]]; then
      layer_suffix="${layer_suffix}_${model_subdir}"
    fi
    layer_dir="${ROOT}/analysis/figures/layer_${layer_suffix}"
    # Compact result syncs intentionally omit the per-layer rows used by this
    # optional diagnostic.  layer_performance_plots exits successfully with a
    # helpful message in that case, so do not attempt to compile a nonexistent
    # figure directory and abort the headline tables.
    if [[ ! -f "${layer_dir}/layer_figures.tex" ]]; then
      echo "Skipping layerwise figure book for ${model}: no per-layer artifacts synced."
      continue
    fi
    echo "Compiling titled layerwise figure book for ${model} ..."
    pushd "${layer_dir}" >/dev/null
    set +e
    case "${ENGINE}" in
      latexmk)
        latexmk -g -pdf -interaction=nonstopmode -halt-on-error layer_figures.tex
        COMPILE_STATUS=$?
        [[ "${COMPILE_STATUS}" -eq 0 ]] && latexmk -c layer_figures.tex >/dev/null 2>&1 || true
        ;;
      wsl-pdflatex)
        wsl pdflatex -interaction=nonstopmode -halt-on-error layer_figures.tex \
          && wsl pdflatex -interaction=nonstopmode -halt-on-error layer_figures.tex
        COMPILE_STATUS=$?
        rm -f layer_figures.aux layer_figures.out
        ;;
      *)
        "${ENGINE}" -interaction=nonstopmode -halt-on-error layer_figures.tex \
          && "${ENGINE}" -interaction=nonstopmode -halt-on-error layer_figures.tex
        COMPILE_STATUS=$?
        rm -f layer_figures.aux layer_figures.out
        ;;
    esac
    set -e
    popd >/dev/null
    if [[ "${COMPILE_STATUS}" -ne 0 || ! -f "${layer_dir}/layer_figures.pdf" ]]; then
      echo "error: failed to compile titled layerwise figure book for ${model}" >&2
      exit 1
    fi
  done
fi

# Use a private marker for this invocation rather than summary_latex.py's own
# mtime. A full run can take long enough for that source file to be edited
# after phase 1; those edits must not make freshly regenerated snippets look
# stale at phase 3.
FRESHNESS_MARKER="$(mktemp "${TMPDIR:-/tmp}/summary-tables-freshness.XXXXXX")"
trap 'rm -f -- "${FRESHNESS_MARKER}"' EXIT

summary_cache_valid() {
  local model="$1" book_out="$2" source_dir="$3"
  local summary_csv="${book_out}/summary_merged_${model}.csv"
  [[ "${FORCE}" != "1" && "${INCLUDE_FULL_METHOD_SET}" != "1" && -f "${summary_csv}" ]] || return 1
  [[ ! analysis/summary_latex.py -nt "${summary_csv}" ]] || return 1
  [[ ! analysis/method_groups.py -nt "${summary_csv}" ]] || return 1
  ! find "${source_dir}" -name results.json -newer "${summary_csv}" -print -quit | grep -q .
}

# Phase 1: regenerate every per-model snippet + summary_merged CSV first. The
# scatter (phase 2) reads only these CSVs, so it must never depend on a PDF
# compile succeeding -- a PDF left open in a viewer used to abort the script
# before the scatter step, leaving figures stale against fresh tables.
for job_index in "${!BOOK_MODELS[@]}"; do
  book_models="${BOOK_MODELS[$job_index]}"
  book_out="${BOOK_OUTS[$job_index]}"
  mkdir -p "${book_out}"
  book_out="$(cd "${book_out}" && pwd)"
  echo "Regenerating summary snippets for ${book_models} ..."
  IFS=',' read -r -a BOOK_MODEL_ARR <<< "${book_models}"
  for model in "${BOOK_MODEL_ARR[@]}"; do
    MODEL_RUNS_ARGS=("${RUNS_ARGS[@]}")
    model_subdir="$(model_source_subdir "${model}")"
    if [[ -n "${model_subdir}" ]]; then
      MODEL_RUNS_ARGS+=(--subdir "${model_subdir}")
    fi
    source_dir="${RUNS:-${ROOT}/runs}/${model}"
    [[ -n "${model_subdir}" ]] && source_dir="${source_dir}/${model_subdir}"
    if summary_cache_valid "${model}" "${book_out}" "${source_dir}"; then
      echo "  cache hit: summary_merged_${model}.csv (set FORCE=1 to rebuild)"
    else
      echo "  ${PYTHON_BIN} analysis/summary_latex.py --model ${model} ${MODEL_RUNS_ARGS[*]:-}"
      "${PYTHON[@]}" analysis/summary_latex.py --model "${model}" --out "$(python_path "${book_out}")" "${CAUSAL_FALLBACK_ARGS[@]}" "${MODEL_RUNS_ARGS[@]}"
    fi
    if [[ "${INCLUDE_FULL_METHOD_SET}" == "1" ]]; then
      "${PYTHON[@]}" analysis/summary_latex.py --model "${model}" --method-set full --out "$(python_path "${book_out}")" "${CAUSAL_FALLBACK_ARGS[@]}" "${MODEL_RUNS_ARGS[@]}"
    fi
  done
done

# Across-model artifacts consume the freshly generated canonical CSVs above.
# Keeping this after phase 1 prevents a new run from mixing fresh headline
# tables with stale per-model detail matrices or heatmaps.
HEADLINE_COMPLETE90_AVAILABLE=1
if [[ "${SKIP_ACROSS_MODELS:-0}" != "1" ]]; then
  HEADLINE_OUT="${ROOT}/analysis/tables/latex_across_models"
  mkdir -p "${HEADLINE_OUT}"
  HEADLINE_OUT="$(cd "${HEADLINE_OUT}" && pwd)"
  echo "Regenerating across-model headline and per-task Det./Int. artifacts ..."
  HEADLINE_CSV_ARGS=()
  COMPLETE90_CSV_ARGS=()
  DETAIL_SOURCE_ARGS=()
  for job_index in "${!BOOK_MODELS[@]}"; do
    IFS=',' read -r -a BOOK_MODEL_ARR <<< "${BOOK_MODELS[$job_index]}"
    for model in "${BOOK_MODEL_ARR[@]}"; do
      model_subdir="$(model_source_subdir "${model}")"
      detail_csv="${BOOK_OUTS[$job_index]}/summary_merged_${model}.csv"
      HEADLINE_CSV_ARGS+=("--summary-csv" "${model}=$(python_path "${detail_csv}")")
      DETAIL_SOURCE_ARGS+=("--source" "${model}=$(python_path "${detail_csv}")")
    done
  done
  IFS=',' read -r -a CANDIDATE_MODEL_ARR <<< "${ABLATION_CANDIDATE_MODELS}"
  for model in "${CANDIDATE_MODEL_ARR[@]}"; do
    model="${model//[[:space:]]/}"
    [[ -z "${model}" ]] && continue
    model_subdir="$(model_source_subdir "${model}")"
    model_out="${ROOT}/analysis/tables/latex_${model}"
    if [[ -n "${model_subdir}" ]]; then
      model_out="${model_out}_${model_subdir}"
    fi
    candidate_csv="${model_out}/summary_merged_${model}.csv"
    if [[ -f "${candidate_csv}" ]]; then
      COMPLETE90_CSV_ARGS+=("--complete90-summary-csv" "${model}=$(python_path "${candidate_csv}")")
    else
      echo "warning: >=90% candidate omits ${model}: no ${candidate_csv}" >&2
    fi
  done
  headline_generation_log="$(mktemp)"
  if ! "${PYTHON[@]}" analysis/summary_latex.py --across-models \
    --report-mode "${REPORT_MODE}" --out "$(python_path "${HEADLINE_OUT}")" \
    "${HEADLINE_CSV_ARGS[@]}" "${COMPLETE90_CSV_ARGS[@]}" \
    | tee "${headline_generation_log}"; then
    rm -f -- "${headline_generation_log}"
    exit 1
  fi
  if grep -q "Skipping >= 90% headline: no eligible model" "${headline_generation_log}"; then
    HEADLINE_COMPLETE90_AVAILABLE=0
  fi
  rm -f -- "${headline_generation_log}"
  "${PYTHON[@]}" analysis/across_model_detail.py \
    --out "$(python_path "${HEADLINE_OUT}")" "${DETAIL_SOURCE_ARGS[@]}"
fi

# Phase 2: scatter figures from all per-model summary_merged CSVs. The output
# directory is shared across books, so this runs once after every CSV exists
# (running it per book overwrote GPT-2's points with Pythia's).
if [[ "${INCLUDE_HEADLINE_SCATTER:-1}" == "1" && "${SKIP_ACROSS_MODELS:-0}" != "1" ]]; then
  echo "Regenerating Detection-vs-Intervention scatter figures from all canonical summary CSVs ..."
  SCATTER_ARGS=("--runs" "$(python_path "${RUNS:-${ROOT}/runs}")")
  for job_index in "${!BOOK_MODELS[@]}"; do
    book_models="${BOOK_MODELS[$job_index]}"
    book_out="$(cd "${BOOK_OUTS[$job_index]}" && pwd)"
    IFS=',' read -r -a BOOK_MODEL_ARR <<< "${book_models}"
    for model in "${BOOK_MODEL_ARR[@]}"; do
      model_subdir="$(model_source_subdir "${model}")"
      scatter_source="${model}"
      if [[ -n "${model_subdir}" ]]; then
        scatter_source="${scatter_source}=${model_subdir}"
      fi
      SCATTER_ARGS+=("--source" "${scatter_source}")
      SCATTER_ARGS+=("--csv" "${model}=$(python_path "${book_out}/summary_merged_${model}.csv")")
    done
  done
  # The ordinary paper scatter follows PAPER_MODELS, not a second hard-coded
  # list.  model_source_subdir preserves suite_full2 for GPT-2/Pythia.
  IFS=',' read -r -a PAPER_MODEL_ARR <<< "${PAPER_MODELS}"
  for model in "${PAPER_MODEL_ARR[@]}"; do
    model="${model//[[:space:]]/}"
    figure_subdir="$(model_source_subdir "${model}")"
    model_out="${ROOT}/analysis/tables/latex_${model}"
    if [[ -n "${figure_subdir}" ]]; then
      model_out="${model_out}_${figure_subdir}"
    fi
    table_csv="${model_out}/summary_methods_${model}.csv"
    if [[ -f "${table_csv}" ]]; then
      SCATTER_ARGS+=("--table-csv" "${model}=$(python_path "${table_csv}")")
    else
      echo "warning: headline_scatter.pdf omits ${model}: no ${table_csv} (build its tables first)" >&2
    fi
  done
  "${PYTHON[@]}" analysis/headline_scatter.py --report-mode "${REPORT_MODE}" "${SCATTER_ARGS[@]}"
  # Paired headline figures use precisely the source set recorded by the
  # task-weighted 90%-completion table, never a separately guessed list.
  complete_manifest="${HEADLINE_OUT}/summary_across_models_manifest_complete90.json"
  if [[ "${REPORT_MODE}" == "full" && -f "${complete_manifest}" ]]; then
    "${PYTHON[@]}" analysis/headline_scatter.py "${SCATTER_ARGS[@]}" \
      --report-mode full --completion-manifest "$(python_path "${complete_manifest}")"
  elif [[ "${REPORT_MODE}" == "full" ]]; then
    echo "Skipping >=90%-complete scatter: no eligible model yet."
  fi
fi

# The across-model book must be written *after* Phase 2: its first figure is
# the pooled all-cell headline scatter.  The earlier across-model pass creates
# the numeric snippets; this lightweight rewrite only inserts freshly written
# figure paths and never rereads results.json.
if [[ "${SKIP_ACROSS_MODELS:-0}" != "1" ]]; then
  echo "Refreshing across-model headline books after scatter generation ..."
  "${PYTHON[@]}" analysis/summary_latex.py --out "$(python_path "${HEADLINE_OUT}")" \
    --write-headline-books --report-mode "${REPORT_MODE}"
fi

# Autorank consumes the same compact normalized CSVs as the other headline
# aggregates. It therefore never reopens a task results.json during a normal
# local paper refresh. Set SKIP_AUTORANK=1 only to skip this presentation step.
if [[ "${SKIP_ACROSS_MODELS:-0}" != "1" && "${SKIP_AUTORANK:-0}" != "1" ]]; then
  echo "Regenerating local headline Autorank tables and posterior maps ..."
  AUTORANK_CSV_ARGS=()
  for job_index in "${!BOOK_MODELS[@]}"; do
    IFS=',' read -r -a BOOK_MODEL_ARR <<< "${BOOK_MODELS[$job_index]}"
    for model in "${BOOK_MODEL_ARR[@]}"; do
      detail_csv="${BOOK_OUTS[$job_index]}/summary_merged_${model}.csv"
      AUTORANK_CSV_ARGS+=("--csv" "${model}=$(python_path "${detail_csv}")")
    done
  done
  "${PYTHON[@]}" analysis/method_statistics.py \
    --out "$(python_path "${ROOT}/analysis/tables/autorank")" "${AUTORANK_CSV_ARGS[@]}"
fi

# Paper-ready detailed-appendix artifacts. These are separate from headline
# aggregation on purpose: their scope is the two fully mature small-model
# studies, and appendix_evidence.py prints that restriction again itself.
if [[ "${INCLUDE_APPENDIX_ARTIFACTS}" == "1" ]]; then
  echo "Regenerating compact SAE appendix evidence (all paper models) ..."
  "${PYTHON[@]}" analysis/appendix_evidence.py --scan-kstar
  echo "Regenerating layer-selection appendix figures/tables (all paper models) ..."
  APPENDIX_LAYER_ARGS=()
  if [[ -n "${RUNS:-}" ]]; then
    APPENDIX_LAYER_ARGS+=(--runs "$(python_path "${RUNS}")")
  fi
  # Sources come from the paper-model policy, never from a second list here.
  APPENDIX_LAYER_SOURCE_ARGS=()
  while IFS= read -r layer_source; do
    layer_source="${layer_source%$'\r'}"
    [[ -n "${layer_source}" ]] && APPENDIX_LAYER_SOURCE_ARGS+=(--source "${layer_source}")
  done < <("${PYTHON[@]}" analysis/report_model_policy.py --set paper_models --format sources)
  "${PYTHON[@]}" analysis/layer_selection_appendix.py "${APPENDIX_LAYER_ARGS[@]}" \
    "${APPENDIX_LAYER_SOURCE_ARGS[@]}"
  "${PYTHON[@]}" analysis/appendix_ablation_inventory.py
fi

# Phase 3: write and compile the books.
for job_index in "${!BOOK_MODELS[@]}"; do
  book_models="${BOOK_MODELS[$job_index]}"
  book_out="$(cd "${BOOK_OUTS[$job_index]}" && pwd)"

  if [[ "${REPORT_MODE}" == "essential" ]]; then
    report_specs=(mean:combined mean:detection mean:intervention)
  else
    report_specs=(
      mean:combined mean:pairwise mean:one_pole mean:detection mean:intervention
      median:combined median:pairwise median:one_pole
    )
  fi
  if [[ "${INCLUDE_DIAGNOSTICS}" == "1" ]]; then
    report_specs+=(none:diagnostics)
  fi

  for method_set in "${METHOD_SETS[@]}"; do
  for report_spec in "${report_specs[@]}"; do
    IFS=: read -r aggregation view <<< "${report_spec}"
    if [[ "${view}" == diagnostics ]]; then
      [[ "${method_set}" == full ]] && continue
      base=summary_diagnostics
    elif [[ "${view}" == detection || "${view}" == intervention ]]; then
      base="summary_tables_${view}"
    elif [[ "${view}" == combined ]]; then
      base=summary_tables
    else
      base="summary_tables_${view}"
    fi
    if [[ "${method_set}" == full ]]; then
      base="${base}_full"
    fi
    if [[ "${aggregation}" == median ]]; then
      base="${base}_median"
    fi
    book_tex="${book_out}/${base}.tex"
    echo "Writing ${book_tex}"
    if [[ "${view}" == diagnostics ]]; then
      "${PYTHON[@]}" analysis/summary_latex.py --out "$(python_path "${book_out}")" \
        --write-diagnostics-book "${book_models}"
    elif [[ "${view}" == detection || "${view}" == intervention ]]; then
      "${PYTHON[@]}" analysis/summary_latex.py --out "$(python_path "${book_out}")" \
        --write-concatenated-book "${view}:${book_models}" --method-set "${method_set}"
    else
      "${PYTHON[@]}" analysis/summary_latex.py --out "$(python_path "${book_out}")" \
        --write-book "${book_models}" --book-view "${view}" \
        --aggregation "${aggregation}" --method-set "${method_set}" \
        --freshness-reference "$(python_path "${FRESHNESS_MARKER}")"
    fi

    pdf="${book_out}/${base}.pdf"
    log="${book_out}/${base}.log"
    echo "Compiling ${book_models} (${view}) with ${ENGINE} ..."
    pushd "${book_out}" >/dev/null
    set +e
    case "${ENGINE}" in
      latexmk)
        latexmk -g -pdf -interaction=nonstopmode -halt-on-error "${base}.tex"
        COMPILE_STATUS=$?
        if [[ "${COMPILE_STATUS}" -eq 0 ]]; then
          latexmk -c "${base}.tex" >/dev/null 2>&1 || true
        fi
        ;;
      wsl-pdflatex)
        wsl pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex" \
          && wsl pdflatex -interaction=nonstopmode -halt-on-error "${base}.tex"
        COMPILE_STATUS=$?
        rm -f "${base}.aux" "${base}.out"
        ;;
      *)
        "${ENGINE}" -interaction=nonstopmode -halt-on-error "${base}.tex" \
          && "${ENGINE}" -interaction=nonstopmode -halt-on-error "${base}.tex"
        COMPILE_STATUS=$?
        rm -f "${base}.aux" "${base}.out"
        ;;
    esac
    set -e
    popd >/dev/null

    if [[ "${COMPILE_STATUS}" -ne 0 ]]; then
      if [[ -f "${log}" ]] && grep -q "I can't write on file" "${log}"; then
        echo "error: pdflatex couldn't write ${pdf} — it's locked by another process" >&2
        echo "       (likely open in a PDF viewer or browser tab). Close it and re-run." >&2
      else
        echo "error: LaTeX compile failed (exit ${COMPILE_STATUS}). See ${log} for details." >&2
      fi
      exit 1
    fi

    if [[ ! -f "${pdf}" ]]; then
      echo "error: PDF was not produced: ${pdf}" >&2
      exit 1
    fi
    echo "Wrote ${pdf}"

    if [[ "${INCLUDE_LAYER_PERFORMANCE_PLOTS:-1}" == "1" && "${book_models}" != *,* \
      && "${view}" != detection && "${view}" != intervention ]]; then
      layer_subdir="$(model_source_subdir "${book_models}")"
      layer_suffix="${book_models}"
      if [[ -n "${layer_subdir}" ]]; then
        layer_suffix="${layer_suffix}_${layer_subdir}"
      fi
      layer_dir="${ROOT}/analysis/figures/layer_${layer_suffix}"
      if [[ ! -f "${layer_dir}/layer_figures.pdf" ]]; then
        echo "Skipping layerwise PDF merge for ${book_models}: figure book unavailable."
        continue
      fi
      with_figures="${book_out}/${base}_with_figures.tmp.pdf"
      "${PYTHON[@]}" analysis/merge_pdfs.py \
        --tables "$(python_path "${pdf}")" \
        --figures "$(python_path "${layer_dir}/layer_figures.pdf")" \
        --out "$(python_path "${with_figures}")"
      mv -f "${with_figures}" "${pdf}"
      echo "Updated ${pdf} with layerwise figures"
    fi
  done
  done
done

if [[ "${SKIP_ACROSS_MODELS:-0}" == "1" ]]; then
  echo "Skipping across-model headline tables and book (SKIP_ACROSS_MODELS=1)."
  exit 0
fi

if [[ "${REPORT_MODE}" == "essential" ]]; then
  HEADLINE_PDFS=(summary_headline_tables summary_model_task_tables)
else
  HEADLINE_PDFS=(summary_headline_tables summary_headline_tables_complete90 summary_headline_tables_median summary_headline_tables_median_complete90 summary_headline_tables_full summary_headline_tables_full_complete90 summary_headline_tables_full_median summary_headline_tables_full_median_complete90 summary_model_task_tables)
fi
for headline_base in "${HEADLINE_PDFS[@]}"; do
  headline_pdf="${HEADLINE_OUT}/${headline_base}.pdf"
  headline_log="${HEADLINE_OUT}/${headline_base}.log"
  if [[ "${HEADLINE_COMPLETE90_AVAILABLE}" == "0" && "${headline_base}" == *complete90* ]]; then
    echo "Skipping ${headline_base}: no requested model reaches 90% headline coverage."
    continue
  fi
  if [[ ! -f "${HEADLINE_OUT}/${headline_base}.tex" ]]; then
    echo "Skipping ${headline_base}: no eligible source set yet."
    continue
  fi
  echo "Compiling across-model headline tables (${headline_base}) with ${ENGINE} ..."
  pushd "${HEADLINE_OUT}" >/dev/null
  set +e
  case "${ENGINE}" in
  latexmk)
    latexmk -g -pdf -interaction=nonstopmode -halt-on-error "${headline_base}.tex"
    COMPILE_STATUS=$?
    if [[ "${COMPILE_STATUS}" -eq 0 ]]; then
      latexmk -c "${headline_base}.tex" >/dev/null 2>&1 || true
    fi
    ;;
  wsl-pdflatex)
    wsl pdflatex -interaction=nonstopmode -halt-on-error "${headline_base}.tex" \
      && wsl pdflatex -interaction=nonstopmode -halt-on-error "${headline_base}.tex"
    COMPILE_STATUS=$?
    rm -f "${headline_base}.aux" "${headline_base}.out"
    ;;
  *)
    "${ENGINE}" -interaction=nonstopmode -halt-on-error "${headline_base}.tex" \
      && "${ENGINE}" -interaction=nonstopmode -halt-on-error "${headline_base}.tex"
    COMPILE_STATUS=$?
    rm -f "${headline_base}.aux" "${headline_base}.out"
    ;;
  esac
  set -e
  popd >/dev/null

  if [[ "${COMPILE_STATUS}" -ne 0 || ! -f "${headline_pdf}" ]]; then
    echo "error: headline PDF compile failed. See ${headline_log}." >&2
    exit 1
  fi
  echo "Wrote ${headline_pdf}"
done
