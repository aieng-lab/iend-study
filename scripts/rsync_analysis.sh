#!/usr/bin/env bash
# Pull only the files needed to regenerate (or view) study analysis/summaries.
#
# Required to rebuild TABLES / REPORT / plots / analysis/summarize_runs.py:
#   runs/<model>/<task>/results.json
#
# Optional already-rendered views (small; skip with MODE=results):
#   REPORT.md  TABLES.txt  metrics.csv  plots/
#   analysis/tables/          (cross-task CSVs; also regenerable from results.json)
#   causal/summary.json       (redundant with results.json raw.causal)
#   done.json                 (under artifacts/; train LR / collapsed_encoder)
#
# Intentionally skipped (huge step logs; encoder metrics in results.json suffice):
#   artifacts/**/training.json
#
# Not needed (and usually huge) — this script never copies them by default:
#   artifacts/**/training.json  modified_models/  decoder*/
#   activation_cache*/  decoder_grid_cache.json
#   causal/samples_*.jsonl  causal/curve_samples_*.jsonl
#   checkpoints / *.pt / *.bin / *.safetensors
#
# Opt-in: IEND_WEIGHTS=1 additionally pulls the *small* IEND encoder/decoder
# weights (an ACTIEND model.safetensors is ~110 KB, not a base model), so
# decoder-norm / reachability analysis can run locally instead of costing a GPU
# job to read a number off a checkpoint. Base models cannot ride along:
# modified_models/ is pruned and IEND_WEIGHTS_MAX_SIZE (default 8m) caps the
# rest. Per-step training.json logs stay excluded.
#   IEND_WEIGHTS=1 bash scripts/rsync_analysis.sh --go
#
# Opt-in: IEND_TRAINING_LOGS=1 also pulls per-seed ``training.json`` histories.
# Use it with MODEL and TASK: these histories are intentionally not part of the
# ordinary result sync, but are required to diagnose whether a non-convergent
# IEND run is under-updating, unstable, or merely misses a final threshold.
#   MODEL=llama-3.1-8b TASK=language IEND_TRAINING_LOGS=1 \\
#     bash scripts/rsync_analysis.sh --go
#
# Opt-in: DECODER_GRIDS=1 also pulls the split-clean decoder causal grids
# (``decoder_grid_validation.json`` / ``decoder_grid_test.json`` and their
# ``_weaken_`` counterparts) that ``causal_study.py::_evaluate_decoder_split_clean``
# writes directly into each GRADIEND/ACTIEND/CGA artifact dir
# (e.g. ``artifacts/cga__onepole__F/``). These are NOT the excluded
# ``decoder_grid_cache.json``/``decoder*/`` dirs above -- they are the frozen
# per-LR validation-selection and test-confirmation grids themselves. Needed to
# check, without any GPU/model work, whether a validation-selection fix changes
# which LR is chosen and whether the OLD test grid already has a score at the
# newly-selected LR (recoverable from disk) or only ever scored one LR on test
# (needs a fresh forward pass at the corrected LR).
#   MODEL=pythia-70m-deduped OUTPUT_SUBDIR=suite_full2 DECODER_GRIDS=1 \\
#     bash scripts/rsync_analysis.sh --go
#
# Also syncs every runs/axbench*/ flat dump dir (including smoke runs) — the
# AxBench pipeline's own flat dump dirs, not runs/<model>/<task>/. Official
# AxBench writes its own filenames (unknown schema, unlike this study's
# results.json/done.json), so that block pulls broadly by extension
# (json/jsonl/csv/txt/log) instead of a fixed whitelist, still excluding
# checkpoints/binaries and caps each transferred file at AXBENCH_MAX_SIZE
# (default 20m) since it isn't a schema whitelist. Runs only when MODEL/TASK
# are unset; skip with SYNC_AXBENCH=0. Set AXBENCH_DUMP_DIR to sync only one
# named AxBench dump directory.
#
# Usage (from repo root):
#   bash scripts/rsync_analysis.sh                  # dry-run first
#   bash scripts/rsync_analysis.sh --go             # actually copy
#   MODEL=gpt2-small TASK=language bash scripts/rsync_analysis.sh --go
#   MODEL=gpt2-small OUTPUT_SUBDIR=suite_full TASK=induction \
#     bash scripts/rsync_analysis.sh --go
#   TASK=language bash scripts/rsync_analysis.sh --go  # also pulls processed language CSVs
#   bash scripts/rsync_analysis.sh --go  # also discovers axbench_smoke10, etc.
# PowerShell:
#   .\scripts\rsync_analysis.ps1
#   .\scripts\rsync_analysis.ps1 -Go
#
# Password: this script opens one SSH master (ControlMaster) and reuses it
# for every rsync. To never type a password, install a key:
#   ssh-copy-id "${REMOTE:-slurm}"
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/_cluster_env.sh"

: "${REMOTE:=slurm}"
: "${REMOTE_REPO:=${CLUSTER_USER_DIR}/gradiend-sae}"
: "${LOCAL_REPO:=${ROOT}}"
: "${MODE:=analysis}"   # analysis | results
: "${MODEL:=}"          # empty = all models under runs/
: "${TASK:=}"           # empty = all tasks
: "${OUTPUT_SUBDIR:=}"  # optional nested runs subdir: runs/<model>/<subdir>/<task>/
: "${SAE_SELECTION_ARCHIVE:=1}" # retain compact SAE choices separately per source host
: "${SAE_SELECTION_SOURCE:=cluster}"
: "${IEND_WEIGHTS:=0}"  # 1 = also pull the small IEND encoder/decoder weights
: "${IEND_TRAINING_LOGS:=0}"  # 1 = also pull per-step seed training histories
: "${DECODER_GRIDS:=0}"  # 1 = also pull the split-clean decoder validation/test grids

DRY_RUN=(--dry-run)
if [[ "${1:-}" == "--go" || "${1:-}" == "-y" ]]; then
  DRY_RUN=()
  shift || true
fi

if [[ "${MODE}" != "analysis" && "${MODE}" != "results" ]]; then
  echo "MODE must be analysis or results (got ${MODE})" >&2
  exit 1
fi

# Reuse one SSH TCP connection so rsync (and dry-run then --go) share a password.
# Start an explicit background master instead of relying on ``ssh host true`` to
# leave one behind.  The snapshot below can take long enough for an interrupted
# or stale master to otherwise turn the main transfer into a fresh login.
mkdir -p "${HOME}/.ssh"
MUX="${HOME}/.ssh/cm-gradiend-sae-%C"
SSH_MUX=(
  -o ControlMaster=auto
  -o ControlPersist=12h
  -o ControlPath="${MUX}"
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=3
)
export RSYNC_RSH="ssh ${SSH_MUX[*]}"

ensure_ssh_master() {
  if ssh "${SSH_MUX[@]}" -O check "${REMOTE}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Opening SSH to ${REMOTE} (one password; reused for 12 h after last use)..."
  ssh "${SSH_MUX[@]}" -MNf "${REMOTE}"
}

if ! ensure_ssh_master; then
  echo "note: SSH multiplexing failed; each rsync may ask for the password again." >&2
  echo "      lasting fix: ssh-copy-id ${REMOTE}" >&2
  unset RSYNC_RSH
fi

# Snapshot + merge guard.  A plain rsync replaces results.json WHOLESALE with
# the remote copy, silently dropping anything only the local copy has.  This is
# local work. It runs AFTER the SSH login below so the password is asked for
# immediately, not after minutes of silent snapshotting; the master connection
# is kept alive (ServerAliveInterval) while this runs.
PY_BIN="python"
command -v python >/dev/null 2>&1 || PY_BIN="python3"

# Under WSL a repo on /mnt/c is ~30x slower to walk than natively (runs/ has ~44k
# files: 80 s per walk in WSL vs 2 s with Windows python), and the snapshot/merge
# walks it several times. Use Windows' python.exe there (stdlib-only script).
smart_merge() {
  if [[ "${ROOT}" == /mnt/* ]] && command -v python.exe >/dev/null 2>&1 \
      && command -v wslpath >/dev/null 2>&1; then
    local a args=()
    for a in "$@"; do
      if [[ "$a" == /* ]]; then args+=("$(wslpath -w "$a")"); else args+=("$a"); fi
    done
    python.exe "$(wslpath -w "${ROOT}/scripts/smart_merge_pulled_results.py")" "${args[@]}"
  else
    "${PY_BIN}" scripts/smart_merge_pulled_results.py "$@"
  fi
}
MERGE_ROOT="${LOCAL_REPO}/runs"
if [[ -n "${MODEL}" ]]; then
  MERGE_ROOT="${LOCAL_REPO}/runs/${MODEL}"
  if [[ -n "${TASK}" ]]; then
    if [[ -n "${OUTPUT_SUBDIR}" ]]; then
      MERGE_ROOT+="/${OUTPUT_SUBDIR}"
    fi
    MERGE_ROOT+="/${TASK}"
  fi
fi
if ((${#DRY_RUN[@]} == 0)); then
  echo "[1/2] merging leftover results.json snapshots under ${MERGE_ROOT} (local, SSH stays open) ..."
  smart_merge "${MERGE_ROOT}" \
    || echo "warning: unmerged snapshots from an earlier pull remain (kept, not overwritten)" >&2
  echo "[2/2] snapshotting local results.json files before the pull ..."
  smart_merge --snapshot "${MERGE_ROOT}" \
    || { echo "ABORT: could not snapshot local results.json files; nothing was pulled." >&2; exit 1; }
  echo "local snapshot done; starting transfer."
fi

# Slurm job logs live outside the repository and are short-lived, so pull only
# recent ones and prune old local copies. This runs first: if a later analysis
# transfer fails or is interrupted, its diagnostic output is still available.
#
# SYNC_SLURM_LOGS=0             skip entirely
# SLURM_LOG_FETCH_DAYS=2        how far back to fetch from the cluster
# SLURM_LOG_RETENTION_DAYS=2    delete local logs older than this
if [[ -z "${MODEL}" && -z "${TASK}" && "${SYNC_SLURM_LOGS:-1}" != "0" ]]; then
  : "${REMOTE_LOG_DIR:=${CLUSTER_USER_DIR}/slurm-logs}"
  : "${SLURM_LOG_FETCH_DAYS:=2}"
  : "${SLURM_LOG_RETENTION_DAYS:=2}"
  LOCAL_LOG_DIR="${LOCAL_REPO}/slurm-logs"

  echo
  echo "Slurm logs: last ${SLURM_LOG_FETCH_DAYS}d from ${REMOTE_LOG_DIR}"

  RECENT_LOGS="$(ssh "${SSH_MUX[@]}" "${REMOTE}" \
    "find '${REMOTE_LOG_DIR}' -maxdepth 1 -type f -name 'slurm-*' -mtime -${SLURM_LOG_FETCH_DAYS} -printf '%P\n' 2>/dev/null" \
    || true)"

  if [[ -z "${RECENT_LOGS}" ]]; then
    echo "  (none in the last ${SLURM_LOG_FETCH_DAYS}d, or log dir missing)"
  else
    mkdir -p "${LOCAL_LOG_DIR}"
    printf '%s\n' "${RECENT_LOGS}" | rsync -az --human-readable \
      "${DRY_RUN[@]}" \
      --files-from=- \
      --max-size="${SLURM_LOG_MAX_SIZE:-32m}" \
      "${REMOTE}:${REMOTE_LOG_DIR}/" "${LOCAL_LOG_DIR}/" \
      || echo "note: slurm log sync failed (non-fatal)"
    echo "  -> ${LOCAL_LOG_DIR}"
  fi

  if [[ "${LOCAL_LOG_DIR}" == */slurm-logs && -d "${LOCAL_LOG_DIR}" ]]; then
    if ((${#DRY_RUN[@]})); then
      PRUNE_N="$(find "${LOCAL_LOG_DIR}" -maxdepth 1 -type f -name 'slurm-*' \
        -mtime "+${SLURM_LOG_RETENTION_DAYS}" | wc -l | tr -d ' ')"
      echo "  (dry-run) would delete ${PRUNE_N} local log(s) older than ${SLURM_LOG_RETENTION_DAYS}d"
    else
      PRUNE_N="$(find "${LOCAL_LOG_DIR}" -maxdepth 1 -type f -name 'slurm-*' \
        -mtime "+${SLURM_LOG_RETENTION_DAYS}" -print -delete | wc -l | tr -d ' ')"
      echo "  pruned ${PRUNE_N} local log(s) older than ${SLURM_LOG_RETENTION_DAYS}d"
    fi
  fi
fi

# Explicit, out-of-band fetch for specific job IDs regardless of age --
# e.g. a job whose diagnostic log already aged out of the rolling
# SLURM_LOG_RETENTION_DAYS window above (or that finished >2 days ago and
# was never pulled). Comma-separated job ids or globs, matched as
# 'slurm-<entry>*' against the remote log dir (so both a bare job id like
# '4182967' and an array task like '4182967_0' work; '4182967_*' fetches
# every array index of that job). Writes to slurm-logs/archive/, NOT
# slurm-logs/ directly, so the -maxdepth 1 retention prune above (and any
# future run of it) never touches these -- they are not subject to
# SLURM_LOG_RETENTION_DAYS and persist until manually removed.
#
#   SLURM_LOG_JOBS=4182967 bash scripts/rsync_analysis.sh --go
#   SLURM_LOG_JOBS=4182967_0,4182967_1 bash scripts/rsync_analysis.sh --go
#   SLURM_LOG_JOBS='4182967_*,4155756_*' bash scripts/rsync_analysis.sh --go
if [[ -n "${SLURM_LOG_JOBS:-}" ]]; then
  : "${REMOTE_LOG_DIR:=${CLUSTER_USER_DIR}/slurm-logs}"
  ARCHIVE_LOG_DIR="${LOCAL_REPO}/slurm-logs/archive"
  mkdir -p "${ARCHIVE_LOG_DIR}"

  echo
  echo "Slurm logs: explicit fetch for job id(s)/pattern(s): ${SLURM_LOG_JOBS}"

  ARCHIVE_FILTERS=()
  IFS=',' read -ra _SLURM_LOG_JOB_ENTRIES <<< "${SLURM_LOG_JOBS}"
  for _entry in "${_SLURM_LOG_JOB_ENTRIES[@]}"; do
    _entry="$(echo "${_entry}" | xargs)"
    [[ -n "${_entry}" ]] || continue
    ARCHIVE_FILTERS+=(--include="slurm-${_entry}*")
  done

  if ((${#ARCHIVE_FILTERS[@]} == 0)); then
    echo "  (SLURM_LOG_JOBS set but empty after parsing; nothing to fetch)"
  else
    rsync -az --human-readable \
      "${DRY_RUN[@]}" \
      "${ARCHIVE_FILTERS[@]}" \
      --exclude='*' \
      --max-size="${SLURM_LOG_MAX_SIZE:-32m}" \
      "${REMOTE}:${REMOTE_LOG_DIR}/" "${ARCHIVE_LOG_DIR}/" \
      || echo "note: explicit slurm log fetch failed (non-fatal)"
    echo "  -> ${ARCHIVE_LOG_DIR} (exempt from SLURM_LOG_RETENTION_DAYS pruning)"
  fi
fi

SRC_RUNS="${REMOTE}:${REMOTE_REPO}/runs/"
DST_RUNS="${LOCAL_REPO}/runs/"
if [[ -n "${MODEL}" ]]; then
  SRC_RUNS="${REMOTE}:${REMOTE_REPO}/runs/${MODEL}/"
  DST_RUNS="${LOCAL_REPO}/runs/${MODEL}/"
  if [[ -n "${TASK}" ]]; then
    if [[ -n "${OUTPUT_SUBDIR}" ]]; then
      SRC_RUNS="${REMOTE}:${REMOTE_REPO}/runs/${MODEL}/${OUTPUT_SUBDIR}/${TASK}/"
      DST_RUNS="${LOCAL_REPO}/runs/${MODEL}/${OUTPUT_SUBDIR}/${TASK}/"
    else
      SRC_RUNS="${REMOTE}:${REMOTE_REPO}/runs/${MODEL}/${TASK}/"
      DST_RUNS="${LOCAL_REPO}/runs/${MODEL}/${TASK}/"
    fi
  fi
fi

# rsync filter order is first-match-wins.  In particular, exclusions for whole
# directory trees MUST precede --include='*/': putting them after that rule
# makes rsync enter every cache/checkpoint directory and stat all of its files
# even though the final --exclude='*' prevents those files from being copied.
# On the absorption runs alone that meant walking tens of thousands of .pt
# cache files on every sync.
FILTERS=(
  --exclude='activation_cache*/'
  --exclude='modified_models/'
  --exclude='decoder/'
  --exclude='decoder_raw/'
  --exclude='decoder_by_layer/'
  --exclude='checkpoints/'
  --exclude='checkpoint-*/'
)

# Seed/model directories contain only train checkpoints and their metadata.
# Enter them only for the explicit small-IEND-weight sync below.
if [[ "${IEND_TRAINING_LOGS}" != "1" ]]; then
  FILTERS+=(--exclude='seeds/')
fi
if [[ "${IEND_WEIGHTS}" != "1" ]]; then
  FILTERS+=(--exclude='model/')
fi

# AxBench has a separate, schema-appropriate pass below.  Excluding it here
# avoids walking the same (potentially very large) dump trees twice.
if [[ -z "${MODEL}" && -z "${TASK}" ]]; then
  FILTERS+=(--exclude='axbench*/')
fi

# Keep all remaining directories so rsync can reach the small named outputs.
FILTERS+=(
  --include='*/'
  --include='results.json'
  --include='done.json'
  --include='artifacts/sae/encode_method_rows.json'
  # Small theory-analysis artifacts that are not embedded in results.json.
  --include='causal_seed_rows.csv'
  --include='paired_causal_seed_rows.csv'
  --include='causal_paired_summary.csv'
  --include='trajectory_rows.csv'
  --include='pole_rows.csv'
  --include='pair_rows.csv'
  --include='task_rows.csv'
  --include='associations.csv'
  # Per-cell provenance and small theory-run tables. cell_report.csv in
  # particular exists so a run that silently covers only part of its cells is
  # diagnosable from synced data rather than only from the Slurm log — omitting
  # it here would reintroduce exactly the blind spot it was added to close.
  --include='cell_report.csv'
  # The preflight screen writes screen_results.json / screen_summary.csv, which
  # the 'results.json' rule above does NOT match.
  --include='screen_results.json'
  --include='screen_summary.csv'
  --include='decomposition_rows.csv'
  --include='e0h_rows.csv'
  --include='chi_a_rows.csv'
  --include='chi_a_transfer_by_source_rows.csv'
  # C1 one-dimensional recoverability diagnostics (runs/theory/iend_spectral/...).
  --include='spectral_rows.csv'
  --include='signal_space_contrasts.csv'
  --include='signal_space_summaries.csv'
  # C3 intervention/behavior-gradient alignment (runs/theory/iend_c3_alignment/...).
  --include='c3_alignment_rows.csv'
  # CAGA vs CAA vs CGA causal (runs/theory/caga_causal/...).
  --include='caga_causal_rows.csv'
  --include='learned_alignment_rows.csv'
  --include='arm_summary.csv'
  --include='decoder_norm_trajectory.csv'
  --include='decoder_scale_steps.csv'
  --include='decoder_scale_summary.csv'
  # Study-local gradient moderator diagnostics (runs/theory/...).
  --include='gradient_moderators.json'
  --include='gradient_moderators.csv'
  --include='gradient_samples.csv'
)
if [[ "${IEND_WEIGHTS}" == "1" ]]; then
  # IEND encoder/decoder weights only. These are genuinely small -- an ACTIEND
  # checkpoint's model.safetensors is ~110 KB (a 9216x1 decoder plus a 1x9216
  # encoder), not a base model -- so pulling them lets decoder-norm and
  # reachability analysis (IEND_THEORY_PLAN 2.7) run locally instead of costing
  # a GPU job just to read a number off a checkpoint.
  #
  # Base-model weights cannot ride along: modified_models/ and raw decoder
  # trees are pruned above, and IEND_WEIGHTS_MAX_SIZE caps whatever remains.
  # The multi-MB per-step training.json logs stay excluded (they are not
  # whitelisted).
  # rsync's --max-size is global, so it also applies to the whitelisted
  # json/csv above. The largest of those observed is ~0.2 MB, so 8m leaves ample
  # headroom while still blocking a base-model checkpoint (500 MB+) by two
  # orders of magnitude.
  : "${IEND_WEIGHTS_MAX_SIZE:=8m}"
  FILTERS+=(
    --max-size="${IEND_WEIGHTS_MAX_SIZE}"
    --include='model.safetensors'
    --include='gradiend_context.json'
    --include='config.json'
  )
fi
if [[ "${IEND_TRAINING_LOGS}" == "1" ]]; then
  FILTERS+=(--include='training.json')
fi
if [[ "${DECODER_GRIDS}" == "1" ]]; then
  FILTERS+=(
    --include='decoder_grid_validation.json'
    --include='decoder_grid_test.json'
    --include='decoder_grid_weaken_validation.json'
    --include='decoder_grid_weaken_test.json'
  )
fi
if [[ "${MODE}" == "analysis" ]]; then
  FILTERS+=(
    --include='REPORT.md'
    --include='TABLES.txt'
    --include='metrics.csv'
    --include='decoder_artifacts.json'
    --include='causal/summary.json'
    --include='plots/***'
  )
fi
FILTERS+=(
  --exclude='*.jsonl'
  --exclude='*.pt'
  --exclude='*.bin'
  --exclude='*.safetensors'
  --exclude='*.ckpt'
  --exclude='*'
)

echo "MODE=${MODE}  remote=${SRC_RUNS}"
echo "dest=${DST_RUNS}"
if ((${#DRY_RUN[@]})); then
  echo "(dry-run; pass --go to copy)"
fi

# Check the master directly before the critical pull.  The local snapshot ran
# before SSH was opened, so this should normally reuse the same one connection.
if [[ -n "${RSYNC_RSH:-}" ]] && ! ensure_ssh_master; then
  echo "note: SSH master was unavailable before the main transfer; falling back to ordinary SSH." >&2
  unset RSYNC_RSH
fi

# --update: a locally smart-merged results.json is stamped 1 s newer than the remote copy it
# was merged with (smart_merge_pulled_results.py), so it is skipped until the cluster writes
# it again instead of being re-downloaded and re-merged on every pull.
rsync -avz --update --human-readable --prune-empty-dirs \
  "${DRY_RUN[@]}" \
  "${FILTERS[@]}" \
  "${SRC_RUNS}" "${DST_RUNS}" \
  "$@"

# ``results.json`` is smart-merged below, but a plain rsync cannot merge the
# separate SAE encode artifacts of two machines. Keep a tiny, source-namespaced
# copy for the k* audit so a later pull can never erase another host's choice.
if [[ "${SAE_SELECTION_ARCHIVE}" == "1" ]]; then
  SAE_ARCHIVE_DEST="${LOCAL_REPO}/runs/_sae_selection_sources/${SAE_SELECTION_SOURCE}/"
  # Mirror the selected source's position below runs/; otherwise a scoped
  # MODEL= pull would lose its model/subdir path and become undiscoverable by
  # the canonical-paper-source matcher.
  if [[ -n "${MODEL}" ]]; then
    SAE_ARCHIVE_DEST+="${MODEL}/"
    if [[ -n "${TASK}" ]]; then
      [[ -n "${OUTPUT_SUBDIR}" ]] && SAE_ARCHIVE_DEST+="${OUTPUT_SUBDIR}/"
      SAE_ARCHIVE_DEST+="${TASK}/"
    fi
  fi
  echo "Archiving compact SAE selection artifacts: ${SAE_SELECTION_SOURCE} -> ${SAE_ARCHIVE_DEST}"
  rsync -avz --human-readable --prune-empty-dirs \
    "${DRY_RUN[@]}" \
    --include='*/' \
    --include='artifacts/sae/encode_method_rows.json' \
    --exclude='*' \
    "${SRC_RUNS}" "${SAE_ARCHIVE_DEST}" \
    || echo "note: SAE selection artifact archive failed (non-fatal)"
fi

if ((${#DRY_RUN[@]} == 0)); then
  if ! smart_merge "${MERGE_ROOT}"; then
    {
      echo
      echo "############################################################"
      echo "# SYNC INCOMPLETE: results.json files were overwritten and NOT"
      echo "# merged with your local copies (kept as results.json.presync)."
      echo "# Analysis refuses to run on them until repaired:"
      echo "#   python scripts/smart_merge_pulled_results.py ${MERGE_ROOT}"
      echo "############################################################"
    } >&2
    exit 1
  fi
fi

# The language-ID builder runs on the cluster because it downloads and filters OPUS
# corpora. Its final, study-ready artifacts are small enough to keep locally
# for dataset audits and paper counts; the raw OPUS/MUSE caches stay remote.
if [[ "${SYNC_LANGUAGE_DATA:-1}" != "0" && ( -z "${TASK}" || "${TASK}" == "language" ) ]]; then
  echo
  echo "Also syncing processed language-ID CSVs + metadata (not raw corpora)"
  mkdir -p "${LOCAL_REPO}/data/synthetic"
  rsync -avz --human-readable \
    "${DRY_RUN[@]}" \
    --include='language.csv' \
    --include='language.meta.json' \
    --include='language_neutral.csv' \
    --include='language_neutral.meta.json' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_REPO}/data/synthetic/" \
    "${LOCAL_REPO}/data/synthetic/" \
    || echo "note: processed language-ID data missing remotely (run the language task first)"
fi

# Sync root error log (small, useful for diagnosing missing/partial rows).
if [[ -z "${MODEL}" && -z "${TASK}" ]]; then
  rsync -avz --human-readable \
    "${DRY_RUN[@]}" \
    "${REMOTE}:${REMOTE_REPO}/error_report.jsonl" \
    "${LOCAL_REPO}/error_report.jsonl" \
    || echo "note: remote error_report.jsonl missing (ok)"
fi

# Compact merged output from the split ACTIEND LR/activation-scale launcher.
# Checkpoints remain under runs/<model>/<task>/ and are excluded by the main
# filter; only the comparison and per-case JSON shards are copied here.
if [[ -z "${MODEL}" && -z "${TASK}" && "${SYNC_ACTIEND_LR_ABLATION:-1}" != "0" ]]; then
  # v3 is the first sweep generated after the explicit-raw and one-pole
  # protocol fixes.  Keeping v2 as the default silently leaves the current
  # comparison/causal report on the cluster even though the per-cell
  # done.json files are synced by the main filter above.
  ACTIEND_LR_REPORT_DIR="${ACTIEND_LR_REPORT_DIR:-_ablation_actiend_lr_scale_v3}"
  echo
  echo "Also syncing runs/${ACTIEND_LR_REPORT_DIR}/ (compact LR/scale ablation reports)"
  rsync -avz --human-readable --prune-empty-dirs \
    "${DRY_RUN[@]}" \
    --include='*/' \
    --include='comparison.json' \
    --include='comparison.txt' \
    --include='shards/*.json' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_REPO}/runs/${ACTIEND_LR_REPORT_DIR}/" \
    "${LOCAL_REPO}/runs/${ACTIEND_LR_REPORT_DIR}/" \
    || echo "note: remote runs/${ACTIEND_LR_REPORT_DIR}/ missing (ok; nothing run there yet)"
fi

# AxBench dump dirs are separate flat trees (runs/axbench*/, not
# runs/<model>/<task>/). Discover them so normal syncs include smoke outputs
# too. Set AXBENCH_DUMP_DIR explicitly to limit this to one named directory.
# We don't know AxBench's own output filenames, so pull broadly by extension
# (including evaluator and inference parquet tables) rather than the study's
# fixed whitelist above — but cap per-file size
# (--max-size) since that filter isn't a schema whitelist and a large
# activation/generation dump could otherwise slip through under an included
# extension.
: "${AXBENCH_MAX_SIZE:=20m}"
if [[ -z "${MODEL}" && -z "${TASK}" && "${SYNC_AXBENCH:-1}" != "0" ]]; then
  if [[ -n "${AXBENCH_DUMP_DIR:-}" ]]; then
    AXBENCH_DUMP_DIRS=("${AXBENCH_DUMP_DIR}")
  else
    mapfile -t AXBENCH_DUMP_DIRS < <(
      ssh "${SSH_MUX[@]}" "${REMOTE}" \
        "find '${REMOTE_REPO}/runs' -mindepth 1 -maxdepth 1 -type d -name 'axbench*' -printf '%f\\n'" \
        2>/dev/null || true
    )
  fi
  for axbench_dump_dir in "${AXBENCH_DUMP_DIRS[@]}"; do
    [[ -n "${axbench_dump_dir}" ]] || continue
    echo
    echo "Also syncing runs/${axbench_dump_dir}/ (AxBench dump dir; broad filter, no checkpoints, max ${AXBENCH_MAX_SIZE}/file)"
    rsync -avz --human-readable --prune-empty-dirs \
      --max-size="${AXBENCH_MAX_SIZE}" \
      "${DRY_RUN[@]}" \
      --include='*/' \
      --include='*.json' \
      --include='*.jsonl' \
      --include='*.csv' \
      --include='*.parquet' \
      --include='*.pkl' \
      --include='*.txt' \
      --include='*.log' \
      --exclude='*.pt' \
      --exclude='*.bin' \
      --exclude='*.safetensors' \
      --exclude='*.ckpt' \
      --exclude='*' \
      "${REMOTE}:${REMOTE_REPO}/runs/${axbench_dump_dir}/" \
      "${LOCAL_REPO}/runs/${axbench_dump_dir}/" \
      || echo "note: remote runs/${axbench_dump_dir}/ missing (ok; nothing run there yet)"
  done
fi

if [[ "${MODE}" == "analysis" && -z "${TASK}" ]]; then
  echo
  echo "Also syncing analysis/tables/ (cross-task CSVs; regenerable from results.json)"
  rsync -avz --human-readable --prune-empty-dirs \
    "${DRY_RUN[@]}" \
    --include='*/' \
    --update \
    --include='*.csv' \
    --include='*.txt' \
    --include='*.tex' \
    --include='*.pdf' \
    --include='appendix_evidence/evidence_report.md' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_REPO}/analysis/tables/" \
    "${LOCAL_REPO}/analysis/tables/" \
    || echo "note: remote analysis/tables/ missing or empty (ok; regenerate locally)"

  echo "Syncing paper-facing headline and layer figures"
  rsync -avz --human-readable --prune-empty-dirs \
    "${DRY_RUN[@]}" \
    --update \
    --include='headline_*.pdf' \
    --include='layer_*.pdf' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_REPO}/analysis/figures/" \
    "${LOCAL_REPO}/analysis/figures/" \
    || echo "note: remote headline/layer figures missing or empty (ok; regenerate on the cluster)"
fi

echo
if ((${#DRY_RUN[@]})); then
  echo "(dry-run done; pass --go to copy + regenerate family overview)"
else
  echo "Regenerating overviews (family best-per-class + method-group pivots)..."
  python analysis/summarize_runs.py ${MODEL:+--model "${MODEL}"} || \
    echo "note: summarize_runs failed (ok to run manually)"
fi
echo "SSH master stays up 12 h after last use; rerun --go without typing the password again."
echo "To skip passwords forever: ssh-copy-id ${REMOTE}"
echo "Canonical tables: analysis/tables/family/family_overview_all.txt"
