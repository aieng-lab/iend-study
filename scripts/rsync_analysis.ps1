# Pull only files needed for analysis/summaries (PowerShell entry point).
# Also copies artifacts/**/done.json (train LR / collapsed_encoder). Not training.json.
# Dry-run by default; pass -Go to copy.
#
#   .\scripts\rsync_analysis.ps1
#   .\scripts\rsync_analysis.ps1 -Go
#   .\scripts\rsync_analysis.ps1 -Go -Mode results
#   .\scripts\rsync_analysis.ps1 -Go -Model gpt2-small
#
# Also syncs runs/<AxbenchDumpDir>/ (default axbench_gemma2_2b_l20) — the
# AxBench pipeline's own flat dump dir, not runs/<model>/<task>/. Only when
# -Model/-Task are unset; skip with -NoAxbench.
#
# Opens one SSH master (ControlMaster, 10 min) so you type the password once.
# Lasting fix (no password): ssh-copy-id slurm
#
param(
    [switch]$Go,
    [ValidateSet("analysis", "results")]
    [string]$Mode = $(if ($env:MODE) { $env:MODE } else { "analysis" }),
    [string]$Remote = $(if ($env:REMOTE) { $env:REMOTE } else { "slurm" }),
    [string]$RemoteRepo = $env:REMOTE_REPO,
    [string]$Model = $(if ($env:MODEL) { $env:MODEL } else { "" }),
    [string]$Task = $(if ($env:TASK) { $env:TASK } else { "" }),
    [string]$AxbenchDumpDir = $(if ($env:AXBENCH_DUMP_DIR) { $env:AXBENCH_DUMP_DIR } else { "axbench_gemma2_2b_l20" }),
    [string]$AxbenchMaxSize = $(if ($env:AXBENCH_MAX_SIZE) { $env:AXBENCH_MAX_SIZE } else { "20m" }),
    [switch]$NoAxbench
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
. (Join-Path $PSScriptRoot "_cluster_env.ps1")
Import-ClusterEnv $Root
if (-not $RemoteRepo) { $RemoteRepo = "$env:CLUSTER_USER_DIR/gradiend-sae" }

function Find-Rsync {
    $cmd = Get-Command rsync -ErrorAction SilentlyContinue
    if ($cmd) { return @{ Exe = $cmd.Source; Prefix = @() } }
    $gitBash = @(
        "$env:ProgramFiles\Git\usr\bin\rsync.exe",
        "${env:ProgramFiles(x86)}\Git\usr\bin\rsync.exe",
        "$env:LOCALAPPDATA\Programs\Git\usr\bin\rsync.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($gitBash) { return @{ Exe = $gitBash; Prefix = @() } }
    $wsl = Get-Command wsl -ErrorAction SilentlyContinue
    if ($wsl) { return @{ Exe = "wsl"; Prefix = @("rsync") } }
    $bash = Get-Command bash -ErrorAction SilentlyContinue
    if ($bash) {
        Write-Host "No rsync on PATH; using Git Bash for scripts/rsync_analysis.sh"
        $args = @()
        if ($Go) { $args += "--go" }
        $env:MODE = $Mode
        $env:REMOTE = $Remote
        $env:REMOTE_REPO = $RemoteRepo
        $env:MODEL = $Model
        $env:TASK = $Task
        & bash (Join-Path $PSScriptRoot "rsync_analysis.sh") @args
        exit $LASTEXITCODE
    }
    throw "Need rsync, WSL, or Git Bash. Install Git for Windows + rsync, or use WSL."
}

$sshDir = Join-Path $HOME ".ssh"
if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Path $sshDir | Out-Null }
# %C is an OpenSSH token (not a PowerShell variable).
$mux = "$HOME/.ssh/cm-gradiend-sae-%C"
# The path routinely contains a space (C:\Users\First Last\...). ssh splits an
# unquoted -o value on whitespace and fails with "keyword controlpath extra
# arguments at end of line", silently dropping multiplexing so every rsync
# prompts for the password again.
$muxQuoted = '"' + $mux + '"'
$sshMux = @(
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=12h",
    "-o", "ControlPath=$muxQuoted"
)
$env:RSYNC_RSH = "ssh -o ControlMaster=auto -o ControlPersist=12h -o ControlPath=$muxQuoted"
Write-Host "Opening SSH to $Remote (one password; reused for 10 min)..."
& ssh @sshMux $Remote true
if ($LASTEXITCODE -ne 0) {
    Write-Host "note: SSH multiplexing failed; each rsync may ask for the password again."
    Write-Host "      lasting fix: ssh-copy-id $Remote"
    Remove-Item Env:RSYNC_RSH -ErrorAction SilentlyContinue
}

$rs = Find-Rsync

function Convert-RsyncPath {
    <#
      Render a Windows path in the form the chosen rsync understands.

      rsync parses "host:path", so a bare C:\Git\... is read as host "C" and
      fails with "The source and destination cannot both be remote". Git/msys
      rsync wants /c/Git/..., WSL rsync wants /mnt/c/Git/...
    #>
    param([string]$Path, [bool]$IsWsl)
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -match '^([A-Za-z]):[\\/](.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = $Matches[2] -replace '\\', '/'
        $prefix = if ($IsWsl) { "/mnt/$drive" } else { "/$drive" }
        return "$prefix/$rest"
    }
    return ($full -replace '\\', '/')
}

$isWsl = ($rs.Prefix.Count -gt 0)
$srcRuns = "${Remote}:${RemoteRepo}/runs/"
$dstRuns = Join-Path $Root "runs"
if ($Model) {
    $srcRuns = "${Remote}:${RemoteRepo}/runs/${Model}/"
    $dstRuns = Join-Path $Root "runs\$Model"
    if ($Task) {
        $srcRuns = "${Remote}:${RemoteRepo}/runs/${Model}/${Task}/"
        $dstRuns = Join-Path $Root "runs\$Model\$Task"
    }
}

# Rsync filters are first-match-wins. Directory exclusions must therefore
# precede --include=*/; otherwise rsync walks every cache/checkpoint file even
# though the final --exclude=* prevents it from being copied.
$filters = @(
    "--exclude=activation_cache*/",
    "--exclude=modified_models/",
    "--exclude=decoder/",
    "--exclude=decoder_raw/",
    "--exclude=decoder_by_layer/",
    "--exclude=checkpoints/",
    "--exclude=checkpoint-*/"
)

if ($env:IEND_WEIGHTS -ne "1") {
    # Seed/model trees contain checkpoints and their metadata. Enter them only
    # when the caller explicitly requests the small IEND weights.
    $filters += @(
        "--exclude=seeds/",
        "--exclude=model/"
    )
}

if (-not $Model -and -not $Task) {
    # AxBench is synced by its dedicated pass below; do not scan it twice.
    $filters += "--exclude=axbench*/"
}

$filters += @(
    "--include=*/",
    "--include=results.json",
    "--include=done.json",
    "--include=artifacts/sae/encode_method_rows.json",
    "--include=causal_seed_rows.csv",
    "--include=paired_causal_seed_rows.csv",
    "--include=causal_paired_summary.csv",
    "--include=trajectory_rows.csv",
    "--include=pole_rows.csv",
    "--include=pair_rows.csv",
    "--include=task_rows.csv",
    "--include=associations.csv",
    # Per-cell provenance and small theory-run tables; mirrors the bash script.
    # cell_report.csv makes a partially-covered run diagnosable from synced data
    # instead of only from the Slurm log.
    "--include=cell_report.csv",
    # screen_results.json is not matched by the 'results.json' rule above.
    "--include=screen_results.json",
    "--include=screen_summary.csv",
    "--include=decomposition_rows.csv",
    "--include=e0h_rows.csv",
    "--include=chi_a_rows.csv",
    "--include=arm_summary.csv",
    "--include=decoder_norm_trajectory.csv",
    "--include=decoder_scale_steps.csv",
    "--include=decoder_scale_summary.csv"
)
if ($env:IEND_WEIGHTS -eq "1") {
    # Mirrors the bash script's IEND_WEIGHTS opt-in: pull only the small IEND
    # encoder/decoder weights (~110 KB per ACTIEND checkpoint, not a base model)
    # so decoder-norm / reachability analysis can run locally. Modified models
    # and raw decoder trees are pruned above; rsync's --max-size is global, and
    # the largest whitelisted json/csv observed is ~0.2 MB, so 8m is ample
    # headroom while still blocking a 500 MB checkpoint.
    $maxSize = if ($env:IEND_WEIGHTS_MAX_SIZE) { $env:IEND_WEIGHTS_MAX_SIZE } else { "8m" }
    $filters += @(
        "--max-size=$maxSize",
        "--include=model.safetensors",
        "--include=gradiend_context.json",
        "--include=config.json"
    )
}
if ($Mode -eq "analysis") {
    $filters += @(
        "--include=REPORT.md",
        "--include=TABLES.txt",
        "--include=metrics.csv",
        "--include=decoder_artifacts.json",
        "--include=causal/summary.json",
        "--include=plots/***"
    )
}
$filters += @(
    "--exclude=*.jsonl",
    "--exclude=*.pt",
    "--exclude=*.bin",
    "--exclude=*.safetensors",
    "--exclude=*.ckpt",
    "--exclude=*"
)

$common = @("-avz", "--human-readable", "--prune-empty-dirs")
if (-not $Go) { $common = @("--dry-run") + $common }

Write-Host "MODE=$Mode  remote=$srcRuns"
$dstRunsRsync = Convert-RsyncPath -Path $dstRuns -IsWsl $isWsl
Write-Host "dest=$dstRuns"
if (-not $Go) { Write-Host "(dry-run; pass -Go to copy)" }

# A plain rsync replaces results.json WHOLESALE with the cluster's copy and would
# silently drop whatever only the local copy has (e.g. CAA/CGA computed on a
# workstation).  Snapshot every local results.json (hardlink, O(1)) before the
# pull and merge it back after -- the same guard scripts/rsync_analysis.sh has.
$mergeRoot = if (Test-Path $dstRuns) { $dstRuns } else { Join-Path $Root "runs" }
$mergePy = Join-Path $PSScriptRoot "smart_merge_pulled_results.py"
if ($Go) {
    & python $mergePy $mergeRoot   # heal snapshots an earlier pull left unmerged
    if ($LASTEXITCODE -ne 0) { Write-Host "warning: unmerged snapshots from an earlier pull remain (kept, not overwritten)" }
    & python $mergePy --snapshot $mergeRoot
    if ($LASTEXITCODE -ne 0) { Write-Error "ABORT: could not snapshot local results.json files; nothing was pulled."; exit 1 }
}

# --update: a smart-merged results.json is stamped 1 s newer than the remote copy it was merged
# with, so it is skipped until the cluster writes it again (see smart_merge_pulled_results.py).
& $rs.Exe @($rs.Prefix + $common + @("--update") + $filters + @($srcRuns, "$dstRunsRsync/"))
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Go) {
    & python $mergePy $mergeRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "############################################################"
        Write-Host "# SYNC INCOMPLETE: results.json files were overwritten and NOT"
        Write-Host "# merged with your local copies (kept as results.json.presync)."
        Write-Host "# Analysis refuses to run on them until repaired:"
        Write-Host "#   python scripts/smart_merge_pulled_results.py $mergeRoot"
        Write-Host "############################################################"
        exit 1
    }
}

# The canonical runs tree is merged at the results.json level, but its SAE
# encode artifact is one file and would be overwritten by a later source.
# Preserve this source's tiny k* selection record independently so the local
# audit can combine the cluster/a second cluster/workstation data without fetching raw activations.
if ($env:SAE_SELECTION_ARCHIVE -ne "0") {
    $sourceName = if ($env:SAE_SELECTION_SOURCE) { $env:SAE_SELECTION_SOURCE } else { "cluster" }
    $archiveRuns = Join-Path $Root "runs\_sae_selection_sources\$sourceName"
    if ($Model) {
        $archiveRuns = Join-Path $archiveRuns $Model
        if ($Task) { $archiveRuns = Join-Path $archiveRuns $Task }
    }
    $archiveRsync = Convert-RsyncPath -Path $archiveRuns -IsWsl $isWsl
    $saeArtifactFilters = @(
        "--include=*/",
        "--include=artifacts/sae/encode_method_rows.json",
        "--exclude=*"
    )
    Write-Host "Archiving compact SAE selection artifacts: $sourceName -> $archiveRuns"
    & $rs.Exe @($rs.Prefix + $common + $saeArtifactFilters + @($srcRuns, "$archiveRsync/"))
    if ($LASTEXITCODE -ne 0) {
        Write-Host "note: SAE selection artifact archive failed (non-fatal)"
    }
}

# Pull only the processed language-ID datasets needed for local audits/paper
# counts. Raw OPUS/MUSE caches remain on the cluster. Set SYNC_LANGUAGE_DATA=0 to skip.
if ($env:SYNC_LANGUAGE_DATA -ne "0" -and (-not $Task -or $Task -eq "language")) {
    Write-Host ""
    Write-Host "Also syncing processed language-ID CSVs + metadata (not raw corpora)"
    $srcLanguage = "${Remote}:${RemoteRepo}/data/synthetic/"
    $dstLanguage = Join-Path $Root "data\synthetic"
    $dstLanguageRsync = Convert-RsyncPath -Path $dstLanguage -IsWsl $isWsl
    $languageFilters = @(
        "--include=language.csv",
        "--include=language.meta.json",
        "--include=language_neutral.csv",
        "--include=language_neutral.meta.json",
        "--exclude=*"
    )
    & $rs.Exe @($rs.Prefix + $common + $languageFilters + @($srcLanguage, "$dstLanguageRsync/"))
    if ($LASTEXITCODE -ne 0) {
        Write-Host "note: processed language-ID data missing remotely (run the language task first)"
    }
}

if ($Mode -eq "analysis" -and -not $Task) {
    Write-Host ""
    Write-Host "Also syncing analysis/tables/ (cross-task CSVs; regenerable from results.json)"
    $srcTables = "${Remote}:${RemoteRepo}/analysis/tables/"
    $dstTables = Join-Path $Root "analysis\tables"
    $tableFilters = @("--update", "--include=*/", "--include=*.csv", "--include=*.txt", "--exclude=*")
    $dstTablesRsync = Convert-RsyncPath -Path $dstTables -IsWsl $isWsl
    & $rs.Exe @($rs.Prefix + $common + $tableFilters + @($srcTables, "$dstTablesRsync/"))
    if ($LASTEXITCODE -ne 0) {
        Write-Host "note: remote analysis/tables/ missing or empty (ok; regenerate locally)"
    }
}

# AxBench dump dir (separate flat tree: runs/<AxbenchDumpDir>/, not
# runs/<model>/<task>/). AxBench's own output filenames are unknown here, so
# pull broadly by extension instead of the study's fixed whitelist above.
if (-not $Model -and -not $Task -and -not $NoAxbench) {
    Write-Host ""
    Write-Host "Also syncing runs/$AxbenchDumpDir/ (AxBench dump dir; broad filter, no checkpoints, max $AxbenchMaxSize/file)"
    $srcAxbench = "${Remote}:${RemoteRepo}/runs/$AxbenchDumpDir/"
    $dstAxbench = Join-Path $Root "runs\$AxbenchDumpDir"
    $axbenchFilters = @(
        "--max-size=$AxbenchMaxSize",
        "--include=*/",
        "--include=*.json",
        "--include=*.jsonl",
        "--include=*.csv",
        "--include=*.parquet",
        "--include=*.pkl",
        "--include=*.txt",
        "--include=*.log",
        "--exclude=*.pt",
        "--exclude=*.bin",
        "--exclude=*.safetensors",
        "--exclude=*.ckpt",
        "--exclude=*"
    )
    $dstAxbenchRsync = Convert-RsyncPath -Path $dstAxbench -IsWsl $isWsl
    & $rs.Exe @($rs.Prefix + $common + $axbenchFilters + @($srcAxbench, "$dstAxbenchRsync/"))
    if ($LASTEXITCODE -ne 0) {
        Write-Host "note: remote runs/$AxbenchDumpDir/ missing (ok; nothing run there yet)"
    }
}

Write-Host ""
if (-not $Go) {
    Write-Host "(dry-run done; pass -Go to copy + regenerate family overview)"
} else {
    Write-Host "Regenerating overviews (family best-per-class + method-group pivots)..."
    $overviewArgs = @()
    if ($Model) { $overviewArgs += @("--model", $Model) }
    & python analysis/summarize_runs.py @overviewArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "note: summarize_runs failed (ok to run manually)"
    }
    Write-Host "Canonical tables: analysis\tables\family\family_overview_all.txt"
}
Write-Host "SSH master stays up ~10 min; rerun -Go without typing the password again."
Write-Host "To skip passwords forever: ssh-copy-id $Remote"
