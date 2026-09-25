# Push project files to the Slurm host (PowerShell entry point).
# Delegates to scripts/rsync_to_slurm.sh, which uses ONE ssh connection (one
# password); Windows OpenSSH cannot multiplex, so the old multi-rsync flow
# prompted repeatedly. Dry-run by default; pass -Go to copy.
#
#   .\scripts\rsync_to_slurm.ps1
#   .\scripts\rsync_to_slurm.ps1 -Go
param([switch]$Go)

$ErrorActionPreference = "Stop"
$sh = Join-Path $PSScriptRoot "rsync_to_slurm.sh"
$gitBash = @(
    "$env:ProgramFiles\Git\bin\bash.exe",
    "${env:ProgramFiles(x86)}\Git\bin\bash.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $gitBash) { throw "Git Bash not found (needed for tar/ssh); install Git for Windows." }

$args = @($sh)
if ($Go) { $args += "--go" }
& $gitBash @args
exit $LASTEXITCODE
