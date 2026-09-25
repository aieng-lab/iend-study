# Dot-sourced by scripts/rsync_*.ps1. Reads the gitignored <repo>/cluster.env
# (KEY=VALUE lines) into process env vars that are not already set, and
# requires CLUSTER_USER_DIR (the only site-specific value).
function Import-ClusterEnv([string]$Root) {
    $f = Join-Path $Root "cluster.env"
    if (Test-Path $f) {
        foreach ($line in Get-Content $f) {
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$' -and $line -notmatch '^\s*#') {
                if (-not [Environment]::GetEnvironmentVariable($Matches[1])) {
                    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"', "'"))
                }
            }
        }
    }
    if (-not $env:CLUSTER_USER_DIR) {
        throw "CLUSTER_USER_DIR is not set: copy cluster.env.example to cluster.env and fill it in."
    }
}
