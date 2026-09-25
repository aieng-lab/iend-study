#!/usr/bin/env bash
# Single array-task body: map SLURM_ARRAY_TASK_ID → job_table row.
# Python lives in _array_task.py (no heredoc: CRLF from Windows rsync cannot
# leak print(... join(cmd)) into bash as "syntax error near unexpected token '('").
set -euo pipefail
cd /workspace

: "${SUBMIT_DIR:?SUBMIT_DIR must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"

exec python -u /workspace/slurm/_array_task.py
