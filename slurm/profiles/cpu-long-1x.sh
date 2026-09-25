# CPU-only long analysis / AxBench evaluation. the cluster: -p general, no GPU.
# Use for API-bound evaluators such as AxBench LMJudge on comparison runs,
# where wall time is dominated by provider latency/cache writes rather than CPU.
SBATCH_PARTITION=general
SBATCH_GRES=
SBATCH_CONSTRAINT=
SBATCH_CPUS=4
SBATCH_MEM=8G
SBATCH_TIME=24:00:00
