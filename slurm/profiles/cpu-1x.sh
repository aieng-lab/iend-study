# CPU-only analysis / AxBench evaluation. the cluster: -p general, no GPU.
# Keep this deliberately small: general-partition nodes cannot satisfy the
# former 32G request, while these file- and API-bound jobs do not load a model.
SBATCH_PARTITION=general
SBATCH_GRES=
SBATCH_CONSTRAINT=
SBATCH_CPUS=4
SBATCH_MEM=8G
SBATCH_TIME=4:00:00
