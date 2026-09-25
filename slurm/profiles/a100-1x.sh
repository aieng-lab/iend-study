# Optional: pin to a single A100 SXM (prefer gpumem-24-1x unless you need this GRES).
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:a100-sxm:1
SBATCH_CPUS=4
SBATCH_MEM=64G
SBATCH_TIME=48:00:00
