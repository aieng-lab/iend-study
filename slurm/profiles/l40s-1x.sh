# Optional: pin to a single L40S (prefer gpumem-24-1x unless you need this GRES).
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:l40s:1
SBATCH_CPUS=4
SBATCH_MEM=64G
SBATCH_TIME=48:00:00
