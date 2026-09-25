# Optional: pin to L40S for smoke / short jobs.
# Default launchers use gpumem-24-*; set SLURM_PROFILE=l40s-test-1x to pin.
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:l40s:1
SBATCH_CPUS=4
SBATCH_MEM=16G
SBATCH_TIME=2:00:00

