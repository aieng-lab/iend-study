# Default single-GPU study profile: NVIDIA GPU with >= 24 GiB VRAM.
# the cluster: gpumem.24gib + generic --gres=gpu:1 (no specific GRES type).
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:1
SBATCH_CONSTRAINT='gpumem.24gib&nvidiagpu'
SBATCH_CPUS=4
SBATCH_MEM=64G
SBATCH_TIME=48:00:00
