# Any NVIDIA GPU with >= 16 GiB VRAM (the cluster gpumem.16gib).
# nvidiagpu: skip Chimaira AMD RX 9060 XT (16 GiB, amdgpu).
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:1
SBATCH_CONSTRAINT='gpumem.16gib&nvidiagpu'
SBATCH_CPUS=4
SBATCH_MEM=64G
SBATCH_TIME=24:00:00
