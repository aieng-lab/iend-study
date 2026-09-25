# Any NVIDIA GPU with >= 11 GiB VRAM (the cluster gpumem.11gib).
# nvidiagpu: skip Chimaira AMD RX 9060 XT (16 GiB) which also matches 11/16.
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:1
SBATCH_CONSTRAINT='gpumem.11gib&nvidiagpu'
SBATCH_CPUS=4
SBATCH_MEM=64G
SBATCH_TIME=24:00:00
