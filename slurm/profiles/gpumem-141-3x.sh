# Three H200s for Gemma-27B's explicit base_model_device_map: auto setting.
# Other models retain the package's default automatic-loading policy.
SBATCH_PARTITION="gpu"
SBATCH_GRES="gpu:3"
SBATCH_CONSTRAINT="gpumem.141gib&nvidiagpu"
SBATCH_CPUS="12"
SBATCH_MEM="192G"
SBATCH_TIME="48:00:00"
