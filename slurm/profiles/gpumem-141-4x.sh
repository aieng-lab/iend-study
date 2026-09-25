# Four H200s for the 70B model's weight-gradient methods: with the package's
# device map GPU 0 carries no backbone weights (GRADIEND head only), so the
# backbone shards over GPUs 1-3 (~390 GiB) for ~141 GiB weights + ~137 GiB grads.
SBATCH_PARTITION="gpu"
SBATCH_GRES="gpu:4"
SBATCH_CONSTRAINT="gpumem.141gib&nvidiagpu"
SBATCH_CPUS="16"
SBATCH_MEM="256G"
SBATCH_TIME="48:00:00"
