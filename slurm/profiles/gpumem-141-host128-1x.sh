# NVIDIA H200 SXM plus the host-RAM headroom required by CGA's closed-form
# model-width direction.  CGA deliberately keeps its inert encoder on CPU;
# at 8B that matrix is ~26 GiB, in addition to transient backbone loading.
SBATCH_PARTITION=gpu
SBATCH_GRES=gpu:1
SBATCH_CONSTRAINT='gpumem.141gib&nvidiagpu'
SBATCH_CPUS=4
SBATCH_MEM=128G
SBATCH_TIME=24:00:00
