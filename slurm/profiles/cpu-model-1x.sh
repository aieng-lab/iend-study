# CPU-only model-backed diagnostics on the cluster's high-memory general nodes.
# Unlike cpu-1x/cpu-long-1x (small report/API jobs), theory extraction loads a
# backbone, autograd state, and sometimes an SAE dictionary. Keep the project's
# production host-memory default even though no GPU is requested.
SBATCH_PARTITION=general
SBATCH_GRES=
SBATCH_CONSTRAINT=
SBATCH_CPUS=4
SBATCH_MEM=64G
SBATCH_TIME=24:00:00
