#!/bin/bash
#SBATCH --partition=all
#SBATCH --job-name=ps_cpu
#SBATCH --output=runs/prestage/logs/%x_%j.out
#SBATCH --time=2-00:00:00
# CPU-only job (xeon6530-h100nvl): runs the commands of a file (one per line), $SLURM_CPUS_PER_TASK at a time, 1 thread each.
# usage (from the experiment directory): sbatch --cpus-per-task=N --mem=<..>G -J <name> prestage/cpu_cmds.sbatch <cmdfile>
set -u
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/experiments/gpn_w16u_al
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=${PY:-python}
echo "start $(date) host $(hostname) cpus ${SLURM_CPUS_PER_TASK} cmds $1 ($(wc -l < "$1") lines)"
xargs -a "$1" -d '\n' -P "${SLURM_CPUS_PER_TASK}" -I{} bash -c '{}'
echo "PS_CPU_DONE rc=$? $(date)"
