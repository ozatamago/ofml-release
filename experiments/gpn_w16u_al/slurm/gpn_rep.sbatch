#!/bin/bash
# OFML-GPN, PB-W16U active learning: one array task = one rep (teacher), all four policies (shared warm-up), 1 GPU.
#   ssh xeon6530-h100nvl "cd <gpn_w16u_al> && OUT=runs/<date>_<tag> sbatch --job-name=w16u_gpn --array=0-15%2 slurm/gpn_rep.sbatch"
# Production BENCH9 PB-W16U inference flags are the runner defaults (w16u_gpn.PROD_FLAGS).
# Resume: resubmit the same command; finished (rep, policy) files are skipped and checkpoints resumed.
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm_logs/%x_%A_%a.out
set -euo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/experiments/gpn_w16u_al
: "${OUT:?set OUT=runs/<date>_<tag>}"
REP=${REP:-${SLURM_ARRAY_TASK_ID}}
mkdir -p "${OUT}"
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} rep=${REP} gpus=${SLURM_JOB_GPUS:-?} start=$(date -Is)"
${PY:-python} -u w16u_gpn.py --reps "${REP}" --out "${OUT}" \
  --checkpoint-every 10 --skip-existing --compile-cache runs/jax_cache ${EXTRA:-}
echo "end=$(date -Is)"
