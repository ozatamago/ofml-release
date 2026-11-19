#!/bin/bash
# Production trajectories -- DO NOT submit before the main session approves.  One array task = one (method, rep),
# one CPU thread (xeon6530-h100nvl; same implementation as the search: CPU, compile on).
#   ssh xeon6530-h100nvl "cd <p1_neural> && sbatch --job-name=p1n_main --array=0-59 --export=ALL,OUT=runs/main_<date>,CFGDIR=runs/hpo_<date> slurm/main_traj.sbatch"
# task k -> method METHODS[k / NREP], rep REPS[k % NREP]; REPS defaults to "0 ... 11" (replacements: e.g. REPS="12 13").
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=0-12:00:00
#SBATCH --output=runs/slurm_logs/%x_%A_%a.out
set -euo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/experiments/p1_neural
: "${OUT:?}"; : "${CFGDIR:?}"
METHODS=(shared dagnn dagnn_cat sharedcell ofml)
read -r -a REPA <<< "${REPS:-0 1 2 3 4 5 6 7 8 9 10 11}"
NREP=${#REPA[@]}
K=${SLURM_ARRAY_TASK_ID}
M=${METHODS[$((K / NREP))]}; R=${REPA[$((K % NREP))]}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-}_${K} method=${M} rep=${R} start=$(date -Is)"
${PY:-python} -u p1_run.py --method "${M}" --problem "rep:${R}" --out "${OUT}" \
  --config "${CFGDIR}/selected_${M}.json" --device cpu --threads 1 --compile 1 --skip-existing
echo "end=$(date -Is)"
