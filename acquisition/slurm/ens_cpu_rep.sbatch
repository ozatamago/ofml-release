#!/bin/bash
# OFML-Ens, PB-W16U active learning, CPU only: one array task = one rep; the four policies run as four
# single-thread processes, each trajectory in its own batch (B=1), so a trajectory's result depends only on
# (rep, policy, config) -- not on which other trajectories share the batch (float32 batched kernels differ with
# the batch size; see EXP_W16U_AL_IMPLEMENTATION.md).
#   ssh xeon6530-h100nvl "cd <acquisition> && OUT=runs/<date>_<tag> CONFIG=configs/ens_selected_2026-11-04.json sbatch --job-name=w16u_ens --array=0-15 slurm/ens_cpu_rep.sbatch"
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=runs/slurm_logs/%x_%A_%a.out
set -euo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/acquisition
: "${OUT:?set OUT}"; : "${CONFIG:?set CONFIG}"; MODEL_DIR=${MODEL_DIR:-ens}
REP=${REP:-${SLURM_ARRAY_TASK_ID}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} rep=${REP} start=$(date -Is)"
mkdir -p "${OUT}/${MODEL_DIR}_logs"
pids=()
for P in ivr maxvar spacefill random; do
  ${PY:-python} -u w16u_ens.py --reps "${REP}" --policies "${P}" --out "${OUT}" \
    --config "${CONFIG}" --device cpu --threads 1 --tag "r${REP}_${P}" --skip-existing --model-dir "${MODEL_DIR}" > "${OUT}/${MODEL_DIR}_logs/rep_$(printf %04d ${REP})_${P}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "end=$(date -Is) rc=${rc}"; exit ${rc}
