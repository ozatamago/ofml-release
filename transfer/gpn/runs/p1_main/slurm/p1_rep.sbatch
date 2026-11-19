#!/bin/bash
# Purpose 1 (c) GP-matched benchmark: one array task = one problem (true function), one method, 1 GPU.
#   METHOD=main  -> run_p1_main.py  (OFML-GPN,        out runs/p1_main/ofml_gpn)
#   METHOD=indep -> run_p1_indep.py (Independent-GPN, out runs/p1_main/indep_gpn)
#   KIND=rep (default) or tune; the array index is the problem number (rep_XX.npz / tune_XX.npz).
# Submit with submit_p1.sh (xeon6130-v100 locally, xeon6530-h100nvl over ssh).  Resume: resubmit the same task; finished problems
# are skipped (--skip-existing) and unfinished ones continue from runs/p1_main/<out>/ckpt/.
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/p1_main/slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/transfer/gpn
PY=${PY:-python}
: "${METHOD:?set METHOD=main|indep}"
KIND=${KIND:-rep}
IDX=${IDX:-${SLURM_ARRAY_TASK_ID}}
PROB=$(printf "runs/p1_main/problems/%s_%02d.npz" "$KIND" "$IDX")
case "$METHOD" in
  main)  SCRIPT=run_p1_main.py;  OUT=${OUT:-runs/p1_main/ofml_gpn} ;;
  indep) SCRIPT=run_p1_indep.py; OUT=${OUT:-runs/p1_main/indep_gpn} ;;
  *) echo "bad METHOD=$METHOD"; exit 2 ;;
esac
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} method=$METHOD problem=$PROB out=$OUT gpus=${SLURM_JOB_GPUS:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
START=$(date +%s)
$PY -u $SCRIPT --problem "$PROB" --out "$OUT" --skip-existing --compile-cache runs/p1_main/jax_cache ${EXTRA:-}
RC=$?
echo "rc=$RC wall=$(( $(date +%s) - START ))s end=$(date -Is)"
exit $RC
