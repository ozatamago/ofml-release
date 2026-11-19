#!/bin/bash
# Purpose 1 (c) automatic replacement: one OFML-GPN function (REP), stopped as soon as the ESS rule is hit.
# Submitted only by p1_auto.py (launch).  On success it submits Independent-GPN and the neural methods for REP;
# in every case it then asks p1_auto.py to launch the next function if more are needed.
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/p1_main/auto/logs/%x_%j.out
set -uo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/experiments/gpn_graph
PY=${PY:-python}
AUTO="$PY runs/p1_main/slurm/p1_auto.py"
: "${REP:?}"
HOST=$(hostname)
PROB=$(printf "runs/p1_main/problems/rep_%02d.npz" "$REP")
echo "host=$HOST job=$SLURM_JOB_ID rep=$REP problem=$PROB start=$(date -Is)"
$AUTO set "$REP" ofml running
$PY -u run_p1_main.py --problem "$PROB" --out runs/p1_main/ofml_gpn --skip-existing \
  --compile-cache runs/p1_main/jax_cache --abort-on-exclusion
RC=$?
echo "rc=$RC end=$(date -Is)"
case $RC in
  0) $AUTO set "$REP" ofml ok; $AUTO followups "$REP" "$HOST" ;;
  3) $AUTO set "$REP" ofml excluded ;;
  *) $AUTO set "$REP" ofml failed ;;
esac
$AUTO launch "$HOST"
exit 0
