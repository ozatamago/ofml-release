#!/bin/bash
# Purpose 1 (c) automatic replacement: Independent-GPN for a function (REP) on which OFML-GPN passed the ESS rule.
# Submitted only by p1_auto.py (followups).  If this run hits the rule, the function is excluded and a new
# OFML-GPN worker is launched when needed.
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/p1_main/auto/logs/%x_%j.out
set -uo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/transfer/gpn
PY=${PY:-python}
AUTO="$PY runs/p1_main/slurm/p1_auto.py"
: "${REP:?}"
HOST=$(hostname)
PROB=$(printf "runs/p1_main/problems/rep_%02d.npz" "$REP")
echo "host=$HOST job=$SLURM_JOB_ID rep=$REP problem=$PROB start=$(date -Is)"
$AUTO set "$REP" indep running
$PY -u run_p1_indep.py --problem "$PROB" --out runs/p1_main/indep_gpn --skip-existing \
  --compile-cache runs/p1_main/jax_cache --abort-on-exclusion
RC=$?
echo "rc=$RC end=$(date -Is)"
case $RC in
  0) $AUTO set "$REP" indep ok ;;
  3) $AUTO set "$REP" indep excluded ;;
  *) $AUTO set "$REP" indep failed ;;
esac
$AUTO launch "$HOST"
exit 0
