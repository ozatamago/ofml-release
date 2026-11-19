#!/bin/bash
# usage: submit_p1.sh HOST METHOD LIST [KIND] [EXTRA]
#   HOST   xeon6130-v100 | xeon6530-h100nvl        (xeon6530-h100nvl is reached with ssh; the file system is shared)
#   METHOD main | indep
#   LIST   Slurm array list of problem numbers, e.g. "0-5", "0,3,7", "6-11%2" (%N = at most N at once)
#   KIND   rep (default) | tune
#   EXTRA  extra runner flags, e.g. "--T 10 --particles 2048" (tests only; use another OUT for tests)
# Env OUT overrides the output directory (default runs/p1_main/ofml_gpn | indep_gpn).
set -euo pipefail
HOST=$1; METHOD=$2; LIST=$3; KIND=${4:-rep}; EXTRA=${5:-}
D="${OFML_ROOT:?set OFML_ROOT to the repository root}"/experiments/gpn_graph
NAME="p1${METHOD}_${KIND}"
ENVS="METHOD=${METHOD} KIND=${KIND} EXTRA=$(printf %q "$EXTRA")${OUT:+ OUT=$(printf %q "$OUT")}"
CMD="cd $D && ${ENVS} sbatch --parsable --job-name=${NAME} --array=${LIST} runs/p1_main/slurm/p1_rep.sbatch"
case "$HOST" in
  xeon6130-v100)   if [ "$(hostname)" = xeon6130-v100 ]; then bash -c "$CMD"; else ssh xeon6130-v100 "$CMD"; fi ;;
  xeon6530-h100nvl) if [ "$(hostname)" = xeon6530-h100nvl ]; then bash -c "$CMD"; else ssh xeon6530-h100nvl "$CMD"; fi ;;
  *) echo "bad HOST=$HOST"; exit 2 ;;
esac
