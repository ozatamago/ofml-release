#!/bin/bash
# Bayesian-optimisation search of ONE neural method (design 1.8.6) on xeon6530-h100nvl CPUs: 50 trials x 4 tuning functions,
# every trajectory exactly as production (T = 200, IVR, cold refit each t, 400 steps, 5 members).  The 4 x PARALLEL
# trajectories in flight run as single-thread processes (torch.compile'd loss; CPU is bit-reproducible across
# processes, the GPU gives no throughput gain for these small nets and is left to the GP methods).
# Resumable: resubmit the same command (trials on file are kept).
#   ssh xeon6530-h100nvl "cd <transfer/ensembles> && sbatch --job-name=p1n_hpo_<m> --export=ALL,METHOD=<m>,OUT=runs/hpo_<date> slurm/hpo_method.sbatch"
#   (PARALLEL=4 -> --cpus-per-task=16 on the command line)
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm_logs/%x_%j.out
set -euo pipefail
cd "${OFML_ROOT:?set OFML_ROOT to the repository root}"/transfer/ensembles
: "${METHOD:?}"; : "${OUT:?}"; PARALLEL=${PARALLEL:-1}; DAGNN_MAP=${DAGNN_MAP:-state_latent}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# INIT_FROM=<dir>: if OUT/METHOD.jsonl does not exist yet, start it from <dir>/METHOD.jsonl (continue an earlier search
# in a wider --space, e.g. EXTRA="--space wide --trials 80").
if [ -n "${INIT_FROM:-}" ] && [ ! -e "${OUT}/${METHOD}.jsonl" ]; then mkdir -p "${OUT}"; cp "${INIT_FROM}/${METHOD}.jsonl" "${OUT}/${METHOD}.jsonl"; fi
echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} method=${METHOD} parallel=${PARALLEL} start=$(date -Is)"
${PY:-python} -u p1_hpo.py --method "${METHOD}" --out "${OUT}/${METHOD}.jsonl" \
  --trials 50 --parallel "${PARALLEL}" --device cpu --threads 1 --compile 1 --dagnn-map "${DAGNN_MAP}" ${EXTRA:-}
echo "end=$(date -Is)"
