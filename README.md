# OFML: code and run records of the experiments (Sections 6.2 and 6.3)

This repository contains the code and the run records of the two experiments of the paper on OFML
(surrogate models composed of operation-type modules along a typed protocol DAG):

- **Section 6.2, transfer across protocols** (internal name `p1`): OFML-GPN, Independent-GPN and four deep-ensemble
  surrogates (OFML-Ens, Shared-DAG-Ens, DAGNN-Cat-Ens, Shared-Ens) on 12 true functions.
- **Section 6.3, acquisition functions and uniform consistency** (internal name `w16u`, the family PB-W16U):
  OFML-GPN and OFML-Ens with IVR, MaxVar, space-filling and random acquisition on 16 true functions.

The run records are sufficient to regenerate every table and figure of these sections and of the corresponding
appendices without running the surrogate models again. The scripts that ran the experiments are included with the
settings used for the paper.

## Contents

```
src/ofml/                      the ofml package (operation models of the deep ensembles)
third_party/gpn_smc/           OFML-GPN engine used by Section 6.3 (copy; see its PROVENANCE.md)
scripts/                       paths (ofml_paths.py), figure style and colours, Japanese font conversion
experiments/
  gpn_graph/                   Section 6.2, OFML-GPN and Independent-GPN (engine: gpn_graph/ofml_gpn/, see PROVENANCE.md)
    runs/p1_main/problems/     the true functions (rep_NN.npz) and the tuning functions (tune_NN.npz)
    runs/p1_main/ofml_gpn/     run records of OFML-GPN (rep_NN.json, rep_NN.npz; stopped/ = runs stopped early)
    runs/p1_main/indep_gpn/    run records of Independent-GPN
    runs/p1_main/exclusion/    judgments.csv: the exclusion decision for every drawn true function
    runs/p1_main/auto/         log of the replacement procedure (rep 16 onwards)
    runs/p1_main/diag/         the test with 4x particles / 4x pCN updates
  p1_neural/                   Section 6.2, the four deep ensembles
    runs/main_2026-11-08/      run records (ofml, sharedcell = Shared-DAG-Ens, dagnn_cat, shared)
    runs/hpo_2026-11-05*/      hyperparameter search records and the selected settings
  analysis_transfer/           Section 6.2: tables, figures, coverage, fit check (fit12/ = its outputs)
  gpn_w16u_al/                 Section 6.3, OFML-GPN (w16u_gpn.py) and OFML-Ens (w16u_ens.py)
    teachers/                  the true functions (rep_XXXX.npz)
    runs/2026-11-04_main/gpn/  run records of OFML-GPN
    runs/prestage/prod_eval_fz/{s04,p02,shallow}/ens/   run records of OFML-Ens (s04 = the setting of the main text)
    runs/2026-11-04_main/ens_v2/                        run records of OFML-Ens with the old setting
    runs/prestage/{e4,e5,e6,configs}/                   selection of the OFML-Ens setting
    prestage/                  scripts of the setting selection; fz.py = the production wrapper of w16u_ens.py
  analysis_tables/             appendix aggregates (exclusion, run time, fit, OFML-Ens settings, sensitivity)
```

The run records do not contain checkpoints. Machine names are replaced by `xeon6130-v100` (2x Xeon Gold 6130,
8x Tesla V100 SXM2) and `xeon6530-h100nvl` (2x Xeon Gold 6530, 2x H100 NVL); Slurm job numbers are removed.

## Installation

Python 3.12 was used for the paper (Python >= 3.10 is required).

```bash
git clone https://github.com/ozatamago/ofml-release.git
cd ofml-release
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

`requirements.txt` pins the versions used for the paper; the `--extra-index-url` option installs the CPU build of
PyTorch (the deep ensembles ran on the CPU). For OFML-GPN on a GPU, additionally install the CUDA build of JAX:

```bash
pip install "jax[cuda12]==0.10.2"
```

OFML-GPN requires a GPU. The engine `ofml_gpn` of Section 6.2 has a CPU switch (`OFML_GPN_ALLOW_CPU=1`) for tests,
but CPU runs are too slow for anything but a few observations.

## Tutorial

All steps run from the repository root. Steps 1 and 2 use the CPU only and take about 6 minutes in total; step 3 needs a
GPU and takes about 5 minutes (measured on one H100 NVL). Outputs go to `outputs/` (tables and figures)
and `tutorial_runs/` (runs), both ignored by git.

### 1. Regenerate the tables of Sections 6.2 and 6.3 from the run records

```bash
cd experiments/analysis_transfer
python make_table_obj1.py --reps 2,4,5,9,11,12,15,17,19,23,27,28 --suffix _v12   # tab:exp-transfer
python make_table_obj1_paired.py                                                 # tab:gpm-paired
cd ../gpn_w16u_al
python analyze_w16u.py runs/2026-11-04_main --ens-dir ../prestage/prod_eval_fz/s04/ens \
    --outdir runs/2026-11-04_main/analysis_s04
python make_table_obj2_paired.py runs/2026-11-04_main analysis_s04               # tab:acq-paired
cd ../analysis_tables
python table_exclusion.py                 # tab:gpm-exclusion and the per-function decisions
python table_repro_time.py                # median run times of sec:repro
python table_gpm_fit.py                   # tab:gpm-fit
python table_acq_ens_settings.py          # tab:acq-ens-settings
python sensitivity_p1_rule_fig4b.py       # sec:acq-sensitivity
cd ../..
```

The `.tex` tables are written to `outputs/tables/`. The scripts in `analysis_tables/` print each value next to the
value printed in the paper and mark differences. `outputs/tables/table_obj1_v12_en.tex` should contain, for example,
the row of OFML-GPN of tab:exp-transfer.

### 2. Run the deep ensembles with small settings (CPU)

```bash
cd experiments/p1_neural
python p1_run.py --method ofml --problem rep:2 --T 10 --compile 1 --threads 1 \
    --config runs/hpo_2026-11-05_wide/selected_ofml.json --out ../../tutorial_runs/p1_neural
cd ../gpn_w16u_al
python prestage/fz.py w16u_ens.py --reps 0 --policies ivr --config runs/prestage/configs/e5_shallow.json \
    --device cpu --threads 1 --out ../../tutorial_runs/w16u
cd ../..
python scripts/check_tutorial.py
```

The first command runs OFML-Ens of Section 6.2 on true function 2 for the first 10 acquisitions (the paper:
`--T 200`; about 1 minute). The second runs OFML-Ens of Section 6.3 on true function 0 with IVR and the shallow
reference setting of the appendix (all 92 acquisitions; about 3 minutes; the main text uses `e5_s04.json`, which
takes about 13 hours per run). Both use the settings of the paper's runs (CPU, one thread, `--compile 1` in
Section 6.2, `prestage/fz.py` in Section 6.3), so `check_tutorial.py` finds that their error curves equal the
published run records (`p1_neural/runs/main_2026-11-08/ofml/rep_02.json` and
`gpn_w16u_al/runs/prestage/prod_eval_fz/shallow/ens/rep_0000_ivr.json`). Without `--compile 1` or with more threads
the curves differ slightly, because the floating-point operations are ordered differently.

### 3. Run OFML-GPN with small settings (GPU)

```bash
cd experiments/gpn_graph
python run_p1_main.py --problem runs/p1_main/problems/rep_02.npz --particles 1024 --pcn-steps 8 --T 5 \
    --out ../../tutorial_runs/ofml_gpn
cd ../gpn_w16u_al
python w16u_gpn.py --reps 0 --policies ivr --particles 1024 --pcn-steps 8 --out ../../tutorial_runs/w16u
cd ../..
python scripts/check_tutorial.py --gpu
```

The first command assimilates the 80 observations of the source protocols (D_0) and 5 target observations (about
2.5 minutes); the second runs all 92 acquisitions of Section 6.3 (about 2 minutes). The paper used 16384 particles
and 96 pCN updates (`--particles 16384 --pcn-steps 96`, the defaults) and `--T 200` in Section 6.2, so the numbers
differ from the run records; with 1024 particles the exclusion rule of Section 6.2 may be hit (the record then says
`"excluded": true`). `--pcn-steps` must be a multiple of 8.

### 4. Figures

```bash
cd experiments/analysis_transfer
python plot_fig3_f4p_v2.py --reps 2,4,5,9,11,12,15,17,19,23,27,28 --lang en --out ../../outputs/figures/fig_obj1_v2
python plot_protocols.py --lang en
cd ../gpn_w16u_al
python make_fig_obj2_f4p_v2.py runs/2026-11-04_main --ens-dir runs/prestage/prod_eval_fz/s04/ens \
    --ens-label "setting selected in the pre-stage" --lang en --out ../../outputs/figures/fig_obj2_v2_s04
python plot_protocols_w16u.py --lang en
cd ../..
```

The Japanese figures (`--lang ja`) need the Harano Aji Gothic font (SIL Open Font License 1.1,
<https://github.com/trueroad/HaranoAjiFonts>). Put its OpenType files in `fonts/` (or set `OFML_JA_FONT_DIR`) and run
`python scripts/otf2ttf_haranoaji.py fonts` (needs `pip install fonttools`) to create the TrueType files used by the
figure scripts. The Japanese figures of the paper were made with `--ens-label 前段階で選んだ設定`.

## Reproducing the runs

The commands of the paper's runs are recorded next to the run records:

- Section 6.2, OFML-GPN and Independent-GPN: `experiments/gpn_graph/runs/p1_main/slurm/` (`p1_rep.sbatch` for the
  first 16 true functions, `p1_auto.py` for the replacement procedure; `--abort-on-exclusion` stops a run when the
  exclusion rule is hit).
- Section 6.2, deep ensembles: `experiments/p1_neural/slurm/main_traj.sbatch` (one CPU thread per run) and
  `hpo_method.sbatch` (hyperparameter search).
- Section 6.3, OFML-GPN: `experiments/gpn_w16u_al/slurm/gpn_rep.sbatch`.
- Section 6.3, OFML-Ens: `experiments/gpn_w16u_al/runs/prestage/prod_eval/prod_eval_fz.cmds` (one command per run,
  through `prestage/fz.py`, which flushes subnormal numbers to zero) and `prestage/cpu_cmds.sbatch`.

The Slurm scripts are records of the cluster used for the paper; set `OFML_ROOT` to the repository root and `PY` to
the Python of the environment, and adapt the `#SBATCH` lines to your cluster. Comments in the code refer to design
notes (`REPORTS/...`) that are not part of this repository.

The true functions of Section 6.3 (`teachers/`) were generated by `make_teachers.py` with the functions of an
unpublished repository of the authors (`OFML_CX_BAYES`); the cache is checked by `w16u_common.load_teacher` when it is
read. The GPN record stores `teacher_sha256` of (truth, noisy - truth) and the OFML-Ens record of (truth, noise); the two
hashes differ by floating-point rounding only.

## Exclusion decisions (Appendix E.6)

`experiments/gpn_graph/runs/p1_main/exclusion/judgments.csv` lists, for each true function drawn in Section 6.2
(rep 0-15: the first 16; rep 16-28: drawn for replacement; rep 29-33: run after the five replacements were
collected), the minimum of the effective sample size divided by the number of particles, whether the threshold 0.001
was crossed, and at which stage (`source` = assimilation of D_0, `target` = target-protocol observations), for
OFML-GPN and Independent-GPN, and whether the function entered the comparison. It is written by
`experiments/analysis_tables/table_exclusion.py` from the run records; runs stopped before T = 200 are read from
`<method>/stopped/rep_NN_rounds.json` (extracted from their checkpoints by
`experiments/gpn_graph/scripts/extract_stopped_rounds.py`).

## License

MIT (see `LICENSE`). The run records are distributed under the same terms. The Harano Aji fonts used for the
Japanese figures are not included (SIL Open Font License 1.1).
