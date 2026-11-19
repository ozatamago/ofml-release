#!/usr/bin/env bash
# Regenerate the tables, figures and appendix values of Sections 6.2 and 6.3 from the published run records and
# compare them with the paper. CPU only; about 1 minute. Outputs: outputs/tables, outputs/figures, and the printed log.
# usage: bash scripts/reproduce_paper.sh            (set PY to choose the Python interpreter)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
REPS=2,4,5,9,11,12,15,17,19,23,27,28   # the 12 true functions of the comparison in Section 6.2

echo "== Section 6.2: tab:exp-transfer, tab:gpm-paired"
cd "$ROOT/transfer/analysis"
$PY make_table_obj1.py --reps $REPS --suffix _v12 > /dev/null
$PY make_table_obj1_paired.py > /dev/null
echo "== Section 6.2: tab:gpm-exclusion and the per-function exclusion decisions"
$PY table_exclusion.py
echo "== Section 6.2: tab:gpm-fit"
$PY table_gpm_fit.py
echo "== Section 6.2: 95% coverage of OFML-GPN (sec:gpm-exclusion); the paper quotes the row 'mean' of 全体 (all 448 points)"
$PY coverage_by_t.py $REPS | grep -A0 -E "^## |平均"
echo "== Section 6.2: input range of the merge operation M (sec:gpm-bench; the paper: fraction beyond the length scale 0.15 = 1 - mean)"
(cd "$ROOT/transfer/gpn" && JAX_PLATFORMS=cpu OFML_GPN_ALLOW_ANY_GRID=1 $PY analysis/p1_type_ranges.py | grep -m1 "^M ")
echo "== Section 6.2: fig:exp-transfer, fig:exp-protocols"
$PY plot_fig3_f4p_v2.py --reps $REPS --lang en --out "$ROOT/outputs/figures/fig_obj1_v2" > /dev/null
$PY plot_protocols.py --lang en > /dev/null

echo "== Section 6.3: tab:acq-paired"
cd "$ROOT/acquisition"
$PY analyze_w16u.py runs/2026-11-04_main --ens-dir ../prestage/prod_eval_fz/s04/ens \
    --outdir runs/2026-11-04_main/analysis_s04 > /dev/null
$PY make_table_obj2_paired.py runs/2026-11-04_main analysis_s04 > /dev/null
echo "== Section 6.3: values of fig:exp-acquisition (b) quoted in the text"
$PY fig_acquisition_values.py | tail -3
echo "== Section 6.3: tab:acq-ens-settings"
$PY table_acq_ens_settings.py
echo "== Section 6.3: tab:acq-sd-err"
$PY sd_err_calibration.py runs/2026-11-04_main "$ROOT/acquisition/runs/prestage/prod_eval_fz/s04/ens" | tail -8
echo "== Section 6.3: sec:acq-sensitivity"
$PY sensitivity_p1_rule.py runs/2026-11-04_main
$PY sensitivity_p1_rule_fig4b.py
echo "== Section 6.3: fig:exp-acquisition, fig:exp-acq-protocols"
$PY make_fig_obj2_f4p_v2.py runs/2026-11-04_main --ens-dir runs/prestage/prod_eval_fz/s04/ens \
    --ens-label "selected setting" --lang en --out "$ROOT/outputs/figures/fig_obj2_v2_s04" > /dev/null
$PY plot_protocols_w16u.py --lang en > /dev/null

cd "$ROOT"
echo "== Appendix G: median run times"
$PY scripts/table_repro_time.py
echo "== Comparison of the .tex tables with the paper"
$PY scripts/compare_tables.py
echo "figures: $(ls outputs/figures/*.pdf | xargs -n1 basename | tr '\n' ' ')"
