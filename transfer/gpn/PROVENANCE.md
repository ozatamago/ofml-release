# ofml_gpn -- provenance

Origin: `cx_bayes/gpn_smc/` (an unpublished repository of the authors; the copy used by Section 6.3 is `third_party/gpn_smc/`) (package `gpn_smc`, the cx_bayes OFML-GPN engine). cx_bayes is not a
git repository at this point, so there is no commit id: the origin is identified by the md5 of every file below
(copied 2026-11-04T17:52 JST; `graph_backends.py` 73deefb7ac8f601a2cf28bfe3bd8b87d = the `MD5_final.txt` of
`cx_bayes/ucb/noisy_ucb_experiments/results/u15_comparison/bench9_mrg22_mixing_2026-11-04/`, i.e. the 2026-11-04
inner-collapsed final). All `*.py` files were copied (no `.bak*`, no `__pycache__`). The package was renamed
`ofml_gpn` so that it cannot collide with `gpn_smc` imported elsewhere in this repo (e.g. `acquisition/`);
all intra-package imports are relative, so no import line had to change.

| file | origin md5 (at copy) | ofml_gpn md5 (now) | status |
|---|---|---|---|
| __init__.py | fe573f21d46dee57364f656108e19052 | e3131c8ef20aac983f3a9e62b1561545 | MODIFIED (see below) |
| assimilate.py | cc030308564fb9e29a24cecc10e8fc45 | cc030308564fb9e29a24cecc10e8fc45 | unchanged |
| backends.py | 8cc7666c9d2331b1c5b505233199144d | 8cc7666c9d2331b1c5b505233199144d | unchanged |
| constants.py | a26e7bf2877a997c376bd836f3666580 | a26e7bf2877a997c376bd836f3666580 | unchanged |
| coverage_quantiles.py | f727db0cc9af650446144027c000227d | f727db0cc9af650446144027c000227d | unchanged |
| factorised_independent.py | ff5c28b60d02295ae0bd522ff3b8a127 | ff5c28b60d02295ae0bd522ff3b8a127 | unchanged |
| graph_backends.py | 73deefb7ac8f601a2cf28bfe3bd8b87d | 55520c114a8938a9a5a9fa33bc8623b1 | MODIFIED (see below) |
| graph_library.py | 334604dc9c6b588e5fc62ee5a16a9185 | 58dd445f81a77baa57b45fe03375a88a | MODIFIED (see below) |
| library.py | 8ac8da0354dd9d9b83614bcc875a9888 | 8ac8da0354dd9d9b83614bcc875a9888 | unchanged |
| numerics.py | a173da280bdbbb2ad7378a93acd6e4b3 | a173da280bdbbb2ad7378a93acd6e4b3 | unchanged |
| pcn.py | 1cf1e8383d49e7dfcf1370e85a2c46d5 | 1cf1e8383d49e7dfcf1370e85a2c46d5 | unchanged |
| pushforward.py | 737d697be2a25cbcebf23d5cc8a160de | 737d697be2a25cbcebf23d5cc8a160de | unchanged |
| quantile_acq.py | 8a62364d5218301b61096b3f3778d221 | 8a62364d5218301b61096b3f3778d221 | unchanged |
| term_library.py | - | 78c2c6091fb70411915265944d99607a | new (ofml_gpn) |
| terminal_rb.py | d1a8acd2acf393f7e1de45a55d76e0a1 | d1a8acd2acf393f7e1de45a55d76e0a1 | unchanged |

Modified / new files:
* `graph_library.py`: self-feeding types (evaluation-block schedule, Cholesky order, generalised teacher); the
  origin's "parent in an EARLIER type" rule is removed (`old_valid` reports it). Origin libraries unchanged.
* `graph_backends.py`: (i)-(iii) of `REPORTS/GPN_GRAPH_EXTENSION.md`; CPU / grid test switches
  (`OFML_GPN_ALLOW_CPU=1`, `OFML_GPN_ALLOW_ANY_GRID=1`); gate-only hooks `_flat_lik`, `_cm_variant`.
* `__init__.py`: docstring header only.
* `term_library.py` (new): term-string library builder and the purpose-1 DRAFT library.

Gates (`../gates/`): Gate 0 = the copy (before any change except the CPU switch) vs cx_bayes, Gate 1 = the
modified copy vs cx_bayes on origin libraries, both bit for bit; see the report.

(table regenerated 2026-11-04T21:00:44)
