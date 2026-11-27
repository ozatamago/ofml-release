# OfmlAppendix: Lean checks of selected proof steps in Appendices B, C, and D

This Lean 4 project machine-checks self-contained steps of the proofs in Appendices B, C, and D of the paper.
It does not formalize the whole proof (stochastic processes, regular conditional distributions, and Bochner integrals are outside its scope).
`SPEC.md` (in Japanese) lists each Lean theorem, the corresponding step of the paper, and the parts that are not formalized.

| Lean theorem | Paper |
|---|---|
| `OfmlAppendix.summable_of_drift` | Proof of Lemma B.5: telescoping sum and Tonelli step |
| `OfmlAppendix.affine_predictor_bound` | Proof of Lemma B.6: comparison with the best affine predictor (projection inequality) |
| `OfmlAppendix.inf_ball_measure_pos` | Proof of Lemma B.7: uniform positive lower bound of ball measures |
| `OfmlAppendix.covariance_near_maximizer` | Proof of Lemma B.7: a radius ρ common to the whole subsequence |
| `OfmlAppendix.integral_lower_bound` | Proof of Lemma B.7: lower bound of the integral |
| `OfmlAppendix.lipschitz_assembly` | Appendix C: construction of L_f and ω |
| `OfmlAppendix.approx_maximizer_gap` | Lemma D.1: inequality for an approximate acquisition function |
| `OfmlAppendix.approx_tolerance_tendsto` | Lemma D.1: the tolerance converges to zero |
| `OfmlAppendix.ciSup_eq_ciSup_dense` | Proof of Proposition D.2: supremum over a dense set |
| `OfmlAppendix.measurable_ciSup_of_continuous` | Proof of Proposition D.2: measurability of the supremum |
| `OfmlAppendix.summable_of_drift_le` | Proof of Proposition D.2: telescoping and Tonelli step with the drift inequality |
| `OfmlAppendix.sup_tendsto_zero_of_selection` | Proof of Proposition D.2: the supremum of the lower bound converges to zero |

## Build and check

Requirements: [elan](https://github.com/leanprover/elan). The toolchain (`lean-toolchain`) is Lean v4.32.0-rc1 and Mathlib is pinned to v4.32.0-rc1 (`lake-manifest.json`).

```bash
cd lean/OfmlAppendix
lake exe cache get   # download prebuilt Mathlib
lake build
./tests/check.sh
```

`tests/check.sh` verifies that the project builds, that no `sorry`, `admit`, `axiom`, `native_decide`, or `implemented_by` is used, that every theorem depends only on the axioms `propext`, `Classical.choice`, and `Quot.sound` (`tests/AxiomAudit.lean`), and that the hypotheses of every theorem are satisfiable by a concrete example (`tests/NonVacuous.lean`).
