import OfmlAppendix

open MeasureTheory OfmlAppendix
open scoped BigOperators Topology ENNReal InnerProductSpace

namespace OfmlAppendixTests

noncomputable def testU : EuclideanSpace ℝ (Fin 2) :=
  EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1

noncomputable def testPredictor : EuclideanSpace ℝ (Fin 2) :=
  EuclideanSpace.single 0 1

-- The vectors are (1, 1) and (1, 0), so the bound need not be an equality.
example :
    ⟪testU, testPredictor⟫_ℝ ^ 2 / ‖testPredictor‖ ^ 2 ≤
      ‖((⊤ : Submodule ℝ (EuclideanSpace ℝ (Fin 2))).orthogonalProjectionOnto
        testU : EuclideanSpace ℝ (Fin 2))‖ ^ 2 := by
  apply affine_predictor_bound
  · simp
  · simp [testPredictor]

abbrev SingletonInterval := Set.Icc (0 : ℝ) 0

noncomputable def singletonPoint : SingletonInterval :=
  ⟨0, by simp⟩

example :
    ∃ c : ℝ≥0∞, 0 < c ∧
      ∀ x : SingletonInterval,
        c ≤ (Measure.dirac singletonPoint) (Metric.ball x 1) := by
  apply inf_ball_measure_pos
    (ν := Measure.dirac singletonPoint) (ρ := 1)
  · intro x r hr
    have hx : x = singletonPoint := by
      apply Subtype.ext
      exact le_antisymm x.property.2 x.property.1
    subst x
    simp [Measure.dirac_apply, hr]
  · norm_num

example :
    ∃ ρ : ℝ, 0 < ρ ∧
      ∀ x : ℝ, dist 0 x < ρ → (1 : ℝ) / 2 ≤ 2 := by
  obtain ⟨ρ, hρ, hbound⟩ :=
    covariance_near_maximizer
      (X := ℝ) (s2min := 1) (C := 1) (ω := fun r => r)
      rfl continuousAt_id (by norm_num) (by norm_num)
  refine ⟨ρ, hρ, ?_⟩
  apply hbound (0 : ℝ) (fun _ => 2) 2
  · norm_num
  · intro x
    simpa using (dist_nonneg : (0 : ℝ) ≤ dist (0 : ℝ) x)

example :
    (1 : ℝ) ^ 2 *
        ((Measure.dirac (0 : ℝ)) (Metric.ball 0 1)).toReal /
        (4 * (1 + 1)) ≤
      ∫ _x : ℝ, (1 : ℝ) ^ 2 / (1 + 1) ∂Measure.dirac 0 := by
  apply integral_lower_bound
    (ν := Measure.dirac (0 : ℝ))
    (x₀ := 0) (k := fun _ => 1)
    (s2 := 1) (s2min := 1) (S := 1) (ξ2 := 1) (ρ := 1)
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · intro x hx
    norm_num
  · exact integrable_const _

noncomputable def testV (t : ℕ) (_x : ℝ) : ℝ :=
  if t = 0 then 1 else 0

noncomputable def testDelta (t : ℕ) (_x : ℝ) : ℝ :=
  if t = 0 then 1 else 0

example :
    (∀ n : ℕ,
      (∑ t ∈ Finset.range n,
        ∫ x : ℝ, testDelta t x ∂Measure.dirac 0) ≤
        ∫ x : ℝ, testV 0 x ∂Measure.dirac 0) ∧
      Summable
        (fun t => ∫ x : ℝ, testDelta t x ∂Measure.dirac 0) ∧
      (∀ᵐ x : ℝ ∂Measure.dirac 0,
        Summable (fun t => testDelta t x)) := by
  apply summable_of_drift
    (μ := Measure.dirac (0 : ℝ)) (V := testV) (Δ := testDelta)
  · intro t x
    dsimp [testV]
    split_ifs <;> norm_num
  · intro t x
    dsimp [testDelta]
    split_ifs <;> norm_num
  · intro t
    exact integrable_const _
  · intro t
    exact integrable_const _
  · intro t
    exact measurable_const.aemeasurable
  · intro t
    cases t with
    | zero => simp [testV, testDelta]
    | succ t => simp [testV, testDelta]

noncomputable def protocolDistance (x y : Bool) : ℝ :=
  if x = y then 0 else 1

noncomputable def protocolSignal (x : Bool) : ℝ :=
  if x = true then 1 else -1

example :
    ∀ x y : Bool,
      |protocolSignal x - protocolSignal y| ≤
        max (0 : ℝ) (2 * 1) * protocolDistance x y := by
  apply lipschitz_assembly
    (prot := fun x : Bool => x)
    (d := protocolDistance) (f := protocolSignal)
    (B := 1) (K := 0)
  · intro x y
    dsimp [protocolDistance]
    split_ifs <;> norm_num
  · intro x y hne
    simp [protocolDistance, hne]
  · intro x
    cases x <;> norm_num [protocolSignal]
  · intro x y hsame
    change x = y at hsame
    subst y
    simp

/-! ## Appendix D -/

noncomputable def testGain (b : Bool) : ℝ := if b then 1 else 0
noncomputable def testGainHat (b : Bool) : ℝ := if b then 1 else 1 / 2

-- The selected point `false` is not a maximizer, and `δ = 1/2` is attained.
example :
    (⨆ x, testGain x) - testGain false ≤
      2 * (1 / 2 : ℝ) + ((⨆ x, testGainHat x) - testGainHat false) := by
  apply approx_maximizer_gap
  · exact (Set.finite_range testGainHat).bddAbove
  · intro x
    cases x <;> norm_num [testGain, testGainHat, abs_of_nonneg]

example :
    Filter.Tendsto (fun t : ℕ => 2 * (1 / ((t : ℝ) + 1)) + (1 / ((t : ℝ) + 1)))
      Filter.atTop (𝓝 0) :=
  approx_tolerance_tendsto (fun t : ℕ => 1 / ((t : ℝ) + 1)) (fun t : ℕ => 1 / ((t : ℝ) + 1))
    tendsto_one_div_add_atTop_nhds_zero_nat tendsto_one_div_add_atTop_nhds_zero_nat

example :
    (⨆ x : ℝ, Real.cos x) = ⨆ q : Set.range ((↑) : ℚ → ℝ), Real.cos q := by
  apply ciSup_eq_ciSup_dense
  · exact Rat.denseRange_cast
  · exact Real.continuous_cos
  · exact ⟨1, by rintro _ ⟨x, rfl⟩; exact Real.cos_le_one x⟩

example :
    Measurable fun ω : ℝ => ⨆ x : ℝ, Real.cos (ω + x) := by
  apply measurable_ciSup_of_continuous (Set.range ((↑) : ℚ → ℝ))
  · exact Set.countable_range _
  · exact Rat.denseRange_cast
  · intro ω
    exact Real.continuous_cos.comp (continuous_const.add continuous_id)
  · intro ω
    exact ⟨1, by rintro _ ⟨x, rfl⟩; exact Real.cos_le_one _⟩
  · intro q _
    exact (Real.continuous_cos.comp (continuous_id.add continuous_const)).measurable

noncomputable def testV2 (t : ℕ) (_x : ℝ) : ℝ :=
  if t = 0 then 2 else 0

-- The drift inequality is strict at `t = 0` (0 ≤ 2 - 1).
example :
    (∀ n : ℕ,
      (∑ t ∈ Finset.range n,
        ∫ x : ℝ, testDelta t x ∂Measure.dirac 0) ≤
        ∫ x : ℝ, testV2 0 x ∂Measure.dirac 0) ∧
      Summable
        (fun t => ∫ x : ℝ, testDelta t x ∂Measure.dirac 0) ∧
      (∀ᵐ x : ℝ ∂Measure.dirac 0,
        Summable (fun t => testDelta t x)) := by
  apply summable_of_drift_le
    (μ := Measure.dirac (0 : ℝ)) (V := testV2) (Δ := testDelta)
  · intro t x
    dsimp [testV2]
    split_ifs <;> norm_num
  · intro t x
    dsimp [testDelta]
    split_ifs <;> norm_num
  · intro t
    exact integrable_const _
  · intro t
    exact integrable_const _
  · intro t
    exact measurable_const.aemeasurable
  · intro t
    cases t with
    | zero => simp [testV2, testDelta]
    | succ t => simp [testV2, testDelta]

example :
    Filter.Tendsto (fun t : ℕ => ((1 : ℝ) / 2) ^ t) Filter.atTop (𝓝 0) := by
  apply sup_tendsto_zero_of_selection
    (fun t : ℕ => ((1 : ℝ) / 2) ^ t) (fun t : ℕ => ((1 : ℝ) / 2) ^ t) (fun _ => 0)
  · exact summable_geometric_of_lt_one (by norm_num) (by norm_num)
  · exact tendsto_const_nhds
  · intro t
    positivity
  · intro t
    simp

end OfmlAppendixTests
