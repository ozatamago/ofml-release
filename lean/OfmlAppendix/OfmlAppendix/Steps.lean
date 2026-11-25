import Mathlib

open MeasureTheory
open scoped BigOperators Topology ENNReal InnerProductSpace

namespace OfmlAppendix

/-- Expectation under the uniform probability measure on `Bool`. -/
noncomputable def twoPointMean (F : Bool → ℝ) : ℝ :=
  (F false + F true) / 2

/-- Variance under the uniform probability measure on `Bool`. -/
noncomputable def twoPointVariance (F : Bool → ℝ) : ℝ :=
  ((F false - twoPointMean F) ^ 2 +
    (F true - twoPointMean F) ^ 2) / 2

/-- Covariance under the uniform probability measure on `Bool`. -/
noncomputable def twoPointCovariance (F Y : Bool → ℝ) : ℝ :=
  ((F false - twoPointMean F) * (Y false - twoPointMean Y) +
    (F true - twoPointMean F) * (Y true - twoPointMean Y)) / 2

/--
The finite-partition formula for conditional expectation given `Y`
under the uniform probability measure on `Bool`.
-/
noncomputable def twoPointConditionalMean (F Y : Bool → ℝ) : Bool → ℝ := by
  classical
  exact fun b =>
    if Y false = Y true then twoPointMean F else F b

-- SCOPE: This formalizes the Hilbert-space projection inequality.
-- In the intended application, K is the closed subspace L²(σ(y^x)),
-- u represents the centred F, and v is a nonzero centred affine
-- function of y^x belonging to K. The identification of conditional
-- expectation with this orthogonal projection, of covariance with
-- the inner product, and of variances with these squared norms is
-- not formalized here.
theorem affine_predictor_bound
    {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [CompleteSpace E]
    (K : Submodule ℝ E) [K.HasOrthogonalProjection]
    (u v : E) (hv : v ∈ K) (hv0 : v ≠ 0) :
    ⟪u, v⟫_ℝ ^ 2 / ‖v‖ ^ 2 ≤
      ‖(K.orthogonalProjectionOnto u : E)‖ ^ 2 := by
  have hinner :
      ⟪u, v⟫_ℝ = ⟪(K.orthogonalProjectionOnto u : E), v⟫_ℝ := by
    have horth := Submodule.starProjection_inner_eq_zero (K := K) u v hv
    change
      ⟪u - (K.orthogonalProjectionOnto u : E), v⟫_ℝ = 0 at horth
    rw [inner_sub_left] at horth
    exact sub_eq_zero.mp horth
  have hnorm : 0 < ‖v‖ ^ 2 :=
    pow_pos (norm_pos_iff.mpr hv0) 2
  apply (div_le_iff₀ hnorm).2
  rw [hinner]
  simpa only [pow_two, real_inner_self_eq_norm_mul_norm] using
    real_inner_mul_inner_self_le (K.orthogonalProjectionOnto u : E) v

theorem inf_ball_measure_pos
    {X : Type*} [MetricSpace X] [CompactSpace X]
    [MeasurableSpace X] [BorelSpace X]
    (ν : Measure X) [IsFiniteMeasure ν]
    (hpos : ∀ x : X, ∀ r : ℝ, 0 < r →
      0 < ν (Metric.ball x r))
    (ρ : ℝ) (hρ : 0 < ρ) :
    ∃ c : ℝ≥0∞, 0 < c ∧
      ∀ x : X, c ≤ ν (Metric.ball x ρ) := by
  classical
  have hhalf : 0 < ρ / 2 := by linarith
  obtain ⟨s, hs⟩ :=
    isCompact_univ.elim_finite_subcover
      (fun x : X => Metric.ball x (ρ / 2))
      (fun _ => Metric.isOpen_ball)
      (by
        intro x _
        exact Set.mem_iUnion.mpr ⟨x, by simp [hhalf]⟩)
  have hfinite :
      ∀ t : Finset X, ∃ c : ℝ≥0∞, 0 < c ∧
        ∀ y ∈ t, c ≤ ν (Metric.ball y (ρ / 2)) := by
    intro t
    induction t using Finset.induction_on with
    | empty =>
        exact ⟨1, by norm_num, by simp⟩
    | @insert a t ha ih =>
        obtain ⟨c, hc, hcb⟩ := ih
        refine ⟨min c (ν (Metric.ball a (ρ / 2))),
          lt_min_iff.mpr ⟨hc, hpos a (ρ / 2) hhalf⟩, ?_⟩
        intro y hy
        rcases Finset.mem_insert.mp hy with rfl | hy
        · exact min_le_right _ _
        · exact le_trans (min_le_left _ _) (hcb y hy)
  obtain ⟨c, hc, hcb⟩ := hfinite s
  refine ⟨c, hc, ?_⟩
  intro x
  have hx : x ∈ ⋃ y ∈ s, Metric.ball y (ρ / 2) :=
    hs (Set.mem_univ x)
  simp only [Set.mem_iUnion] at hx
  obtain ⟨y, hy, hxy⟩ := hx
  apply le_trans (hcb y hy)
  apply measure_mono
  intro z hz
  change dist z x < ρ
  change dist z y < ρ / 2 at hz
  have hyx : dist y x < ρ / 2 := by
    rw [dist_comm]
    exact hxy
  have htriangle := dist_triangle z y x
  linarith

theorem covariance_near_maximizer
    {X : Type*} [MetricSpace X]
    (s2min C : ℝ) (ω : ℝ → ℝ)
    (hω0 : ω 0 = 0)
    (hω : ContinuousAt ω 0)
    (hC : 0 ≤ C)
    (hsmin : 0 < s2min) :
    ∃ ρ : ℝ, 0 < ρ ∧
      ∀ (x₀ : X) (k : X → ℝ) (s2 : ℝ),
        s2min ≤ s2 →
        (∀ x, |k x - s2| ≤ C * ω (dist x₀ x)) →
        ∀ x, dist x₀ x < ρ → s2min / 2 ≤ k x := by
  have hcont : ContinuousAt (fun r : ℝ => C * ω r) 0 :=
    continuousAt_const.mul hω
  obtain ⟨ρ, hρ, hsmall⟩ :=
    Metric.continuousAt_iff.mp hcont (s2min / 2) (by linarith)
  refine ⟨ρ, hρ, ?_⟩
  intro x₀ k s2 hmins2 hmod x hx
  have hout :
      dist (C * ω (dist x₀ x)) (C * ω 0) < s2min / 2 := by
    apply hsmall
    simpa only [Real.dist_eq, sub_zero, abs_of_nonneg dist_nonneg] using hx
  rw [hω0, mul_zero, Real.dist_eq, sub_zero] at hout
  have hu : C * ω (dist x₀ x) < s2min / 2 :=
    lt_of_le_of_lt (le_abs_self _) hout
  have hl := (abs_le.mp (hmod x)).1
  linarith

theorem integral_lower_bound
    {X : Type*} [MetricSpace X]
    [MeasurableSpace X] [BorelSpace X]
    (ν : Measure X) [IsFiniteMeasure ν]
    (x₀ : X) (k : X → ℝ)
    (s2 s2min S ξ2 ρ : ℝ)
    (hsmin : 0 < s2min)
    (hmins2 : s2min ≤ s2)
    (hsS : s2 ≤ S)
    (hξ : 0 < ξ2)
    (hρ : 0 < ρ)
    (hk : ∀ x ∈ Metric.ball x₀ ρ, s2min / 2 ≤ k x)
    (hint : Integrable (fun x => k x ^ 2 / (s2 + ξ2)) ν) :
    s2min ^ 2 * (ν (Metric.ball x₀ ρ)).toReal /
        (4 * (S + ξ2)) ≤
      ∫ x, k x ^ 2 / (s2 + ξ2) ∂ν := by
  classical
  have hden : 0 < s2 + ξ2 := by linarith
  have hDen : 0 < S + ξ2 := by linarith
  let a : ℝ := s2min ^ 2 / (4 * (S + ξ2))
  have hi :
      Integrable ((Metric.ball x₀ ρ).indicator (fun _ : X => a)) ν :=
    (integrable_const a).indicator measurableSet_ball
  have hpoint :
      ∀ x : X,
        (Metric.ball x₀ ρ).indicator (fun _ : X => a) x ≤
          k x ^ 2 / (s2 + ξ2) := by
    intro x
    by_cases hx : x ∈ Metric.ball x₀ ρ
    · rw [Set.indicator_of_mem hx]
      have hkx := hk x hx
      have hp :
          0 ≤ (k x - s2min / 2) * (k x + s2min / 2) :=
        mul_nonneg (by linarith) (by linarith)
      have hsq : 0 ≤ 4 * k x ^ 2 - s2min ^ 2 := by
        nlinarith [hp]
      have hgap :=
        mul_nonneg (sq_nonneg (k x)) (sub_nonneg.mpr hsS)
      have hmain := mul_nonneg hsq (le_of_lt hden)
      dsimp [a]
      apply (div_le_div_iff₀
        (by positivity : 0 < 4 * (S + ξ2)) hden).2
      nlinarith
    · rw [Set.indicator_of_notMem hx]
      exact div_nonneg (sq_nonneg _) (le_of_lt hden)
  have heq :
      (∫ x, (Metric.ball x₀ ρ).indicator (fun _ : X => a) x ∂ν) =
        a * (ν (Metric.ball x₀ ρ)).toReal := by
    rw [integral_indicator measurableSet_ball, integral_const]
    change
      ((ν.restrict (Metric.ball x₀ ρ)) Set.univ).toReal * a =
        a * (ν (Metric.ball x₀ ρ)).toReal
    simp [mul_comm]
  have hle := integral_mono hi hint hpoint
  rw [heq] at hle
  calc
    s2min ^ 2 * (ν (Metric.ball x₀ ρ)).toReal /
        (4 * (S + ξ2)) =
        a * (ν (Metric.ball x₀ ρ)).toReal := by
          dsimp [a]
          ring
    _ ≤ ∫ x, k x ^ 2 / (s2 + ξ2) ∂ν := hle

/--
An integrated nonnegative drift gives bounded partial sums, summable
expectations, and almost-everywhere summability via Tonelli's theorem.
-/
theorem summable_of_drift
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (V Δ : ℕ → Ω → ℝ)
    (hV : ∀ t x, 0 ≤ V t x)
    (hΔ : ∀ t x, 0 ≤ Δ t x)
    (hVint : ∀ t, Integrable (V t) μ)
    (hΔint : ∀ t, Integrable (Δ t) μ)
    (hΔmeas : ∀ t, AEMeasurable (Δ t) μ)
    (hdrift : ∀ t,
      (∫ x, V (t + 1) x ∂μ) =
        (∫ x, V t x ∂μ) - (∫ x, Δ t x ∂μ)) :
    (∀ n : ℕ,
      (∑ t ∈ Finset.range n, ∫ x, Δ t x ∂μ) ≤
        ∫ x, V 0 x ∂μ) ∧
      Summable (fun t => ∫ x, Δ t x ∂μ) ∧
      (∀ᵐ x ∂μ, Summable (fun t => Δ t x)) := by
  have htel : ∀ n : ℕ,
      (∑ t ∈ Finset.range n, ∫ x, Δ t x ∂μ) +
        (∫ x, V n x ∂μ) = ∫ x, V 0 x ∂μ := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
        rw [Finset.sum_range_succ, hdrift n]
        linarith
  have hbound : ∀ n : ℕ,
      (∑ t ∈ Finset.range n, ∫ x, Δ t x ∂μ) ≤
        ∫ x, V 0 x ∂μ := by
    intro n
    have hv : 0 ≤ ∫ x, V n x ∂μ := integral_nonneg (hV n)
    linarith [htel n]
  have hnonneg : ∀ t, 0 ≤ ∫ x, Δ t x ∂μ := by
    intro t
    exact integral_nonneg (hΔ t)
  have hsum : Summable (fun t => ∫ x, Δ t x ∂μ) :=
    summable_of_sum_range_le hnonneg hbound
  have hmeas :
      ∀ t, AEMeasurable (fun x => ENNReal.ofReal (Δ t x)) μ := by
    intro t
    exact (hΔmeas t).ennreal_ofReal
  have htonelli :
      (∫⁻ x, ∑' t, ENNReal.ofReal (Δ t x) ∂μ) =
        ENNReal.ofReal (∑' t, ∫ x, Δ t x ∂μ) := by
    calc
      (∫⁻ x, ∑' t, ENNReal.ofReal (Δ t x) ∂μ) =
          ∑' t, ∫⁻ x, ENNReal.ofReal (Δ t x) ∂μ :=
        lintegral_tsum hmeas
      _ = ∑' t, ENNReal.ofReal (∫ x, Δ t x ∂μ) := by
        apply tsum_congr
        intro t
        exact
          (ofReal_integral_eq_lintegral_ofReal
            (hΔint t) (ae_of_all μ (hΔ t))).symm
      _ = ENNReal.ofReal (∑' t, ∫ x, Δ t x ∂μ) :=
        (ENNReal.ofReal_tsum_of_nonneg hnonneg hsum).symm
  have hfinite :
      (∫⁻ x, ∑' t, ENNReal.ofReal (Δ t x) ∂μ) ≠ ∞ := by
    rw [htonelli]
    exact ENNReal.ofReal_ne_top
  have hae :
      ∀ᵐ x ∂μ, (∑' t, ENNReal.ofReal (Δ t x)) < ∞ :=
    ae_lt_top' (AEMeasurable.ennreal_tsum hmeas) hfinite
  refine ⟨hbound, hsum, ?_⟩
  filter_upwards [hae] with x hx
  have hreal :
      (fun t => (ENNReal.ofReal (Δ t x)).toReal) =
        (fun t => Δ t x) := by
    funext t
    exact ENNReal.toReal_ofReal (hΔ t x)
  rw [← hreal]
  exact ENNReal.summable_toReal (ne_of_lt hx)

theorem lipschitz_assembly
    {X P : Type*}
    (prot : X → P) (d : X → X → ℝ)
    (f : X → ℝ) (B K : ℝ)
    (hd : ∀ x y, 0 ≤ d x y)
    (hdiff : ∀ x y, prot x ≠ prot y → d x y = 1)
    (hB : ∀ x, |f x| ≤ B)
    (hwithin : ∀ x y, prot x = prot y →
      |f x - f y| ≤ K * d x y) :
    ∀ x y, |f x - f y| ≤ max K (2 * B) * d x y := by
  intro x y
  by_cases hsame : prot x = prot y
  · exact le_trans (hwithin x y hsame)
      (mul_le_mul_of_nonneg_right (le_max_left K (2 * B)) (hd x y))
  · rw [hdiff x y hsame, mul_one]
    calc
      |f x - f y| = |f x + -f y| := by rw [sub_eq_add_neg]
      _ ≤ |f x| + |-f y| := abs_add_le _ _
      _ = |f x| + |f y| := by rw [abs_neg]
      _ ≤ 2 * B := by linarith [hB x, hB y]
      _ ≤ max K (2 * B) := le_max_right _ _

end OfmlAppendix
