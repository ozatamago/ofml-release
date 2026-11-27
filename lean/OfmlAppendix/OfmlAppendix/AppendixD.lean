import Mathlib

open MeasureTheory Filter
open scoped BigOperators Topology ENNReal

namespace OfmlAppendix

/-!
# Appendix D: self-contained steps

* `approx_maximizer_gap`: the inequality of Lemma D.1.
* `approx_tolerance_tendsto`: the tolerance of Lemma D.1 converges to zero.
* `ciSup_eq_ciSup_dense`, `measurable_ciSup_of_continuous`: the first paragraph
  of the proof of Proposition D.2 (the supremum of a continuous function equals
  the supremum over a countable dense set, hence is measurable).
* `summable_of_drift_le`: the telescoping and Tonelli step of Lemma B.5 with the
  drift identity weakened to an inequality, as used in the proof of
  Proposition D.2.
* `sup_tendsto_zero_of_selection`: the display in the second paragraph of the
  proof of Proposition D.2.
-/

-- Lemma D.1. `δ` is any uniform bound on the approximation error; the
-- uniform norm used in the paper is one such bound. The last summand on the
-- right is the maximization error `η^opt`.
theorem approx_maximizer_gap
    {X : Type*} [Nonempty X]
    (Δ Δhat : X → ℝ) (δ : ℝ) (xhat : X)
    (hbdd : BddAbove (Set.range Δhat))
    (hδ : ∀ x, |Δhat x - Δ x| ≤ δ) :
    (⨆ x, Δ x) - Δ xhat ≤
      2 * δ + ((⨆ x, Δhat x) - Δhat xhat) := by
  have hup : ∀ x, Δ x ≤ (⨆ y, Δhat y) + δ := by
    intro x
    have h1 : Δ x ≤ Δhat x + δ := by
      have := hδ x
      rw [abs_le] at this
      linarith [this.1]
    have h2 : Δhat x ≤ ⨆ y, Δhat y := le_ciSup hbdd x
    linarith
  have hsup : (⨆ x, Δ x) ≤ (⨆ y, Δhat y) + δ := ciSup_le hup
  have hlow : Δhat xhat - δ ≤ Δ xhat := by
    have := hδ xhat
    rw [abs_le] at this
    linarith [this.2]
  linarith

-- Lemma D.1, second part: the tolerance `2 δ_t + η_t^opt` converges to zero.
theorem approx_tolerance_tendsto
    (δ ηopt : ℕ → ℝ)
    (hδ : Tendsto δ atTop (𝓝 0)) (hη : Tendsto ηopt atTop (𝓝 0)) :
    Tendsto (fun t => 2 * δ t + ηopt t) atTop (𝓝 0) := by
  have := (hδ.const_mul 2).add hη
  simpa using this

-- Proof of Proposition D.2, first paragraph: for a continuous function that is
-- bounded above, the supremum over the space equals the supremum over a dense
-- set.
theorem ciSup_eq_ciSup_dense
    {X : Type*} [MetricSpace X] [Nonempty X]
    (Q : Set X) (hQ : Dense Q)
    (g : X → ℝ) (hg : Continuous g) (hbdd : BddAbove (Set.range g)) :
    (⨆ x, g x) = ⨆ q : Q, g q := by
  haveI : Nonempty Q := hQ.nonempty.to_subtype
  have hbddQ : BddAbove (Set.range fun q : Q => g q) := by
    obtain ⟨M, hM⟩ := hbdd
    refine ⟨M, ?_⟩
    rintro _ ⟨q, rfl⟩
    exact hM ⟨q, rfl⟩
  apply le_antisymm
  · apply ciSup_le
    intro x
    have hx : x ∈ closure Q := hQ x
    obtain ⟨u, huQ, hu⟩ := mem_closure_iff_seq_limit.mp hx
    have hgu : Tendsto (fun n => g (u n)) atTop (𝓝 (g x)) :=
      (hg.tendsto x).comp hu
    refine le_of_tendsto' hgu ?_
    intro n
    exact le_ciSup hbddQ ⟨u n, huQ n⟩
  · apply ciSup_le
    intro q
    exact le_ciSup hbdd (q : X)

-- Proof of Proposition D.2, first paragraph: if `g ω` is continuous and bounded
-- above for every sample point `ω`, and `ω ↦ g ω q` is measurable for every `q`
-- in a countable dense set, then `ω ↦ sup_x g ω x` is measurable.
theorem measurable_ciSup_of_continuous
    {Ω X : Type*} [MeasurableSpace Ω] [MetricSpace X] [Nonempty X]
    (Q : Set X) (hQc : Q.Countable) (hQ : Dense Q)
    (g : Ω → X → ℝ)
    (hcont : ∀ ω, Continuous (g ω))
    (hbdd : ∀ ω, BddAbove (Set.range (g ω)))
    (hmeas : ∀ q ∈ Q, Measurable fun ω => g ω q) :
    Measurable fun ω => ⨆ x, g ω x := by
  haveI : Countable Q := hQc.to_subtype
  have heq : (fun ω => ⨆ x, g ω x) = fun ω => ⨆ q : Q, g ω q := by
    funext ω
    exact ciSup_eq_ciSup_dense Q hQ (g ω) (hcont ω) (hbdd ω)
  rw [heq]
  exact Measurable.iSup fun q => hmeas q q.2

-- Lemma B.5 / Proposition D.2: telescoping and Tonelli with the expected
-- one-step decrease bounded below by `Δ_t`.
theorem summable_of_drift_le
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (V Δ : ℕ → Ω → ℝ)
    (hV : ∀ t x, 0 ≤ V t x)
    (hΔ : ∀ t x, 0 ≤ Δ t x)
    (hVint : ∀ t, Integrable (V t) μ)
    (hΔint : ∀ t, Integrable (Δ t) μ)
    (hΔmeas : ∀ t, AEMeasurable (Δ t) μ)
    (hdrift : ∀ t,
      (∫ x, V (t + 1) x ∂μ) ≤
        (∫ x, V t x ∂μ) - (∫ x, Δ t x ∂μ)) :
    (∀ n : ℕ,
      (∑ t ∈ Finset.range n, ∫ x, Δ t x ∂μ) ≤
        ∫ x, V 0 x ∂μ) ∧
      Summable (fun t => ∫ x, Δ t x ∂μ) ∧
      (∀ᵐ x ∂μ, Summable (fun t => Δ t x)) := by
  have htel : ∀ n : ℕ,
      (∑ t ∈ Finset.range n, ∫ x, Δ t x ∂μ) +
        (∫ x, V n x ∂μ) ≤ ∫ x, V 0 x ∂μ := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
        rw [Finset.sum_range_succ]
        linarith [hdrift n]
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

-- Proof of Proposition D.2, second paragraph, at one sample point:
-- `S t` is `sup_x Δ_t(x)`, `a t` is `Δ_t(x_{t+1})`, and `η t` is `η_t^lb`.
theorem sup_tendsto_zero_of_selection
    (S a η : ℕ → ℝ)
    (hsum : Summable a)
    (hη : Tendsto η atTop (𝓝 0))
    (hS0 : ∀ t, 0 ≤ S t) (hsel : ∀ t, S t ≤ a t + η t) :
    Tendsto S atTop (𝓝 0) := by
  have ha0 : Tendsto a atTop (𝓝 0) := hsum.tendsto_atTop_zero
  have hup : Tendsto (fun t => a t + η t) atTop (𝓝 0) := by
    simpa using ha0.add hη
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le
    tendsto_const_nhds hup hS0 hsel

end OfmlAppendix
