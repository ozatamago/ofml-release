# OfmlAppendix: Lean による付録 A〜D の確認（2026-11-25）

目的：付録 B〜C の証明の中の、自己完結に述べられる段階を Lean 4 + Mathlib（v4.32.0-rc1）で確かめる。
確率過程・正則条件付き分布・Bochner 積分の全体は形式化しない（著者の指示「Lean は確認のためのみ」）。

## 対象の命題と論文の対応

| Lean の名前 | 論文の箇所 | 内容 |
|---|---|---|
| `OfmlAppendix.affine_predictor_bound` | 補題 B.6 の証明（最適なアフィン予測子との比較） | 二乗可積分な F, Y で Var(Y)>0 のとき、Var(E[F | σ(Y)]) ≥ Cov(F,Y)^2 / Var(Y) |
| `OfmlAppendix.inf_ball_measure_pos` | 補題 B.7 の証明（ν の下界 ν̲(ρ)>0） | コンパクト距離空間上の有限測度で、すべての開球の測度が正なら、任意の ρ>0 で inf_x ν(B(x,ρ)) > 0 |
| `OfmlAppendix.covariance_near_maximizer` | 補題 B.7 の証明（ρ の選び方） | |k(x̃,x') − s²| ≤ C ω(d(x̃,x'))、ω(0)=0、ω は 0 で連続、s² ≥ s̲² > 0 なら、ある ρ>0 があり d(x̃,x')<ρ ⇒ k(x̃,x') ≥ s̲²/2 |
| `OfmlAppendix.integral_lower_bound` | 補題 B.7 の式 (S9) 相当 | 上の結論と s² ≤ S、ξ²>0 から ∫ k(x̃,x')²/(s²+ξ²) dν(x') ≥ s̲⁴ ν(B(x̃,ρ)) / (4(S+ξ²)) |
| `OfmlAppendix.summable_of_drift` | 補題 B.5 の証明（望遠鏡和と Tonelli） | V_t ≥ 0 可積分、Δ_t ≥ 0 可積分、E[V_{t+1}] = E[V_t] − E[Δ_t] なら Σ_t E[Δ_t] ≤ E[V_0] かつ Σ_t Δ_t < ∞ a.s. |
| `OfmlAppendix.lipschitz_assembly` | 付録 C（L_f と ω の構成） | プロトコル内で |f(x)−f(x')| ≤ K d(x,x')、異なるプロトコルで d = 1 なら |f(x)−f(x')| ≤ max{K, 2‖f‖∞} d(x,x') |

## 満たすべき条件（tests/check.sh が確かめる）

- R1 `lake build` が通る。
- R2 `sorry`、`admit`、`axiom` 宣言、`native_decide`、`implemented_by` を使わない。
- R3 `#print axioms` で、各命題が依存する公理が propext、Classical.choice、Quot.sound だけである。
- R4 各命題について、仮定を満たす具体例（tests/NonVacuous.lean）がコンパイルできる。仮定が矛盾して命題が空虚に真になることを防ぐ。
- R5 命題の文は、上の表の論文の主張と同じ量化子と定数をもつ（人が読んで確かめる。GPT にも照合させる）。

## 結果（2026-11-25、GPT と 4 往復）

- R1〜R4 はすべて通った（tests/check.sh）。6 つの命題は、propext、Classical.choice、Quot.sound だけに依存する。
- R5（命題の文の照合）：GPT と Claude がそれぞれ照合した。補題 B.7 の半径 ρ を、部分列のすべての j に共通にとれる形（ρ を点と関数より前に量化）に直した。
- 範囲（SCOPE）：
  - `affine_predictor_bound` は、補題 B.6 の証明の射影の不等式だけを形式化した。条件付き期待値を L²(σ(y^x)) への直交射影と同一視すること、共分散と分散を内積とノルムと同一視することは形式化していない。
  - `summable_of_drift` は、期待値の等式 E[V_{t+1}] = E[V_t] − E[Δ_t] を仮定として与えた。条件付き全分散の公式からこの等式を導く段階は形式化していない。Tonelli の段階（ほとんど確実な総和の有限性）は形式化した。
  - `integral_lower_bound` は球の測度 ν(B(x̃,ρ)) を使う。`inf_ball_measure_pos` と合わせると、論文の ν̲(ρ) > 0 による一様な下界になる。
