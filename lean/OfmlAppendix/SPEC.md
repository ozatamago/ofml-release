# OfmlAppendix: Lean による付録 B〜D の確認（2026-11-25、付録 D は 2026-11-27 に追加）

目的：付録 B〜D の証明の中の、自己完結に述べられる段階を Lean 4 + Mathlib（v4.32.0-rc1）で確かめる。
確率過程・正則条件付き分布・Bochner 積分の全体は形式化しない（著者の指示「Lean は確認のためのみ」）。

## 対象の命題と論文の対応

| Lean の名前 | 論文の箇所 | 内容 |
|---|---|---|
| `OfmlAppendix.affine_predictor_bound` | 補題 B.6 の証明（最適なアフィン予測子との比較） | 二乗可積分な F, Y で Var(Y)>0 のとき、Var(E[F | σ(Y)]) ≥ Cov(F,Y)^2 / Var(Y) |
| `OfmlAppendix.inf_ball_measure_pos` | 補題 B.7 の証明（ν の下界 ν̲(ρ)>0） | コンパクト距離空間上の有限測度で、すべての開球の測度が正なら、任意の ρ>0 で inf_x ν(B(x,ρ)) > 0 |
| `OfmlAppendix.covariance_near_maximizer` | 補題 B.7 の証明（ρ の選び方） | |k(x̃,x') − s²| ≤ C ω(d(x̃,x'))、ω(0)=0、ω は 0 で連続、s² ≥ s̲² > 0 なら、ある ρ>0 があり d(x̃,x')<ρ ⇒ k(x̃,x') ≥ s̲²/2 |
| `OfmlAppendix.integral_lower_bound` | 補題 B.7 の式 (S5) 相当 | 上の結論と s² ≤ S、ξ²>0 から ∫ k(x̃,x')²/(s²+ξ²) dν(x') ≥ s̲⁴ ν(B(x̃,ρ)) / (4(S+ξ²)) |
| `OfmlAppendix.summable_of_drift` | 補題 B.5 の証明（望遠鏡和と Tonelli） | V_t ≥ 0 可積分、Δ_t ≥ 0 可積分、E[V_{t+1}] = E[V_t] − E[Δ_t] なら Σ_t E[Δ_t] ≤ E[V_0] かつ Σ_t Δ_t < ∞ a.s. |
| `OfmlAppendix.lipschitz_assembly` | 付録 C（L_f と ω の構成） | プロトコル内で |f(x)−f(x')| ≤ K d(x,x')、異なるプロトコルで d = 1 なら |f(x)−f(x')| ≤ max{K, 2‖f‖∞} d(x,x') |
| `OfmlAppendix.approx_maximizer_gap` | 補題 D.1 の不等式 | X が空でなく、Δ̂ が上に有界で、すべての x で |Δ̂(x) − Δ(x)| ≤ δ なら、sup Δ − Δ(x̂) ≤ 2δ + (sup Δ̂ − Δ̂(x̂)) |
| `OfmlAppendix.approx_tolerance_tendsto` | 補題 D.1 の後半（許容値の収束） | δ_t → 0、η_t^opt → 0 なら 2δ_t + η_t^opt → 0 |
| `OfmlAppendix.ciSup_eq_ciSup_dense` | 命題 D.2 の証明の第 1 段落 | 距離空間の上の連続で上に有界な関数の上限は、稠密集合の上での上限に等しい |
| `OfmlAppendix.measurable_ciSup_of_continuous` | 命題 D.2 の証明の第 1 段落（上限の可測性） | 各標本点で連続かつ上に有界で、可算稠密集合の各点で標本点について可測なら、上限は標本点について可測 |
| `OfmlAppendix.summable_of_drift_le` | 補題 B.5 の望遠鏡和と Tonelli を、命題 D.2 の証明の第 2 段落で使う形 | `summable_of_drift` の仮定を不等式 E[V_{t+1}] ≤ E[V_t] − E[Δ_t] に弱めたもの。結論は同じ |
| `OfmlAppendix.sup_tendsto_zero_of_selection` | 命題 D.2 の証明の第 2 段落の表示式（1 つの標本点で） | a_t の和が有限、η_t → 0、0 ≤ S_t ≤ a_t + η_t なら S_t → 0。S_t = sup_x Δ̲_t(x)、a_t = Δ̲_t(x_{t+1})、η_t = η_t^lb |

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

## 付録 D の追加（2026-11-27）

- ファイル：`OfmlAppendix/AppendixD.lean`。6 つの命題を加え、`tests/NonVacuous.lean` に具体例、`tests/AxiomAudit.lean` に公理の監査を加えた。
- R1〜R4：すべて通った（tests/check.sh）。12 の命題すべてが propext、Classical.choice、Quot.sound だけに依存する。
- R4 の具体例：`approx_maximizer_gap` は選んだ点が最大化点でない場合、`summable_of_drift_le` は不等式が真に成り立つ場合（等式にならない場合）を使った。
- R5：Claude が論文の主張と照合した。GPT との照合はまだ行っていない。
- 範囲（SCOPE）：
  - `approx_maximizer_gap` の δ は近似誤差の任意の一様な上界である。論文の δ_t（一様ノルム）はその一つなので、論文の主張を含む。論文が暗に使う「Δ̂ の上限が有限である」ことを仮定として明示した。
  - `summable_of_drift_le` は、期待値をとった不等式を仮定として与えた。命題 D.2 の証明が導く条件付きの不等式 V_t − E[V_{t+1} | F_t] ≥ Δ̲_t(x_{t+1})（補題 B.6 の論法を実際の観測 y_{t+1} に適用したもの）から、期待値をとってこの仮定を得る段階は形式化していない。
  - `measurable_ciSup_of_continuous` の可測性は任意の σ 加法族についての主張であり、論文では F_t に適用する。k_t と s_t^2 の版が連続で可測であること（補題 B.4）から Δ̲_t がこの仮定を満たすことは形式化していない。
  - 命題 D.2 の第 3 段落と第 4 段落は、付録 B の補題 B.7 と定理 B.3 の証明を参照するだけなので、新しい命題はない。補題 B.7 の段階は `covariance_near_maximizer`、`integral_lower_bound`、`inf_ball_measure_pos` が扱う。
- 式の番号は、2026-11-27 の原稿で (S5)。
