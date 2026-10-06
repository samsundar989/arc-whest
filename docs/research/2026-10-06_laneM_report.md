# Lane M report — ARC's dropped same-order diagrams (draft, 2026-10-06)

**Question.** ARC's factored k_max = 3 AUGMENT propagator warns that it drops diagrams "with
the same Θ(n^-3) squared size as the extra terms that are kept". How much accuracy do those
drops cost, which diagrams matter, and does the loss persist as width grows?

**Answer (M2).** About **16 % of final-layer MSE**, flat in width from n = 384 to 768, and it
is entirely the **κ3 × covariance-edge** family. The covariance triangle contributes nothing.
By the handoff's M3 rule this is a **GO**. Whether it transfers to v73 is a separate question
(§4): v73 does not carry the channel where most of the gain lands.

## 1. Setup

- Reference: `external/mlp_cumulant_propagation` (ARC, commit 93d091a), `.venv-kprop`, torch
  GPU, float32, TF32 off. Our nets: He N(0, 2/n) on every layer, 16 × [W, ReLU], no biases.
  ARC's `MLP` gives layer 0 a weight variance of 1/n, so `init_scale` is overridden to 2 on
  every layer and outputs are read at `act0..act15`.
- Nets: 8 per width from reserved seeds (`tools/fresh/gen.py lane_m_seed`, whestbench's
  own mint). Labels: our GPU MC, N = 1e8 (`tools/fresh/mc.py`); every MSE below has the
  label floor (≈ 8e-10) subtracted.
- Driver: `experiments/2026-10-06_laneM/m2_variants.py`; table: `analyze_m2.py` →
  `m2_table.md`, `m2_summary.json`; raw: `m2_w{256,384,512,768}.json`.

## 2. M0 — what factoring drops

`kprop_harmonic.factored_keeps_term` rejects exactly **4 of the 91** AUGMENT terms
(int_part, vec_part, multiplicity):

| tag | term | meaning |
|---|---|---|
| T1 | (1,1,1) ((0,1,1),(1,0,1),(1,1,0)) ×1 | covariance triangle into κ3's all-distinct slice |
| E3 | (1,1,1) ((0,1,1),(1,1,1)) ×3 | κ3 × edge into κ3's all-distinct slice |
| E21a | (2,1,1) ((0,1,1),(1,1,1)) ×1 | κ3 × edge on the two single indices, into κ4 |
| E21b | (2,1,1) ((1,0,1),(1,1,1)) ×2 | κ3 × edge touching the repeated index, into κ4 |

Variant (d0) — unfactored AUGMENT restricted to the 87 kept terms, by the same
`get_all_terms_iso` filter ARC's `tests/test_factor_k3.py` uses — reproduces factored
AUGMENT (b) to float32 rounding (max |Δ| 1.7e-6 on outputs ~0.6) at every width.

## 3. M2 — results

Final-layer MSE (floor-free), mean of 8 nets:

| n | a: factored SIMPLE | b: factored AUGMENT | c: unfactored AUGMENT | d+E (all κ3×edge) | d+E21b alone |
|---|---|---|---|---|---|
| 256 | 2.10e-6 | 1.65e-6 | 1.21e-6 | 1.22e-6 | 1.27e-6 |
| 384 | 8.24e-7 | 4.52e-7 | 3.68e-7 | 3.69e-7 | 3.79e-7 |
| 512 | 3.51e-7 | 1.76e-7 | 1.49e-7 | 1.49e-7 | 1.53e-7 |
| 768 | 9.08e-8 | 4.57e-8 | 3.81e-8 | 3.79e-8 | 3.96e-8 |

Key quantities (paired over 8 nets, studentised t 95 %):

| n | G_drop = (b − c)/b, final | wins | G_drop, all layers | G_aug = (a − b)/a, final |
|---|---|---|---|---|
| 256 | +26.6 % [16.5, 36.6] | 8/8 | +19.4 % | +21.4 % |
| 384 | +18.6 % [11.8, 25.5] | 8/8 | +16.1 % | +45.1 % |
| 512 | +15.6 % [7.8, 23.5] | 8/8 | +12.1 % | +49.7 % |
| 768 | +16.8 % [11.3, 22.3] | 8/8 | +14.4 % | +49.6 % |

Per family at n = 512 (each added back ALONE to the kept set):

| family | Δ final MSE vs b | Δ all layers |
|---|---|---|
| T1 triangle | +0.2 % [−0.5, +0.8] | 0.0 % |
| E3 | **−24.3 %** (worse, 0/8) | −16.4 % |
| E21a | **−14.1 %** (worse, 0/8) | −8.8 % |
| E21b | +13.1 % [4.9, 21.2] | +10.0 % |
| E = E3 + E21a + E21b | +15.6 % [8.0, 23.2] | +12.1 % |

- The κ3 × edge family as a SET reproduces the full gap (d+E within 0.2 pp of c at every
  width). Its members are not separately addable: E3 and E21a alone make the estimate worse
  (they cancel against each other and against E21b). E21b alone gives ~80 % of the gain
  (13.5 of 16.8 pp at n = 768).
- The gap grows with depth (per-layer (b − c)/b at n = 512: 0 % at layer 1 → 16 % at layer
  16), as a compounding propagation error should.
- n = 1024 unfactored needs ~52 GiB (peak 22 GiB at n = 768, scaling n³) — not run.

**Predictions vs outcome.** Strategist: G_drop 5–25 % at n = 512, κ3 × edge dominant — both
right. Worker: 3 % — wrong by ~5×, extrapolated from the rms output shift of one unlabelled
depth-4 net (a squared shift is not an MSE change when the shift is aligned with the error).
Worker's trend call after n = 256/512 (power-law decline to ~11 % at 768): wrong; it is flat.

## 4. M3 — decision and the transfer problem

**GO by the handoff rule** (G_drop ≥ 10 % at n = 512; trend flat 384 → 768).

The transfer to v73 is NOT automatic, and the reason is structural:

- In ARC, E21b lands in the **κ4 core** — the n×n matrix A of κ4 = Sym(A ⊗ I) — through
  the same projection as the KEPT (2,1,1) κ3 term (`factor_k3.py:499-521`):
  kept `A_jk += Σ_r (Σ_i w2_i a_ir) b_jr c_kr`; E21b `A_jk += Σ_r b_jr c_kr (Σ_i W22_i a_ir C_ik)`.
  With C_off ≈ Σ_m s_m v_m v_mᵀ the extra factor folds into the right operand elementwise
  (`c'_kr = c_kr(α_r + Σ_m s_m v_mk h_mr)`), so in ARC's representation it is nearly free.
- **v73 does not carry ARC's κ4 core.** It carries a surrogate κ4 = m·Sym(G ⊗ I) with
  G = diag(dG) + λ_l·C_off, λ fitted per layer, and only dG is transported
  (`estimator_k3_v73_gb.py:3924-3940`, Q31 at 595-604). The kept (2,1,1) κ3→κ4 projection is
  not computed either. λ·C_off may already absorb part of what E21b supplies.
- So the prototype must ADD an off-diagonal κ4 core channel to v73: compute the E21b
  (and, since the family only works as a set, E21a/E3) contribution per layer, transport it
  (Wᵀ A W: two n³ matmuls a layer if dense, ≈ +15 % FLOPs; low rank much less), and feed
  it to the consumers (Q31, the hub routes, D22's off-diagonal).

**Next (cheap, running).** ARC a and b at n = 1024 on 16 organizer DEV nets, against v73's
cached per-net DEV MSE (`m2_mini1024.json`): if v73 already beats ARC-b by more than ~16 %,
its surrogates likely capture much of the family, and the prototype's expected gain shrinks
accordingly.

## 5. Transfer check and pricing (2026-10-06, later)

**ARC at full width on organizer DEV nets** (16 mini nets, n = 1024, organizer N = 1e9 labels;
`m2_mini1024.json`): factored SIMPLE (a) 3.93e-8, factored AUGMENT (b) **1.79e-8**, v73 2.32e-8
(cached DEV preds). b beats v73 raw on 16/16 nets (v73 is 29 % ± 6 % worse); c ≈ b × 0.835 ≈
1.50e-8. This is known in kind (CLAIMS_LEDGER #39, the d8b README's "k3af teacher 1.8023e-8 on
mini-8"): AUGMENT's 8n-column lineages cost ~8× and lose on SCORE. It does mean v73 sits
between SIMPLE and AUGMENT on accuracy, so the dropped family is not obviously "already
absorbed" — the gap to b alone is 23 %.

**Where E21b would land in v73 and what it costs.** ARC zeroes κ3's repeated entries before
the (2,1,1) → κ4-core projection (`factor_k3.py:502-505`), so E21b writes only the
OFF-DIAGONAL κ4 core B_jk ∝ W11_j W21_k Σ_i W22_i C_ik κ3_ijk (j ≠ k) — the part v73
replaces with the memoryless surrogate λ_l·C_off (its per-neuron core g_out, `:4229-4233`,
is the r = 1 projection's diagonal and has no K211 term to extend). Computing B from v73's
tower needs κ3 contracted with a vector (rank-1 C = s v vᵀ → c = W22 ⊙ v): for every carried
lineage that is `M diag(Aᵀc) Mᵀ`-type work, i.e. at least one (n, kn) @ (kn, n) product per
layer. At k n ≈ 5n that is ≈ 1.6e11 FLOPs over 15 layers before Strassen, **≈ +25–35 % of
v73's 4.72e11** — and the E family only works as a set, so E3/E21a add similar work. Under
PLAN §1's marginal rule +30 % FLOPs needs ≥ 23 % lower raw; the full ARC G_drop (16 %) would
not pay even if it transferred completely. **The exact form fails the cost bar.**

**What could pay.** The youngest lineage has M = I, so its contraction with c is O(n²)
elementwise work — essentially free. Next measurement (reference-side, before any twin
work): implement E21b in ARC's FACTORED path (validated against unfactored d+E21b), then
restrict it to the most recently born factor columns, and read what share of the 13–16 pp
the young-only version keeps. GO to a v73 twin prototype only if young-only keeps ≥ ~1/3
(≥ 5 % raw) — the handoff's ≥ 5 % raw at ≤ +5 % FLOPs bar.

## 6. Young-lineage screen and the M3 decision: **CLOSE**

Factored E21b, written as an add-on to ARC's factored AUGMENT κ4 core
(`m3_young.py`), reproduces the unfactored d+E21b to **2.3e-13 relative** at coefficient
24/(2n+8) = ARC's 12/(2n+8) × the term's multiplicity 2 (`m3_calibrate.json`). Restricting
it to the κ3 columns born at the last g layers (8 nets each, `m3_young_w{512,768}.json`):

| E21b reads | n = 512 gain vs b | wins | n = 768 gain vs b | wins |
|---|---|---|---|---|
| all columns | +13.1 % | 8/8 | +13.5 % | 8/8 |
| births of last 1 layer (M = I: O(n²) in v73) | +2.8 % | 4/8 | +2.5 % | 6/8 |
| last 2 layers | +6.5 % | 8/8 | +5.6 % | 7/8 |
| last 3 layers | +8.7 % | 7/8 | +8.0 % | 8/8 |

Worker prediction was 40 / 65 / 80 % of the gain for g = 1/2/3; measured ≈ 20 / 45 / 63 %.

**Pricing in v73.** Whatever form is used, v73 has no off-diagonal κ4-core channel to put B
in (λ·C_off is memoryless; only the diagonal is transported). Consuming B at the next layer
needs Wᵀ B W: two n³ products per layer ≈ 6.4e10 FLOPs ≈ **+14 %** of 4.72e11. Then:

- g = 1: B is O(n²), total ≈ +14 % FLOPs for ≤ 2.5–2.8 % raw (marginal rule needs ≥ 12 %).
- g = 3: two extra n³ lineage contractions per layer (+14 %) + transport ≈ +27 % for ≤ 8–9 %
  raw (needs ≥ 21 %).
- all columns: ≥ +40 % for ≤ 13–16 %.

None reaches the handoff bar (≥ 5 % raw at ≤ +5 % FLOPs), and these are ARC-chain gains,
an optimistic ceiling for v73. **M3 = CLOSE.** The GO-by-rule result stands as a
measurement for the write-up: ARC's own warning is quantitatively right (≈ 16 % of final-
layer MSE, flat in width, all of it κ3 × edge, the triangle worthless), and the exact
factored form of the dominant term (E21b) is now validated.

What would reopen it: a low-rank transport of B (rank ≤ ~64 would make the channel ≈ +1 %),
which this lane has not measured; only worth testing if a future chain carries an
off-diagonal κ4 core for another reason.

## 7. M4 — neighbor-feature corrector screen: **CLOSED**

Twin arm `nbr` (`tools/twin/twin.py`, lane-m): a zero-initialised additive head on μ_out
beside the shipped corrector, 56 detached per-net-normalised features per neuron (own α, Φ,
φ, γ3; the top-8 |ρ| partners' signed ρ, α, Φ, φ, γ3; similarity-kernel means
exp(−β(1−ρ²)), β ∈ {8, 32}, and log kernel mass; ridge (C⁺α)_i and (R⁺1)_i), MLP 56-32-1;
only the head trains, from twin-v73 (S2t3 step-736 leaves + G1bg gates). One epoch on TRAIN
900, VAL every 28 steps. VAL raw: init 2.213009e-08 (equals v73's recorded VAL) → +0.79 %,
+0.26 %, +0.10 %; early stop at step 84. Never beat init: **< 2 % → killed**. Deviation from
the handoff: the kernel uses similarity exp(−β(1−ρ²)), because exp(−βρ²) weights the
uncorrelated bulk. Feature blocks were not ablated separately: the verdict is on the set.
