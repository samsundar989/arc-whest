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
