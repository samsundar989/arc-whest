# Anatomy 2 — predictions, recorded 2026-09-25 ~05:00 UTC, BEFORE any chain run

Panel: 20 **FIT** nets (full rows 0–19), not DEV. Reason: `full1000-N1e9` holds only
the `full` split, which is disjoint from `mini` (card + `data/panels/dev.json`
`mini_full_overlap = 0`), so the dense pre-activation labels do not exist for any
DEV net. Diagnosis only; nothing here is fitted except the κ4 ridge (10 nets fit,
10 evaluate), and DEV stays untouched for selection.

Two facts from reading the code before predicting (they change what the tests can show):
1. The layer-l MEAN reads only the marginal objects (μ_pre, var, D3, D4). C_off,
   D21, D22 and Q31 reach the mean one layer later (through cov_out, the tower and
   g_out). So the one-step table substitutes marginals at l, and off-diagonal
   objects at l−1 with the mean read at l (a two-step branch).
2. The carried κ4 is m·Sym(G⊗I), G = diag(dG) + λ·C_off. Expanding: κ4_iiii = m·dG,
   **κ4_iijj = (m/6)(dG_i + dG_j) — λ·C does NOT enter it**, κ4_iiij = (m/2)·λ·C_ij.
   So "λ·C_off R² on κ4_iijj" tests a form the chain never uses for that slice; the
   chain's λ·C lives in Q31 (and the unlabelled κ4_ijkk).

## Strategist (PLAN rev 13 §3 Step 1)
- covariance substitution recovers < 15% of one-step error; D21 30–50%; κ4 slices 20–40%
- λ·C_off R² on κ4_iijj < 0.5 by layer 10, 4-mode model > 0.8
- old-vs-young split of D21 error: unknown

## Mine
- **Object errors (rel. Frobenius, layer 16 / index 15):** var ≲ 1%; C_off 3–8%;
  D3 10–25%; D21 25–45%; D4 30–60%; D22 > 60% (wrong form); Q31 > 50%.
  Error in the covariance is mostly traceless (> 80% of ‖E‖²).
- **One-step (mean at l, μ labelled):** var 25–40% recovered, D3 20–35%, D4 10–20%,
  all four ≥ 70% (the rest is closure truncation + corrector response).
  Two-step off-diagonal: D21 the largest single off-diagonal share, C_off second.
- **Rolled (substitute at every layer, read layer 16):** cov (var + C_off) 30–50% of
  final error; κ3 (D3 + D21) 30–50%; κ4 (D4 + D22 + Q31) 10–25%; everything but μ 70–85%.
  I expect cov to own more of the PROPAGATED error than the strategist does, because
  ∂E[ReLU]/∂σ = φ(α) is the largest sensitivity at α ≈ 0 and our 24 Sep residual sat there.
- **κ4 model:** pure λ*·C_off on κ4_iijj R² < 0.3 from layer 3 on; the chain's own
  (m/6)(dG_i+dG_j) baseline also poor (< 0.3); 4-mode ridge 0.5–0.8, carried by C∘C.
  λ·C on κ4_iiij: R² 0.3–0.7, falling with depth.
- **Old vs young D21:** the unshared T0 chain's D21 error is 60–80% of v59's, i.e.
  compression owns 20–40% and birth/closure the majority.
