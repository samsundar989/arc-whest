# Step 2 (HANDOFF 09-27) — refit ceilings: predictions before any run (2026-09-26 ~21:20 UTC)

Twin (`tools/twin/twin.py --subst X`), candidate v62_s1 with the S3 parameter set (= v67/v68
accuracy), stage S3 leaves (corrector pages, λ/θ/feed, term gains, q31). TRAIN = FIT rows 0–79,
HOLDOUT-20 = FIT rows 80–99 (the only full-split nets with exact labels are rows 0–99).
X substituted exact at every layer ≥ 1 (1-based 2..16) in train AND holdout forward passes:
cov = var + C11; k3 = D3 + D21; k4 = dG + D22 + Q31; all = all seven (NOT μ — substituting
μ_pre at every layer makes the final mean exact by construction). Refit: lr 3e-4, 220 steps of
8 nets (S2t length ≈ 224), eval every 20, patience 3. Control = same refit, nothing substituted.
ceiling_X = 1 − MSE_X / MSE_ctrl (HOLDOUT-20 final layer). Unrefit share = each run's [init] read
vs the control's [init] (Anatomy 2, v59 on FIT 0–19: cov 42, κ3 51, κ4 15, all-but-μ 80%).

- Strategist: cov ≥ 45%, κ3 ≥ 45%, κ4 30–45%.
- Mine: cov 45–55%; κ3 50–62%; κ4 12–25% (refit can undo the κ4-compensation, but Anatomy's
  "exact D4/D22 make it worse" says the channel's own value is small); all 82–92%.
  Caveat on the control: refitting on 80 nets with 1e3 parameters/layer can overfit; the control
  absorbs the same effect, so ratios are still comparable.
