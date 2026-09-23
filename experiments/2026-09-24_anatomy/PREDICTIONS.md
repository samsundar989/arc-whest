# Step 1 anatomy — predictions recorded BEFORE the run

Written 2026-09-23 ~23:40 UTC, before `tools/anatomy.py` existed and before any
number was read. Handoff 09-24 Step 1 requires this.

## Strategist (docs/STRATEGIST_RESPONSE_2026-09-24.md §3, HANDOFF Step 1)

1. The one-step residual is **< 20%** of the accumulated residual at layer 16.
2. It correlates **> 0.5** with (mu_pre/sigma_pre, D3/sigma^3) at layers >= 8.

## Worker (mine)

1. **One-step / accumulated at layer 16: 10-30% of the MSE.** Reasoning: if each
   of 16 layers contributed an independent, equal-variance increment the ratio
   would be ~6%, and #330001 already measured propagated drift at 5.2x the local
   one-step (=> ~19%). But this one-step probe substitutes only the *mean*
   channel's true value and keeps the chain's own drifted covariance, kappa3 and
   kappa4, so the residual it reports still contains cumulant drift and will NOT
   fall to 1/16. I expect it above the naive 6% and at or below the strategist's
   20% bound. **I agree with the direction of his claim but expect the number to
   sit near its top end.**

2. **Correlations weaker than he predicts.** |corr(one-step residual, mu_pre/sigma_pre)|
   **0.3-0.7** at layers >= 8; |corr(., D3/sigma^3)| **0.2-0.6**. #330001's 0.975
   was measured against a *fitted combination* of Edgeworth features, not a single
   raw feature; a single-feature Pearson correlation is a weaker statistic and I
   expect it to come in under 0.5 for at least one of the two.

3. **MSE by layer:** layer 1 at the label-noise floor (~1e-11 or below; it is a
   closed form, `||W[:,j]||/sqrt(2*pi)`, not a chain output). Growth ratio
   MSE_l/MSE_{l-1} largest in the early layers and **declining** toward a roughly
   geometric 1.15-1.35x by layers 10-16.

4. **The bin decision: no formula fix implicated.** I predict the top-2 bins of
   the strongest axis carry **25-45%** of the total squared error, i.e. below the
   handoff's >50%-with-consistent-sign-at-3-layers trigger, and that Step 1
   therefore hands over to Step 2 rather than to a closure-term fix. Reasoning:
   the final-layer residual is a random projection of layer-15 residuals (measured
   correlation -0.05 with local features, #330001), D2's fit already removed the
   smooth part of the local closure error (-4.5% raw, then 0.00% from 302 further
   evaluations), and the two exact-algebra closure fixes tried this week (exact
   Gaussian bivariate tail -0.23%, adaptive lambda -0.39%) both came in tiny.
   If a single bucket held >50% of the error, D2's 16-parameter fit would very
   likely have found it.

5. **Bias vs RMS:** I expect the per-bin *mean* residual (bias) to be small
   relative to the per-bin RMS at layers >= 12 — i.e. the accumulated error is
   mostly spread, not offset — with a visible systematic sign only at the
   extreme alpha = mu_pre/sigma_pre bins.
