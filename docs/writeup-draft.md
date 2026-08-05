# Write-up draft — PRIVATE, structure only

*Started 2026-08-05. **Nothing here is posted anywhere.** Publication is an endgame
decision, not a default. Do not post to the forum, and do not describe the variance
anatomy, the common mode, or any point-set work publicly before we decide.*

Target: the Algorithmic Contribution prize (Phase 2 write-up due **Sep 26**). The
judging criteria (thread 18041) reward mechanistic insight and honest analysis over
leaderboard position — which suits what we actually have, because our strongest
results this month are **negative results with mechanisms attached**.

**Provisional title:** *What actually limits white-box mean estimation at depth 32:
a cost-variance account, and five ways of not predicting the common mode*

---

## 1. The scoring identity, and why sample count is not a lever

**Claim.** Above the multiplier floor the adjusted score is

    adjusted = MSE × max(0.1, C/B) = (V/N) × (N·c/B) = **V·c/B**

per-sample variance × per-sample cost, with N cancelling identically.

**Evidence to include.** Verification against our own graded submission (V = 1.4893e-6
× 16384 = 2.44e-2, c = 4.273e6, B = 2.72e11 → 3.83e-7 against a graded 3.893e-7, 2%).
The measured flatness: adjusted 7.783e-7 at C/B 0.075 versus 7.771e-7 at C/B 0.15.
The corollary that most of the field appears not to use: **cheapening a sample is
worth exactly as much as de-noising it**, which redirects effort from variance
reduction (crowded) to cost reduction (empty).

**Refinement to state carefully.** The identity holds only for the *marginal* cost.
Per-MLP fixed costs F do not cancel: adjusted = (V·c_marginal + V·F/N)/B, so fixed
scaffolding argues for larger N. We measured this and it lost anyway, because
residual wall time scales with chunk count — worth reporting as a caution.

## 2. Where the error lives: a depth-generated rank-1 mode

**Claim.** ~48% of a well-tuned RQMC estimator's final-layer error is a single
per-MLP scalar — the whole layer scaled by (1+g). It is *generated through depth*,
not inherited from the input.

**Evidence.** corr(g_k, g_L) climbing 0.22 → 0.99 across layers; sd(g) growing
0.00021 → 0.00088. Removing g exactly is worth **1.80–1.90×**.

**The negative result, and its geometry.** Five independent attempts to predict g:

| observable | result |
|---|---|
| closure reference at the final layer | R² 0.094 |
| closure reference at mid-depth | LOO R² 0.155 |
| point-set radial imbalance | LOO R² −0.010 |
| exact closed-form layer 1 | LOO R² −0.037 |
| learned regressor on the gain trajectory (3000 MLPs × 8 draws) | out-of-fold corr 0.012 |

**Mechanism (the contribution).** g = ⟨ŝ − t, t⟩/⟨t,t⟩ is a projection of the error
onto the *unknown truth direction*. Any estimate of it needs a proxy for that
direction, and every available proxy carries more error along it than the quantity
being measured — the closure's own scale error has sd 0.00276 against g's 0.00088,
a 3× contamination. This predicts the measured R² ≈ 0.09 *before* it was measured
(we predicted 0.092, measured 0.094), which is the strongest form of the argument.
Conclusion: the mode is **irreducible variance**, not a correctable bias.

## 3. Why the whole analytic-assist family fails

**Claim.** Gaussian-closure residual at depth 32 is *conditionally-centred noise*,
not smooth bias — so it cannot be corrected, fused, or learned.

**Evidence.** Conditioning on α = μ_pre/σ, the residual's conditional mean is only
0.1–0.5 of its conditional rms; a fitted α-curve removes **3.5%** of the variance
held-out. A learned model on 50,000 minted MLPs (12.8M neuron examples, split by
MLP) reaches held-out **R² = 0.012** — with label noise at 2% of target variance, so
the ceiling was 0.98 and the signal genuinely is not there.

**The unfavourable arithmetic to spell out.** Because the closure's residual
variance is 18.2× the sampling variance, R² converts to score terribly: R² = 0.5
buys only +9.9% fused. The threshold that matters is R² > 0.9, not 0.5 — a point we
have not seen made elsewhere and that would have saved us the experiment.

**Corroboration.** Independently reported by two Phase-1 write-ups (18063, 18085) and
consistent with 18097's finding that the depth-32 failure is *error dynamics* (a
measured 16:1 bias-amplification-to-noise-attenuation exchange rate), not closure
order.

## 4. The cost channel, which is where the room is

**Claim.** Under shape-based FLOP billing, structural sparsity in the *network* can
be converted into smaller dense matrices, and this is worth more than any variance
trick we found.

**Evidence.** flopscope bills matmul as (2K−1)·M·N from shapes alone, so sparsity
earns nothing unless materialised. Predicting dead neurons analytically (α ≤ −2.5),
dropping them from **both** sides of each matmul, and patching their contribution
with the closed-form rectified-Gaussian mean gives **33.7% cheaper samples for +2.6%
MSE = 1.47×**, with the pruned neurons' own MSE at 3.8e-9.

**The sharp optimum, worth explaining.** τ=−3.0 → 1.38×, τ=−2.5 → 1.47×, τ=−2.0 →
1.23× (MSE +35%). The cliff is where the analytic patch stops being accurate.

**Ablation to include.** α beats both Var(h) and the sensitivity-weighted criterion
Var(h)·‖∂z_L/∂h‖² at matched cost — the criterion that looks more principled loses.

## 5. What didn't work, and why it *couldn't*

The section that earns the prize if written honestly.

- **Point-set engineering is bounded by theory.** For discontinuous integrands the
  RQMC rate bonus is O(1/(4d−2)) — 1/1022 at d=256 (He & Wang 2015). We measured
  **α = 1.00** exactly on 100 MLPs. Sobol buys a constant, never a rate. Every
  variant we tried — Owen scrambling, LMS, Korobov lattices, radial stratification,
  antithetic pairing, and direct gradient optimisation of the point matrix — lands
  within ±15% of plain digitally-shifted Sobol.
- **Rate recovery needs the kinks concentrated**, which is why an input rotation is
  the only thing that moved α (0.87 → 1.06). Since x ~ N(0,I) is rotation-invariant,
  the rotation is free and folds into W₁.
- **Moment-matching a QMC set biases it** (RESEARCH_NOTES §2e). Forcing exact
  per-coordinate mean 0 / variance 1 cost **2.35×** with no optimisation at all. A
  QMC set is unbiased because it is the correct quantile transform of an
  equidistributed net, not because its sample moments are exactly (0,1). Corollary
  worth stating: the Sobol net's most extreme point (z = −6.12 in every coordinate)
  is *load-bearing* — deleting it costs 0.734×.
- **A rank-r factor scaffold cannot replace the closure**, and the reason is
  structural: the input covariance I has no low-rank part, so the factor initialises
  at zero and the multiplicative update keeps it there — it degenerates silently to
  diagonal propagation. Representing Wᵀdiag(d)W needs a full-rank re-factorisation
  every layer.

## 6. Method as shipped

Sobol RQMC, digitally shifted from `mlp.seed`, float32 throughout, exact closed-form
first layer, dead-neuron pruning with analytic tail patch, alignment rotation folded
into W₁. Direction numbers embedded so the submission stays a single file.

## 7. Reproducibility

Everything measured against the public dataset's 1e9-sample ground truth (noise
~2e-10) on the **full 100-MLP** split — a policy adopted after three separate
apparent wins evaporated on re-measurement from 40-MLP subsets. That methodological
point deserves its own paragraph: at these effect sizes, subset measurement is not
evidence.

---

## Open items before this is publishable

- [ ] Decide whether to publish at all, and what to withhold (the cost channel is
      our live advantage; the negative results are not).
- [ ] Phase 2 numbers throughout — everything above is Phase 1 (256×32, B=2.72e11).
- [ ] Reproduce the headline table on the Phase 2 shape once known.
- [ ] Figures: error-vs-N with the α=1.00 fit; the α-histogram with the τ cliff;
      corr(g_k, g_L) versus depth.
- [ ] Cite 18053, 18063, 18085, 18097, 18106 properly; He & Wang 2015; Owen on
      scrambled nets.
