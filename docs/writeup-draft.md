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
order. Quote 18097 directly on this — "*the chained estimator attenuates zero-mean
noise while amplifying small coherent biases (measured end-to-end at a ~16:1
exchange rate)*" — because it is the same phenomenon our fusion family died of,
arrived at from a different direction and with a number attached.

### 3b. The one place the analytic side genuinely wins, and why it is unreachable

**The cross-confirmation worth leading with.** 18097 reports that their closure run
*from exact input moments* beats sampling, 1.16e-6 against 1.61e-6. We measured the
same comparison at the last layer, on 100 MLPs against 1e9-sample truth: closure
from reference moments **1.035e-6** against sampling's **1.526e-6**. Two independent
setups, ratios 0.72 and 0.68. So the analytic step is not inaccurate — *it is
starved*.

**And the starvation is structural, which is the contribution.** The last-layer
closure needs only the 256 scalar pre-activation variances, because

    diag(Wᵀ Σ̂ W) = mean_s(z²) − mean_s(z)²

exactly. There is no covariance matrix to estimate and therefore no
Marchenko-Pastur regime to shrink away — we verified the identity to 1.8e-8
relative, and the sample variances are already accurate to **1.13% rms** at
N=16384. The whole error decomposes as **41% irreducible model bias / 57.9%
mean-estimation noise / 0.9% variance-estimation noise**, and the mean-estimation
noise *is* the sampling noise. Replacing the variances with their exact values —
the oracle for every denoising scheme there is, linear or nonlinear — moves the
analytic tail from 0.605× to 0.611× of the sampling baseline.

**The sharpest single fact.** In the α ≥ 3 band, which carries **76.5% of the
final-layer MSE**, the correlation between the analytic and sampled estimates is
**1.000** — with α that large, Φ = 1 and φ = 0, so the rectified-Gaussian formula
*is algebraically the sample mean*. Last-layer Rao-Blackwellisation is a no-op
exactly where the error lives. That is worth stating plainly, because "smooth the
final ReLU" is the standard remedy for a QMC estimator with α = 1.

### 3c. Rank-k propagation: when the accuracy rank equals the break-even rank

The sample-to-sample fluctuation of the activation vector does collapse with depth
— effective rank 25.9 → 8.8 → 4.4 → 2.8 at layers 8/16/24/32, with 96.2% of the
energy in the top 64 by layer 16 — and a **precomputable** basis finds it: the
Gaussian closure's own propagated covariance captures within one point of the data
PCA at every depth, while W Wᵀ eigenvectors capture exactly k/256, i.e. nothing.
That is a genuinely useful structural fact about deep random ReLU nets.

It still cannot be spent. Propagating deviations in a k-dimensional subspace is a
rank-k factorisation of each weight matrix, and under shape-based FLOP billing two
factors cost `(2k−1)·n + (2n−1)·k` against `(2n−1)·n` dense — **break-even at
k = n/2**, exactly 128 here. And 128 is also the rank at which truncation stops
costing accuracy: at layer 16 with k=64 the retained 96% of energy still costs 6.3×
in final-layer MSE, because the dropped variance compounds through every remaining
ReLU. Adding the dropped variance back through the closure formula recovers most of
it, but even an **oracle** correction priced at **zero** tops out at 1.093×
end-to-end. The accuracy rank and the break-even rank are the same number.

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

### 4b. The on-neuron reduction, and exactly what it is worth

**The theorem, stated once and used twice.** For a neuron whose firing index α is
large, Φ(α) → 1 and φ(α) → 0, so the rectified-Gaussian mean collapses to the
plain mean: E[ReLU(z)] = E[z]. Two consequences, and they are different claims:

* *At the scored layer* the neuron needs **no per-sample value at all** — its
  answer is wᵀ(incoming sample mean) plus the exact rectification correction
  σ[φ(α) − αΦ(−α)]. Measured: the correction extends the usable threshold down by
  about one unit of α, and at α > 3 it is unnecessary.
* *At a feed-forward layer* ReLU is the identity, so the neuron is **linear** and
  its path composes with the next layer's weights into a precomputed matrix. We
  verified this survives propagation: linearising every α > 2.5 neuron in the last
  eight layers changes the final-layer MSE by **0.07%**.

The distinction matters because the *other* reading — replace the on-neuron by its
mean — is catastrophic: 48× worse MSE at k=2 and three orders of magnitude by k=8.
Deleting a neuron's fluctuation biases every downstream ReLU, since ReLU is convex
in it. Same mechanism as the projection result in §3c, and worth showing side by
side.

**And it buys 1.02×.** This is the part we would most want a reader to take away,
because it is where a plausible-sounding structural insight meets shape-based
billing and loses. Under `(2K−1)MN` pricing, what costs money is the matmul, and
linearising a neuron removes its ReLU, not its multiply: the contribution still has
to be carried, either explicitly per sample or implicitly through a composed matrix
whose source width grows with every kink activation retained. Re-materialising to
reset that width costs *more* than a dense layer. Pricing the exact optimum of that
trade-off (a DP over which layers re-materialise) gives 1.025× cheaper samples at
0.06% MSE.

The ceiling is arithmetic, not a property of our implementation. In our per-sample
budget the scored layer is **3.67%** of the cost, so making it entirely free is
worth **1.038×**; making the last *eight* layers free would be needed for 1.32×.
**Track T's theorem buys exactly one layer, and one layer is 3.67%.** State it that
way — the general lesson is that a structural reduction is worth the share of the
cost model it actually touches, and under shape-based billing the nonlinearity is
never the expensive part.

*A methodological aside worth one sentence*: the variance rescale κ = 1.363 that
Track T fitted to correct the closure's final-layer variance made things **worse**
when reused inside this correction term. A constant fitted for one role does not
transfer to another, even within the same layer of the same estimator.

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

---

## Appendix F (PRIVATE — not part of any submitted PDF unless explicitly decided)

*Kept with the write-up because it is the same kind of object — a claim with an
arithmetic consequence and a labelled confidence — and because the decision about
what to publish should be made once, at the end, with all of it visible. Default is
**do not publish**. Nothing here has been posted anywhere. Full record and the
scripts that reproduce it: RESEARCH_NOTES §2g.*

**F.1 What the scoring channel actually is (verified).** The public Phase-1 score is
the mean over exactly **50 fixed MLPs** (indices 0–49), and every submission returns
those MLPs' individual `final_layer_mse`, `all_layers_mse`, `adjusted_final_layer_score`
and a 32-long `per_layer_mse` in `results.public_scores`. The mean of the 50
reproduces all three reported aggregates bit-for-bit, and their min/max reproduce the
reported best/worst MLP. The suite is fixed: the 100 `mlp_name`s are identical and
identically ordered across four of our submissions spanning two days. It is not the
public dataset (1/100 name overlap with `mini`, 0/100 with `full`).

**F.2 The consequence (derived).** For a scored MLP j, a submission of prediction p
returns ‖p − t_j‖²/256 to 17 significant digits, so differencing two submissions
gives a *linear* equation in t_j's 256 unknowns — one per scored MLP per pair.
**257 submissions determine every scored reference vector exactly.**

**F.3 Why the wall-time bypass is not the explanation (derived, with its
sensitivity).** Reaching 5.6e-9 at the multiplier floor needs MSE 5.6e-8, i.e.
~2.4e5 samples ⇒ 7.5e11 real FLOPs ⇒ 2.75× the budget. Billed only as λ·wall, that
stays inside 0.1B only at sustained ≥2.8 TFLOP/s f32 on a 16-vCPU grader; realistic
AVX-512 throughput is 0.1–0.5, where the bypass is worth ~4× and takes the best
published method to ~4e-8. The conclusion depends on a throughput figure we have not
measured, and the appendix should say so.

**F.4 Timeline (verified).** Phase-1 record improvements ran 2.13e-6 → 4.87e-8 in
~25 gradual steps from Jun 19 to Aug 5 06:04Z, then **4.87e-8 → 5.6e-9 in one step
at 2026-08-05T17:21Z**. The entry has 917 submissions against a top-30 median of 153.
Implied V·c = 1.5e3: 28× below the best published method and 49× below ours.

**F.5 What does not fit.** That entry ran at multiplier 0.98, leaving the 10× floor
discount unused — which a party that already knows the references has no reason to
do. Recorded rather than explained away.

**F.6 Two of our own hypotheses died here**, and the appendix is more useful for
saying so than for the accusation it declines to make:
* *"`sampling_mse` is reference label noise, so nothing honest can beat ≈6.47e-8."*
  Refuted — 89 of 200 board entries show a final-layer MSE below that value, in a
  smooth continuum with no gap there.
* *"917 submissions cannot leak enough to matter."* Wrong, and wrong because it
  reasoned about the report schema instead of reading it.

**F.7 Our position.** We do not use this channel; it is not estimation. The
methodological point that *is* publishable, and is worth publishing on its own, is
narrower and blames no one: **a leaderboard that scores a fixed set and returns
per-item errors is a fixed-design regression problem, not an estimation benchmark,
and the only defence is the private re-run on fresh instances.** That sentence can
stand in a public write-up without any of F.1–F.6 behind it.

## 8. Robustness, and why the distribution is the deliverable

The competition's final ranking comes from re-running one designated submission on
MLPs it has never seen. That makes the *distribution* of per-MLP score the object
to engineer — a method with a good mean and a fragile tail loses a fresh re-run,
and a single failed MLP scores as zeros. We think this framing is under-discussed
in the write-ups so far, all of which (ours included, until now) report a mean.

**Measured**, on 1200 freshly minted He MLPs with 4.19M-sample MC labels: mean
adjusted 3.01e-7, median 2.19e-7, p90 5.69e-7, p99 1.65e-6, **max/median 12.9×**.
The worst 1% of networks carry 6.9% of the total score. Bootstrapping the re-run
score by set size R: at **R=10 the p90 is 50% above the median and the p99 is
2.05×**; at R=100 the p90 spread is 13%; at R=1000, 4%. A competition that
re-runs on ten fresh MLPs is measuring draw luck as much as method.

**And the skew is mostly the metric, not the method** — the finding we would lead
with. The score is an *unnormalised* MSE, and He-init draws vary about 5× in
activation scale: `corr(log per-MLP MSE, log mean(truth²)) = +0.548`, and
normalising by that scale cuts p99/median from 6.87× to 4.06×. The five worst
networks have mean(truth²) of 1.5–3.7 against a median of 0.70. So the tail is
largely inherited from the target, shared by every competitor, and not something
estimator engineering can remove.

**Robustness of the machinery, separately**: across 1800 fresh and deliberately
deformed networks (inflated row norms, a collapsed layer, an amplified layer, a
mean-shifted layer) there were no non-finite outputs, the alive count after
pruning never fell below 117/256, the degenerate-layer fallback never fired, and
the analytic patch stayed bounded at 6.3e-3 with α reaching ±12.4. The contract
suite on the shipped file passes 24/24, including an identically-zero layer
(σ → 0 in α = μ/σ), 10⁴ dynamic range, seed 2⁶³−1, widths 64–256, depths 2–64,
budget 0, and input-array immutability.

**The design point worth arguing**: our pruning rule fires on a *predicted*
quantity (the closure's α), so its failure mode on an atypical network is
misclassification, not a crash — and the analytic patch that replaces a pruned
neuron is a closed form that cannot diverge. Rules that depend on sampled
quantities do not have that property.

## Open items before this is publishable

- [ ] Decide whether to publish at all, and what to withhold. Appendix F stays out
      by default and until the organizers have had a chance to respond to the
      disclosure; §7's one-sentence methodological version can stand alone.
- [ ] Phase 2 numbers throughout — everything above is Phase 1 (256×32, B=2.72e11).
- [ ] Reproduce the headline table on the Phase 2 shape once known.
- [ ] Figures, in priority order: (1) the fresh-MLP score distribution with the
      bootstrap-by-R inset — it is the most under-reported thing we have;
      (2) per-MLP MSE against mean(truth²) on log axes, showing the +0.55 slope
      that explains the tail; (3) the α-histogram by depth showing bimodality
      emerging (0.007 → 0.45 above α=3 from layer 4 to layer 31), with the τ cliff
      marked; (4) error-vs-N with the α=1.00 fit; (5) corr(g_k, g_L) versus depth.
- [ ] One table that carries the whole cost argument: each structural idea against
      the share of the per-sample cost model it touches, and the resulting ceiling
      (pruning 33.7% → 1.47×; scored-layer on-neurons 3.67% → 1.038×; rank-k
      propagation, break-even k = n/2 → 1.09× with a free oracle correction).
- [ ] Cite 18053, 18063, 18085, 18097, 18106 properly; He & Wang 2015; Owen on
      scrambled nets. For §8, Blum & Hardt (arXiv:1502.04585) and Dwork et al.
      (arXiv:1506.02629) on why a fixed evaluation set with per-item feedback is a
      different object from a held-out test set.
- [ ] Decide whether to report the eigensolver realisation effect: the alignment
      basis is only determined in its leading directions, so LAPACK and cuSOLVER
      give different realised errors from identical code. It is a real caveat for
      anyone reproducing QMC results and we have not seen it stated.
