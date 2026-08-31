# Write-up draft — PRIVATE, structure only

> **STALE DRAFT — DO NOT SUBMIT OR QUOTE.** This manuscript predates Phase 2 and
> still contains obsolete rules, dates, leaderboard values and method claims,
> including “one designated submission” in §8. Correcting or adopting any of those
> claims is STRATEGIC and requires a preregistered claim review plus Samuel's
> prediction. Current authority is `CURRENT_STATE.md`, `GAME_PLAN.md`, and
> `PHASE2_FRONTIER_AUDIT_2026-08-31.md`.

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

## Abstract  *(draft 2026-08-06)*

We give a quantitative account of what limits estimation of the expected
activations of a wide random ReLU MLP under a compute budget, at the depth where
the published methods stop working. Our organising result is an identity: above the
scoring floor the adjusted score factors as `V·c/B`, per-sample variance times
per-sample cost, with the sample count cancelling exactly. Sample count is
therefore not a design variable, and de-noising a sample and cheapening a sample
are worth precisely the same.

Read through that identity, the field's remaining levers are small and we can say
how small. On the variance side we report five independent attempts to predict the
dominant error mode — a single per-network scalar carrying 48 % of the squared
error and worth 1.9× if removed exactly — that fail at held-out R² between −0.04
and 0.15, and we give the reason: the target is a projection onto the unknown truth
direction, and every available proxy for that direction carries more error along it
than the target itself. We predicted R² ≈ 0.09 before measuring 0.094. On the rate
side, randomised QMC cannot help: for discontinuous integrands the rate bonus is
1/(4d−2), which at d = 256 is 0.001, and we measure the convergence exponent at
1.00 as theory requires.

On the cost side we prove and price a reduction that does work — a confidently-on
ReLU is the identity, so such a neuron needs no per-sample value at the scored
layer and is linear and composable at a feed-forward layer, verified to within
0.07 % over eight layers — and then show it cannot pay, because shape-based FLOP
billing charges for the multiply and not the nonlinearity. The scored layer is
3.67 % of the per-sample cost, so eliminating it entirely is worth 1.038×. The same
arithmetic bounds rank-k propagation: the rank at which truncation stops losing
accuracy is the rank at which factorisation stops saving compute, and both equal
half the width.

We also report the evaluation-methodology findings, which we believe generalise
beyond this task. Ground-truth generation must be verified bit-for-bit, not
distributionally: our own generator matched the evaluator's distribution to 3e-4
while sharing none of its values, and coupled the estimator's random seed to the
weights in a way the evaluator does not. A fixed evaluation set that returns
per-item feedback is not a held-out test set, and we give the arithmetic for how
many queries determine it outright. We describe the three-tier data protocol,
sealed holdout, and invariant suite we adopted in response, including a permutation
-equivariance check that verifies the estimator computes the right function rather
than merely a finite one.

Our final estimator — dead-neuron pruning with a closed-form analytic patch, an
alignment rotation folded into the first layer, and a scrambled Sobol point set —
reaches an adjusted score of 2.72e-7 on the public evaluation. We argue the honest
sampling frontier for this task is near 2.7e-7 and explain what would have to
change for it to move.

---

## 0. Introduction

The task is deceptively simple to state. Draw a ReLU MLP of width 256 and depth 32
with He-initialised weights and no biases. For inputs x ~ N(0, I), predict the
expected post-ReLU activation of every neuron. You are scored on the final layer's
mean squared error, multiplied by the fraction of a fixed compute budget you spend.
Everything about the data-generating process is known: there is no distribution
shift, no label ambiguity, no missing information. The only question is how
efficiently you can turn compute into accuracy.

That transparency is what makes the problem worth studying. When a method fails
here it fails for a reason you can write down, and this paper is mostly a
collection of such reasons. Three shape the argument.

**First, the scoring rule is a factorisation, and it removes a degree of freedom
most participants spend effort on.** Because the multiplier is proportional to
compute and the error is inversely proportional to sample count, the sample count
cancels: the score depends only on the variance of a single sample and its cost.
Choosing "how many samples to draw" is not a lever at all above the floor, and
every real improvement is either a cheaper sample or a quieter one. Section 1.

**Second, the error at depth 32 is not where the literature looks for it.** The
dominant component is not per-neuron noise but a single scalar per network — the
whole output layer scaled by 1 + g — carrying 48 % of the squared error. It looks
exactly like a bias, it is stable enough to be worth 1.9× if removed, and it is not
predictable from anything we can observe. We give five failed predictors and the
geometric argument for why a sixth will also fail. Sections 2 and 3.

**Third, the analytic and sampling estimators fail in ways that do not compose.**
Gaussian closure at depth 32 is 18× worse than sampling and its residual is not a
learnable function of the network (held-out R² = 0.012 over 50,000 networks). Its
errors do not cancel against sampling's either: the chain attenuates zero-mean
noise while amplifying coherent bias, which we and one other group measure
independently. Section 3.

The cost channel is the one with room left, and Section 4 is our positive result:
a proof that confidently-on neurons need no sampling, an exact price for it, and
the demonstration that the price is set by matrix shapes rather than by
nonlinearities — which caps the whole structure-aware family well below what its
reception suggests.

Section 7 is about measurement rather than estimation, and is the part we would
most like to see adopted. Working on a benchmark whose data are *generated* and
whose evaluation set returns *per-item feedback* creates two failure modes that
ordinary machine-learning hygiene does not cover, and we walked into the first one.
We describe both and the protocol we now use.

We report negative results at length and deliberately. Every one below is
accompanied by a mechanism and, where we could construct one, an oracle bound —
the value of the exact quantity the method was trying to estimate, priced at zero.
An oracle bound converts "we could not make this work" into "this cannot work,"
and that is the difference between a lab note and a result.

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
MSE = 1.47×**. The original 3.8e-9 estimate for the pruned values themselves was
withdrawn after the canonical-mint audit measured 1.043e-8; the independent
end-to-end sweep nevertheless selects the same sharp optimum, τ=-2.5.

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

### The table that carries the argument

Every structural idea we tried, against the share of the per-sample cost model it
touches and the ceiling that share implies. This is the section's centerpiece: the
ceiling column is computable *before* implementing anything, and in every case it
predicted the outcome.

| idea | what it removes | share of per-sample cost | ceiling | measured |
|---|---|---|---|---|
| **dead-neuron pruning** (τ=−2.5) | whole rows *and* columns → smaller dense matmuls | 33.7% of the matmul | 1.51× | **1.47×** |
| on-neurons at the scored layer | that layer's per-sample values entirely | 3.67% | 1.038× | 1.02× |
| on-neurons linearised, last *k* layers | the ReLU only — never the multiply | 0% extra | see below | 1.025× at k=8 |
| rank-*k* deviation propagation | width, via a rank-*k* weight factorisation | break-even at k = n/2 | 1.09× (free oracle correction) | not built |
| moment denoising at the tail (Track T) | nothing in the cost model | 0% | — | 1.0001× |

The third row is the instructive one and needs the DP to state honestly.
Propagating on-neurons implicitly means carrying a composed matrix whose *source
width grows with every kink activation retained*, and re-materialising to reset
that width costs `2·src·n > 2n²`, i.e. more than a dense layer. So the scheme is
neither "always compose" nor "always dense": the cheapest schedule is a choice per
layer, which we solve exactly by dynamic programming over which layers
re-materialise. Even at that optimum it is 1.025×. **A structural insight with no
denominator in the cost model is worth nothing, however true it is.**

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
  per-coordinate mean 0 / variance 1 cost **2.35×** with no optimisation at all.
  **[REWRITE REQUIRED — 2026-08-19.** The explanation that followed here, that a
  QMC set is unbiased because it is the correct quantile transform rather than
  because its sample moments are (0,1), is *true* but does not explain a 2.35×
  regression. Moment matching's bias is **O(1/N)** (Glasserman §4.5.1 p.245), i.e.
  second order, not first. And our shipped Sobol set's **diagonal** second moment is
  already 107× better than iid, so forcing it is a measured **no-op: 1.0000×, CI
  [0.999, 1.001]** (LOG 2026-08-19), independently reproduced at "1.000× to three
  digits" in forum 18151. The 2.35× may be real — the likely confounds are that our
  `normalise_` also *centres*, and that it ran inside `g2_optimize.py` against a
  9.156e-6 baseline rather than the shipped configuration — but **this section must
  not be published until the number is re-measured in the shipped configuration.**
  The publishable finding in its place is the *off-diagonal* one: our net's
  off-diagonal second moment is only 5.9× better than iid, and forcing *that* is
  also worth nothing, because the rotation already collects it.**]** Corollary
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

## 7. Validation: how we know any of this

Every number above is a prediction about networks nobody has seen, because that is
what the competition scores. A prediction is worth its instrument, so this section
describes the instrument. We think it is the most transferable part of the work:
the estimator is specific to this task, but the failure modes below are generic to
any benchmark that reports per-item feedback on a fixed evaluation set.

### 7.1 The defect that motivated all of it

Our offline networks were not the evaluator's networks. Both are He-initialised
ReLU MLPs of the same shape, drawn from the same distribution, and for a month
that seemed like enough. It is not. The evaluator draws each weight matrix in
float64 and casts once at the end; we drew natively in float32, which consumes the
random bit stream by a different algorithm and produces entirely different numbers.
More seriously, the evaluator expands one root seed into **three independent
substreams** — weights, ground-truth samples, and the seed handed to the estimator
— while we used a single seed for all three. Our estimator derives its randomised
QMC digital shift from that seed, so on our own data the point set was a
deterministic function of the very network it was scoring. The evaluator has no
such coupling.

Neither defect invalidated a statistical result: the distribution was right to
within 3e-4 in weight standard deviation. Both made per-seed reproducibility
against the evaluator impossible, and the second was a live path to a result that
would have evaporated on unseen data.

The fix is a reimplementation of the evaluator's generation protocol checked for
**bit-identity on raw integer representations**, not to a tolerance: 100/100 across
five shapes × twenty seeds, weights and seed semantics both. The old generator
never matched, at a maximum relative elementwise divergence of 1.87.

**The general lesson: "same distribution" is not a reproducibility claim.** If a
benchmark's data are generated rather than fixed, verify the generator bit-for-bit
or you have a different experiment than the one you are being scored on.

### 7.2 A three-tier data protocol

The reason the defect survived a month is that we had one pool of data doing three
incompatible jobs. Separating them:

| tier | data | may be used to | may not |
|---|---|---|---|
| **decide** | freshly minted networks, a new seed block per experiment, never reused | choose constants, compare methods, kill tracks | — |
| **confirm** | the organizers' public split, 1e9-sample truth | check that a decision survives on their data | choose anything; read-only |
| **vault** | 500 sealed networks, 2²⁵-sample truth, touch-counted | estimate the level expected on the private re-run | iterate |

The middle tier is the one people get wrong, and we did. A split that has steered a
month of decisions has an optimum that is partly a memory of the search that
visited it; a constant fitted there carries that memory into the final evaluation.
The split remains the best data available — it is the organizers' own, with truth
three orders below any effect we chase — but only as a check.

The first tier rotates seeds because re-measuring a constant on the seeds that
chose it is not independent evidence, it is the first measurement's noise served
twice.

### 7.3 The vault

Five hundred networks, ground truth at 2²⁵ = 33.5M samples each (Monte-Carlo noise
1.6e-9, or 0.11 % of the effect we act on), sealed by hashing the contents and
committing the hashes. Every evaluation increments a counter and appends a line to
a log, so a reported baseline can be checked against the number of attempts that
produced it. Three touches are permitted: a baseline, one per estimator change, and
designation day.

The seal-time v2.4.1 baseline used exactly one touch: adjusted median
**2.4651e-7**, mean **3.2669e-7**, p99 **2.0535e-6**, maximum 3.0469e-6, with
zero non-finite predictions. Ground-truth noise was 0.09% of measured raw MSE.

Only seeds and ground truth are stored — not weights. Once the generation protocol
is proven bit-exact, 4 KB of seeds pins down what would otherwise be 4.2 GB of
float32, which is the provenance proof paying rent immediately.

We are aware that a self-imposed touch counter is not a cryptographic commitment
and that a sufficiently motivated author could ignore it. It is not aimed at that
threat. It is aimed at the ordinary and far more common one: running an evaluation
five times, remembering the third.

### 7.4 Invariants, not just smoke tests

A contract suite that only asks *did it crash?* passes happily on code that
computes a plausible wrong answer. Five properties of the *true* answer are
therefore checked on the shipped file at every change:

* **A closed form where one exists.** For x ~ N(0, I) the first layer's expected
  post-ReLU activation is ‖w‖/√(2π) exactly. Measured agreement: 1.8e-16.
* **Permutation equivariance.** Relabelling a hidden layer's neurons leaves the
  network's function unchanged, so predictions must permute with it and every other
  layer must be untouched. This is the only test in the suite that catches a
  transposed index or an axis swapped inside a reduction — errors that are finite,
  plausible, and wrong. The permuted row matches to 2.2e-15; other rows move by
  ≤4.4e-8, which is not slop but the reordering of summations inside downstream
  matmuls, and is nine orders below the decision signal.
* **Budget monotonicity.** More budget never buys fewer samples, and every rung
  down to zero degrades rather than throws.
* **Bit-identical determinism**, including from a freshly constructed estimator —
  the case that catches state leaking between items through a cache.
* **Ground-truth noise reported with every MSE**, so an "improvement" smaller than
  the label noise cannot be mistaken for one.

### 7.5 Constants audit

Every frozen constant was re-measured on fresh, never-reused seeds, asking not "is
there a better value" — there always is, by noise — but "is the frozen value inside
the measurement error of the fresh optimum?" Paired comparisons with common random
numbers and 32-shift averaging bring the resolution to ~1 %, where a single
100-network run cannot resolve 15 %: two realisations of the *same* estimator
differ by ~18 % in mean MSE.

| item | fresh result | verdict |
|---|---|---|
| pruning threshold | τ=-2.5 is the exact sweep optimum | keep |
| rotation rank | full rank appears 1.0108× better, only 0.7σ | keep rank 32 |
| operating point | N=65,536 beats 16,384 by 1.0408× at 3.2σ in the FLOP-only audit | real-stack residual billing leaves ~1.007×; do not adopt |
| pruned-value patch | 1.043e-8 vs the old 3.8e-9 bound | withdraw the bound; end-to-end τ verdict unchanged |
| v0 calibration | ≤0.07% fresh/frozen drift | keep values; fallback wiring debt recorded |

### 7.6 Measurement discipline, stated as a result

Three separate apparent wins in this project evaporated on re-measurement from
larger samples. At these effect sizes subset measurement is not evidence, and we
report the rule we adopted as one of the findings: **nothing is decided on fewer
than the full split, and any comparison of point sets or bases is averaged over
many independent randomisations rather than measured once.**

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

## 7b. Why the fluctuations collapse — NNGP correlation growth  *(STUB — strategy session to draft)*

*Placeholder. The numbers this section has to explain are all measured and in the
repo; the theory is being written elsewhere. Do not fill this in here without
merging that draft.*

**The measurements it must account for** (figure 1, and `figure_data.py`):

| layer | 1 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| effective rank (participation ratio) | **165.0** | 25.9 | 8.8 | 4.4 | **2.8** |

and the companion facts:

* top-64 principal directions hold 90.9% of the fluctuation energy at layer 8,
  **96.2%** at layer 16, 98.2% at 24, **99.1%** at 32;
* the **Gaussian closure's own propagated covariance** finds the same subspace —
  its top-k captures within one point of the data PCA at every depth — while
  eigenvectors of W Wᵀ capture exactly k/256, i.e. nothing;
* JJᵀ for the linearised end-to-end map is numerically rank ~68: λ₃₂/λ₀ = 1.2e-6,
  λ₆₄/λ₀ = 6.4e-12, λ₉₆/λ₀ = 4.5e-17;
* the firing index does **not** become bimodal — it stays unimodal and widens, so
  the tails fill from both ends (figure 3).

**The expected shape of the argument** (for the drafter, not a claim yet): He init
is ReLU-critical, so norms are preserved but input–input correlations approach 1
only algebraically with depth (Poole 1606.05340; Schoenholz 1611.01232; Hayou
1902.06853). As the correlation between two inputs' representations → 1, the
covariance of the activation vector across inputs approaches a rank-one object,
which is exactly a participation ratio falling toward 1. Our measurement says it
reaches ~3 at depth 32 rather than 1, and the gap is the interesting part.

### Verification runs — measured 2026-08-06, `local_sim/theory_probe.py`

*These are measurements against pre-stated, parameter-free predictions, run on
fresh mints (seed block D07). They are reported here for the drafter; the prose
of §7b is still to be written and must not be back-fitted to them.*

**P1 — the correlation map is parameter-free and it holds.** For He-init ReLU the
uncentered correlation of two inputs' representations should obey
`c_{l+1} = R(c_l)` with `R(c) = [√(1−c²) + c(π − arccos c)] / π` — the arc-cosine
kernel, no fitted constant, because the 2/n scaling makes the map exactly
norm-preserving. Measured against R iterated from the input correlation, over four
starting correlations × 32 layers × 8 networks × 2048 input pairs:

| layer | c₀ = 0.0 meas / pred | c₀ = 0.3 | c₀ = 0.6 | c₀ = 0.9 |
|---|---|---|---|---|
| 17 | 0.9440 / 0.9351 | 0.9279 / 0.9397 | 0.9486 / 0.9482 | 0.9701 / 0.9709 |
| 25 | 0.9681 / 0.9629 | 0.9603 / 0.9649 | 0.9718 / 0.9687 | 0.9802 / 0.9802 |
| 32 | 0.9777 / 0.9747 | 0.9655 / 0.9758 | 0.9801 / 0.9780 | 0.9862 / 0.9851 |

**Maximum absolute deviation over every layer and every start: 0.0133**, against a
pre-registered threshold of 0.02. The correlation really does approach 1
algebraically rather than geometrically — by depth 32 it is only ~0.98, which is
the quantitative reason the participation ratio stalls near 3 instead of reaching 1.

**P3 — and the collapse is toward the mean direction specifically.** This is the
sharper result and it was not obvious in advance: low rank does not by itself say
*which* direction survives. Measuring the leading eigenvector v₁ of the
across-input fluctuation covariance against the mean activation direction μ:

| layer | 1 | 5 | 9 | 13 | 17 | 21 | 25 | 29 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| \|⟨v₁, μ̂⟩\| | 0.079 | 0.866 | 0.947 | 0.963 | 0.970 | 0.980 | 0.987 | 0.993 | **0.994** |
| participation ratio | 159.3 | 45.0 | 21.5 | 12.8 | 8.6 | 6.4 | 4.2 | 3.1 | **2.4** |
| λ₁ share of energy | 0.013 | 0.067 | 0.149 | 0.228 | 0.305 | 0.366 | 0.476 | 0.571 | **0.651** |

At the scored layer the dominant fluctuation mode carries 65 % of the energy and is
**99.4 % aligned with the mean**. That is the mechanism behind the common mode of
§2 stated in a different vocabulary: an error along μ̂ is exactly a rescaling of the
whole output layer, which is what `1 + g` is. The two sections are describing one
phenomenon and the write-up should say so.

*Caveat on the PR column: a 256-dimensional covariance estimated from 4096 samples
is downward-biased, which is why layer 1 reads 159.3 against the 165.0 of figure 1
(measured at higher sample count on an older mint). The trend is unaffected; the
absolute values in figure 1 are the ones to quote.*

**P2** — the two-sector participation-ratio fit held out at layers 12/20/28 — is
**not run**. Its functional form is in the theory draft, which is not yet in the
repo, and inventing a two-sector form here and then reporting it as a
verification would invert the point of the exercise.

## 8. Robustness, and why the distribution is the deliverable

The competition's final ranking comes from re-running one designated submission on
MLPs it has never seen. That makes the *distribution* of per-MLP score the object
to engineer — a method with a good mean and a fragile tail loses a fresh re-run,
and a single failed MLP scores as zeros. We think this framing is under-discussed
in the write-ups so far, all of which (ours included, until now) report a mean.

**Measured**, on 1200 freshly minted He MLPs with 4.19M-sample MC labels: mean
adjusted 3.01e-7, median 2.19e-7, p90 5.69e-7, p99 1.65e-6, **max/median 12.9×**.
The worst 1% of networks carry 6.9% of the total score, the worst 10% carry 31.9%.

Bootstrapping the score a re-run of size R would produce (20k resamples):

| R | median | p90 | p99 | p90/median |
|---|---|---|---|---|
| **10** | 2.84e-7 | 4.26e-7 | 5.83e-7 | **1.50×** |
| 25 | 2.94e-7 | 3.77e-7 | 4.60e-7 | 1.28× |
| 50 | 2.98e-7 | 3.54e-7 | 4.11e-7 | 1.19× |
| 100 | 2.99e-7 | 3.39e-7 | 3.74e-7 | 1.13× |
| 250 | 3.00e-7 | 3.25e-7 | 3.47e-7 | 1.08× |
| 1000 | 3.01e-7 | 3.13e-7 | 3.23e-7 | 1.04× |

**The nuance that keeps this honest**: the re-run suite is *common to all teams*,
so MLP hardness is a common-mode term. It inflates or deflates everyone's score
together and mostly cancels in the *ranking*. What does not cancel is the
interaction between draw and method — different estimators are not uniformly
better across the difficulty range — so the ranking jitter between two
similarly-placed teams at R=10 is on the order of **±10–20%**, not the ±50% the
absolute spread suggests. Both numbers belong in the paper, and confusing them
overstates the case.

**And the skew is mostly the metric, not the method** — the finding we would lead
with, and the one an organizer could act on:

| quantity | median | p99/median | max/median |
|---|---|---|---|
| raw per-MLP MSE | 1.19e-6 | 6.87× | 8.92× |
| **MSE / mean(truth²)** | 1.78e-6 | **4.06×** | **5.24×** |
| mean(truth²) itself | 0.704 | 3.95× | 5.41× |

The score is an *unnormalised* MSE and He-init draws vary about 5× in activation
scale: `corr(log per-MLP MSE, log mean(truth²)) = +0.548`, and normalising by that
scale removes roughly half the excess dispersion. The five worst networks have
mean(truth²) of 1.5–3.7 against a median of 0.70. So a large part of the tail is
inherited from the target, is shared by every competitor, and is not something
estimator engineering can remove — **reporting a scale-normalised MSE alongside the
raw one would make the benchmark measure the estimator rather than the draw.**

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
- [x] **Figures 1–3 drafted** (`docs/figures/`, rendered by
      `local_sim/make_figures.py` from `local_sim/figure_data.py`):
      (1) effective-rank collapse by depth, 165 → 26 → 9 → 4 → 3, with a
      10–90th-percentile band — the empirical fact the NNGP section must explain;
      (2) error-vs-N in two panels (never two y-scales): MSE ∝ N^−0.988 above,
      adjusted score below, showing it fall steeply while the multiplier is pinned
      at the 0.1 floor and go flat the moment the floor is cleared;
      (3) the α density by layer, per-column scaled, over a stacked dead/kink/on
      share panel.
- [ ] Remaining figures: (4) the fresh-MLP score distribution with the
      bootstrap-by-R inset — the most under-reported thing we have;
      (5) per-MLP MSE against mean(truth²) on log axes, showing the +0.55 slope
      that explains the tail; (6) corr(g_k, g_L) versus depth.
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
