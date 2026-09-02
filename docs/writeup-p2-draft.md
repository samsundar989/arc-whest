# Phase-2 contribution write-up — WORKING DRAFT

> **DRAFT. NOT SUBMITTED, NOT POSTED, NOT QUOTED OUTSIDE THIS REPOSITORY.**
> Started 2026-09-02 under the strategist's track-2 instruction (the contribution
> prize is co-equal with the leaderboard prize; the deliverable is this document).
> Every claim below carries its artifact path and its evidence status. A claim
> that enters the *submitted* version is STRATEGIC and needs an independent
> Skeptic pass first — `AGENTS.md` § *Work classification*.
>
> This file **replaces** `docs/writeup-draft.md` as the live draft. That file is
> the Phase-1 manuscript, is marked stale, and stays untouched: correcting it in
> place would be a claim review, which is a separate slice.

**Deadline:** contribution write-up due **2026-10-24 23:59 UTC**. Phase 2 closes
**2026-10-17 23:59 UTC**.

---

## Status of this draft

| section | state | blocking evidence |
|---|---|---|
| 1. The score identity | **populated, evidence complete** | — |
| 2. What the identity forbids | **populated, evidence complete** | — |
| 3. The amplification law | **populated, evidence complete** | — |
| 4. Where the error actually is | **populated, evidence complete** | — |
| 5. Compressing the third cumulant | **populated, evidence complete** | — |
| 6. What an estimator may spend, and the two levers | **populated, evidence complete** (new, 2026-09-02) | — |
| 7. The off-diagonal lane | **populated** — the lane is closed on a measurement | — |
| 8. Method and protocol | **populated** | — |
| 9. What we got wrong | **populated — nine entries, and this list grows** | — |
| 10. Result and nomination | **PENDING** | end of Phase 2; Samuel's gate |

---

## Working title

*What a compute-metered leaderboard actually rewards: a score identity, an
amplification law, and eleven closed lanes in white-box activation estimation*

**Alternative, if §7 lands positively:** *Regenerating the third cumulant from its
own read-slices: K3-class accuracy at the multiplier floor*

---

## Abstract *(skeleton — rewrite last)*

The task: predict the expected post-ReLU activation of every neuron of a random
He-init ReLU MLP (1024 wide, 16 deep, no biases) under `x ~ N(0, I)`, scored as
`final-layer MSE x max(0.1, FLOPs / 2^41)`.

We report three things we believe generalise beyond this benchmark, and eleven
lanes we closed with a measured reason for each.

1. **A score identity.** Above the multiplier floor, any estimator whose error
   falls at the Monte-Carlo rate in its own cost has an adjusted score
   *independent of that cost*. Sampling cannot climb a compute-metered
   leaderboard; it can only choose a constant. We verify this on a frozen
   eight-point quadrature panel where the adjusted score is flat to `0.62%` over
   a `4x` cost range, and show the flatness is an algebraic identity of the
   closed-form census rather than a measurement.
2. **An amplification law.** The Gaussian-closure tower is a *linear amplifier*
   of intermediate state error, with excess final MSE scaling as the square of
   the injected relative error across three decades, gain `~18x` coherent and
   `~4.2x` incoherent over 14 layers, ratio `4.3 ~ sqrt(14)`. The practical
   consequence is a method: **convert a lane's oracle ceiling into an accuracy
   requirement before building the lane.** One sensitivity sweep does it, and it
   would have retired two of our lanes before we spent 96 fresh networks on them.
3. **A cost budget in the only unit that matters.** The metering package bills a
   dense `(n, n) @ (n, n)` matmul at `4 n^3`, twice the textbook figure. Restated:
   at the multiplier floor, an entire `1024 x 16` estimator is allowed
   **3.2 dense matmuls per layer**. A second-order Gaussian closure spends 2.0. A
   third-order flow that regenerates its cumulant from its own read-slices spends
   5.0. That single sentence prices the whole design space.

We also report a negative that we think is the most useful thing in the paper: a
Gaussian closure is an *internally consistent approximation, not a set of
independently improvable marginals*. Correcting its marginal variance **toward
truth**, at zero fit error, makes the final answer `16x` **worse**, because it
breaks a cancellation against the truncated off-diagonal. The mean channel of the
same correction, also at zero fit error, is `6.9x` better. The two channels have
opposite signs.

---

## 1. The score identity, and why sample count is not a design variable

**Evidence: CONFIRMED.** `experiments/LOG.md` 2026-09-02 (SCREEN WP12);
frozen panel `experiments/2026-08-31_wp2_objective/carrier_panel_1024x16.json`,
content-addressed with its rule and gate inside the digest.

The scoring rule is `error x max(0.1, F / B)`. Above the floor the multiplier is
`F / B`, so for any method whose error obeys `E = C1 / k + C0` at cost `F = c k`
— the Monte-Carlo rate — the adjusted score is

    (C1 / k) x (c k / B)  =  C1 c / B ,   independent of k.

Measured on an eight-point quadrature-carrier panel, `k = 1 .. 16`:

| `k` | `F / B` | multiplier | `MMD^2` | `MMD^2 x multiplier` |
|---:|---:|---:|---:|---:|
| 1 | `0.0312` | `0.1000` | `2.0758e-05` | `2.0758e-06` |
| 4 | `0.1250` | `0.1250` | `5.1816e-06` | `6.4769e-07` |
| 8 | `0.2500` | `0.2500` | `2.5854e-06` | `6.4635e-07` |
| 16 | `0.5000` | `0.5000` | `1.2874e-06` | `6.4368e-07` |

**Flat to `0.62%` across a `4x` cost range.** Every operating point is the same
point. Choosing a sample count was never a decision.

*Scope, stated in the paper:* `MMD^2` is a truth-free proxy; the step to
final-layer MSE is the pre-registered assumption that the objective is monotone in
the error. The one measurement linking the family to real MSE puts the `k = 8`
carrier `1.42x` **worse adjusted** than our floor-pinned baseline.

**What the identity does NOT say, and we correct our own earlier claim in print.**
We once concluded from board forensics that "the free zone is a trap" because 33
of 57 entrants were floor-pinned and the best of them was rank 16. That inference
was wrong. The floor caps *price*, not accuracy: a raw error of `3.0e-8` delivered
at `F <= 0.1 B` scores `3.0e-9` and leads. **The floor is the destination of the
compression race, not a trap** — and the counter-evidence was in our own data, an
entrant moving multiplier `0.3065 -> 0.2698` with raw error flat.

---

## 2. What the identity forbids, and what it permits

Only two kinds of method can move the score:

1. error falling **faster than `F^-1`**, or
2. error falling at any rate **entirely inside the free zone**, where the
   multiplier is pinned at `0.1` and additional FLOPs are literally free.

Category 2 is where the whole design problem lives, and it has a hard edge: at
the floor the estimator's *entire* budget is `0.1 x 2^41 = 2.199e11` counted
FLOPs. §6 turns that into an operation count.

*(This section also carries the frontier observation that the leaders' raw error
is roughly flat across their budgets — which is the same identity seen from
outside. Board snapshots are dated and private; the paper quotes the structure,
not another team's numbers.)*

---

## 3. The amplification law

**Evidence: CONFIRMED.** `experiments/LOG.md` 2026-09-02 (WP10);
`experiments/2026-09-01_wp9_corrector/`.

Inject a relative error `c` into the closure's intermediate state at every layer
and measure the excess final-layer MSE. Across **three decades**, in both a
coherent and an incoherent regime, the excess scales as `c^2` — i.e. the tower is
a **linear amplifier**, gain `~18x` coherent and `~4.2x` incoherent over 14
layers, ratio `4.3 ~ sqrt(14)`.

The positive control matters and is reported: at `c = 0` the perturbed arm
reproduces the baseline to `0.000e+00`, exactly, so the instrument is measured
not assumed.

**The method this yields.** An oracle ceiling and an accuracy requirement are the
same measurement seen from two ends. Our in-flight mean corrector has a genuine
`6.872x` ceiling at zero fit error — and the amplification law converts that into

> the corrector must reach held-out `R^2 >= 99.85%` on its target merely to
> **match** the uncorrected shipped file.

Our best fitted arm reached `R^2 = -0.53`; an earlier, easier readout target
reached `+0.50`. `(1 - R^2)` had to fall `330x`. **The lane was retired by one
sensitivity sweep that cost minutes, after 96 freshly minted networks had already
been spent on it.** Running the sweep first is the recommendation.

---

## 4. Where the error actually is: three channels, three closures

**Evidence: CONFIRMED.** WP9 / WP10, `experiments/LOG.md` 2026-09-02.

We asked whether the residual error of a calibrated second-order closure is a
**per-neuron** phenomenon. It is decomposable into three channels and all three
are now closed, each with a measured reason:

| channel | oracle ceiling at ZERO fit error | closure |
|---|---:|---|
| **readout** (correct the final map) | `1.047x` | exhausted by sixteen literal per-layer constants; a 354-parameter network predicts the per-neuron residual at `R^2 = 0.499-0.522`, and a **single** per-layer constant reaches `0.5097` |
| **in-flight variance** | **`0.059x`** — `16x` WORSE than doing nothing | *destructive*; see below |
| **in-flight mean** | `6.872x` | real, but behind a `99.85%` `R^2` entry price (§3) |

### The result we think is the most transferable

> **Correcting a closure's marginal variance toward truth is destructive.**

Not "hard" — destructive, at **zero** fit error, by `16x`. The mechanism: a
Gaussian closure is an internally consistent approximation, not a set of
independently improvable marginals. Making the marginal variance exact while the
off-diagonal covariance stays truncated at Price order 1 **breaks a cancellation**
between the two errors. The two channels have opposite signs, and moving them
together — which is what a naive per-neuron corrector does — lands at `0.547x`,
i.e. worse than doing nothing, while the mean channel alone is `6.872x` better.

A related exact fact, worth one line in the paper: the closure is **positively
homogeneous of degree 1** in its state (verified to `1.6e-30`), so scaling
`(mu, Sigma) -> (g mu, g^2 Sigma)` leaves the ReLU's shape parameter unchanged and
scales every downstream mean by exactly `g`. **Every purely multiplicative state
correction is one output rescale in disguise**, whatever its parameter count. This
retires an entire family of "richer calibration" proposals analytically.

---

## 5. Compressing the third cumulant: four ways that do not work, and why

**Evidence: CONFIRMED**, with one retraction in the record (below).
`experiments/2026-08-28_pb_oracle/`, `experiments/LOG.md` 2026-08-27/28.

The reference third-order flow reaches raw `3.92e-8` on the organisers' data
against a second-order closure's `4.05e-6` — a `100x` accuracy gap — but at
`1.48 x` budget, so it scores worse than a floor-pinned second-order method. The
whole question is compression, and we tried four families:

| family | what it does | result |
|---|---|---|
| rank capping by column norm | keep the top-`r` CP columns by Frobenius weight | at `r = n/8` the "compressed K3" is **`14x` closer to plain K2 than to K3** |
| best rank-`r` CP (an oracle) | ALS, warm-started, unreachable in deployment | drift `18.8 x MSE_full` at rank `n` |
| diagonal-blocks-only | drop the all-distinct block, keep both read-slices exact | drift `22.1 x MSE_full` |
| exact-diagonal + truncated remainder | protect the slices, truncate only the remainder | needs rank `4n` (`0.56 x` budget) to reach parity |

**The mechanism, and it is the paper's point.** The flow reads `kappa_3` through
**exactly two diagonal slices** — the fully-repeated `(3,)` slice and the
partial-diagonal `(2, 1)` slice — and through nothing else. Meanwhile `n` of the
`2n` columns added per layer are of the form `D[:, j] (x) e_j (x) e_j`. So
capping below rank `n` **deletes whole neurons' `kappa_3` diagonal**: it is
**coordinate pruning, not compression**, which is exactly why the drift curve is
flat from rank 32 to rank 128.

And the deeper one: every one of these compressors minimises a **Frobenius**
error on an `n^3` tensor whose only readout is an `n x n` diagonal slice. That is
the wrong norm for the functional. §7 is the experiment that asks whether the
right norm changes the answer.

**We report a retraction.** Our first reading of the "exact-diagonal" family was
wrong by `4.5x` in the wrong direction, because the truncated remainder leaked
onto the `(2, 1)` slice — reaching `40%` of that slice's true norm by layer 15 —
and the flow reads nothing else. An independent falsification pass caught it. The
corrected cell is the one tabled above.

---

## 6. What an estimator may actually spend *(new, 2026-09-02)*

**Evidence: CONFIRMED.** `experiments/2026-09-02_wp13_slice/e0_cost.json`,
executed in the official metering package inside a real budget context.

The metering package bills a dense `(n, n) @ (n, n)` matmul at **`4 n^3`**, twice
the textbook `2 n^3`; elementwise operations and transposes are `O(n^2)` or free.
Restating the budget in that unit is the single most useful thing we did for
design:

> At the multiplier floor, an entire `1024 x 16` estimator is allowed
> **3.2 dense matmuls per layer.**

- A calibrated second-order closure spends **2.00** — that is `W^T Sigma W` and
  essentially nothing else. It is `0.063 B`, and it is what we shipped.
- A third-order flow that **regenerates** its cumulant from its own two read
  slices every layer — so the carried rank never exceeds `2n` instead of growing
  to `32n` — spends **5.0**, i.e. `0.153 B`.

The structural fact that gets it to 5.0 rather than 10: rebuilding the degree-3
tensor from its slices produces factors `(diag(S3) + 3 S21^T, I, I)` in which
**two of the three legs are exactly the identity**. Contraction on an identity
leg is free and a diagonal leg collapses a slice einsum to a row scaling, so a
generic implementation pays six matmuls a layer for what three do — a measured
`2.08x`, found by reading the reference implementation rather than by searching.

### The two levers on the multiplier, both measured

`5.0` matmuls a layer is `0.153 B`, i.e. a multiplier of `0.153` when `0.100` was
available — **53% of the cost buying nothing.** Two levers were available and
exactly one works.

**Symmetry does not help.** The covariance is symmetric and its congruence
`W^T Sigma W` is two of the five matmuls, so a symmetry-aware meter looked like a
free discount. Tagging the operand as symmetric bills the *identical*
`8,585,740,288` — ratio exactly `1.0000` — and the tagging call itself costs
`0.0137 n^3`. **A symmetric operand is not a cheaper operand.**

**Strassen does.** Measured under the same meter:

| levels | vs a plain matmul | flow's `F / B` | multiplier | leaf matmuls | array ops |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.0000` | `0.1527` | `0.1527` | 1 | 1 |
| 1 | `0.8777` | `0.1341` | `0.1341` | 7 | 36 |
| 2 | `0.7732` | `0.1181` | `0.1181` | 49 | 281 |
| 3 | `0.6861` | `0.1048` | `0.1048` | 343 | 1,996 |
| **4** | `0.6174` | `0.0943` | **`0.1000`** | 2,401 | 14,001 |

**At four levels the flow pins the multiplier floor**, with `5.7%` of the free
zone unused. A 1969 algorithm and one structural reading of a reference
implementation together move a third-order method from `0.153` to `0.100` — which
is the difference between a ceiling of `6.1e-9` and one of `4.0e-9`.

**And the paper must say why that table is not a recommendation.** The scored
budget is not the only budget: the runtime also enforces a per-MLP *residual wall
time* that tracks the **number of operations** and the size of the temporaries
they free, not FLOPs — and an overrun is not a penalty, it zeroes the prediction
*and* forces the multiplier to `1.0`, turning a `1e-7` row into a `~0.92` row, a
`~66,000x` loss. The `array ops` column is the warning label. `L = 2` — `281`
operations for `4.7e-9` — is very likely the honest operating point, and no level
is authorised until it has been soaked at scale. Strassen's own numerical error
is a second unmeasured cost.

---

## 7. The off-diagonal lane: the information is findable in principle and not in practice

**Evidence: CONFIRMED**, pre-registered as `CONTRACT WP13` with nine numbered
predictions committed before any data — four of which predicted the lane would
fail — and reviewed by an independent falsifier that **retracted our stated
reason for the kill** while leaving the kill standing.

The second-order closure's first approximation is the Price order-1 truncation of
the post-ReLU off-diagonal covariance. A third-order flow that regenerates its
cumulant from its own read-slices (§6) discards one object: the all-distinct
block, worth `22x` the reference method's own MSE in drift. We asked three
questions about it.

**How accurate must the slices be?** Perturbing both read-slices by `(1 + eps)` at
every layer gives `excess = 1.35e-4 * eps^2.07`, `R^2 = 0.998` over three decades,
so **about `1%` relative accuracy holds the end-to-end excess at `1e-8`**. The
coherent regime is `8.8x` worse in amplitude than the incoherent one.

**Is the discarded information low rank?** **Yes, and increasingly so with
depth** — the finding we did not expect. The `n x n` residual's `r90` falls
`264 -> 78` across sixteen layers while the covariance's own `r90` falls
`237 -> 26`. Against an iid-Gaussian baseline of `r90 = 523` it is below isotropic
at the first layer and `6.7x` below at the last, monotone on 8 of 8 networks. And
it is **cheap to carry**: correcting the slice directly is one matmul per layer,
so a rank-512 correction costs a `9%` multiplier penalty and a rank-128 one is
free at the floor.

**Can an affordable selector find it?** **No, and this is the result.** Five
`O(nR)` column-weight selectors — Frobenius-weighted and slice-weighted
truncation, and three importance-sampling variants — were tested at equal cost.
At every budget, **every one of them is `1.45x` to `2.10x` worse than not
correcting at all.** Two sub-results explain why:

- The slice-targeted weight beats the Frobenius weight by **`1.0019x`**. On this
  factorisation the two are collinear, so **there was no wrong norm to fix** — the
  hypothesis that the compression lane had been minimising the wrong objective is
  refuted, not confirmed.
- Randomised importance sampling is **`2.48x` worse** than deterministic
  truncation. The unbiased `1/(k p_r)` rescaling has variance that dominates,
  because the column-weight distribution is not heavy-tailed enough for that trade
  to pay.

So the lane closes, and it closes in an unusual place: **not at the information,
not at the budget, but at the search.** The oracle that finds the subspace costs
orders of magnitude more than the object it selects — which is the same wall the
norm-optimal compression hit, now measured on the readout functional instead of
the Frobenius norm, where it was supposed not to exist.

**What we would tell the next person.** The open question this leaves is sharp
and cheap: the residual's rank collapse tracks the covariance's, which the flow
already holds for free. If even part of the residual's leading subspace lies in
the covariance's, the selector is free. It cannot be all of it — the ratio of
their effective ranks grows to `2.94` — so the question is a containment
*fraction*, and it is the one experiment we would run next.

---

## 8. Method and protocol — what we would tell someone starting this task

**Evidence: CONFIRMED**, and each item cost us something to learn.

1. **Pre-register.** Question, mechanism, numbered predictions, gate threshold,
   falsifier, cost model, and seed block, committed to version control *before*
   any data exists. Git timestamps make it checkable.
2. **Verify that a null control is null.** We built three controls in three
   consecutive slices and all three failed to do their job; one turned out to be
   the *most damaging perturbation in its grid*, presented as a null. A control's
   failure means nothing until the control is measured null.
3. **The binding statistic is the paired arithmetic score difference** with a
   studentised-t or basic-bootstrap interval — the competition's own identity. We
   measured the percentile bootstrap and BCa at false-promotion rates of
   `6.8-8.2%` and `9.1-11.5%` against a nominal `5%`, and refuse both. A
   geometric mean of per-item ratios is a stability diagnostic with no authority
   over an outcome.
4. **Quote every error with its ground-truth noise floor.** Ours is `54,165x`
   below the error we were measuring — which is the only reason the measurement
   means anything.
5. **A fixed evaluation set that returns per-item feedback is not a held-out test
   set.** Fit nothing on it, ever.
6. **Verify ground-truth generation bit-for-bit, not distributionally.** Our own
   generator once matched the evaluator's distribution to `3e-4` while sharing
   none of its values.
7. **Independent falsification, in a context that never saw the builder's
   reasoning.** Six passes; six found something real; one *retracted a negative*
   and reopened a lane we had closed on a measurement of the wrong method.

---

## 9. What we got wrong *(this list is a feature, and it grows)*

- **"The free zone is a trap."** Withdrawn the same day. Right measurement,
  wrong inference; the floor is the destination, not the trap. (§1)
- **"The per-neuron half of the fork is empty."** Retracted by an independent
  pass that ran the oracle cells we never ran and found `6.42x` sitting in the
  channel we had killed. (§4)
- **"Rank capping is closed for good."** Softened to a measured negative; the
  stronger claim was not supported by what we had measured. (§5)
- **"Structure-preserving compression works."** Refuted by `8x` in the wrong
  direction, by our own pre-registered falsifier. (§5)
- **A prediction was swapped after the fact.** We scored a different slice's
  prediction in place of the contract's, and an independent pass caught it. The
  real prediction was then answerable, and held exactly. (§8.1)
- **A cost model assumed `2 n^3` per matmul.** It is `4 n^3`. Every hand estimate
  we had made was a factor of two optimistic. (§6)
- **A three-hour certification suite reported `45` failures of `300` and could
  not say why**, because it counted failures and discarded the reason, and
  because its own median was computed over rows that never executed. Both are
  now unit-tested in milliseconds. The failures did not reproduce.
- **We scored a gate in a currency our own contract forbade.** The rank
  threshold in §7's pre-registration was inherited from a reference
  implementation's runtime counter, while the contract said in terms that no gate
  may be scored on that counter. Re-priced in the metering package that actually
  bills, the quantity we had called unaffordable costs a `9%` penalty. The
  conclusion survived — a different gate failed, decisively — but **the reason we
  published for it was wrong**, and an independent pass caught it. This is the
  *second* time in a week: the previous one swapped a prediction between slices.
  The lesson is narrow and mechanical: **write every threshold in the unit the
  gate will be scored in, in the same sentence.**
- **Four consecutive null controls could not do their job.** The most recent
  compares a deterministic call with itself: the perturbation branch is skipped
  when the perturbation is zero, so the control returns exactly zero and *cannot*
  fail. It is a determinism check wearing a control's name. We had previously
  written the rule as "verify that a null control is null"; that is not strong
  enough, and the form we now use is: **a null control must differ from the
  treatment in code path, not only in parameter value.**

---

## 10. Result and nomination *(PENDING — end of Phase 2)*

To be written against whatever is nominated. The standing plan is **two diverse
nominations with no shared failure mode**, one of which is the robust
second-order fallback that has already graded cleanly twice with zero failed
MLPs.

---

## Appendix A — reproducibility

Everything in this paper is reproducible from a public dataset tag plus the code
in this repository: grader-identical network minting proven bit-identical to the
organisers' published weights, an append-only seed ledger, an executable
promotion gate, and a certification battery. *(Decision pending on what is
open-sourced and when — `RESEARCH_NOTES.md` §2g is private.)*

## Appendix B — the eleven closed lanes, with the measured reason for each

Every row was closed on a measurement, not on intuition, and every row's number
is reproducible from the artifact named. Three of the eleven were closed *twice*,
because the first closure was wrong and an independent pass caught it — those are
marked, because the correction is the more useful record.

| # | lane | measured reason it closed | evidence |
|---|---|---|---|
| 1 | per-neuron **readout** corrector | exhausted by sixteen literal per-layer constants. A 354-parameter network on eight scale-invariant features, trained on 72 networks, ties the *single* per-layer constant (`R^2 0.499-0.522` vs `0.5097`) and does not beat it: `0.9945x`, `t` CI straddling zero | WP9, `experiments/2026-09-01_wp9_corrector/` |
| 2 | per-neuron **in-flight variance** corrector | **destructive**: injecting the *true* variance at **zero** fit error is `16x` worse than doing nothing. Mechanism: it breaks a cancellation against the Price-order-1 off-diagonal (§4) | WP9/WP10 |
| 3 | per-neuron **in-flight mean** corrector | real `6.872x` oracle ceiling, but the amplification law prices break-even at held-out `R^2 >= 99.85%`. Best achieved `+0.50` on an easier target; `(1 - R^2)` had to fall `330x` | WP10 |
| 4 | per-layer **scalar** calibration | exhausted at `2.96x` (and shipped). Any purely multiplicative state correction is provably one output rescale in disguise: the closure is positively homogeneous of degree 1, verified to `1.6e-30` | P2-CAL |
| 5 | higher **Price/Hermite orders** in the covariance map | plateau at `1.05x` | P2-CAL part 2 |
| 6 | marginal **Edgeworth** correction of the final ReLU map | capped at `1.047x` | P2-CAL part 2 |
| 7 | **quadrature / sampling carriers** (Kerdock, MUB, orthogonal blocks) | closed on the score identity itself (§1): adjusted score flat to `0.62%` over a `4x` cost range, and the flatness is an algebraic identity rather than a fit. The structured-over-random advantage at width 1024 is `0.155%` | WP2, WP12, `carrier_panel_1024x16.json` |
| 8 | plain **IID and random-orthogonal** direction sets | lose to the second-order closure outright | Phase-1 sprints |
| 9 | **exact sparse / equality-aware K3** ("W-EFF") | `1.073 B` under the official meter against a pre-registered `<= 0.35 B` screen — fails by `3.07x` | `experiments/2026-08-28_weff/` |
| 10 | **rank-capping** the factored third cumulant | accuracy falls faster than cost; at rank `n` the object is `14x` closer to the second-order closure than to the third-order one. It is coordinate pruning, not compression (§5). **Closed twice** — the first closure claimed "for good" and was softened to a measured negative | P-B, ORACLE-1/2/2b |
| 11 | **late / age-windowed K3** | every window arm's provable score floor is `2.1x`-`4.7x` worse than the shipped file, and `89%` of a window's error is its cheap prefix, not the missing memory | `experiments/2026-08-28_window/` |

Two further lanes are *not* in this table because they are not closed, and saying
so is part of the honesty: **query-directed / adjoint K3** had its kill
**retracted** when its linearity statistic was shown to kill a provably linear
system at every depth (the instrument is broken, the lane is unproven), and
**faithful augmented-K2** is measured free at the floor (`1.037x`) but its
accuracy at the live width is unmeasured — even its sign.

### The one sentence we would keep

**`kappa_3` is compressible and that is not the problem.** A rank-`n`
approximation captures `96.7%` of its Frobenius norm. But the flow reads it
through **two diagonal slices and nothing else**, and norm-optimal compression
spends its rank on the all-distinct bulk that carries most of the norm and none
of the readout. The question was never *"is it low rank"*; it is *"is there a
compressor whose error is orthogonal to those two slices"* — and §§6-7 are our
attempt to answer it by not compressing at all.
