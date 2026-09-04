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
| 6. What an estimator may spend, and the two levers | **populated, evidence complete** (new 2026-09-02; §6.6-6.7 new 2026-09-03) | — |
| 7. The off-diagonal lane | **populated. FIVE routes, five measured ceilings** — §7.1 memory depth, §7.2 the vector ceiling, §7.3 the matrix ceiling plus the sensitivity, §7.4 the mechanistic result and the joint `11.1x` oracle, §7.5 the carried route at `3.33x` against a family ceiling of `3.87x` (new 2026-09-04, independently audited and the audit changed the answer) | — |
| 8. Method and protocol | **populated** | — |
| 9. What we got wrong | **populated — fifteen entries, and this list grows** | — |
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

## 6. What an estimator may actually spend *(new 2026-09-02; **materially corrected the same day** — see 9.4; **executed rather than projected** — see 6.5; **found to be measuring the free half of the objective** — see 6.6, 6.7; **and confirmed on 100 graded networks, which then moved the binding constraint again** — see 6.8)*

**Evidence: CONFIRMED.** `experiments/2026-09-02_wp14_range/e0_cost_f64.json` and
`e0_cost_f32.json`, executed in the official metering package inside a real budget
context, at both dtypes.

### 6.1 A FLOP count is not well-formed without its dtype

We first wrote this section around a sentence we had measured and believed:

> *"The metering package bills a dense `(n, n) @ (n, n)` matmul at `4 n^3`, twice
> the textbook `2 n^3`."*

**That sentence is false, and the way it is false is the most transferable thing
in this paper.** The meter bills the textbook `2 n^3` and then scales *every*
operation by a per-dtype rate taken from its own packaged weight table:

    float64 = 2.0    int64 = 2.0    complex128 = 2.0
    float32 = 1.0    float16 = 1.0  int32 = 1.0    complex64 = 1.0

Our probe built its arrays with a generator that returns `float64` by default. So
we had measured the f64 column and written it down as a property of matrix
multiplication — and then compared it against accuracy evidence gathered, in a
different tool, in `float32`. Measured at `n = 1024`: `4,292,870,144` FLOPs at f64
against `2,146,435,072` at f32.

> **The rule we would give anyone doing this: write the dtype in the same sentence
> as the price, every time. "This costs `X` FLOPs" is not a complete statement.**

A related trap sits one level down. Under NEP-50 a *numpy* scalar such as
`np.float64(3.0)` is **concrete**, not weak, so a single one anywhere in an f32
expression promotes the whole downstream chain and the meter bills all of it at
rate `2.0`. A **Python** `3.0` stays weak and bills f32. Our cost probe contained
three of them. The instrument now asserts the dtype of its own carried state and
refuses rather than reporting a silently promoted number.

### 6.2 The budget, restated correctly

At the multiplier floor, an entire `1024 x 16` estimator is allowed

> **6.4 dense float32 matmuls per layer, or 3.2 float64 ones.**

- A calibrated second-order closure spends `W^T Sigma W` and essentially nothing
  else: **`137,600,326,656`** FLOPs at f64 (`0.0626 B`), **`68,802,510,336`** at
  f32 (`0.0313 B`). The f32 ratio is `0.50002`, stable across five shapes and
  bit-identical across nine networks.
- A third-order flow that **regenerates** its cumulant from its own two read
  slices every layer — so the carried rank never exceeds `2n` instead of growing
  to `32n` — adds **`198,233,321,472`** at f64 and **`99,116,660,736`** at f32.

The structural fact that keeps the third-order increment to about three matmuls a
layer rather than six: rebuilding the degree-3 tensor from its slices produces
factors `(diag(S3) + 3 S21^T, I, I)` in which **two of the three legs are exactly
the identity**. Contraction on an identity leg is free, and a diagonal leg
collapses a slice contraction to a row scaling, so a generic implementation pays
six matmuls a layer for what three do — a measured `2.08x`, found by reading the
reference implementation rather than by searching.

### 6.3 Composing the two halves, which is where the answer was

| composition | second order | third order | total | `F / B` | multiplier |
|---|---:|---:|---:|---:|---:|
| all float64 | `137,600,326,656` | `198,233,321,472` | `335,833,648,128` | `0.1527` | `0.1527` |
| third order f32 only | `137,600,326,656` | `99,116,660,736` | `236,716,987,392` | `0.1076` | `0.1076` |
| **both float32** | `68,802,510,336` | `99,116,660,736` | **`167,919,171,072`** | **`0.0764`** | **`0.1000`** |

> **A regenerating third-order flow reaches the multiplier floor with `24%` of the
> free zone unspent, in plain float32, with no fast matrix multiplication and no
> rank cap.**

Both halves of that composition were measured, independently reviewed, and already
in our own repository. The f32 second-order figure had been reviewed six days
earlier — including the question a reader should ask, whether float32 breaks the
closure numerically: catastrophic cancellation in `var = max(E[z^2] - mu^2, 0)`
fires on `1,497` of `1,572,864` neuron-layers across 96 networks, the variance
clamp fires **zero** times, there are zero non-finite outputs, and the float32
final-layer error against ground truth is *marginally better* than float64. The
third-order figure was measured the day this section was written. **Nobody had
added them.** The cost lane was never short of measurement; it was short of one
addition, and a units error was hiding the reason to try.

### 6.4 The lever we thought we needed, and no longer do

Before the dtype correction, the flow priced at `0.153 B` and the gap to the floor
looked like it needed fast matrix multiplication. Measured under the same meter:

| Strassen levels | vs a plain matmul | flow `F / B` (f64) | leaf matmuls | array ops |
|---:|---:|---:|---:|---:|
| 0 | `1.0000` | `0.1527` | 1 | 1 |
| 1 | `0.8777` | `0.1341` | 7 | 36 |
| 2 | `0.7732` | `0.1181` | 49 | 281 |
| 3 | `0.6861` | `0.1048` | 343 | 1,996 |
| 4 | `0.6174` | `0.0943` | 2,401 | 14,001 |

**Symmetry, the other candidate lever, buys exactly nothing.** The covariance is
symmetric and its congruence is two of the flow's matmuls, so a symmetry-aware
meter looked like a free discount. Tagging the operand bills the *identical*
`8,585,740,288` — ratio exactly `1.0000` — and the tagging call itself costs
`0.0137 n^3`. A symmetric operand is not a cheaper operand.

**And the paper must say why the Strassen table was never a recommendation.** The
scored budget is not the only budget: the runtime also enforces a per-MLP
*residual wall time* that tracks the **number of operations** and the temporaries
they free, not FLOPs — and an overrun does not cost a penalty, it zeroes the
prediction *and* forces the multiplier to its maximum, turning a `1e-7` row into a
`~0.92` row, a `~66,000x` loss. The `array ops` column is the warning label:
`14,001` operations per MLP against a `0.4 s` cliff is not an operating point, it
is a table entry.

> **The general lesson, and it is worth more than the table: we nearly bought a
> `1.6x` cost reduction with a 1969 algorithm, `14,001` array operations per MLP
> and an unquantified numerical risk, when a `2x` reduction was available from
> declaring the right dtype. Check your units before you optimise. The cheapest
> lever is usually the one hidden inside a number you have already measured and
> mislabelled.**

### 6.5 And then we executed it, because a sum of two measurements is not a measurement

An independent review of §6.3 accepted the arithmetic and refused the conclusion,
for one reason: **the two halves had never been executed as one program.** Its
own dispatch census of the reference implementation put a real regenerating layer
at `19.4` matmul-equivalents, four times our composed `4.89`, and it named two
live traps our projection could not price — the metering package's promotion of
Gaussian CDF/PDF inputs to `float64`, and the fact that our cost probe's spelling
was admitted to be numerically wrong while being cost-correct.

So we wrote the layer properly and metered it. One layer at `n = 1024`, in
`float32`, as one program, inside a real budget context:

| | counted FLOPs | dense `1024`-matmuls | `x 16` layers | `F / B` |
|---|---:|---:|---:|---:|
| `float64` | `21,675,713,540` | `5.047` | `346,811,416,640` | `0.15773` |
| **`float32`** | **`10,837,935,618`** | **`5.047`** | `173,406,969,888` | **`0.07886`** |
| the §6.3 projection | | `4.89` | `167,919,171,072` | `0.07636` |

*(These are the figures as measured at the time. §6.6 and §6.7 supersede them
downward — the layer was computing a block that turns out to be dead work — but
the reasoning in this section is about the gap between a projection and an
execution, and re-stating it with later numbers would destroy the point.)*

**The conclusion survives and the number moves.** `0.07886` against `0.07636`:
the projection was **3.3% optimistic**, the floor still holds, and the free zone
under it is `0.02114` of budget — `1.35` float32 matmuls per layer rather than
`1.51`.

The missing `0.157` matmuls per layer are not a mystery and are worth naming,
because they are the part a composed estimate structurally cannot see: the 25
diagram terms, the 16 Wick coefficients, the power-cumulant-to-cumulant
conversion, the degree-4 harmonic projection, the rebuild of the carried tensor
and the diagonal masks. Every one is `O(n^2)`. **A sum of two `n^3` measurements
omits exactly the `O(n^2)` glue that joining them requires**, and at `n = 1024`
that glue was 3.3% — small, but the same order as the margin we were reasoning
about.

Two details that make the number trustworthy, and that we would insist on from
anyone reporting one:

- **The `float32 / float64` ratio is `0.50000` exactly.** That is the check that
  no operation silently promoted. The Gaussian CDF and PDF *do* return `float64`
  from a `float32` input — we recorded both calls doing it — and a single
  un-cast result would have re-promoted the whole graph and billed it at rate
  `2.0`. An audit of all 56 intermediate arrays reported zero dtype violations.
- **The price is a price of something.** Setting the third and fourth cumulants
  to zero collapses the layer to a case with a closed-form answer, and we checked
  three: the mean and variance reproduce the second-order closure exactly
  (`0` and `4.9e-16`); the covariance reproduces it **plus one term the
  second-order closure does not have** (`5.1e-16`, and the check requires that
  term to be present — it is `0.55%` of the covariance, and a layer matching the
  plain closure exactly would be *wrong*); and the third-cumulant diagonal
  matches independent numerical integration to `1.1e-14`.

> **A projection is a hypothesis about a program you have not written. Ours was
> right to 3.3%, which is a good projection and still not a measurement — and the
> 3.3% lived entirely in the category of work neither half could have charged
> for.**

---

### 6.6 The cost we were measuring was free, and the cost that could kill us was an operation count *(new 2026-09-03)*

Everything in §6.1–6.5 is about FLOPs, and for this Phase FLOPs below the floor
**cost nothing at all**. The round sets the cost exponent to zero, so the score
is `MSE x max(0.1, F/B)` and any estimator under `0.1 B` pays the same
multiplier as one at zero. We reached that floor and then kept optimising the
half of the objective that had already stopped mattering.

**What can still ruin a run is the residual wall-time cliff.** The harness
measures the time your estimator spends outside the metered operations —
interpreter overhead, allocation, the glue between array calls — and if it
exceeds `0.4 s` for a single network, that network does not merely score badly:
**it fails, and its prediction is replaced by zeros.** One failed network out of
a hundred moves a `1e-7` score to roughly `9e-3`.

That number does not track FLOPs. It tracks **how many array operations you
issue**, because each one carries a fixed Python-side cost that the FLOP meter
does not see. Measured on the same host, in the same process, in the same run:

| | counted FLOPs | array **operations** |
|---|---:|---:|
| our shipped second-order closure | `68,782,549,504` | **`453`** |
| the third-order flow, as first written | `173,381,273,195` | **`5,327`** |
| ratio | `2.5x` | **`11.8x`** |

**`2.5x` the arithmetic, `11.8x` the operations.** The third-order flow is not
expensive because it computes more; it is expensive because it computes in more
*pieces*. Its arithmetic is dominated by five matrix products per layer, and its
operation count by roughly three hundred small elementwise steps around them.

**We deliberately do not put a residual column in that table.** We measured one
four times on the same machine on the same day and the *control* moved by 55%
between readings, because other work was running; the ratio between the two arms
moved with it. A wall-time residual is a property of the machine that measured
it, and the only reading with authority is the evaluator's own.

The reading that does have authority: on the official runner, the flow as first
written had **`1.21x`** margin against the cliff where our shipped file has
`6.8x`. That version has since been retired for an unrelated reason (§6.7) and
its successor's operation count is `2,646`; neither number has been re-measured
on the evaluator at the time of writing.

> **Under a zero cost exponent, "how much arithmetic" is the wrong question and
> "how many operations" is the right one. They are not proportional, and only one
> of them can fail your submission.**

### 6.7 Two programs, one of them scored: the parity check we had deferred *(new 2026-09-03)*

Our third-order flow existed twice. A research implementation in PyTorch, where
every accuracy number was measured; and a port into the restricted dialect the
grader accepts, which has no PyTorch and no scientific stack. We checked the port
three ways — a degenerate limit with a closed-form answer, independent numerical
integration, and agreement between two independently written copies — and all
three passed. **We never checked it against the implementation whose accuracy we
were quoting**, because that required both frameworks in one process and we kept
deferring it.

Then we ran the port on the real evaluator, and it scored **`3.3x` worse than the
research implementation had predicted**. A control submitted in the same run —
our shipped file, whose score we already knew — transferred to `3.2%`, so the
data, the networks and the harness were all fine. The discrepancy was ours.

**It was one line.** The reference implementation removes a correction term from
one stored component to avoid double-counting, and then adds the whole object
back in a way that silently discards the correction. Net, the term is neither
added nor subtracted. Our port subtracted it. On eight networks the fixed port
reproduces the reference's error to four significant figures (`8.5075e-07`
against `8.5075e-07`) where the defective one gives `2.6890e-06` — `3.16x`,
against the `3.31x` the evaluator measured on twenty different networks.

Three things we would tell anyone porting a research implementation:

- **A degenerate limit tests the part of the program the degenerate limit
  reaches.** Ours set the third cumulant to zero, which made the defective term
  vanish from the one component the check inspected. Seven deliberately
  value-destroying mutations of the port all passed all three checks: **0 of 7**.
  That number was available months before the failure and we read it as a
  curiosity rather than a warning.
- **Agreement between two copies of your own transcription is not verification.**
  Both copies were wrong in the same way, and the check comparing them stayed
  green throughout.
- **Relative agreement in the output is not agreement in the score.** The
  defective port's predictions differ from the correct ones by `0.12%`. Had we
  gated on end-to-end drift with a `1%` tolerance — which is what we would
  naturally have chosen — the bug would have passed the gate at every network
  size we tried. **The predictions are `O(1)` and the error being chased is
  `O(1e-3)`; a difference invisible against the signal is enormous against the
  residual.** The gate that caught it compares the port against independently
  minted ground truth, not against another program.

The fix costs nothing and saves a little: the discarded term was, in a flow that
rebuilds its carried tensor each layer, **dead work** — the object it corrects is
thrown away at the rebuild. Removing it took `9,441,280` FLOPs and 208 array
operations out of every network, and left the answer correct.

> **The cheapest verification we now own is a 46 KB file: the reference
> implementation's own output on two small networks, committed and hashed. It
> turns every future run of the mutation battery into a real oracle inside the
> restricted environment, and it took the kill rate from 0 of 7 to 9 of 9 on the
> live code path. The framework barrier we had treated as a reason not to check
> only had to be crossed once.**

---

### 6.8 The evaluator confirmed the fix on 100 of its own networks — and then told us the binding cost was somewhere else entirely *(new 2026-09-03)*

**Evidence: CONFIRMED**, on the organisers' own labels, with a matched control in
the same run.

§6.7's fix was verified against our own minted ground truth. That leaves one risk
a self-check cannot bound: a defect shared by *all three* of our programs — the
reference implementation, the flow driver, and the network builder — would be
invisible to every comparison among them. So we ran the fixed port on the
evaluator, on 100 networks, with our shipped file as a control in the same run.

| 100 graded networks | raw MSE | adjusted | failed |
|---|---:|---:|---:|
| shipped second-order file (control) | `4.05e-06` | `4.05e-07` | 0 / 100 |
| **fixed third-order flow** | **`8.75e-07`** | **`8.75e-08`** | **0 / 100** |
| | **`4.63x` better** | | |

The control reproduced the figure we had carried for it since August **to
`0.01%`**, which is what licenses reading the other row. And the defective port
had scored `2.8192e-06` on twenty networks; the fixed one scores `8.75e-07` on a
hundred — `3.22x`, against the `3.16x` our offline paired measurement predicted.
**The shared-defect risk is now bounded everywhere, not just where our own
oracle could see, because the grader's labels are independent of all three of our
programs.**

**Then the same run told us something we had been measuring wrongly for weeks.**
Phase 2's residual limit is not a cost — it is a `0.4 s` per-network cliff that
*fails* the network. We had set ourselves a safety bar of `4x` headroom against
it. That bar was calibrated on a run-to-run spread of about `2.4x`, measured on
our own workstation.

On the evaluator, the per-network spread across 100 rows is **`1.20x` from
minimum to maximum**, with the worst network `13%` above the 99th percentile.

| per-network residual, 100 graded rows | seconds |
|---|---:|
| minimum | `0.1493` |
| median | `0.1537` |
| 99th percentile | `0.1593` |
| **maximum** | **`0.1797`** |
| headroom against the `0.4 s` cliff | **`2.23x`** |

**The bar was calibrated against noise the evaluator does not have.** Two things
follow, and they point in opposite directions, which is why we are writing both
down.

- The safety margin we demanded was set by *our* measurement noise rather than by
  the *evaluator's* dispersion. A `2.23x` margin against a distribution whose
  observed spread is `1.20x` is a different proposition from a `2.23x` margin
  against a `2.4x` spread, and we had been treating them as the same.
- **But our own instrument is worse than we thought, in exactly the direction
  that makes local measurement useless here.** Two runs of the *same* evaluator,
  on files differing by `2.3%` in operation count, reported mean residuals
  differing by **`31%`**. An operation count cannot do that. The runner is
  sensitive to host load, and the only defensible statement is the narrow one:
  *the evaluator is the only thing that can fail a network, and it failed none.*

**The lesson, and it is the one we would most want a reader to take from §6 as a
whole.** We spent weeks reducing a FLOP count that turned out to be free — the
score multiplier was already pinned at its floor, so arithmetic was not being
charged at the margin (§6.6). We then spent days measuring a wall-clock residual
on a machine that cannot measure it. **The quantity that actually binds is the
number of array *operations* per network**, and the evidence for that is the one
clean coefficient this project owns: the evaluator's mean-residual ratio between
our two files was `5.82x` where their local *operation* ratio was `5.98x` —
agreement to `3%`, where every attempt to establish the same thing from local
timings had been contaminated. **Count the operations, not the arithmetic, and
take the timing from the deployment authority or not at all.**

> **A `raw` report does not print the number a tail gate is written on.** Our
> requirement was on the *worst* network. The first graded run gave us an
> aggregate, and four hours of local residual measurement went into trying to
> reconstruct what one command-line flag (`--detail full`) returned directly.

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

**What we would tell the next person.** The open question this left was sharp and
cheap — the residual's rank collapse tracks the covariance's, which the flow
already holds for free, so if any of the residual's leading subspace lay inside
the covariance's, the selector would be free. **We ran it, and the answer is
exactly nothing.** The containment fraction equals `r/n` to three decimal places
at every depth, every rank, in both single and double precision, with a positive
control at `1.000` — that is, the covariance's dominant range contains the
residual's leading subspace *exactly as much as a random subspace of the same
dimension does*. Six selectors are now dead: five column-weight rules and the one
free subspace the flow already carries.

**So we stopped looking for a selector and asked whether the discarded object can
be REGENERATED.** A selector can only choose a subspace of something the flow
already holds, and the flow does not hold this object at all — it is created and
destroyed within a single layer. That is a different question, and it has a
mechanism: the discarded block's leading term is a specific quadratic Hadamard
form in the covariance under diagonal rescalings by functions of the mean and
standard deviation, both of which the flow *does* carry.

### 7.1 How deep is the discarded object's memory? *(new 2026-09-03)*

**Evidence: CONFIRMED**, pre-registered, and **the gate FAILED**.

The object discarded at layer `l` is a superposition of blocks generated at
`l-1, l-2, …`, propagated forward. If that sum is dominated by its youngest
terms, it can be rebuilt from a short window and carried cheaply. We measured the
fraction of its Frobenius energy that a `k`-generation memory recovers — as an
`R²`, so a window that recovered something orthogonal would get no credit — on 16
networks at the live shape, layer-averaged over the layers whose slice is
actually read downstream, with a double-precision control on two of them.

| generations kept | fraction of the discarded energy recovered |
|---|---:|
| 1 (age `l-1`) | `0.338` |
| **2 (ages `l-1`, `l-2`)** | **`0.573`** |
| 3 | `0.732` |

Per-network spread at two generations is `0.553`–`0.593`, so this is not a
sampling accident, and single and double precision agree to `1.6%`, so it is not
a precision artifact.

**Our pre-registered prediction said one generation would explain `≥ 50%` and two
would explain `≥ 70%`. Both halves are wrong**, and the direction came from an
earlier locality result whose single-layer share we had read as `0.42`–`0.70`; the
true single-layer share is `0.34`, half of what that range suggested.

**The gate was written on two generations and two generations give `0.573`. It
fails.** Three generations give `0.732`, which clears the number — and *moving the
gate to three because three passes is the exact post-hoc adjustment we had
already caught ourselves making once*, so we refused it in writing and recorded
the refusal in the same commit as the result.

**What the failure costs, in arithmetic rather than in evidence.** Carrying one
extra generation exactly doubles the rank of the carried tensor, which costs two
extra full matrix products per layer and breaks a symmetry collapse that saves a
third — about **five extra matrix products per layer against a margin of
`1.35`**. Three generations is worse. A low-rank sketch replaces each product
with a thin one at `r/n` of the cost, and the affordability horizon is `r ≤ 192`,
not the `512` we had assumed. So the mechanistic route survives only as *a
rank-`≤192` sketch of three generations recovering `~73%` of one of the two
discarded slices* — and **`73%` of that slice is about `44%` of the flow's total
error**, because the other discarded slice carries roughly `40%` of it and this
measurement says nothing about that one.

**We killed the route on that arithmetic, not on the measurement.** Either
discarded slice, corrected *exactly*, saturates at about `2.4x` end-to-end
improvement — we measured `2.53x` for one and `1.55x` for the other — so a route
that adds operations on the axis that is currently binding the whole submission
(§6.8) to recover a fraction of one of them is not a trade available to us. Its
closed-form generating terms survive as *features* of the supervised map that
replaced it, where they cost what an elementwise term costs rather than what a
carried generation costs.

> **The most useful thing this measurement did was fail.** It was designed so
> that a specific arithmetic consequence followed from a specific number, and the
> number came in where the consequence was fatal. That is worth more than a
> passing gate on a route we would then have had to price under pressure.


---

### 7.2 Can the discarded object be REGENERATED from what the flow carries? Not the cheap half, and a ceiling says so *(new 2026-09-03)*

**Evidence: CONFIRMED**, pre-registered, and **the gate FAILED** — at a ceiling
rather than at a fit, which is the stronger form of failure.

§7 and §7.1 close the *selection* route: six ways of choosing a subspace of what
the flow already holds, all dead. That leaves a different question. The flow does
not merely mis-select the discarded object — it never holds it at all; the object
is created and destroyed inside a single layer. So: can it be **rebuilt** from the
state the flow does carry?

There are two discarded pieces. One is an `n × n` matrix carrying roughly 60% of
the error; the other is an **`n`-vector** carrying roughly 40%. We started with the
vector, for three reasons that all looked good: it is `O(n)` so it is free in
arithmetic; correcting it *exactly* is already known to buy `1.55×`; and a
per-neuron map — neuron `i`'s correction from neuron `i`'s own carried scalars —
is **permutation-equivariant by construction**, which is the property every
previous learned component in this project died without.

We fit it properly. 128 freshly minted networks at the live shape, 11 layers,
1,024 neurons each — **1,441,792 labelled rows against 24 features**. Ridge, one
coefficient vector across all layers, leave-networks-out over 8 folds, the
regularisation strength chosen by an *inner* split on the training folds only, and
all five fitting choices written into the code's docstring before the labels
existed. The labels themselves are exact — they are the difference between two of
our own flows, and the labeller asserts both endpoints (`0` and `1`) to the last
bit on every run.

**Held-out `R²`: `0.10`, against a pre-registered bar of `0.90`.** Negative at the
shallowest layer — worse than predicting zero. Our written prediction had been
`≥ 0.95`.

A failed fit has two very different explanations, and the gate cannot tell them
apart: the features do not explain the target, or they explain it *within* a layer
and one fixed coefficient vector cannot span the layers. The second would be
cheaply fixable. So we measured the ceiling — a coefficient vector fitted **on the
very block it is then scored on**. In-sample by construction, so it is not a
result; it is the most any map built from these features could ever achieve.

| | layer-averaged |
|---|---:|
| held-out, one vector for all layers (**the gate**) | `0.10` |
| held-out, one vector per layer | `0.16` |
| **in-sample ceiling of the 24-feature basis** | **`0.26`** |

**The basis is the problem, not the fit.** No amount of per-layer freedom,
regularisation, or data reaches `0.90` from a ceiling of `0.26`.

**And the mechanism was already written down, one run earlier, in the right
words.** The discarded block's own contribution to the vector slice is **exactly
zero**: its three factors are built from the off-diagonal covariance, whose
diagonal is zero by definition, and the slice reads exactly that diagonal. So the
discarded mass does not appear in the vector at the layer that generates it — **it
arrives there only after a linear step mixes the indices**, as a triple Hadamard
product of three *differently* weight-transformed matrices. That is a third-order
functional of the weight row against the previous layer's covariance, and the
covariance the flow carries at that point does not determine it. Two networks can
agree on everything the flow holds and disagree on the correction.

We had recorded that zero in the module docstring as a hint about *where to take
the features from*. It was in fact a statement that **the pointwise information is
insufficient in principle**, and we read it as the weaker thing.

Three things worth carrying out of this:

- **Equivariance being free is not the same as the feature set being
  sufficient.** Our argument went "the target is a vector, so a per-neuron map is
  equivariant by construction, so a per-neuron map is the right object". The first
  two clauses are true and the third does not follow. Equivariance is a constraint
  a map must satisfy, not evidence that a map exists.
- **Measure the ceiling before you tune the fit.** One extra command on data we
  already had converted "our fit is bad" into "no fit built from these features
  can work", which is a claim about the problem rather than about our effort. It
  is also the only version of the result that is worth writing down.
- **The fit gets BETTER with depth** — `0.045` at the shallowest layer to `0.28` at
  the deepest — which is the opposite of the usual intuition and is consistent
  with the missing information being weight-history that the carried state slowly
  accumulates.

**What this costs the plan — and a correction we published for four hours.**

The first version of this section closed by predicting the end-to-end
requirement would go unmet, on the argument that *"a fitted map cannot beat its
own oracle; the two oracles are `2.53×` and `1.55×`."* **That arithmetic was
wrong, and our own measurement table said so before we wrote it.** Those two
figures are the *single-slice* oracles — correcting the matrix alone, or the
vector alone. The cell that matters is the one where **both** are corrected:

| oracle correction, 8 networks, paired | end-to-end improvement | per-network range |
|---|---:|---|
| vector only, exact | `1.55×` | 1.42–1.65 |
| matrix only, rank 256 | `2.38×` | 2.14–2.67 |
| **both, rank 128** | **`11.1×`** | **9.86–12.63** |
| both, rank 256 | `32.2×` | 28.0–37.1 |
| both, rank 512 | `334×` | 302–373 |

**Correcting either piece alone saturates near `2.4×` however good the
correction; correcting both at rank 128 gives `11×`.** We had written exactly
that sentence in our own log a day earlier, and then contradicted it by reading
a single-slice row as if it were the joint one. An independent reviewer caught
it. It is left in the record rather than quietly fixed, because the failure mode
is the interesting part: **a component ablation whose parts do not sum to the
headline is a signal that you have mis-read your own table**, and the arithmetic
was load-bearing for a decision about whether to keep going.

So the honest position is narrower than either version. What is measured:

- the **joint** oracle clears a `10×` requirement at rank 128, on 8 of 8
  networks, with the affordable rank;
- a **pointwise map** from the flow's carried per-neuron scalars to the vector
  reaches an in-sample ceiling of `0.26`;
- and — the reviewer's finding, not ours — **the vector is the diagonal of the
  same discarded object whose zeroed off-diagonal is the matrix**. Reading the
  diagonal off a rank-192 reconstruction of the matrix, with *one* fitted scalar,
  recovers `0.61`–`0.66` of the vector: **2.5× better than our entire
  24-feature basis, at no extra cost, using no new features.**

What is **not** measured, and what we will not predict again without measuring:
what an *imperfect* matrix correction does to that `11.1×` joint cell. Every
number above is an oracle. The one experiment that settles it is cheap and
specified: correct the matrix with its rank-`r` truncation and the vector with
the **diagonal of that same reconstruction**, and read the end-to-end result.

**The lesson we would hand over is the one we needed twice.** In §6.7 a
degenerate-limit check passed a defective port because the defect vanished in the
limit. Here a component table was read one row off. Both times the correct
information was already in a file we owned. **Verification is not only about
building the right check — it is about reading the check you already built.**


---

### 7.3 The lane, closed four ways — and the sensitivity that explains why it was never close *(new 2026-09-03)*

**Evidence: CONFIRMED**, pre-registered, and **the falsifier fired.**

§7.2 closed one route. Two more measurements closed the rest, and together they
turn "we could not make it work" into something a reader can use: **the state
this class of flow carries does not determine what it discards, and here is the
ceiling on each of the four ways we tried to get it back.**

**Can the matrix be predicted from the carried state?** We fit it over 29
permutation-equivariant terms — Hadamard powers of the covariance, diagonal
rescalings by the ReLU Wick coefficients, the carried third-order slice, outer
products, and six matrix-product terms — on 128 networks, leave-networks-out, the
basis committed to version control before a single inner product with the target
was computed.

| | layer-averaged, held out |
|---|---:|
| the fit | **`0.14`** |
| the basis's own in-sample ceiling | **`0.17`** |

The fit is within `0.03` of the most any map built from those terms could
achieve. **The shortfall is representational, not statistical.** We checked the
ceiling was not an artifact of the solver — three methods and five truncation
settings agree to six decimals, and the design matrix is full rank.

**And the expensive half of the basis buys nothing.** Our compute budget permits
at most one matrix-product term per layer; each costs a full `2n³` while a
Hadamard or diagonal term costs about a hundredth of that. So we ran the fit
eight ways. **At full precision the eight answers differ in the sixth decimal,
and two of the six expensive terms make it worse.** They correlate 0.23–0.91
with cheap terms already in the basis — they are not new directions. We had
spent real effort deriving how many expensive terms we could afford; the answer
turned out not to matter.

**Why none of it was ever close.** The decisive number is a sensitivity, measured
on 8 networks at the live shape with the *matrix* correction held at oracle:

| quality of the vector correction | end-to-end improvement |
|---|---:|
| absent | `2.2×` |
| the best free route we found (`R² = 0.45`) | `3.7×` |
| exact | `18.9×` |

**Going from a 0.45-quality vector correction to an exact one multiplies the
end-to-end gain by five.** The curve is steeply convex: the last half of that
object's variance is worth four times the first half. A `10×` end-to-end
requirement therefore needs *both* pieces near-exact — and we can predict neither
above `0.15`.

That is the whole story, and it is why the four ceilings matter more than the
four failures:

| route | outcome |
|---|---|
| select a subspace of what the flow already holds (5 rules) | **1.45×–2.10× worse than not correcting** |
| select using the covariance's dominant range | containment equals `r/n` to three decimals — **chance** |
| predict the vector pointwise from carried per-neuron scalars | held out `0.10`, **ceiling `0.26`** |
| predict the matrix from 29 equivariant terms | held out `0.14`, **ceiling `0.17`** |

**One thing that did work, and should not be lost.** The two discarded objects are
the diagonal and the zeroed off-diagonal of a single tensor. Reading the diagonal
off a rank-192 reconstruction of the off-diagonal — **one fitted scalar, no new
features, on an object the matrix correction has already materialised** —
recovers `R² = 0.45`, with a fitted scale of `1.019`, i.e. essentially unbiased.
It is also non-destructive: better than no correction on **8 of 8 networks at
three ranks**, which mattered because an earlier experiment of ours had found
that injecting exact information mid-flow can make things *worse*. That component
is free and correct. It is unusable only because it needs the matrix we cannot
predict.

**What we would tell the next person.** Three things, in order of how much they
cost us:

1. **Measure the ceiling, not just the fit.** Both of our learned components died
   at a ceiling, and a ceiling is a claim about the problem where a poor
   held-out score is only a claim about your effort. Ours came almost free: for a
   basis linear in its coefficients you never need to store the target, only its
   Gram matrix and moments, and the ceiling is then a closed form.
2. **A negative result needs a sensitivity to be worth reading.** "Our map got
   0.14" is not useful. "The end-to-end requirement needs both pieces near-exact,
   and here is the measured curve showing how fast quality converts into result"
   is a thing someone else can plan against.
3. **Check which half of your cost model is binding before you optimise it.** We
   derived an elaborate constraint on how many expensive terms we could afford.
   The expensive terms turned out to contribute nothing, twice over — the same
   shape of error as §6.6, where we spent weeks reducing a FLOP count that the
   score was not charging us for.



### 7.4 The complete result: the target is reachable at an affordable rank, and the carried state does not determine it *(new 2026-09-03)*

**Evidence: CONFIRMED.** This section states what §§7.1–7.3 add up to. We think
it is the most transferable thing in the paper, and it is a *positive*
statement about the structure of the problem that happens to be delivered by
four negative experiments.

**The two discarded objects are one object.** The flow throws away an
all-distinct third-order block. What the scored means actually read of it are
two slices: a vector (the diagonal of that block) and a matrix (the same block
with its diagonal zeroed). We verified this against the upstream reference
rather than assuming it — the library's own slice accessor returns exactly the
diagonal, to `0.0`, not to a tolerance.

**Corrected jointly, at a rank we can afford, the target is reached.** With
*both* slices supplied exactly and the matrix truncated to rank 128 of 1024 —
one eighth — the end-to-end drift falls by **`11.1×`**, on 8 networks at the
live shape, paired:

| what is corrected, at rank 128 of 1024 | end-to-end |
|---|---:|
| nothing | `1.0×` |
| the matrix only | `2.1×` |
| **both slices** | **`11.1×`** |
| both slices, rank 192 | `18.9×` |

**This is the number that makes the negative result interesting.** Every other
compression lane we closed (§5) was closed because the *thing itself* was not
compressible at a price we could pay. This one is not. A rank-128 sketch is
about `2nr = 0.26M` numbers against the block's `n² = 1M`, it costs a fraction
of our FLOP margin, and it delivers an order of magnitude. **The lane is not
blocked by rank, and it is not blocked by cost.**

**It is blocked by predictability, and that is a statement about information,
not about effort.** Everything the flow still carries at the moment of the
discard — the mean vector, the covariance, and the two repeated slices it
regenerates from — was given to two different learned maps:

| target | best held-out | **the basis's own ceiling** | what the ceiling is over |
|---|---:|---:|---|
| the vector | `0.10` | **`0.26`** | 24 per-neuron features, +300 pairwise products, +10 row contractions |
| the matrix | `0.14` | **`0.17`** | 29 permutation-equivariant terms, including 6 at full `2n³` |

Both maps land within `0.03`–`0.16` of the most *any* map over those terms could
achieve. Adding features moved the vector's honest cross-validated score from
`0.292` to `0.306`. The expensive terms moved the matrix's by `~1e-6`, two of
them negatively.

So, stated as plainly as we can:

> **A regenerating cumulant flow's carried state does not determine what that
> flow discards.** The discarded object is reachable — an eighth-rank sketch of
> it is worth an order of magnitude end to end — and it is not recoverable from
> what remains. Those two facts are not in tension; together they say the
> information is *destroyed* by the regeneration, not merely *compressed* by it.

**Why we believe this rather than suspecting it.** A poor held-out score is a
claim about the modeller. A ceiling is a claim about the problem, and we
computed one for each map, five and three ways respectively, on artifacts rather
than on a summary. The ceilings agree to six decimals across solvers and
truncation settings, and the design matrices are full rank with no silent
truncation. We also checked the failure is not a *harmful* correction rather
than an absent one: the injection is better than no correction on `8/8` networks
at three ranks, so what we built is not wrong, it is not there.

**What this predicts, and what we did about it.** If the information is
destroyed rather than hidden, then the only way to have it is to **keep** it —
and keeping is a different problem from predicting, with a different cost. We
measured how deep the memory runs (§7.1): three generations of the discarded
block carry `0.73` of its energy at the live shape, and the increments
(`0.34 / 0.23 / 0.16`) say the tail is diffuse rather than long. **§7.5 builds
that flow and measures it.** It is the only one of our five routes that
recovers the vector half at all — and it reaches `3.87×` where we needed `10×`.

**The one caveat that travels with all of it.** Every joint number in this
section uses an *oracle*: the true slices, truncated. They are ceilings, not
operating points, and we have been careful throughout not to quote a ceiling as
a result — an earlier draft of ours did exactly that, cited a single-slice cell
as if it were the joint one, and had to be withdrawn (§9).



### 7.5 The fifth route: *keeping* what the flow discards, and the ceiling that closes the section *(new 2026-09-04)*

**Evidence: CONFIRMED**, pre-registered, and independently audited — the audit
changed the answer, which is why it is reported here in full.

§7.4 ended on an open question: if the carried state does not *determine* the
discarded object, the only way to have it is to **keep** it. Keeping is a
different problem from predicting, with a different cost, and §7.1 had measured
that three generations of the discarded block carry `0.73` of its energy.

So we built it: a flow that keeps the last three generations, compresses the
matrix half to rank 192 of 1024, and injects both halves back. Pre-registered
with its gate, its falsifier and its predictions committed before any data
existed, and with one commitment that turned out to matter — **we said we would
report it either way.**

**The first answer was wrong, and an independent auditor found out why.** Our
scored cell reconstructed the *vector* half from leakage in the compressed
*matrix* half, rather than simply keeping the vector — a mechanism we had
imported from an earlier experiment where the vector genuinely was unavailable.
Here it costs `n` numbers and we already had it. The auditor spotted that the
carried vector sat unused two lines above the code that discarded it, fitted a
response surface to the run's own cells to predict what the corrected version
would give (`3.18×`), and told us not to publish until we had measured it.

| construction, rank 192 of 1024 | end-to-end |
|---|---:|
| no correction | `1.00×` |
| a random rank-192 sketch (null) | `1.07×` |
| **compress the combined object and read both halves back off it** | **`0.78×` — worse than nothing** |
| reconstruct the vector from the matrix's leakage *(our first answer)* | `2.15×` |
| **keep the vector, compress only the matrix** | **`3.33×`** |
| keep both exactly — the ceiling on the whole family | **`3.87×`** |

Measured `3.33×` against the auditor's predicted `3.18×`. Keeping the vector is
worth `2.15× → 3.33×`, better on **16 of 16** networks, `t = −20.6`, for an
object that costs `n` numbers a layer.

**Three things in that table are worth more than the headline.**

**One: the compression is not the problem.** Compressing the matrix half to
one-eighth rank costs `0.08` of its recovery. The vector needs no compression at
all. Our first answer looked like a story about low-rank approximation destroying
a diagonal, and it was a story about our own cell throwing that diagonal away.

**Two: the obvious construction is actively harmful.** Compressing the *combined*
object — the thing that most deserves the name "a rank-`r` sketch of what we
discarded" — lands at `0.78×`, **worse than making no correction at all**, with
the vector half recovered at `R² = −1.70`. The combined object is dominated by
its matrix part, so a rank-`r` budget spends itself there and returns a vector
that is confidently wrong. Two quantities that are read separately must be
carried separately. This is the second time in this project that injecting
*correct* information mid-flow has made the answer worse, and we would not have
believed it the first time either.

**Three, and it is the section's real conclusion: the ceiling is `3.87×`, and it
is a ceiling on the whole family.** That cell keeps both halves *exactly* — no
compression, no estimation, no fit. It is what a three-generation memory can
possibly be worth. Our requirement was `10×`.

So the carried route is the only one of the five that **works** — it is the only
route that recovers the vector half at all, at `R² = 0.79` against `0.10` and
`0.14` for the two learned maps — and it still falls short by a factor of three.
**The binding limit is the depth of the memory, not the compression and not the
predictability.** A fourth generation is worth about `0.10` of the energy on our
measurements, and the end-to-end response is convex in quality, so buying the
missing factor of three this way would mean carrying most of the history — which
is the uncompressed object we were trying to avoid.

**The five ceilings, which is what we would want a reader to take away:**

| route | ceiling |
|---|---|
| select a subspace of what the flow already holds | `1.45×`–`2.10×` **worse than not correcting** |
| select using the covariance's dominant range | containment `= r/n` — **chance** |
| predict the vector from carried per-neuron scalars | `R² = 0.26` |
| predict the matrix from 29 equivariant terms | `R² = 0.17` |
| **keep three generations of both, exactly** | **`3.87×` end-to-end** |
| *(what `10×` needs: the vector near-exact)* | *`18.9×`* |

**What we would tell the next person, and it is a different lesson from §7.4's.**
Our pre-registration fixed the gate, the falsifier, the statistic, the block and
the predictions — everything except *whether the cell we wrote implemented the
hypothesis we stated*. It did not, and every check we had passed anyway: the
instrument self-test passed, the in-flight algebra probe passed at `3e-08`, the
null sat at chance, an independent cross-run anchor agreed to `0.5%`, and the
promotion statistic was clean on 64 paired networks. **Nothing was broken.** The
cell was a correct implementation of the wrong construction, and no amount of
verification can see that — only an adversarial reader who asks what the code
does with each object it was handed. We now write the check into the protocol.


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
- **A cost model assumed `2 n^3` per matmul, and we "corrected" it to `4 n^3`.**
  **Both statements are incomplete and the correction was itself wrong.** The
  meter bills `2 n^3` and scales by a per-dtype rate, so the price is `4 n^3` in
  `float64` and `2 n^3` in `float32`; we had measured the `float64` column and
  written it down as a property of matrix multiplication, then compared it
  against accuracy evidence gathered in `float32`. **Two errors in two days on
  the same number, in opposite directions**, and the second one is the one that
  nearly bought us a 1969 algorithm we did not need. (§6.1)
- **A composed cost projection was 3.3% optimistic, and we called it a solved
  problem before executing it.** An independent review refused the conclusion on
  exactly that ground — two spellings never run as one program — and it was
  right to. Writing the program moved `0.07636` to `0.07886` and the free margin
  from `1.51` to `1.35` matmuls per layer. **The conclusion survived; the
  entitlement to state it did not exist yet.** (§6.5)
- **We optimised the free half of the objective for a day.** The cost exponent
  is zero this Phase, so FLOPs below the floor are worth nothing, and the only
  cost that can still ruin a run is a wall-time residual that tracks **operation
  count** rather than arithmetic. Our flow does `2.5x` the arithmetic of our
  shipped file and `11.3x` the operations, and it reached the real evaluator with
  `1.21x` margin against a cliff that zeroes a network's prediction — and the
  local measurements we took to reason about it moved by 55% depending on what
  else the machine was doing. Every FLOP number in §6 is correct and most of
  them did not matter. (§6.6)
- **We shipped a port to the evaluator that we had never compared against the
  implementation whose accuracy we were quoting.** It scored `3.3x` worse. The
  cause was one line, in code that a rebuild makes dead. Three checks passed
  throughout, and a mutation battery had already reported that **0 of 7**
  deliberate value-destroying edits were detectable — a warning we filed rather
  than acted on. The barrier was that the two implementations need different
  Python environments; crossing it once, and committing 46 KB of the reference's
  output, took the battery to **9 of 9** on the live code path. (§6.7)
- **We wrote a verification gate on the wrong quantity, and only a second clause
  saved it.** The natural gate — "the two programs' predictions agree to `1%`" —
  passes the defective port at every network size we tried, because the
  predictions differ by `0.12%` while the score differs by `3.2x`. The gate that
  worked compares against independently minted ground truth. **When the residual
  error is orders below the signal, agreement in the output is not agreement in
  the objective.** (§6.7)
- **Our own written mechanism was off by a factor of two, and the measurement
  corrected it.** The pre-registered hypothesis said the defect made one carried
  object wrong by twice a named quantity; it is wrong by exactly once, measured
  to six digits. Location, mechanism and sign were right, the factor was not —
  which is an argument for pre-registering the *mechanism* and then measuring the
  *magnitude*, rather than trusting either alone. (§6.7)
- **A NaN reference passed a threshold gate, and the gate printed PASS.** The
  quadrature we checked one component against overflows past a certain degree and
  returns NaN weights; `nan > tolerance` is `False`, so three failed comparisons
  reported success. Separately, the same reference was the *wrong quadrature
  rule* — Gauss-Hermite converges only algebraically against a kinked integrand,
  and at moderate degree it disagreed with an exact closed form by a thousand
  times the tolerance, which reads exactly like a bug in the thing under test.
  **Two lessons: reject non-finite input explicitly before comparing, and when a
  check fails, ask whether the reference is what is wrong.** (§6.5)
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
