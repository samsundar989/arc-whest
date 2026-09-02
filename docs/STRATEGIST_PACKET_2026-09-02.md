# Strategist Packet — ARC White-Box Estimation Challenge, Phase 2

**Written 2026-09-02. Self-contained: assumes no repository access and no
shared chat history.** Every number here is measured and traceable to a
committed artifact; where something is an assumption or a proxy it says so.

**The decision requested is in § 7.** Sections 1–6 are the evidence needed to
make it.

---

## 1. The problem, exactly

Given a randomly initialised, bias-free ReLU MLP with He initialisation —
**width 1024, depth 16** — predict every neuron's expected post-ReLU activation
under `x ~ N(0, I)`, as a `(depth, width)` array.

**Score, lower is better:**

```
score = final_layer_MSE  ×  max(0.1, F / 2**41)
```

where `F` is the FLOP count of your `predict()` as billed by the organisers'
`flopscope` instrumentation. `2**41 = 2,199,023,255,552`.

Hard constraints:

| constraint | value |
|---|---|
| FLOP budget `B` | `2**41` |
| multiplier floor | `0.1` — the first `219,902,325,555` FLOPs are **free** |
| residual wall time | `0.4 s` per MLP — a **cliff**: exceeding it zeroes that MLP's prediction *and* forces its multiplier to `1.0` |
| participant wall | `120 s` per MLP |
| setup | `5 s`, once |
| memory | `8192 MB` |
| submissions | 10 per team per UTC day |
| imports allowed | `flopscope`, `flopscope.numpy`, `whestbench`, Python stdlib. **No numpy, scipy or torch** |
| closes | **2026-10-17**, 45 days from now |

Winners are decided by a **private re-run of up to two designated submissions
per team on fresh MLPs**, not by public-board rank.

### The one structural fact that governs everything

Because the multiplier is `max(0.1, F/B)`, there are two regimes:

- **Below `F = 0.1B`** the multiplier is pinned. FLOPs are *free*; only
  accuracy matters.
- **Above it** the multiplier is exactly `F/B`, so the score is
  `MSE × F / B` — cost enters **linearly**.

## 2. Where this entry stands

| | value |
|---|---|
| best graded submission (v2.7) | **`1.3738e-7`** adjusted / `1.3738e-6` raw |
| its multiplier | `0.1` — pinned at the floor, `F = 137,600,326,656` (`C/B = 0.0626`) |
| public rank | **53 of 83** (2026-08-31 snapshot) |
| board leader | **`4.90e-9`** adjusted (raw `2.27e-8`, multiplier `0.216`) |
| **gap to leader** | **`28x` in adjusted score, `61x` in raw MSE** |
| free FLOPs currently unused | `82.3e9` — `37%` of the free zone |

The shipped estimator is a **Gaussian-closure second-cumulant propagator**
("K2"): it propagates a mean vector `mu` and a full covariance `Sigma` layer by
layer, applying the exact rectified-Gaussian mean/variance map to the diagonal
and a first-order Price-series truncation `Phi_i Phi_j c_ij` to the
off-diagonal. Plus sixteen fitted per-layer output constants (a `2.86x` gain,
verified on the real grader).

**Leaders are not samplers.** Their raw error is flat across their budgets,
which is the signature of an analytic method whose accuracy does not improve
with FLOPs. A sampler's raw error would fall as `F^-1`.

## 3. What is CLOSED, and the measured reason for each

This is the most valuable part of the packet: eleven lanes have been closed on
measurements, not on intuition. **Do not propose these again without new
external evidence.**

### 3a. The whole per-neuron correction family — closed 2026-09-02

Prior result: **94.6%** of K2's error is accumulated `(mu, sigma)` state drift;
only `4.5%` is the final readout map. So "correct the state per neuron" looked
like the obvious lever. All three channels are now closed:

| channel | result |
|---|---|
| **readout** (correct the final layer) | **Exhausted.** The shipped file's *single* per-layer constant predicts the per-neuron residual at `R^2 = +0.51`. A 354-parameter neural network on eight scale-invariant features, trained on 72 networks, matches it and does not beat it (`0.9945x`, `t` CI straddling zero) |
| **in-flight variance** | **Destructive.** Replacing the propagated post-ReLU variance with its *true* value, at **zero fit error**, is **17x worse** than no correction at all |
| **in-flight mean** | **Real ceiling, unreachable price.** The true-mean oracle is **`6.87x` better** than the shipped file. But break-even needs held-out **`R^2 >= 99.85%`**; best achieved is `+0.50` on an easier target |

**Why the variance channel is destructive** — this generalises and should be
assumed for any future state method: *the Gaussian closure is an internally
consistent approximation, not a set of independently improvable marginals.*
Making the variance exact while the off-diagonal stays truncated at Price order
1 breaks a cancellation. The variance enters the next layer through
`diag(W^T Sigma W)`, summed against the very off-diagonal terms that are wrong;
the mean enters linearly through `W^T mu` and does not.

### 3b. The amplification law — a reusable property of this estimator

Injecting a **known** perturbation into the propagated state and measuring the
score gives, across three decades and in both regimes, **excess MSE ∝ c²** —
the final-layer error is **linear** in in-flight state error. The tower is an
amplifier, not a chaotic system:

| perturbation | amplification (final RMS error / injected, absolute) |
|---|---|
| coherent (same shift every layer) | **~18x** |
| incoherent (independent per neuron per layer) | **~4.2x** |

Their ratio, `4.3`, is `sqrt(14)` for 14 corrected layers — coherent
accumulation, exactly as it should be. A positive control at zero perturbation
reproduces the baseline to `0.000e+00`.

**Consequence:** any method that computes the state *approximately* pays this
amplification. Accuracy requirements on intermediate state are far tighter than
intuition suggests.

### 3c. Sampling / quadrature carriers — closed on the score identity itself

A structured-quadrature carrier (Kerdock/MUB point sets) was the other main
candidate family. Its measured error obeys `MMD² = C1/k + C0` — the plain
Monte-Carlo rate — and this was proven to be an **exact algebraic identity**,
not an empirical fit. Its cost is linear in `k`. Therefore, above the floor:

```
adjusted  ∝  (C1/k) × (c·k)  =  C1·c      — independent of k
```

Measured over a 4× cost range: the adjusted score is flat to **0.62%**.

**A pure sampler cannot climb this leaderboard.** Its error falls at exactly
the rate its cost rises. It can only choose a constant — and the structured
(Kerdock) advantage over plain random-orthogonal blocks at width 1024 measures
**0.155%**.

This also explains the leaders' flat raw-vs-budget curve from outside: they are
not samplers, because samplers have nothing to gain from a budget.

### 3d. Third-cumulant (K3) methods — priced out, twice

Full factorised third-cumulant propagation reaches raw MSE **`3.92e-8`**
(confirmed on the organisers' own data, against a label-noise floor of
`7.5e-11`). **Delivered at the `0.1` floor that would score `3.9e-9` and lead
the board.**

It costs `10.7x` too much:

| variant | cost | outcome |
|---|---|---|
| exact sparse/equality-aware spelling ("W-EFF") | **`1.073 B`** measured under the official meter | vs a pre-registered `<= 0.35 B` screen — **fails by `3.07x`** |
| rank-capping the factored tensor | `0.083–1.15 B` | **accuracy falls faster than cost.** At rank `n` the "compressed K3" is `14x` closer to plain K2 than to K3 — it *is* K2. Structurally so: capping below `n` deletes whole neurons' `kappa_3` diagonal. It is coordinate pruning, not compression |
| late/age-windowed K3 (expensive machinery only in the last `m` layers) | `m<=3` fits free | **every window arm's provable score floor is `2.1x`–`4.7x` worse than the shipped file.** And `89%` of a window's error is its cheap prefix, not the missing memory |

**The mechanism of the compression failure is understood and is the useful
part:** `kappa_3` *is* compressible — a rank-`n` approximation captures `96.7%`
of the Frobenius norm — but **the flow reads it only through two diagonal
slices**, and norm-optimal compression spends its rank on the all-distinct bulk
that carries most of the norm and none of the readout. The open question was
never "is it low-rank" but "is there a compressor whose error is *orthogonal to
those two slices*". No affordable such mechanism has been found; the best
oracle-selected rank-`n` object still drifts `14.5x` the full-K3 MSE, and the
oracle's own selector costs four orders of magnitude more than the object it
selects.

### 3e. Smaller closed lanes

- **Per-layer scalar calibration:** exhausted at `2.96x` (already shipped). Any
  purely multiplicative state correction is provably one output rescale in
  disguise — the closure is exactly positively homogeneous of degree 1
  (verified to `1.6e-30`).
- **Higher Price/Hermite orders in the covariance map:** plateau at `1.05x`.
- **Marginal Edgeworth correction of the final ReLU map:** capped at `1.047x`.
- **Plain IID and random-orthogonal direction sets:** lose to K2.
- **Query-directed / adjoint K3:** open but speculative, with a known-broken
  instrument (its kill was retracted when the statistic was shown to kill a
  provably linear system).

## 4. What is OPEN

1. **"Add information" — the off-diagonal.** K2's first approximation at width
   1024 is the Price order-1 truncation of the post-ReLU off-diagonal
   covariance. It is the error source that generates the accumulated drift, and
   the per-neuron family closed *because* it was trying to compensate
   downstream for an error generated there. **It has no label** — the truth
   generator produces per-layer means and variances, i.e. the diagonal. A
   corrector cannot be fitted by regression on a target that does not exist; it
   needs either true covariances (a `1024 × 1024 × 16` label per network) or an
   end-to-end gradient through the 16-layer rollout.
2. **Faithful augmented-K2 (`AUGMENT`).** Propagates a degree-3 `(n,)` vector
   and a degree-4 scalar alongside `(mu, Sigma)` — `1,025` extra numbers on
   K2's `1,049,600`. **Measured cost: `1.037x` K2, i.e. free at the floor.**
   Its accuracy is **unmeasured at the live shape** (`1.8%` movement at width
   64, sign unknown). No implementation exists in the grader's restricted
   language; that port is a multi-day effort with a substantial parity burden,
   and an independent review flagged that it costs `4.4x` the operation count,
   which matters for the residual cliff.
3. **Adaptive per-MLP budget.** The score is a per-MLP mean with a *per-MLP*
   floor, so cheap networks can subsidise expensive ones. Unexplored.
4. **Harvesting the Phase-1 winner's code** when its mandated open-sourcing
   appears (signed frame reweighting, reported ~5x on a cubature core).

## 5. Assets that are strong

- **The offline-to-graded transfer is verified.** Constants fitted on
  self-minted networks projected a `2.96x` gain; the graded server delivered
  `2.86x` — agreement to `1.9%`. Offline fitting works and is trustworthy.
- **The local FLOP meter reproduces the official one operation for operation,
  to zero**, across 19 operations. (The grading *server* adds an unexplained
  `+2**24` per MLP, `+0.0122%`, which is modelled and does not bind.)
- **The local score is an unbiased proxy for the graded score** — the apparent
  gaps are `0.90` and `0.57` sigma of sampling error.
- Substantial verified infrastructure: grader-identical network minting, a
  three-tier evidence protocol with an append-only seed ledger, an executable
  promotion gate with measured false-promotion rates, and a certification
  battery.

## 6. The two structural facts worth carrying into any new idea

1. **Monte-Carlo-rate error buys nothing.** Any estimator whose error falls as
   `1/F` has an adjusted score *independent of `F`* above the floor. Only
   methods whose error falls **faster than `F^-1`** — or that fit entirely
   inside the free zone — can move the score.
2. **Intermediate state error is amplified `4–18x`.** Approximating the state
   more richly is not enough; it must be approximated *accurately*. Convert
   every proposed lane's oracle ceiling into an **accuracy requirement**
   before building it. Doing so costs one sensitivity sweep and has now
   retired two lanes that looked promising.

## 7. The decision requested

**The shipped entry is `28x` off the lead. Every lane this project has
explored is closed except two, and neither obviously reaches `28x`.**

The specific questions:

**Q1 — Is the third-cumulant lane worth one more mechanism attempt, and if so
which?** It is the *only* explored lane whose accuracy is known to be
sufficient (`3.92e-8` raw → `3.9e-9` adjusted at the floor → would lead). It
fails purely on cost, by `10.7x`. The compression question has been sharpened
to: *is there a rank-reduction of the third-cumulant tensor whose error is
orthogonal to its `(3,)` and `(2,1)` diagonal slices, computable at
`O(n^2 L)`?* Norm-optimal compression provably is not it. Is there a known
construction that is — a slice-preserving sketch, a structured/Kronecker
factorisation, a randomised method with the right invariance — or is this a
research problem too large for 45 days?

**Q2 — Is the off-diagonal correction reachable without a label?** The missing
information is concentrated in a quantity that cannot be supervised directly.
Is an end-to-end gradient through the 16-layer rollout (differentiating the
final-layer MSE w.r.t. a parameterised off-diagonal correction) the right
approach, or is there a cheaper analytic characterisation of the order-1
truncation error that could be corrected in closed form?

**Q3 — Given 45 days, what is the correct allocation?** Concretely: pursue a
differentiator that might reach the front and might produce nothing, or
consolidate — harden what exists, prepare two robust nominations, and invest in
the write-up (a co-equal prize, whose stated wishlist is exactly compression,
learned cumulant features and floor-targeting)? Several results in § 3 are
publishable negatives with clean mechanisms.

**Constraints on any answer:** the grader's restricted import list means every
proposal must be expressible as dense array operations in a numpy-like API with
no external libraries; anything requiring an eigendecomposition in the scoring
path is disfavoured (prior numerical-ambiguity problems); and a candidate is
worthless without measured residual and memory headroom, which historically has
been where promising methods died.

---

*Deliver the answer as a plan that can be turned into a pre-registered
experimental contract: a hypothesis with a mechanism, the smallest
discriminating intervention, a numeric gate, and the falsifier.*
