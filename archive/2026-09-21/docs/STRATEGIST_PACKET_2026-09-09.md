# Strategist Packet — 2026-09-09: AUGMENT reaches the leaders' floor. What gets us below it?

*Self-contained. Assumes no repository access and no shared chat history.
Supersedes `STRATEGIST_PACKET_2026-09-08.md`, whose question ("is `1.9e-8` a
property of the class?") has been **answered** — see §3. Every number below is
measured unless marked PROJECTED or ESTIMATE.*

---

## 1. The contest, in one page

**Task.** Given the weights of a randomly initialised He-normal ReLU MLP — Phase 2
is **1024 wide × 16 deep, no biases** — predict every neuron's expected post-ReLU
activation under `x ~ N(0, I)`, returned as a `(depth, width)` array. No sampling
restriction: any compliant method is allowed, but the organisers expect the best
methods to be mechanistic.

**Score, per MLP, lower is better:**

    score = final_layer_MSE × max(0.1, FLOPs / 2^41)

FLOPs are counted analytically by `flopscope`, a numpy drop-in. λ is zero this
phase, so wall time is not billed — but there is a **120 s per-MLP wall cap** and
an **8 GiB memory cap**, and an MLP that exceeds either is scored as **zeros at
multiplier 1.0**, a ~`3e5` penalty. The leaderboard is the mean over a hidden
100-MLP suite.

**Prizes.** `$50k/$20k/$10k` for best score, plus a **co-equal `$20k` for best
algorithmic contribution**, judged on a technical write-up together with score.

**Decided by a private re-run.** Up to two designated submissions are re-run on
**fresh, unseen MLPs**. Public board rank is not the target.

**Dates.** Submissions close **2026-10-17 23:59 UTC**. Write-up due **2026-10-24**.
Today is 2026-09-09, so **~5.5 weeks of submission runway**.

---

## 2. Where we stand

**Public board:** #1 is **`3.80e-9`** adjusted. A cluster of **six teams sits at
`1.87–2.13e-8` raw** final-layer MSE — the "floor" this packet is about.

**Our three graded submissions** (all `0/100` failures):

| id | what | adjusted | raw | multiplier `B` | ops | worst wall |
|---|---|---:|---:|---:|---:|---:|
| `330212` | exact factorized-K3 chain | `2.9467e-8` | `3.7793e-8` | `0.7797` | 3,249 | `9.54 s` |
| **`330327`** | **+ batched Strassen L=3** | **`2.031e-8`** | `3.7793e-8` | `0.53745` | 12,357 | `61.93 s` |
| `330290` | + batched Strassen L=4 | `1.836e-8` | `3.7793e-8` | `0.48578` | 15,393 | `114.68 s` |

Rank ≈ **#27**. Strassen is **exact** — raw is unchanged to `+0.0036%`; the whole
gain is the multiplier.

**Nominee A = `330327`.** `330290` scores `1.106×` better but its worst row is
`114.68 s` against a `120 s` cap, and we have **measured a `1.60×` spread in
graded wall on identical bytes**, which puts it at `0.65×` margin under
contention. Nominee B = `330212` (the safest thing we own).

---

## 3. What Slice 3 established (all measured this week)

### 3.1 The floor is a property of the METHOD, not the class — confirmed

Our chain was ARC's `kind=SIMPLE` factorized K3, in which the 4th cumulant enters
as **one global scalar**. Switching to ARC's `kind=AUGMENT`, which carries κ₄ as
an `(n,n)` object in flight:

| arm | mean raw MSE (24 nets) | 95% CI |
|---|---:|---|
| K3 `SIMPLE` | `4.004685e-08` | `[3.7943e-08, 4.2151e-08]` |
| K3 **`AUGMENT`** | **`1.912323e-08`** | **`[1.8229e-08, 2.0017e-08]`** |

`2.094×`, **24/24 networks**, paired difference `+2.0924e-08` with CI
`[+1.9441e-08, +2.2406e-08]`. The control reproduced a stored 2026-08-26 baseline
**bit-exactly** on all 24 nets, three times, across two machine reboots.
Arithmetic and geometric ratios agree to `0.2%`, so the gain is broad, not
outlier-driven.

**This lands inside the leaders' `1.87–2.13e-8` cluster.** Two caveats: these 24
networks are a design-tier set, and for an identical file that path reads **`+6.0%`
hot** against the graded 100 — so the expected *graded* raw is **~`1.80e-8`**.

**An independent fingerprint.** Our all-layers MSE is `6.906e-9` against a
final-layer `1.912e-8`; the leaders publish `7–11e-9` and `1.9–2.8e-8`. **We now
match their two-number signature simultaneously**, which is a much harder test
than the final-layer number alone.

### 3.2 The gain is *entirely* κ₄'s `(n,n)` structure

`kind=AUGMENT` is a bundle: it also raises the diagram budget from **40 to 142**
terms. Two ablations separate them:

| arm | diagram budget | κ₄ carried as | raw MSE | vs SIMPLE |
|---|---|---|---:|---:|
| `SIMPLE` | base (40) | scalar | `4.0047e-08` | `1.000×` |
| **A5** | **raised (142)** | **scalar** | `4.3448e-08` | **`1.085×` WORSE** |
| `AUGMENT` | raised (142) | **`(n,n)` matrix** | `1.9123e-08` | `0.478×` |
| A3 | raised (142) | **zero** | `3.6895e-07` | `9.21×` |

**The raised budget on its own is worth nothing — it is slightly negative.** All of
the `2.094×` is κ₄'s matrix structure, and the two are complementary rather than
additive. A3 also shows something worth stating in the write-up: **a
self-inconsistent cumulant state is far worse than a lower-order consistent one.**

### 3.3 Cost engineering is exhausted, and both remaining levers deflated

| lever | previously assumed | **measured at 1024×16** |
|---|---|---|
| dead/cold-column pruning | `~0.80×` | **`0.961×`** free; `0.845×` needs a compensation recovering **`139×` our entire MSE** |
| hybrid Strassen (big leaves, deep savings) | target: L=4's FLOPs at `≤75 s` | **the point does not exist** — Strassen's saving comes *from* depth |

Strassen depth against projected wall (the pricing instrument reproduces the two
graded anchors to `+0.003%`, and its wall model predicts the held-out L=4 row to
`+3.1%`):

| depth | leaf | `B` | projected worst wall | margin under the measured `1.60×` contention |
|---:|---:|---:|---:|---:|
| 2 | 256 | `0.60420` | `33.0 s` | **`2.28×`** |
| 3 (nominee A) | 128 | `0.53743` | `61.9 s` | `1.21×` |
| 4 | 64 | `0.48577` | `118.2 s` | `0.63×` |

So the realistic cost floor is **≈`0.52 B`** (depth 3 × pruning), against the
`≤0.35 B` the old plan assumed. **That target is unreachable by any lever we have
priced.**

### 3.4 Three claims from the 2026-09-08 response are falsified

- **"D3: power cumulants, `1.1–1.5×`"** — they are **already on**. The flag is
  upstream's `use_pK`, default `True`, and both arms already carry it. No lever.
- **"D2 brackets every κ₄-in-flight scheme, including SSC"** — **it does not.**
  The oracle injects κ₄ into an `r=1` state, which can hold only κ₄'s *trace
  matrix*. Measured, that subspace carries **~0.2% of κ₄'s Frobenius mass** at
  `n=1024`. **So the instrument built to gate the traceless-carrier lane cannot
  gate it, in either direction.**
- **The two cost levers** — see §3.3.

### 3.5 The port is feasible; the blocker is gone

The AUGMENT port's obstacle was a dense `n³` array — **exactly `4.000 GiB` at
width 1024 in f32, i.e. exactly the contest's `4 GiB` single-array cap**. It is
now proven unnecessary, three independent ways: the relevant Laplacian term
vanishes identically; a **10× perturbation of that array, with its axis-0 sum
preserved, moves the chain's output by `1.6e-16`** (while a sum-breaking one moves
it by `2.0e-1`); and the generator's own combinatorics give zero valid cap-graphs.
**The fused path's largest array is `4 MB`.** Coefficient tables are generated (79
terms vs 25); what remains is transcription, pricing and certification.

---

## 4. The strategic problem

**AUGMENT gets us to the leaders' floor, not below it.**

    expected graded raw after the port          ~1.80e-8
    multiplier, if the port costs +10%          ~0.59 B   ->  adjusted ~1.06e-8   (~#15)
    multiplier, if the port costs +60%          ~0.86 B   ->  adjusted ~1.55e-8   (~#22)
    board #1                                     3.80e-9

**The port's FLOP cost is the single largest unknown and is NOT yet priced.** A
sanity estimate for the fused degree-4 route came to `~0.31 B` of additional
arithmetic, which if real would put the port near the pessimistic end. It is an
order-of-magnitude check with stated assumptions, **not a price**.

**To reach #1 at our realistic cost floor (`~0.52 B`) we need raw `≤7.3e-9`** — a
further **~2.5×** below where AUGMENT lands. Nothing we have measured offers it,
and the instrument that was supposed to gate the most promising candidate lane
(§3.4) provably cannot.

### The candidate sub-floor levers, and their status

1. **A traceless-κ₄ carrier** (Puffi-style low-rank response modes, or per-neuron
   diagonal κ₄ in flight). Holds the ~99.8% of κ₄ that an `r=1` state cannot.
   **No working go/no-go instrument.** Building one means designing a *different*
   oracle that can represent the traceless part — itself a real piece of work.
2. **A learned in-flight corrector on chain features** (per-layer targets from a
   sample stream; c.f. one leader's published 10-channel/90-coefficient per-layer
   adjustment). **No prior on its gain at this shape.** Needs a training run and
   a held-out protocol.
3. **Unfactored AUGMENT.** The factored path we measured *drops* same-order
   diagrams, so `1.91e-8` is a **lower bound** on what AUGMENT-class information is
   worth. Unfactored is `O(n⁴)` and unreachable at this width — but the gap
   between factored and unfactored has never been measured at any width.
4. **Something else entirely.** We have not found it.

---

## 5. What we would like judged

1. **Is the AUGMENT port still the right first call given it lands at the floor?**
   It is ~2× and takes us to roughly #15. The alternative is to spend the same
   weeks on a sub-floor lever with no instrument and no prior.
2. **What should gate the traceless-carrier lane, now that the planned oracle
   provably cannot?** Is there a cheaper discriminating measurement than building
   a representation-faithful oracle?
3. **Is the learned corrector worth the remaining runway** on the evidence that at
   least one leader appears to use one — or is that inference too weak to spend
   weeks on?
4. **Given #1 looks out of reach, how should effort split between board position
   and the `$20k` contribution write-up?** We hold several clean negative results
   (the memory-depth ceiling; a self-inconsistent cumulant state being 9× worse
   than a lower-order consistent one; the diagram budget being worthless without
   the κ₄ structure it exists to consume) that may be worth more than two board
   places.
5. **Nominee policy for the private re-run.** Is depth 2's `2.28×` wall margin
   worth `12.4%` of score, against nominee A's `1.21×`, on a grading server whose
   contention we cannot observe and where a single timeout is catastrophic?

---

## 6. Two things about this evidence a reader should weigh

**The measurement discipline held, repeatedly.** Every gate was pre-registered
before data; an independent falsifier reviewed each strategic result and forced
real corrections (it caught that `kind=AUGMENT` is a bundle, which is what
produced §3.2); a refusal gate on the oracle's own sampling noise **fired on its
first use** and prevented a lane being closed on noise rather than on the method.

**And the builder's priors were repeatedly wrong.** Two pre-registered predictions
missed in the same direction — both times by assuming the extra diagram terms
carried some of the gain, when they carry none. Two cost estimates were badly
wrong. One derivation would have silently dropped two real terms had it not been
checked numerically. **Every error was caught by measurement, by the falsifier, or
by a graded anchor — none by the builder's reasoning.** Weight the measured
numbers above heavily and the builder's projections (notably the `~0.31 B` port
cost estimate) lightly.
