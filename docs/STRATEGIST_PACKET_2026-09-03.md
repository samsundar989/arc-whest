# STRATEGIST PACKET — 2026-09-03

**Self-contained. Assumes no repository access and no shared chat history.**
Written because repository evidence on the question *"can the third-order
correction be made to work?"* is now exhausted, and what remains is an allocation
decision under a deadline.

There are **two questions** at the end. Everything before them is the evidence
needed to answer them and nothing else.

---

## 1. The competition, in six lines

Given a random He-initialised bias-free ReLU MLP (`1024` wide × `16` deep),
predict every neuron's expected post-ReLU activation under `x ~ N(0, I)`.
Score = mean over MLPs of `final_layer_MSE × max(0.1, FLOPs / 2**41)`, **lower is
better**. Phase 2 sets `lambda = 0`; the installed caps are a **`0.4 s` per-MLP
residual cliff** (exceeding it FAILS the MLP, which then scores as zeros), a
`120 s` wall and `8 GiB`.

**The prize is decided by a private re-run of UP TO TWO designated submissions
per team on fresh MLPs.** Public board rank is not the target. There is a second,
now co-equal objective: a **$20k contribution prize** whose deliverable is a
write-up.

**Dates.** Designate by **Oct 10**. Close **Oct 17**. Write-up due **Oct 24**.
Today is **Sep 3**, so ~5 weeks, of which ~3 are working weeks before designation.

---

## 2. Where we stand — the two candidates

| | raw MSE (100 graded rows) | adjusted | multiplier |
|---|---:|---:|---|
| shipped `v2.6.1` (second-order, submitted) | `4.05e-06` | `4.05e-07` | `0.1` floor |
| `v2.7` (calibrated, submitted) | `1.374e-06` | `1.374e-07` | `0.1` floor |
| **the K3 canary** (third-order regenerating flow, **NOT submitted**) | **`8.75e-07`** | **`8.75e-08`** | `0.1` floor |

The canary is **`4.63×` better than the shipped file on the organisers' own
data**, `0/100` failures, with a matched control in the same run reproducing its
own historical figure to `0.01%`. **This is our best asset and it has never been
submitted.**

**We are at the multiplier floor with ~30% margin**, so arithmetic is free at the
margin and the score is simply `raw MSE / 10`. The binding constraint is not
FLOPs; it is the **number of array operations per MLP**, because that is what the
`0.4 s` residual cliff is paid in. Measured coefficient: the evaluator's
mean-residual ratio between two of our files was `5.82×` where their local
*operation* ratio was `5.98×` — agreement to 3%.

---

## 3. What was attempted since, and the complete result

The third-order flow rebuilds its cumulant from two "read slices" each layer and
**discards** an object. Recovering that object was the whole improvement program.
The discarded object turns out to be **one tensor**: its diagonal (call it
**`E3`**, an `n`-vector) and its zeroed off-diagonal (**`E21`**, `n×n`).

**Four routes, all closed, each with a measured ceiling** (not merely a poor
result — a bound on what any map of that form could achieve):

| route | held-out | **ceiling** |
|---|---:|---:|
| select a subspace of what the flow already holds (5 rules) | — | `1.45×`–`2.10× **worse** than not correcting` |
| select using the covariance's dominant range | — | containment `= r/n` to 3 decimals, i.e. **chance** |
| predict `E3` pointwise from carried per-neuron scalars (24 equivariant features, 1.44M rows, 128 nets) | `0.100` | **`0.258`** |
| predict `E21` from 29 equivariant matrix terms (same 128 nets, basis frozen before its moments were computed) | `0.144` | **`0.167`** |

Each fit is within `0.03`–`0.16` of its own basis's ceiling, so **the shortfall
is representational, not statistical.** Ceilings were checked against solver
artifacts (three solvers, five truncation settings, agreement to six decimals,
full-rank design matrices). Both permuted-coefficient nulls are clean by enormous
margins.

**And the sensitivity that explains why none of it was close.** Measured on 8
networks at the live shape with `E21` held at **oracle** (exact, rank-192
truncated):

| `E3` quality | end-to-end drift reduction |
|---|---:|
| absent (`R² = 0`) | `2.22×` |
| best free route (`R² = 0.45`) | `3.73×` |
| exact (`R² = 1.00`) | **`18.87×`** |

**A `10×` end-to-end target needs BOTH pieces near-exact. We can predict neither
above `0.15`.** The pre-registered falsifier — *"`R² < 0.90` on both legs held
out"* — has fired. The lane is closed.

**One component does work and is worth knowing about.** Because the two pieces
are the diagonal and off-diagonal of one tensor, reading `E3` off a rank-192
reconstruction of `E21` — **one fitted scalar, no new features, on an object the
`E21` correction has already materialised** — gives `R² = 0.45` with a fitted
scale of `1.019` (essentially unbiased), and is **non-destructive on 8/8 networks
at three ranks**. It is free and correct. It is unusable only because it needs an
`E21` we cannot predict.

---

## 4. The canary, and the one thing blocking it

Certifying the canary for the first time found it was a **research artifact**: it
scored `5 of 30` on our invariant suite — NaN output on degenerate networks, 151
of 1024 **scored-row** entries negative, and a FLOP count that depended on
evaluation order. All fixed; it is now **`29/30`**, at the multiplier floor, with
`0/100` failures, and the real `8 GiB` enforcement passing `13/13` with both
controls behaving.

**The one remaining failure is a float32-precision property**: a permutation-
equivariance invariant whose threshold (`d_k/scale < 1e-6`) our f32 flow misses at
`1.1e-6`, while the *same invariant's* implied-MSE clause passes by five orders of
magnitude. Verified as precision and not a bug: the same flow in f64 gives
`2.4e-15`, a ratio of `~1e9`, exactly the eps ratio. **Curing it requires float64
in the covariance backbone, which doubles the billed rate, pushes the multiplier
off the floor, and costs `1.53×` on score — throwing away the entire `4.63×`
advantage.**

Our own pre-registered gate says any red item means the canary is not presented
as gate-green. **Three independent adversarial reviews all concluded the bytes
are operationally safe at the live shape** and that nothing found can change what
the grader computes at `1024 × 16`.

**Why we want to submit it at all:** the residual bar for *nomination* is `≥ 4×`
worst-row headroom, and **only the grader's per-row telemetry can measure it.**
Our own runner swings `31%` with host load for a `2.3%` operation change. A
public, undesignated canary returns that telemetry for all 100 rows. The current
local reading is `2.23×` (conservative reading `1.54×`), so **we do not know
whether our best file clears the nomination bar.**

---

## 5. What is left that could still move the score

1. **Operation-count reduction.** The remaining cuts are *reassociations* (not
   IEEE identities), so each needs a measured drift check. Target is ~`1,470`
   operations/MLP against today's `2,673`. **This is the only remaining lever on
   the nomination, and it is gated on a graded number we do not have.**
2. **The write-up.** Now carries a complete four-ceiling negative result with a
   measured sensitivity curve, plus several methodology findings (a score
   identity that makes arithmetic free at the floor; an amplification law; the
   discovery that our binding cost was operations, not FLOPs).
3. **Nothing else.** Every other lane is closed on measurement.

---

## 6. THE TWO QUESTIONS

**Q1 — Allocation.** The kill line's condition is met three weeks early. The
standing instruction is *freeze, nominate the canaried uncorrected flow plus
`v2.7`, finish the write-up*. Given that (a) the nomination path is blocked on a
graded number only a submission can produce, (b) operation-count reduction is
the only remaining score lever and is itself gated on that number, and (c) the
write-up is a co-equal `$20k` objective with a now-complete result — **how should
the remaining ~3 working weeks before Oct 10 be split, and what is the trigger
for abandoning the op-count work entirely?**

**Q2 — The canary at `29/30`.** Our canary bar (which you set) reads
"`certify_candidate` green". We are at `29/30` with one red item that is a
float32 property the graded distribution cannot observe and that no competitive
Phase-2 candidate can satisfy. **Does that satisfy the bar, or does the bar want
amending?** Note the asymmetry you identified originally still holds: an
undesignated public submission has no prize consequence, an overrun costs one
daily submission slot, and the private re-run grades only designated entries.

---

## 7. Two premise corrections you should have, regardless of the answers

Both rulings you gave on 2026-09-02 rested partly on inputs we supplied, and one
input was wrong.

1. **"`E3` first — nothing targets it yet"** rested on `E3` being a separate cheap
   leg. It is not: it is the **diagonal of the same tensor** whose zeroed
   off-diagonal is `E21`. The pointwise route we took because of that framing died
   at a ceiling of `0.26`; the diagonal route we did not think to try reaches
   `0.45` for one scalar. The ordering was wrong, though the outcome is the same.

2. **We told ourselves the joint oracle ceiling was `~3×`.** It is not. Correcting
   *either* piece alone saturates near `2.4×`, but correcting **both** at the
   affordable rank gives **`11.1×`** (`9.86`–`12.63` on 8 of 8 networks), and at
   rank 512 gives `334×`. We had this in our own log a day before we contradicted
   it, and an independent review caught the misread. **The gate was always
   reachable in principle; what is not reachable is predicting the two pieces.**
   Any reasoning you did that used our `~3×` figure should be re-run against
   `11.1×`.

---

## 8. Things that are settled and should not be re-litigated

- `estimator.py` is untouched at `v2.6.1`. Nothing has been submitted or
  designated this session.
- The carrier lane, AUGMENT, Strassen, symmetry tagging, W-EFF, and all five
  column-weight selectors are closed on measurements recorded at the time.
- Arm M (a mechanistic three-generation reconstruction) was killed on cost: `~5`
  extra matrix products per layer against a margin of `1.35`. Its closed-form
  generator survives as a basis term, where it is one of the four terms that
  actually point at `E21` — and still insufficient.
- `P06` (128 DESIGN seeds reserved for the end-to-end gate) is **unspent**, since
  that gate's falsifier fired upstream of it.
