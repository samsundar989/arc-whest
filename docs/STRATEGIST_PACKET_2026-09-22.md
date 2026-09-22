# Strategist Packet — 2026-09-22

**Self-contained. Assumes no repository access and no shared chat history.**
Everything needed to answer is below; nothing is referenced that is not quoted.

Written because the plan this project has been executing (`docs/PLAN.md` rev 12,
2026-09-21) has had three of its four technical theses falsified by measurement
in one day, and its remaining deliverables were sized against that model. We are
not stuck — we improved 1.574× today — but we are now navigating without a map
we trust.

---

## 1. The competition, exactly

ARC White-Box Estimation Challenge 2026, Phase 2. Given a random He-initialised
ReLU MLP — **1024 wide, 16 deep, no biases** — predict every neuron's expected
post-ReLU activation under x ~ N(0, I), as a `(depth, width)` array. Only the
**final layer** is scored.

    score = mean over MLPs of   final_layer_MSE × max(0.1, FLOPs / 2^41)

Lower is better. FLOPs are counted by `flopscope`, a metering NumPy wrapper: every
operation is billed, and the count is deterministic and exactly reproducible
offline. λ = 0 this phase, so cost is purely the FLOP multiplier.

**Hard limits, per MLP.** Wall 120 s. A separate "residual" wall-time budget of
0.4 s, which is the time spent *outside* metered kernels — Python dispatch,
essentially proportional to the number of flopscope operation calls. Memory
8,192 MB. **Breaching any of them fails that MLP, which then scores as zeros —
about 0.9 against a good score of 6e-9.** There is no partial credit.

Submissions are graded on 100 nets (50 public, 50 sealed). The prize is decided
by a **private re-run of up to two designated submissions on fresh MLPs**, so
robustness matters as much as score. Closes Oct 17; 25 days left.

## 2. Where we stand — five graded rows, all ours

| # | candidate | score | raw MSE | cost (F/2^41) | ops | wall | residual |
|---|---|---:|---:|---:|---:|---:|---:|
| 330950 | F21 | 1.5957e-08 | 3.779e-08 | 0.42221 | 14,091 | 93.16 s | 0.1562 s |
| 331317 | v31 | 8.7520e-09 | 2.9074e-08 | 0.30103 | 13,053 | 68.13 s | 0.1402 s |
| 331816 | C10 | 6.4914e-09 | 2.9686e-08 | 0.21867 | 23,421 | 87.56 s | 0.2363 s |
| 331863 | v50 | 6.1919e-09 | 2.8761e-08 | 0.21529 | 25,728 | 88.75 s | 0.2576 s |
| **331880** | **v53** | **5.5603e-09** | **2.4340e-08** | **0.22848** | **25,140** | **96.49 s** | **0.2685 s** |

All five: 100/100 nets, zero failures.

**The public board.** #1 J2W 2.10e-9 (cost 0.131). #2 marius_binner 2.40e-9
(0.146). #3 a_s6 2.50e-9 (0.126). #10 mliston 3.50e-9. Ranks 13–41 are trimmed
derivatives of a public MIT-licensed reference implementation at 4.2–5.3e-9;
ranks 42–57 are sixteen byte-identical clones of it at 5.40e-9. Our 5.5603e-9
sits just below that clone cluster, ~rank 58 of 184 teams.

**The gap, stated precisely:**

| | score | cost | implied raw MSE |
|---|---:|---:|---:|
| J2W #1 | 2.10e-9 | 0.131 | 1.60e-8 |
| a_s6 #3 | 2.50e-9 | 0.126 | 1.98e-8 |
| **us** | **5.56e-9** | **0.228** | **2.43e-8** |

**The leaders are ~1.5× better on raw MSE while spending 57% of our FLOPs.**
We need 2.65× on score to reach #1, 1.59× for top-10, 1.24× for the #13–41 band.

## 3. What we run

Deterministic cumulant propagation through the layers. Mean and full covariance
carried exactly; the **third cumulant** carried in factored form as a set of
per-layer "sources" (rank-1-ish legs, one born per layer) that are transported
forward and contracted into the next layer's mean; the **fourth cumulant**
regenerated each layer rather than carried (exact diagonal plus λ·C_off with a
fitted per-layer λ). Products use Strassen at depth 5. float32. Single file,
stdlib + flopscope only.

Sources older than `_SHARED_AGE` layers are **compressed into a shared rank-r
basis** `Q` (n × r), with the old legs reconstructed as `Q @ [FA | FM]` when
needed. Currently age 5, rank 256.

This is, we believe, the same family the leaders use: every top-12 entry has
chain-like telemetry (deterministic, 77–117 s wall, cost 0.13–0.19), and the
error profile matches — our all-layers/final ratio is 0.41, the leader's 0.41–0.45,
i.e. depth-accumulated truncation error rather than a per-layer defect.

## 4. What we measured today, and what it killed

Every number below is from the real flopscope meter and 100-net panels against
organizer labels whose Monte-Carlo noise floor is ~9e-11 — three orders below
anything quoted.

### 4a. Fitted coefficients are a small lever (the plan's "main accuracy bet")

Six per-layer profiles parametrised as 4-knot monotone splines, fitted end-to-end
by CMA-ES on 100 held-out nets disjoint from the scoring panel by weight hash.

**Result: +2.7%**, not the 1.15–1.75× the plan projected. It generalises cleanly
(DEV +2.687%, HOLDOUT +2.871%, 0.18 pp apart), so it is real — just small. The
likely reason: the λ table being refitted was *already* fitted.

**And it does not go further.** Refitting on the current structure, with a
fourth profile added (a gain on the old-tier readout — the component §4d shows
carries the accuracy), returns **0.00%** over 302 evaluations. The search was
exploring properly: 9 evaluations within +1% of the incumbent, closest
non-baseline +0.196%, none better. The coefficients are at a local optimum and
16 parameters do no better than 12. **On this chain, fitting is a 2.7% lever and
that is all it is.**

### 4b. Billing polish is a small lever, and two of its five items are impossible

**Result: −2.26%**, not the −6–10% projected.

- Routing more products through Strassen: −2.6% FLOPs but **+30% op calls**,
  which spends the residual budget. 63% of our op budget is already Strassen
  glue (Strassen off: 8,724 ops and cost 0.329; on: 23,421 ops and cost 0.219).
- Winograd's 15-add form instead of 18-add: −0.69% FLOPs, −8.2% ops, but **+29%
  peak memory**, which we could not afford.
- **Pooled `out=` buffers: impossible.** `fnp.add(a,b,out=buf)` bills exactly
  what `a+b` bills, and slice assignment raises *"flopscope arrays are
  immutable"*.
- **Symmetry tagging: worthless.** `as_symmetric` discounts *elementwise* ops 2×
  but bills matmuls identically (33,488,896 FLOPs tagged or untagged), and costs
  ~7n² to verify. Matmuls are essentially the whole bill.

### 4c. Richer term sets lose on score — including K4

We ran the reference implementation's own algorithms against its own
Monte-Carlo truth at (n=128, L=8) and (n=256, L=8), with 8 trials and 20M
samples, then priced each with its own FLOP counter. Against factored AUGMENT
K3 (roughly what we run):

| arm | n=128 cost× | MSE× | **score×** | n=256 cost× | MSE× | **score×** |
|---|---:|---:|---:|---:|---:|---:|
| **factored SIMPLE K3** | 0.125 | 2.744 | **0.343** | 0.124 | 2.509 | **0.312** |
| unfactored AUGMENT K3 | 1.361 | 0.843 | 1.147 | 2.359 | 0.924 | 2.180 |
| **factored K4** | **21.19** | 0.294 | **6.222** | **42.15** | 0.154 | **6.483** |

**K4 costs 21–42× the FLOPs for 3.4–6.5× the accuracy** — a 6.2–6.5× *worse*
score — and the cost ratio doubles with width, so it degrades toward n=1024.
Closed.

**The unexplored result is the first row.** A *cheaper* term set (SIMPLE) beats
the richer one (AUGMENT) on score by ~3× at both widths: the extra Edgeworth
diagrams cost 8× and buy only 2.5–2.7×. We have not acted on this.

*Caveats: a different FLOP counter with different conventions, an unoptimised
reference implementation, shapes far from (1024, 16), and our chain is not
exactly any of these arms. Ratios only.*

### 4d. The plan's cost thesis is backwards — and this was the day's big win

The plan asserts *"Cost (0.25 → 0.13 B) is the old tier"* and proposed three ways
to cheapen it. Measured, on 100 scoring nets:

| old-tier setting | cost | raw MSE | **score** |
|---|---:|---:|---:|
| rank 64 | −23.21% | +545.6% | +395.8% |
| rank 128 | −15.64% | +170.4% | +128.1% |
| rank 192 | −6.81% | +43.9% | +34.1% |
| **age 4, rank 256** (was shipped) | — | — | — |
| rank 320 | +9.34% | −14.89% | −6.95% |
| **age 5** (now shipped) | +6.13% | −15.50% | **−10.32%** |
| age 6 | +11.59% | −20.17% | −10.91% |
| age 7 | +16.40% | −21.07% | −8.13% |
| age 8 | +20.54% | −21.42% | −5.28% |

**The old tier is under-funded, not overpriced.** Note the contrast with §4a:
one integer here was worth −10.32%, while the whole fitted-coefficient apparatus
is worth −2.7% and is exhausted. On this chain, structure dominates fitting. Even the mildest truncation —
6.8% of cost — destroys 44% of raw MSE. Conversely, compressing two layers
*later* costs 6.1% and returns 15.5%, a −10.3% score improvement **from changing
one integer**. That single constant outperformed the fitted-coefficient work
(−2.7%) and the billing work (−2.26%) combined, each of which took a day.

Accuracy saturates near −21% raw beyond age 6 while cost keeps climbing, so
there is a genuine optimum, and it is at age 5–6.

## 5. The strategic problem

Our improvement path is **spending more**: cost went 0.21867 → 0.22848 while the
score improved. The leaders sit at **0.126–0.146 with better raw MSE than ours**.
We are moving away from their operating point while chasing their accuracy.

And we are near hard walls:

| constraint | limit | v53 | headroom |
|---|---|---|---|
| memory | 8,192 MB | 6,457 MiB VmPeak | 79% used; **age 6 hits 87%** |
| wall | 120 s | **96.49 s** | already **over** the 95 s bound we set for a designable entry |
| residual | 0.4 s | 0.2685 s | 1.49× |

So "keep raising the age" has perhaps one step left before memory and wall stop
it, and it would leave us around 5.0–5.2e-9 — still outside the top 40.

## 6. The questions

1. **Is our cost 1.7× the leaders' because of a structural choice we have not
   identified?** They achieve 1.60e-8 raw at 0.131; we achieve 2.43e-8 at 0.228.
   What class of design gets both? We carry a full n×n covariance and factored
   third-cumulant sources with an n×r compressed tail; is there a representation
   that is cheaper *and* more accurate, rather than trading between them?

2. **Is "spend more on the exact tier" a local optimum?** It has produced our
   largest single gain, but it is bounded by memory (87% at age 6) and wall (96.5 s
   already). Is there a reason to expect it continues, or is this the end of that
   direction?

3. **Should we move to a cheaper term set?** Our own measurement says factored
   SIMPLE beats factored AUGMENT on *score* by ~3× at two widths, which
   contradicts our chain's design. Is that transferable to (1024, 16) and to an
   optimised implementation, and what would migration cost? Note we cannot
   assume "cheaper term set plus heavier fitting" rescues the accuracy: §4a
   shows fitting is exhausted at 2.7% on the chain we have.

4. **Only the final layer is scored.** We prune hard at the last layer but carry
   the full apparatus everywhere else. Is there a principled way to propagate
   only what reaches the final mean — a linear-response or adjoint formulation —
   that we are missing?

5. **Given 25 days and a private re-run on two designated entries**, is the right
   play to push score, or to bank a robust entry and spend the remaining time on
   the write-up (a co-equal $20k prize)?

## 7. What we are not asking

Do not re-litigate: sampling / QMC / control-variate sidecars, Gaussian-mixture
closures, generic post-hoc correctors, exact augmented κ4, CP/Tucker leg
compression, f64, or the K4 and unfactored-AUGMENT lanes in §4c. All are measured
dead here, most with numbers quoted above.

## 8. Confidence and caveats

- The five graded rows are ground truth. Our offline meter predicts the server's
  FLOP count to **491 parts in 4.7e11** and its score to within 1%.
- §4d's table is one 100-net panel per setting on a GPU evaluator that agrees
  with our reference implementation to 3.2e-05 on a panel mean but only ~1e-3
  per net. The *ordering* is solid; the third digit is not.
- §4c is the weakest evidence here: a different counter, a different
  implementation, and shapes 4–8× smaller than ours.
- We found four instrument defects today, all caught by gates rather than by
  review, including a wall-time model that was 23% wrong for two submissions.
  Treat any single unreplicated number below with that in mind.
