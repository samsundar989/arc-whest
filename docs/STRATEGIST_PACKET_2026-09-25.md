# Strategist Packet — 2026-09-25: the corrector landed, every other lever is closed, two decisions are blocked

**Self-contained. Assumes no repository access and no shared chat history.**
Answers `docs/STRATEGIST_RESPONSE_2026-09-24.md` and `docs/HANDOFF_2026-09-24.md`.
Every number is measured unless marked **[I]** (inference). Paths are given so a
Skeptic can check them.

**The short version.** §3 (the in-loop learned corrector) **landed and is the
largest accuracy win this project has produced**: a new best graded score of
5.0443e-09, then a further −1.35% raw on the server from a wider training set.
§4 (adaptive age) is **killed at −0.72%** against its 2% gate. Both remaining
items on the list — the recursive feature and more training data — have been
executed: one paid 1.57 pp, the other nothing. **There is no untried accuracy
lever left in the spec.** What now blocks progress is not measurement: it is that
PLAN §5's 3% submission gate has become binding against our own best score, and
that this collides with the designation policy. Both need Samuel.

---

## 1. Where we stand

| sub | candidate | graded score | raw MSE | F/B | ops | worst wall (median) | failed |
|---|---|---:|---:|---:|---:|---|---:|
| 332088 | tr64 | 5.4341e-09 | 2.4610e-08 | 0.22081 | 22,978 | 80.30 (76.90) q | 0 |
| **332129** | **v57** (corrector, 100-net fit) | **5.0443e-09** | 2.3326e-08 | 0.21626 | 24,186 | 92.81 (90.43) q | 0 |
| **332223** | **v60_ws400** (400-net corrector, leaf 64) | **5.3387e-09** | **2.3011e-08** | 0.23200 | 20,345 | **57.59 (55.73)** q | 0 |

Best graded **5.0443e-09** (was 5.3200e-09 at the start of this slice, a 1.0546×
gain). Every row above is 100/100 clean. All three walls were measured on quiet
servers (worst/median 1.026–1.044).

## 2. What the slice delivered

### 2.1 The corrector works, and the result is the FITTING PROCEDURE

A per-layer `33 → 32 → 32 → 1` tanh MLP predicts `label − chain μ_l` from the
layer's own state and adds it to `μ_out` *before* it becomes the next layer's
input. Same architecture, same features, same cost, same panel:

| fit | DEV Δraw | wins |
|---|---:|---:|
| single pass, all 16 correctors on the uncorrected chain | **+0.58%** (worse) | 46/100 |
| sequential teacher forcing (l fitted with 0..l−1 active, frozen) | **−5.60%** | 88/100 |
| sequential, 400 training nets instead of 100 | **−7.17%** | **99/100** |

**6.2 percentage points separate the first two rows, and the difference is
entirely exposure bias.** The teacher-forced ceiling I measured beforehand
(6.16–7.32%) was right about the number and silent about the engineering: the
naive route to it returned exactly zero. Anyone reading a ceiling as "the lever is
worth 6%" would have been right and still shipped nothing.

Generalisation held at every step: HOLDOUT-600 −4.76% (0.84 pp from DEV, kill was
1 pp), fresh 10 nets never in any fit −3.79% (1.81 pp, kill was 2 pp), and the
graded raw MSE moved −5.22% against DEV's −5.60%.

### 2.2 Cost is a non-issue; the wall is the whole constraint

The corrector costs **+0.0326% FLOPs and +541 ops, with 0 scalar
materializations** — metered on the f64 twin, and *identical* between a
zero-weight build and the fitted one, so its cost is value-independent. Essentially
the entire raw gain flows to the score.

The wall law refit continues to hold. Leaf-64 kernel rate is now confirmed on a
third candidate at **94.6 ps/FLOP** against the 94 the model uses, and a
mixed-leaf candidate (tr64) read **141.0**, between leaf-32's 176 and leaf-64's
93 — the first interior anchor the law has.

### 2.3 Four levers closed by measurement

| lever | outcome |
|---|---|
| **residual anatomy** (§3 by-product) | **79% of the final-layer error is propagated drift**, not local. One-step/accumulated falls 0.570 (layer 4) → 0.206 (layer 16). A ReLU layer **contracts** incoming mean error (A_l = 0.60–0.88); all MSE growth is newly created error. The decomposition reconstructs M₁₆ to −0.00%. |
| **a formula fix** | Not implicated. The one-step residual concentrates on α (top-2 of 10 bins = 0.664 at layer 16) but at **α ≈ 0, the ReLU kink**, with per-bin bias 0.1–2.4% of RMS and the same sign either side of zero. Symmetric spread, not offset. |
| **per-lineage adaptive age** (§4) | **KILLED at −0.72%**, gate 2%. τ = 0.28 produces ONE fold schedule across every net — it is fixed age 6 wearing a tolerance — and its +5.45%/−5.40% reproduces D5's age-6 numbers by a different route. Lineage energies are too tightly distributed (p5–p95 spans 1.9× within an age against 1.7× *between* ages) for a tolerance to allocate age per net. |
| **the recursive feature** | **Worth nothing.** `(W_lᵀ c_{l−1})_i` — the one feature in the spec that could not be measured before a corrector existed. Per-layer R² identical to within noise (layer 16 0.01244 vs 0.01260); end to end −7.04% vs −7.17%. `mu_pre = Wᵀμ` already carries it; the corrector was never missing the information, only unable to separate it, and separating it does not matter. |

### 2.4 A calibration correction worth carrying

**The organizer label noise floor is 6.8e-10, not 9e-11** — AGENTS.md was wrong by
7.6×. Layer 1 is the one row the chain does not compute (exact closed form
`‖W[:,j]‖/√(2π)` in float64), so its MSE *is* the labels' Monte-Carlo noise:
measured 6.8815e-10 against an analytic `Var(ReLU)/1e9 = 6.817e-10`. AGENTS.md is
corrected. It binds nobody (2.8% of our MSE, 4.2% of the leader's) but every
"MSE with its noise floor" from here on should quote 6.8e-10.

Separately, **DEV→graded is lowered 1.042 → 1.038** (RESULTS.md and `tools/ev.py`
together). Eight observations, mean 1.0403; the last two graded *worse* than
projected (+0.22%, +0.54%), so 1.042 made §5's 3% gate ~0.4% looser than intended.

## 3. The two things that are blocked, and why they are not mine

### Q1. §5's 3% gate is now binding against our own best score

`estimator_k3_v59_corr400.py` — the same corrector as the graded A, at leaf 32 —
would grade **4.9763e-09**, derived from A's own graded raw MSE
(2.301125e-08 × 0.21625525). That is **the best score this project has produced**
and **1.35% better than the best graded**.

It cannot be submitted. §5 requires 3%, and at 1.35% it fails. The structural
problem is that **the gate measures against our own most recent success**, so every
win raises the bar for the next one: −5.60% cleared it comfortably against a stale
baseline; −7.17% does not clear it against the row that −5.60% produced.

The gate was written to stop slots being wasted on marginal noise. 1.35% at
99/100 wins and ±0.81 is not noise. **Does a lower threshold apply when the
baseline is our own fresh best rather than a stale one?** I have not submitted it
and will not without an answer.

### Q2. The designation pair is degenerate, and the fix needs a rule we do not have

Today's pair is **A = v60_ws400 (5.3387e-09, 69.1 s implied noisy, measured) and
B = v56b (5.7076e-09, ~68 s)** — but **A dominates B on both axes**, so the second
slot is buying nothing.

The pair that would be a real hedge is **A = v59 (4.9763e-09, ~112 s noisy) and
B = v60_ws400 (5.3387e-09, 69 s)**: the best score we have, floored by a safe twin.
It needs two things:

1. a graded row for v59 — i.e. the Q1 answer;
2. **an answer to how the two designated entries combine.** The Rules v14 text
   this repository holds says only that "up to two designated submissions are rerun
   on fresh MLPs". If the **better** counts, the hedge is strictly correct and our
   current A ≤ 105 s policy is too conservative. If a failure in **either** taints
   the team result, v59 should not be designated at all and both legs must be safe.

**Why this matters more than 7% of score.** A net over the 120 s cap is FAILED and
scores ~0.9. One failed net in 100 takes the entry from 5.04e-09 to 9.0e-03 —
**1.8 million times worse**. So the question is not "is v59 7% better", it is "how
much probability of total loss is 7% worth", and that depends entirely on the
combination rule. Agents may not contact the organizers; this one is Samuel's.

For scale: the worst server tail we have ever measured is 1.235. v59 needs 1.327
to breach the cliff; v60_ws400 needs 2.256.

## 4. Two things I would flag without being asked

**The leaders' ±6% per-net cost variation has now killed two hypotheses**, and both
failed the same way: per-net λ (its driving ratio varies only 1.8–9.0% CV) and
adaptive age (lineage energies too tightly distributed). The pattern is that *this
chain's per-net statistics are too uniform to key an adaptive decision on*. A third
hypothesis should look for variation that is **not** built from weight energies —
an iterative solve whose iteration count varies with conditioning rather than
scale. I would not spend a day on it without a specific mechanism.

**Every wall reading we hold is from a quiet or moderate server** (ratios
1.026–1.114). Every "noisy" figure in DESIGNATION.md is the ×1.24 projection, not a
reading. Sep 23 went noisy after ~08:00 UTC but Sep 24's 08:30 UTC run came back
quiet at 1.026, so the tail is not a time of day. A genuine noisy-day A/B reading
has to be taken opportunistically, and that is worth a slot when a same-day row
shows a ratio ≥ 1.15.

## 5. What I am not asking about

Anything closed by measurement in RESULTS.md: single-leg, N1, tiered ranks, P3,
per-net λ, the closure tail, joint refit, deeper Strassen, K4/AUGMENT, adaptive
age, the recursive feature, hidden 64 (uniformly worse than 32), or the
formula-fix reading of the anatomy.

## 6. Confidence and caveats

- Accuracy screens are cupy (panel agreement with the numpy instrument 3e-5); the
  paired control is the *same module* with `_CORR_ON = False`, which is
  bit-identical to v55 rather than merely equivalent.
- Every cost figure is the real flopscope f64 read-only grader twin, which has
  matched the server's FLOP count on every row we hold.
- v59's 4.9763e-09 is **derived**, not graded: it is A's graded raw MSE times
  v59's twin-metered multiplier. Both inputs are measured; the product is not.
- The 400-net fit used FIT 100 + the pinned HOLDOUT-300
  (`data/panels/holdout_train_split.json`). HOLDOUT-600 was read once, for the
  generalisation check, and never fitted.
- This slice's predictions: the worker's were right on adaptive age (−0 to −2%,
  measured −0.72%), on its FLOP-spread sub-prediction (<4%, measured 0.00–3.34%),
  and on the anatomy's formula-fix decision; **wrong** in the pessimistic direction
  on the sequential corrector (predicted 1–4%, measured −5.60%) and wrong in the
  optimistic direction on single-feature correlations. The strategist's were
  optimistic on adaptive age and close on the MLP ceiling.
