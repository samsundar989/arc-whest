# The ARC White-Box Estimation Challenge, explained from zero

*Written for Samuel — CS degree, no ML background assumed. Phase-2 facts updated
2026-08-31. Detailed live provenance is in `PHASE2_FRONTIER_AUDIT_2026-08-31.md`.*

---

## 1. The one-paragraph version

You are given a neural network — not to train, not to use, just to *look at*: a big pile of random numbers (its "weights") that define a mathematical function. The question is: **if you fed this function random inputs forever and averaged what each internal unit outputs, what would those averages be?** The obvious way to answer is to actually feed it lots of random inputs and average (this is "Monte Carlo sampling"). The competition pays you to be smarter: your score is your prediction error *times* the fraction of a compute budget you burned, so the winners are the people who can *reason about the network's internals mathematically* instead of brute-forcing it. That skill — predicting what a network will do by analyzing its weights, called **white-box estimation** — is something the organizers (an AI-safety research group) believe is a stepping stone to auditing real AI systems for rare, dangerous behaviors you can't just wait to observe.

---

## 2. Who runs this and why

The **Alignment Research Center (ARC)** — a non-profit founded by Paul Christiano (one of the inventors of the technique used to train ChatGPT-style models, now a leading AI-safety researcher) — runs the challenge together with **AIcrowd**, a Kaggle-like competition platform. Jacob Hilton and the ARC theory team designed the problem.

ARC's motivating question is: *"Are there unusual situations in which an AI system would undermine human control?"* You can't find a one-in-a-billion behavior by testing a billion inputs — too expensive, and the dangerous case might be deliberately rare. So ARC wants **estimation methods that read the mechanism** (the weights) rather than observe behavior. Their published method works well on *wide, shallow* networks but breaks down as networks get *deeper* — so they built a competition around exactly the regime where their own method struggles, and put **$150k+ in prizes** behind it. They also published a paper describing their best-known approach ("cumulant propagation") and are openly hoping contestants beat it.

One more thing worth knowing: they chose **randomly-initialized** networks (not trained ones) on purpose. Random networks are the "cleanest hard case" — no learned structure to exploit, so methods must genuinely handle the math of depth. Trained networks are planned for future iterations.

---

## 3. Neural networks from zero (in programmer terms)

Forget the brain metaphors. For this competition a neural network is a tiny, fixed program:

```
x = input                     # a list of 1,024 numbers
for W in weight_matrices:     # 16 matrices, each 1,024x1,024
    x = relu(x @ W)           # matrix multiply, then clamp negatives to 0
```

That's the entire thing. Two operations:

- **Matrix multiply (`x @ W`)**: each of the 1,024 output slots is a weighted sum of all 1,024 inputs. The weights are the program's constants.
- **ReLU** ("rectified linear unit"): `max(0, v)` applied to each number. This is the only nonlinearity — the only reason the network isn't just one big matrix multiply.

Vocabulary map: each of the 1,024 slots at each of the 16 steps is a **neuron**
(16,384 total). The value a neuron takes for a given input is its **activation**.
The value *before* the ReLU clamp is the **pre-activation**. The Phase-2 MLP is
1,024 **wide** and 16 **deep**, with **no biases**.

The networks are **random**: every weight is drawn independently from a Gaussian
with mean 0 and variance `2/1024`—the **He initialization**, chosen to keep signal
sizes roughly stable through ReLU layers.

---

## 4. The exact question you must answer

The input `x` is itself random: 1,024 independent draws from the **standard normal
distribution** N(0,1). Since the input is random, every neuron's activation is a
random quantity too. The task:

> For **every one of the 16,384 neurons**, predict its **expected activation**—the
> average over infinitely many random inputs. Return a 16×1,024 grid.

"Expected value" is just the long-run average: E[f(X)] = the number that `mean(f(x) for x in samples)` converges to as samples → ∞. Notation you'll see: **E[ReLU(zᵢ)]** where zᵢ is neuron i's pre-activation.

Only the **final layer's 1,024 numbers are scored**—the other 15 rows are
diagnostics, but the complete finite grid is required.

An ordinary grading report carries 100 per-network telemetry rows and exposes 50
public score rows. Those hidden grading rows are **not** the prize-deciding final
rerun: Rules v14 says the designated submissions are rerun on fresh MLPs after
Phase 2. Separately, the downloadable public dataset has a 100-network mini and a
1,000-network full split with ground truth.

---

## 5. Why this is hard (and why it's interesting)

**The dumb way works, but slowly.** Sample many random inputs, run them through, average: error shrinks like 1/√N (10,000 samples ≈ 1% error; 1,000,000 ≈ 0.1%). The organizers computed their "ground truth" answer keys exactly this way — with **one billion samples per network**.

**The budget makes the dumb way uncompetitive.** Your code runs under a **FLOP
accountant** called `flopscope`. Phase 2 allows `2**41 ≈ 2.199×10¹²` FLOPs per
network, calibrated to roughly 65,536 dense samples. At the 2026-08-31 14:57 UTC
snapshot, the leader's displayed raw MSE was `2.25e-8` (its GT-noise floor is not
published), far below ordinary Monte Carlo at that row count. Several leaders use
roughly `5.5e11` counted FLOPs, consistent with about 16,384 dense directions, but
that cost fits multiple samplers and does not disclose their private method.

**The math way, in one idea.** Random inputs are Gaussians. A weighted sum of Gaussians is Gaussian—so the *first* layer's pre-activations are *exactly* Gaussian, and the average of ReLU(Gaussian) has a closed form. At layer 2, ReLU has made the state non-Gaussian. Covariance propagation pretends a Gaussian state remains adequate and carries mean/covariance forward. Its small state error compounds through all 16 Phase-2 layers; winning methods need a better structured measurement or a compositional correction to that state.

**Why anyone cares:** replace "average activation of neuron 4,201" with "probability the AI does something catastrophic" and you have ARC's research program. Averages of random nets are the training wheels.

---

## 6. Scoring, precisely

Per network m: **score_m = MSE_m × max(0.1, C_m / B)**, averaged over networks.
Lower is better.

- **MSE_m**: mean squared error over the final layer's 1,024 neurons, against the billion-sample ground truth.
- **C_m**: Phase-2 counted flopscope FLOPs. Lambda is zero, so residual time is not added to compute—but residual above 0.4 seconds fails that MLP.
- **B**: `2**41 = 2,199,023,255,552` FLOPs.
- **The 0.1 floor**: use less than 10% of the budget and your multiplier stops improving at 0.1. So a method that's cheap *and* accurate gets a 10× score bonus, but no more. (The challenge overview page shows an older `max(0.5, ·)` example — the starter kit docs, FAQ, and the actual leaderboard arithmetic all confirm **0.1** is what's used.)
- **Failure = zeros**: crash, wrong shape, NaN, or budget overrun on a network ⇒ that network is scored as if you predicted all zeros (MSE ≈ 0.9, multiplier 1.0) — catastrophic but not disqualifying.
- Ground truth itself has Monte Carlo noise because a billion samples is not
  infinity. On the Phase-2 organiser mini, the measured average floor used by our
  K3 CONFIRM evaluation is **`7.478e-11` MSE**; the floor varies with the MLP and
  must be reported for each evaluation rather than treated as one universal number.

Timestamped reference on the organiser mini (GT-noise floor `7.478e-11`): the
uncalibrated K2 path is `4.050566e-6` raw / `4.050566e-7` adjusted and v2.7 is
`1.392983e-6` / `1.392983e-7`. The 2026-08-31 14:57 UTC leader was displayed at
`2.25e-8` raw / `5.60e-9` adjusted; its truth-noise floor is not public. That is the
gap the research program in `GAME_PLAN.md` must close.

---

## 7. How submissions actually work

- You submit **code** (a single `estimator.py`). The official environment is CPU-only with no network; the solution process has an 8-GiB memory limit and the host has separate flopscope workers.
- Inside the sandbox you may import **only** `flopscope`, `whestbench`, and the Python standard library. **No numpy, no scipy, no torch** — all math goes through `flopscope.numpy` (a numpy-compatible, FLOP-counted clone, imported as `fnp`).
- Your class must be named `Estimator`, subclass `BaseEstimator`, and implement `predict(mlp, budget) -> (depth, width) array`. Optional `setup(context)` runs once, off-budget, ≤~5s — you can load precomputed files you shipped (up to 50 MiB / 50 files per submission). **Offline precomputation is legal and unmetered**: you can train models at home for a week and ship the resulting numbers.
- Any randomness must be seeded from `mlp.seed` / `context.seed` (grader-provided). Unseeded or self-seeded randomness risks disqualification; determinism is re-checked when they re-grade.
- **10 submissions per team per UTC day.** Up to two nominated submissions are privately re-run on fresh networks after the public round.
- Arrays are immutable and single arrays are capped at 4 GiB. The installed
  official stack/run config uses 0.4-second residual and 120-second wall limits per
  MLP, 5-second setup, and 8-GiB memory. Public graded ledgers prove at least
  `104.766 s/MLP` passes but do not expose the exact server ceiling. An outer
  AIcrowd job timeout may also exist and is a separate constraint.

---

## 8. Timeline, prizes, and the state of play

**Timeline** (all UTC): Phase 2 closes **October 17, 23:59**; registration/team
freeze is October 2; private re-runs run October 17–November 7; winners are
announced November 15; the algorithmic-contribution write-up is due October 24.

**Prizes**: Phase 2 Best Score $50,000 / $20,000 / $10,000, plus **$20,000 for "Best Algorithmic Contribution"** — judged by ARC on scientific merit, explicitly favoring *mechanistic* (math-based) methods over tuned sampling and "opaque LLM-optimized code". Phase 1 had its own $50k which has already been decided. Winners must open-source their code (OSI license). LLM assistance (e.g., me) is explicitly allowed with disclosure.

**Phase-2 leaderboard snapshot, August 31 at 14:57 UTC (lower is better):**

| Rank | Team | Adjusted | Raw MSE |
|---:|---|---:|---:|
| 1 | J2W | 5.60e-9 | 2.25e-8 |
| 2 | Puffi | 5.90e-9 | 2.26e-8 |
| 3 | Activation-Regions | 7.40e-9 | 2.93e-8 |
| 4 | marius | 7.70e-9 | 1.83e-8 |
| 53 | neopok919 (us) | 1.374e-7 | 1.374e-6 |

The account has two zero-failure Phase-2 entries: v2.6.1 at `3.933e-7` adjusted
and v2.7 at `1.374e-7`. The live board is only public feedback; final ranking
comes from private re-runs of up to two designated submissions. The Rules v14
text now states the two-submission rule explicitly.

**Public frontier evidence:** Puffi's Phase-1 system combined Kerdock/Walsh mutually
unbiased bases, antipodes, exact radial integration, an analytic prefix, and a
mid-network control variate. But the live Phase-2 leaders use raw MSE around
`2.3e-8`--`3.5e-8` at multipliers around `0.285`--`0.476`; their code is private,
so no exact Phase-2 architecture may be inferred from a FLOP row alone. Public
experiments show useful moment-matched sampling and low-error-correlation blends,
while our local full-K3 work shows that high-order state is accurate enough if its
cost can be rebuilt. ARC's reference augmented-K2 method separately shows how
selected joint third/fourth state can be propagated without a full dense K3 object.
The current leaders' code is private: their telemetry is an envelope and must not
be represented as evidence for Kerdock, Sobol, or any other specific carrier.

**Clean-compute rule:** meaningful computation must go through flopscope. The old
wall-time pricing loophole is not a research lane and lambda is now zero. We do not
route numerical work through uninstrumented raw arrays, FFI, subprocesses, threads,
runtime code generation, or bundled compiled libraries.

---

## 9. The science menu (plain English), from baseline to frontier

**Mean propagation**—track each neuron independently and ignore cross-neuron state.
It is a cheap sanity baseline, not a competitive method.

**K2 covariance propagation**—also track how every *pair* of 1,024 neurons
co-varies. Linear layers transform this state exactly; ReLU uses Gaussian formulas.
This is our current Phase-2 fallback. It is stable and cheap but about `61x` behind
the 2026-08-31 leader in displayed raw MSE (`24.6x` in adjusted score).

**Augmented K2**—track only selected joint third/fourth-order harmonic/trace pieces
that repair the state passed to later layers. This is open and distinct from merely
adding skew/kurtosis to the final ReLU mean formula.

**Exact third-cumulant propagation**—carry the leading non-Gaussian interaction
through the network. The measured full version reaches roughly frontier raw
accuracy, but its best exact W-EFF spelling costs `1.0732 B` FLOPs and is closed on
price. Keeping only the last few layer ages is also closed: its K2 prefix already
makes every tested window noncompetitive. The open analytic route is
*query-directed K3*: backpropagate which layer-local K3 contractions affect the
final means, without materialising the full recurrent tower.

**Task-specific structured quadrature**—replace random directions with balanced
points, pair directions with antipodes, and integrate the Gaussian radius separately.
Kerdock/MUB bases are one bounded family, but the selection objective should be the
deep-ReLU kernel induced by this task. The next engineering gate is a submission-
legal stdlib/`fnp` constructor; the current NumPy-built probe cannot ship.
Independent rotations, ordinary orthogonal, IID and low-discrepancy controls keep
the certificate honest.

**Compositional hybrids**—run the structured carrier and analytic state in parallel,
measure their discrepancy at registered checkpoints, and transport a compact
correction forward. This follows where the error develops instead of trying to fit
one opaque final residual.

**Compact learned state correction**—only a conditional final branch. If used, it
must respect hidden-neuron permutation symmetry, train on rolled-forward state, use
DESIGN-only data, and improve a preregistered score. Naive final-residual fitting has
already failed.

The deep truth connecting all of these is **state drift through composition**. The
candidate architecture uses a cheap analytic prefix, exact high-order transport only
over the layer ages that earn their cost, and a kernel-selected carrier/checkpoint
to repair the history that was discarded.

---

## 10. What you need to learn (and what you can skip)

You need surprisingly little classical ML — no training, no backpropagation, no datasets in the usual sense. The relevant stack:

**Priority 1 — the mental model (a weekend):**
- 3Blue1Brown, *Neural Networks* series (YouTube) — episodes 1–2 only; gorgeous intuition for layers/weights/activations. Skip the training episodes for now.
- 3Blue1Brown, *Essence of Linear Algebra* — refresh matrix-multiply-as-transformation.
- Probability: expectation, variance, covariance, the normal distribution, Law of Large Numbers, Central Limit Theorem, Monte Carlo error ~1/√N. Any intro source works; you mainly need fluency with E[·] notation to follow discussions.

**Priority 2 — the tools (as we work):**
- NumPy basics (arrays, broadcasting, `@`, einsum). Everything we write is numpy-flavored.
- The starter kit's own docs (in the repo once cloned): `docs/concepts/problem-setup.md`, `scoring-model.md`, and `reference/estimator-contract.md` are excellent and short.

**Priority 3 — the frontier (weeks 2–3, with me translating):**
- ARC's companion paper: *"Estimating the expected output of wide random MLPs more efficiently than sampling"* (arXiv:2605.05179) — read §1–2 for framing; I'll handle the appendix math.
- The two Phase-1 forum write-ups (linked in RESEARCH_NOTES.md) — the most competition-relevant reading that exists.
- Optional deep background: "Formalizing the Presumption of Independence" (ARC's philosophical foundation), and the townhall recording on YouTube.

**Skip entirely (for this competition):** training neural networks, PyTorch, GPUs-for-inference, transformers/LLMs, backpropagation, datasets/feature engineering.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| MLP | The `for W in weights: x = relu(x @ W)` program above |
| Neuron / activation | One slot in the vector / its value after ReLU |
| Pre-activation | The value before ReLU clamps it |
| He initialization | Weights ~ N(0, 2/width) — keeps signals stable with depth |
| E[X], expectation | Long-run average of a random quantity |
| Variance / covariance | Spread of one quantity / co-movement of a pair |
| Monte Carlo (MC) | Estimate an average by sampling; error ~ 1/√N |
| Quasi-MC (QMC/RQMC) | Sampling with evenly-spread deterministic points; error ~ 1/N |
| Gaussian closure | "Pretend it's still Gaussian" layer-to-layer approximation |
| Cumulants | Systematic measures of non-Gaussianity (3rd = skew, 4th = tails) |
| Control variate | Sample only the difference from an analytic guess |
| Rao-Blackwellization | Replace sampled randomness by its exact conditional average |
| FLOP | One floating-point operation; the competition's currency |
| flopscope | The FLOP accountant wrapping all math in the grader |
| whestbench | The harness: MLP objects, scoring, the `whest` CLI |
| MSE | Mean squared error — average of (prediction − truth)² |
| Adjusted score | MSE × max(0.1, compute_used/budget); lower is better |

---

## 12. Primary sources

- Challenge: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026
- Starter kit: https://github.com/AIcrowd/whest-starterkit · FLOP counter: https://github.com/AIcrowd/flopscope
- Dataset: https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026
- ARC announcement: https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/
- ARC's method paper: https://arxiv.org/abs/2605.05179 (+ code: https://github.com/alignment-research-center/mlp_cumulant_propagation)
- Townhall recording: https://youtu.be/pvCgbUJ-nTo
- Key forum threads: see RESEARCH_NOTES.md (leaderboard snapshot, write-ups, the flopscope controversy)
