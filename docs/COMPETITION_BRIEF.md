# The ARC White-Box Estimation Challenge, explained from zero

*Written for Samuel — CS degree, no ML background assumed. Everything a claim, everything sourced (links at the bottom and in RESEARCH_NOTES.md). Last updated: Aug 4, 2026.*

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
x = input                     # a list of 256 numbers
for W in weight_matrices:     # 32 matrices, each 256x256
    x = relu(x @ W)           # matrix multiply, then clamp negatives to 0
```

That's the entire thing. Two operations:

- **Matrix multiply (`x @ W`)**: each of the 256 output slots is a weighted sum of all 256 inputs. The weights are the program's constants.
- **ReLU** ("rectified linear unit"): `max(0, v)` applied to each number. This is the only nonlinearity — the only reason the network isn't just one big matrix multiply.

Vocabulary map: each of the 256 slots at each of the 32 steps is a **neuron** (8,192 total). The value a neuron takes for a given input is its **activation**. The value *before* the ReLU clamp is the **pre-activation**. The network here is an **MLP** (multi-layer perceptron) that is 256 **wide** and 32 **deep**, with **no biases** (no `+ b` term — simplifies the math).

The networks in this challenge are **random**: every weight is drawn independently from a bell curve (Gaussian) with mean 0 and variance `2/256` — the so-called **He initialization**, chosen because it keeps signal sizes roughly stable as data flows through many ReLU layers instead of exploding or dying out.

---

## 4. The exact question you must answer

The input `x` is itself random: 256 independent draws from the **standard normal distribution** N(0,1) (the classic bell curve). Since the input is random, every neuron's activation is a random quantity too. The task:

> For **every one of the 8,192 neurons**, predict its **expected activation** — the average value it would take over infinitely many random inputs. Return a 32×256 grid of numbers.

"Expected value" is just the long-run average: E[f(X)] = the number that `mean(f(x) for x in samples)` converges to as samples → ∞. Notation you'll see: **E[ReLU(zᵢ)]** where zᵢ is neuron i's pre-activation.

Only the **final layer's 256 numbers are scored** — the other 31 rows are diagnostics. (But the whole grid must be returned, all values finite.)

Concretely, a graded run hands your code ~10 networks, one at a time, as an `MLP` object (`mlp.width`, `mlp.depth`, `mlp.weights` — a list of 256×256 matrices), and your `predict(mlp, budget)` must return the grid. The public practice set has 1,000 such networks with ground-truth answers included.

---

## 5. Why this is hard (and why it's interesting)

**The dumb way works, but slowly.** Sample many random inputs, run them through, average: error shrinks like 1/√N (10,000 samples ≈ 1% error; 1,000,000 ≈ 0.1%). The organizers computed their "ground truth" answer keys exactly this way — with **one billion samples per network**.

**The budget makes the dumb way uncompetitive.** Your code runs under a **FLOP accountant** (a library called `flopscope` that wraps every math operation and counts floating-point operations). Phase 1's budget: **2.72×10¹¹ FLOPs per network**. One forward pass costs ~4.2 million FLOPs, so the budget buys ~65,000 samples — a sampling error around 2×10⁻³ per neuron, mean-squared-error ≈ 3×10⁻⁶. The current leaderboard leader is at ~10⁻⁸ — **hundreds of times more accurate than pure brute force at budget**. They are not sampling harder; they are doing math.

**The math way, in one idea.** Random inputs are Gaussians. A weighted sum of Gaussians is Gaussian — so the *first* layer's pre-activations are *exactly* Gaussian, with mean and covariance you can compute directly from the weights, no samples needed. And the average of ReLU(Gaussian) has a **closed-form formula**. So layer 1 can be answered *exactly* by pure algebra. The trouble starts at layer 2: after a ReLU clamp, values aren't Gaussian anymore (they can't go negative, the distribution has a kink). But they're *approximately* Gaussian, so you can *pretend*, carry the (mean, covariance) bookkeeping forward layer by layer, and get remarkably good answers. That's **covariance propagation** — our starting method. The approximation error compounds with **depth** — each layer's small "it's still Gaussian, trust me" lie stacks — which is precisely why ARC made the networks 32 layers deep.

**Why anyone cares:** replace "average activation of neuron 4,201" with "probability the AI does something catastrophic" and you have ARC's research program. Averages of random nets are the training wheels.

---

## 6. Scoring, precisely

Per network m: **score_m = MSE_m × max(0.1, C_m / B)**, averaged over networks. Lower is better.

- **MSE_m**: mean squared error over the final layer's 256 neurons, against the billion-sample ground truth.
- **C_m**: compute you spent = FLOPs counted by flopscope **plus** a wall-clock term: every second of *uninstrumented* runtime (plain Python work the accountant can't see) is billed as 10¹¹ FLOPs (λ = 1e11 FLOPs/sec).
- **B**: the budget (2.72e11 in Phase 1).
- **The 0.1 floor**: use less than 10% of the budget and your multiplier stops improving at 0.1. So a method that's cheap *and* accurate gets a 10× score bonus, but no more. (The challenge overview page shows an older `max(0.5, ·)` example — the starter kit docs, FAQ, and the actual leaderboard arithmetic all confirm **0.1** is what's used.)
- **Failure = zeros**: crash, wrong shape, NaN, or budget overrun on a network ⇒ that network is scored as if you predicted all zeros (MSE ≈ 0.9, multiplier 1.0) — catastrophic but not disqualifying.
- Ground truth itself has noise ≈ **2×10⁻¹⁰** MSE (a billion samples isn't infinity). That's the physical floor of the game; the leader is within ~100× of it.

Worked example: covariance propagation costs ~3×10⁹ FLOPs ≈ 1% of budget → multiplier floor 0.1. Its raw final-layer MSE ≈ 8.4×10⁻⁵ → adjusted score ≈ 8.4×10⁻⁶. Full-budget Monte Carlo: MSE ≈ 2.8×10⁻⁶ × multiplier 1.0 ≈ 2.8×10⁻⁶. The leader: 1.24×10⁻⁸. That's the landscape.

---

## 7. How submissions actually work

- You submit **code** (a single `estimator.py`), via a CLI: `uv run whest submit --estimator estimator.py`. It runs on AIcrowd's servers: **CPU-only, 16 vCPU, 64 GB RAM, no network, no GPU**.
- Inside the sandbox you may import **only** `flopscope`, `whestbench`, and the Python standard library. **No numpy, no scipy, no torch** — all math goes through `flopscope.numpy` (a numpy-compatible, FLOP-counted clone, imported as `fnp`).
- Your class must be named `Estimator`, subclass `BaseEstimator`, and implement `predict(mlp, budget) -> (depth, width) array`. Optional `setup(context)` runs once, off-budget, ≤~5s — you can load precomputed files you shipped (up to 50 MiB / 50 files per submission). **Offline precomputation is legal and unmetered**: you can train models at home for a week and ship the resulting numbers.
- Any randomness must be seeded from `mlp.seed` / `context.seed` (grader-provided). Unseeded or self-seeded randomness risks disqualification; determinism is re-checked when they re-grade.
- **50 submissions per team per UTC day.** Final ranking: your designated Phase 2 submission is re-run privately on **fresh, unseen networks** Sep 20–30.
- Arrays are immutable in the grader (no `a[i] = v`, no `+=`), single arrays capped at 4 GiB, and there's an undocumented per-network wall-clock cap (readable as `admin_config.wall_time_limit_s` in your own run reports).

---

## 8. Timeline, prizes, and the state of play

**Timeline** (all UTC): Warm-up opened May 28 (width 256 × depth 8). Phase 1 ran Jun 18 – Jul 31 (256 × 32). **Phase 2 — the one that counts — is live now: Aug 1 → Sep 19, 23:59 UTC.** Private re-runs Sep 20–30; results Oct 1; NeurIPS workshop Dec 11–12.

**Prizes**: Phase 2 Best Score $50,000 / $20,000 / $10,000, plus **$20,000 for "Best Algorithmic Contribution"** — judged by ARC on scientific merit, explicitly favoring *mechanistic* (math-based) methods over tuned sampling and "opaque LLM-optimized code". Phase 1 had its own $50k which has already been decided. Winners must open-source their code (OSI license). LLM assistance (e.g., me) is explicitly allowed with disclosure.

**Leaderboard as of late July (Phase 1, lower = better, adjusted score):**

| Rank | Team | Adjusted score |
|---|---|---|
| 1 | joe_wanza | 1.24e-8 |
| 2 | abhinav_gorrepati | 2.30e-8 |
| 3 | daddy_yours | 2.32e-8 |
| 4 | mliston | 4.63e-8 |
| 10 | Puffi | 1.06e-7 |
| ~50 | (tail of board) | ~2.2e-7 |

Reference points: random guessing ≈ 7.5e-2, mean propagation ≈ 9.5e-5, covariance propagation ≈ 8.4e-6, pure Monte Carlo at full budget ≈ 2.8e-6. 636 participants / 76 teams / 13,000+ submissions so far. **Phase 2 leaderboards restart** — everyone re-proves themselves on (possibly) new settings, and the final ranking comes from the private re-run.

**Public intel from Phase 1 write-ups** (details + links in RESEARCH_NOTES.md): one team hit 1.55e-7 with a hybrid "classify neurons as dead/on/kinked, sample the ambiguous ones, treat the rest analytically" approach. Another documented that covariance propagation's error is a **stable, systematic ~0.8% scale bias** at depth 32 — fixable with a single multiplicative constant for a ~3× accuracy gain (we've gone further and fit a *per-layer* correction curve — see below). ARC's townhall dropped hints: they expect **cumulant propagation** (tracking third/fourth-order statistics, not just mean/covariance) to eventually win, and are exploring **low-rank refinements** and fancy series-resummation.

**A live controversy you should know about**: participants discovered the FLOP accountant can be partially **bypassed** — computation routed through raw arrays gets billed only via the wall-clock λ rate, which is *cheaper than real CPU throughput*, effectively inflating scores of teams that exploit it (evidence suggests some top entries do). The #4 team publicly proposed fixes; organizers hadn't formally responded as of the last crawl. **We will not exploit this** — it's against the stated rules ("must not exploit" per organizers), it's worthless for the algorithmic prize, and it could be patched (and offenders re-scored) at any time. Strategically it means: don't panic-chase leaderboard numbers that may be partly artifacts; build methods that survive a fair re-grade.

---

## 9. The science menu (plain English), from baseline to frontier

**Mean propagation** — track each neuron's (mean, variance) independently; ignore correlations between neurons. Cheap (11M FLOPs), crude (MSE ~9.5e-4). The "hello world".

**Covariance propagation** — also track how every *pair* of neurons co-varies (a 256×256 table). Linear layers transform the table exactly; ReLU steps use closed-form Gaussian formulas plus one approximation for the pair terms. MSE ~8.4e-5. **This is our v0** — with our fitted per-layer bias correction on top.

**Cumulant propagation (ARC's paper)** — go beyond pairs: track triple and quadruple co-variation ("cumulants" — think higher-order Taylor terms of a distribution's shape). More accurate in theory; error still compounds like (depth)^K, and the paper admits it *loses* to plain sampling by depth 8+ in its current form. Making a depth-robust version = the open research problem = the algorithmic prize.

**Smart sampling (quasi-Monte Carlo)** — sample, but with cleverness: use low-discrepancy point sets instead of raw random draws (error ~1/N instead of 1/√N), pair each sample x with −x ("antithetic" — cancels errors by symmetry), and integrate analytically wherever possible. A public write-up validated this family at ~4e-7.

**Hybrids (the current winning family, most likely)** — exact math for layer 1; smart sampling through the middle layers; then for the *last* layer(s), don't sample the ReLU — compute its conditional average in closed form given the penultimate layer ("Rao-Blackwellization": replace sampled randomness with its exact average wherever you can). Subtract an analytic prediction as a "control variate" so you only sample the *difference* between reality and theory — small differences = small variance = fewer samples needed.

**Learned correctors (our secret weapon candidate)** — the networks come from a *known random recipe*. So we can generate unlimited practice networks at home, compute near-perfect answers offline (GPU Monte Carlo), and *train a small model that predicts the analytic method's residual error* from cheap features of the weights. Ship the trained corrector inside the submission (offline compute is free!). Public evidence says naive versions underperform (the remaining error near layer 32 is mostly ground-truth noise + hard-to-learn structure), so this is a refinement layer, not a magic bullet.

The deep truth connecting all of these: **depth compounds errors**. Every method's story is "how do I stop layer-32 error from being 32 layers of accumulated lies?" ARC's theory says the natural expansion parameter is depth/width = 32/256 — small but not negligible, which is why corrections work and why they're needed.

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
