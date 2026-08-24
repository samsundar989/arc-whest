# Phase 2 live competition and research audit — 2026-08-24

This dated report is the external/repository evidence behind the current strategy.
It is not a second action plan. Use `CURRENT_STATE.md` for status and
`GAME_PLAN.md` for the sole active architecture and execution sequence. Historical
experiment results remain authoritative within their recorded scope; historical
claims about live rules, package versions, leaderboard state, and research ordering
do not.

Evidence labels used below:

- **CONFIRMED** — official rules/package, a public graded report, or a frozen local
  artifact under the repository protocol.
- **INFERRED** — the evidence strongly constrains the explanation, but competitors
  have not published their Phase-2 code.
- **SPECULATIVE** — a mechanism-backed experiment proposal, not a result.

No seed block or vault was touched for this audit. `estimator.py` and all
instruments were left unchanged.

## Executive decision

The repository has a strong scientific protocol and the right *kind* of leading
idea — propagating distribution state rather than relying on terminal Monte Carlo
— but it does not have a competitive Phase-2 submission architecture yet.

The shipped file routes every width above 256 to the old first-order Gaussian
closure. On the Phase-2 shape that means none of the Phase-1 Sobol, pruning,
rotation, or Strassen sampling machinery runs. The current Phase-2 path is only
`k2p1`. Its measured raw MSE is about `4e-6`; the live leaders are at
`3.6e-8`–`3.9e-8`. The raw-accuracy gap is therefore roughly **100x**, and the
adjusted-score gap is roughly **22x**.

The prior plan's projected `~1.30x` coupled-sampling improvement cannot close that
gap, and its float32 cost premise is not in the frozen artifact. It is not the
winning research lane.

The strongest live clue is Puffi's exact FLOP count: `1,099,802,385,000`, only
`0.0264%` above `32,768 * 16 * 2 * 1024^2`, the leading dense cost of exactly
32,768 forward directions. And `32,768 = 16 bases * 1,024 axes * 2 antipodes`.
Combined with Puffi's published Phase-1 Kerdock/MUB architecture, this makes a
**16-basis structured angular carrier with exact radial integration** the
highest-value reproduction experiment. The Phase-2 code is private, so the
mechanism is a strong fingerprint, not a confirmed disclosure.

The second strategic experiment is a **faithful implementation of ARC's augmented
K2 joint-state propagation**. P2-CP3R did not test or upper-bound that algorithm:
it supplied true *marginal* moments to the scalar ReLU mean map, whereas augmented
K2 propagates selected *joint* third/fourth cumulant components that change future
mean/variance state.

The evaluator must be repaired before any such candidate is promoted. The main
environment and lifecycle defaults are Phase 1, the local runner does not meter
weight-method operations exactly like the grader, and the 0.4-second residual
failure cliff is absent from the frozen P2 ladder.

## Live competition snapshot

### Official Phase-2 contract — CONFIRMED

Authoritative sources are the [Phase-2 launch
announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197),
the [WhestBench 0.16.0
release](https://github.com/AIcrowd/whestbench/releases/tag/v0.16.0), and the
[challenge resources
page](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents).

| item | live value |
|---|---:|
| round | Phase 2, live |
| closes | **2026-10-17 23:59 UTC** |
| registration/team freeze | **2026-10-02 23:59 UTC** |
| algorithmic-contribution write-up | **2026-10-24 23:59 UTC** |
| final private rerun | 2026-10-17 through 2026-11-07 |
| winners announced | 2026-11-15 |
| shape | `1024 x 16`, no biases, no separate output layer |
| input | `x ~ N(0, I)` |
| prediction | all post-ReLU means, shape `16 x 1024` |
| dataset | `aicrowd/arc-whestbench-public-2026@v2-phase2` |
| public dataset | mini 100 + full 1,000, disjoint, `1e9` truth samples/MLP |
| private evaluation | 100 MLPs: 50 public feedback + 50 sealed |
| budget | `B = 2**41 = 2,199,023,255,552` FLOPs/MLP |
| score | `final_MSE * max(0.1, F/B)`; Phase 2 has `C=F` |
| residual | not priced, but **hard-fails above 0.4 s/MLP** |
| total wall cap | 120 s/MLP |
| solution memory | 8 GiB; instance 64 GiB |
| CPU | solution 1 physical core / 2 vCPU; flopscope 7 cores / 14 vCPU |
| submissions | 10/team/day; temporary first-week failed-submission allowance |
| final choice | up to two designated entries/team; default is the two highest-ranked |
| required stack | `whestbench>=0.16,<0.17`, `flopscope>=0.12,<0.13` |

Phase 2 prohibits meaningful computation outside flopscope. Bundled compiled
libraries/kernels, FFI, subprocesses, threads, runtime code generation, and
computation concurrent with an in-flight flopscope operation are prohibited.
Residual time is for plumbing, not free estimation. A Phase-1-legal estimator can
be disqualified in Phase 2.

Two official pages are stale and must not drive configuration: Insights displayed
`3.91e-8` as the best adjusted score while the live board displayed `1.80e-8`, and
Resources still mentioned the old minimum package pair in its dataset snippet.
The release and launch announcement win any conflict.

### Live leaderboard — CONFIRMED at 2026-08-24 17:59 page time

[Live Phase-2 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2)

| rank | team/participant | adjusted | raw final MSE | mean FLOPs | mean budget | all-layer MSE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Puffi | `1.80e-8` | `3.603e-8` | `1.10e12` | 50.01% | `1.41e-8` |
| 2 | pranav_devarinti | `1.87e-8` | `3.878e-8` | `1.06e12` | 48.25% | `1.57e-8` |
| 3 | Activation-Regions | `3.08e-8` | `3.780e-8` | `1.79e12` | 81.47% | `1.56e-8` |
| 4 | andrew_epstein | `9.20e-8` | `9.20e-7` | floor-bound | 10% or less | — |
| 10 | trim_qewas | `1.22e-7` | `1.22e-6` | floor-bound | 10% or less | — |
| 19 cluster | several teams | `3.93e-7` | `3.93e-6` | floor-bound | 10% or less | — |

Submission detail:

- [Puffi #328223](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328223)
- [pranav_devarinti #328267](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328267)
- [Activation-Regions #328196](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328196)

The three leading solutions share essentially the same raw-error scale. The first
two use about half a budget; third place spends about 81%. Puffi's exact cost is
only `290,757,224` FLOPs above 32,768 dense forward directions. That is a much
sharper fingerprint than the rounded budget percentage.

The repository's P2-CP1 `k2p2` result (`3.9505e-6` raw, `3.9598e-7` adjusted)
lands almost exactly on the live rank-19 cluster. This cross-check says the local
fresh-network accuracy measurement is directionally right even though the runtime
instrument is stale.

### Our public state — CONFIRMED

The signed-in Submissions view lists six Phase-1 entries and **no Phase-2 entry**.
The best Phase-1 entry is #327743 at `2.63e-7` adjusted / `1.33e-6` raw and the UI
marks it selected. The current Phase-1 board shows neopok919 around rank 104.

The browser selection state contradicts `CURRENT_STATE.md`, which says no Phase-1
entry was selected and therefore the default two applied. Phase 1 is closed, so
this is now a provenance/documentation issue rather than an action to take without
organizer confirmation.

## Repository verdict

### What is strong

- Fresh generator parity is proven against published Phase-2 weights.
- The DECIDE / CONFIRM / VAULT separation is conceptually right.
- Paired common-random-number comparisons and the arithmetic competition-score
  difference are the correct promotion statistic.
- Truth is measured with two independent replicas and noise is reported.
- Immutable evidence rows, seed claims/finalization, and protected submission
  paths are unusually strong.
- The current submission is shape-valid and completed a one-network official
  subprocess smoke with no failure.

### What is not ready

1. `estimator.py` dispatches every width above 256 to `_analytic_only`, so Phase 2
   runs only the old first-order K2 closure. No versioned Phase-2 candidate exists.
2. `pyproject.toml` pins WhestBench 0.14 and flopscope 0.10.
3. `competition_score.py` defaults to Phase-1 lambda `1e11`; official Phase 2 is
   lambda zero.
4. `lifecycle_eval.py` defaults to the Phase-1 budget, 60-second wall cap, no
   residual cap, and 64-GiB memory.
5. The local lifecycle mints raw NumPy weights. The grader supplies flopscope
   arrays, so calls such as `w.astype(...)` can be unmetered locally.
6. The local price for the current path is short by `37,779,456` FLOPs/MLP
   (0.0275%). Small now, but arm-dependent and enough to falsify exact cost parity.
7. The frozen P2 ladder disabled the official 0.4-second residual failure cliff.
   Local readings exceed it for some high-order and near-budget arms. Local timing
   is not transferable, but those deployment claims are provisional.
8. Memory is recorded, not enforced; native-array memory is understated by
   `tracemalloc`, and worker death/restart is not emulated.
9. Preflight can be green on the stale Phase-1 stack and still be wrong for Phase 2.
10. The live checklist reads a stale cached manifest; docs, deadlines, seed counts,
    and strategy ordering contradict one another.
11. P2-CP1 used P01, a reusable DESIGN block on which no promotion gate is decided,
    to make global lane closures. The narrow measurements survive; the binding
    closures do not satisfy the DECIDE-tier rule.
12. The claimed half-cost float32 closure and its 4,497-sample headroom are not in
    the frozen ladder. The recorded `k2p1_f32` cost is `133.266e9` versus
    `137.563e9`, only 3.1% lower. The `1.305x` coupled-sampling projection must be
    treated as unsupported until a registered real float32 arm is measured.

The official one-network subprocess anchor for the current file is:

| metric | value |
|---|---:|
| score/raw MSE | `4.501854e-7` / `4.501854e-6` |
| FLOPs | `137,600,326,656` (`C/B=0.0625734`) |
| residual / wall | `0.05368 s` / `0.35487 s` |
| failures | 0 |

This is a smoke anchor, not a distributional score estimate.

## The critical scientific correction

P2-CP3R is valuable, but its lane closure is over-broad and its binding KILL was
post-hoc.

What it actually measured:

- `O0`: honest marginal Gaussian map from the closure state.
- `O1`: true per-neuron `(mu, sigma)` inserted into that scalar map.
- `O2/O3`: true per-neuron marginal skew/kurtosis added to the scalar Edgeworth
  mean formula.

It convincingly establishes that 94.61% of K2's error is accumulated mean/variance
state drift, while a marginal scalar mean-map correction accounts for 4.47%.

It does **not** establish that the state drift requires full factorized K3 at 4.05
budgets, and it does not bound ARC's augmented K2. The official augmented K2
propagates selected joint `d=3` and `d=4` harmonic/trace components precisely so
they can change the next layer's covariance/variance state. Its projected leading
cost is `C/B≈0.087`, still below the score floor before the residual gate.

The registered CP3R gate was `O3` versus `O0`; it passed at 99.08%. The repository
then substituted `O1` versus `O3` after observing the data and used that different
estimand to kill the lane. The mechanistic observation is real; the strategic
closure is neither protocol-clean nor scoped to actual augmented K2.

**Decision:** reopen faithful augmented K2 by external evidence and correct scope.
Do not reopen marginal final-mean Edgeworth correction.

Primary references:

- [ARC companion paper](https://arxiv.org/abs/2605.05179)
- [ARC reference implementation](https://github.com/alignment-research-center/mlp_cumulant_propagation)

## What the public frontier teaches

### Confirmed public methods

Team Puffi's Phase-1 score system combined exact radial integration, a
Kerdock/Walsh angular design, a compact high-order analytic prefix, and a
mid-network control variate. Its analytic state carried mean, full covariance,
marginal third/fourth moments, a paired fourth-order surrogate, and a finite
history of mixed third-order responses. A ridge response then used the discrepancy
between sampled and analytic intermediate states to correct the terminal estimate.

Puffi's separate algorithmic-contribution entry, “Separate-Universe Response
Closure,” represented fourth cumulants by a small signed set of covariance-response
modes. It was not score-competitive in Phase 1, but it is direct proof that compact
joint-state closures exist; the 1024x16 shape is much more favorable to them.

Zachary Martinot's Phase-1 note independently argues that an efficient estimator
must follow the network composition. It proposes a Kalman-like predict/correct
closure rather than a control variate that tries to cut through the whole depth,
and reports a contracting angular attractor. The public ceiling study reports a
Phase-1 backward effective rank near 2.7. Neither Phase-1 observation automatically
transfers to Phase 2, but together they motivate low-dimensional *response modes*,
not arbitrary low-rank covariance truncation.

Relevant forum threads:

- [Puffi Phase-1 write-ups](https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175)
- [Honest Phase-1 frontier/field guide](https://discourse.aicrowd.com/t/phase-1-solution-ideas-for-phase-2-and-the-honest-public-frontier-all-methods-combined-and-explained/18193)
- [Measured method-family ceilings](https://discourse.aicrowd.com/t/write-up-submission-326725-measuring-the-ceiling-of-a-method-class-how-we-closed-54-families-of-estimators-without-building-them/18182)
- [Notes on the problem](https://discourse.aicrowd.com/t/notes-on-the-problem-phase-1/18188)

### What can be inferred about the Phase-2 leaders

**INFERRED, not confirmed:** Puffi's Phase-2 carrier is a 16-basis Kerdock/MUB-style
angular construction with antipodes and exact radial integration.

The evidence is unusually specific:

- exact cost is within 0.0264% of 32,768 dense directions;
- `32,768 = 16 * 1024 * 2` is exactly 16 width-1024 orthogonal bases with
  antipodes;
- Puffi's published Phase-1 carrier is precisely a Kerdock/Walsh mutually
  unbiased-basis construction with antipodes and exact radial integration;
- Team Puffi reached first place with one Phase-2 entry.

This would be a structured sample/quadrature estimator, not ordinary IID Monte
Carlo. Its `3.603e-8` raw MSE is about 74x below the IID variance estimate at the
same 32,768 rows, far above the repo's 2.7x sampling-reopening threshold. External
evidence has therefore reopened structured sampling even though the code is not
public. Random orthogonal blocks and the broken two-cycle lattice do not bound a
mutually unbiased Kerdock subset.

An expensive higher-order analytic system is no longer the best explanation for
Puffi's *leading* cost. It may still be used as a tiny reference/control or by
other teams. The leaders' raw MSE also happens to match CP3R's `O3` oracle scale
(`3.8785e-8`), but that numerical coincidence is not proof of a cumulant mechanism.

## Target architecture

The best evidence-backed target is now a **structured angular carrier with a
compositional analytic control branch**:

```text
16 width-1024 mutually unbiased/Kerdock bases
  -> antipodal directions + exact Gaussian radial factor
  -> dense 16-layer flopscope forward propagation (32,768 directions)
  -> mean at every layer
  -> optional shallow augmented-K2 / response-state reference
  -> optional cross-fitted checkpoint discrepancy control
  -> final all-layer means
```

Design principles:

1. **Reproduce the apparent carrier before inventing past it.** The cost fingerprint
   is the cheapest discriminating evidence available.
2. **Separate radial and angular integration.** Positive homogeneity makes the
   Gaussian radial mean exactly integrable; do not spend samples on radius.
3. **Use mutual unbiasedness, not merely random orthogonality.** CP1's orthogonal
   arm does not test the Kerdock structure the leader appears to use.
4. **Correct state, not only the final mean.** If an analytic/control branch is
   added, the oracle says accumulated `(mu, variance)` drift is the dominant target.
5. **Use joint structured cumulants.** Marginal skew/kurtosis is not the same
   object as the harmonic/response pieces that feed future covariance.
6. **Preserve composition.** A control variate should compare sampled and analytic
   state at a checkpoint, not simply blend two terminal means.
7. **Spend where accuracy pays.** The leader spends about 0.5B. The objective is
   MSE times multiplier, not staying at the 0.1 floor.
8. **Keep two deployable branches.** A conservative K2p2 entry is a failure-safe
   anchor; the structured carrier/control entry is the prize bid.

## Ranked research program

### 1. Reproduce the 32,768-direction Kerdock fingerprint — highest EV

Adapt the repository's width-256 Kerdock derivation to dimension 1024 and generate
a legal subset of 16 mutually unbiased bases. Evaluate both signs of all 1024 axes
and integrate the Gaussian radial factor analytically. Match the leader's leading
dense cost exactly before adding any control branch.

On a fresh DESIGN block compare:

- IID at 32,768 rows;
- randomized orthogonal blocks at 32,768 rows;
- Kerdock/MUB 16 bases with and without antipodes/radial separation;
- K2p2 as the deterministic reference.

The first falsifier is variance reduction below 2.7x versus IID or adjusted score
failing to beat K2. If the carrier approaches the public `3.6e-8` raw scale, freeze
its basis-selection rule before a new DECIDE block. Do not tune on the public 50.

### 2. Faithful ARC augmented K2 — highest-confidence analytic lane

Port the official harmonic/diagonal-slice equations faithfully. Validate on small
width/depth against the reference implementation before building a flopscope
spelling. Measure per-layer errors in mean, variance, and final mean, not just the
terminal output.

Kill only on a preregistered score/state gate or official residual/memory failure.
P2-CP3R is not a valid kill for this lane.

### 3. Public A716/response-closure replication — high value

Reproduce the public compact carriers incrementally:

- full covariance + marginal third/fourth state;
- paired fourth-order covariance-response modes;
- finite mixed `(2,1)` third-cumulant history;
- a small number of Hermite/Wick response contractions;
- layerwise coefficients fit on DESIGN trajectories, with rollout rather than
  teacher-forced evaluation.

Fit to the next-layer/state error first. A final-readout fit can hide accumulating
state mistakes and does not transfer as cleanly.

### 4. Recursive cross-fitted control variate — medium/high value

At one or more checkpoints, compare a small structured sample ensemble with the
analytic state, shrink the innovation, and propagate only the downstream effect.
Cross-fit pilot and main samples so the final estimate is not biased by fitting its
own coefficient. Candidate angular carriers are valid scrambled Sobol, randomized
orthogonal blocks, or a reduced public Kerdock family; the invalid two-cycle lattice
is excluded.

### 5. Shared-scale Gaussian mixture — medium evidence, novel application

Track a small latent positive scale/log-norm fluctuation and condition the Gaussian
closure on 2–3 fixed quadrature nodes. This targets multiplicative norm/kurtosis
fluctuations without component explosion. Continue only if residual state error
correlates with the predicted scale variance and the gain pays for the extra
components.

Supporting literature: finite ReLU priors approach a normal-lognormal mixture in a
deep limit, and finite-width corrections accumulate in powers of `L/n`:

- [Noci et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ae4503ec3da32f5e9033604744ec45ae-Abstract.html)
- [Hanin, perturbative hierarchy](https://arxiv.org/abs/2204.01058)
- [Gaussian-mixture propagation, ICML 2023](https://proceedings.mlr.press/v202/monchot23a.html)

### 6. Permutation-equivariant trajectory corrector — medium evidence

Train a tiny shared update for mean/variance from the closure trajectory, row norms,
firing statistics, trace/skew/kurtosis order parameters, and depth. It must respect
hidden-neuron permutation symmetry, train on its own rolled-forward states, and
generalize to unseen seeds. This is not the dead Phase-1 terminal residual model.

Reference: [DWSNets, ICML 2023](https://proceedings.mlr.press/v202/navon23a.html).

### 7. Proper scrambled-Sobol reopening — bounded side lane

CP1 never tested a functioning QMC carrier. Test randomized scrambles, exact radial
integration, and multiple independent scrambles only after the Kerdock fingerprint.
It needs at least **2.7x variance reduction over IID at N=6,546** merely to beat
the existing K2 result. High-dimensional nonsmooth QMC theory gives no reason to
assume success; the live Kerdock inference is evidence for structured algebra, not
for Sobol in general.

### Deprioritized

- marginal final-mean Edgeworth correction;
- higher Price/Hermite covariance orders beyond p2;
- ordinary IID/random-orthogonal sampling as the whole estimator;
- full 1024-dimensional Kerdock/MUB completion (~1.05M directions, ~16 budgets);
- generic sigma-point/degree-5 cubature;
- full uncompressed factorized K3;
- conditional line integration unless a cheap region-count mechanism is first
  demonstrated.

## Validation architecture required before promotion

### Foundation repair — IMPORTANT, no seed block required

1. Upgrade the lock to WhestBench 0.16 / flopscope 0.12 while retaining explicit
   named Phase-1 replay constants.
2. Create a single round-aware configuration source. Phase 2 is `B=2**41`, lambda
   zero, residual `.4`, wall `120`, memory `8192`, setup `5`.
3. Build local MLP weights as official immutable flopscope arrays and add an exact
   operation/FLOP regression against the official one-row subprocess anchor.
4. Make official `whest run --runner subprocess` the deployment evaluator. Keep
   the in-process lifecycle for paired research only and label its memory/process
   limitations.
5. Test the residual cliff, worker death/restart, 8-GiB native memory, over-budget
   zero fallback, and all 1024x16 shape/dtype paths.
6. Rebuild the manifest/checklist and make preflight assert the live stack/config,
   Phase-2 mini smoke, and Phase-2 shape/GPU parity.
7. Re-label P2-CP1 operational cost/failure claims provisional where the official
   residual cap was omitted. Its paired MSE comparisons remain useful.

This is an IMPORTANT instrument slice under `AGENTS.md`: Samuel's written
prediction, preflight, a knowledge-delta LOG entry, and an independent Skeptic are
required before it is changed.

### Research evidence tiers

- **DESIGN:** separate reusable seeds for coefficient/rank/checkpoint selection.
- **DECIDE:** a new SINGLE_USE block for every strategic comparison. P01 is a
  reusable DESIGN block and already consumed 144/600 seeds; 456 remain, not 504.
- **CONFIRM:** public mini is read-only and never fits a constant.
- **VAULT:** create a separate Phase-2 vault only with Samuel's authorization; the
  Phase-1 256x32 vault cannot be reused.

Every experiment reports:

- raw and truth-noise-corrected MSE by layer;
- paired arithmetic score difference and valid t/basic-bootstrap interval;
- exact official FLOPs and multiplier;
- residual, wall, native memory, and failures;
- biggest-mover/tail sensitivity;
- fit/design/decision seed separation;
- a declared falsifier and an independent Skeptic verdict.

For fitted state models, evaluation must be rollout-based: the model sees the
trajectory its own corrections create. Teacher-forced one-step accuracy is only a
diagnostic.

## Execution plan to October 17

| window | outcome |
|---|---|
| Aug 24–27 | official-stack/evaluator repair; live docs/manifest; one-row and mini anchors |
| Aug 27–Sep 3 | width-1024 Kerdock/MUB generator; reproduce the 16-basis/antipodal/radial cost fingerprint; K2p2 conservative candidate |
| Sep 3–12 | structured-carrier DECIDE; faithful augmented-K2 small-shape parity and DESIGN evidence |
| Sep 12–24 | augmented-K2 decision; Puffi-style checkpoint control and response/scale branches; kill weak lanes quickly |
| Sep 24–Oct 5 | aggressive architecture decision on fresh seeds; 300–600-network robustness; create Phase-2 vault only if authorized |
| Oct 5–12 | public mini CONFIRM, official subprocess tails/memory, package/code-policy audit, contribution-write-up evidence |
| Oct 12–17 | freeze two candidate hashes, nominate up to two, no late unmeasured changes |
| Oct 17–24 | finish algorithmic-contribution write-up tied to one successfully graded Phase-2 submission |

Do not submit without the repository's existing explicit-approval sequence:
official validate, clean mini subprocess with zero failures, candidate hash and
evidence presented to Samuel, then ask.

## Immediate next decision at audit time

The first implementation is not yet authorized by this audit because it changes
the instrument and then reopens a strategic lane. Before data exists, record
Samuel's predictions for:

1. how much the official-stack repair changes the current K2 score;
2. the expected variance reduction of the 16-basis Kerdock carrier versus IID at
   32,768 directions;
3. the expected raw-MSE gain and residual-failure risk of faithful augmented K2.

The authoritative continuation and exit gates now live in `GAME_PLAN.md`: commit
the IMPORTANT foundation contract, repair the evaluator, obtain an independent
Skeptic verdict, and only after that preregister the structured-carrier
DESIGN/DECIDE sequence and the separate augmented-K2 decision block.
