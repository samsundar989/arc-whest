# Research notes & intel (living document)

*Everything we know, with sources. Snapshot dates matter — AIcrowd pages were crawled ~Jul 28–29 and re-checked Aug 3–4, 2026. Update as Phase 2 develops.*

> **Live Phase-2 override — 2026-08-31.** This is a chronological research notebook,
> so the Phase-1/pre-launch tables below remain historical rather than being
> rewritten. Do not use them for live constants or research ordering. Current
> official facts and forum/leaderboard evidence are in
> `PHASE2_FRONTIER_AUDIT_2026-08-31.md`; the sole active research program is
> `GAME_PLAN.md`.
> Phase 2 is 1024x16 on WhestBench 0.16 / flopscope 0.12 with `B=2**41`, lambda 0,
> an installed residual cap of 0.4 s/MLP, an installed participant wall default of
> 120 s/MLP (public graded ledgers prove only that `>=104.766 s/MLP` passes), and
> solution memory 8 GiB.

## 1. Historical confirmed Phase-1 competition spec

| Item | Value | Source |
|---|---|---|
| Network | ReLU MLP, width 256 × depth 32, **no biases**, square layers | starter kit `docs/concepts/problem-setup.md`; HF card |
| Weights | He init, i.i.d. N(0, 2/256), float32; forward `y = ReLU(W.T @ x)`, weights stored (input, output) | problem-setup.md; `local_engine.py` |
| Input dist. | x ~ N(0, I₂₅₆) | overview; problem-setup.md |
| Predict | (depth, width) array of per-neuron post-ReLU means; **final layer scored** | estimator-contract.md; townhall |
| Metric | mean over MLPs of `final_layer_mse × max(0.1, C/B)`; failures → zeros + multiplier 1.0 | scoring-model.md; leaderboard math confirms 0.1 floor (overview page's 0.5 is stale text) |
| Compute | `C = flopscope FLOPs + 1e11 × residual_wall_seconds`; B = 2.72e11 (Phase 1); warm-up was 6.8e10 (forum; overview said 3.4e10 — unresolved discrepancy, moot now) | scoring-model.md; forum 18026 |
| Ground truth | MC, **N = 1e9** samples/MLP → GT noise ≈ 2e-10 MSE | ground-truth.md; HF card |
| Grader | CPU-only, 16 vCPU / 64 GB, no network; only flopscope + whestbench + stdlib importable; arrays immutable; 4 GiB/array cap; hidden per-MLP wall cap (read `admin_config.wall_time_limit_s` in run reports); `setup()` off-budget ≤~5s | forum 17997; stage-5 docs; FAQ |
| Submission | **Historical Phase-1 row:** `whest submit --estimator estimator.py`; 50 MiB / 50 files; `.npz` (no pickle). **Live Phase 2:** 10/day/team and up to two designated submissions are privately rerun on fresh MLPs | Phase-2 Rules v14; forum 18143 |
| Seeds | all predict-time randomness from `mlp.seed`; setup-time from `ctx.seed`; violations risk DQ; determinism re-checked on re-grade | 01_random.py; code-patterns.md |
| Dataset | HF `aicrowd/arc-whestbench-public-2026` — tags `v1-warmup`, `v1-phase1` (mini=100 / full=1,000 MLPs); **no v1-phase2 tag as of Aug 4** | HF API refs |
| Timeline | Phase 2: Aug 1 – Sep 19 23:59 UTC; private re-runs Sep 20–30; results Oct 1; NeurIPS ws Dec 11–12; contribution write-up due Sep 26 | overview (re-fetched Aug 3); forum 18041 |
| Prizes | Phase 2: $50k/$20k/$10k + $20k contribution (Phase 1's $50k separate, closed). Winners: OSI open-source; LLM use allowed w/ disclosure | overview; forum 18026, 18041; townhall |

### 1b. Phase-2 toolchain — read-only reconnaissance, 2026-08-19

Round 1429 is **UPCOMING, not open** (`phase2_checklist`): window
`2026-08-20T23:59Z -> 2026-09-19T23:59Z`, leaderboard 1480 exists with **0
entries**, HF tags still `['v1-phase1', 'v1-warmup']`. **width, depth, biases,
input distribution, `n_mlps`, `B`, λ, the multiplier floor and the number of
designatable submissions remain UNKNOWN for 1429.** Nothing in the table above is
confirmed for Phase 2; it is Phase 1 carried forward.

**whestbench 0.15.0 shipped 2026-08-17** (installed: 0.14.0) and requires
`flopscope>=0.11.0,<0.12.0` (installed: 0.10.0). The wheel was downloaded to a
scratch directory and diffed **without installing it**:

| module | 0.14.0 vs 0.15.0 |
|---|---|
| `scoring.py`, `budget.py`, `domain.py`, `generation.py` | **byte-identical** |
| `simulation.py`, `dataset_torch.py` | comments only (version strings) |
| `runner.py` | a predict-response timeout grace |
| `cli.py` | changed |

**So the score identity, the 0.1 floor, λ = 1e11, and He-init / no-bias MLP
generation are unchanged in the newest published evaluator.** `MLP` still has no
bias field at all, so a Phase 2 *with* biases would require a package change that
has not shipped.

**What this does NOT tell us, and must not be read as telling us:**
1. the **contest configuration** (width/depth/`n_mlps`/`B`/λ/floor) is a runtime
   spec, not a package constant — it is passed in as a `ContestSpec`;
2. **what flopscope 0.11 does to the price table is unknown.** flopscope is the
   *pricing* engine, i.e. the cost half of the score and this project's weakest
   half. A repriced gather or matmul would move every cost claim we have.

Verified by the P0.1 slice: the reported leaderboard score is the **arithmetic
mean over the 50 PUBLIC MLPs**, reproducible offline bitwise from a graded report.
A failure is priced at multiplier **1.0**, not the 0.1 floor.

## 2. Baselines & reference numbers (256×32, B=2.72e11, mini split)

| Method | Final-layer MSE | FLOPs | Adjusted |
|---|---|---|---|
| zeros | ~0.91 | ~0 | ~9.1e-2 |
| random U[0,1) | ~0.75 | small | ~7.5e-2 |
| mean propagation | ~9.5e-4 | 11.2M (0.004% B) | ~9.5e-5 |
| covariance propagation | ~8.4e-5 | 1.6e9 (0.6% B) | ~8.4e-6 |
| + published scalar bias fix | ~2.8e-5 | ~same | ~2.8e-6 |
| **ours: + per-layer calibration** | see experiments/LOG.md | ~3.3e9 (f64) | see LOG |
| plain MC at full budget (~65k samples) | ~2.8e-6 | 2.72e11 | ~2.8e-6 |
| RQMC + exact layer 1 (public) | — | C/B=0.42 | 4.10e-7 |
| hybrid structure-aware (public) | 2.18e-7 | — | 1.551e-7 |
| **Phase 1 #1** | 3.12e-8 (col.) | ~40% B | **1.24e-8** |
| GT noise floor | ~2e-10 | — | — |

~~Phase 1 top-10 adjusted: 1.24e-8, 2.30e-8, 2.32e-8, 4.63e-8, 5.59e-8, 6.38e-8, 9.18e-8, 9.47e-8, 9.76e-8, 1.06e-7. 636 participants / 76 teams / 13.2k submissions (Jul 28 snapshot).~~ **SUPERSEDED** — see §5b for the live board measured Aug 4 (#1 4.87e-8, #10 9.85e-8, #200 6.40e-7). The Jul 28 numbers match no live round; do not use them for target-setting. Warm-up (256×8) winner: 3.91e-8 (round 1425 board, confirms the "<4e-8" Manifold resolution).

## 2e. What makes a QMC estimator unbiased — a lesson that cost us a 2.35× regression

**Do not "standardise" a QMC point set.** Forcing exact per-coordinate mean 0 and
variance 1 on the shipped Sobol set, with **no optimisation at all**, moved held-out
MSE from 9.156e-6 to 2.152e-5 — a **2.35× regression** (measured 2026-08-05,
`local_sim/g2_optimize.py`).

Mechanism: a QMC estimator is unbiased because its points are the correct **quantile
transform** of an equidistributed net, *not* because their sample moments happen to
equal the distribution's moments. The Sobol set's per-coordinate sample moments are
mean **−0.00149**, std **1.00278**. Rescaling to force (0,1) applies a ~0.3% affine
map to every point, and a systematic 0.3% scale error on activations of ~0.45 is far
larger than the ~1e-6 MSE we are chasing. Moment matching trades a negligible
variance reduction for a **first-order bias **[WITHDRAWN 2026-08-19 — the mechanism is wrong. Moment matching's bias is O(1/N) (Glasserman 4.5.1 p.245), and diagonal forcing measures at exactly 1.000x both here (LOG 2026-08-19) and in forum 18151. The 2.35x may be real; this explanation of it is not.]****.

Corollary, equally counter-intuitive: the net's most extreme point — the first Sobol
point, which maps to z = −6.12 in *every* coordinate, norm 97.9 against a typical
16.0 — is **load-bearing**. Deleting it as an "outlier" costs 0.734×. The extreme
points are what balance the low-discrepancy structure.

(For plain iid Monte Carlo, standardising is a legitimate and well-known variance
reduction. The lesson is specifically that it does not transfer to QMC.)

## 2g. Scoring-channel forensics — measured 2026-08-05. **PRIVATE: never the forum, never a public write-up without an explicit decision.**

*Framing: every line below is labelled **verified** (read off a report or the board
and reproducible with a named script), **derived** (arithmetic on verified facts),
or **hypothesis** (fits the evidence, not established). Two of our own hypotheses
died in this section; that is the point of keeping the labels.*

### Verified

| # | fact | how |
|---|---|---|
| V1 | Board columns are `score` = **Adjusted Score** and `scoreSecondary` = **Final Layer MSE** — the board's own `scoreTitle`/`scoreSecondaryTitle`. Our row (rank 87, display name **`neopok919`**) reads 2.719e-7 / 1.3748e-6, matching submission 324420 exactly. So `score/scoreSecondary` is each entry's realised `max(0.1, C/B)`. | `board_snapshot.py`, `board_forensics.py` |
| V2 | **The evaluation suite is fixed across submissions.** The 100 `per_mlp` `mlp_name`s are identical *and identically ordered* in the v0, v1.0, v1.1 and v2.4 graded reports (Aug 4 → Aug 5). | `twin_diff.py --all` |
| V3 | The graded suite is **not** the public data: 1/100 name overlap with public `mini`, 0/100 with `full`. | same |
| V4 | **The public score is the mean over exactly 50 MLPs (indices 0–49)**, and each submission returns *those* MLPs' individual `final_layer_mse`, `all_layers_mse`, `adjusted_final_layer_score` and a 32-long `per_layer_mse` in `results.public_scores`. Their mean reproduces the aggregate `final_layer_mse`, `score` and `all_layers_mse` **bit-for-bit**, and their min/max reproduce the reported best/worst MLP. | report schema |
| V5 | `sampling_mse` = 6.469470189211361e-07, identical to 17 digits across all four submissions — a property of the suite, not of our estimator. Still **undocumented** in `score-report-fields.md`. | reports |
| V6 | Phase-1 board (Aug 5, 18:54 UTC): #1 5.6e-9 with **917 entries**; #2 5.08e-8; top-30 median entries 153. #1's `scoreHistory` shows 4.87e-8 at 06:04Z and **5.6e-9 at 17:21Z the same day**, after a month of ≤1.1× steps. | archived snapshot |

### Derived

**D1 — the reconstruction arithmetic.** For a scored MLP j, submitting prediction p
returns `‖p − t_j‖²/256` to 17 significant digits (V4). Differencing two submissions:

    ⟨p_a − p_b, t_j⟩ = ( ‖p_a‖² − ‖p_b‖² − 256·(MSE_a − MSE_b) ) / 2

a **linear** equation in t_j's 256 unknowns, supplied for all 50 scored MLPs by the
same submission pair. **257 submissions determine every scored reference exactly**;
the leader has 917 (3.6×).

**D2 — implied physics.** Above the floor `adjusted = V·c/B`, so `V·c = adjusted·B`
is comparable across teams. Ours 7.4e4; best *published* method (18106) 4.2e4;
#1 **1.5e3** — 28× below anything published and 49× below us. Flag threshold used in
`board_forensics.py`: >3× below the best published.

**D3 — the bypass cannot reach it.** To hit 5.6e-9 at the 0.1 floor you need MSE
5.6e-8, i.e. N ≈ 2.4e5 samples at the best known per-sample variance ⇒ 7.5e11 real
FLOPs = 2.75×B. Charged only as `λ·wall`, staying inside 0.1B needs sustained
**≥2.8 TFLOP/s f32** on the 16-vCPU grader; a fast AVX-512 CPU delivers ~0.1–0.5.
At 0.4 TFLOP/s the bypass is worth ~4×, taking the best published 1.55e-7 to about
**4e-8** — an order of magnitude short of 5.6e-9. *(Sensitivity stated deliberately:
the conclusion depends on grader throughput, which we have not measured.)*

### Hypothesis (not established)

**H1** — the sub-1e-8 cluster is reference reconstruction against the fixed scored
set, not estimation. Supported by V2+V4+D1+D2+V6. **Does not fit**: #1 ran at
multiplier 0.98, leaving the 10× floor discount unused, which a party that already
knows the answers has no reason to do. Recorded, not explained away.

### Two hypotheses that died here (kept as a record of the method)

* **"`sampling_mse` is reference label noise, so nothing honest can score below
  ≈6.47e-8 adjusted."** **Refuted.** If it were additive noise, no entry could show
  a final-layer MSE below 6.4695e-7; **89 of 200 do**, in a smooth continuum with no
  gap there. The bound recorded in LOG.md on Aug 5 is withdrawn. Best remaining
  reading (hypothesis): the MSE of a reference full-budget plain-MC sampler on the
  fixed suite — ~6–7e-7, which matches.
* **"917 submissions can't leak enough to matter."** Also wrong, and wrong because
  it reasoned about the schema instead of reading it. The aggregate-only channel
  would have left probing ~100× short; `public_scores` is what makes it
  over-determined. **Read the payload before doing the information budget.**

### Standing rules and Phase-2 re-checks

1. **We do not use this channel.** This is not the wall-time grey area of rule 3 —
   solving for reference values is not estimation. It also cannot survive the
   private re-run on fresh MLPs, which is what sets the final ranking.
2. Nothing here goes on the forum. The organizer email (`docs/email-draft-organizers.md`)
   states V2/V4 as neutral observations and asks whether they are intended; it names
   no one and draws no inference.
3. **Read our own graded scores correctly**: they are a deterministic function of our
   estimator on 50 *fixed* MLPs. Our submission-to-submission comparisons are
   therefore *paired* and cleaner than we assumed — but the level carries a fixed
   7–20% offset versus the public 100 and is never evidence a change helped.
   Decisions use freshly claimed canonical blocks; graded/public data is external
   calibration and CONFIRM only.
4. **Historical forensics item, now superseded:** the first Phase-2 graded reports
   exist and their conclusions are recorded in `CURRENT_STATE.md`. Do not rerun the
   old fixed-suite investigation as a model-selection loop. Board snapshots remain
   read-only monitoring artifacts.

## 2h. Evaluation mechanics — what actually decides the prize

> **Current correction, 2026-08-28:** Challenge Rules v14 explicitly permits a
> team to designate **up to two Phase-2 submissions** for fresh private re-runs.
> If the team designates none, Rules v14 §6 selects the two best-scoring valid
> submissions from that Phase's public leaderboard. The older one-submission text
> below is retained as a historical record and is superseded.

> Placeholder for the strategy session's own §2h and
> `docs/eval-mechanics-and-reconstruction.md`, **neither of which had synced to this
> repo** when this was written. Merge rather than overwrite.

**The frame, as currently confirmed:** prizes are decided by a **private re-run of
up to two designated submissions per team on fresh MLPs**. The public leaderboard is not the
ranking; combined with §2g (a fixed 50-MLP scored set with exact per-MLP feedback)
it is a reconstruction game, and rank on it carries no information about how a
method will do on unseen draws.

**Operational consequences, which is why this section exists:**

1. **Decide on fresh canonical mints; confirm on the public mini.** The public
   100-MLP mini is CONFIRM-only and is never used for fitting or model choice.
   Binding decisions use newly claimed single-use blocks under `VALIDATION.md`;
   robustness soaks run only after the decision.
2. **The distribution matters, not the mean.** Up to two designated submissions,
   fresh draws, and a failed MLP scores as zeros — so p99 and max are first-class metrics.
   `local_sim/edge_cases.py` covers the contract failure modes on the shipped file.
3. **Don't chase published numbers.** 18106's 1.551e-7 is attributed to the
   dead/on/kink family that Sprint 3 measured end-to-end at ~1.02×; either the
   method differs from its description or the number reflects the board. Trust
   local measurement.
4. **Open questions for the organizers** (drafted in `docs/email-draft-organizers.md`,
   not yet sent): how designation works and by when; the re-run set's size and
   architecture (the townhall said "finalized closer to the time"); and whether the
   contribution-prize top-K screening pool comes from the public board or the
   private re-run.

## 3. Key forum threads (discourse.aicrowd.com)

- **17997** challenge open — grader specs, 50/day. `/t/arc-white-box-estimation-challenge-is-now-open/17997`
- **18026** Phase 1 launch — 256×32, B=2.72e11, +$50k, `v1-phase1`. `/t/phase-1-launch-deeper-models-and-increased-prizes/18026`
- **18041** contribution-prize criteria — "mechanistic" bias, write-up format, deadlines (P2: Sep 26). `/t/algorithmic-contribution-prize-guidelines-.../18041`
- **18053** RQMC + Rao-Blackwell write-up (evaaaz) — 4.10e-7, unbiasedness proof, rank-1 lattice. `/t/unbiased-randomized-qmc-rao-blackwell-.../18053`
- **18063** Gaussian-closure scale bias (pscamillo) — cov-prop overestimates final means by stable ~0.992×; scalar fix → 3× MSE cut; residual signal decays exp(depth/τ), τ≈5.1 layers; negative results: K=3 cumulants, low-rank corrections, learned correctors all failed to beat the scalar at depth 32. `/t/phase-1-write-up-characterizing-a-systematic-scale-bias-.../18063`
- **18078** townhall summary + recording (youtu.be/pvCgbUJ-nTo) — final-layer-only scoring; OSI; fresh private MLPs; ARC hints: cumulant prop > QMC long-term, low-rank refinements, Edgeworth divergence/Borel resummation. `/t/townhall-summary-recording/18078`
- **18085** variance/bias budget write-up (radiant-allomancer) — RQMC prefix + shrinkage + Edgeworth + offline ridge corrector = 4.47e-7; **≥58% of residual is bias not variance**; control-variate coupling dies by layer 32 (0.86→0.25); third-cumulant transport loses ~98% signal. `/t/phase-1-write-up-a-variance-bias-budget-.../18085`
- **18099** ⚠️ flopscope bypass bug (Kerensa) — raw NumPy-backed arrays escape FLOP counting; billed only wall-time at λ=1e11 < real AVX throughput; jtel: top submissions show instrumented/effective ratios <0.001. No organizer reply as of crawl. `/t/potential-flopscope-accounting-bypass-bug/18099`
- **18106** structure-aware hybrid write-up (natasha_stewart) — dead/on/kink classification, antithetic Sobol to layer 30, linearized "on" neurons last layers → **1.551e-7**. Still the strongest published Phase-1 method, 2.5× ahead of our 3.893e-7. `/t/phase-i-submission-structure-aware-estimation-.../18106`
- **18097** trajectory-calibrated moment chain (submission 314695) — read in full 2026-08-05, see §3e for the verdict.
- **18108** mliston (#4) proposal to fix wall-time channel (cap share / raise λ to 1e12 / require instrumented APIs). `/t/recommendation-restoring-the-estimation-framing-.../18108`

## 3b. The re-score story — RESOLVED 2026-08-05 (hypothesis partly wrong)

**Working hypothesis was**: "flopscope 0.10 closed the wall-time bypass and the board was re-scored, forcing the extension." **Verdict: the re-score and extension are confirmed, but the bypass is NOT the cause and appears still open.**

Timeline, from the threads:

| date | thread | what happened |
|---|---|---|
| Jul 16–17 | **18076** | Zero-sized contractions charged **negative** FLOPs — "increases the remaining budget inside a BudgetContext". Fixed, PR #146 + regression tests. |
| Jul 23 | flopscope 0.9.0 | "dtype-aware four-factor cost model + reviewer-driven re-tiering (#150)" — a full accounting overhaul. |
| Jul 23–25 | **18082** | `fnp.linalg.solve` undercharged batched RHS by the **whole batch factor** (256×256 @ 256×1024 and 40×256×1024 both billed 145,402,538). Exploitable by "rewriting each layer `A @ W` as `solve(inv(W.T), A.T)`". Organizer confirmed and fixed: *"once the fix is in, the affected submissions will automatically be evaluated using the patched flopscope."* |
| Jul 25–26 | **18099** | The raw-NumPy-array bypass. jtel showed top submissions with instrumented/effective **< 0.001**. **No organizer reply.** |
| Jul 28 | **18108** | mliston's three fixes (cap residual / raise λ to 1e12 / analytical-only scoring). **No organizer reply.** |
| Jul 30 | flopscope 0.10.0 | Prices **destination writes**; **voids symmetry-tag discounts when the tagged buffer is rewritten**; refused ops uniformly free; stops charging for results the server cannot deliver. **Nothing about the raw-array bypass or λ.** |
| ~Jul 31 | **18109** | pranay212 asked for an extension: board "re-evaluated only a few days ago", bugs/PRs still landing, *"we don't currently know what the real scores are"*. Organizer (Mohanty): *"We have therefore decided to extend the Phase 1 deadline by one week."* → **Phase 1 Jul 31 → Aug 7**; algorithmic write-up → **Aug 17**. inverted_donkey pushed back: *"Every flopscope update effectively changes the rules of the competition."* |

**So**: the re-score was driven by the `solve` undercount fix (18082) plus the 0.9/0.10 cost-model overhaul, and *that* forced the extension. The wall-time bypass of 18099/18108 has **no organizer response and no changelog entry** — assume it is still open and that some of the top board remains inflated. Our position is unchanged: build clean (rule 3), and note the organizers' standing commitment that **"every submission made in Phase 1 until the change will be re-evaluated on the updated cost model"** — a fair re-grade helps us.

**Grader stack now**: flopscope **0.10.0**, whestbench **0.14.0** (confirmed in our own graded report). Phase 1 originally evaluated on 0.8.0rc1/0.12.0rc0 — the upgrade is real.

## 3c. Cost-model facts that constrain how we write estimators

1. **Never call `budget_summary_dict()` / `budget_summary()` inside `predict`** (18092). Measured: 0.40 s → **62.4 s** residual on the 100-MLP mini split, score 2.03e-5 → 4.60e-5. It aggregates the whole per-op history, so it is O(ops so far). At λ=1e11 those 62 s are ~6.2e12 FLOP-equivalents ≈ **23× the entire budget**. Compute sample counts analytically instead; if you must calibrate, do it once per shape and cache.
2. **Freeing big temporaries is billed as residual wall time** (18101): ~100–120 µs/op when temporaries are released between instrumented calls vs ~6–8 µs when release is deferred — 13–16×. `fnp.matmul` has no `out=`, so it is not fully avoidable. Practical rule: **fewer, larger ops** — a sampling estimator should use a small number of big chunks, not many small ones.
3. **Destination writes are priced** as of 0.10.0, so `out=` is not a free lunch.
4. **Symmetry tags are voided if the tagged buffer is rewritten** (0.10.0) — our v0 rebuilds `cov` each layer rather than mutating, so its `as_symmetric` discount stands (measured 2.94e7 FLOPs/MLP).
5. λ = 1e11 FLOP/s independently confirmed by 18092's arithmetic, matching our own derivation from per-MLP telemetry.
6. Historic exploit, now closed: batched-RHS `solve` (18082). Do not build on `solve` tricks.

## 3d. flopscope pricing — VERIFIED from the docs (2026-08-05). Read before designing anything.

| item | price | consequence for us |
|---|---|---|
| **float16 / int8 / bool** | **rate 1.0 — same as float32** | **There is no dtype trick.** Do not spend time on half-precision or quantised paths. float64 is rate 2.0, so f32 remains the right choice, and that is the whole of it. |
| matmul (M,K)@(K,N) | **(2K−1)·M·N — by SHAPES ONLY** | Sparsity is never rewarded. Pruning pays **only** by materialising smaller *dense* matrices (drop dead output columns of W_k *and* the matching input rows of W_{k+1}). Verified: (256,256)@(256,256) f32 = 33,488,896 = 511×65536 exactly. |
| gather / where / sort | 4 / element | Slicing to prune is cheap relative to the matmul it saves, but not free. |
| transcendentals | 16 / element | `stats.norm.ppf` measured at 168/elt (it is more than one transcendental). |
| thin SVD | ≈26n³ ≈ **4.4e8** (0.16% B) | Per-MLP linear algebra is affordable — a basis change costs less than 1% of budget. |
| eigh | 9n³ ≈ 1.5e8 | " |
| inv | 2n³ | " |

**The scoring identity that follows.** Above the 0.1 multiplier floor,

    adjusted = MSE × C/B = (V/N) × (N·c/B) = **V·c/B**

— per-sample variance × per-sample cost, with N cancelling entirely. (Verified: V = 1.4893e-6 × 16384 = 2.44e-2, c = 4.273e6, B = 2.72e11 → 3.83e-7 vs v1.1's graded 3.893e-7.) So *cheapening a sample is worth exactly as much as de-noising it*, and choosing N is not a lever at all once past the floor. This is why dead-neuron pruning — pure cost reduction — is the strongest Sprint-1 item.

**Why α = 1.00 (theory, He & Wang 2015).** For discontinuous integrands the RQMC convergence-rate bonus over plain MC is 1/(4d−2), which at d=256 is ≈0.001. Sobol can therefore only ever buy us a **constant factor**, never a better rate — exactly what we measured (α = 1.00 on the full 100-MLP split). Rate recovery requires concentrating the discontinuities (ReLU kinks) into a few coordinates, which is precisely why an input rotation moved α (0.87 → 1.01/1.06) and is the only structural lever on the rate.

## 3e. Thread 18097 — "trajectory-calibrated moment chain" — read 2026-08-05. **Verdict: no analytic assist; write-up material only.**

**What it is.** A k=2 moment chain (not K≥3 — "*the depth-32 failure of k=2 propagation is error dynamics, not closure order*") with **per-layer linear corrections fitted on the chain's own rolled-forward trajectories** — a DAgger-style calibration. The key claim is that corrections must be fitted against the chain's *own* behaviour, because successive layers' errors anticorrelate and partly cancel; fitting to *true* cumulants reportedly makes things **worse**.

**Its numbers.**

| | |
|---|---|
| raw final-layer MSE | 5.89e-6 |
| adjusted, post-erratum | **1.76842e-6** (multiplier 0.300) |
| adjusted as originally quoted | 9.97e-7 (multiplier 0.169) — superseded by the flopscope 0.10.0 repricing |
| cost | 7.40e10 FLOPs/MLP in f64 (3.70e10 f32-equivalent) |
| its own summary | "*well short of the sampling frontier*"; "*Sampling wins Phase 1's raw metric, and the write-up says so plainly*" |

**Verdict for us: it does not re-open the analytic track.** Their adjusted 1.768e-6 is **4.5× worse than our v1.1's graded 3.893e-7**, at roughly the same compute. Not usable as a pruned-tail patch (our patch is the closed-form rectified-Gaussian mean, already at 3.8e-9 on pruned neurons) nor as a low-weight variance-free base (that is exactly the B1 fusion we measured at net ~0, and their residual variance would be *larger* than the closure's).

**But two of their findings corroborate ours precisely, and one refines a result:**

1. **"the chained estimator attenuates zero-mean noise while amplifying small coherent biases (measured end-to-end at a ~16:1 exchange rate)"** — an independent, quantified version of why our fusion family died. Same closure "*is 52× worse*" when chained.
2. Their **true-input closure beats sampling: 1.16e-6 vs 1.61e-6.** This *refines* our B2 kill. We measured analytic-tail-from-**sampled**-moments at −67% (k=1); they measure analytic-tail-from-**exact**-moments as a win. The entire gap is therefore **moment-estimation noise** — a 256×256 covariance estimated from 16k samples — not closure bias. Our k=0 recommendation stands for the practical case, but the reason is sharper than we had it, and it says a cheaper/better *moment estimator* (not a better closure) is the only thing that could revive the tail.
3. Even with reconstructed true third cumulants (cosine similarity 0.83) they "*still lose 4.5× to a plain N=4096 MC probe, 10/10 IDs*" — third-order structure does not rescue it.

**No other new write-ups or organizer posts as of 2026-08-05** (re-crawled): the set remains 18053, 18063, 18085, 18097, 18106. 18099/18108 still have **no organizer reply**.

## 4. The science (papers that matter, ranked)

**Directly on-task:**
- ARC's method paper: *Estimating the expected output of wide random MLPs more efficiently than sampling* — arXiv:2605.05179 (Wu, Lecomte, Winer, Robinson, Hilton, Christiano). Cumulant propagation w/ Hermite expansions; K-th order → MSE ~O(1/n^K) but cost O(n^{K+1}L); factorized variants O(n^K L²); **known failure: error ~L^K depth compounding; loses to sampling at depth ≥8 for K=4**. Code: github.com/alignment-research-center/mlp_cumulant_propagation
- *Formalizing the Presumption of Independence* — arXiv:2211.06738. The conceptual foundation (heuristic estimators; Wick propagation as worked example).
- ARC blog: *Estimating Tail Risk in Neural Networks* (layer-by-layer activation modeling frame + its known blind spot: cross-layer correlations); *A bird's eye view of ARC's research*; Wu & Hilton arXiv:2410.13211 (low-probability estimation — importance sampling beats extrapolation).

**Toolbox:**
- Gaussian-ReLU closed forms: E[ReLU(X)] = μΦ(μ/σ)+σφ(μ/σ); arc-cosine kernels (Cho & Saul, NIPS 2009) for zero-mean pair covariances; **exact nonzero-mean pair covariance**: Wright et al., AISTATS 2024, arXiv:2403.16163 (bivariate-CDF formulas — the exact alternative to the gain/linearized cross-term; candidate upgrade to our v0).
- Mean-field/NNGP depth theory: Poole et al. 1606.05340; Schoenholz et al. 1611.01232; Lee et al. 1711.00165. He init is ReLU-critical: norms preserved, correlations approach 1 only algebraically (Hayou 1902.06853) → deep layers become near-constant functions (final-layer variance shrinks with depth; observed avg_variance ≈ 0.18 at depth 8).
- Finite-width corrections (the *why* of the bias we calibrate away): Yaida 1910.00019; Roberts/Yaida/Hanin 2106.10165 (expansion parameter L/n = 0.125 here); Hanin 2204.01058 (exactly-our-setting cumulant hierarchies); Hanin & Nica 1909.05989.
- Variance reduction: Owen's MC book (artowen.su.domains/mc); scrambled Sobol; control variates for QMC (Hickernell et al. 2005); MLMC (Giles, Acta Numerica 2015); piecewise-linearity of ReLU nets along lines — Hanin & Rolnick 1901.09021/1906.00904 (few kinks per ray at init → exact radial integration is plausible).
- Weight-space learned models (Phase C): DWSNets 2301.12780; NFN 2302.14040; Universal Neural Functionals 2402.05232; Graph Metanetworks (ICLR 2024); property-from-weights precedent 2002.11448.

## 5. Strategic reading of the field

1. Top of Phase 1 board (~1e-8) is ~100× above GT noise — headroom exists, but part of the top may be wall-time-loophole inflated (thread 18099: instrumented/effective <0.001). A fair re-grade likely compresses the board; **build for the re-grade**.
2. Two independent write-ups agree: at depth 32 the *bias* of Gaussian closure (not sampling variance) is the binding constraint (≥58% of residual; exp-decaying exploitable signal, τ≈5.1). Conclusion: pure sampling plateaus ~1e-7; pure analytics plateaus ~1e-5-ish uncorrected; **the win is bias-corrected analytics + smart sampling of the residual**.
3. ARC's own wishlist (townhall) = Phase D of our plan: depth-robust low-rank cumulants + resummation. They have effectively pre-announced what impresses them.
4. Everyone's negative results (failed learned correctors, dead control variates at depth 32) used *generic* regressors on *final-layer* residuals. Our angle: correct *per-layer* along the way (stop compounding early) rather than patch the end.

## 5b. Round structure — MEASURED 2026-08-04 (supersedes §1 timeline row)

Read directly off the challenge page's round objects:

| round id | name | window (UTC) | state | leaderboard |
|---|---|---|---|---|
| 1425 | warm-up | — | closed | 79 entries, top 3.91e-8 |
| **1428** | **Phase 1** | **2026-06-18 → 2026-08-07 23:59** | **LIVE** | 200+ entries, top 4.87e-8, #10 9.85e-8, #200 6.40e-7 |
| 1429 | Phase 2 | 2026-08-08 → 2026-09-19 23:59 | UPCOMING | empty |

Phase 2 had **not** started on Aug 4 (§1 and the game plan assumed an Aug 1 start — wrong).
Our submission 323872 graded into round 1428. Note the live Phase-1 top (4.87e-8) does not
match the "Phase 1 #1 = 1.24e-8" figure in §2's Jul 28 snapshot; the numbers in this table
are freshly measured and should be preferred.

**Phase 1 config confirmed** from the graded report: 256×32, n_mlps 100, B=2.72e11,
λ=1.000e11 (derived exactly from per-MLP telemetry on all 100 MLPs), multiplier floor 0.1.
`admin_config` is **admin-only and never shown to participants**; `wall_time_limit_s` is
therefore still unknown (lower bound ≥0.6 s). Details in `experiments/LOG.md`.

## 6. Open questions / to verify next

- [x] ~~Phase 1 config~~ confirmed 2026-08-04 (see §5b): 256×32, B=2.72e11, λ=1e11, floor 0.1.
- [ ] Phase 2 config: cannot be checked before the round opens **Aug 8**. First graded run in round 1429 + any `v1-phase2` HF tag will settle it.
- [ ] `admin_config.wall_time_limit_s` — not in the participant report; infer from a deliberately slow submission or ask on the forum.
- [ ] Did organizers respond to 18099/18108 (loophole)? Any flopscope ≥0.10 patch notes?
- [x] ~~flopscope pricing~~ measured on the mini split (per MLP, 256×32, f64): `einsum` W^T Cov W 3.219e9 (98.5% of our total), `as_symmetric` 2.94e7, `outer` 2.1e6, `matmul` 8.4e6, `stats.norm.cdf` 7.9e5 / `pdf` 4.4e5, `diagonal` 0. `standard_normal` untested (v0 draws no samples) — measure when Phase A lands.
- [x] ~~Real per-run wall time~~: grader 0.48 s/MLP wall, of which **6.74 ms is uninstrumented residual** → 6.74e8 FLOP-equiv at λ=1e11 = 17% of our billed compute. Local box: 0.12 s/MLP, 4 ms residual. flopscope's own overhead is *not* charged.
- [x] ~~`fnp.diagonal`/`fnp.outer`/`flops.stats` names~~ all correct — `whest validate` passed with no estimator changes.
- [ ] Team-formation rules for Phase 2 (Phase 1 merges closed Jul 31).
- [ ] Phase 1 final standings & any released winner write-ups (Aug 7 write-up deadline) — free intel, check ~Aug 8.

## 7. Our experiment provenance

Local cloud sim (2 vCPU): He MLPs self-generated; MC truth 200k samples (measurement floor ~2e-6·(0.4)≈; MC-vs-MC reported in experiments/calibration.json); the exact `estimator.py` runs through numpy shims (`local_sim/`). Per-layer calibration factors fitted at 256×32 (K=5 MLPs), validated on disjoint seeds (validate_estimator.py). Numbers in experiments/LOG.md.
