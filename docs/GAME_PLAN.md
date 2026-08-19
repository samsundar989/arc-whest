# Game plan — ARC White-Box Estimation Challenge 2026

> **REV 13 POINTER — 2026-08-19. Read this before any earlier revision.**
>
> Rev 12's strategic thesis — that the cost half of `V·c` is empty and variance is
> crowded, so effort should go to cost — was **half right, and right for the wrong
> reason**. The 2026-08-19 carrier audit measured our *variance* carrier at ~85-95%
> of the achievable ceiling for input-space methods (every device lands at ~1.0x:
> whitening 0.997x, sphere 1.001x, tent 0.981x, antithetic 1.016x, degree-3 design
> 1.047x), and found the remaining headroom on the **cost** side after all —
> Strassen-Winograd, worth 1.150x counted, which rev 12 had closed on the wrong
> carrier. But rev 6's "honest sampling frontier ~2.7e-7" is **withdrawn**: the
> published honest frontier is **9.10e-8** and we are ~2.6x behind it.
>
> The governing mechanism, measured three ways, is **SUBSTITUTION**: two devices
> that annihilate the same low-order structure do not compose, and whichever is
> added first takes the credit. Read every competitor ladder for what its baseline
> arm contains, never for its largest rung.
>
> Current plan of record is **`docs/CURRENT_STATE.md` § Next actions**, not this
> file. Sections of rev 12 that survive: the scoring identity, the cost-ceiling
> table, and the closed-lane reasoning. Sections superseded: the variance/cost
> allocation argument and every frontier estimate.


*Goal: win Phase 2 (Best Score, $50k) and contend for Best Algorithmic Contribution ($20k). Written Aug 4, 2026; **round dates corrected Aug 5** after reading them off the challenge page.*

## Round calendar (measured, authoritative)

| round | window (UTC) | state as of Aug 5 |
|---|---|---|
| warm-up (1425) | — | closed |
| **Phase 1 (1428)** | **Jun 18 → Aug 7, 23:59** | **LIVE — closes in days** |
| Phase 2 (1429) | **Aug 8 → Sep 19, 23:59** | upcoming, empty leaderboard, same grader stack |

Earlier drafts assumed Phase 2 opened Aug 1. Wrong — Phase 1 is still live and closes **Aug 7 23:59 UTC**, so near-term work targets the Phase 1 board; Phase 2 opens Aug 8 and re-verification of its config happens with our first submission there.

## The strategic picture

- **The bar** (live Phase-1 board, measured Aug 4): **#1 = 4.87e-8**, #10 = 9.85e-8, #200 = 6.40e-7. The warm-up round topped out at 3.91e-8. ~~Phase 1's leader sat at 1.24e-8~~ — that late-July figure is **superseded**; it does not match any live round and should not be used for target-setting. Assume winning Phase 2 means **≲4e-8**. Ground-truth noise (~2e-10) says there's real room below the current leader.
- **The two currencies**: leaderboard score (engineering + statistics) and the contribution prize (mechanistic insight + a clean PDF write-up). These pull in the same direction until late September, when we split effort.
- **Our structural advantages**: (1) the data distribution is *fully known* — He-init MLPs — so we can mint unlimited labeled data offline; offline compute is unmetered and we have a recent NVIDIA GPU for it; (2) the cloud workspace runs the exact submission file against self-generated ground truth (local_sim/), so iteration never waits on the grader; (3) 50 submissions/day is effectively unlimited — the grader is our oracle for anything ambiguous (flopscope pricing, wall-time caps, Phase 2 config).
- **Fair play**: the flopscope wall-time loophole (RESEARCH_NOTES §5) is off-limits. It's explicitly against organizer statements, poisons the contribution prize, and a patch + re-grade could vaporize scores built on it. We build methods that win a *fair* re-run. If organizers patch it mid-phase, our relative position improves for free.
- **Unknown to resolve immediately**: Phase 2's exact config (same 256×32? new budget? floor still 0.1? λ still 1e11?). No new dataset revision exists yet and the kit still pins `v1-phase1`, so same-config is the working assumption — but our first graded run report (`admin_config`) + the forum will confirm within days.

## Phase 0 — pipeline online (this week, target: first submission within ~48h)

1. ✅ Cloud repo built: submission-ready `estimator.py` (covariance propagation + offline-fitted per-layer calibration), local simulation harness, docs.
2. Windows setup (SETUP_WINDOWS.md): install `uv`, `uv sync`, `whest login` with Samuel's AIcrowd API key.
3. `whest validate` → fix any API-surface mismatches (shim vs real flopscope) → `whest run` on the public mini split (real flopscope FLOP counts, real scoring) → **submit v0**.
4. Read the graded run report: `admin_config.wall_time_limit_s`, actual budget B, λ, floor, dataset shape → update RESEARCH_NOTES and this plan if Phase 2 differs from Phase 1.
5. Join the Discord / watch the forum for the Phase 2 announcements we haven't seen yet.

*Expected v0 landing: adjusted ~2–3e-6 (calibrated cov-prop at the 0.1 floor). Not competitive — but the pipeline, the score reports, and the calibration machinery all get proven, and we beat the published baseline from day one.*

## Phase A — the honest sampling floor (week 1–2)

Build the tuned RQMC estimator: scrambled Sobol or rank-1 lattice points (implemented in pure `fnp` — Sobol direction numbers ship as shipped weights), antithetic pairs, exact closed-form layer 1, dead/saturated-neuron pruning via propagated firing probabilities. All randomness seeded from `mlp.seed`.

*Target: ~1.5–4e-7 adjusted (validated publicly by two Phase-1 write-ups). Decision data: measure error-vs-samples slope to choose the C/B operating point (for 1/N methods, anywhere between 10% and 100% budget scores the same — spend where the curve says).*

## Phase B — FUSION (revised Aug 5 after Phase A; the expected winner)

Phase A's measurements rewrote this section. The original plan (control variates, radius stratification, Rao-Blackwell tail) is **dead on arrival**: measured against real ground truth, the layer-1 control variate's ceiling is 1.80× and it adds *nothing* on top of QMC (both remove the same variance); radius stratification is 1.20×; antithetic is 1.03×. See `experiments/LOG.md` 2026-08-05.

What the data *did* show is that our two estimators fail in orthogonal ways:

| | error character | size |
|---|---|---|
| v0 covariance propagation | **pure smooth bias**, per-neuron, ~zero common mode | MSE 2.5e-5 |
| v1.1 Sobol RQMC | **pure variance**, flat in N (α=1.00), **42% in one common scale mode** | MSE 1.49e-6 |

Two estimators with uncorrelated errors are worth more together than apart. Hence:

### B1 — precision-weighted fusion (`estimator_v2.py`)
Run both (covprop costs 3.3e9 FLOPs ≈ 4.7% on top of sampling's 7.0e10), then combine in closed form, O(n) online:
1. **Scale-mode kill**: estimate the sampling run's common scale fluctuation ĝ by regressing s on p across the 256 neurons, weighted by 1/Var(ε_i); shrink ĝ by its own estimation noise; correct s′ = s/(1+γĝ). This attacks the 42% directly, using covprop as the reference direction.
2. **Per-neuron GLS blend**: t̂_i = w_i p_i + (1−w_i) s′_i with w_i = Var(s′_i)/(Var(s′_i)+Var(covprop residual_i)).

Error models are fitted offline on the **full 1000-MLP split** (1e9-sample truth = zero label noise) and ship as literals: sampling noise from covprop's own in-flight σ̂² (rescaled — covprop underestimates post-ReLU variance ~30%), covprop residual variance as a low-dimensional function of (σ̂_i, α̂_i).

*Target: graded ≤2.4e-7. Gates: mini ≤3.2e-7, 0 failures, C/B 0.26–0.30, determinism.*

### B2 — push the last k layers analytically
The fusion-hurt map (neurons where the covprop error model is wrong) tells us where the analytic side is untrustworthy. B2 replaces the last k layers' sampled ReLUs with a better analytic treatment, k chosen from that map.

### B3 — GPU learnability probe (2-day box, parallel)
Mint 50k He MLPs on the RTX 5090 (38.3 TFLOP/s measured), label with 64k–256k-sample MC, train a small per-neuron regressor to predict the *calibrated covprop residual*. **R² > 0.5 → Phase C is alive** (a learned error model replaces B1's static one); **R² < 0.15 → drop Phase C**, GPU pivots to B2 calibration data.

*Overall target: ≤5e-8 adjusted. Milestone gate: if not ≤2e-7 by Aug 21, consolidate instead of adding pieces.*

## REV 12 (2026-08-06) — Sprint 5 ran to conclusion. **The packed lane is killed on D14.** Moonshot G0 opens.

Rev 11 sent Sprint 5 to compress the carrier before spending telemetry. It did,
measured every lever, and the lane missed its gate.

**D14 (locked, single-use, now spent): paired `V·C` = 1.1216× [1.1188, 1.1244]
against a pre-registered 1.15×. FAIL.** 100 MLPs × 3 shifts, design and residual
model frozen and written into `experiments/LOG.md` before the block was touched.

Three results are banked and three interpretations are corrected:

1. **The published arithmetic target is reached, counted-to-counted.** Extrapolated
   to 319341's N = 84,992 our frozen carrier costs **2.137e6 counted FLOPs/sample
   against their 2.144e6** — 0.4%. Rev 11 was right that the Sprint-4 headline
   mixed units; corrected, the arithmetic is not merely reachable but reached.
2. **Exact packing moves no variance, definitively.** On D14 control and candidate
   MSE agree to five significant figures (1.5586e-06). D12's 1.2079× is retired.
3. **Operation count is not the residual driver on this backend.** Batched
   depth-2 Strassen is bitwise identical to the recursion and cuts the shipped
   file from 10,263 to 4,823 operations — and measured **worse** residual at all
   three sample counts. Shipped off, kept behind a flag, fully tested.
4. **N is not a lever.** `MSE × N` is flat over {16,384, 32,768, 65,536} once
   measured with 8-shift averaging; the 14% "optimum" a 1-shift pass showed was
   one heavy-tailed realisation. Residual is ~96% marginal in N, so amortisation
   buys 1.11× ± 0.11 locally against 1.002× on CONFIRM.
5. **The structural verdict.** flopscope prices a gather at 4 FLOPs/element;
   packing's entire saving is substituting gathers for dense multiply-adds. A
   random-access gather costs far more *wall time* per element than a BLAS
   multiply-add, and flopscope charges wall time back at λ = 1e11. The lane
   trades an operation that is cheap in the price table for one that is expensive
   in physics. No amount of further engineering changes that exchange rate.

**Priority order from here.**

0. **Round operations stay event-triggered.** Checked 2026-08-06: round 1428 still
   LIVE (closes Aug 7 23:59 UTC), 1429 UPCOMING with an empty board, dataset tags
   still `['v1-phase1','v1-warmup']`. On 1429 opening, the first Phase-2 action is
   still **frozen v2.4.1 for configuration telemetry, and only with Samuel's
   explicit approval.** **No telemetry submission is requested for any packed
   candidate.** `estimator.py` untouched; vault untouched (counter 1).
1. **Moonshot G0 only** (`docs/MOONSHOT.md`). Establish layer-convention parity,
   build the separate calibrated-K2 control path, and treat
   `lowrank_cumulant_flow.py` as an accuracy oracle. **Do not touch D08 and do
   not claim compressed cost.** G1 stays closed until G0's parity is proven.
2. **319341's variance advantage is the only score question left open.** Their
   `MSE × N` is 0.018520 against our graded 0.022525 — **1.216×**, and it is not
   arithmetic, not packing, and not antithetic. It is the highest-value unknown
   on the score lane and it is a *variance* question.
3. **Write-up.** Sprint 5's negative result is unusually clean contribution
   material: a pre-registered gate, a conservative model frozen in advance that
   was later shown to matter (defining it as the competitor anchor alone would
   have passed a carrier that does not deserve it), and a mechanism — price-table
   arbitrage against physics — that generalises past this competition.

## REV 11 (2026-08-06, independent audit) — do not spend telemetry yet; compress the carrier first

Rev 10's core result survives audit: exact activation packing is real, the FLOP
oracle is accurate, and the current v2.5 file must not replace v2.4.1. Three
interpretations do not survive:

1. **S1 is borderline, not a cost-only pass.** The quoted 1.2026x is a ratio of
   mean costs. The pre-registered paired statistic is the geometric mean of the
   per-MLP ratios; recomputed on D12 with variance held neutral it is **1.1996x,
   95% CI [1.1963, 1.2029]**. The full 1.4070x result still exists, but depends on
   the D12-only MSE draw that did not reproduce on D11 or the public mini. Record
   S1 as **inconclusive/borderline** rather than independently passed by cost.
2. **`SETUP_TIMEOUT` is not a demonstrated candidate or memory failure.** The
   local subprocess runner has a hard five-second worker-start deadline. In an
   independent six-start audit, v2.4.1 itself timed out 2/6 times while v2.5
   passed 6/6; v2.5 also completed a real one-MLP subprocess prediction. A 3.9 GB
   process does not establish exhaustion of a 64 GB address-space cap. Keep
   memory bounded, but remove this alleged shipping blocker until reproduced in
   a candidate-specific way.
3. **The like-for-like published variance gap is about 1.22x, not 1.40x.** Use
   graded v2.4.1 `MSE*N = 0.022525` against 319341's `0.018520` (1.216x). The
   1.40x figure compares a D11/local level with a different graded suite and is
   only a diagnostic. Exact packing cannot create a variance gain, but the size
   of the remaining gap must not be overstated.
4. **The published effective-cost target was not reproduced yet.** Our 2.359e6
   number is counted FLOPs/sample. Submission 319341's 2.374e6 number is
   effective compute/sample, including residual; its like-for-like counted
   figure is about **2.144e6/sample**. Under Sprint 4's own pre-registered 66 ms
   residual model, our current candidate is about **2.762e6 effective/sample**.
   The packing arithmetic is validated, but the external cost target remains
   open and must be compared counted-to-counted and effective-to-effective.

The local 100-MLP result remains a valid warning: current v2.5 is 26% worse on
this machine because residual time rises 5.47x. It is not the final remote bill.
Flopscope's published model assigns client/server transport to framework
overhead, while participant Python and temporary release can still land in
residual. The current implementation's 10,266 calls and large temporary churn
are therefore an engineering target, not a reason to jump straight to a graded
oracle.

### Sprint 5 — round-trip compression and fixed-cost amortisation

0. **Round operations remain event-triggered.** First Phase-2 action, only with
   Samuel's approval, is still frozen v2.4.1 for configuration telemetry. Do not
   submit the current v2.5 file.
1. **Compress Strassen calls.** Batch the 49 depth-2 leaf products into batched
   contractions and then reconstruct the quadrants. A clean-room microbenchmark
   was bit-identical and changed one representative block from 219 to 173 calls,
   residual -16%, wall time -40%, for +1% counted FLOPs. Validate on full layers.
2. **Build an operation/FLOP Pareto for packing.** Compare the existing six
   support buckets with one- and two-bucket designs, one row sort plus segment
   boundaries, and packing-only / Strassen-only hybrids. Fewer remote operations
   may be worth modestly more counted arithmetic. Never threshold a nonzero.
3. **Reopen N for the packed carrier.** Rev 10 compares our N=16,384 carrier with
   319341 at N=84,992. With fixed closure, setup, Python, and temporary-release
   costs, N does not cancel completely: `score = V*(c_marginal + F/N)/B`. Sweep
   16,384 / 32,768 / 65,536 on a fresh design block, with per-array size below
   4 GiB and total live memory below 64 GB. The published competitor demonstrates
   that a large-N packed carrier can be viable; do not assume our old dense-N
   conclusion transfers.
4. **Allocate D13/D14 before measurement.** D13 is design/Pareto work. D14 is a
   locked, single-use decision block. D12 stays spent and must never be reused.
5. **Promotion gates.** Preserve S0 exactness. On D14 require a pre-registered
   candidate to improve paired `V*C` by at least 1.15x with its 95% interval
   excluding 1.0 under a conservative residual model, and confirm that the gain
   is not supplied by a one-block variance anomaly. Then require repeated
   subprocess starts for both control and candidate, a multi-MLP subprocess run,
   the full public mini, and all existing invariants. Only after those pass may
   Samuel be asked to approve one optimized Phase-2 telemetry submission.
6. **Moonshot stays next, not concurrent noise.** If the optimized carrier misses
   D14 or real-stack gates, bank the exact-sparsity result and move to Moonshot
   G0 (calibrated K2 control plus layer-convention parity). Do not keep tuning a
   failed packed carrier on confirmation data.

## REV 10 (2026-08-06, late) — Sprint 4: the arithmetic reproduces, the shipping does not. **S1 pass, S2 fail, no promotion**

Rev 9 reopened one score lane on the strength of submission 319341's published
telemetry. Sprint 4 ran it to conclusion. **The mechanism is real and reproduces
the published arithmetic exactly. It still loses on the real stack, because
flopscope charges for the operations that value-dependent packing needs.**

The one-line version: **1.38x cheaper in FLOPs, 5.5x more expensive in residual
wall time, 26% worse in adjusted score on the organizers' own 100 MLPs.**

**What was built.** `local_sim/packed_sparse_probe.py`, a laboratory with a
v2.4.1-equivalent dense reference (**bit-identical** to the shipped file) and an
exactly equivalent packed path, both running against **real flopscope 0.10.0**,
plus a *separate* analytic cost oracle that reproduces the instrumented bill to
0.01%. 62 focused tests. Fresh seed blocks D11 (design) and D12 (locked,
single-use) allocated before any measurement.

**What was measured.**

| | v2.4.1 | v2.5 candidate | 319341 (published) |
|---|---|---|---|
| effective FLOPs/sample, arithmetic | 3.253e6 | **2.359e6** | 2.374e6 |
| counted FLOPs/MLP | 5.333e10 | 3.866e10 | — |
| operations/MLP | 1,416 | **10,266** | — |
| S1 paired `V*c` (pre-registered) | 1.000x | **1.407x** [1.266, 1.564] | — |
| S1 cost half alone | 1.000x | **1.203x** | — |
| **S2: public mini, real stack** | **3.26e-7** | **4.11e-7 (26% WORSE)** | — |
| ... residual wall time / MLP | 70.8 ms | **387.6 ms** | — |

**S1's headline was not the method.** D12's 1.407x leans on an MSE ratio of 1.2079
that the design block puts at 1.0199 and that the organizers' own 100 MLPs put at
1.007. The cost half is the real gain and it does reproduce (0.04% across blocks).
But S2 then measured what the cost half actually costs: **residual wall time went
70.8 -> 387.6 ms/MLP, which at lambda = 1e11 is 3.9e10 of effective compute — as
much again as the entire counted bill.**

**The lane does not promote.** `estimator.py` is untouched, the vault is untouched,
v2.4.1 stays designated. Two things are nonetheless banked:

* **The published arithmetic target is reachable** — 2.359e6 effective FLOPs/sample
  against 319341's 2.374e6, rebuilt independently and exact to 1.5e-8.
* **Their advantage is therefore mostly NOT the packing.** Their `MSE*N` is
  1.852e-2 against our 2.58e-2 — a **1.40x per-sample variance** advantage that
  packing cannot produce and none of our components deliver. That is where their
  score lives, and it is now the open question worth attacking.

**Priority order from here.**

0. **Round operations stay event-triggered.** Round 1428 is still LIVE (closes
   Aug 7 23:59 UTC by the round object); 1429 is UPCOMING with an empty board and
   no `v1-phase2` dataset tag. On 1429 opening, submit frozen **v2.4.1** first for
   configuration telemetry. **No submission without Samuel's explicit approval.**
1. **One graded run settles the only open question.** Everything above is
   arithmetic we can bill ourselves; what we cannot see offline is how the grader
   prices the **7.2x operation count** and the **~310 data-dependent host
   round-trips per MLP** that value-dependent packing requires (`fnp.ndarray` is a
   `RemoteArray` on the eval servers — 18122). Our pre-registered model says
   66 ms/MLP of residual against v2.4's 9.84 ms; the break-even is 75.7 ms. A
   single Phase-2 telemetry submission of v2.5 resolves it and nothing else can.
2. **v2.4.1 remains the designated fallback** until that telemetry exists. v2.5 is
   a *candidate*: separate file, `whest validate` passed, invariants 30/30, public
   mini confirmed read-only, vault **not** touched.
3. **The variance half of 319341 is still unexplained and is the bigger prize.**
   Their `MSE*N` is 1.852e-2 against our 2.58e-2 — a 1.40x per-sample variance
   advantage that packing cannot explain (packing is exact) and that our ablated
   components do not deliver. Whatever buys it is a *variance* mechanism. That is
   now the most valuable open question on the score lane, and it is a better
   target than another 5% of cost.
4. **Moonshot and write-up unchanged** from rev 9.

**Method note that belongs in the write-up.** Two "findings" in this sprint were
noise that arrived wearing a mechanism: a 12-MLP run said antithetic sampling cost
12% of variance (the 30x8 run says it *gains* 0.9%), and D12 said the candidate
wins 21% of variance (the design block says 2%). Both had a tidy causal story
available on demand. The defence that worked was the same one every time — refuse
to accept a number that no ablation predicted until it reproduces on a different
block.

## REV 9 (2026-08-06) — reopen one score lane: packed activation sparsity was never tested

Rev 8 correctly fixed the measurement instrument, but rev 6 closed score work one
experiment too early. Sprint 3 tested **static late-layer on-neuron composition**.
The strongest published honest competitor (submission 319341, regraded under
flopscope 0.10.0) uses a materially different mechanism: after dead-neuron removal
it sorts sampled activation columns by empirical firing rate, packs a cold prefix,
buckets rows by 0/1/2/4/8 active prefix entries, skips the exact zero rectangles,
and uses Strassen-style dense blocks. This is value-dependent exact arithmetic,
not the static shape reduction Sprint 3 priced.

The live telemetry makes the gap concrete. Our v2.4 spends 16,384 samples at
3.303e6 effective FLOPs/sample and has variance proxy `MSE*N = 2.252e-2`.
Submission 319341 spends 84,992 samples at 2.374e6 effective FLOPs/sample and has
`MSE*N = 1.852e-2`. Its regraded adjusted score is 1.599e-7 versus our 2.719e-7:
about **1.39x cheaper samples and 1.22x lower per-sample variance, 1.70x combined**.
That is too large, too public, and too mechanistically specific to dismiss without
reproduction. The old "honest sampling frontier is 2.7e-7" statement is therefore
withdrawn; 2.7e-7 is the frontier of **our current dense sampler**.

### Revised priority order

0. **Round operations are event-triggered.** The live round object still closes
   Phase 1 on Aug 7 and opens Phase 2 on Aug 8, while organizer thread 18125 says
   Aug 10. Trust `phase2_checklist.py`'s live state, not either date. On round 1429
   opening, submit frozen v2.4.1 first for configuration telemetry, then branch.
   No external submission is made without Samuel's approval.
1. **Sprint 4 — packed activation propagation (one tightly gated score sprint).**
   Reproduce 319341's mechanism independently on canonical fresh mints, first as
   a GPU/NumPy accuracy-and-cost oracle and then in a separate flopscope candidate.
   Keep each ingredient ablated: antithetic Sobol, pilot reclassification, cold
   prefix + row buckets, and Strassen. The packed multiply must be an arithmetic
   reorganization of the same sampled forward pass; never approximate a sampled
   nonzero as zero. Allocate fresh D11/D12 blocks before measuring. Do not touch
   `estimator.py` during exploration.
2. **Moonshot lane continues, but the current adapter is an accuracy oracle, not
   yet a shippable low-rank algorithm.** `lowrank_cumulant_flow.py` projects K3/K4
   factors back into full-width objects; that tests whether the subspace carries
   useful signal but does not itself realize r^3 online cost. Before D08/G1, add
   the missing separate calibrated-K2 control path and prove layer conventions.
   If G1 passes, require a compressed representation plus honest flopscope cost
   model before spending D09/G2. No implied symmetric-tensor-kernel discounts.
3. **Write-ups become deadline-specific, not a reason to stop research.** Produce
   a sanitized Phase-1 PDF by Aug 17, tied to one graded artifact and sent privately;
   Appendix F stays out. Continue the Phase-2 paper as a living lab notebook, with
   one bounded writing block per week until the Sep 5 full-draft target.
4. **Stewardship stays strict.** v2.4.1 remains the fallback. Every candidate still
   passes provenance, 30/30 invariants, contract validation, read-only public
   confirmation, GPU parity, and exactly one vault touch only after it is frozen.

### Sprint-4 gates

1. **S0 — semantic parity.** On identical points, packed and dense propagation
   agree to the documented floating-point tolerance; any MSE change must come from
   the separately ablated point-set/pilot choice, not silently dropped arithmetic.
2. **S1 — fresh-mint decision.** On a new registered seed block with paired shifts,
   `V*c` improves by **at least 1.20x** over v2.4.1, with the paired uncertainty
   reported. Anything smaller is killed as complexity risk.
3. **S2 — real-stack confirmation.** The separate candidate passes `whest validate`,
   a full 100-MLP subprocess mini run, determinism, zero failures, and stays below
   the Phase-2 compute/wall envelope. The public mini is confirmation only and
   chooses no threshold.
4. **S3 — promotion.** Only after S0–S2 does the candidate replace `estimator.py`
   and spend one vault touch. Target: at least the published 1.6e-7 class on the
   current shape. Passing this gate does not win by itself; it creates the cheaper
   sampling carrier that a successful mechanistic correction can beat or augment.

### Phase-1 artifact decision, reopened only by new organizer evidence

The earlier no-submit decision assumed there was no selection mechanism. Thread
18125 now says teams will receive nomination instructions and may explicitly choose
two submissions. Therefore, **if Samuel approves before the live round closes**,
grade v2.4.1 in Phase 1 so the reproducible artifact can anchor the Phase-1 PDF,
then nominate `{v2.4.1, v1.1}` when the email arrives. If approval or the nomination
path is absent, keep the existing no-submit decision. Claude Code must stop before
any submission command and ask Samuel.

## REV 8 (2026-08-06) — the truth harness: measurement becomes the deliverable

Rev 7 said the write-up is primary and the estimator is a designation candidate, not
a work in progress. Rev 8 acts on that by fixing the *instrument*, because we found
it was broken in a way that had gone unnoticed for a month.

**The finding.** Our offline networks were not the grader's networks. Same
distribution — weight std within 3e-4 of He — but a different realisation, because
`whestbench.sample_mlp` draws in float64 and casts once while we drew natively in
float32. Worse, the evaluator expands one root seed into **three independent
substreams** (weights / ground-truth draws / the estimator's seed) and we used one
seed for all three, so our Sobol digital shift was a deterministic function of the
network it was scoring. `local_sim/mint_provenance.py` now reproduces the protocol
**bit-identically, 100/100** across five shapes × twenty seeds, and every mint site
(CPU and GPU) routes through it. No past *statistical* conclusion changes; per-seed
reproducibility against the evaluator never existed until today.

**The protocol that follows** (`docs/VALIDATION.md` is now canon for measurement):

1. **DECIDE** on fresh mints, a new seed block per experiment, never reused
   (`local_sim/seed_blocks.py`).
2. **CONFIRM** on the public split — **read-only. No constant is ever fitted there
   again.** A split that has steered a month of decisions has an optimum that is
   partly a memory of our own search.
3. **VAULT** — 500 sealed networks, GT at 2²⁵ samples (noise ~1.6e-9), hashes
   committed, **touch-counted**. Three permitted touches: baseline, one per
   estimator change, designation day.

**Priorities, updated.** Rev 7's four workstreams stand, with the harness inserted
above them because everything else reports through it:

0. **The harness** (this rev). Done: provenance proof, three-tier protocol, sealed
   vault, invariant suite 24/24 → **30/30** (layer-1 closed form exact to 1.8e-16,
   permutation equivariance, budget monotonicity, determinism ×3, seed hygiene),
   constants audit, public–private self-check.
   The audit kept τ=-2.5, rank 32, and the v0 calibration values. It withdrew the
   old 3.8e-9 pruned-value bound (canonical-mint value 1.043e-8, while the
   independent end-to-end τ optimum remains unchanged). A 65,536-point arm won
   1.0408× in the FLOP-only mint model but only ~1.007× under real local residual
   pricing, so the shipped 16,384-point operating point stays frozen and the
   larger arm is only a Phase-2 telemetry candidate.
1. **Write-up — primary.** Now has an abstract, an introduction, and §7 as a
   validation section in its own right. The measurement findings are contribution
   material, not housekeeping: "same distribution is not a reproducibility claim"
   and "a fixed evaluation set with per-item feedback is not a held-out test set"
   both generalise well beyond this task.
2. **Governance / stewardship / opportunistic** — unchanged from rev 7.
3. **Moonshot lane** (`docs/MOONSHOT.md`), newly opened and strictly fenced:
   low-rank cumulant flow with **pre-registered** gates G1/G2/G3, ≤2 GPU-days per
   week, kill-by-default Sep 1, never touches `estimator.py`. Both outcomes ship.

**Theory (§7b) moved from stub to partly verified.** P1 — the parameter-free ReLU
correlation map — **passes** at max absolute deviation 0.0133 with no fitted
constant. P3 shows the fluctuation collapse is toward the **mean direction
specifically**: |⟨v₁, μ̂⟩| runs 0.079 → **0.994** over 32 layers while λ₁'s energy
share reaches 0.651. That makes §2's common mode and §7b's collapse **one
phenomenon**, which is the strongest structural claim in the write-up. P2 awaits
the theory draft.

**Phase-1 close.** No further Phase-1 submission (decided 2026-08-06): there is no
participant-side selection mechanism, so 18118's top-2 default governs and picks
{v2.4, v1.1}; submitting v2.4.1 would displace the only rotation-free hedge.
v2.4.1 debuts in round 1429.

## REV 7 (2026-08-06) — endgame doctrine: write-up primary, governance, stewardship, opportunistic

*Repo docs are canon; strategy verdicts arrive in prompts and get folded in here.
Rev 6 below is the Claude Code rev — the strategy session's own rev 6 and §2h never
synced, so both were merged into this file rather than either overwriting the
other.*

**Score work is closed** (rev 6) and stays closed unless *external* evidence
reopens it. The four workstreams from here, in priority order:

1. **Write-up — primary.** `docs/writeup-draft.md`, full draft by **Sep 5**, due
   Sep 26. Now has the cost-ceiling table and DP result as §4b's centrepiece, §8's
   R-bootstrap and metric-tail decomposition, three rendered figures in
   `docs/figures/`, and a stub at §7b for the NNGP correlation-collapse theory with
   the measured numbers it has to explain. Appendix F stays private.
2. **Governance.** The organizer email is final and waits on Samuel. Note that two
   of its five paragraphs were *retired by evidence* during the Aug-6 crawl — the
   organizers ruled on residual billing in 18122/18132 — which is the model for how
   this workstream should behave: crawl, then ask only what is actually open.
3. **Stewardship.** The estimator is a designation candidate, not a work in
   progress. v2.4.1 hardened the one real defect (an irreproducible rotation).
   Remaining stewardship: keep `edge_cases.py` at 24/24, keep the fresh-MLP
   distribution current if the shape changes, and re-validate pricing on any
   flopscope/whestbench release (done for 0.10.0 — all our graded numbers are
   already post-change).
4. **Opportunistic.** Phase-2 opening (event-triggered), the Phase-1 write-up
   harvest through Aug 17, and anything the organizers answer.

**Ranking-noise nuance, corrected.** The fresh-MLP score distribution is heavy
(max/median 12.9×, p90/median 1.50× at R=10), but the re-run suite is **common
across teams**, so hardness is a common-mode term that inflates or deflates
everyone together and largely cancels in the *ranking*. What survives is the
interaction between draw and method. Ranking jitter between two similarly-placed
teams at R=10 is therefore **±10–20%, not ±50%**. Both numbers go in the write-up;
conflating them overstates the case.

## REV 6 (2026-08-05, late) — the honest sampling frontier is declared at ~2.7e-7

**Sprint 3 (structure-aware FLOP allocation) came back at 1.0246× against a 1.20×
gate, and the ceiling is arithmetic.** Details in LOG. The short form: for a
confidently-on neuron ReLU is the identity, so at the scored layer it needs no
per-sample value and at a feed-forward layer it is linear and composes — both
verified, the second to within 0.07% MSE over eight layers. But under `(2K−1)MN`
billing the multiply is the cost, not the nonlinearity, so the reduction is worth
the share of the cost model it touches: **the scored layer is 3.67% of the
per-sample budget, so freeing it entirely is 1.038×.** Clearing 1.20× would need
six of the last layers to cost nothing.

**Therefore: we stop optimising score.** Every remaining known lever is small and
additive — a hand-rolled normal quantile in place of `stats.norm.ppf` (~1.4%), the
Sprint-3 scheme (~2.5%) — perhaps 1.04× in total, for new code paths on an
estimator we are scored on **once**, on MLPs we have never seen. That is a bad
trade. v2.4 stands as the designated submission unless Phase 2 changes shape.

**What the effort goes to instead, in priority order:**

1. **Robustness on fresh MLPs** (`local_sim/robustness.py`, `edge_cases.py`). The
   private re-run scores one designated submission on unseen draws, so the
   distribution's tail is the risk, not its mean. Edge-case suite on the shipped
   file: **24/24 pass**, including degenerate pruning, a identically-zero layer,
   1e4 dynamic range, seed 2⁶³−1, widths 64/128, depths 2–64, budget 0, and input
   immutability.
2. **The write-up** (`docs/writeup-draft.md`), due Sep 26. This is now where the
   remaining expected value is. The negative results carry mechanisms *and* oracle
   bounds, and the on-neuron reduction gives it a theorem with a price tag.
3. **Phase 2 config verification** (Aug 8). If the shape changes, re-measure before
   optimising: α-histogram → error-vs-N → τ → stack. v1.1 stays the shape-generic
   fallback.
4. **Governance**: the organizer email is drafted and waits on Samuel. Nothing on
   the forum.

**Scoring policy, restated because it now drives everything**: judge on the local
100-MLP mini split and on ≥1000 fresh minted MLPs. Public-board rank is not a
target — §2g establishes the board is a fixed-suite reconstruction game — and our
own graded runs are *paired re-measurements* on 50 fixed MLPs, useful only against
our own prior submissions.

## REV 5 (2026-08-05, night) — both analytic revivals are bounded out; the board is signal-poor; optimise for FRESH MLPs

Rev 4 retired variance reduction as a growth area. Rev 5 closes the two tracks that
were meant to reopen it, and changes what we optimise *for*.

**Track T (shrinkage-moment tail) — dead, 0.9998× against a 1.10× gate.** Three
independent bounds, any one of which is sufficient (LOG 2026-08-05 night):
1. Only `diag(WᵀΣW)` enters the final-layer means, and that equals
   `mean(z²) − mean(z)²` exactly — so there is no covariance matrix to estimate,
   no Marchenko-Pastur regime, and v̂ is already accurate to **1.13% rms**.
   Replacing it with the *exact* variance — the oracle for every denoising scheme,
   linear or not — still gives only **0.611×** of v2.4.
2. The analytic tail's residual is **41% irreducible model bias**; with exact
   moments it beats sampling **1.474×** (independently reproducing 18097), but the
   exact moments are the very thing sampling estimates, and at N=16384 the
   mean-estimation noise alone is 57.9% of the error.
3. Blending is capped by ρ = **0.77** between the two errors. In the α ≥ 3 band —
   **76.5% of the MSE** — ρ = **1.000**, because there Φ=1 and the "analytic tail"
   *is* the sample mean. Oracle per-band blending: 1.0003×.

**Track P (rank-k deviation propagation) — bounded out at 1.093× against 1.25×.**
Stage 1's energy gate passes handsomely (top-64 holds 96.2% at layer 16, and the
closure's own covariance eigenvectors find the subspace for free), but propagating
in a rank-k subspace *is* a rank-k factorisation of each weight matrix, and under
shape-based billing two factors only beat a dense 256×256 matmul for **k < 128** —
which is exactly the rank at which truncation stops costing accuracy. Even pricing
an oracle-accurate Jensen correction at **zero**, the best cell in the whole grid is
1.093×. No prototype: the stage-1 data already bounds one.

**The board is signal-poor, so stop steering by it.** Measured Aug 5: the public
score is the mean over **50 fixed MLPs whose exact per-MLP MSEs are returned with
every submission**, the suite is identical across all four of our submissions, and
257 submissions suffice to solve for those references outright (RESEARCH_NOTES §2g).
The Phase-1 leader has 917. Consequences for planning:

* **Optimise expected score on FRESH MLPs.** The final ranking is a private re-run
  on MLPs nobody has seen; that is also what our local 100-MLP split proxies.
  Rank on the public board is now a weak target and a misleading one.
* **Our own graded runs are paired re-measurements**, not fresh samples. Useful for
  A/B against our own prior submission, useless as evidence about the population.
* **We do not touch that channel** (§2g rule 1). It is not the wall-time grey area.
* The email to organizers states the observation neutrally and asks whether it is
  intended. Nothing goes on the forum.

**What is left that could still move the score**, in order of expected value:
1. **Cost (c).** Still the only channel with headroom, and the one the field ignores.
   v2.4 spends 2.86e6 FLOPs/sample; the irreducible dense-matmul floor at the
   measured 80% alive fraction is ~2.8e6, so *this* design is nearly optimal — the
   next win has to change the design, not tune it. Pruning τ is at a sharp optimum,
   the op-count work is banked, and residual wall time is 1.83% on the grader.
2. **The write-up** (`docs/writeup-draft.md`). Our negative results now come with
   mechanisms *and* oracle bounds, which is the strongest form of the genre. Due
   Sep 26. This is where the remaining expected value actually is.
3. **Phase 2 config verification** (Aug 8) and a shape-generic fallback. If the shape
   changes, re-measure before optimising: α-histogram → error-vs-N → τ → stack.

**Standing measurement policy is unchanged and was vindicated twice more**: the
Track T and Track P verdicts both came from the full 100-MLP split with oracle arms
that bound the entire family, not from a promising subset.

## REV 4 (2026-08-05, evening) — variance is at its floor; we work cost, governance, and the write-up

Rev 3 retired the analytic family. Rev 4 retires **variance reduction itself** as a growth area, on evidence:

- **The common mode is irreducible variance, not correctable bias.** Five independent attempts to predict the per-MLP scale fluctuation g — closure reference at the final layer (R² 0.094) and mid-depth (0.155), point-set radial imbalance (−0.010), the exact closed-form layer 1 (−0.037), and a learned regressor over 3000 MLPs × 8 draws (out-of-fold corr 0.012). The official reason is **projection geometry**: g = ⟨ŝ−t, t⟩/⟨t,t⟩ projects onto the unknown truth direction, and every proxy for that direction carries more error along it than g itself. This predicted R² ≈ 0.09 before measurement. **No further prediction attempts.**
- **Point sets are bounded by theory.** He & Wang 2015: the RQMC rate bonus for discontinuous integrands is 1/(4d−2) = 1/1022 at d=256. Measured α = 1.00. Every variant lands within ±15% of plain Sobol, including direct gradient optimisation (G2 plateaued at 1.06×).

**Operating frame: adjusted ≈ V·c/B, V is at floor, so we work c.** Three tracks:

1. **Cost (c).** Dead-neuron pruning was the largest single win of the month (1.47×). Op-count engineering is the untouched channel — residual wall time was 10.8% of billed compute at 1794 ops/MLP.
2. **Governance.** The 18099 wall-time bypass has no organizer reply. A neutral, technical clarification email is drafted at `docs/email-draft-organizers.md` for Samuel to send from his own address. Never posted to the forum.
3. **The write-up** (`docs/writeup-draft.md`, private). Our negative results have mechanisms attached, which is what the contribution prize rewards. Due Sep 26.

## REV 3 (2026-08-05) — the analytic family is retired

Everything on the analytic side has now been measured against 1e9-sample truth and killed: B1 fusion (net ~0), B2 analytic tail (−67% to −214% at every k), B3 learned covprop residual (held-out R² = 0.012 on 50k MLPs). Covariance propagation is 18× worse than sampling at layer 32 and its residual is not a learnable bias. **We stop trying to make the analytic estimator contribute.**

What replaces it follows directly from the scoring identity `adjusted = V·c/B` (RESEARCH_NOTES §3d): the only two levers are per-sample **variance** and per-sample **cost**.

### Sprint 1 — cheap wins stacked on v1.1 (before Phase 1 closes Aug 7 23:59 UTC)
1. **Dead-neuron pruning** (cost lever). Predict dead neurons from covprop's α, slice both sides of each matmul to smaller dense shapes, compensate with the exact constant offset, report pruned neurons with their closed-form rectified-Gaussian mean.
2. **Rotation + radius** (variance/rate lever). Fold an alignment basis Q into W1 (zero marginal per-sample cost); put an exactly stratified chi₂₅₆ radius on one Sobol coordinate.
3. **Offline base-set choice** (free). Compare the shipped Sobol set against orthogonal-block + antithetic variants and Owen scrambling.

*Gate: combined ≥1.2× over v1.1 (mini adjusted ≤3.4e-7), n_failed=0, determinism → spend ONE reserve on Phase 1 before close. Otherwise submit nothing.*

### Track G — GPU, through ~Aug 12
- **G1 amortized common-mode regressor.** 48% of the error is one per-MLP scalar worth 1.9× if removed exactly. Train offline over (MLP × K sample-set draws) to predict it from cheap online observables; cross-fitted and James-Stein shrunk by out-of-fold R² so it cannot add net bias. Ship as inline float literals. *Gate: ≥1.3× held-out, confirmed on real mini.*
- **G2 task-optimized point set.** Optimise the base point matrix directly against minted-MLP final-layer error; regularise with exact moment matching and drift penalties. *Gate: ≥1.3× held-out + mini.* G1 and G2 overlap on the mode — report the **combined** gain, never the product of solo gains.

**Standing measurement policy**: no operating-point or variant decision on fewer than the full 100-MLP mini split. Three separate "wins" (LMS scrambling, C/B=0.5, a +33% rotation) evaporated when re-measured.

## Phase C — learned residual corrector — RETIRED (see REV 3; B3 measured R² = 0.012)

Mint 10⁵–10⁶ He MLPs on the RTX GPU with high-precision MC labels (torch, float32, batched — each 256×32 MLP takes ~seconds at 1e6+ samples). Train a *small* per-neuron corrector (permutation-equivariant features: analytic μ̂/σ̂, firing probability, row norms, Gram-overlap stats, depth index) to predict the hybrid's residual. Distill to pure-`fnp` inference (a few matmuls, ~1e7 FLOPs) + shipped `.npz` weights (≤50 MiB).

*Target: 1.5–3× on top of Phase B → ~1–2e-8. Risk: publicly-reported failures of naive correctors near layer 32 (residual ≈ GT noise). Mitigation: train to predict the *bias* component only (averaged over input noise), validated on held-out self-generated MLPs against 1e9-sample local truth for a handful of networks.*

## Phase D — the mechanistic bet (parallel, best-effort; the $20k track)

Depth-robust cumulant propagation: start from ARC's public `mlp_cumulant_propagation` code; add (a) low-rank K=3/K=4 cumulant tracking in the top-k covariance eigendirections, (b) per-layer renormalization fitted offline (the generalization of our per-layer calibration — same trick, deeper object). This is ARC's own stated wishlist (townhall). Even a partial result + clean analysis = strong contribution-prize entry. Phase 1's write-up deadline showed the format: one PDF + one submission ID; Phase 2 deadline **Sep 26**.

## Operating cadence

- **Cloud workspace** = development + local_sim validation (exact submission file, self-generated truth). **Windows PC** = official `whest run` + submissions + GPU offline training. Repo lives in `Documents\arc-whest`; sync via this Claude session.
- Every experiment gets a row in `experiments/LOG.md` (idea → local MSE → graded score → keep/kill).
- Submit *at least* daily once the pipeline works — the grader is free information (real FLOP pricing, wall-time headroom, private-split variance). Keep the best-scoring submission clearly tagged; Phase 2 requires designating one.
- Weekly: re-crawl forum/Discord for rule changes (esp. flopscope patches), re-run the leaderboard snapshot, update RESEARCH_NOTES.
- Writing: keep a running `docs/writeup-draft.md` from week 2 — the contribution prize rewards analysis quality, and it's brutal to reconstruct in the last week.

## Risk register

| Risk | Odds | Mitigation |
|---|---|---|
| Phase 2 changes architecture/budget | Medium | Estimator is shape-generic; calibration keyed by (width, depth) with safe fallback; refit within a day of any announcement (GPU makes refits cheap) |
| Wall-time loophole patched → leaderboard reshuffles | Medium-high | We never depend on it; pure upside for us |
| Loophole NOT patched → exploiters keep inflated scores | Medium | Final ranking is a private re-run; organizers signaled enforcement ("must not exploit"); contribution prize unaffected; if it stands, honest ceiling is still top-tier |
| flopscope API drift breaks submission (fnp surface mismatches) | Low-medium | `whest validate` catches locally before submitting; keep estimator's API footprint minimal and documented |
| Sobol/QMC implementation cost under flopscope pricing | Medium | Ship precomputed point sets as `.npz` (loading is ~free); measure real pricing in run reports before committing |
| GT noise floor limits gains below ~1e-8 | Low | That's *good* for us — compresses the top; win on consistency across the private set |
| Time (Samuel learning + building from zero) | — | Claude drives implementation; Samuel runs pipeline + learns via BRIEF curriculum; scope gates at each phase |

## Success criteria checkpoints

- **Aug 6**: v0 submitted, graded, run report parsed. Phase 2 config confirmed.
- **Aug 14**: Phase A live, ≤4e-7. | **Aug 28**: Phase B live, ≤1e-7. | **Sep 11**: Phase C live, ≤3e-8, freeze-candidate chosen. | **Sep 16–19**: hardening only — determinism re-runs, failure-mode audit (the private re-run punishes fragility, not slowness). | **Sep 26**: contribution write-up submitted.
