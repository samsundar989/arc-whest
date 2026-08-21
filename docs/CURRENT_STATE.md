# CURRENT_STATE.md — read this first

*Snapshot, not narrative. Last updated **2026-08-21** — the session that repaired
the decision layer (CP0.1-0.3), built and proved a 66,048-point exact design
(CP1.1), and then found the Phase-2 shape had moved to **1024 x 16**, which
obsoletes that design. Four slices, four independent Skeptic reviews. Update this
file when a lane opens or closes, a number is graded, or the phase changes — not
on every experiment. The narrative record is `experiments/LOG.md`.*

*Documentation note, 2026-08-20:* the external repository/field audit and its
non-binding frontier execution blueprint are now durable in
`docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`; the copy-ready Builder handoff is
`docs/CLAUDE_CODE_FRONTIER_HANDOFF.md`. They do not change the plan of record,
registered gates, or closed-lane statuses. Reconcile them through `next-slice`
before implementation.

---

## The clock

*Last polled **2026-08-21**. Round 1429 opens **2026-08-21T23:59Z**, leaderboard 1480 still has 0 entries — and the HF tags now read `['v2-phase2', 'v1-phase1', 'v1-warmup']`. **The Phase-2 dataset is published and the shape has moved: 1024 x 16.** See the boxed section below.*

| event | when | status |
|---|---|---|
| Round 1428 "Phase 1" | closed **2026-08-10 23:59 UTC** | **CLOSED.** Our last submission is 327743 (v2.4.1) |
| Phase 1 submission selection | closed **2026-08-11 23:59 UTC** | **PASSED.** We nominated nothing, so the default applied: our **two** highest-ranked public entries |
| Phase 1 write-up | closed **2026-08-17 23:59 UTC** | **PASSED, MISSED.** No extension was issued. Write-ups posted after it (18184, 18185) drew no organiser objection, but that is not permission |
| Round 1429 "Phase 2" | opens **2026-08-21 23:59 UTC** | **IMMINENT.** Re-polled 2026-08-21: leaderboard 1480 exists with **0 entries**, but the HF dataset now carries a **`v2-phase2`** tag baked 2026-08-13 — **width 1024, depth 16**. Slipped **four** times: Aug 8 → Aug 18 → Aug 19 → Aug 20 → **Aug 21**. Organiser 18186 (2026-08-18): *"still ironing out some details … the deadline for Phase 2 submissions will also be shifted"* |
| Phase 2 close | nominally 2026-09-19 23:59 UTC | **WILL MOVE** — organiser says so explicitly |
| Registration / team freeze | **2026-09-05 23:59 UTC** | 17 days out |
| Results | 2026-10-01 | private re-run of **UP TO TWO** designated submissions on fresh MLPs (forum 18143; the Rules tab still says one and the contradiction is unresolved) |

**Phase-2 opening is event-triggered, never date-triggered, and the date has now
moved four times.** Poll with `uv run python local_sim/phase2_checklist.py`.

**HALF the configuration is now published and half is not.** The dataset gives
**width 1024, depth 16**, GT 1e9, mini 100 / full 1000. It does **not** give
**`B`, λ, the multiplier floor, or whether biases are absent** — those live in the
round config and surface in a graded report's `config_snapshot`. The challenge
page's timeline and its `max(0.5, ·)` worked example remain **stale Phase-1 copy —
do not read constants off it**.

**The shape DID move, so the first action is no longer conditional:** re-fit
`_PRUNE_TAU` and `_BASIS_RANK` at 1024×16 on a DESIGN block *before* gating
anything. Both are frozen at 256×32 and neither transfers.

### ⚠️ THE PHASE-2 SHAPE IS **1024 x 16** — CONFIRMED 2026-08-21 FROM THE DATASET

The HF dataset gained a **`v2-phase2`** tag. `metadata.json` gives **width 1024,
depth 16**, GT 1e9, mini 100 / full 1000, baked with **whestbench 0.16.0 /
flopscope 0.12.0**. Round 1429 opens **2026-08-21T23:59Z**.

**This falsifies the working assumption below**, on its own first-listed falsifier
("a width that is not 256"). Kept for the record rather than deleted.

**Dead on arrival at the new shape:** the CP1.1 Kerdock/MUB-129 carrier (a
width-256 object -- `256 = 4^4`, `GR(4,7)`, `d^2 = 2^16`); the dual-BCH array and
its 131,072 linear floor; `_PRUNE_TAU = -2.5`; `_BASIS_RANK = 32`. The Kerdock
*mathematics* generalises to `GR(4,9)` at 1,050,624 points; the *object* does not.

**Reopened, on its own pre-registered condition:** the mechanistic / cumulant lane.
`L/n` fell from 1/8 to **16/1024 = 1/64**, and `L^K < n` now holds at K = 2
(`256 < 1024`). Open question #1 said a wider or shallower Phase 2 would reopen it
"wholesale". It did.

**Still unknown: `B`, lambda, the multiplier floor, and whether biases are absent.**
None is in the dataset. **Every `C/B` in this document is a Phase-1 number.** Naive
arithmetic: per-sample propagation is ~8x Phase 1 (16x the layer cost, half the
depth), so if `B` did not grow proportionally the affordable sample count collapses.

Full record: `experiments/LOG.md`, 2026-08-21.

### SAMUEL'S WORKING ASSUMPTION, 2026-08-20 — FALSIFIED 2026-08-21, kept for the record

Nothing about Phase 2 has been released. Rather than block, Samuel's instruction is
to **assume Phase 2 is more or less Phase 1 with different MLPs, possibly a little
harder**, and to build against that.

**This is a stated assumption, not evidence, and it is labelled as such wherever it
is load-bearing.** Its consequences:

- Build and validate at **256 x 32, no biases, He init, `x ~ N(0, I)`**, budget
  `B = 2.72e11`, lambda = 1e11, multiplier floor 0.1. These are the Phase-1
  constants from organiser thread 18125 and the installed whestbench 0.14.0.
- Keep everything **shape-generic** anyway (hard rule 7). "A little harder" most
  plausibly means more depth or a tighter budget; both change `C/B` and neither
  changes the score identity.
- **Do not spend D17 or D19** — the two clean single-use decision blocks — until the
  real shape is published. A block spent on a shape that then moves is wasted, and
  that is the whole reason both were reserved.
- **Re-poll before acting on any deadline** (`local_sim/phase2_checklist.py`). The
  open date has now slipped four times, most recently to 2026-08-21T23:59Z.

**What would falsify the assumption**, in rough order of damage: biases (kills the
exact radial reduction and the layer-1 closed form), a non-Gaussian input
distribution (a new angular/radial derivation), a width that is not 256 (rebuilds
every design-carrier construction, and 2^8 is exactly what makes the MUB/BCH
arithmetic work), a materially different `L/n` ratio (reopens the mechanistic
cumulant lane wholesale), or a changed flopscope price table (flopscope >= 0.11 is
required by whestbench 0.15.0 and its pricing is unverified).

## Where we stand

| instrument | tier | value | date |
|---|---|---|---|
| **graded, submission 327743 (v2.4.1 — THE SHIPPED FILE)** | graded | **2.6314e-7** adjusted · 1.3323e-6 MSE · mult 0.19896 · residual 10.022 ms/MLP · **0 of 100 failed** | 2026-08-10 |
| graded, submission 324420 (v2.4) | graded | 2.7194e-7 adjusted · 1.3748e-6 MSE · mult 0.1990 · residual 9.845 ms/MLP | 2026-08-05 |
| graded, 324094 (v1.1) | graded | 3.893e-7 | 2026-08-05 |
| vault, 500 sealed MLPs, v2.4.1 | VAULT | median **2.4651e-7** · mean 3.2669e-7 · p99 2.0535e-6 · max 3.0469e-6 · 0 non-finite | touch #1, 2026-08-06 |
| public mini (100), v2.4.1 | CONFIRM | 3.172e-7 adjusted · 1.4895e-6 MSE · mult 0.216 · 0 failed | 2026-08-06 |
| invariant suite on the shipped file | — | **30/30** | verified 2026-08-19 |
| shape parity (v2.6.1 fallback vs v2.4.1) | — | **27/27**, dense paths bit-identical | 2026-08-19 |
| canonical scorer vs installed whestbench | — | **71/71**, bitwise | 2026-08-19 |
| seed-ledger enforcement | — | **44/44** negative tests (was 40/40; three snapshot assertions converted to invariants 2026-08-20) | 2026-08-20 |
| **the preflight gate itself** | — | **12 checks**, 10 by default and 12 under `-Full`. Was 6 | 2026-08-20 |
| grader-lifecycle evaluator | — | **37/37**, all six failure paths provoked | 2026-08-20 |
| carrier design strength | — | **48/48** (was 27/27) | 2026-08-21 |
| **Kerdock + axial, an EXACT spherical 4-design** | design theory | **66,048 points on 256 dims**, degrees 1–5 at machine precision, strength 5 proven by exact Delsarte transform. `C/B = 0.6934` **at the Phase-1 shape**. **OBSOLETED by the 1024×16 shape the same day** | 2026-08-21 |
| **the degree-4 design carrier** | DECIDE (D18, 48 nets) | **1.4775× [+14.4%, +45.9%]** at 131,072 points, where `C/B = 1.3509` is **over budget**. At **66,048** points the cost line fitted on our own arms gives **`C/B = 0.6932` counted — under the cliff** (CP0.3) | 2026-08-20 |
| the binding promotion gate | — | **arithmetic**, studentised-t / basic bootstrap, executable as `competition_score.decide()`. Re-scoring the whole record flipped **nothing** | 2026-08-20 |
| board position | — | raw #90/200; **#82 of 192 plausible-honest** on 5 entries | 2026-08-06 10:03Z |

**Shipped file, corrected 2026-08-19**: `estimator.py` == **`estimator_v2_6_1.py`**
(byte-identical modulo line endings — `scripts/preflight.ps1` step 4 asserts this).
The **graded** file is `estimator_v2_4_1.py` (327743, 2026-08-10, Phase 1): 0
failures, safe to designate. **These are not the same file** — v2.6.1 is built,
green and deliberately un-gated, and the line above said "v2.4.1" until an
independent audit caught it. Vault touch counter: **1**.

**v2.4.1 vs v2.4 is a DRAW, not a 3.2% improvement.** The headline adjusted score
fell 2.7194e-7 → 2.6314e-7, but both were scored on the *same* 100 networks
(`mlp_name` lists identical, `sampling_mse` identical to 17 digits), so the paired
statistic applies: geometric mean of per-MLP ratios **0.9665×, 95% CI [0.7969,
1.1721]**, better on **22 of 50** (exact p = 0.839). The aggregate delta is a small
residue of near-cancelling sums (+1.91e-5 of gains against −1.70e-5 of regressions);
three networks each individually flip its sign. Counted FLOPs agree to 0.02%, and
layers 0–30 are exactly float-equal on **all 50** networks — only the scored final
layer differs. **Quote v2.4.1 as *equivalent to* v2.4, never as better.**

*Precisely: "no **resolvable** effect." At n = 50 with sd(log ratio) = 0.696 this
design can only exclude |effect| > ~20%. The better-powered answer already existed
and agrees — 0.9959× over 32 draws/MLP (`docs/GOTCHAS.md`). The grader run confirmed
it; it did not discover it.* Full analysis and the independent audit that corrected
three claims in this section: `experiments/LOG.md` 2026-08-10.

**Field anchors — corrected 2026-08-19.** Best *published and explained* honest
score is **9.10e-8** (Team Puffi, analytic+QMC hybrid); the best published raw MSE
is 1.546e-7. GT noise floor of the metric ~2e-10. **We are ~2.6× behind the best
published honest method, and behind roughly twelve of them.**

⚠️ **"Our declared honest sampling frontier ~2.7e-7" is WITHDRAWN** (it described
*our own method's* frontier, not the field's), and so is the framing that the
cost half of `V·c` is "empty" while variance is "crowded" — the 2026-08-19 carrier
audit measured our variance carrier at ~85–95% of the achievable ceiling for
input-space methods, and found the remaining headroom on the **cost** side.
Full table: `docs/BENCHMARK.md`.

---

## Shipped file — v2.6.1 BUILT, NOT PROMOTED (2026-08-19)

`estimator.py` is **v2.6.1** (Strassen-Winograd carrier + the P0.3 odd-width
repair), archive `estimator_v2_6_1.py`, preflight **GREEN 11/11 under `-Full`**
(2026-08-20; the gate was 6 steps when this line was written and is now 11 —
see `docs/VALIDATION.md` §7), invariants **30/30**, shape parity **27/27**,
vault touches still **1**. **It has not been gated, not been submitted, and no
decision block has been spent.**

**v2.6 had a network-destroying bug on every odd width and it is fixed.** The
Strassen trim ran *before* the eligibility test, and the trim itself
(`keep = a > alpha[alive].min()`) sheds every neuron tied at the minimum — and at
layer 0 **every alpha is exactly 0.0**, because `x ~ N(0, I)` makes the
pre-activation mean zero. So on any odd width v2.6 deleted the entire first
layer's sampled path: 7 of 17 shapes wrong, up to **0.58 max relative error** and
**0.33 prediction MSE** against v2.4.1. v2.6.1 decides eligibility from `n` and
`width` first, rebuilds untrimmed if trimming cannot make every dimension even,
and refuses any trim that would shed more than one neuron.

**Nothing recorded is invalidated**: the competition shape is 256x32, layer 0's
alive count there is 256 (even), the trim never fired, and v2.6.1 is bit-identical
to v2.6 at even widths **including its counted FLOP count to the unit**. The
independent Skeptic confirmed `use_str` was True on 100/100 public-mini networks.
Full record: `experiments/LOG.md` 2026-08-19 (night), P0.3.

The graded number of record is still **v2.4.1 = 2.6314e-7** (submission 327743).

**Why it is held.** The deciding quantity is the grader's residual price per
operation, which this document classifies **E1 — directional only**, and the
same byte-identical paired comparison reads:

| | cost ratio | net V.c |
|---|---|---|
| **fitted line on 5 graded submissions (2026-08-19 — QUOTE THIS)** | **1.1004x** | **1.0940x** |
| grader per-op price 7.0 us/op (ratio through origin) | 1.0990x | 1.0926x |
| transfer the local residual ratio (the old conservative choice) | 1.0740x | 1.0677x |
| **this machine, as measured** — never decides | 0.9577x | 0.9522x |

**The first row supersedes the rest.** Projected graded residual for v2.6.1 is
**16.251 ms** against v2.4.1's observed 10.022 ms; counted FLOPs fall 10.5%,
residual rises 62%, and because residual is only 2–3% of `C` on the grader the
counted saving wins. Known to well under 1% on the cost side, against the
17-point bracket below whose **sign flipped**. Remaining caveat: v2.6.1 sits at
2,370 ops and the fit is calibrated on 329–1,352 — a **1.75× extrapolation**,
which is now the load-bearing assumption and is much weaker than the old one.

The sign flips between the local harness and the grader, because local residual
runs **4.37x** the grader's (42.99 ms/MLP vs 9.845) while the op count transfers
exactly (1,417 local vs 1,416 graded). Break-even is 55.5 us/op; the grader is
at 7.0, this machine at 30-51. **Only a graded run can decide it.** MSE is
settled and costs 0.51% (paired 0.9942x [0.9870, 1.0013]), all of it the
even-width trim — Strassen itself contributes nothing (LOG 2026-08-19).

## CP0 — the decision layer, 2026-08-20 (two slices, two independent Skeptics)

*Working against `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` rev 2 as a non-binding
strategist packet. **No seed block drawn, no vault touch, no estimator edit,
nothing submitted.** Vault counter still **1 of 3**.*

**Ten proposed validation blockers were audited against the code with file/line
evidence.** Two were already partially fixed (2, 8); eight were open. Two are now
repaired:

| slice | class | outcome |
|---|---|---|
| **CP0.1** the gate | IMPORTANT | `scripts/preflight.ps1` **6 → 12 checks**. 165 previously ungated checks are now gated. Skeptic **PROVISIONAL**, found 3 real defects in the repair (plus 1 I introduced fixing them); all fixed |
| **CP0.2** the lifecycle evaluator | IMPORTANT | `local_sim/lifecycle_eval.py`, **37/37**. Skeptic **PROVISIONAL**, and its lead finding invalidated the slice's own licence; 12 findings fixed, self-test 26/26 → 37/37. Terminal state PROMOTE_AND_CONTINUE |

**The gate is now two-tier and the denominator is always 12.** Default runs 10
checks in ~135–160 s and prints `GREEN 10/12, 2 SKIPPED` with the skips named in
red; `-Full` runs all 12 in ~755 s and is **required** before any promotion,
submission or vault touch. **Quote the tier** — "preflight GREEN" now means two
different things. Protocol: `docs/VALIDATION.md` §7.

**Five things every future number depends on**, added to the three from Stage 0:

1. **A setup failure kills the WHOLE submission, not one MLP.** `cli.py:2357` calls
   `runner.start()` outside the try wrapping `evaluate_estimator`. This is the most
   expensive failure mode a carrier candidate has, and no evaluator here modelled
   it until 2026-08-20.
2. **We have never observed the grader's timing or memory limits.** Every graded
   `config_snapshot` carries exactly `n_mlps`, `depth`, `width`, `flop_budget`.
   `setup_timeout_s` = 5.0, `wall_time_limit_s` = 60.0, `predict_timeout_s` = 30.0,
   `memory_limit_mb` = 65,536 are **package defaults**. Only λ = 1e11 is
   corroborated. **`memory_limit_mb` is RLIMIT_AS and already killed v2.5.**
3. **The old evaluators over-charge MLP #0 and only MLP #0** — measured
   **+2.5273e8** counted FLOPs (+0.554%, +155 ops); every other MLP bit-identical,
   because `_BASE_POINTS` is a module global. **The vault baseline and the coverage
   study are NOT invalidated.** The narrowness is an accident of v2.6.1's design,
   not a property of the old evaluator: a carrier held in *instance* state would be
   billed on all 500.
4. **A local evaluator's objective is not a graded objective.** Counted FLOPs
   transfer exactly; residual seconds do not and they enter `C`. λ·R is ~10–13% of
   `C` locally against 2–3% graded.
5. **`mean(target²)` is float32.** A float64 version of the failure-path MSE differs
   by 1.96e-9 relative. Go through `competition_score.zero_prediction_mse`.

**Still open from the CP0 audit, untouched:** blocker 4 (the ledger scan misses a
literal already covered by another record; dynamic block access invisible; two D18
artifact hashes blank), 6 (a vault touch is not transactional — an interrupt
between opening the truth and writing the seal reads the truth and logs nothing),
7 (the residual model fits five per-submission *aggregate* means and cannot score
per MLP), 9 (truth seeds masked to 31 bits), 10 (single truth replica). Blocker 1 —
the arithmetic gate — is **STRATEGIC and waiting on Samuel**; see § *Next actions*.

**`local_sim/vault.py` and `local_sim/coverage_lab.py` were deliberately NOT
rewired** to the new evaluator. Both have recorded results attached and `vault.py`
is guarded; that is its own slice with its own prediction.

---

## Stage 0 — the instrument, repaired (2026-08-19 night)

Five slices, all recorded in `experiments/LOG.md` with contracts written before
the data existed. **Nothing was promoted, no decision block was spent, no vault
touch was taken, nothing was submitted.**

| # | what | state |
|---|---|---|
| **P0.1** | `local_sim/competition_score.py` — the grader's identity in one place, bitwise-equal to whestbench 0.14.0 on the scalar primitives, on the aggregate, and on the real scoring loop including both failure paths. **71/71** | done, Skeptic PROVISIONAL |
| **P0.4** | `local_sim/seed_ledger.py` + `experiments/seed_ledger.jsonl` — consumption recorded, enforced, and statically scanned. **39/39** negative tests. Preflight step **6** | done |
| **coverage** | `local_sim/coverage_lab.py` — which promotion interval actually covers 95%, on 192 fresh networks (D18) | done |
| **P0.3** | `estimator.py` -> **v2.6.1**, the odd-width repair. **23/23** shape parity | done, Skeptic **ACCEPT** |
| **P0.2** | `local_sim/vault.py` — evaluates a named candidate FILE under real flopscope, `--dry-run` deleted, touch bound to a sha256, maximum enforced, atomic seal update | done |

**Three things every future number depends on:**

1. **The competition objective is the ARITHMETIC MEAN of per-MLP scores over the
   FIFTY public MLPs**, and it is now reproducible offline **bitwise** from the
   durable graded reports — `2.719427529595e-07` (324420) and
   `2.631432177242e-07` (327743), exactly. A failure is priced at multiplier
   **1.0**, not the 0.1 floor: ~5x, on top of the zeroed prediction.

2. **The arithmetic mean and the geometric mean of per-MLP ratios select
   OPPOSITE winners on our own graded data.** v2.4 -> v2.4.1 on the same 50 rows:
   arithmetic **1.0334x** (v2.4.1 better), geometric **0.9666x** (v2.4 better),
   22/50 wins, bootstrap CI spanning zero. **RESOLVED 2026-08-20 (CP0.3): Samuel
   adopted the arithmetic statistic.** `docs/VALIDATION.md` §2b-i is now binding
   and the gate is executable as `competition_score.decide()`. Re-scoring every
   recorded verdict under both estimands flipped **nothing**.

3. **Use the studentised t interval or the basic bootstrap. Do NOT gate on the
   percentile bootstrap or on BCa at n ~ 50.** Measured false-promotion rate
   against a true zero (nominal 5%): t **4.1-5.4%**, basic **4.0-5.2%**,
   percentile 6.8-8.2%, **BCa 9.1-11.5% — roughly double**. The reason is that
   common random numbers make the *paired difference* nearly symmetric (skew
   +0.21, -0.55, -0.46) even though each arm's scores are strongly right-skewed,
   so BCa's skew correction is pure added variance. BCa only starts winning past
   skew ~6, which no real arm reaches.

**The v2.6 gate, re-derived under the right estimand** (read-only, from the two
durable CONFIRM artifacts; `local_sim/reconstruct_v26_gate.py`). The scorer
reproduces both artifacts' per-MLP and aggregate scores at **0.000e+00** relative
deviation:

| residual model | arithmetic gain | 95% CI (relative) | geometric gain |
|---|---|---|---|
| local, as measured — never decides | 0.9529x | [-5.77%, -4.11%] | 0.9521x |
| per-op at the graded 6.95 us/op | **1.0961x** | [+8.01%, +9.52%] | 1.0927x |
| transfer (conservative) | **1.0695x** | [+5.75%, +7.25%] | 1.0666x |
| counted FLOPs only | 1.1147x | [+9.54%, +11.03%] | 1.1109x |

**The estimand correction does not change this decision** — arithmetic and
geometric agree in sign in all four models, and the arithmetic reading is slightly
*more* favourable to v2.6. The sign still flips on the grader's residual price,
which is **E1**, and only a graded run settles it.

**Seed blocks after the ledger repair:** D16 **DESIGN, part-spent** (270 of 600
drawn by five probes; 330 free in four gaps). D17 **LOCKED SINGLE-USE**, verified
never drawn. D18 DESIGN, **192 drawn** by the coverage study, 408 free. D19
**SINGLE-USE, unspent**, held for the v2.6 promotion gate so a Phase-2 shape
change cannot burn the only clean decision block.

---

## Closed lanes — do not reopen without new *external* evidence

Each was killed by a measurement, not by a preference. The reopening condition, if
any, is stated. Reopening one is STRATEGIC work and needs a Slice Contract.

| lane | verdict | gate vs result | evidence |
|---|---|---|---|
| Analytic family (B1 fusion / B2 tail / B3 learned residual) | **DEAD** | fusion net ~0 · tail −67…−214% · R²=0.012 | GAME_PLAN rev 3 |
| Track T — shrinkage-moment tail | **CLOSED** | 1.10× gate → 0.9998× | LOG 2026-08-05 |
| Track P — rank-k deviation propagation | **CLOSED** | 1.25× gate → 1.093× even with a free oracle | LOG 2026-08-05. Break-even rank = width/2 |
| Sprint 3 — structure-aware FLOP allocation | **FAIL** | → 1.0246× | LOG 2026-08-05 |
| Sprint 4/5 — packed activation sparsity | **KILLED** | pre-registered 1.15× → **1.1216×** [1.1188, 1.1244] on D14 | LOG 2026-08-06. D12 and **D14 are SPENT — never reuse** |
| N as an operating-point lever | **CLOSED** | `MSE×N` flat over {16k, 32k, 65k} at 8 shifts | the 14% "optimum" was one heavy-tailed realisation |
| Batched depth-2 Strassen | **OFF** | bitwise identical, −53% ops, **worse** residual | shipped behind `_STRASSEN_BATCHED`, tests kept |
| **Control variates (layer-1 exact-mean, LINEAR forms only)** | **DEAD** | **1.010×, 95% CI [0.979, 1.042]** on our Sobol+digital-shift path — CI includes 1.0. On plain MC the same correction gives 1.383× and the fitted optimum 1.584×: QMC already removes that variance | Skeptic probe 2026-08-11, 14 MLPs × 6 shifts, paired geo-mean, CONFIRM tier. **Covers recursive/per-layer forms** — any correction telescoping to a linear map of the layer-1 sample-mean error is in this class. **Does NOT cover** corrections using an *empirical* per-layer gain (`mean(z>0)`), which are nonlinear and unmeasured. The older "1.80× standalone" is a ratio of means over 3 MLPs and should not be quoted |
| Score optimisation generally | ~~CLOSED at ~2.7e-7~~ **REOPENED 2026-08-10** | — | GAME_PLAN rev 6's frontier described *our method*, not the field's. Board decomposition shows honest teams 4.8× better on MSE at our compute. See § *Open questions* #1 |

**The mechanism behind the packed lane's death, which generalises**: flopscope
prices a gather at 4 FLOPs/element, and packing's entire saving is substituting
gathers for dense multiply-adds. A random-access gather costs far more *wall time*
per element than a BLAS multiply-add, and wall time is charged back at λ = 1e11.
The lane arbitrages the price table against physics, and loses.

---

### Added 2026-08-19 — the carrier audit (LOG 2026-08-19)

| lane | verdict | number | note |
|---|---|---|---|
| **Point-set whitening (full 256x256 ZCA)** | **DEAD on our carrier** | **0.9972x, CI [0.986, 1.009]** | 24 MLPs x 24 shifts, paired, GT 2^23. Worth **1.23x on iid** — the device works, our Sobol already collects it. The competitor "2.46x" is a rung against an iid+antithetic baseline; that author's own ablation reads 0.97x / 0.994x in both orders and the census records 1.05x |
| **Diagonal moment standardisation** | **DEAD, and it was always a no-op** | **1.0000x, CI [0.999, 1.001]** | Our shipped Sobol's diagonal second moment is already **107x better than iid**; its off-diagonal is only 5.9x better. The 2026-08-05 kill and the radial-imbalance predictor both probed the already-exact half |
| **ReLU-homogeneity sphere projection** | **DEAD on our carrier** | 1.0010x, CI [0.998, 1.004] | Exact and free, but Sobol already balances the radius. Worth +4% *inside* a design carrier |
| **Tent / baker fold** | **DEAD alone** | 0.9810x, CI [0.928, 1.037] | Only `tent+white` clears, at **1.0931x [1.044, 1.148]** — the one surviving input-space combination |
| **Degree-3 structured design carrier** | **DEAD against the shipped stack** | **1.047x, CI [0.979, 1.120]** | Bare it is 1.182x [1.092, 1.290]; the rank-32 rotation collects it. **Does NOT close a true degree-4/5 design** (strength-4 OA / Kuperberg cubature), whose external ceiling is ~1.46x |
| ~~Strassen-Winograd~~ | **REOPENED** | **1.267x counted (depth 2), MSE-exact; ~1.10-1.16x net after residual** | Measured on the real flopscope 0.10.0. Our 2026-08-06 negative was on the *packed* v2.5 carrier (tiny irregular blocks). Now the largest bankable lever in the project |

**The governing mechanism, measured three ways today: SUBSTITUTION.** Whitening
vs Sobol, design vs rotation, and (previously) CV vs QMC. Two devices that
annihilate the same low-order structure do not compose — whichever is added first
takes the credit. **Read every competitor ladder for what its baseline arm
contains, never for its largest rung.**

⚠️ **Correction, 2026-08-19 (P0.4): two of those three ways are ONE sample.**
`probe_whitening.py` and `probe_design_carrier.py` both hardcode
`SEED0 = 30_806_016_200` with `N_MLPS = 24` — **the same 24 networks**. The
whitening kill (0.9972x) and the degree-3 design-carrier kill (1.047x) are
therefore **correlated evidence, not two independent readings**. Both are DESIGN
tier and no gate was decided on either, so neither verdict changes; the claim of
independence does. The CV-vs-QMC leg is on different data and still stands. The
seed ledger now prints this every time anyone runs `preflight` step 6.

~~**New standing number: 7.0 us/op**~~ **SUPERSEDED 2026-08-19.** That was a
ratio *through the origin* from a single submission (327743: 1,416 ops, 9.845 ms).
Fitted on **all five** graded submissions the grader's residual is a **line**:

    residual_graded  ≈  2.551 ms  +  5.780 µs/op        R² = 0.898, LOO 0.829 ms

**2.551 ms is FIXED overhead charged to every MLP before our first operation** —
25% of v2.4.1's whole residual, 2.55e8 FLOP-equivalents. A ratio through the
origin cannot see it and mis-prices every candidate whose op count differs from
the one it was fitted on. The discriminating evidence: **v1.0 has 12.5× v0's
counted FLOPs and LESS residual** (4.491 ms vs 6.743 ms), which no function of
FLOPs can produce. Price any counted-FLOPs-for-op-count trade with the line, and
note that a candidate far outside 329–1,352 ops is an extrapolation.

**Correction owed to a durable doc.** `docs/writeup-draft.md` and
`docs/RESEARCH_NOTES.md` attribute the 2.35x moment-standardisation regression to
"a first-order bias". Moment matching's bias is **O(1/N)** (Glasserman 4.5.1
p.245) and diagonal forcing measures at exactly 1.000x here and in forum 18151.
The number may be real; the mechanism is not. **Do not ship it.**

## Open questions, ranked

*Re-ranked 2026-08-19. Questions 1 and 2 of the previous ranking are now answered
and have moved to § Closed lanes; what is left is genuinely short.*

1. **THE PHASE-2 CONFIGURATION — blocking, and it lands within hours.** Shape,
   `n_mlps`, `B`, λ and the floor are unpublished. This is the highest-ranked
   open question not because it is deep but because **everything downstream is
   conditioned on it**: `_PRUNE_TAU = −2.5` and `_BASIS_RANK = 32` are fitted at
   256×32, D17 is single-use, and any design-carrier work is width-specific.
   **If `L/n` falls materially — a wider or shallower Phase 2 — the mechanistic
   lane reopens wholesale**, because ARC's own cumulant propagation breaks even
   only when `L^K < n` (L < 16 for K=2, L < 6.3 for K=3). At 256×32 we sit at
   L/n = 1/8 and every variant of it is either over budget or worse than MC.

2. **Does v2.6 actually win?** Built, preflight GREEN, MSE settled at 0.9949×.
   The cost claim reads **1.068–1.093× projected on the grader** and **0.952×
   measured on this machine**, and the sign flips on the residual price — a
   quantity this document classifies **E1, directional only**. See § *Shipped
   file*. **Only a graded run decides it.** A Skeptic has not seen the build; the
   transfer argument (that v2.4.1's measured 4.34× per-op inflation carries to a
   file with 1.7× the operation count) is the load-bearing assumption and is
   exactly what an independent falsifier should attack.

2b. **The design lane is ALIVE, and the missing rung is now identified.**
   Measured 2026-08-19 (`local_sim/test_carriers.py`, 18/18): the shipped Sobol
   carrier's degree-4 moment error is **0.510× iid's** — it collects about half
   of degree 4 and none of degree 6. A Haar frame bank is **exactly a 3-design**
   (machine precision at k=1,2,3) and **0.700× iid at k=4**, which is the
   mechanistic explanation of the audit's degree-3 kill: it brings nothing the
   rotation had not already taken. **The remaining prize is specifically the
   degree-3 → degree-4/5 step**, and nothing above degree 4 is reachable with
   anything we have (every carrier sits at 0.76–1.09× iid at k=6). Two further
   mechanisms fell out: sphere projection is **angularly inert** (it moves the
   radius only, which is why it measured 1.0010×), and antipodal symmetrisation
   **costs ~2× at even degrees** because it uses n/2 distinct points.
   **BUILT, same day.** `local_sim/carriers.py:bch_oa` — the extended dual-BCH
   orthogonal array, `{Tr(ax + bx^3)}` plus the all-ones coset, 2^17 = 131,072
   rows on 256 coordinates. **Strength 5, verified combinatorially** (every random
   4- and 5-subset of columns perfectly balanced, 320/320). Its degree-4 moment
   error is ~~**0.0305× iid — 17× better than the shipped carrier** — and **0.138×
   at degree 6**~~ **← BOTH NUMBERS ARE WRONG, CORRECTED 2026-08-20 (CP1.1).**
   `test_carriers.py` divided *every* carrier by iid at **N = 16,384**, including
   this 131,072-row array. iid's own moment error falls as `1/√n`, so an array
   measured at 8× the points was flattered by ≈ `√8 = 2.83×`. **At matched n the
   BCH array reads 0.0756–0.0908× at degree 4 and 0.343–0.421× at degree 6**, and
   "17× better than the shipped carrier" is **≈ 5.8×**. The harness is fixed —
   an iid baseline is now built for every distinct point count — and the ranges
   quoted here are two independent realisations, because the iid denominator
   itself carries **±3.5%** per draw. Nothing that was *decided* changes: the
   array's absolute degree-4 error (7.2321e-04) never moved, and the end-to-end
   1.4775× was measured against a Sobol arm, not against iid. No
   Kerdock/Z₄ machinery was needed.
   **MEASURED END TO END, same night: the design is worth 1.4775x and we cannot
   afford it.** 48 fresh D18 networks, GT 2^24, arms matched at 131,072 points and
   at cost (counted FLOPs differ by 0.008%): **adjusted gain 1.4775×, studentised
   95% CI [+3.80e-08, +1.81e-07] excluding zero, relative [+14.4%, +45.9%], better
   on 33/48, sign-test p = 0.013.** Raw-MSE gain 1.4826×. **This is the largest
   measured effect in the project** — every other input-space device died between
   0.98× and 1.09× — and it lands within 1.5% of the ~1.46× ceiling derived
   independently from the harmonic spectrum.
   **The blocker is cost, and it is hard.** `C/B = 1.3509`: **48 of 48 MLPs over
   budget**, which the grader scores as a *failure* (multiplier 1.0 *and* zeroed
   predictions). The array only works whole — halving it gives a constant column
   and centroid norm 1.0 (41× worse end to end), and a *random* half is worse than
   Sobol at k=2 and k=4. **And FWHT is not the fix**: the first layer is only
   **4.7%** of the 32-layer chain (1.718e10 of 3.674e11), so even a free first
   layer leaves `C/B = 1.29`. That earlier speculation is withdrawn.
   **The construction question is SETTLED for linear codes: 131,072 is the
   floor.** Proven by exhaustive enumeration, not argued — the BCH parity-check
   columns `(x, x³)` form a Sidon set of size **255** in F₂¹⁶ (all 32,385 pairwise
   XORs distinct), and **all 65,535 nonzero vectors are 3-subset sums**, so a
   256th column always creates a weight-4 dual codeword. *Zero* candidates.
   **At width 255 the array would be affordable** (65,536 rows, `C/B ≈ 0.68`);
   width 256 rather than 2⁸−1 costs exactly **2×** on this lane.
   **Both known cost levers together are not enough**: depth-2 Strassen (1.267×)
   and an FWHT first layer (1.047×) leave `C/B = 1.0184` — still over, and far too
   close to a cliff that scores as a *failure*.
   **The subspace route is CLOSED too, and the reason is the mechanism.** Probed
   along directions confined to 32 dimensions of its own frame, the *unrotated*
   array reads **5.378e-02** — 4.6× *worse* than Sobol's 1.164e-02 — while the
   *rotated* array reads **7.676e-04**, flat across every confinement level.
   Matching `(3 − 2S)·258/(3·256)` with `S = Σaᵢ⁴ = 3/(k+2)` at every rung
   (predicted 0.00 / 0.77 / 2.27 / 5.15% low at k = 256/128/64/32).
   **The random rotation IS the mechanism**: it turns every response direction
   into a spread one, which is the only regime where a ±1 cube array is exact —
   and that requires balance on **all 256** columns. You cannot have the cheap
   array and the rotation at once.
   **Two of four routes are now closed with proofs.** What remains: a non-linear
   array or spherical 4-design near the 33,152 floor (**research**), or a Phase-2
   budget larger relative to the shape (**not ours to choose**). If either lands,
   `2.6314e-7 / 1.4775 = 1.781e-7` — roughly half the remaining gap to the best
   published honest score (9.10e-8), in one step.

   **✅ THE NAMED RESEARCH ROUTE LANDED, 2026-08-20 (CP1.1).** The length-256
   **Kerdock** code — non-linear, so untouched by the 131,072 linear floor —
   plus the `2d` axial points is an **exact spherical 4-design at 66,048
   points**, i.e. **33,024 distinct directions, below the 33,152 the route was
   specified at**. Degrees 1–5 measure 2.5e-16 … 7.0e-15; degree 6 is equivalent
   to the BCH array once scaled to equal n (1.016).
   **Strength 5 is PROVEN, not sampled**: the exact Delsarte transform of the
   weight enumerator gives `B₁…B₅ = 0` and `B₆ = 11,606,784`.
   **The degree-4 exactness is an identity** — the `2d` axial points contribute
   precisely the `2S = 2Σaᵢ⁴` the cube array is short by, for *every* direction
   at uniform weights — so it holds under **confinement to a rank-32 subspace**,
   which is exactly what killed the subspace route for a bare array.
   **Both halves are load-bearing** (measured counter-examples): random antipodal
   signs + axial reads 1.93e-02, and a strength-5 array of the *wrong size*
   (BCH at 2¹⁷) + axial reads 3.94e-04. The set must be strength ≥ 4 **and** have
   exactly `d² = 65,536` points.
   **Cost: `C/B = 0.6934` counted** — under the 1.0 failure cliff, against the
   131,072-row array's 1.3509 — *provided the random rotation is folded into `W₁`
   rather than applied to the points*: `W₁ᵀ(Qu) = (QᵀW₁)ᵀu`, verified to 1.7e-13,
   and **258× cheaper** (6.70e7 vs 1.73e10 FLOPs, i.e. 0.0002 of B instead of
   0.064).
   **What is NOT established: any end-to-end MSE gain.** The 1.4775× on record is
   bchOA-vs-sobol at matched N = 131,072; this is a different design at half the
   points. That measurement needs fresh networks on a **DESIGN** block (D16 has
   330 free, D18 has free seeds) — **not D17 or D19**. Artifact:
   `experiments/2026-08-20_kerdock/kerdock_profile.json`.

3. **Where does the ~2.6× to the published honest frontier come from?**
   Unresolved and possibly unresolvable from public data. Their sample counts are
   not published, so their **variance advantage cannot be separated from their
   cost advantage**. Their write-ups attribute it to a strength-4 orthogonal-array
   design plus dead/on/kink routing plus FWHT. The one variance lane the carrier
   audit did *not* close is a **true degree-4/5 design** (BCH strength-4 OA, or
   Kuperberg's 2¹⁶+512-point cubature which is exact to degree 5 for the sphere
   *and* for N(0,I) simultaneously). External ceiling from the layer-32 harmonic
   spectrum (deg ≤5 share ≈ 0.36–0.39): **≈1.46×**. Width-specific — do not build
   it before #1 is answered.

4. **Moonshot G1 — blocked on the D10 renormalization fit, and it now has a
   deployable gate for the first time.** G0 (2026-08-07) proved the layer
   convention, built the calibrated-K2 baseline (depth 8 → 1.2067e-05, G1 must
   reach ≤4.02e-06), and read the oracle: full-rank K3 beats the control 12.786×,
   but pre-registered r ∈ {4, 8, 16} is *worse* and the ≥3× gate needs r ≈ 128 =
   width/2 — where Track P found factorisation stops paying. **New, better gate
   (2026-08-19): a closure whose mid-layer mean is 3–6× more accurate than the
   present one licenses the mid-layer control-variate lane** (`r_c = δ_c/SE_c < 1`
   at some depth where ρ² is large), which the oracle prices at up to ~100×. That
   is a *deployable* success criterion, unlike the old ≥3× accuracy bar which was
   never tied to a use. Kill-by-default **2026-09-01**. D08/D09/D10 untouched.

5. **Is there a Phase 2 write-up round?** Unverified and worth one email. The
   Phase 1 window closed 2026-08-17 and we missed it. The "Sep 26" figure in older
   docs is **uncorroborated by any source**. This matters more than it looks: the
   contribution prize is where our comparative advantage now sits (§ *Standing
   risks*), and the answer changes how the month is allocated.

---

## Standing risks

- **Uncommitted work — PARTIALLY resolved, and the remainder is load-bearing.**
  The *experiment* record landed across five commits through `7040f41` (mint
  provenance, vault, invariant suite, Sprints 4–5, Moonshot G0, records), so LOG
  entries can cite commits again. **What did not land is the cross-agent harness
  itself**: `AGENTS.md`, `ai/`, `.agents/`, `.codex/`, `scripts/guard_paths.py`,
  `scripts/verify_portability.py`, `scripts/git-hooks/`, `scripts/human_model_state.py`,
  `scripts/install_git_hooks.py`, `.claude/hooks/guard_hook.py`, plus modifications to
  all six SKILL.md adapters, `CLAUDE.md` and `README.md` — all **untracked** as of
  2026-08-10. `7040f41` committed only the Claude-side skills.
  This is sharper than the original risk: `CLAUDE.md` `@`-imports `AGENTS.md` as the
  operating contract, and `verify_portability.py` reports 174/174 against files that
  exist **only in the working tree**. Losing this tree loses the entire
  agent-independent layer and both agents' enforcement.
- **The cost half of the evaluator is not trustworthy.** See below. **New, 2026-08-10:
  the unreliable half is OUR measurement, not the grader's** — see the transfer-ratio
  note in § *Evaluator trust*.
- **Deadline conflict**, unresolved by the organizers since Aug 5 (thread 18130).
- **The metric has a heavy tail.** Fresh-MLP max/median = 12.9×; the worst 1% of
  MLPs carry 6.9% of the score. At a re-run of size R = 10 the p90 score is 50%
  above the median. Tail risk, not mean improvement, is what the re-run measures.

---

## Evaluator trust — E1/E2, and the split is the point

The score is `MSE × max(0.1, C/B)`, and its two halves are measured to completely
different standards. We are currently at `C/B ≈ 0.20`, well above the 0.1 floor, so
**both halves are live**.

| half | level | why |
|---|---|---|
| **Accuracy (MSE)** | **E2, approaching E3** | mint proven bit-identical to the grader (100/100, uint32); seed-block registry with machine-readable single-use blocks; gates pre-registered in LOG before the block is touched; paired common-random-number statistics with CIs; 32-shift averaging; every MSE quoted against its GT-noise floor; 30/30 invariants including permutation equivariance; touch-counted sealed vault |
| **Cost (`C` = counted FLOPs + λ·residual seconds)** | **E1 → E2 PROPOSED, 2026-08-19** | counted FLOPs are exact and reproducible. Local wall time is not measurable here — but **we never needed it**. The grader measured residual for us **five times** and the telemetry was sitting unused in `experiments/run_reports/`. Fitted on all five: **`residual ≈ 2.551 ms + 5.780 µs/op`**, R² = 0.898, leave-one-out 0.829 ms. Residual is only **1.9–3.3% of `C` on the grader**, so `C` is now determined to well under 1%. **The reclassification is STRATEGIC and is Samuel's to make** — it gates the "no autonomous loop while E1" rule. See `experiments/LOG.md` 2026-08-19 (night) |

**The local residual has no quotable level — established 2026-08-10 (submission
327743), corrected the same day after independent falsification.** Same
byte-identical `estimator.py`, same 100-MLP mini split:

| reading | ms/MLP | vs grader 10.022 |
|---|---|---|
| `--runner local`, 2026-08-06 | 70.85 | 7.07× |
| `--runner subprocess`, 2026-08-06 | **54.47** | **5.44×** |
| `--runner subprocess`, 2026-08-10 | 42.94 | 4.28× |

Runner effect, same day **1.301×**; machine-state effect, runner fixed **1.269×**;
they multiply to the observed 1.65×. So the spread is **half path, half time** — not
"purely the runner", and the overstatement is ~30%, not 65%.

`AGENTS.md` § *Measurement law* states "~7×" as binding and **must stay unedited**
until repeatability is measured. If three same-day repeats span >10%, the correct fix
is not "4.3× instead of 7×" but *"this machine has no quotable absolute residual
level"*, with the line replaced by a range carrying its measurement path.

**What this does NOT contaminate.** The claim that Sprint 4 S2 and the D14 residual
model "scaled by ~7" was **false and is withdrawn** — two different sevens were
conflated. D14's deciding model is *local **ratio** × graded anchor*
(`9.84 ms × 490.4/62.3`); the 7.87 there is a within-run candidate/control ratio,
invariant to any uniform rescaling of this machine, and the local→grader level never
enters. The local-absolute model is explicitly flagged "never decides". **The real
exposure, newly identified: that candidate/control ratio itself reads 5.47× in one
local harness and 7.87× in another — a 44% spread on the quantity that does decide.**

**Grader-side residual: a detected shift, not stability.** Paired 1.0180× [1.0119,
1.0241] excludes 1.0. But n = 1 submission per file, so the n = 100 CI is
pseudoreplication; and `residual_s` is a subtraction remainder whose components moved
~78 ms while the remainder moved 0.177 ms. Do not attribute it to the `eigh` path.

**Consequences, binding:**

- **No autonomous / Ralph-style optimization loop.** The framework rule is
  evaluator ≥ E2 before bounded autonomy; the composite here is not.
- Cost claims follow the three-model rule (`docs/VALIDATION.md` §2b): report three,
  freeze one *before* the decision block, decide on
  `max(local-ratio-on-graded-anchor, competitor-anchored)`. The `max` is
  load-bearing — the competitor anchor alone would have passed a carrier that
  did not deserve it.
- **Improving the cost evaluator outranks any further score optimization.** The only
  instrument that would do it is a graded run, which costs a submission.

---

## Harness — what is enabled and why

| capability | layer chosen | bottleneck it removes |
|---|---|---|
| Evaluation | the project's own scripts (`edge_cases`, `mint_provenance`, `vault`, `whest run`) | already best-in-class; wrapped, not replaced |
| Change gate | `scripts/preflight.ps1` | four commands that must all run on every estimator change, previously prose in `VALIDATION.md` §7; adds the ship-file identity check that existed nowhere |
| Pre-registration | skill `slice-contract` | the repo's #1 historical failure mode: a residual model or gate chosen after seeing the answer (Sprint 4 twice) |
| Independent falsification | agent `skeptic` | two "findings with tidy causal stories" turned out to be noise; the one independent audit (rev 11) corrected four claims |
| Orientation | this file + `docs/GOTCHAS.md` | a fresh session previously had to read ~150 KB across 12 stacked GAME_PLAN revisions to learn what was decided |
| Human comprehension | `docs/HUMAN_MODEL.md` (Samuel-owned, write-protected from agents) | no comprehension checkpoint existed anywhere in the record |
| Prompt friction | `.claude/settings.json` allowlist | read-only lab commands prompted on every invocation |
| **Agent portability** (audit 2026-08-07) | `AGENTS.md` + `ai/workflows/` + shared `scripts/` enforcement | the whole harness lived under `.claude/`. Exhausting one agent's quota meant losing the methodology, not just the session |

**The portability layer, in one paragraph.** `AGENTS.md` is now the cross-agent
operating contract and `CLAUDE.md` imports it, so no project behaviour is
agent-specific. The six workflow commands are defined once in `ai/workflows/`;
`.claude/skills/` and `.agents/skills/` (Codex) are thin adapters that load them, and
`scripts/verify_portability.py` (162 checks) fails if either side drifts, grows a rule
of its own, or names a product where it should name a role. The Skeptic exists for
both agents (`.claude/agents/`, `.codex/agents/`) over one shared brief. Enforcement is
one implementation, `scripts/guard_paths.py`, called by both agents' hooks *and* by
`scripts/git-hooks/pre-commit` — the backstop that holds even for an agent whose hooks
nobody has wired. **Switching drivers is: start the agent in the repo root, run
`session-start`, continue with `next-slice`.** Not verified in this environment: the
Codex hook *registration* schema, because Codex is not installed here — see
`.codex/hooks/README.md`. The pre-commit backstop covers the gap meanwhile.

**Deliberately not enabled** — each was considered and rejected for a stated reason:

| rejected | reason |
|---|---|
| Ralph / autonomous loops | cost half of the evaluator is E1. Forbidden until a graded anchor exists |
| MLflow / Phoenix / Langfuse | ~30 decisive experiments total, and the artifacts already exceed what these capture. None can model a single-use seed block or a touch-counted holdout — which are the properties that actually protect this project |
| Context7 | two pinned dependencies; the real API surface is already verified against the installed package and recorded |
| Serena | ~70 source files; `rg` + Read is sufficient |
| Repomix | the repo is local; subagents read it directly |
| Promptfoo | there is no LLM in the product |
| Playwright / Chrome DevTools / design skills | no UI anywhere |
| Exa / Firecrawl / last30days | the research surface is one forum and one leaderboard, already scripted (`board_snapshot`, `board_forensics`, `board_census`, `phase2_checklist`) |
| GitHub MCP | `git` + `gh` cover it |
| Symphony / Agent Deck / workflow orchestration | one operator, one working tree, evidence-gated work. Parallel agents here would create merge and provenance risk, not throughput |
| Google Drive / Slack / Gmail / Calendar MCPs | irrelevant to this repo; Gmail and Calendar are also unauthenticated in this environment |

---

## Next actions

*Rewritten **2026-08-21** at session close, immediately after the Phase-2 shape
landed. Everything above this line describes a 256x32 world; **Phase 2 is
1024 x 16**, and this list is the bridge. Nothing here depends on chat history.*

**Start here.** `session-start`, then this list **in order** — the ordering is
load-bearing, because steps 2 and 3 cannot be costed or trusted until step 1 is done.

### The shape moved. Do these first, in this order.

1. **Upgrade the toolchain and re-prove mint provenance at 1024 x 16.**
   Installed: whestbench **0.14.0**, flopscope **0.10.0**. The Phase-2 dataset was
   baked with whestbench **0.16.0**, flopscope **0.12.0**. Until this is done we
   cannot load the dataset, and `local_sim/mint_provenance.py` has never been run
   at the new shape. **Everything rests on our networks being the grader's
   networks; that proof is currently only established at 256x32 and four other
   Phase-1 shapes.** IMPORTANT (changes the instrument).
   *Watch for:* flopscope 0.10 -> 0.12 is two minor versions and **the price table
   is exactly the unverified thing** (`docs/CURRENT_STATE.md` § *Evaluator trust*).
   Re-run `local_sim/residual_model.py`'s counted-FLOP legs and see if they move.

2. **Establish `B`, λ, the multiplier floor, and whether biases are absent.**
   None is in the dataset. Sources, cheapest first: the round/challenge config via
   `whest`; `whestbench` 0.16.0's own defaults; and — decisively — a graded
   report's `config_snapshot`, which requires a submission. **Every `C/B` in this
   document is a Phase-1 number and none of them transfers.**
   *Naive arithmetic, stated as arithmetic:* a 1024-wide layer costs **16x** a
   256-wide one and there are **half** as many, so per-sample propagation is
   **~8x Phase 1**. If `B` did not grow to match, the affordable sample count
   collapses and the whole operating point moves.

3. **Re-fit `_PRUNE_TAU` and `_BASIS_RANK` at 1024 x 16 on a DESIGN block.**
   -2.5 and 32 are Phase-1 values and neither transfers. `_BASIS_RANK` in
   particular was chosen as `width/8`; at 1024 the analogue is a different number
   and possibly a different *mechanism*, since Track P's break-even was
   `rank = width/2`. **Allocate a NEW design block** (`local_sim/seed_blocks.py`) —
   D16's 330 free seeds and D18's remainder are 256x32 networks and are **not
   reusable at the new shape**. STRATEGIC.

4. **Only then** re-open the carrier and estimator questions. See § *Open research*.

### Waiting on Samuel — no agent may do these

5. **Submit early for telemetry, or not?** A Phase-2 submission of the frozen
   **v2.4.1** would return `config_snapshot` — `B`, λ, the floor, the real price
   table — from the grader's own numbers, unblocking step 2 immediately. It costs
   a submission slot and posts a Phase-1-tuned score to a fresh board. **Samuel's
   call**, and hard rule 6 applies regardless: `whest validate` -> clean mini run
   with `n_failed_mlps: 0` -> ask.
   *Note:* v2.4.1 and v2.6.1 are both shape-generic and will *run* at 1024x16, but
   their constants are fitted at 256x32, so a score from either is telemetry, not
   a competitive entry.

6. **The E1 -> E2 reclassification of the cost half.** `residual ≈ 2.551 ms +
   5.780 µs/op`, R² = 0.898, fitted on five graded submissions. **Now doubly
   caveated:** those five are Phase-1 submissions at 256x32 under flopscope 0.10,
   and both the shape and the price table have moved. It may need re-fitting from
   scratch. STRATEGIC — the E1 label gates the "no autonomous loop" rule.

7. **The vault is Phase-1 and cannot be re-baselined into Phase 2.** 500 sealed
   256x32 networks, touch **1 of 3**. A 1024x16 vault is a **new mint**, and the
   old one becomes historical. Decide whether to build one and what it costs.
   Do **not** difference a Phase-2 touch against touch #1.

8. **Team freeze 2026-09-05 23:59 UTC.** Solo or not.

9. **Two questions to the organisers** — `docs/email-draft-organizers.md`.
   (a) Is there a Phase 2 write-up round, and when? (b) One designated submission
   or two — announcement 18143 says two, the Rules tab says one, still unanswered.

### Open research, ranked for the NEW shape

10. **The cumulant / mechanistic lane, REOPENED on its own pre-registered
    condition.** Open question #1 said a wider or shallower Phase 2 would reopen it
    "wholesale" because ARC's propagation breaks even only when `L^K < n`.
    **`L/n` fell from 32/256 = 1/8 to 16/1024 = 1/64**, and `L² = 256 < 1024`
    now holds at K = 2 (K = 3 does not: `4096 > 1024`). This is a reopening by
    **external fact**, which is the standard `AGENTS.md` requires. Start from
    `docs/MOONSHOT.md` and the G0 baseline. **This is now the highest-value
    research lane**, and it is the one the shape change *helped*.

11. **Generalise the Kerdock design to width 1024, and price it.** The CP1.1
    object is width-256 and dead at the new shape, but the *mathematics* carries:
    `1024 = 4^5`, the real-MUB bound is `d/2 + 1 = 513`, the Galois ring is
    `GR(4,9)`, and the analogue is `d² = 2^20 = 1,048,576` sign vectors plus 2,048
    axial = **1,050,624 points**. The axial identity
    `[(3-2S) + 2S]/(d² + 2d) = 3/(d(d+2))` is dimension-free and should hold
    unchanged. **The open question is entirely cost**: 1,050,624 points at ~8x the
    per-sample cost is 16x Phase 1's operating point, and affordability depends
    wholly on step 2. `local_sim/carriers.py` already has the derivation to adapt.

12. **Moonshot D10 renormalization fit**, kill-by-default **2026-09-01** — but see
    #10: the break-even arithmetic that made this marginal has just changed in its
    favour, so re-read the kill condition before honouring it.

**Explicitly NOT next.** Do not run **CP1.2**, the end-to-end Kerdock arm on D18 —
it is suspended, not killed, and its 192 networks are 256x32. Measuring a carrier
for a shape that is no longer scored is the most expensive mistake available right
now. Do not spend **D17 or D19**; both are single-use, both are unspent, and both
are 256x32 allocations that should be re-thought at the new shape.

