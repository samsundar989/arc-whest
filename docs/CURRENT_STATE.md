# CURRENT_STATE.md — read this first

*Snapshot, not narrative. Last updated **2026-08-07** (harness audit).
Update this file when a lane opens or closes, a number is graded, or the phase
changes — not on every experiment. The narrative record is `experiments/LOG.md`.*

---

## The clock

| event | when | status |
|---|---|---|
| Round 1428 "Phase 1" | closes **2026-08-07 23:59 UTC** (challenge page) / Aug 10 (organizer thread 18125) | **LIVE, conflicting dates — plan against Aug 7** |
| Round 1429 "Phase 2" | opens Aug 8 00:00 UTC, closes **2026-09-19 23:59 UTC** | UPCOMING, empty board, no `v1-phase2` HF tag yet |
| Write-up due | **2026-08-17** (forum 18109) | the contribution-prize deliverable |
| Registration / team freeze | 2026-09-05 | |
| Results | 2026-10-01 | private re-run of ONE designated submission on fresh MLPs |

**Phase-2 opening is event-triggered, never date-triggered.** Poll with
`uv run python local_sim/phase2_checklist.py`. The runbook is at the tail of
`experiments/LOG.md` (§ *Phase-2 opening runbook*). First action on 1429 going
active is **frozen v2.4.1 for configuration telemetry, and only with Samuel's
explicit approval.**

---

## Where we stand

| instrument | tier | value | date |
|---|---|---|---|
| **graded, submission 324420 (v2.4)** | graded | **2.7194e-7** adjusted · 1.3748e-6 MSE · mult 0.1990 | 2026-08-05 |
| graded, 324094 (v1.1) | graded | 3.893e-7 | 2026-08-05 |
| vault, 500 sealed MLPs, v2.4.1 | VAULT | median **2.4651e-7** · mean 3.2669e-7 · p99 2.0535e-6 · max 3.0469e-6 · 0 non-finite | touch #1, 2026-08-06 |
| public mini (100), v2.4.1 | CONFIRM | 3.172e-7 adjusted · 1.4895e-6 MSE · mult 0.216 · 0 failed | 2026-08-06 |
| invariant suite on the shipped file | — | **30/30** | verified 2026-08-07 |
| board position | — | raw #90/200; **#82 of 192 plausible-honest** on 5 entries | 2026-08-06 10:03Z |

**Shipped file**: `estimator.py` == `estimator_v2_4_1.py` (byte-identical modulo
line endings — `scripts/preflight.ps1` asserts this). v2.4.1 has **never been
graded**: no submission was made in the Phase-1 window, and selection was left at
the default. Vault touch counter: **1**.

Field anchors: best published honest method 1.551e-7 · our declared honest sampling
frontier ~2.7e-7 · GT noise floor of the metric ~2e-10. Full table: `docs/BENCHMARK.md`.

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
| Score optimisation generally | **CLOSED at ~2.7e-7** | GAME_PLAN rev 6 | draw variance dwarfs any remaining ~2% gain |

**The mechanism behind the packed lane's death, which generalises**: flopscope
prices a gather at 4 FLOPs/element, and packing's entire saving is substituting
gathers for dense multiply-adds. A random-access gather costs far more *wall time*
per element than a BLAS multiply-add, and wall time is charged back at λ = 1e11.
The lane arbitrages the price table against physics, and loses.

---

## Open questions, ranked

1. **319341's variance advantage — the only open score question.** Their `MSE × N`
   is 0.018520 against our graded 0.022525: **1.216×**. Ruled out: arithmetic,
   packing, antithetic sampling. It is a *variance* mechanism and we do not know
   which one.
2. **Moonshot G1 — blocked on the D10 renormalization fit.** G0 (done 2026-08-07)
   proved the layer convention, built the registered calibrated-K2 baseline
   (depth 8 → 1.2067e-05; G1 must reach ≤4.02e-06), and read the oracle: full-rank
   K3 flow beats the control **12.786×**, but the pre-registered r ∈ {4, 8, 16} is
   *worse* than the control and the ≥3× gate needs r ≈ 128 = width/2 — exactly
   where Track P found rank-k factorisation stops paying. **Next step is the D10
   renormalization fit, not D08.** D08/D09 untouched.
3. **Phase-2 configuration.** Shape, `n_mlps`, budget, λ, floor are all unverified
   for round 1429. `wall_time_limit_s` has never been observable.
4. **The write-up (due Aug 17).** Sprint 5's negative result is unusually clean
   contribution material: a pre-registered gate, a conservative model frozen in
   advance that was later shown to be load-bearing, and a transferable mechanism.

---

## Standing risks

- **Uncommitted work.** As of the audit: 38 untracked files and ~2,250 modified
  lines — the entire 2026-08-06/07 body of work (mint provenance fix, vault,
  invariant suite, constants audit, Sprints 4 and 5, Moonshot G0). Last commit is
  `2dde612`, 2026-08-05. **No LOG entry can currently cite a commit.** Highest-value
  fix in the repository; see § *Harness* below.
- **The cost half of the evaluator is not trustworthy.** See below.
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
| **Cost (`C` = counted FLOPs + λ·residual seconds)** | **E1 — directional only** | counted FLOPs are exact and reproducible. **Uninstrumented wall time is not measurable on this machine at all**: it bills the same file ~7× what the grader billed it. Every dense-vs-non-dense cost comparison therefore rests on a *model*, and the deciding model is a frozen assumption, not an observation |

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

1. **Commit the outstanding work** (see § *Standing risks*). Restores the evidence
   chain that every LOG entry implicitly claims.
2. **Samuel fills in `docs/HUMAN_MODEL.md`.** ~15 minutes, once. It is the input to
   every future comprehension checkpoint and the thing that lets him direct rather
   than ratify.
3. **Poll for round 1429** and follow the runbook when it flips active.
4. **Write-up** — Aug 17, and it is now co-equal with the score lane.
5. **D10 renormalization fit** if the Moonshot lane continues; do not touch D08.
