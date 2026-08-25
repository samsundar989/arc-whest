# CURRENT_STATE.md — read this first

**Snapshot: 2026-08-25. Phase 2 is live.** This file says where execution stands;
[`GAME_PLAN.md`](GAME_PLAN.md) is the only active architecture/roadmap;
[`VALIDATION.md`](VALIDATION.md) is binding for measurement; the newest entries at
the bottom of `experiments/LOG.md` contain the evidence narrative.

## One-minute orientation

- Phase 2 closes **2026-10-17 23:59 UTC**. Team/registration freeze is
  **2026-10-02 23:59 UTC**; contribution write-up is due **2026-10-24 23:59 UTC**.
- Shape is `1024 x 16`; budget is `2**41`; multiplier floor is `0.1`.
- Official stack is WhestBench 0.16 / flopscope 0.12. Lambda is zero; residual cap
  is 0.4 seconds; total wall cap 120 seconds; solution memory 8 GiB.
- The live leader is Puffi at `1.802e-8` adjusted / `3.603e-8` raw with
  `1.099802385e12` FLOPs. There were about 30 entries at the snapshot.
- This account has **no Phase-2 submission**.
- The current shipped file routes Phase-2 width to basic `k2p1`. Expected Phase-2
  performance is around `3.96e-7` adjusted / `3.95e-6` raw: about 22x and 100x
  behind the leader respectively.
- The repo's scientific decision protocol is strong. Its deployment evaluator was
  not Phase-2-truthful; **P2-P0 repaired the deployment parity** and the local
  meter now reproduces the official runner op by op, to zero.
- **Phase 0 is CLOSED.** Its all-of gate G1–G7 is met, with a literal independent
  **ACCEPT** on reviewed HEAD `952bd61`, recorded in closure commit `7367fdb` at
  the bottom of `experiments/LOG.md`.
  It took four review passes — PROVISIONAL, PROVISIONAL, REJECT, ACCEPT — and the
  first three each found something real, including a gate clause satisfied by a
  check that could not fail.
- **P1-PRE has LANDED and is awaiting its verdict.** D1, D2 and D3 are closed in
  code, `PREFLIGHT GREEN — 16/16`, portability 99/0, and eight deliberate fault
  injections all turn the gate red. **It is not yet accepted:** two independent
  passes returned **PROVISIONAL**, each finding a real defect in the repair
  itself, and a third review of the corrected tree is outstanding. See the
  P1-PRE entries at the bottom of `experiments/LOG.md`.
- **Immediate next action: obtain that third verdict.** Nothing else may start
  until it reads ACCEPT — in particular Phase 1A does not begin.
- **Then Phase 1A — construction only.** Build the width-1024 Kerdock/MUB
  generator, independent verifier and dense reference with **no seed block
  claimed, no `estimator.py` edit, and no truth measurement**. Phase 1B (DESIGN
  evidence) begins only after 1A is independently accepted and its strategic
  contract is reviewed.

## Authoritative document map

| need | source |
|---|---|
| current status and next action | this file |
| target architecture and full execution plan | `docs/GAME_PLAN.md` |
| live rules/leaderboard/research evidence | `docs/PHASE2_LIVE_AUDIT_2026-08-24.md` |
| measurement protocol | `docs/VALIDATION.md` |
| operational traps | `docs/GOTCHAS.md` |
| contracts, results, Skeptic verdicts | bottom of `experiments/LOG.md` |
| independent review brief | `docs/SKEPTIC.md` |
| Claude implementation prompt | `docs/CLAUDE_CODE_HANDOFF.md` |
| contribution manuscript | `docs/writeup-draft.md` |

The old frontier packets and handoffs are superseded. Stub files remain only where
source comments/tools link to their historical names; they redirect here and must
not be treated as plans.

## Current deployed and research state

### Shipped estimator

- `estimator.py` is still v2.6.1 and must remain byte-identical to its frozen archive
  unless a STRATEGIC contract authorizes a versioned change.
- The graded Phase-1 file remains v2.4.1.
- Phase-2 width takes `_analytic_only`; the effective method is K2p1.
- None of the Phase-1 rotation/Sobol/pruning/Strassen carrier is active at width
  1024.
- Official 0.16 validation passes.
- One official Phase-2 subprocess smoke produced:

| metric | one-row anchor |
|---|---:|
| score | `4.501854e-7` |
| final-layer MSE | `4.501854e-6` |
| multiplier | `0.1` |
| counted FLOPs | `137,600,326,656` |
| `C/B` | `0.0625734` |
| residual | `0.05368 s` |
| total wall | `0.35487 s` |
| failures | `0` |

This is an operational smoke, not a score estimate.

### Phase-2 local evidence that survives audit

P2-CP1 used 96 fresh 1024x16 networks and two independent `2**22` truth replicas.
There were no truth failures. Its narrow conclusions remain credible:

- K2p2 improved K2p1 about 1.052x for about 0.18% extra counted cost.
- Tested IID and ordinary random-orthogonal arms lost to K2.
- Higher Price/Hermite covariance-map orders plateaued.
- Those arms do not bound valid Kerdock/MUB structured quadrature.
- P01 was a DESIGN block, so it cannot support binding global lane kills.

P2-CP3R's durable mechanistic result:

- 94.61% of K2 error is accumulated mean/variance-state drift;
- marginal third/fourth corrections in the scalar ReLU mean map recover 4.47%;
- the registered O3-versus-O0 gate passed, then the recorded KILL used a different
  post-hoc O1-versus-O3 estimand;
- the oracle reset to true marginal state every layer and did not propagate ARC's
  augmented-K2 joint state.

Therefore marginal final-mean Edgeworth correction remains low value, but recurrent
joint-state augmented K2 is open.

### Seed and vault state

- P01 has consumed **144/600** seeds; 456 DESIGN seeds remain.
- P01 is not a DECIDE block.
- Existing SINGLE_USE Phase-1 blocks keep their recorded state in the ledger.
- The Phase-1 vault cannot be used as a Phase-2 vault.
- No Phase-2 vault exists. Creating or touching one requires Samuel's explicit
  authorization and a STRATEGIC preregistration.

## Live frontier and architecture decision

The strongest external clue is Puffi's cost:

```text
1,099,802,385,000 FLOPs ~= cost of 32,768 full dense directions
32,768 = 16 Kerdock/MUB bases * 1,024 axes * 2 antipodes
```

Puffi publicly used Kerdock/Walsh mutually unbiased bases, antipodes, exact radial
integration, analytic state, and checkpoint control in Phase 1. Phase-2 code is
private, so the specific carrier is **inferred**, not confirmed. The factorization
and roughly 74x improvement over matched-row IID are nevertheless strong enough to
reopen structured sampling under the repository's own rule.

The target architecture is now:

1. 16-basis width-1024 Kerdock/MUB angular carrier;
2. antipodes and exact radial integration;
3. K2p2 analytic state reference;
4. faithful recurrent augmented-K2 joint state if it passes its own decision;
5. compact mid-network sampled-versus-analytic checkpoint response;
6. optional shared-scale/equivariant state correction only if residual diagnostics
   justify it;
7. two final nominees: a conservative carrier/reference candidate and an aggressive
   supported state/response hybrid.

Coupled residual sampling is deprioritized. Its former 1.305x projection assumed a
float32 K2 cost reduction that the frozen artifact does not show: measured cost fell
only 3.1%, not 50%.

## Evaluator state — P2-P0 repaired these, and the slice is closed

All nine Phase-0 blockers were repaired or explicitly bounded, deployment parity
is restored within the documented host limits, and the gate is closed on a literal
independent ACCEPT. Raw evidence:
`experiments/2026-08-24_p2_p0/`; contract, result, two reviews, a retraction, a
REJECT and the ACCEPT at the bottom of `experiments/LOG.md`.

**Four defects were carried forward** from the accepting review (D1–D4 in that
entry). None falsifies a Phase-0 gate clause. D4 — `VALIDATION.md` §7 describing
the old twelve-step gate — is fixed. **D1–D3 are CLOSED IN CODE by P1-PRE
(2026-08-25), and P1-PRE is not yet accepted.**

| carried defect | required repair | state |
|---|---|---|
| D1 — literal-`True` worker-restart witness | replace it with a structured result emitted by this suite; `--skip-worker-death` must make the witness fail/unprovable, never green | **closed** — deferred to a witness-closure section that runs after the provocations; injections A, D, G |
| D2 — witness `suite` field ignored | resolve the named suite to its own result artifact and refuse unknown/mismatched suites with executable negative tests | **closed** — `WitnessResolver`; also repaired on the artifact-writing path, where it had survived; injection B |
| D3 — `--no-write` rewrites lifecycle evidence | prevent regeneration and every tracked write; assert before/after hashes and a clean worktree | **closed** — a read-only latch that raises, plus an external proof by child process, sha256 and git; injection C |

**What P1-PRE cost, and it is the durable lesson:** two independent passes
returned PROVISIONAL, and both found the *repair* re-committing the defect it was
closing — first a guard whose two sides were the same object
(`adjudicated == len(deferred)`), then a scan recorded as covering a defect family
while catching three of its eleven spellings. **Closing an instance of this
repository's signature failure is a fresh opportunity to create one.** The
falsification harness is now committed as `local_sim/p1pre_fault_inject.py` and
carries eight injections.

| # | blocker | state |
|---|---|---|
| 1 | `pyproject.toml` pinned WhestBench 0.14 / flopscope 0.10 | **fixed** — 0.16 / 0.12 pinned and locked; the lock diff is exactly those two packages |
| 2 | `competition_score.py` defaulted to `1e11` lambda | **fixed** — defaults come from `official_config`; live lambda is **0** |
| 3 | `lifecycle_eval.py` defaulted to Phase-1 wall/residual/memory/budget | **fixed** — `LifecycleSpec.for_round()` takes a round's six constants together and records deviations |
| 4 | local weights were raw NumPy, so weight-method ops were unmetered | **fixed** — `fnp.array`, as `MLP.from_row` builds them |
| 5 | local price low by 37,779,456 FLOPs vs the official anchor | **fixed** — reconciled to **zero**, op by op (see below) |
| 6 | the 0.4 s residual cliff was not enforced | **fixed** — armed by default and PROVOKED in the self-test |
| 7 | 8-GiB memory and worker death/restart not emulated | **partly** — worker restart is provoked for real through the official runner; **memory cannot be enforced on Windows at all** and is now an accounting constraint, not a test |
| 8 | manifest/checklist/preflight could be green on the stale stack | **fixed** — preflight step 1 asserts the live stack; the manifest reports `PACKAGE PARITY: ACHIEVED` |
| 9 | shape/GPU parity did not prove the 1024x16 live path | **fixed** — `mint_provenance` now checks 1024x16 FIRST, bit-identical against the installed 0.16 |

### The anchor, reconciled

The 37,779,456-FLOP shortfall had two mechanisms, and neither was the version bump
(flopscope 0.10 -> 0.12 moved **nothing** the shipped file executes):

```text
local, historical spelling      137,562,547,200
  + weight representation            37,746,688   fnp.array weights: astype 33,554,432
                                                  over 16 calls, plus multiply and sum
  + harness materialisation              32,768   whestbench 0.16 calls
                                                  materialise_predictions IN-window
full grader spelling            137,600,326,656   == official, unexplained 0
```

19 operations compared; every name, call count and FLOP figure agrees.

**A candidate whose final assembly is NATIVELY float32 avoids 32,768 FLOPs.**
`materialise_predictions` is free on a float32 return and 2 FLOPs/element on a
float64 one. The shipped file returns float64. The saving is CONDITIONAL: adding a
cast to reach float32 costs the same 2 FLOPs/element and nets zero, so it is only
free if the last array is built in float32. Taking it means changing
`estimator.py`, which is STRATEGIC and needs its own contract.

### What the evaluator still cannot do

- **Enforce the 8,192 MB memory limit.** Windows has no `resource` module, so the
  subprocess worker's `RLIMIT_AS` call is skipped and `--runner local` is advisory
  by design. Measured accounting for the shipped file: weights 64.0 MB, peak
  working set 440.8 MB, peak commit 918.0 MB -> **at most 8.9x headroom, an UPPER
  BOUND**. RLIMIT_AS caps virtual address space, which exceeds both counters by an
  amount no Windows counter reports, so the true headroom is SMALLER than 8.9x by
  an unmeasured factor. This is the optimistic reading, not a safety margin.
- **Observe the grader's own configuration.** Every limit above is a PACKAGE fact,
  and they are not equally well corroborated. A real `whest run --json`
  `run_config` records what the runner USED for `width`, `depth`, `flop_budget`,
  `lambda_flops_per_second`, `residual_wall_time_limit_s`, `wall_time_limit_s` and
  `setup_timeout_s` — and **not** for `memory_limit_mb` or `predict_timeout_s`,
  which remain `ContestSpec` class defaults only. So the **8,192 MB limit — the one
  constraint this host cannot test at all — is also the one with the weakest
  provenance**. No graded report publishes any of them, and none exists for
  Phase 2.
- **Transfer a local second.** The deployment authority for residual is
  `whest run --runner subprocess`, never the in-process figure.
- **Enforce the 5-second setup ceiling meaningfully.** `import whestbench` alone
  is 1.3 s on a quiet host and 4.7–9.1 s under load; our own estimator's setup is
  0.6 **ms**. The in-process timer measures a strictly smaller window than the
  grader's subprocess one, so its "419 ms of 5000 ms" is not a transferable
  margin. Matters directly: the carrier builds its point set in `setup()`.

### The residual budget Phase 1 must design against

Best case, one quiet session, `--runner subprocess` as the authority, 20 MLPs per
cell (`experiments/2026-08-24_p2_p0/p2_residual_paths.json`):

| candidate | median | worst reading | over the 0.4 s cap |
|---|---:|---:|---:|
| `estimator.py` (shipped K2p1) | 56.6 ms | 83.6 ms | 0/20 |
| `arm_k2p1` | 55.9 / 59.4 ms | 95.5 ms | 0/20 |
| `arm_k2p2` — the architecture's reference | 94.6 / 95.1 ms | 140.6 ms | 0/20 |

**Those are best-case and must not be used alone.** The same `arm_k2p2` reads
**260.7 ms median / 390.6 ms worst — 1.02x headroom** in the frozen P2-CP1 ladder
of 2026-08-21 — that figure is `lifecycle_eval` **in-process**, against the
subprocess table above, so the two are not directly comparable; the decomposition
is instrument-matched. A first write-up blamed machine contention; **that is retracted**.
Three experiments decompose the 2.35x: the weight-representation repair is a fixed
**~19 ms**, flopscope 0.10 vs 0.12 is **0.91x** (not the cause), and **~2.35x is
run/session state and remains unexplained.**

**So the carrier's allowance is bounded by the bad day, not the good one:**
`k2p2` at 260.7 ms leaves **~139 ms**, and that day's worst MLP already sat at
1.02x headroom **with no carrier at all**. Phase 1 designs against that, and its
contract needs a falsifier on residual rather than a check performed afterwards.

Two cautions on the numbers above, both from an independent Skeptic: the "worst
MLP" is frequently the **first** MLP (a warm-up transient, which still fails an MLP
under a cliff but is not network heaviness), and **n = 20 is thin for a maximum**.

The paired arithmetic score test, common-random-number design, ground-truth noise
reporting, seed ledger, public-CONFIRM rule, and vault discipline remain sound.

## Active execution queue

| order | slice | class | status |
|---:|---|---|---|
| 0 | official-stack/evaluator repair | IMPORTANT | **CLOSED 2026-08-24** — G1–G7 all met, independent ACCEPT recorded |
| 0H | P1-PRE: D1–D3 evaluator hardening | **IMPORTANT** | **LANDED, VERDICT OUTSTANDING** — code closed, gate green 16/16, 8 injections all caught; two passes returned PROVISIONAL and both were closed; a third review is required before this row reads CLOSED |
| 1A | Kerdock/MUB mathematical spec, direct generator, independent verifier, dense reference | **IMPORTANT** | **BLOCKED** on 0H ACCEPT; no seed, estimator edit, or truth measurement |
| 1B | structured-carrier DESIGN evidence | STRATEGIC | after 1A ACCEPT; contract reviewed before any block is claimed, on a NEW block (P02) |
| 2 | structured-carrier DECIDE | STRATEGIC | pending; new SINGLE_USE block after design freeze |
| 3 | recurrent augmented-K2 parity / DESIGN / DECIDE | IMPORTANT then STRATEGIC | pending |
| 4 | checkpoint response DESIGN/DECIDE | STRATEGIC | pending |
| 5 | integrated frontier and robustness | STRATEGIC | pending |
| 6 | public CONFIRM / authorized Phase-2 vault | STRATEGIC | pending |
| 7 | two frozen submissions and nomination | STRATEGIC | pending |
| 8 | contribution write-up | STRATEGIC claim review | pending |

## Samuel's standing predictions — recorded 2026-08-24, before the data

Taken before the Phase-0 repair and before any carrier or augmented-K2 slice
exists. S2 and S3 are **not yet tested**; they will be restated in the contracts
that test them, and they must not be edited in the meantime.

| # | question | Samuel | outcome |
|---|---|---|---|
| S1 | official-stack repair changes the current K2 score by | `<1%` | **HIT** — exactly 0%; the multiplier is pinned at the 0.1 floor before and after |
| S2 | Kerdock16 vs matched-row IID at 32,768 directions | `2.7x`–`20x` | open (Phase 2 of the plan) |
| S3a | raw-MSE gain of faithful recurrent augmented K2 over K2p2, standalone | `<1.1x` | open (Phase 3) |
| S3b | official 0.4 s residual-cap failure risk | aug-K2 at risk, carrier safe | open |

S2 sits **below** the ~74x implied by the Puffi cost fingerprint. If the carrier
lands in Samuel's band rather than near 74x, the fingerprint inference is
incomplete and the leader has a second mechanism. That disagreement is registered
and worth watching.

## Closed and open lanes

| lane | state | scope |
|---|---|---|
| basic K2/K2p2 | control / conservative branch | sound but not winning alone |
| higher Gaussian covariance-map orders | low value | measured plateau near 1.052x |
| marginal ReLU mean Edgeworth | closed | capped near 1.047x in CP3R |
| recurrent augmented K2 joint state | **open** | never tested by CP3R |
| 16-basis Kerdock/MUB + radial | **highest priority** | reopened by external fingerprint |
| ordinary IID/random orthogonal | control only | tested arms lose |
| broken two-cycle lattice | closed | does not bound valid QMC/MUB |
| checkpoint response closure | **open after carrier** | compositional mechanism |
| shared-scale mixture | conditional | run only on matching residual evidence |
| equivariant learned state corrector | conditional | rollout/cross-fit required |
| dense full K3 | deferred | projected multiple budgets |
| coupled residual sampler | deprioritized | float32/headroom premise unsupported |

## Authority boundaries

- Do not edit `docs/HUMAN_MODEL.md`.
- Do not edit `estimator.py`, claim a seed block, reopen a killed lane, touch a
  vault, change a gate/residual model/protocol, or make a write-up claim without the
  required STRATEGIC sequence.
- Do not submit without Samuel's explicit approval after official validate and a
  clean Phase-2 subprocess mini with zero failures.
- After every IMPORTANT or STRATEGIC slice, obtain an independent Skeptic verdict
  from raw artifacts before believing or propagating the result.
- Keep `CURRENT_STATE.md` updated when a lane, baseline, live rule, or next action
  changes. Do not stack another override on top of this file.
