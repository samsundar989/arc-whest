# CURRENT_STATE.md — read this first

**Snapshot: 2026-08-26. Phase 2 is live.** This file says where execution stands;
[`GAME_PLAN.md`](GAME_PLAN.md) is the only active architecture/roadmap;
[`VALIDATION.md`](VALIDATION.md) is binding for measurement; the newest entries at
the bottom of `experiments/LOG.md` contain the evidence narrative.

## One-minute orientation

- Phase 2 closes **2026-10-17 23:59 UTC**. Team/registration freeze is
  **2026-10-02 23:59 UTC**; contribution write-up is due **2026-10-24 23:59 UTC**.
- Shape is `1024 x 16`; budget is `2**41`; multiplier floor is `0.1`.
- Official stack is WhestBench 0.16 / flopscope 0.12. Lambda is zero; residual cap
  is 0.4 seconds; total wall cap 120 seconds; solution memory 8 GiB.
- The live Phase-2 board had **50 entrants on the 2026-08-26 UTC recheck** (48 the
  previous day; leader unchanged). Rank 1 was Puffi at `9.50e-9` adjusted / `3.10e-8` raw; the K2
  covariance-propagation tie occupied roughly ranks 37--41 at `3.93e-7` /
  `3.93e-6`. This is a
  live-board observation, not a promotion dataset.
- The signed-in team/Phase-2 submission view says **no submissions match**. This
  account therefore has **zero Phase-2 submissions** as of 2026-08-26; the
  authenticated eligibility capture of that date is consistent with it but does
  **not** by itself enumerate submissions.
- The current shipped file routes Phase-2 width to basic `k2p1`. Its official
  100-MLP subprocess mini measured `4.050566e-7` adjusted / `4.050566e-6` raw,
  with 0 failures, fixed `137,600,326,656` FLOPs/MLP and **4.29x LOCAL,
  one-session** worst-case residual headroom. At the current board that projects
  to about rank 42 and is **42.6x worse adjusted / 130.7x worse raw** than the
  leader. It is a safe
  calibration baseline, not a materially improved Phase-2 system.
- The repo's scientific decision protocol is strong. Its deployment evaluator was
  not Phase-2-truthful; **P2-P0 repaired the deployment parity** and the local
  meter now reproduces the official runner op by op, to zero.
- **Phase 0 is CLOSED.** Its all-of gate G1–G7 is met, with a literal independent
  **ACCEPT** on reviewed HEAD `952bd61`, recorded in closure commit `7367fdb` at
  the bottom of `experiments/LOG.md`.
  It took four review passes — PROVISIONAL, PROVISIONAL, REJECT, ACCEPT — and the
  first three each found something real, including a gate clause satisfied by a
  check that could not fail.
- **P1-PRE is CLOSED on a literal independent ACCEPT** (seventh pass, reviewed
  HEAD `6d05d77`). D1, D2 and D3 are closed; so are the six review-5 findings and
  the four review-6 blocking items. 15 mutations fire their named guard, the
  harness cannot write the repository (containment via `realpath`, measured after
  the write), preflight is 16/16 and portability 99/99.
  **Seven passes: PROVISIONAL x6, then ACCEPT — and five of the six found the
  defect in the repair rather than in the original defect.** Recorded limitations,
  none a false green today: the Linux branch and the child-`setrlimit` Linux
  answer have never executed, and the five design observations of the sixth pass
  were deliberately not pursued.
- **Track 0S is technically safe but procedurally REJECTED at current HEAD.** The
  frozen K2p1 candidate
  (`00240eb1e60810f5…`, byte-identical to `estimator_v2_6_1.py`) scores
  `4.050566369642183e-7` adjusted on the official 100-MLP subprocess mini with
  **zero failures**, fixed `137,600,326,656` FLOPs/MLP and a floor-pinned
  multiplier. The GT-noise floor is recorded (`7.47823932957716e-11`, 0.001846% of
  raw MSE, so the result is **54,165× above** the truth's own noise); the
  account/allowance evidence is preserved with **transcription honestly
  distinguished from capture**; the submission contract and **Samuel's prediction
  (`3.6e-7`, no failures, transfers)** are committed *before* any result. Two
  independent passes ran against an earlier tree, and every finding of both is
  closed.
  **The stale-witness defect is now FIXED:** `preflight -Full` is
  **GREEN 16/16 witnessed at HEAD `576ad46`** (transcript
  `experiments/2026-08-25_trackA/preflight_full_trackA.txt`, which records that
  SHA in its own header), committed as the single evidence commit `240f8df`, so
  the submitting HEAD differs from the witnessed HEAD **only by that commit** —
  nothing under `local_sim/`, `scripts/` or `estimator.py` differs. That is the
  shape the contract's reworded gate item 6 requires.
  **Still outstanding before approval may even be requested, and both are
  blocking:** (1) a **live signed-in recheck** of the account state and the
  10/team/UTC-day allowance — the committed figures are honestly labelled
  *transcribed, not captured*, and this session had no browser or credentials;
  (2) a **focused independent ACCEPT** on the corrected record. **Do not request
  approval or submit until both exist.**
- **P2-CAL (2026-08-26) measured a 2.96x accuracy gain for ~zero FLOPs, and its
  own falsifier fired.** Calibrating the K2 state improves held-out final-layer
  MSE from `4.141e-6` to `1.398e-6` on 96 DESIGN networks (two `2**22` replicas,
  GT noise `1.8e-8`), for `16,384` counted FLOPs against a `2.199e11` free zone —
  the multiplier stays pinned at the `0.1` floor. **But the in-loop, global and
  post-hoc arms are identical to twelve significant figures**, because the K2
  closure is exactly positively homogeneous of degree 1 in its state: scaling
  `(mu, cov) -> (g*mu, g^2*cov)` leaves `alpha` unchanged, so every downstream
  mean is scaled by exactly `g` (verified at `1.6e-30`). **Every purely
  multiplicative state correction is one output rescale in disguise**, whatever
  its parameter count. The contract's prediction 4 is REFUTED, the pre-registered
  falsifier fired, and because the continuation gate was conjunctive on it,
  **the gate is NOT met and no DECIDE block is authorised.** The 2.96x is real
  and holds on both replicas; turning it into a submission needs a NEW contract
  with a new gate and Samuel's prediction taken first. The remaining power is in
  corrections that change `alpha` and therefore do not commute — those are the
  live question, not this one.
- **Track 0S blocker (1) is half closed.** The account state is now **captured,
  not transcribed**: an authenticated `whoami` + `eligibility` call for
  `neopok919` (id 513543) returned `submissions_allowed: true`,
  `rules_accepted: true`, `participation_terms_accepted: true`, Phase 2 active
  until `2026-10-17T23:59Z`
  (`experiments/2026-08-26_p2_live/account_eligibility_capture.json`).
  **The `10 per team per UTC day` allowance is STILL NOT captured** — no reachable
  AIcrowd API exposes a per-day count or balance and the rules page renders
  client-side, so that half stands and needs a browser session.
- **Phase 1A is mathematically valid but PROVISIONAL.** 34/34 verifier and 16/16
  probe. Exact `1/32` cross-basis coherence; the phase-degree distribution is
  exactly **`{0: 1, 2: 511}`**, and **all 130,816** pairwise differences have rank
  10; the rejected
  coordinate mapping is an executable negative control with a raw artifact.
  **K6 is WITHDRAWN** — the verifier is a property checker, not an independent
  reference; CCKS Example 9.2 was attempted and abandoned rather than tuned into
  agreement. **K8 is quantified**: the carrier-aligned fourth moment is exactly
  **21.6881103515625×** the spherical target while 64 random directions give
  `[0.968, 1.023]`, so the subset is **not** a 4-design and random probing cannot
  establish it. The dense carrier's official flopscope ledger is exactly
  `1,099,511,627,776 = 2**40`, hence `C/B = 0.5` and multiplier `0.5` before any
  analytic/control overhead: a **5x score penalty** relative to K2's `0.1` floor.
  The unseen leader's overhead is unknown. The current probe wraps a NumPy-built
  carrier in `fnp`; it is not yet a grader-legal construction path, and its setup
  and Windows memory checks cannot clear the official boundaries. **No seed
  block, no `estimator.py` edit, no truth measurement, and no performance claim.**
  Phase 1B stays blocked until the bounded repair and a fresh independent ACCEPT.

## Authoritative document map

| need | source |
|---|---|
| current status and next action | this file |
| target architecture and full execution plan | `docs/GAME_PLAN.md` |
| live rules/leaderboard/research evidence | `docs/PHASE2_LIVE_AUDIT_2026-08-24.md` |
| the external Phase-2 mission brief, checked item by item | `docs/PHASE2_BRIEF_RECONCILIATION_2026-08-26.md` |
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
float32 K2 cost reduction that the frozen artifact did not show: measured cost fell
only 3.1%, not 50%.

**CORRECTED 2026-08-26 — that 3.1% was measured on a mislabelled arm.**
`local_sim/p2_arms/arm_k2p1_f32.py` calls `flops.stats.norm.cdf` and `.pdf`, which
flopscope 0.12 **promotes from float32 to float64** (it emits a `FlopscopeWarning`
saying so), and the arm never casts back. `Phi`, `phi` and everything downstream of
them — including the covariance accumulator, where nearly all the cost lives — are
float64. The arm is float32 in name only. Measured today
(`local_sim/p2_f32_price_probe.py`, artifact
`experiments/2026-08-26_p2_cal/f32_price_probe.json`), one network:

| arm | counted FLOPs | vs f64 | state dtype |
|---|---:|---:|---|
| f64, shipped spelling | `137,562,584,064` | 1.0000x | float64 |
| f32 as the arm is written | `133,266,036,224` | **0.9688x** | **float64** |
| f32 with the two stats results cast back | `68,782,589,440` | **0.5000x** | float32 |

The recorded 3.1% is reproduced **exactly**; two `.astype` calls recover the full
2x. Divergence from the f64 control on the scored row is `3.3e-13` relative mean
square, eight orders below the error being fought — **on one network only**;
across all 96, and against float32 failure modes, nothing is established.

Consequence for design: the multiplier floor makes the first `219,902,325,555`
counted FLOPs free, and the shipped path spends `137,600,326,656` of them. At f64
the free zone holds `~82.3e9` unused FLOPs — **1.2** dense Kerdock/MUB bases, since
one costs exactly `2**36`. At corrected f32 it holds `~151.1e9` — **2.2** bases.

**The lane is NOT reopened by this.** Reopening requires a STRATEGIC contract with
a stated reopening condition. What is established is narrower and is all that may
be quoted: the cost half of the lane's closure premise was measured on an arm that
was not float32, so the recorded reason no longer stands on its own.

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

**What P1-PRE cost, and the durable lesson:** six independent passes found six
successive instances of a claim outrunning the evidence behind it. D1–D3
themselves survived all six attacks; the overclaim was that a
pinned declaration proves the behaviour named by that declaration. It does not.
A useful check compares an independent expected property with observed behaviour.
Mutation probes add evidence that selected faults are detectable; they are not a
proof that every future edit is safe. The terminal pass therefore pins the exact
witness semantics and platform classification, measures what files/patterns were
actually scanned, and requires the named guard—not merely any non-zero exit—to
fire. It then stops. The repository will not build a second evaluator whose job
is to formally verify the first evaluator.

The historical destructive `git checkout --` path is gone at current HEAD, and
the T6 correction has landed: `local_sim/p1pre_fault_inject.py` **defaults its
transcript to a system temp directory, refuses any `--out` resolving inside the
repository** (tracked or not — the test is containment, not tracking status),
carries the old tracked transcript in its protected hash set, and takes the
untouched-repository measurement **after** the write rather than before it. It is
safe to run with no arguments. Copy its transcript into `experiments/` yourself
if you want it as committed evidence.

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

**There is no trustworthy additive carrier allowance.** Subtracting the 260.7-ms
median suggests ~139 ms, but the cap applies per MLP and that day's 390.6-ms
maximum leaves only **9.4 ms** on its row. Residual is not known to add linearly
between branches. Phase 1 must test the integrated candidate, include the observed
~2.35x session spread in a preregistered tail falsifier, and reject any design that
passes only in a friendly session.

Two cautions on the numbers above, both from an independent Skeptic: the "worst
MLP" is frequently the **first** MLP (a warm-up transient, which still fails an MLP
under a cliff but is not network heaviness), and **n = 20 is thin for a maximum**.

The paired arithmetic score test, common-random-number design, ground-truth noise
reporting, seed ledger, public-CONFIRM rule, and vault discipline remain sound.

## Active execution queue

| order | slice | class | status |
|---:|---|---|---|
| 0 | official-stack/evaluator repair | IMPORTANT | **CLOSED 2026-08-24** — G1–G7 all met, independent ACCEPT recorded |
| 0H | P1-PRE: D1–D3 evaluator hardening | **IMPORTANT** | **CLOSED** — seventh independent pass ACCEPT; 15 named mutations and current durable record verified |
| 0S | first Phase-2 calibration submission | **STRATEGIC** | **REJECTED procedurally at current HEAD** — the candidate is safe, but the full-preflight witness predates later `local_sim/` edits. Correct continuity, produce a clean current-HEAD witness + evidence-only commit, live-recheck account/allowance, obtain focused ACCEPT, then ask for exact-SHA approval. Expected about rank 42; calibration only, never a nominee |
| 0C | P2-CAL: K2 state calibration at 1024x16 | **DESIGN measurement** | **GATE NOT MET (falsifier fired), effect real.** 2.96x held-out for 16,384 FLOPs, but all multiplicative arms are provably one output rescale. Needs a NEW contract + Samuel's prediction before it can ship; C3/C4 (non-commuting corrections) are the live follow-on |
| 1A | verified Kerdock/MUB carrier + grader-legal construction/deployment evidence | **IMPORTANT** | **PROVISIONAL** — 34/34 + 16/16 prove a valid real-MUB carrier and its exact dense cost. Add a stdlib/`fnp` construction path byte-identical to the NumPy reference; remove the memory/setup false-green; reconcile K6 as formally withdrawn; run current-HEAD full preflight/portability and obtain focused ACCEPT. No seed, estimator edit, or truth measurement |
| 1B | cost-indexed structured-carrier DESIGN frontier | STRATEGIC | after 1A ACCEPT; contract and Samuel's restated prediction reviewed before claiming a NEW DESIGN block. Primary estimand is adjusted score over basis count and total measured overhead, not reproduction of the 16-basis fingerprint |
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

S2 is still a valid historical prediction for **Kerdock16 versus matched-row IID**,
but it is not a promotion gate. Dense Kerdock16 must improve raw MSE by **more than
5x versus K2p1 merely to beat the current baseline score**, before adding any
analytic/control cost; matching the live `9.50e-9` leader would require roughly
**213x**. The current leader's raw/adjusted row implies multiplier about `0.306`,
not `0.5`, so the old 16-basis cost fingerprint is evidence for the carrier family,
not evidence that 16 is the winning basis count.

## Closed and open lanes

| lane | state | scope |
|---|---|---|
| basic K2/K2p2 | control / conservative branch | sound but not winning alone |
| higher Gaussian covariance-map orders | low value | measured plateau near 1.052x |
| marginal ReLU mean Edgeworth | closed | capped near 1.047x in CP3R |
| recurrent augmented K2 joint state | **open** | never tested by CP3R |
| cost-indexed Kerdock/MUB + exact radial | **highest priority** | valid carrier; test the adjusted-score frontier over basis count, with 16 only a diagnostic endpoint |
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
- Do not submit without Samuel's explicit approval **naming the exact estimator
  SHA**, and only after the **six-item operational gate** of the Track-0S
  submission contract: hash + archive identity, no dirty TRACKED files,
  `whest validate`, allowance, Samuel's SHA-naming approval, and
  **`preflight.ps1 -Full` GREEN 16/16 with a committed transcript grepped for
  its own verdict line**. `VALIDATION.md` §7 always required that last one;
  the contract first omitted it, and an independent review caught the
  omission.
- After every IMPORTANT or STRATEGIC slice, obtain an independent Skeptic verdict
  from raw artifacts before believing or propagating the result.
- Keep `CURRENT_STATE.md` updated when a lane, baseline, live rule, or next action
  changes. Do not stack another override on top of this file.
