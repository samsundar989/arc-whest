# Claude Code handoff — implement the Phase-2 architecture to a runnable state

> **Historical prompt; do not execute its old work ordering.** The active September 12 handoff is [CODEX_EXECUTION_HANDOFF](CODEX_EXECUTION_HANDOFF.md), with [CURRENT_STATE](CURRENT_STATE.md), [GAME_PLAN](GAME_PLAN.md) and [STRATEGIST_RESPONSE](STRATEGIST_RESPONSE_2026-09-12.md). Samuel has reopened eligibility review and candidate preparation; submission still needs separate approval. Preserved text below does not override that current scope or execution order.

Copy everything under **Prompt** into a new Claude Code session started at
`C:\Users\samsu\Documents\arc-whest`.

This is a completion prompt, not a request for another plan. The required session
outcome is a runnable, tested, fully priced research architecture at the live
`1024 x 16` shape, plus every safe system needed to evaluate it. Statistical
promotion and submission remain separate evidence/human gates.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge
2026 entry. Work in:

```text
C:\Users\samsu\Documents\arc-whest
```

Your mission is to implement `docs/GAME_PLAN.md` through its complete safe,
truth-free engineering state during this session. Do not return a restatement of
the plan. Build the modules, tests, candidate-certification path, and integrated
research candidate; run the gates; resolve defects; obtain independent reviews;
and leave durable commits and evidence.

The desired end state is:

- the grader-legal structured-carrier branch works end to end;
- faithful AUGMENT and query-directed K3 exist as exact-oracle-tested analytic
  branches or have been killed by their preregistered falsifiers;
- all surviving branches plug into one shape-generic research candidate API;
- at least one end-to-end candidate runs at `1024 x 16`, is fully metered, and
  passes candidate certification without reading new truth;
- systems, manifests, test plumbing, scalar-synchronization checks, and memory
  accounting are complete enough that a later strategic decision is not blocked by
  unfinished engineering.

Do not claim a win from truth-free tests. A branch killed cleanly is useful. The
non-negotiable engineering outcome is a working integrated candidate from the
surviving modules, not a forced positive research result.

### Authority and prohibited actions

You may modify research modules under `local_sim/`, tests, candidate-certification
infrastructure, documentation, and experiment records as required by the plan.
Classify the work exactly as `AGENTS.md` requires.

You do **not** have authorization to:

- edit `docs/HUMAN_MODEL.md`;
- touch or evaluate a vault;
- claim or consume any seed/truth block;
- fit or select anything on the public mini;
- modify `estimator.py` or any frozen `estimator_v*.py`;
- submit, designate, post, or message anyone externally.

If Samuel explicitly authorizes a later strategic step in the active session,
follow the repository contract for that exact step. Do not infer that authority
from this implementation prompt.

### 0. Orient and establish a clean execution base

Read completely, in this order:

1. `AGENTS.md` and `CLAUDE.md`;
2. `docs/CURRENT_STATE.md`;
3. the bottom of `experiments/LOG.md` through the latest accepted audit;
4. `docs/GAME_PLAN.md`;
5. `docs/VALIDATION.md`, `docs/GOTCHAS.md`, `docs/KERDOCK_MUB_SPEC.md`,
   `docs/SKEPTIC.md`, and
   `docs/PHASE2_FRONTIER_AUDIT_2026-08-31.md`.

Inspect HEAD, worktree ownership, ship/archive identity, installed official stack,
seed ledger, available GPU/WSL/Linux paths, and existing relevant modules. Preserve
unrelated user changes. Confirm that historical plans are not being followed.

Create a concise task board from `GAME_PLAN.md` WP0–WP5. Continue working while
any safe package remains. Do not stop merely to report that a later strategic step
needs permission; finish all independent truth-free work first.

For every IMPORTANT slice:

1. append and commit its question, evidence, mechanism, intervention, predictions,
   falsifier, scope, and verification plan before result data;
2. make the smallest coherent implementation;
3. run focused tests, default preflight, and portability;
4. write the knowledge delta and raw artifact paths;
5. launch a fresh independent Skeptic that has not seen Builder reasoning, passing
   the contract, diff, commands, protocol, and raw artifacts;
6. resolve its blockers and commit before another implementation consumes the
   result.

Never combine unrelated IMPORTANT changes just to save commits.

### 1. WP0 — eliminate systems and test blockers first

Complete these as separately classified slices:

**A. Phase-2 manifest repair**

- update the executable live tag/pin from the durable comparison at
  `experiments/2026-08-31_phase2_tag_drift/source_manifest.tsv`;
- preserve both commit IDs and the canonical 70-row digest;
- distinguish unchanged source Parquet objects from added prepared artifacts;
- add a regression proving metadata/prepared-only tag movement does not invalidate
  source-byte-dependent evidence;
- run `local_sim/phase2_checklist.py` and the relevant provenance tests.

**B. Enforcement-description repair**

- correct stale preflight subtest labels and the `VALIDATION.md` step mapping;
- do not change executable semantics while repairing descriptions;
- mutation-test any guard whose stated coverage changes.

**C. Candidate-selectable certification**

Add a read-only way to certify a candidate path before it is promoted into
`estimator.py`. Prefer a separate candidate command or a parameterized path whose
default remains the shipped file. It must not weaken the existing ship-file gate.
Cover:

- official `whest validate`;
- package/config identity;
- shape and exact layer-1 checks;
- deterministic fresh-worker execution;
- immutable inputs;
- flopscope operation totals and multiplier;
- residual and per-MLP wall telemetry;
- scalar-materialization/synchronization detection;
- peak and single-array memory accounting;
- clean subprocess smoke and deliberate failing mutations.

**D. Real memory enforcement**

Detect WSL/Linux availability. Implement and, if available, run the exact 8-GiB
enforcement path in a child process. Do not call Windows accounting an enforcement
pass. If no Linux execution environment exists, leave one copy-paste command,
expected artifact path, and explicit deployment blocker; continue all other work.

WP0 exits only when default preflight and portability are green and research
candidates can be certified without modifying the shipped file.

### 2. WP1 — grader-legal structured carrier

Treat `local_sim/kerdock_mub.py` and `docs/KERDOCK_MUB_SPEC.md` as the frozen
mathematical reference. Do not rewrite the reference to make parity easier.

Implement a stdlib/`flopscope.numpy` constructor and streamed full-network forward
path, preferably in:

```text
local_sim/p2_carrier_fnp.py
local_sim/test_p2_carrier_fnp.py
```

The deployed-form path must not construct with raw NumPy and wrap the result.
Preserve basis ordering, Gray convention, normalization, antipodal pairing, and
exact Gaussian radial scaling.

Required proof:

- byte parity at supported reduced sizes and live width where representation
  permits it;
- exhaustive defining Kerdock/MUB properties at reduced sizes;
- deterministic bytes in repeated and fresh processes;
- parity of every propagated layer with the research reference;
- chunk-boundary equivalence, including odd/tail chunks;
- dtype, scaling, ordering, antipode, and seed mutations that fail their named
  tests;
- `mlp.seed`-derived randomization only, if randomization is enabled;
- zero `bool`/`float`/`int` materializations in `predict`;
- setup, first-call, residual, wall, FLOP ledger, multiplier, and explicit memory
  evidence at `1024 x 16`.

Integrate the carrier-only path into the common research API immediately; do not
wait for labels or for the analytic branch.

### 3. WP2 — truth-free carrier objective and frozen panel

Implement the deep-ReLU kernel discrepancy in:

```text
local_sim/p2_carrier_objective.py
local_sim/test_p2_carrier_objective.py
```

Prove algebraic nulls, determinism, global-rotation invariance, permutation
behavior, and sensitivity to known-bad carriers. Before reading any new labels,
freeze and content-address:

- nested 4/6/8/10/12-basis policies;
- equal-row IID antithetic, orthogonal, scrambled-low-discrepancy, and randomized
  orientation controls;
- row-count and counted-cost matching;
- deterministic proxy-to-policy selection;
- later DESIGN falsifier and score gate proposal.

Eight bases is only a panel member. Do not infer it from leaderboard FLOPs. Proxy
success does not select a submission candidate.

### 4. WP3 and WP4 — run the analytic proofs without holding up the carrier

Once WP0 is stable, use separate implementation contexts for WP3 and WP4 if Claude
Code can keep their reasoning and diffs independent. Integrate only after each has
its own synthesis and Skeptic verdict.

**WP3: faithful AUGMENT**

Create:

```text
local_sim/p2_augment.py
local_sim/test_p2_augment.py
```

Freeze the upstream unfactored AUGMENT target first. Compare every retained state
component and every layer at small shapes. `factor=True, kind=AUGMENT` is not an
acceptable reference because it drops diagrams. Test symmetry conventions, float32
and float64 behavior, shape scaling, and standalone versus checkpoint use. Price
the actual `fnp`-compatible float32 spelling before labels.

**WP4: query-directed/adjoint K3**

Create:

```text
local_sim/p2_adjoint_k3.py
local_sim/test_p2_adjoint_k3.py
```

Implement:

- exact small-shape layer-local K3 source extraction;
- backward final-output query propagation;
- direct source-to-query contractions;
- query-rank controls and explicit state/FLOP/memory formulas;
- per-layer and summed response diagnostics.

Compare with exact full K3 on truth-free small-shape states. Preregister and test
source-scaling linearity, sign/cosine agreement, explained squared drift,
covariance-feedback sensitivity, query-rank growth, and counted cost. Kill the lane
if agreement is poor or scaling reconstructs the full tower. Do not fit a rank or
coefficient to hide a failed mechanism.

If AUGMENT or adjoint K3 is killed, preserve its smallest falsifying artifact and
continue integrating the surviving branch. Do not reopen W-EFF or late-window K3.

### 5. WP5 — deliver the working integrated research architecture

Create the common research candidate and focused tests, preferably:

```text
local_sim/p2_candidate_architecture.py
local_sim/test_p2_candidate_architecture.py
```

Expose explicit configurations for:

- calibrated K2 control;
- carrier only;
- AUGMENT only, if supported;
- AUGMENT plus adjoint K3, if supported;
- one zero-weight identity hybrid and one frozen hybrid placeholder whose
  coefficient remains unset until a legal DESIGN contract.

The API must share seed derivation, chunking, assembly, fallback logic, and
telemetry. Do not duplicate branch math inside the integration layer. Make the
implementation convertible to one self-contained estimator file without importing
`local_sim`, but do not create a frozen estimator archive yet.

Mandatory integrated tests:

- `1024 x 16` is exercised first, plus smaller shape-generic cases;
- exact layer-1 closed form and hidden-layer permutation equivariance;
- repeated-run and fresh-worker determinism;
- branch isolation and zero-weight blend identities;
- immutable grader arrays and finite output of exact shape;
- zero scalar synchronizations;
- chunk boundaries, first-call behavior, and pathological/degenerate networks;
- official counted FLOPs and multiplier per configuration;
- residual, wall, peak memory, and single-array accounting;
- subprocess lifecycle with zero failures;
- deliberate mutations proving every claimed guard can fail.

Use the WP0 candidate-certification path on every live-shape configuration. Fix
engineering failures now. Do not defer them to promotion.

WP5 is complete only when at least one full research candidate executes at live
shape, returns all layer means, has a stable cost/memory envelope, and passes every
applicable truth-free certification check.

### 6. Human and statistical boundary

No Phase-2 SINGLE_USE decision block or independent DESIGN-replication block exists.
P01's unused DESIGN seeds cannot serve as an independent replica.

After WP0–WP5 and a frozen power analysis, prepare—but do not execute without the
required authority—the IMPORTANT block-allocation contract for:

- a disjoint Phase-2 DESIGN-replication block; and
- a distinct Phase-2 SINGLE_USE DECIDE block.

If Samuel is present and supplies the required prediction/authorization, you may
continue through the exact `GAME_PLAN.md` sequence. Otherwise stop only this
truth-consuming branch, not the remaining engineering.

Any later decision uses common random networks and only
`competition_score.decide(gate=...)`, with paired arithmetic adjusted-score
difference and a studentised-t or basic-bootstrap interval. Percentile bootstrap
and BCa are refused. Every MSE carries its ground-truth noise floor. Public mini is
CONFIRM only.

`estimator.py`, archive creation, full deployment, CONFIRM, vault, designation,
and submission remain separate STRATEGIC or human-only steps. Never submit without
Samuel approving the exact hash in the active session.

### 7. Parallelism and persistence

Use parallel subagents only for bounded, independent work such as:

- inspecting upstream AUGMENT formulas while the Builder closes WP0;
- implementing WP3 while another fresh context prototypes WP4;
- running slow shape/cost tests while the Builder works on independent docs/tests;
- cold Skeptic review after a package is finished.

Do not let multiple agents edit the same files concurrently. The primary Builder
owns integration, contracts, and final synthesis.

Continue until the completion definition is met or an external requirement truly
blocks every remaining safe task. If a command is slow, monitor it. If a test
fails, diagnose and repair it. Do not end with “the next agent should implement.”

### 8. Required final handoff

Before ending:

1. update `experiments/LOG.md` with every contract, result, raw artifact path,
   cost, memory, failure evidence, and Skeptic verdict;
2. update `docs/CURRENT_STATE.md` so another agent needs no transcript;
3. run focused tests, default preflight, portability, and `git diff --check`;
4. commit coherent completed slices and leave the worktree clean unless preserving
   named user changes;
5. report:
   - working configurations and exact commands;
   - killed branches and falsifiers;
   - live-shape FLOPs, multiplier, wall/residual, and memory;
   - tests and gate results;
   - files and commits;
   - the single next strategic decision or external blocker.

Do not post, submit, designate, touch a vault, or edit the human model.
