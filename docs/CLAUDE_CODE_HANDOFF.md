# Claude Code handoff — open the two Phase-2 candidate paths

Copy everything under **Prompt** into a new Claude Code session started at the
repository root. This handoff reflects the live-board and evaluator audit of
2026-08-31. Repository evidence wins if it becomes stale.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge
2026 entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

Your objective is to turn the strongest open Phase-2 idea into a grader-legal,
truth-ready research arm. Continue through safe truth-free work while every gate
remains green. Do not spend a seed block, touch the vault, change `estimator.py`,
or submit unless the repository contract and Samuel's explicit approvals permit
that later step.

You do **not** have permission to call `whest submit`, designate entries, touch the
vault, or edit `docs/HUMAN_MODEL.md`.

### 0. Orient from durable state

Read, in order:

1. `AGENTS.md` and `CLAUDE.md`;
2. `docs/CURRENT_STATE.md`;
3. the bottom of `experiments/LOG.md` far enough to understand the latest results,
   corrections and independent reviews;
4. `docs/GAME_PLAN.md`;
5. `docs/PHASE2_FRONTIER_AUDIT_2026-08-31.md`;
6. `docs/VALIDATION.md`, `docs/GOTCHAS.md`, `docs/KERDOCK_MUB_SPEC.md`, and
   `docs/SKEPTIC.md`.

Inspect the current branch, HEAD, worktree, version archives, official installed
stack, seed ledger, public submission records and ship/archive identity. Preserve
unrelated user changes.

The current evaluator fact that must not regress is: the installed official stack
sets `wall_time_limit_s = 120` and applies it **per MLP/predict call**, not over the
100-network suite. This exact ceiling is not exposed in a graded config field;
public ledgers prove that at least `104.766 s/MLP` passes, consistent with the
installed value. The installed residual cliff is `0.4 s` per MLP. Continue to
report both provenance and headroom, and never claim that Windows enforces 8 GiB.

### 1. Respect the closed lanes

Do not reopen either lane without new external evidence that changes its mechanism:

- full-tower exact K3/W-EFF was measured at about `1.0732 B` FLOPs and failed its
  own `<=0.35 B` affordability screen;
- late-window/age-window K3 has provable adjusted-score floors `2.09x`–`4.65x`
  worse than v2.7 because its K2 prefix is already too inaccurate.

The optional live-shape `k3pre` rerun is attribution cleanup, not a blocker and not
a candidate. Do not let it delay the open paths.

### 2. First slice — grader-legal carrier construction (IMPORTANT, truth-free)

The current P1A Kerdock/MUB mathematics is a research reference, not deployable
code: its probe constructs arrays with NumPy before wrapping them in `fnp`. The
first bounded task is to close that legality and deployment gap without labels.

Before generating result data:

1. write a complete IMPORTANT contract and prediction in `experiments/LOG.md`;
2. commit the contract so its timestamp precedes the evidence;
3. state explicitly that no seed-ledger block, public truth or vault is involved.

Implement a deterministic stdlib/`flopscope.numpy` constructor for the frozen
Kerdock/MUB object. It may use `setup(context)` where appropriate, but every array
used by `predict` must be legal in the official sandbox. Preserve the component
specification; do not silently change basis ordering, Gray convention, scaling,
antipodes or radial normalisation.

Required evidence:

- byte-identical carrier rows versus the frozen NumPy reference at supported small
  sizes and the live width;
- exact exhaustive integer properties already required by
  `docs/KERDOCK_MUB_SPEC.md` at reduced sizes;
- deterministic bytes across repeated runs and a fresh process;
- exact or tolerance-frozen live propagation parity with the research reference;
- zero `__bool__`, `__float__` and `__int__` array materialisations in `predict`;
- official flopscope operation ledger, counted total and multiplier;
- `setup` duration/headroom, first-predict warm-up, subprocess residual headroom,
  and explicit peak/single-array memory accounting;
- focused tests for row order, antipodal pairing, chunk boundaries, dtype and
  radial scaling;
- shape-generic behaviour rather than width/depth constants in candidate code.

Avoid the unresolved large stacked-array failure mode reported on the forum until
a smaller spelling has official-subprocess evidence. Do not infer the leaders'
private sampler from their FLOP totals: roughly `5.5e11` FLOPs identifies an
operating envelope shared by several possible 16k-row methods.

Run the default preflight and portability check. Then launch a fresh independent
Skeptic context that has never seen the Builder's reasoning. Give it the contract,
diff, protocol, commands and raw artifact paths—not a prose summary. Resolve and
record its verdict before consuming this result.

### 3. Second carrier slice — objective and frozen panel (IMPORTANT, truth-free)

Only after the construction slice is synthesised, write a separate contract for
the selection objective. Implement the deep-ReLU kernel energy proposed in
`GAME_PLAN.md` and validate its algebraic nulls, including global-rotation
invariance.

Before any labels, freeze:

- the ordered nested Kerdock/MUB basis/sample counts;
- independent rotations or scrambles, if any;
- IID antithetic, ordinary orthogonal and scrambled-low-discrepancy controls;
- row counts and counted-cost matching;
- the deterministic rule that maps the proxy to one policy;
- the later paired adjusted-score gate and falsifier.

This IMPORTANT slice proves the instrument only. A later STRATEGIC contract,
Samuel prediction, preregistered DESIGN selection sample, and disjoint DESIGN
replication are required to learn whether the proxy predicts finite-network MSE.
Never fit basis indices, weights or blends on the public mini.

### 4. Parallel analytic proofs — separate contracts, no labels

After the first carrier slice is stable, two bounded analytic investigations may
run independently. Do not combine them into one contract or let one result tune
the other.

**AUGMENT parity and price.** Port ARC's unfactored AUGMENT recurrence at small
shape and compare every retained state component at every layer. Freeze the exact
upstream parity target first: `factor=True, kind=AUGMENT` drops diagrams and is not
equivalent. Only after parity, price the actual f32 spelling and its checkpoint
use. Standalone AUGMENT and checkpoint initialization are separate estimands.

**Query-directed/adjoint K3.** At small shape, linearise the final means with
respect to layer-local third-cumulant sources. Backpropagate final-output adjoints
and contract each local K3 source directly into those queries instead of storing
the recurrent K3 tower. Compare with exact K3 for layerwise source contributions,
final drift explained, cosine/sign agreement, source-scaling linearity, adjoint
rank growth, FLOPs and memory. Kill the lane if covariance feedback or adjoint rank
restores the full-tower scaling.

Each is IMPORTANT instrument/probe work: contract and prediction first, default
preflight green, knowledge delta in `LOG.md`, and a fresh independent Skeptic
before belief. Neither consumes truth until its truth-free mechanism, parity and
cost gate pass.

### 5. Strategic decision only after a truth-free path passes

No suitable later blocks currently exist: P01 is DESIGN with 432 unused seeds, but
it is neither an independent replica nor a decision block. After the truth-free
gates determine the power requirement and **before any new labels**, run a separate
IMPORTANT instrument slice to register at least one disjoint Phase-2
DESIGN-replication block and one distinct Phase-2 SINGLE_USE DECIDE block in
`seed_blocks.py`. Take Samuel's prediction first, test the ledger mutations, run
default preflight, record the knowledge delta, and require an independent Skeptic.
Do not invent identifiers or sizes in advance of that contract.

Next, under a STRATEGIC DESIGN contract, use a preregistered selection sample and
require the frozen policy to reproduce on the new disjoint DESIGN block. Synthesize
that result and obtain an independent Skeptic before it can feed a decision.

Do not claim the SINGLE_USE block merely because code runs or because DESIGN
passes. First freeze the complete candidate, control, coefficients, carrier policy,
cost model, fallbacks and exact estimand in a Rule-1 STRATEGIC contract; obtain
Samuel's three short predictions; commit the contract; only then claim the already
registered Phase-2 SINGLE_USE block through `local_sim/seed_ledger.py`.

Compare against exact v2.7 with common random networks. Promotion authority is
only `local_sim/competition_score.py::decide(gate=...)`, using the paired
arithmetic adjusted-score difference and a studentised-t or basic-bootstrap
interval. Percentile bootstrap and BCa are forbidden. Every MSE must carry its
ground-truth noise floor. The current 24-network `2**26` Phase-2 truth set has an
average floor near `1.3e-9`; write an explicit power plan before deciding whether
it is sufficient for a frontier-scale gate.

Report counted FLOPs including assembly and the server `+2**24` offset near a
boundary, per-MLP residual headroom, per-MLP wall headroom, failures, tails and
memory accounting. The downloadable public split is CONFIRM only and may fit no
constant. Obtain the required fresh independent Skeptic verdict from raw artifacts
before promotion or synthesis.

### 6. Candidate architecture and nomination discipline

The two intended nominee families are deliberately different:

```text
Nominee A: grader-legal structured angular carrier
           + exact radial integration
           + frozen deterministic policy/calibration

Nominee B: corrected-f32 K2 / faithful AUGMENT prefix
           + supported query-directed K3 response
           + tiny neuron-specific shape correction only if separately reproduced
```

A hybrid is allowed only if its interaction and diversity cost were preregistered;
do not collapse the two nominees into correlated variants by default. Never add a
module because it is attractive on paper. Every shipped module needs its own green
evidence path.

Only after formal promotion may you create a new immutable `estimator_v*.py`
archive and place identical bytes in `estimator.py`. Then require exact archive
identity, official validation, default and full preflight with a durable `GREEN
16/16` transcript, clean official Phase-2 mini with zero failures, three
official-subprocess 100-MLP soaks, zero scalar materialisations, and real Linux
8-GiB enforcement or an explicit deployment blocker.

Stop with an exact-SHA approval packet containing git commit, estimator SHA-256,
FLOPs/multiplier, decision artifact and interval, ground-truth noise, per-MLP wall
and residual headroom, memory status, failure counts, Skeptic verdict and existing
submission IDs `#328662` and `#328664`. Ask Samuel to approve or refuse that exact
hash. Never submit without his explicit active-session approval.

### 7. Continuity

At every stop, update `experiments/LOG.md` and `docs/CURRENT_STATE.md` with the raw
artifact paths, exact result, changed paths, cost/memory/failure evidence, Skeptic
verdict and next action. Never allow one IMPORTANT or STRATEGIC result to flow into
the next implementation without synthesis.

Do not modify the quarantined write-up, publish private benchmark intelligence,
post to the discussion board, touch a vault, designate submissions or force a
third public entry. A failed preregistered gate is a valid completion state.
