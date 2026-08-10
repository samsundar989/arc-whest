# next-slice — arc-whest

*Canonical workflow. Agent-independent. Adapters: `.claude/skills/next-slice/`,
`.agents/skills/next-slice/`.*

**`next-slice` does not mean "write more code."** In this repository the last five
sound moves were: a provenance repair, an invariant suite, a constants audit, a
pre-registered gate that failed, and a parity proof. Only one of them shipped a line
into `estimator.py`. Selecting the right *kind* of move is most of the value.

## Step 1 — inspect evidence before proposing anything

Read `docs/CURRENT_STATE.md`, the tail of `experiments/LOG.md`, and — if a lane or a
number is involved — `docs/GOTCHAS.md`. Check `git status`. Then answer five
questions explicitly, in the output:

1. **What is the current bottleneck?** (cite the measurement, not an impression)
2. **What is the important uncertainty?**
3. **Do we already know enough to act?** If yes, research is waste.
4. **Is evaluation trustworthy enough to judge the result?** In this repo:
   accuracy is **E2**, cost is **E1**. A move judged on a cost delta is being judged
   by the weaker instrument — say so.
5. **Is research actually necessary,** or would repository evidence answer it?

If a proposed move touches a lane marked CLOSED in `docs/CURRENT_STATE.md`, stop and
say so. Reopening needs **new external evidence** — a competitor's telemetry, an
organizer ruling, a paper — not a new idea about the same evidence.

## Step 2 — route the move

Emit exactly one route, named, with one sentence of justification.

| route | when | what happens |
|---|---|---|
| **LOCAL_EXECUTION** | we know what to do and the evaluator can judge it | proceed to Step 3 |
| **LOCAL_DIAGNOSTIC** | the mechanism is unclear but the repo can answer it | probes, ablations, traces, `board_forensics`, `twin_diff`, existing artifacts under `experiments/`. Design blocks only, no decision block |
| **EVALUATION_REPAIR** | the instrument cannot judge the thing we care about | **outranks optimization.** The live example: cost is E1 because local residual is ~7× the grader's. Fixing that beats another accuracy experiment |
| **STRATEGIC_RESEARCH** | bottleneck known, repo evidence exhausted, external prior art would change the decision | emit a Strategist Packet (below). Do not start research here |
| **STRATEGIC_DECISION** | evidence exists but the direction needs deeper reasoning or competing mechanisms weighed | emit a Strategist Packet |

Valid move types include: diagnostic · instrumentation · evaluation repair · baseline
reproduction · error analysis · targeted research · hypothesis test · implementation ·
ablation · benchmark · data validation · security/integrity check · architecture
experiment. Name which one.

**Do not escalate to research by default.** Research is an escalation layer. This
project's plateau is not from lack of ideas.

## Step 3 — classify, then delegate

Use the repository's own definitions (`AGENTS.md` § *Work classification*), which are
more specific than any generic scheme:

### ROUTINE
Docs, figures, board snapshots, report fetching, tests, lab scripts touching no seed
block and no shipped file.

→ **Follow the `routine` workflow** (`ai/workflows/routine.md`). No strategic ceremony.

### IMPORTANT
Changing *the instrument*: `local_sim/harness.py`, `mint_provenance.py`,
`edge_cases.py`, `seed_blocks.py`, `vault.py`, a probe's math, grader-report parsing,
or anything under `ai/workflows/` and `scripts/`.

→ Concise plan → **written prediction before running** → implement → `.\scripts\preflight.ps1`
must be GREEN → targeted evaluation → LOG entry with a Knowledge Delta.

Instrument changes are IMPORTANT rather than ROUTINE because a bug here does not
produce an error — it produces a plausible wrong number that survives review.

### STRATEGIC
`estimator.py`; spending a decision block; a vault touch; a submission; changing a
gate, a residual model, or the three-tier protocol; a write-up claim; reopening a
closed lane.

→ **Follow `ai/workflows/slice-contract.md`.** Do not restate or re-derive its
steps here — it owns the pre-registration protocol, the seed-block allocation rules,
the residual-model freeze, Samuel's prediction checkpoint, and the Evidence Packet
format. `next-slice` decides *that* a strategic slice is right; `slice-contract`
governs *how* it runs.

## Step 4 — execute or hand off

- ROUTINE / IMPORTANT / LOCAL_DIAGNOSTIC / EVALUATION_REPAIR / LOCAL_EXECUTION:
  do the work now, then tell Samuel to run `review-slice` if the slice was IMPORTANT
  or STRATEGIC.
- STRATEGIC: run it under `slice-contract`, which ends at the Evidence Packet.
  `review-slice` closes the loop.
- STRATEGIC_RESEARCH / STRATEGIC_DECISION: emit the packet and **stop**. Do not
  implement against a question you have just declared unanswerable here.

---

## Strategist Packet

Emitted for STRATEGIC_RESEARCH and STRATEGIC_DECISION. It is addressed to **the
strongest reasoning capability available to Samuel at that moment** — a role, not a
product. The recipient may be a different vendor, a different model generation, or a
human collaborator; the packet must work for all three.

Self-contained: **assume the recipient has no repository access and no shared chat
history.** Never open with "how can we improve this project" — a packet without a
specific strategic question is a request for noise.

```markdown
# Strategist Packet — arc-whest

## Objective
Predict every neuron's E[post-ReLU activation] of a random He-init ReLU MLP
(256 wide × 32 deep, no biases) under x ~ N(0,I), from weights alone.
Score = final_layer_MSE × max(0.1, C/B), C = counted FLOPs + 1e11 × uninstrumented
seconds, B = 2.72e11. Decided by a PRIVATE RE-RUN of one designated submission on
fresh networks. Co-equal second objective: the contribution prize (write-up).

## Current architecture relevant to this question
The shipped path and the specific symbols involved — estimator.py:477 predict,
:574 _closure, :598 _alignment_basis, :632 _slice_weights, plus the constants that
matter (_PRUNE_TAU=-2.5, _BASIS_RANK=32, N=16,384). Include only what bears on the
question.

## Validated baseline
Graded, vault, and CONFIRM numbers with their tiers and dates. State the
ground-truth-noise floor for each. Never quote a local absolute residual (this
machine bills ~7× the grader).

## Observed bottleneck
With the measurement that established it.

## Evidence
Tag CONFIRMED / INFERRED / SPECULATIVE. Include the seed block and the statistic
(paired geometric mean of per-MLP ratios, n shifts, CI).

## Experiments already tried and rejected
The closed-lane table from docs/CURRENT_STATE.md, with each gate and its result.
This is the most valuable section — it prevents the recipient re-proposing a
mechanism already measured and killed.

## Current hypotheses
## Existing relevant internal research
Pointers into experiments/LOG.md, docs/BENCHMARK.md, docs/MOONSHOT.md.

## Exact strategic question
One sentence. Answerable. Decision-relevant.

## Research requested
## Constraints
- estimator.py: only flopscope, whestbench, stdlib. Immutable grader arrays.
- Determinism from mlp.seed. No wall-time-accounting exploits, ever.
- Cost evaluation is E1: residual wall time is not measurable locally. Any proposal
  whose value rests on a residual claim must state what would verify it.
- flopscope pricing: matmul (2K−1)·M·N by shapes; gather/where 4/elt;
  transcendentals 16/elt; no dtype discount.
- The metric has a heavy tail: fresh-MLP max/median ≈ 12.9×.

## Required output
- confirmed evidence / inferred conclusions / speculative ideas, separated;
- up to three MECHANISTICALLY DIFFERENT approaches;
- compatibility with the current architecture;
- evidence for and against each;
- cost and complexity;
- a falsifier for each;
- the cheapest discriminating experiment;
- a recommended Slice Contract in this repo's format. The template is reproduced
  in full in ai/workflows/slice-contract.md § Step 3; paste it into the packet so
  the recipient does not need repository access.
```

Bring the returned Slice Contract back through `next-slice`, which will re-verify its
assumptions against the code before anything is built. External research is an input
to the belief system, not an instruction oracle.
