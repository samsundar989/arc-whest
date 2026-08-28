# Claude Code handoff — build the strongest supportable Phase-2 candidate today

Copy everything under **Prompt** into a new Claude Code session started at the
repository root. It is intentionally completion-oriented: continue through
implementation and validation while every gate remains green, and stop only at a
falsifier or a decision reserved for Samuel.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge
2026 entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

Your objective is to produce the strongest **evidence-backed, grader-legal
Phase-2 submission candidate supportable today**, following the target architecture
in `docs/GAME_PLAN.md`. Do not stop after proposing code or writing another plan.
Implement, test, measure, review, integrate, and certify as far as the registered
gates allow. “Completion” is either:

1. a materially stronger frozen candidate plus an exact-SHA approval packet for
   Samuel; or
2. an evidence-backed kill at the first failed gate, fully recorded, with no weak
   submission made for the sake of the date.

You do **not** have permission to submit. Never call `whest submit`. If every gate
passes, stop with a concise request for Samuel to approve or refuse the exact
candidate SHA.

### 0. Load the repository contract and establish the starting point

First read, in this order:

1. `AGENTS.md` and `CLAUDE.md`;
2. `docs/CURRENT_STATE.md`;
3. the bottom of `experiments/LOG.md` far enough to understand the latest contract,
   result, corrections, and independent review;
4. `docs/GAME_PLAN.md`, especially §4;
5. `docs/PHASE2_FRONTIER_AUDIT_2026-08-28.md`;
6. `docs/VALIDATION.md`, `docs/GOTCHAS.md`, and `docs/SKEPTIC.md`.

Inspect the current branch, HEAD, worktree, version archives, installed official
stack, seed ledger, public submissions, and current estimator/archive identity.
Repository evidence wins if any sentence in this prompt has become stale. Preserve
unrelated user changes. Never edit `docs/HUMAN_MODEL.md`.

Do not treat archive stubs or the 2026-08-24/26 reports as action plans. The only
active status/strategy files are `CURRENT_STATE.md` and `GAME_PLAN.md`.

### 1. Mandatory first interaction: capture Samuel's predictions

Before profiling, benchmarking, minting networks, or generating result data, ask
Samuel these three short questions and accept “I don't know” as an answer:

1. At what integrated flopscope cost do you expect an exact W-EFF K3 path to land:
   `<=0.15 B`, `0.15–0.35 B`, or `>0.35 B`?
2. What is the smallest final exact-K3 window you expect to imitate full K3 well
   enough to remain viable: `m=2`, `4`, `6`, `8`, or none?
3. Will you adopt today's proposed minimum submission-candidate bar: at least a
   `2x` adjusted-score point improvement over v2.7 with the valid paired interval
   excluding zero? If not, ask him to state the bar now.

Append the answers verbatim to the first contract. Do not edit
`docs/HUMAN_MODEL.md`.

### 2. W-EFF exact-K3 cost proof — IMPORTANT, truth-free

Write the complete Rule-1 contract in `experiments/LOG.md` and commit it **before
result data exists**. The isolated question is whether the exact factorized-K3
mathematics can be spelled cheaply enough under flopscope 0.12. This is not an MSE
selection experiment and spends no truth block.

Use `local_sim/p2_pa_kprop.py`, `local_sim/p2_pb_rank.py`, the upstream ARC
factorized implementation they load, and the frozen P-A/P-B artifacts as the
reference. Identify the two operations responsible for the measured `95.35%` cost.
Keep fresh identity/one-hot factor modes symbolic; replace transformations of
identity with reuse/indexed operations; fuse factor transforms with the exact
diagonal-slice queries where algebra permits; separate fresh sparse blocks from
dense aged blocks.

Required evidence:

- exact or explicitly tolerance-bounded layerwise parity with the unmodified dense
  reference at several small shapes and depths;
- parity of final predictions, retained factor rank, dtype and checkpoint state;
- an official flopscope 0.12 operation ledger for one representative width-1024
  layer and a defensible whole-path projection including casts, assembly and
  materialization;
- zero `__bool__`, `__float__`, or `__int__` array materializations inside the
  priced path;
- counted-FLOP total and multiplier, subprocess residual headroom against the
  worst observed run, and memory accounting without claiming Windows enforcement;
- focused tests that would fail if the sparse shortcut dropped, duplicated, or
  permuted a factor block.

The preregistered affordability screen is an integrated `<=0.35 B` path. Preserve
the exact K3 output; an approximate result belongs to the next slice. If parity
fails, the saving uses uninstrumented arrays, or the measured spelling cannot
plausibly meet that screen, kill the full-tower W-EFF port and move only to the
late-window diagnostic below.

Run the default preflight and portability check. Then launch a **fresh independent
Skeptic context that has never seen your reasoning**, using `docs/SKEPTIC.md` and
giving it the contract, diff, protocol, commands, and raw artifacts—not a prose
summary. Record and resolve its verdict before today's next implementation consumes
the result.

### 3. Layer-age and late-window K3 — IMPORTANT, label-free

Write and commit a separate IMPORTANT contract after the W-EFF synthesis and
review. Freeze these arms before running them:

- full uncapped factorized K3 control;
- K2 prefix followed by exact K3 for `m in {2,4,6,8}`;
- zero, faithful augmented-K2 trace/harmonic, and any already-supported structured
  checkpoint initializers;
- a matched-retained-rank arbitrary-cap control.

Use unregistered probe networks and no labels. Do not claim a seed-ledger block.
Attribute full-K3 output and diagonal-slice state by factor-block age before
truncating. For every arm report final prediction RMS drift `D`, layerwise
diagonal-slice drift, retained rank, flopscope cost, memory, and residual. Compare
`D` with the frozen full-K3 RMSE interval only through the valid triangle bracket:

```text
max(0, RMSE_full - D) <= RMSE_candidate <= RMSE_full + D
```

Carry population-transfer and anchor uncertainty. Do not call drift an MSE
increment. Use the drift/cost thresholds frozen in the contract; do not choose `m`
after seeing the panel unless the contract's deterministic rule does so.

If no window clears the screen, stop today's candidate path and record the negative.
Do not fall back to arbitrary K3 rank capping, another scalar K2 calibration, or
public-board tuning. If one survives, complete a fresh independent Skeptic review
before building on it.

### 4. Build the narrowest supported submission architecture — STRATEGIC

The ideal long-run architecture remains:

```text
f32 K2 / faithful augmented-K2 prefix
  -> task-specific angular checkpoint
  -> exact sparse-aware K3 over the supported final layer window
  -> tiny rolled-forward response correction
  -> frozen calibration and safe fallback
```

For today's candidate, include **only modules whose own evidence path is already
green**. Do not rush the kernel-carrier, augmented-K2, K4, or learned-response tracks
into the ship file just because they are part of the ideal design. The likely
same-day candidate is the supported K2-prefix plus exact final-window K3 path; the
diagnostic, not this prompt, determines its window and checkpoint.

Implement first as a research arm with tests. Before touching `estimator.py` or a
truth block, write a complete STRATEGIC promotion/decision contract in
`experiments/LOG.md`, obtain Samuel's prediction, and commit the contract. Then:

- keep the implementation shape-generic in width and depth;
- use only stdlib, `flopscope`, `flopscope.numpy as fnp`, and `whestbench` in the
  eventual ship file;
- use immutable array construction—no item assignment or in-place array mutation;
- route every numerical operation through flopscope and never exploit raw-array
  work;
- derive any randomness only from `mlp.seed`/`context.seed` as appropriate;
- preserve the exact first-layer answer and return every row
  (`len(mlp.weights)`, which is 16 in the live round);
- avoid budget-summary calls and all scalar synchronization in `predict`;
- include deterministic finite/shape/instability fallbacks whose own cost and
  behavior are tested;
- stay under the single-array and total-memory limits with explicit accounting.

Create a new immutable `estimator_v*.py` archive only after the candidate is frozen.
Formally promote the same bytes into `estimator.py` so the current preflight
certifies the actual ship file. Update version/status docs and archive-identity
checks together. Do not spend today's clock rewriting preflight to accept arbitrary
candidates unless formal promotion is genuinely impossible; that would be a
separate IMPORTANT instrument slice with its own prediction and review.

### 5. One fresh paired decision — STRATEGIC

Only after the candidate is frozen and the contract commit precedes data, claim one
fresh SINGLE_USE Phase-2 DECIDE block through `local_sim/seed_ledger.py`. Never type
a literal seed and never reuse a block. Compare the candidate against the exact v2.7
control using common random networks.

Promotion authority is only `local_sim/competition_score.py::decide()` with the
gate recorded before data. Use the paired arithmetic adjusted-score difference and
a studentised-t or basic-bootstrap interval. Percentile bootstrap and BCa are
forbidden. Report:

- per-arm raw final-layer MSE and its GT-noise floor;
- paired adjusted-score difference, interval, and promotion boolean;
- counted FLOPs, `F/2**41`, multiplier, and the observed `+2**24` grading-server
  offset treatment near any boundary;
- zero/nonzero failures and their priced effect;
- subprocess residual headroom from the worst observed run, never a bare local
  wall time;
- memory accounting and what remains untested on Windows.

The candidate proceeds only if `decide()` promotes it **and** it clears Samuel's
preregistered materially-stronger bar. If it loses, freeze the artifact and stop.
Do not inspect public feedback to revise the same decision.

Obtain the required fresh independent Skeptic verdict from the raw packet before
changing `CURRENT_STATE.md` or treating the candidate as supported.

### 6. Deployment certification and approval packet

For a promoted candidate, run and preserve evidence for:

1. exact archive/ship-file identity and clean tracked tree;
2. `uv run whest validate --estimator estimator.py`;
3. default preflight, then `preflight.ps1 -Full -Log <durable transcript path>`,
   followed by committing the final transcript with the candidate packet;
4. a clean official Phase-2 mini run under `--runner subprocess` with
   `n_failed_mlps: 0`;
5. the required official-subprocess soak/tail checks, including first-MLP warm-up,
   p50, p99 and max;
6. zero scalar materializations in `predict`;
7. Linux/WSL memory-limit evidence or a clearly stated unresolved deployment block;
8. the exact estimator SHA-256, git commit, total FLOPs, multiplier, residual
   headroom, memory status, decision artifact, GT-noise floor, Skeptic verdict, and
   existing submission IDs `#328662` and `#328664`.

The full transcript must contain its own `GREEN 16/16` verdict line; do not infer
that from an exit code or a partial transcript. If any deployment item fails, repair
it under the appropriate new contract and do not submit.

When every item is green, stop and show Samuel a compact exact-SHA approval packet.
Ask: “Do you explicitly approve submitting estimator SHA `<sha>`?” Do not submit
until he answers yes to that exact hash in the active session.

### 7. Continuity requirements

At every stop point, update `experiments/LOG.md` and `docs/CURRENT_STATE.md` with the
actual result, raw artifact paths, current SHA, changed paths, operation ledger,
confidence interval, cost multiplier, residual headroom, memory status, failure
counts, Skeptic verdict, and the next Samuel-only decision. Never allow one
IMPORTANT or STRATEGIC result to flow silently into the next implementation.

Do not modify the quarantined write-up or organizer-email draft during this build.
Do not touch a vault. Do not designate submissions. Do not publish private benchmark
or discussion intelligence. Continue autonomously through all safe steps; pause only
for the required prediction answers, a failed preregistered gate, the independent
review's unresolved blocker, or Samuel's exact-SHA submission decision.
