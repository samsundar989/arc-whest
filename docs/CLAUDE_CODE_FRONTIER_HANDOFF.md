# Claude Code handoff — Phase 2 frontier implementation

Copy the prompt below into Claude Code from the repository root.

---

You are the repo-local Builder for:

`C:\Users\samsu\Documents\arc-whest`

The Phase 2 dataset has now been published at width 1024 x depth 16. The previous
frontier checkpoint sequence assumed width 256 and must not be continued. CP0's
canonical scorer/lifecycle work is largely sound; CP1.1's 66,048-point Kerdock
proof was accepted, but the complete width-1024 analogue is unaffordable and is
not the Phase 2 backbone.

Your objective is to execute the replacement Phase 2 plan as far as evidence and
the repository's authority permit. Optimize for the fastest route to two safe,
validated candidates: a conservative full-dimensional K2-family estimator and an
aggressive goal-directed cumulant/residual correction. Do not implement every
speculative component. End each checkpoint in GO, KILL, PIVOT, or BLOCKED and
stop lanes that fail their frozen gates.

Read completely, in this order:

1. `AGENTS.md`
2. `docs/CURRENT_STATE.md`
3. `docs/PHASE2_FRONTIER_ARCHITECTURE.md`
4. `docs/VALIDATION.md`
5. `docs/GOTCHAS.md`
6. the CP0 and CP1.1 entries at the end of `experiments/LOG.md`
7. `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` only as historical Phase 1 context

Do not edit `docs/HUMAN_MODEL.md`. Preserve the Phase 1 estimator archives and
the existing 256 x 32 vault. Do not submit or touch any vault without Samuel's
explicit approval.

## First response and reconciliation

Begin with a read-only `session-start` and a concise reconciliation. Confirm from
official sources at execution time:

- whether round 1429 is LIVE or only the dataset is public;
- the `v2-phase2` commit and its width, depth, no-bias/no-output-layer contract,
  2^41 budget, 0.1 floor, truth size, and split sizes;
- whether whestbench 0.16.0 is publicly installable yet;
- whether flopscope 0.12.0 is the latest official release; and
- whether the one-vs-two final nomination rule has been clarified.

The official dataset card/metadata outrank stale challenge-page examples. Do not
guess a package version or change the main lockfile to a Git revision merely to
make the versions look current.

Then report the smallest correctly classified slice inside P2-CP0. Continue
autonomously through ROUTINE work. For IMPORTANT/STRATEGIC work, use the required
prediction, contract, Samuel prediction, review, and Skeptic machinery exactly as
`AGENTS.md` specifies.

## What has already been validated

Do not redo these from scratch unless the new stack invalidates them:

- canonical scorer: 88/88 on installed whestbench 0.14;
- canonical lifecycle self-test: 37/37;
- Kerdock/design math: 48/48;
- CP1.1 independent Skeptic verdict: ACCEPT;
- `estimator.py` and `estimator_v2_6_1.py` identity on the Phase 1 file.

Treat the following as open defects, not completed CP0 work:

1. main env is pinned to whestbench 0.14 / flopscope 0.10;
2. preflight can report RED on a cp1252 PowerShell because it does not own UTF-8
   for `whest validate`; validation passes with `PYTHONIOENCODING=utf-8`;
3. vault/coverage main paths still do new Estimator per MLP/arm and omit setup;
4. seed scan can miss hardcoded reuse inside an already claimed range, dynamic
   block selection can evade attribution, and D18 truth hashes are blank;
5. vault truth can be evaluated before the touch is durably logged;
6. four torch truth paths mask 32-bit official sample seeds to 31 bits;
7. no 1024 x 16 shape smoke exists; and
8. current `estimator.py` uses analytic-only fallback for width > 256.
9. `local_sim/phase2_checklist.py` finds the new revision but prints `v1-phase2`
   and recommends the obsolete width-256 fallback; repair its displayed tag and
   action guidance before relying on it operationally.

The official Phase 2 metadata contains both provenance versions 0.16 / 0.12 and
legacy `*_version_reported` values 0.14 / 0.10. Preserve and explain both in the
manifest. Do not silently choose whichever pair is convenient.

Fix only what is on the critical path to trustworthy P2-CP1 evidence. Do not turn
all eight into one mega-slice. Vault transactionality can be explicitly deferred
until before a Phase 2 vault exists, but lifecycle, seed provenance, artifact
hashing, exact stack, and shape coverage cannot be deferred past decisive data.

## Replacement checkpoints

The live sequence is:

- P2-CP0: official-stack and evidence readiness
- P2-CP1: Phase 2 baseline ladder
- P2-CP2: deploy conservative K2
- P2-CP3: augmented K2
- P2-CP4: goal-directed K3 capacity
- P2-CP5: deploy aggressive correction or coupled residual fallback
- P2-CP6: composition and broad fresh-network robustness
- P2-CP7: grader telemetry and nomination, only with Samuel approval

Do not execute the old K129 adaptive-reweighting CP2–CP5 sequence. Its capacity
result was Phase 1-specific and its complete main carrier now costs about 16
budgets. A 32-basis orthogonal arm is allowed only as a P2-CP1/P2-CP5 sampling
comparator; never describe it as the completed degree-5 Kerdock rule.

## P2-CP0 — official-stack and evidence readiness

Goal: make one trustworthy path capable of producing immutable per-MLP evidence
at 1024 x 16 on the Phase 2 stack.

Required outcome:

- a machine-readable official Phase 2 manifest;
- corrected checklist output naming `v2-phase2` and the 1024 x 16 migration;
- exact package status recorded; use an isolated compatibility environment while
  comparing stacks, and update the project pin only through a reviewed slice;
- UTF-8-safe preflight behavior;
- at least a 1024 x 16 shape/fallback smoke without allocating a giant truth set;
- new Phase 2 experiments routed through one Estimator + setup once;
- full 32-bit sample seeds and nonblank immutable artifact hashes;
- seed attribution that fails closed for literal and dynamic reuse; and
- no decision seed or vault touch spent.

If whestbench 0.16 is not public, mark exact package parity BLOCKED but continue
safe distributional DESIGN work in an isolated lab. Stamp that limitation on
every artifact and rerun parity as soon as 0.16 is official.

## P2-CP1 — Phase 2 baseline ladder

This is the next research slice and it comes before any estimator rewrite or
Phase 1 constant refit.

On fresh 1024 x 16 DESIGN networks, compare under one runner:

1. exact layer-1 control;
2. K1/raw closure;
3. full-dimensional simple K2;
4. full-dimensional augmented K2;
5. naive/Sobol sampling around the effective 65,536-sample reference; and
6. 32 randomized orthogonal signed bases at matched rows.

Use two independent truth replicas on decisive networks and report the ordinary
MSEs, replica noise, and the cross-product noise-corrected MSE. Fit nothing on
public mini/full. Use a DESIGN block for any calibration/rank/implementation
choice and a fresh single-use block for the later gate.

Required diagnostics:

- final-layer raw MSE and canonical arithmetic score per MLP;
- MSE by depth to test the predicted `(L/n)^K` mechanism;
- exact flopscope 0.12 counted cost, operation mix, residual, peak memory, and
  failures;
- paired interval and biggest-mover/tail share;
- truth-noise level; and
- explicit comparison with the paper projection rather than silently treating it
  as fact.

P2-CP1 ends by freezing the conservative statistic and the deployment gate. It
does not edit `estimator.py`.

## P2-CP2 — conservative K2 deployment

Port only the selected K2 spelling to grader-legal `fnp`. Create a new versioned
Phase 2 archive; never edit a historical `estimator_v*.py` file. Preserve exact
layer 1, row-vector convention, full covariance, immutable arrays, all-layer
shape, deterministic seed behavior, and one-setup lifecycle.

Require reference parity, zero failures, memory/setup safety, and a fresh DECIDE
interval against the best P2-CP1 sampling baseline. The primary statistic is the
arithmetic mean of per-MLP scores, never the geometric ratio or a product of
aggregate means.

## P2-CP3 — augmented K2

ARC's leading polynomial predicts simple K2 near C/B 0.055 and augmented K2 near
0.087, both below the 0.1 score floor before residual. Reproduce exact 0.12 costs.

Continue only if augmented K2 improves exact arithmetic score with an interval
excluding zero; 1.25x is the proposed strong-pass target and must be frozen in the
Slice Contract. If residual pushes C/B over 0.1, charge it exactly. If augmented
K2 loses after cost, keep simple K2 and close this spelling.

## P2-CP4 — goal-directed K3 capacity

Full factorized K3 is projected near C/B 4.05 and is not a production spelling.
Measure a full or reduced-depth offline K3 oracle, then test whether its final-mean
correction is visible in a small weight-only output/adjoint basis Q, with a frozen
ladder such as q in {4, 8, 16, 32}.

This is not the closed Phase 1 covariance-eigenbasis projection. The hypothesis is
to retain only higher-cumulant contractions observable by the final mean. Require
at least 50% of oracle correction gain on both fresh halves at a deployable q and
credible exact cost. If q approaches width/2, KILL immediately as the old rank
wall repeating.

## P2-CP5 — aggressive deployment/fallback

Deploy the passing goal-directed cumulant correction. If CP4 gives PIVOT rather
than GO, run the coupled residual ladder:

- Sobol versus 32 orthogonal bases;
- exact-mean first-layer/linearized controls transported by the K2 Jacobian; and
- independent pilot/main fitting only if coefficients are adaptive.

Require at least 1.25x arithmetic-score improvement after full cost, zero
failures, stable p95/max, and a demonstrated independence boundary. Preserve the
conservative candidate unchanged.

## P2-CP6/P2-CP7 — robustness and telemetry

Run factorial composition and 300–600 fresh networks with two-replica truth where
needed. Freeze conservative and aggressive hashes separately. Simulate the
private-rerun arithmetic objective and report tail/failure risk and residual-price
sensitivity.

No submission is authorized. For each telemetry request: `whest validate`, clean
Phase 2 mini with zero failed MLPs, present candidate hash and evidence to Samuel,
receive explicit approval, submit, then ingest the graded config and telemetry.

## Rules that govern every slice

1. Do not fit anything on public mini/full. Public is CONFIRM only.
2. Allocate/claim a new block before any strategic experiment and record the
   contract before data exists.
3. Use common random numbers only within an MLP across arms; preserve per-MLP
   MSE/cost correlation and exact failure scoring.
4. Report truth noise beside every MSE and exact stack/provenance beside every
   artifact.
5. Never call a predicted leading FLOP polynomial a measured cost.
6. Never multiply standalone gains; measure composition.
7. After every IMPORTANT/STRATEGIC slice, run `review-slice` and an independent
   Skeptic on raw evidence.
8. Preserve a clean conservative candidate while developing the aggressive one.
9. Stop a failed spelling at its registered gate; do not tune it indefinitely.
10. Keep `docs/CURRENT_STATE.md`, `experiments/LOG.md`, and machine-readable
    artifacts sufficient for a new agent to resume without chat history.

At the end of every checkpoint, send Samuel exactly:

```text
CHECKPOINT <id> — <name>
Status: GO | KILL | PIVOT | BLOCKED
Question:
What changed:
Evidence tier and exact seeds/touches used:
Stack and candidate/truth hashes:
Primary arithmetic-score result and interval:
Raw/noise-corrected MSE and truth noise:
Mechanism diagnostics:
Counted/effective cost, residual, memory, failures:
Gate comparison:
What the evidence means:
What it does not mean:
Files/artifacts:
Next action:
Samuel decision needed: none | <one concrete decision>
```

Begin now with the read-only reconciliation and the smallest P2-CP0 critical-path
slice. Do safe work autonomously. Pause only for Samuel's prediction/approval,
vault authority, submission approval, an actual external package blocker, or a
decision that would materially change the registered strategy.

---
