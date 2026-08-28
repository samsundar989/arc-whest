# Claude Code handoff — Phase-2 frontier program

Copy the prompt below into a new Claude Code session started at the repository
root. It assumes no chat history. Repository evidence is authoritative.

---

## Prompt

You are the repo-local Builder for Samuel's ARC White-Box Estimation Challenge
2026 entry. Work from:

```text
C:\Users\samsu\Documents\arc-whest
```

First read `AGENTS.md`, `docs/CURRENT_STATE.md`, the bottom of
`experiments/LOG.md`, `docs/GAME_PLAN.md`,
`docs/PHASE2_FRONTIER_AUDIT_2026-08-28.md`, `docs/VALIDATION.md`,
`docs/GOTCHAS.md`, and `docs/SKEPTIC.md`. Inspect HEAD and the worktree before
editing. Do not depend on this prompt where a current repository document differs.

Never edit `docs/HUMAN_MODEL.md`. Never submit, touch a vault, claim a seed block,
change a gate or residual model, or edit `estimator.py` without the exact approvals
and preregistration required by `AGENTS.md`. Phase 2 permits up to two designated
submissions and two public entries already exist; do not create a duplicate
submission or assume either entry should be designated.

## Current conclusion

- At the **2026-08-28 16:40 UTC** snapshot, the public Phase-2 frontier was around
  `7.4e-9` adjusted and `2.3e-8` to
  `3.5e-8` raw. Our best public entry is about `1.374e-7` adjusted and
  `1.374e-6` raw. Scalar calibration helped, but it cannot close this gap.
- The local official-package evaluator is well reconciled. The default preflight
  is green 12/16 and the local official subprocess FLOP ledger matches exactly.
  The grading server adds a small `2**24` total-FLOP offset, Windows cannot enforce
  the 8 GiB process cap, and `preflight.ps1` currently certifies only
  `estimator.py`.
- Full third-cumulant propagation measures `4.0047e-8` on DESIGN (GT-noise floor
  `1.155e-9`) and `3.9249e-8` on organiser CONFIRM (GT-noise floor `7.478e-11`),
  but not yet at a winning measured cost. The immediate bottleneck is making its
  exact mathematics cheap, then learning which layer ages need to be retained.
- Kerdock/MUB carriers are valid candidates, not the architecture by fiat. Carrier
  policies must be selected with a task-specific deep-ReLU kernel proxy and then
  tested against independent rotated small blocks, orthogonal, IID and low-discrepancy
  controls.

## Execute in this order

### 1. Track 1: `W-EFF` exact-mathematics cost proof — IMPORTANT

Before implementation, obtain Samuel's written prediction and append a knowledge
contract to `experiments/LOG.md`. Preserve full-K3 mathematics and outputs exactly.
Profile the full run, prove representative-layer equality, and produce an official
flopscope operation ledger for the proposed sparse/block-aware contractions. The
claim to test is cost only; do not spend truth or choose a model from MSE.

Stop if exact parity cannot be proved, the projected integrated multiplier remains
too high, or the saving relies on uninstrumented/raw-array work. Run default
preflight, portability and the required fresh independent Skeptic before believing
the result.

### 2. Track 2: late-window / layer-age K3 diagnostic — IMPORTANT

Only after an IMPORTANT knowledge contract and Samuel's prediction are recorded,
use **unregistered, label-free probe networks** to compare full K3 with K2-prefix +
exact-K3 windows `m in {2,4,6,8}`. Include a matched-rank arbitrary-cap control.
Measure final prediction drift, propagated third-cumulant slice drift, exact FLOPs
and residual headroom. Drift is not an error increment: carry the full-K3 RMSE
interval and report the triangle-inequality bracket, with population-transfer
uncertainty. This is an imitation diagnostic, not a promotion decision and not
authority to read or spend any truth block.

No seed-ledger block is claimed for this label-free diagnostic. If its scope changes
to inspect labels, select a model, or make a promotion claim, stop and restart it as
STRATEGIC under Rule 1 with a freshly claimed DESIGN block.

Kill the late-window claim if early third-cumulant blocks materially determine the
final result, checkpoint initialization dominates the error, or measured cost does
not fall with retained layer age as predicted.

### 3. Track 3: task-specific carrier objective — IMPORTANT, then STRATEGIC

Implement and verify the prior-average deep-ReLU-kernel quadrature objective from
the Game Plan. Freeze its recurrence, candidate panel and selection rule before
looking at MLP truth. First test whether the truth-free kernel energy ranks carrier
policies in the same direction as finite-width MLP MSE. Global rotations of one
fixed Gram matrix are a null control; independent rotations between small blocks are
a genuinely different candidate.

Do not promote a carrier because it is a spherical design or has a good generic
moment certificate. Promote only if the preregistered kernel proxy transfers to
the task and the integrated paired adjusted-score gate passes.

### 4. Integrate only after the modules survive independently

The conservative candidate is float32 K2/augmented-K2 plus a kernel-selected carrier
checkpoint and frozen response correction. The aggressive candidate is K2 prefix,
carrier-initialized exact K3 over the final four to six layers, a tiny cross-fitted
response correction and frozen calibration. Bill all branches together and apply
the current server-offset fixture near a multiplier boundary.

Promotion requires paired arithmetic adjusted-score improvement on a fresh DECIDE
block through `competition_score.decide()`, the allowed interval excluding zero,
reported GT-noise floors, zero failures, server-parity mini validation, deployment
headroom and an independent Skeptic verdict. Percentile bootstrap, BCa, public-split
fitting and reused blocks are forbidden.

## Report back

At each stop, report the exact contract, current SHA, changed paths, raw artifact
paths, operation ledger, confidence interval, failure counts, cost multiplier,
residual headroom, memory status, Skeptic verdict and the next Samuel-only decision.
Do not silently flow one strategic result into the next implementation.
