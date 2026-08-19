# Claude Code implementation handoff — frontier architecture

Copy the prompt below into Claude Code from the repository root.

```text
You are taking over implementation in:

  C:\Users\samsu\Documents\arc-whest

The repository is Samuel's ARC White-Box Estimation Challenge 2026 entry. Read
AGENTS.md completely and obey it. Then read, in order:

1. docs/CURRENT_STATE.md
2. docs/FRONTIER_ARCHITECTURE_PROPOSAL.md
3. docs/VALIDATION.md
4. docs/GOTCHAS.md
5. docs/MOONSHOT.md
6. the newest relevant entries in experiments/LOG.md

Do not repeat broad literature or leaderboard research. The architecture audit is
already durable in FRONTIER_ARCHITECTURE_PROPOSAL.md. Your job is to begin careful
implementation, starting with the validation blockers—not to edit estimator.py or
build the final Kerdock/control architecture immediately.

First establish the live Phase-2 configuration using the repo's read-only checklist
and any organizer material already available. Record, or explicitly report as still
unknown: width, depth, biases, input distribution, n_mlps, budget B, lambda,
multiplier floor, flopscope version, whestbench version, and whether one or two final
submissions may be designated. Do not infer missing constants from stale challenge
page examples.

Then use the repository's normal next-slice/work-classification machinery. The first
implementation objective is Stage 0, P0.1 only:

  Build one canonical local scoring primitive that matches the installed official
  whestbench scorer exactly:

    score_i = mse_i * max(floor,
                          (counted_flops_i + lambda * residual_seconds_i) / B)
    objective = arithmetic_mean(score_i)

  It must preserve per-MLP MSE/cost correlation and reproduce official failure
  behavior. The paired arithmetic score difference and its MLP-level interval are
  the primary decision statistic. Geometric log-ratio remains available only as a
  secondary diagnostic.

Before editing, inspect the installed whestbench implementation, docs/VALIDATION.md,
and every current probe that computes a promotion statistic. Write the required
prediction/contract because changing how a number is computed is IMPORTANT or
STRATEGIC under AGENTS.md. Keep this first slice narrow: do not silently rewrite all
probes or alter the three-tier protocol in the same change.

Required evidence for the first slice:

- unit tests matching installed whestbench on synthetic per-MLP rows, including the
  0.1/future-configured floor, failures, heterogeneous budgets, residual time, and
  MSE/cost correlation;
- a regression fixture where arithmetic aggregation and geometric ratios select
  opposite winners;
- a read-only replay of existing v2.4/v2.4.1 rows if the durable report contains the
  necessary per-MLP fields;
- project checks required by AGENTS.md;
- an Evidence Packet with exact commands, outputs, changed files, and unresolved
  assumptions; and
- an independent Skeptic review before allowing the new scorer to decide anything.

Hard boundaries:

- Do not edit estimator.py in this first slice.
- Do not allocate or consume D17 or any other decision block.
- Do not evaluate or inspect vault truth.
- Do not use vault.py --dry-run.
- Do not submit anything.
- Do not edit docs/HUMAN_MODEL.md; it already has user-owned changes.
- Do not fit constants on the public split.
- Do not exploit uninstrumented wall time or FLOP-accounting gaps.
- Preserve unrelated working-tree changes.

After P0.1 is independently accepted, propose—not silently begin—the next smallest
slice from this ordered list:

1. fix v2.6 odd-width/fallback correctness and add parity tests;
2. replace seed-block prose with enforceable range consumption and reconcile D16/D17;
3. make the vault candidate-aware and remove unlogged truth evaluation;
4. reconstruct the v2.6 decision gate from durable code using the canonical scorer;
5. after the instrument is trustworthy, implement MUB-129 and BCH carriers behind
   one experimental interface;
6. reproduce a fixed layer-9 analytic-centre control;
7. compare K3 compression bases and only then try a frozen trajectory corrector;
8. test network-conditioned frame weighting last.

For every later estimator experiment, use fresh-network arithmetic score as the
primary objective, common random numbers only within an MLP across arms, independent
streams across MLPs or a crossed bootstrap, one-shift deployment-risk reporting,
ground-truth-noise propagation, and exact per-MLP cost before aggregation.

Stop and report instead of guessing if the Phase-2 specification is still missing or
would invalidate the Phase-1 homogeneity/design assumptions. End the first turn with
the Stage-0/P0.1 scope, classification, written prediction/contract, implementation,
tests, Evidence Packet, and Skeptic verdict—or with a precise blocker if one of those
cannot be completed safely.
```
