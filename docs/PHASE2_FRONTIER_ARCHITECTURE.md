# Phase 2 frontier architecture — rev 1 — 2026-08-21

**Status:** evidence-backed strategist packet and replacement execution plan for
Phase 2. It supersedes the checkpoint order in
`docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` for new work, because that plan assumed
width 256. It does not authorize an estimator edit, seed-block spend, vault touch,
or submission. Those still go through `next-slice`, a registered Slice Contract,
and `review-slice` as required by `AGENTS.md`.

## Decision

The Phase 2 dataset is genuinely public. The official `v2-phase2` revision is a
public Hugging Face commit dated 2026-08-21 and contains 100 mini plus 1,000 full
MLPs. Its authoritative configuration is:

| item | Phase 2 value |
|---|---:|
| width x depth | **1024 x 16** |
| biases | **none** |
| separate output layer | **none**; all 16 matrices are followed by ReLU |
| input | `x ~ N(0, I)` |
| weights | i.i.d. `N(0, 2/width)` |
| truth | 1,000,000,000 samples per MLP |
| budget | **2^41 = 2,199,023,255,552 FLOPs / MLP** |
| score floor | **0.1** |
| effective naive samples | approximately **65,536** |
| bake stack | whestbench 0.16.0 / flopscope 0.12.0 |

At the time of the audit, the files were public but competition round 1429 was
still `UPCOMING`, scheduled to open at 2026-08-21 23:59 UTC. Dataset publication
and submission-round activation are separate events.

This shape invalidates the old plan's main carrier but creates a much stronger
opening elsewhere:

> **Build the conservative candidate around full-dimensional cumulant propagation
> K2, test augmented K2 while it remains below the score floor, and develop the
> aggressive candidate as a goal-directed/adjoint-compressed K3 or Edgeworth
> correction. Use reduced orthogonal sampling only to estimate the remaining
> residual, not as the primary estimator.**

Do **not** continue old CP2–CP7 as written. Do **not** start by refitting
`_PRUNE_TAU` or `_BASIS_RANK`; both belong to a width-256 sampling architecture
that Phase 2 has already displaced. First establish which analytic order now owns
the accuracy/cost frontier.

## Why the architecture flips

ARC's primary paper predicts cumulant-propagation error approximately

```text
MSE_K ~= c_K (L / n)^K
```

and measures the expected `n^-K` and approximately `L^K` scaling. From Phase 1
to Phase 2, `L/n` falls from `32/256 = 1/8` to `16/1024 = 1/64`. For K2 this is an
**8x smaller expansion parameter and a 64x MSE reduction**, before any new
calibration. This is the exact external-evidence condition that reopened the
mechanistic lane in `CURRENT_STATE.md`.

Using the paper's leading FLOP polynomials gives the following design budget.
These are planning estimates, not substitutes for flopscope 0.12 measurements:

| method | Phase 2 scale | leading counted C/B | implication |
|---|---:|---:|---|
| naive propagation, 65,536 samples | 65,536 x 33,554,432 | about 1.0 | sampling reference |
| simple K1 | `4 n^2 L + ...` | 0.000031 | diagnostic only |
| **simple K2** | `7 n^3 L + ...` | **0.0549** | safely below the 0.1 score floor before residual |
| **augmented K2** | `11 n^3 L + ...` | **0.0871** | potentially higher accuracy at the same 0.1 multiplier, but residual headroom is narrow |
| full factorized K3 | `30 n^3 L^2 + 39 n^3 L + ...` | **4.05** | cannot be the direct production spelling |
| full 513-basis Kerdock design | 1,050,624 signed rows | **about 16.0** | mathematically valid, operationally dead |
| 32 orthogonal signed bases | 65,536 rows | about 1.0 before overhead | useful sampling comparator/residual arm, not a degree-5 completion |

The Phase 1 raw closure measured about `6.28e-5` at depth 32 and calibrated K2
about `2.44e-5`. A pure scaling projection gives Phase 2 values near `9.8e-7`
and `3.8e-7`, respectively. At the 0.1 multiplier the latter would be around
`3.8e-8` adjusted. This is a **hypothesis to test on fresh Phase 2-shaped mints**,
not a result. It is nevertheless strong enough to determine the order of work.

## Audit of work already completed

### CP0: strong core, not fully closed

What is correct and banked:

- `competition_score.py` implements the arithmetic per-MLP competition objective;
  its current suite passes 88/88 against installed whestbench 0.14.
- `decide()` uses the arithmetic gain, a one-sided positive-difference condition,
  and only studentised-t or basic-bootstrap intervals. It refuses percentile and
  BCa for promotion.
- `lifecycle_eval.py` correctly models one Estimator, one off-budget `setup()`, and
  the six provokable grader failure paths; its suite passes 37/37.
- the default preflight now wires mint provenance, invariants, scorer, seed-ledger
  refusals, vault hashes, and lifecycle tests.

What remains open before Phase 2 evidence can decide architecture:

1. The repo is pinned to whestbench 0.14 / flopscope 0.10. The dataset was baked
   with 0.16 / 0.12; flopscope 0.12 is public, but whestbench 0.16 was not yet on
   PyPI at audit time. The official metadata also retains legacy
   `*_version_reported` values (0.14 / 0.10) beside the newer provenance values;
   preserve both fields in the Phase 2 manifest and resolve the distinction
   against the public package or first graded report rather than normalising it
   away.
2. `preflight.ps1` does not force UTF-8 for `whest validate`; in a cp1252
   PowerShell it reports RED after the CLI succeeds internally but crashes while
   printing its Unicode table. Explicit `PYTHONIOENCODING=utf-8` makes validation
   pass. The gate should own this portability detail.
3. The canonical lifecycle runner exists but `vault.py` and `coverage_lab.py`
   still instantiate a new Estimator per MLP/arm and omit `setup()` on their main
   paths. Existing coverage calibration should be rerun after migration.
4. The seed scan ignores a hardcoded seed if some prior claim already covers that
   number, dynamic block selection can evade attribution, and the two D18 truth
   claims still have blank artifact hashes.
5. `vault.py` opens and evaluates against truth before durably recording the
   touch, so interruption can create an unlogged touch.
6. Four torch truth paths mask official 32-bit sample seeds to 31 bits. The labels
   remain unbiased but are not bit-reproducible to whestbench.
7. The old vault is 256 x 32. It cannot validate a Phase 2 candidate. Do not
   repurpose or overwrite it; define a separate Phase 2 vault only if the protocol
   authorizes one.
8. The current shape suite has no 1024 x 16 smoke. `estimator.py` deliberately
   takes analytic-only fallback whenever `width > 256`, so it is shape-valid but
   not a competitive Phase 2 baseline.
9. `local_sim/phase2_checklist.py` detects the new tag but prints the nonexistent
   name `v1-phase2` and recommends a width-256 Sobol fallback. Its round and board
   polling remain useful; its tag/action text is stale and must be repaired before
   it is treated as an operator checklist.

CP0 therefore ends **PIVOT**, not GO: the decision statistic is fixed, while the
Phase 2 toolchain and evidence producers still need a narrow readiness slice.

### CP1.1: accepted mathematics, obsolete production object

The worker's Kerdock result is correct:

- the 256-dimensional code fingerprint and exact weight enumerator are verified;
- the Delsarte transform proves binary strength 5;
- Kerdock plus 512 axial points gives an exact spherical degree-4 rule, with odd
  degrees exact by antipodality;
- the carrier tests pass 48/48;
- the independent Skeptic accepted the slice; and
- the review caught and repaired both an IID-denominator bug and the mistaken
  point-rotation cost, replacing it with rotation of the first weight matrix.

No end-to-end K129 MSE was measured and no decision block was spent. That was
correct restraint.

At width 1024 the exact analogue needs 513 real MUB bases and 1,050,624 signed
rows, about 16 budgets of dense propagation. Thus the theorem and generator are
reusable research assets, but the complete carrier is **KILL as a Phase 2
backbone**. A reduced 32-basis orthogonal arm remains worth one comparison, but it
has only the frame/3-design guarantees; it is not the completed degree-5 rule.

## Target architecture

```text
official Phase-2 adapter and canonical lifecycle evidence
                         |
       exact layer 1 + full-dimensional simple/augmented K2
                         |
          per-layer means/covariances and uncertainty diagnostics
                         |
       +-----------------+------------------+
       |                                    |
 conservative output              goal-directed K3/Edgeworth
 simple or augmented K2           correction in a weight-only
                                  adjoint/output error basis Q
                                                |
                                  small coupled residual sampler
                                  (Sobol vs 32 orthogonal bases)
                                                |
                                      aggressive output
```

Maintain two candidates from the first deployment slice:

1. **Conservative:** the strongest full-dimensional K2-family estimator whose
   total effective cost remains at or near the 0.1 floor.
2. **Aggressive:** the conservative centre plus a separately gated cumulant or
   coupled-sampling residual correction. It must be removable with one flag and
   must never endanger the conservative artifact.

### A. Full-dimensional K2 core

Port the already-parity-checked ARC K2 trajectory to grader-legal `fnp` and make
it the shared centre for every branch. Preserve:

- exact layer-1 mean;
- row-vector weight convention (`h @ W`, equivalently `W.T @ h`);
- full covariance, not a rank scaffold;
- immutable array operations;
- all-layer output shape even though only the final layer scores; and
- exact per-MLP counted/effective evidence.

Do not fit a Phase 2 calibration until raw K2 and augmented K2 have been compared
on a DESIGN split. If a calibration is used, fit on one fresh block and decide on
a disjoint block. Never fit it on public mini/full.

### B. Augmented K2 as the first frontier experiment

ARC's augmented K2 keeps additional leading Edgeworth diagrams while preserving
`O(n^3 L)` scaling. The paper's leading count lands around 8.7% of B, which may
still receive the same 0.1 multiplier as simple K2.

This is the highest-value near-term experiment because a raw-accuracy gain inside
the floor is nearly free in competition score. The implementation must measure
the exact 0.12 price and residual; 8.7% leaves only about 1.3% of B as headroom.
If residual pushes it over the floor, use the exact arithmetic score rather than
calling the additional work free.

### C. Goal-directed compressed K3/Edgeworth correction

Full factorized K3 is about four budgets at 1024 x 16. The production question is
therefore not whether K3 works, but whether the part of K3 that changes the final
mean can be computed without transporting the full state.

Build a weight-computable output/adjoint basis `Q` and propagate only the
higher-cumulant contractions that can reach `Q`. Start with a frozen ladder such
as `q in {4, 8, 16, 32}`. The public Phase 1 evidence that fixed-carrier output
error was low-dimensional is a reason to test this; it is not permission to
assume transfer to the new shape.

The decisive diagnostics are:

- full or reduced-depth K3 oracle gain over K2;
- fraction of that correction energy retained by `Q`;
- agreement across fresh network and output-coordinate folds;
- exact counted cost projected to the production spelling; and
- whether the corrected arithmetic score improves after its multiplier.

This differs from the closed Phase 1 rank-k covariance lane. That lane projected
the entire forward state into covariance eigenvectors and required rank n/2.
The new hypothesis is **goal-oriented**: retain only contractions observable in
the final mean/adjoint subspace. If it also needs q near n/2, kill it immediately.

### D. Coupled residual sampling

If deterministic K2 leaves a correctable residual, use at most the budget that
improves the exact score. Compare:

1. ordinary/Sobol Gaussian samples;
2. 32 randomized orthogonal signed bases (65,536 rows at full sampling budget);
3. a linearized or shallow exact-mean control coupled to the same samples; and
4. an independent pilot/main fit only if coefficients are network-adaptive.

The complete 513-basis Kerdock rule is excluded. Reduced frames are useful because
orthogonal coupling has both public Phase 1 evidence and general Monte Carlo
theory, not because they preserve degree-5 exactness.

The most valuable control to test is a pathwise zero-mean linear/first-layer
surrogate transported through the K2 Jacobian. Its expectation is known, so it can
reduce sampling variance without introducing target-fitted bias. Kill it if the
coupled variance reduction does not pay for propagation after exact cost.

## Validation architecture for Phase 2

The Phase 2 evaluator must produce one immutable row per MLP containing candidate
hash, truth artifact hash, input/estimator/sample seeds, stack versions, raw MSE,
truth-noise estimate, counted FLOPs, residual time/model, failure status, and
canonical score.

Use two independent truth replicas for decisive fresh-mint comparisons. For a
prediction `p` and truth replicas `y1=t+e1`, `y2=t+e2`, report both ordinary MSEs
and the cross-product estimate

```text
mean_j (p_j - y1_j) (p_j - y2_j)
```

whose expectation is the prediction's MSE when replica noises are independent.
This removes the additive label-noise bias without pretending the labels are
exact. Quote its uncertainty and the per-replica noise floor.

Public mini/full remain CONFIRM only. A public result may falsify a choice made on
fresh mints, but it may not choose a constant, arm, rank, layer, or calibration.

## Replacement checkpoints

Every checkpoint ends in `GO`, `KILL`, `PIVOT`, or `BLOCKED` and emits the receipt
specified in `docs/CLAUDE_CODE_FRONTIER_HANDOFF.md`.

### P2-CP0 — official-stack and evidence readiness

**Question:** Can this repo generate and score trustworthy 1024 x 16 evidence on
the actual Phase 2 stack?

Deliverables:

- machine-readable Phase 2 manifest from the official dataset metadata/card;
- corrected Phase 2 checklist output (`v2-phase2`, 1024 x 16, no width-256 fallback);
- isolated upgrade/parity path for flopscope 0.12 and whestbench 0.16 once public;
- UTF-8-safe preflight and a 1024 x 16 shape/fallback smoke;
- lifecycle runner used by new Phase 2 experiments, not merely self-tested;
- full 32-bit truth seeds, nonblank artifact hashes, and no ledger bypass;
- explicit resolution or deferral of vault transactionality; and
- no Phase 1 archive or vault mutation.

Exit: official-stack scorer/mint/lifecycle parity, all applicable checks green,
and no decision seed or vault touch spent. If whestbench 0.16 remains unpublished,
mark exact package parity `BLOCKED` but continue distributional DESIGN work in an
isolated lab with that limitation stamped on every artifact.

### P2-CP1 — Phase 2 baseline ladder

**Question:** Which family owns the raw accuracy/cost frontier at 1024 x 16?

On fresh DESIGN networks with two truth replicas, compare exact layer 1, K1/raw
closure, simple K2, augmented K2, naive/Sobol sampling near 65k, and 32 orthogonal
bases. Measure every layer but decide on final-layer arithmetic score.

Exit: a frozen conservative statistic, an error-vs-depth mechanism curve, exact
0.12 cost, and a preregisterable gate for deployment. No public selection.

### P2-CP2 — deploy the conservative K2 candidate

**Question:** Does the selected K2 spelling survive grader constraints and beat
the best sampling baseline after full cost?

Build a new versioned Phase 2 estimator; do not overwrite a Phase 1 archive. Require
grader-legal imports, setup/predict lifecycle parity, zero failures, all-layer
shape, exact layer 1, dense reference parity, and a fresh DECIDE interval. A
candidate with raw MSE but unsafe residual/memory does not pass.

### P2-CP3 — augmented K2

**Question:** Can additional K2-order Edgeworth terms improve raw MSE while total
effective cost remains competitively priced?

Gate on exact arithmetic score, with a strong-pass target of at least 1.25x over
simple K2 and an interval excluding zero. If augmented K2 costs more but does not
improve score, keep simple K2 and close this spelling.

### P2-CP4 — goal-directed K3 capacity

**Question:** Is the K3 correction observable in a small weight-only adjoint basis?

First measure an offline oracle or reduced-depth K3 correction. Then test the
frozen q ladder on disjoint networks/output folds. Continue only if a deployable
q retains at least 50% of the oracle gain on both halves and has a credible exact
cost below the benefit. If q approaches n/2, `KILL` as the previously observed
rank wall.

### P2-CP5 — deployable aggressive correction

**Question:** Does adjoint K3 or coupled residual sampling improve the actual score?

Try the passing CP4 correction first. If CP4 pivots, test the coupled residual
ladder. Require at least 1.25x arithmetic-score gain after full cost, zero failures,
stable p95/max tails, and independence wherever a pilot fit is used.

### P2-CP6 — composition and private-rerun robustness

Run factorial ablations and 300–600 fresh networks. Freeze separate conservative
and aggressive hashes. Report truth noise, arithmetic interval, failure/tail risk,
one-shift deployment distribution, and sensitivity to residual pricing.

### P2-CP7 — grader telemetry and nomination

No submission is authorized by this plan. Required order: validate, clean local
mini with zero failures, present hash/evidence to Samuel, receive explicit approval,
submit, and ingest the graded config/telemetry. Confirm whether one or two Phase 2
entries may be nominated before making the final choice.

## Immediate order of work

1. Stop the old CP2–CP7 queue and record CP1.1 as accepted-but-obsoleted.
2. Finish only the critical P2-CP0 repairs needed by the next experiment.
3. Recheck whether whestbench 0.16 and round 1429 are live; do not mutate the main
   lockfile against a guessed package.
4. Build the P2-CP1 lab ladder before touching `estimator.py` or refitting Phase 1
   constants.
5. Promote simple/augmented K2 according to fresh arithmetic evidence.
6. Spend frontier time on goal-directed K3, with reduced orthogonal sampling as
   the planned fallback.

## Sources

- Phase 2 dataset/card: <https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v2-phase2>
- Phase 2 metadata: <https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v2-phase2/metadata.json>
- ARC cumulant paper: <https://arxiv.org/html/2605.05179>
- ARC reference code: <https://github.com/alignment-research-center/mlp_cumulant_propagation>
- flopscope 0.12 release: <https://github.com/AIcrowd/flopscope/releases/tag/v0.12.0>
- finite-width Edgeworth evidence: <https://arxiv.org/abs/2605.24072>
- orthogonal Monte Carlo: <https://proceedings.mlr.press/v97/choromanski19a.html>
