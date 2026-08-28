# Phase-2 frontier audit — 2026-08-28

This is the dated evidence report behind the current `GAME_PLAN.md`. It is not a
second roadmap. Competition observations are timestamped; research proposals are
labelled as hypotheses; no public or organiser data was used to fit a constant.

No decision block or vault was touched for this audit. New live artifacts are the
read-only leaderboard snapshot under
`experiments/board_snapshots/2026-08-28T1640Z_phase2_frontier_audit_board.*` and
minimal Chrome captures under `experiments/2026-08-28_p2_live/` for Rules v14,
its Phase-2 default-selection clause, forum provenance and leader precision.

## Executive verdict

The validation core is strong enough to direct research, but the shipped method
is not a frontier architecture. `estimator.py` is K2 covariance propagation at
Phase-2 width; v2.7 adds a scalar output calibration. That calibration transfers
well but is structurally exhausted. On the 2026-08-28 board the best entry is
`18.6x` better in adjusted score and `52.9x` better in raw final-layer MSE.

The latest repository evidence changes the research order:

1. **Exact/factorized K3 is the strongest accuracy anchor.** It measures
   `3.9249e-8` raw MSE on the organiser mini (GT-noise floor `7.478e-11`),
   agreeing within `1.99%` with the independent DESIGN reading (`4.0047e-8`,
   GT-noise floor `1.155e-9`). The missing information is real.
2. **The existing K3 implementation is not deployable.** Its `1.4796 B` cost is
   measured by the upstream Torch counter, not flopscope 0.12. Arbitrary rank
   caps damage the diagonal-slice queries the recurrence consumes; the best tested
   rank-`n` object is a strong negative, not an impossibility theorem.
3. **W-EFF is the immediate priced question.** About `95.35%` of the upstream K3
   cost sits in two operations with exploitable fresh-block sparsity, but the
   `5.49x` cross-meter ratio is an affordability target, not a W-EFF saving; one
   exact `fnp` operating point must be billed by flopscope.
4. **Late-window K3 is the highest-value new hypothesis.** Start exact K3 only for
   the final `m` layers, or retain complete factor blocks by age. This preserves a
   mechanistic unit—the contribution of a whole nonlinearity—rather than deleting
   arbitrary columns. Screen it against full-K3 predictions before reading truth.
5. **Structured angular sampling remains live but is no longer presumed to be the
   sole backbone.** The Kerdock/MUB carrier is valid and public frontier evidence
   supports the family. The fixed 16-basis carrier is anisotropic at fourth order,
   and the winning basis policy is unknown. Carrier design should be screened by
   deep-ReLU-kernel discrepancy, then validated against finite-width MLPs.

The intended end state is a hybrid: f32 K2 or augmented K2 in the prefix, a
task-specific angular checkpoint, sparse-aware exact K3 in the final window, and
only then a small cross-fitted response/calibration layer.

## Live competition truth

Captured at **2026-08-28 16:40 UTC**. The board is public feedback, not the final
fresh-suite ranking.

| item | current value |
|---|---:|
| teams visible | `67` |
| Puffi, rank 1 | displayed `7.4e-9` adjusted / `2.6e-8` raw / `~0.285 B` |
| best raw MSE (jamespayor) | `2.33e-8` at a higher multiplier |
| neopok919, rank 41 | `1.374e-7` / `1.374e-6` / floor-pinned |
| adjusted gap | `18.6x` |
| raw-MSE gap | `52.9x` |

The top nine entries have displayed raw MSEs in a narrow `2.3e-8`–`3.5e-8` band while
their implied multipliers range roughly `0.285`–`0.476`. The frontier is jointly
optimising accuracy and cost. Our deficit is mostly accuracy, not price: v2.7 is
already on the `0.1` multiplier floor.

Two Phase-2 submissions are graded, both with zero failures:

| id | estimator | adjusted | raw final MSE |
|---|---|---:|---:|
| [328662](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328662) | v2.6.1 canary | `3.933e-7` | `3.933e-6` |
| [328664](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328664) | v2.7 calibration | `1.374e-7` | `1.374e-6` |

Rules v14 now explicitly permits **up to two designated Phase-2 submissions** for
the fresh private rerun. The old “Rules says one / forum says two” contradiction is
resolved. It also says that when a team designates none, §6 selects the two
best-scoring valid submissions from that Phase's public leaderboard. Raw captures:
`experiments/2026-08-28_p2_live/2026-08-28T170620Z_rules_v14_phase2_designation_capture.json`
and `2026-08-28T171010Z_rules_v14_phase2_default_selection_capture.json`.

Chrome was signed out during this audit. The public submissions are verified; the
account's current designations and remaining daily allowance are not.

### Latest discussion-board evidence

The most operationally important new thread is the
[Phase-2 residual synchronization investigation](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202).
It isolated `bool()`, `float()` and `int()` conversions of flopscope arrays,
implicit conditionals, assertions and `allclose` as graph-synchronization points
charged to residual time. Removing 166 such materializations changed a five-fail
run into 50/50 clean without changing mathematics, FLOPs or call count. Every new
candidate therefore needs a **zero mid-predict materialization** invariant, not
merely a count of flopscope calls.

Other current public evidence:

- A [Phase-2 covariance/whitened-antithetic blend](https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200)
  reports a `24.85%` local adjusted-score reduction and very low branch-error
  correlation. This supports complementary estimators, not another tiny covariance
  refinement.
- The [moment-matched sampling write-up](https://discourse.aicrowd.com/t/phase-1-write-up-variance-reduction-beats-closure-refinement-a-moment-matched-sampling-estimator-submission-324497/18201)
  reports gains from antithetic and moment matching while Cho–Saul, Price and exact
  bivariate off-diagonal refinements did not help. Its stated mechanism is error
  cancellation, consistent with this repository's closure diagnostics.
- The [Phase-2 launch announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
  and current rules establish the live stack and code policy. Meaningful numerical
  work outside flopscope remains prohibited even when residual time is below the
  cap.

## Repository and evaluator audit

Read-only checks on the current clean tree:

- `official_config.py`: `73/73`.
- `p2_flop_anchor.py --gate`: exact against the stored official-package local
  subprocess report, 19 operations and zero disagreements.
- default preflight: **GREEN 12/16**, with shape parity, carrier strength,
  deployment envelope and official Phase-2 smoke explicitly skipped.
- portability: `99/99`.
- canonical minting, the score identity, failure pricing, paired arithmetic gate,
  seed refusals and vault seal all pass their current executable checks.

The key qualification is terminology. The local meter exactly matches the stored
official-package subprocess anchor at `137,600,326,656` FLOPs for v2.6.1. The
grading server charged `137,617,103,872`, an unexplained `+2**24`; v2.7 carries
almost the same offset and a 32-FLOP difference in the candidate-minus-control
delta. This is only `0.0122%` and cannot move a floor-pinned result, but “grading
server parity op by op to zero” is false. Until explained, planning near the
`0.1` boundary must include the observed server offset.

Other gaps:

- P01 consumption is **168/600**, not 144/600; 432 DESIGN seeds remain. The
  append-only ledger is authoritative.
- Windows cannot enforce the 8-GiB virtual-address-space limit. A large candidate
  needs an actual Linux `RLIMIT_AS` run.
- No Phase-2 vault exists. The Phase-1 vault is a different shape and cannot be
  reused.
- `preflight.ps1` certifies `estimator.py`, not an arbitrary archived candidate.
  Candidate-selectable certification is an IMPORTANT enforcement change and needs
  its own prediction, tests and independent review.
- The public mini and graded set are disjoint draws. Their observed agreement
  supports unbiased transfer; it does not license a fitted correction factor.

## Literature-grounded research opportunities

### 1. Late-window or age-window K3 — highest new expected value

ARC's factorized K3 cost grows quadratically in depth because layer-generated
factor blocks continue to be transformed. Starting K3 only for the final `m`
layers or retaining whole blocks by age may replace a depth-16 history with a much
shorter one. A leading-term `m^2/L^2` projection is encouraging but deliberately
not treated as a cost result.

The first experiment is truth-free: `m in {2,4,6,8}`, exact full-K3 control on the
same unregistered networks, compare final prediction drift, layerwise diagonal-slice
drift, retained rank and flopscope cost. Compare age-window retention with arbitrary
rank capping at matched retained rank. Prediction drift is not an error increment:
carry the full-K3 RMSE interval and report only the triangle-inequality bracket
`max(0, RMSE_full-D) <= RMSE_window <= RMSE_full+D`, with its population-transfer
qualification. Kill it if recent blocks do not dominate or checkpoint initialization
error makes `m<=6` K2-like.

Primary sources: [ARC paper](https://arxiv.org/abs/2605.05179),
[ARC reference implementation](https://github.com/alignment-research-center/mlp_cumulant_propagation).

### 2. Deep-ReLU-kernel carrier screening

For a stochastic function prior with kernel `K`, the prior-averaged squared
quadrature error is a kernel discrepancy. The infinite-width deep ReLU kernel is
available recursively from the arc-cosine/NNGP recurrence. Candidate point sets can
therefore be screened from their Gram matrices without target activations or
competition truth.

The concrete hypothesis is that several independently rotated small MUB blocks
retain within-block orthogonality and antipodal cancellation while producing a
more spherical cross-block Gram distribution than one rigid prefix. A single
global rotation is a null intervention because it leaves the Gram matrix unchanged.

Before using this proxy to select anything, test whether kernel energy ranks a
frozen panel of carrier designs similarly to finite-width MLP MSE. If rank
correlation is weak, retire the proxy.

Primary sources: [Cho and Saul](https://proceedings.neurips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html),
[Lee et al.](https://arxiv.org/abs/1711.00165),
[Bach](https://www.jmlr.org/papers/v18/15-178.html),
[Briol et al.](https://proceedings.mlr.press/v70/briol17a.html),
[CCKS Kerdock/MUB construction](https://doi.org/10.1112/S0024611597000403).

Finite-width theory makes the limitation explicit: the NNGP objective is a proxy,
not the target. See [Yaida](https://proceedings.mlr.press/v107/yaida20a.html) and
[Eldan, Mikulincer and Schramm](https://proceedings.mlr.press/v134/eldan21a.html).

### 3. Exact sparse-aware K3 (W-EFF)

Fresh K3 factor blocks contain one or two identity/one-hot modes. Keep them symbolic,
fuse transforms with diagonal-slice reads, and separate fresh sparse blocks from
aged dense blocks. `W @ I` must become a view/reuse, not a dense matmul. The first
gate is exact small-shape parity and one representative width-1024 flopscope ledger.
The saving is uncertain because one-hot modes become dense after propagation.

### 4. Query-preserving sketches — bounded fallback

If exact K3 remains too expensive, sketch the contracted neuron dimension of aged
dense factors while keeping all semantic factor columns and fresh blocks exact.
Measure downstream diagonal-slice drift, never only Frobenius tensor error. Stable
rank near 1024 would make this unhelpful; the first test is a single-layer
prospective diagnostic.

Primary sources: [stable-rank approximate matrix multiplication](https://arxiv.org/abs/1507.02268),
[SRHT analysis](https://arxiv.org/abs/1204.0062), and
[TensorSketch](https://proceedings.neurips.cc/paper_files/paper/2014/file/ad178e96136f045bbb98ea3d76e25e6c-Paper.pdf).

### Low-priority or closed families

- More Price/Hermite terms within the same Gaussian covariance map.
- Marginal Edgeworth readout as a replacement for recurrent joint state.
- Generic unscented/degree-five cubature at width 1024: generic rules require
  roughly quadratic point counts.
- Exact activation-region enumeration in the full 1024-dimensional input.
- Generic Frobenius-optimal CP/Tucker/TT compression.
- Broad learned final-residual models before the mechanistic state is fixed.
- Scalar per-layer gains beyond the already measured output-scale effect.

## Candidate architecture hypotheses

Two intentionally diverse candidates should be developed:

**Conservative candidate**

```text
f32 K2 / faithful augmented K2
  -> small kernel-screened structured carrier at one checkpoint
  -> rolled-forward state correction
  -> frozen output calibration
```

**Aggressive candidate**

```text
f32 K2 prefix through roughly layer 10–12
  -> task-specific angular checkpoint and K3 initialization
  -> sparse-aware exact K3 for the final 4–6 layers
  -> tiny cross-fitted response map
  -> frozen output calibration
```

Full exact W-EFF K3 remains a valuable alternative if its measured flopscope cost
is unexpectedly low. The **conditional cross-meter target**—not a W-EFF result—is:
if the current upstream total could be divided by `5.49` under flopscope at unchanged
raw MSE, the adjusted score would be about `1.06e-8`, still roughly `1.43x` above
the board's displayed `7.4e-9`. That is why cost reduction, late-window fidelity
and K3-specific calibration are separate gates.

## Implementation order

1. **Repair continuity (ROUTINE/IMPORTANT as classified).** Reconcile the active
   docs; correct P01; distinguish local-runner and grading-server cost parity;
   preregister a candidate-selectable preflight change before touching enforcement.
2. **W-EFF price/parity (IMPORTANT, no truth).** Exact small-shape reference,
   representative per-layer `fnp` spelling, flopscope ledger, zero scalar
   materializations, residual/memory accounting. Stop if the projected full path
   cannot reach `<=0.35 B`.
3. **Late-window K3 imitation (IMPORTANT, no label).** Freeze `m={2,4,6,8}` and
   checkpoint initialization arms; compare to full K3. Treat drift as an imitation
   diagnostic and report the triangle-inequality RMSE bracket with anchor uncertainty.
   Continue only if the preregistered drift diagnostic and a plausible `<=0.20 B`
   integrated-cost gate pass.
4. **Kernel carrier proxy (IMPORTANT then STRATEGIC).** Build the truth-free kernel
   objective and frozen carrier panel; validate proxy rank correlation on DESIGN
   only after the contract and Samuel's prediction.
5. **K3 calibration/response (STRATEGIC).** Fit O1-style renormalisation,
   diagonal/trace K4 readout and a tiny per-neuron residual response separately,
   with disjoint fit/evaluation networks and rolled-forward evaluation.
6. **Integration (STRATEGIC).** Compare exact W-EFF K3, windowed K3, carrier plus
   augmented K2, and the supported hybrid on a new single-use block using paired
   arithmetic score differences and full counted cost.
7. **Deployment and nomination (STRATEGIC).** Three official-subprocess soaks,
   zero scalar materializations, Linux memory enforcement, public CONFIRM, an
   authorised Phase-2 vault if Samuel chooses to create one, exact hashes, full
   candidate-selectable preflight, and explicit approval before either submission.

## Primary competition sources

- [Phase-2 leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2)
- [Rules v14](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules)
- [Resources](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents)
- [Insights](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/insights)
- [Phase-2 launch](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
- [Puffi Phase-1 write-ups](https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175)
