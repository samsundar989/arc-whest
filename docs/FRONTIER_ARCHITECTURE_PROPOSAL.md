# Frontier architecture proposal — 2026-08-19

**Status:** evidence-backed proposal; implementation has not started. This document
does not supersede `CURRENT_STATE.md`, `GAME_PLAN.md`, or any registered gate. It
records the external Phase-1 audit and the implementation sequence to review once
the Phase-2 specification is published.

**Safety boundary:** do not spend D17, touch the vault, edit `estimator.py`, or
submit anything merely because this proposal exists. The validation repairs in
Stage 0 come first and must follow the work classifications in `AGENTS.md`.

## Executive decision

The current Sobol + rank-32 rotation + pruning estimator is a strong, honest
baseline and a useful execution kernel. It is not a plausible winning architecture
through incremental tuning.

The recommended target is a compositional estimator:

> complete Kerdock/MUB or BCH angular design, exact radial integration,
> structure-aware propagation, a shallow accurate analytic centre, and a
> mid-network control variate.

The frontier-breaking research bets are (a) improving the analytic centre in the
downstream-sensitive subspace and (b) learning network-conditioned frame weights or
frame allocation without target leakage.

## Why the present architecture has plateaued

The working `estimator.py` performs:

1. full-covariance Gaussian closure for layer means and firing probabilities;
2. a rank-32 input rotation from the linearised end-to-end map;
3. closure-guided dead-neuron pruning and deterministic mean fill;
4. digitally shifted Sobol Gaussian sampling;
5. one-level Strassen propagation in the unpromoted v2.6 working file; and
6. a sampled final-layer mean.

Repository evidence says the remaining error is approximately 99.4% sampling
variance. Whitening, radial projection, tent transforms, degree-3 designs, and
layer-1 controls mostly substitute for low-order variance already removed by Sobol
plus rotation.

The optimistic composition of the remaining known levers is still insufficient:

```text
current adjusted score              2.631e-7
/ optimistic v2.6 cost gain         1.068
/ external degree-4/5 ceiling       1.46
= optimistic projected score        1.69e-7
```

The best published and explained Phase-1 system scored `9.10e-8`. The public #1
scored `1.84e-8`, but no reproducible method currently explains it. Winning work
should target fresh-network transfer, not reverse-engineer unexplained public rows.

## External frontier evidence

| result | adjusted score | relevant mechanism |
|---|---:|---|
| Team Puffi | `9.10e-8` | complete 66,048-direction Kerdock/MUB design, exact radius, shallow analytic reference through layer 9, online layer-9→32 ridge control, structure-aware execution |
| BCH strength-4 OA | `1.23e-7` | near-complete BCH angular carrier, antipodes, homogeneity, dead/on/kink routing, FWHT |
| mliston | `1.334e-7` | analytic chain through layer 18, layer-22 control, frozen 52k-parameter GRU centre corrector |
| Kerdock 5-design | `1.439e-7` | 66,048 directions, FWHT first layer, Strassen/Winograd execution |
| this repository, 327743 | `2.631e-7` | Sobol RQMC, rank-32 rotation, pruning |

Forum reports are contestant evidence, not organiser-verified theorems. Reproduce
each mechanism locally before relying on its claimed gain.

Primary links:

- Team Puffi: <https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175>
- BCH/OA: <https://discourse.aicrowd.com/t/phase-1-write-up-my-approach-write-up-for-the-submission-327651/18173>
- Kerdock: <https://discourse.aicrowd.com/t/phase-1-write-up-metered-dead-compacted-kerdock-cubature-with-exact-compiler-reductions-submission-326680/18149>
- Mid-layer control: <https://discourse.aicrowd.com/t/phase-1-write-up-monetizing-the-approach-to-the-moment-propagation-frontier-an-intra-network-control-variate-and-a-measured-map-of-the-wall-behind-it-submission-326024/18154>
- Corrected centre + control: <https://discourse.aicrowd.com/t/phase-1-write-up-bias-variance-tradeo-across-control-layers-submission-327723/18170>
- ARC cumulant method: <https://arxiv.org/html/2605.05179>
- Kuperberg degree-5 cubature: <https://arxiv.org/abs/math/0402047>

## Proposed architecture

```text
                         network weights
                              │
                cheap closure / small pilot pass
                ├── dead / on / kink classification
                ├── row and prefix grouping
                ├── downstream sensitivity map
                └── fixed control-layer policy
                              │
   exact-radius complete Kerdock/MUB or BCH angular bank
                              │
         FWHT first layer + structured dense propagation
            + conservative pruning + depth-1/2 Strassen
                              │
                 capture h_c and h_32 together
                              │
weights ──> shallow K2/K3 analytic anchor at c ≈ 7–10
                              │
               optional frozen trajectory corrector
                              │
                online regularised response c → 32
                              │
                   shrinkage CV correction
                              │
                      final-layer estimate
```

### Component A — complete angular carrier

Use a true degree-4/5 construction, not the existing degree-3 proxy:

- complete real Kerdock/MUB-129: `129 × 256 × 2 = 66,048` signed directions; or
- BCH strength-4 orthogonal array at matched cost.

For the Phase-1 bias-free Gaussian problem, replace sampled radii by the exact mean
Gaussian radius. This is an identity from positive homogeneity, not an
approximation. Randomise only in a way that preserves the design's exactness and
supports repeated fresh-network evaluation.

Implementation should expose MUB and BCH behind the same propagation and accounting
engine so the comparison is about the point set rather than unrelated engineering.

### Component B — structure-aware propagation

Candidate exact or conservatively gated reductions:

- FWHT-based first layer for Kerdock/MUB structure;
- antipodal layer-2 identity where algebraically valid;
- pilot-derived dead/on/kink masks with deterministic omitted-neuron mean fill;
- row grouping and common prefixes rather than random-access sparse packing;
- one-level or depth-2 Strassen/true Winograd on regular dense blocks;
- final positive-neuron bypass where it is an exact identity.

Every rewrite must have dense-reference parity tests. Local residual wall time may
rank gross failures but cannot decide close cost variants; only grader telemetry can.

### Component C — shallow analytic centre

Do not attempt a pure analytic answer at depth 32. Produce an accurate centre at a
fixed shallow layer, initially `c ∈ {5, 7, 9, 11}`.

Compare:

1. current calibrated K2 closure;
2. exact Gaussian full-covariance update;
3. shallow full-rank K3 as an accuracy oracle;
4. compressed K3 using a covariance eigenbasis;
5. K3 HOSVD/mode-unfolding bases;
6. downstream-Jacobian-weighted K3 bases; and
7. bases transported between layers instead of recomputed independently.

The existing probe shows full-rank K3 is useful at depth 8 while low ranks fail.
That evidence rejects the current compression basis/rank, not the K3 mechanism.

The deployment metric is not standalone analytic final-layer MSE. For control layer
`c`, report:

```text
sampling error                     e_c
analytic-centre error              e_A
projected centre error             Rᵀ e_A
end-to-end controlled score        score_CV
```

The centre is deployable only when its projected error is smaller than the sampling
error removed by the control.

### Component D — mid-network control variate

Use aligned point trajectories:

```text
mu_hat_32_CV = mean(h_32)
              - beta * R.T @ (mean(h_c) - mu_hat_c_analytic)
```

`R` is an online regularised response fitted from the same aligned rows; `beta` is a
fixed or pre-registered shrinkage rule. Initially reproduce Team Puffi's single
layer-9 control before trying multiple control layers.

### Component E — optional learned centre corrector

Only after the mechanistic centre and control are working:

- train on newly generated MLPs, never the public split;
- split by whole MLP, not neurons or layers;
- expose the model to the same noisy deploy-time features it will receive;
- train on rolled-forward trajectories, not one-step residuals;
- optimise projected centre error or end-to-end controlled loss;
- freeze weights before the decision block; and
- bundle all weights in the single-file submission.

A small recurrent or permutation-equivariant model is preferable to a generic final
residual predictor. The latter was already measured at `R² ≈ 0.012` in this repo and
is not reopened by this proposal.

### Component F — network-conditioned frame allocation

This is the highest-upside and highest-risk lane. Learn frame weights or a subset
policy from shallow, target-free information such as:

- analytic-versus-sampled discrepancies at early layers;
- estimated downstream response;
- kink density and gate margins;
- frame-level shallow residual summaries; and
- downstream-sensitive cumulant projections.

Direct selection against evaluation targets is leakage. Use cross-fitting and compare
against the complete-frame carrier on untouched MLPs.

## Stage 0 validation blockers

Do not begin a strategic estimator experiment until these are repaired.

### P0.1 — decision objective mismatch

The official scorer arithmetic-means per-MLP adjusted scores. The repo currently
gates on the geometric mean of per-MLP ratios (`VALIDATION.md:218`). Those estimands
can select opposite winners on a heavy-tailed suite.

Required canonical objective:

```text
score_i = mse_i * max(floor, (flops_i + lambda * residual_i) / B)
objective = arithmetic_mean(score_i)
```

Primary inference should be a paired arithmetic score difference with a paired
bootstrap or studentised interval over MLPs. Retain geometric log-ratio only as a
secondary stability/mechanism diagnostic.

### P0.2 — vault evaluates the v2.4 mirror

`local_sim/vault.py:227-260` imports `run_v24` and a fixed v2.4 cost model. It does
not evaluate v2.6 predictions or Strassen cost. It also exposes `--dry-run`, which
reads truth without recording a touch.

Required repair:

- execute an explicitly selected candidate through the canonical scorer, or declare
  the vault accuracy-only;
- remove unlogged truth evaluation;
- enforce touch maximum/reason counts and atomic counter/log updates;
- bind a touch to the estimator/archive hash; and
- prove mirror parity on a sacrificial non-vault block first.

### P0.3 — v2.6 odd-width correctness bug

`estimator.py:538` removes an odd survivor before `estimator.py:579` discovers that
Strassen is impossible for an odd global width. The code then runs dense propagation
after needless trimming. A read-only 65×4 parity check against v2.4.1 produced final
maximum relative difference `0.2256` and prediction MSE `0.1037`.

Determine whole-chain Strassen eligibility before changing alive sets. Add widths
63, 65, 96, 127, 192, 255, and 257 plus depth 1 to the invariant matrix. Fallback
paths should require parity with v2.4.1, not merely finite output.

### P0.4 — seed consumption is not enforced

`seed_blocks.py` says D16 was never drawn, while committed artifacts use all 600 D16
seeds. D17 is described as both reusable and single-use. The registry detects
interval overlap but not consumption.

Required repair:

- record D16 as spent design data;
- lock D17 or allocate a fresh decision block after repair;
- add an append-only range-consumption ledger with artifact hash and commit;
- make tools hard-fail on spent/unregistered ranges; and
- have preflight detect duplicate consumption.

## Evaluation protocol for the new architecture

Every candidate/control comparison must:

1. use common random numbers only across arms within an MLP;
2. use independent truth and shift streams across MLPs, or a crossed bootstrap;
3. compute the exact per-MLP competition score before aggregation;
4. preserve MSE/cost correlation;
5. report arithmetic mean difference and private-rerun tail risk;
6. report the one-shift deployed distribution as well as multishift expectation;
7. subtract or propagate ground-truth noise in the candidate-control interval;
8. re-sweep sample count for every biased hybrid;
9. fit nothing on the public split; and
10. treat local residual time as E1 until graded telemetry exists.

## Implementation sequence and gates

### Stage 0 — repair the instrument

Deliverables:

- canonical scorer;
- v2.6 shape/fallback fix and expanded invariants;
- enforceable seed ledger;
- candidate-aware, non-bypassable vault path;
- reproducible v2.6 accuracy/cost gate from checked-in code.

Exit gate: targeted tests and project preflight green; independent review complete;
no decision block or vault touch consumed during construction.

### Stage 1 — reproduce the known carrier frontier

Build MUB and BCH under one engine. Compare design alone, design + current rotation,
design + pruning, and design + pruning + structured multiplication. Do not assume
standalone gains compose.

Exit gate: fresh-network paired arithmetic improvement, zero failures, algebraic
parity for every exact reduction, and raw MSE in the published carrier regime at a
competitive effective-compute model.

### Stage 2 — reproduce the layer-9 control

Implement one fixed shallow centre, online ridge response, and shrinkage.

Exit gate: at least 15% end-to-end arithmetic-score improvement with interval
excluding zero, plus an explicit projected-centre-error decomposition.

### Stage 3 — improve the centre

Run the K3 basis ladder and only then test a frozen trajectory corrector.

Exit gate: centre beats sampled error at a useful control layer and the deployed CV
gain survives untouched networks. A better standalone closure that fails this
deployment test does not pass.

### Stage 4 — frontier-breaking experiments

Test joint shallow controls and network-conditioned frame weighting. Run factorial
combinations rather than multiplying isolated gains.

Exit gate: fresh-network arithmetic score below the reproduced complete-carrier
control with private-rerun tail risk no worse.

### Stage 5 — final candidates

Maintain deliberately different candidates:

1. **conservative:** complete structured design carrier without learned correction;
2. **aggressive:** carrier + analytic/learned centre + mid-layer control and, if
   validated, network-conditioned frame allocation.

Use the public split only as confirmation. Nomination follows fresh validation and
the organiser's final clarification on whether one or two submissions are allowed.

## Phase-2 adaptation rules

Before Stage 1, record the actual Phase-2 width, depth, biases, input distribution,
budget, lambda, multiplier floor, MLP count, and dependency versions.

| Phase-2 change | consequence |
|---|---|
| width > 256 | current estimator falls back to analytic-only; new carrier and Sobol storage must be rebuilt |
| width/depth changes | refit pruning and rotation constants; rebuild MUB/BCH construction; do not reuse 256×32 gates |
| materially smaller `L/n` | analytic K2/K3 becomes more attractive and may deserve more compute |
| biases introduced | positive-homogeneity/exact-radius reduction is invalid |
| non-Gaussian input | spherical carrier and radial identity must be re-derived |
| cost model/floor changes | sample count and the bias/cost trade-off must be re-optimised |

## Explicit non-goals

- no more generic Sobol whitening, sphere, tent, degree-3, or sample-count tuning;
- no full depth-32 K3/K4 implementation before shallow-control evidence;
- no public-leaderboard fitting or target-selected frame subsets;
- no uninstrumented FLOP-accounting workaround;
- no autonomous optimisation loop while cost evaluation remains E1; and
- no attempt to explain the public #1 by speculation.

## Immediate next move

Once the Phase-2 specification is known, begin **Stage 0 only**. The first
implementation slice should create the canonical arithmetic scorer and regression
tests demonstrating the present geometric/arithmetic sign difference. Do not fold
all four P0 repairs into one unreviewable change.
