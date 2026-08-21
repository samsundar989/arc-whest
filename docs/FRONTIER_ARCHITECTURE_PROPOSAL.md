# Frontier architecture and execution plan — rev 2 — 2026-08-20

**Status:** evidence-backed, non-binding proposal. This document records the
2026-08-20 repository, leaderboard, forum, public-submission, and literature audit.
It does **not** supersede `CURRENT_STATE.md`, `GAME_PLAN.md`, `VALIDATION.md`, a
registered Slice Contract, or a closed-lane verdict. Thresholds below are proposed
checkpoint criteria; each strategic experiment must freeze its actual gate in
`experiments/LOG.md` before touching data.

**Safety boundary:** this document authorizes no edit to `estimator.py`, no seed
allocation or consumption, no vault touch, and no submission. Follow `AGENTS.md`.
Any strategic implementation must go through `next-slice` / `slice-contract`, and
each completed strategic slice through `review-slice` and the independent Skeptic.

## Executive decision

The current Sobol + rank-32 rotation + closure-guided pruning estimator is a strong,
honest conservative baseline. It is not a plausible winning endpoint through more
rank, threshold, sample-count, or matrix-multiplication tuning.

The highest-upside architecture is:

> **a Kerdock/MUB angular carrier plus an independently cross-fitted,
> network-adaptive signed correction targeted at the low-dimensional late-layer
> error subspace.**

The crucial research question is no longer "which fixed point set is best?" It is:

> Can the network-specific layer-31 mean defect, or an equivalent basis-reweighting
> direction, be estimated from the weights and a small independent pilot without
> using benchmark targets?

The public evidence says that direction has enough capacity to explain nearly the
entire gap between fixed Kerdock and the Phase-1 leader. No public team has yet
shown a deployable target-free estimator of it.

## What is established, inferred, and conjectured

### Established by this repository

- The current graded record is submission 327743 / v2.4.1 at adjusted score
  `2.6314e-7`, raw final MSE `1.3323e-6`, multiplier `0.19896`, and zero failures.
- `estimator.py` is v2.6.1: built and green, projected around `2.405e-7`, but not
  graded or promoted.
- Approximately 99.4% of the present error is sampling variance; the measured bias
  floor is negligible at the current sample count.
- Sobol, the response-aligned rotation, and several low-order controls overlap.
  Their standalone gains cannot be multiplied.
- The true degree-4/5 carrier experiment measured `1.4775x` accuracy gain but the
  available 131,072-row construction is over budget at `C/B = 1.3509`.
- The current analytic centre is not accurate enough for the proposed mid-layer
  control: centre error already exceeds sampling standard error at layer 2 and
  worsens with depth.

### Established by public reproducible work

- A complete 129-basis Kerdock/MUB construction has 66,048 signed directions in
  dimension 256. Kuperberg's theorem gives a degree-5 spherical cubature at the
  same `2^16 + 2n` count using a strength-5 Kerdock array plus axial points.
- A fully published K129 implementation scored adjusted `1.1433e-7`, raw
  `2.2068e-7`, with 140.7B effective compute and zero failures.
- Target-aware nonnegative reweighting of the 129 realized basis endpoints removed
  95.64% of pooled raw error under a tight weight cap.
- Fitting those weights on half of the final output coordinates and scoring the
  unseen half retained 81.82% raw reduction. The correction is therefore not merely
  129-parameter interpolation.
- The best tested target-free analytic proxy removed only 1.11%; its cosine with
  the required basis-weight delta was 0.018.
- Dense target-side arc-cosine features can recover most of the correction, but
  score-compatible generic query sampling cannot estimate their mean accurately.
  The missing sufficient statistic was localized to the 256-dimensional layer-31
  activation-mean defect.

Primary public evidence:

- live board and #1: <https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-1>
- field frontier and failure map: <https://discourse.aicrowd.com/t/phase-1-solution-ideas-for-phase-2-and-the-honest-public-frontier-all-methods-combined-and-explained/18193>
- technique census: <https://discourse.aicrowd.com/t/a-technique-census-of-phase-1-what-the-mathematics-is-doing-what-walls-it-hit-and-whats-still-open/18157/10>
- reproducible K129 package: <https://github.com/AndreiBulzan/arc-whitebox-submission-and-documentation>
- reweighting receipt: <https://github.com/AndreiBulzan/arc-whitebox-submission-and-documentation/blob/main/ledger/experiments/r90_late_basis_calibration_20260809/VERDICT_BF_XCORR_LMCBR_E1_E2_R1_20260809.md>
- mean-localization receipt: <https://github.com/AndreiBulzan/arc-whitebox-submission-and-documentation/blob/main/ledger/experiments/k129_quenched_arccos_teacher_20260809/VERDICT_QUERY_SCALING_AND_MEAN_LOCALIZATION_R3_R6_20260809.md>
- Kuperberg cubature: <https://arxiv.org/html/math/0402047>
- ARC cumulant propagation: <https://arxiv.org/abs/2605.05179>
- QMC control functionals: <https://arxiv.org/abs/1501.03379>
- StackMC / sample-split controls: <https://arxiv.org/abs/1606.02261>
- ReLU spherical harmonics: <https://arxiv.org/abs/1905.12173>
- Stein control functionals: <https://arxiv.org/abs/1410.2392>

### Strong inference, not disclosure

The current #1 entry has adjusted score `1.845e-8`, raw MSE `3.633e-8`, a
`0.5082` multiplier, final-layer-only output, and zero failures. Its roughly 5--6x
raw advantage over fixed Kerdock is numerically close to the 5.5x capacity retained
by held-output adaptive basis reweighting. This is evidence that a lawful
target-side contraction may explain the leader; it is **not** evidence of the
leader's actual implementation.

### Frontier conjecture

A small independently rotated pilot can learn a network-conditioned signed control
in a low-dimensional downstream-error basis, then correct a separately rotated
Kerdock/MUB main carrier without target leakage or fitting bias.

This is supported by control-functional, StackMC, spherical-harmonic, and Stein
theory. It has not been demonstrated for this deep ReLU estimator.

## Architecture

```text
                         Phase/config manifest
                                  |
              exact radial identity, only if bias-free homogeneous
                                  |
            randomly rotated 129-basis Kerdock/MUB angular carrier
                                  |
            exact shallow repairs + conservative structural routing
                                  |
       regular dense suffix: Tichavsky / one-level Strassen / grouping
                                  |
        +-------------------------+--------------------------+
        |                                                    |
  main endpoint estimate                         independent pilot cloud
  and basis summaries                            and late summaries
        |                                                    |
        +------------ network-conditioned control fit -------+
                                  |
       signed harmonic correction in a ~6D late-error basis Q
                                  |
                         final-layer prediction
```

Maintain two deliberate products throughout:

1. **Conservative:** complete structured Kerdock/MUB carrier with exact identities,
   no learned or adaptive correction.
2. **Aggressive:** the same carrier plus an independently learned signed correction;
   shallow Stein or compressed K3 is added only if it clears its own gate.

### A. Phase/config adapter

Before any carrier work, record width, depth, biases, input distribution, MLP count,
budget `B`, residual price, multiplier floor, package versions, and nomination rule.

- Biases invalidate the exact radial reduction.
- Non-Gaussian inputs require a new angular/radial derivation.
- Width changes require rebuilding the Kerdock/BCH construction.
- Width/depth changes invalidate the current pruning and basis-rank constants.
- Cost-table changes require new operating points; Phase-1 FLOP projections are not
  Phase-2 evidence.

### B. Main angular carrier

Use complete Kerdock/MUB-129 at `129 x 256 x 2 = 66,048` signed directions for the
Phase-1 shape, or the exact Phase-2 analogue. Randomization must preserve the design
and provide independent pilot/main rotations.

The implementation should expose carrier generation, radial handling, propagation,
and readout separately. This makes fixed MUB, Kuperberg/Stroud weights, BCH, and any
future design comparable under one execution engine.

### C. Structure-aware propagation

Reproduce only identities or conservative reductions with dense-reference parity:

- exact radius for homogeneous bias-free networks;
- FWHT/structured first layer where the carrier permits it;
- exact shallow layer repairs;
- dead/on/kink routing with deterministic fill only when the error gate passes;
- regular row groups and common prefixes, not random-access packing;
- rank-40 Tichavsky or one-level Strassen where real counted cost pays;
- final-layer-only output when only that layer is scored.

Cost engineering is secondary to establishing the statistic. Do not optimize a
carrier variant whose raw estimator has not passed.

### D. Low-dimensional error basis

Build a weight-computable output basis `Q in R^(width x q)`, initially `q ~= 6`,
from the downstream Lyapunov/adjoint operator or a reproducible equivalent. Measure:

- fraction of fixed-carrier error energy captured by `Q`;
- stability across networks and output-coordinate folds;
- whether the same `Q` preserves target-aware basis-reweighting capacity; and
- cost and numerical condition.

If six dimensions do not transfer, sweep a small preregistered ladder such as
`q in {3, 6, 12, 24}`. Do not select `q` on the public split.

### E. Sample-split harmonic control

Let `F(u)` be the final activation vector. Choose weight-conditioned directions
`v_j` from the adjoint image of `Q` or sensitivity-ranked first-layer directions.
Construct exact-zero spherical controls

```text
phi_j(u) = C_l^((d-2)/2)(v_j^T u)
```

using Gegenbauer degrees:

- degree 4 as an audit for a 128-basis Kerdock carrier;
- degree 6 for a completed degree-5 129-basis carrier;
- degree 8 only after the lower degree has been measured.

Fit coefficients on pilot rotation `A` only:

```text
C_A = argmin_C || Q^T F(A) - C Phi(A)^T ||^2 + lambda ||C||^2
```

Apply to independent main rotation `B`:

```text
mu_hat = mean_B F - Q C_A mean_B phi
```

Conditional on the pilot, the main randomized control has zero expectation. The
independence boundary is load-bearing: output folds or leave-one-basis folds from
the same rotation are useful diagnostics but do not prove an unbiased deployable
estimator.

Initial implementation grid, to be reduced and frozen by a Slice Contract:

- pilot rows `P in {512, 1024, 2048, 4096}`;
- controls `m in {6, 16, 32, 64}`;
- degree 6, then degree 6+8;
- ridge, capped signed weights, and positivity-constrained weights;
- random, adjoint, and sensitivity-ranked direction families.

The computation must use the low-rank feature system. A full 66k-by-66k isotropic
Bayesian-quadrature solve is both unaffordable and symmetry-blind.

### F. Secondary controls, only after the main hypothesis is tested

1. **Shallow Stein control.** At depth 6--8, use the exact zero-mean identity
   derived from Gaussian/spherical Stein plus homogeneity. Distribute tangent
   directions across point groups so overhead is approximately `c/L`, not six full
   tangent networks.
2. **Compressed shallow K3 centre.** Recompress the canonical K3 factor sum by
   tensor interpolative decomposition or another separation-rank method. This is
   distinct from the closed covariance-eigenbasis projection.
3. **Small independent mean-defect audit cloud.** Estimate only `Q^T` times the
   late mean defect, then use generalized-regression / multi-fidelity correction.

These are fallbacks or additions. Do not build all three before the harmonic
observability audit decides whether the primary mechanism exists locally.

## Validation blockers before frontier evidence is trusted

The core mint and counted-FLOP paths are strong. The decision layer still has
load-bearing mismatches:

1. `VALIDATION.md` binds promotion to the geometric mean of ratios even though the
   official objective is the arithmetic mean of per-MLP scores. Repo evidence shows
   the two can select opposite winners.
2. `paired_compare()` exposes a percentile bootstrap while project guidance calls
   for studentised-t or basic-bootstrap inference.
3. Preflight omits shape parity, canonical scorer, seed-ledger negative tests, and
   full vault hash verification.
4. Seed-ledger scanning can miss hardcoded reuse after a seed has already been
   claimed; dynamic block access is not attributable; completed D18 artifacts have
   blank hashes.
5. Vault and coverage evaluators do not reproduce the grader lifecycle: they create
   a new estimator per MLP and do not call one shared `setup`.
6. Vault truth can be opened before the touch is durably recorded, so interruption
   can create an unlogged touch; previous per-touch prediction evidence is
   overwritten.
7. The five-point residual model extrapolates about 1.75x in operation count into a
   new Strassen mix and factorizes aggregate means instead of scoring per MLP.
8. Public `CONFIRM` is still adaptive if a "survives" result changes selection.
9. Several truth generators mask sample seeds to 31 bits although official
   whestbench uses the full 32-bit value; this is unbiased but not bit-reproducible.
10. One truth replica leaves label-noise bias and a cross-term in decisive MSEs; two
    independent truth replicas enable a cross-product estimate.

Required target architecture for evaluation: one canonical lifecycle evaluator that
emits immutable per-MLP evidence rows containing candidate/truth hashes, exact seed
claims, backend provenance, raw and noise-corrected MSE, counted operations, residual
model ID and uncertainty, exact failure semantics, and the canonical arithmetic
competition score.

Changing the binding gate or validation protocol is **STRATEGIC**. Editing the
instrument is **IMPORTANT**. This proposal does not make those changes.

## Checkpoint protocol

Every checkpoint ends in exactly one status:

- `GO` — its preregistered gate passed; the next checkpoint is authorized.
- `KILL` — the mechanism failed; close it with scope stated precisely.
- `PIVOT` — evidence supports a narrower successor that was named in advance.
- `BLOCKED` — an external fact or required human decision is missing.

After every checkpoint, communicate this compact receipt:

```text
CHECKPOINT <id> — <name>
Status: GO | KILL | PIVOT | BLOCKED
Question:
What changed:
Evidence tier and seeds/touches used:
Primary arithmetic-score result and interval:
Mechanism diagnostics:
Cost/failure result:
Gate comparison:
What the evidence means:
What it does not mean:
Files/artifacts:
Next action:
Samuel decision needed: none | <one concrete decision>
```

No checkpoint may be described as "promising" without a terminal status and a gate
comparison.

## Execution checkpoints

### CP0 — Phase and decision-layer readiness

**Question:** Do we know the real phase configuration, and can the evaluator choose
the correct estimator without optimistic lifecycle, aggregation, seed, or failure
errors?

Deliverables:

- live phase/config manifest;
- arithmetic scorer and t/basic paired intervals;
- canonical grader-lifecycle evaluator;
- full preflight regression wiring;
- finalized seed/artifact ledger transactions;
- transactional, immutable vault evidence;
- truth backend provenance and two-replica option;
- residual-cost sensitivity bands, still E1 until graded telemetry.

Proposed exit criteria:

- parity with installed whestbench on sacrificial data;
- all targeted tests and preflight green;
- a regression fixture where arithmetic and geometric gates choose opposite arms and
  the canonical path chooses the arithmetic winner;
- zero seed-ledger bypasses in negative tests;
- no seed block or vault touch spent merely constructing the instrument.

### CP1 — Reproduce the fixed-carrier frontier

**Question:** Can this repo reproduce a clean, score-compatible K129 carrier before
attempting novel adaptation?

Required ablations under one engine:

1. current carrier;
2. fixed K129 carrier alone;
3. K129 plus exact radius/early identities;
4. K129 plus conservative routing;
5. K129 plus cost engine.

Proposed exit criteria for the Phase-1 shape:

- carrier strength/invariance checks pass exactly;
- zero failures and dense-reference parity for exact rewrites;
- at least 4x raw-MSE improvement over the current carrier, or raw MSE in the
  reproducible public K129 regime (`~1.7e-7` to `2.4e-7`), with truth noise quoted;
- effective compute remains score-compatible;
- paired arithmetic score improves on fresh networks with interval excluding zero.

If raw accuracy passes but cost fails, `PIVOT` to cost engineering. If raw accuracy
fails, `KILL` the implementation spelling before optimization.

### CP2 — Reproduce adaptive-reweighting capacity

**Question:** Does the published target-aware basis direction exist on our freshly
minted networks, and is it low-dimensional and shared across unseen outputs?

Diagnostics:

- capped nonnegative and signed 129-basis oracle weights;
- even/odd and random output-coordinate cross-fitting;
- `Q` dimension ladder;
- correction energy, cosine, effective sample size, and row tails.

Proposed exit criteria:

- held-output cross-fit removes at least 50% of fixed-carrier raw error on both
  fresh halves; 70% is the strong-pass target;
- `Q` retains at least 70% of full-output cross-fit gain at a stable dimension;
- weight caps/ESS exclude degenerate interpolation;
- no public split is used for selection.

Failure means the public capacity result does not transfer to our distribution or
carrier and the adaptive branch is killed before a production proxy is built.

### CP3 — Modal observability audit

**Question:** Do target-free harmonic moment errors point in the oracle correction
direction?

Compare degree 4/6/8 controls and random/adjoint/sensitivity directions using frozen
network and output folds.

Proposed exit criteria:

- cross-fitted controls explain at least 50% of the `Q`-projected oracle correction
  energy, or deliver correction-direction cosine at least 0.5 on both fresh halves;
- the selected direction family beats random directions;
- results are stable to fold assignment and ridge range.

If capacity exists but the named harmonic controls fail, `PIVOT` to CP5 rather than
tuning them indefinitely.

### CP4 — Deployable independent-pilot control

**Question:** Does the target-free, independent pilot/main estimator improve the
actual competition objective after its full cost?

Proposed exit criteria:

- at least 1.5x raw-MSE improvement over fixed K129;
- less than 10% added effective compute for the first promotion, or a larger raw
  gain that improves the exact per-MLP score after full cost;
- paired arithmetic score interval excludes zero on the preregistered decision
  block and improves on both halves;
- zero failures, bounded signed weights, stable p95/max tails;
- an independence test demonstrates that shared-rotation fitting is not silently
  used by the production path.

### CP5 — Fallback control ladder

Run only if CP3/CP4 gives `PIVOT` rather than `KILL`:

1. shallow Stein control;
2. compressed K3 separation-rank diagnostic;
3. independent late-mean audit cloud.

Each mechanism gets one capacity gate and one deployability gate. Continue only if
the measured score gain exceeds its added effective cost by at least 25% and
transfers across fresh halves.

### CP6 — Composition and robustness

**Question:** Do the passing mechanisms compose, and will the private rerun remain
safe?

Deliverables:

- factorial ablation of carrier, routing, cost engine, and adaptive correction;
- 300--600 fresh-network robustness run with exact failure scoring;
- private-suite bootstrap/simulation at the actual competition `n_mlps`;
- one-shift deployed distribution, p95/p99/max, and ground-truth noise;
- conservative and aggressive candidate hashes frozen separately.

Exit criteria:

- arithmetic score gain survives composition rather than multiplying isolated
  gains;
- aggressive candidate has no worse catastrophic tail/failure risk;
- conservative candidate remains a valid independent nomination;
- winning target is evaluated against the live Phase-2 board with uncertainty and
  headroom, not only a point estimate.

### CP7 — Grader telemetry and nomination

No submission is authorized by this checkpoint text. Required order is always:

1. `whest validate`;
2. clean mini run with `n_failed_mlps: 0`;
3. present candidate hash, local evidence, and exact question to Samuel;
4. obtain Samuel's explicit approval;
5. submit and ingest grader telemetry into the canonical evidence bundle.

Nominate the conservative and aggressive candidates only if the organizer confirms
two selections. If only one is allowed, use private-rerun expected score and tail
risk, not public rank, to choose.

## Kill list

Do not reopen without new external evidence:

- generic Sobol whitening, sphere, tent, or sample-count tuning;
- degree-3 designs on top of the current rotation;
- current K2 closure as a mid-layer control centre;
- generic offline final residual regression;
- isotropic full-kernel quadrature on the symmetric K129 frame;
- target-selected frame subsets or weights;
- small ordinary query clouds for the dense layer-31 mean;
- random-access sparse packing;
- full depth-32 K3/K4 before shallow capacity and cost evidence;
- multiplying isolated gains instead of measuring composition.

## Immediate next move

Reconcile this proposal with the current Phase-2 state and complete **CP0 only**.
The first strategic change should repair the binding arithmetic decision gate; the
first instrument change should create the canonical grader-lifecycle evaluator.
Do not combine the gate, evaluator, seed ledger, vault transaction, and K129 carrier
into one slice. Each can silently invalidate the evidence produced by the next.
