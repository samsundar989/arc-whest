# GAME_PLAN.md — Phase-2 frontier research program

**Authoritative from 2026-08-28.** This is the sole active research roadmap. Its
hybrid is a falsifiable candidate architecture, not an established winner.
`CURRENT_STATE.md` says what has happened; `VALIDATION.md` governs every
measurement; `PHASE2_FRONTIER_AUDIT_2026-08-28.md` is the dated competition,
repository and literature evidence behind this revision. Older Kerdock-first and
pre-launch plans are historical.

## 1. Objective and current score geometry

Win Phase 2 while obeying the intended flopscope computation rules and preserve a
contribution-prize-quality mechanistic account.

For each random no-bias `1024 x 16` ReLU MLP, predict all post-ReLU means under
`x ~ N(0, I)`. The score is

```text
mean_MLP[final_layer_MSE * max(0.1, FLOPs / 2**41)]
```

WhestBench 0.16 / flopscope 0.12 is live. Lambda is zero, residual time is a
hard `0.4 s/MLP` cliff, total wall is 120 seconds, setup is 5 seconds and the
solution-memory limit is 8 GiB.

Live board snapshot at **2026-08-28 16:40 UTC** (display precision):

| system | adjusted | raw final MSE | multiplier | status |
|---|---:|---:|---:|---|
| Puffi, rank 1 | `7.4e-9` | `2.6e-8` | `~0.285` | public feedback |
| best raw, jamespayor | `1.11e-8` | `2.33e-8` | `~0.476` | public feedback |
| neopok919 v2.7, rank 41/67 | `1.374e-7` | `1.374e-6` | `0.1` | graded, zero failures |
| exact factorized K3 | projected only | `3.9249e-8` | flopscope cost unknown | organiser mini; GT-noise floor `7.478e-11`; not deployable |

The public gap is `18.6x` adjusted and `52.9x` raw. v2.7 is already floor-pinned;
cheap calibration alone cannot win. The top group shares raw MSE around
`2.3e-8`–`3.5e-8` and trades it against cost.

### Outcome targets

- **Frontier milestone:** honest adjusted score `<=1.2e-8`, zero failures.
- **Winning target:** `<=6e-9` on fresh offline evidence, leaving margin below the
  current moving board.
- **Nominee target:** two complementary frozen candidates, both clean under the
  official subprocess, Linux memory enforcement, public CONFIRM and the final
  operational gate.

These are engineering goals. Promotion is always the preregistered paired
arithmetic score difference returned by `competition_score.decide()`.

## 2. What is established

### Estimator and calibration

1. The shipped Phase-2 path is K2p1 covariance propagation. Width 1024 cannot
   enter the Phase-1 Sobol/rotation/pruning path.
2. v2.7 is K2 plus literal scalar row gains. It improved the graded score
   `2.8629x` and the offline projection transferred within `1.5%`.
3. Positive homogeneity makes purely multiplicative K2 state calibration an output
   scale in disguise. Variance and off-diagonal scalar gains did not transfer.
   The next correction must be neuron-specific or add state information.

### Higher-order state

1. Full factorized K3 measures `4.0047e-8` raw MSE on 24 DESIGN networks
   (GT-noise floor `1.155e-9`) and `3.9249e-8` on 100 organiser networks
   (GT-noise floor `7.478e-11`). The two readings agree within `1.99%`.
2. The current implementation costs about `1.4796 B` under the upstream Torch
   counter. That is not a flopscope price and not a deployable point.
3. Rank capping with the tested selectors loses accuracy faster than it saves
   cost. The best tested rank-`n` object drifts `5.789e-7`, `14.5x` the full-K3
   MSE. This is a measured negative, not a lower bound on all compressors.
4. Frobenius-optimal CP approximation does not target the diagonal-slice queries
   the recurrence consumes. “Preserve the diagonal” was refuted at matched rank,
   and the ALS oracle itself costs 7.9–22 budgets.
5. About `95.35%` of upstream K3 cost is in two operations. Fresh blocks have one
   or two identity/one-hot factor modes. W-EFF is an exact cost-side hypothesis.
   The `5.49x` number is a cross-meter affordability target, not a projected saving;
   no W-EFF operating point has been measured under flopscope.

### Structured sampling

1. The width-1024 Kerdock/MUB construction is algebraically valid. Dense 16 bases
   cost exactly `2**40` before control overhead.
2. The fixed 16-basis carrier is not a spherical 4-design; a carrier-aligned fourth
   moment is `21.6881x` the spherical target.
3. Public Phase-1 evidence supports Kerdock/MUB, antipodes, exact radial
   integration and checkpoint control. Phase-2 leader code remains private.
4. Ordinary IID and random-orthogonal arms lose locally, but they do not bound a
   task-specific structured carrier.

### Evaluator

1. The default gate is green on all 12 executed stages. The four full-only stages
   remain skipped and must not be described as passes.
2. Minting, scoring, failure pricing, the paired arithmetic gate, the seed ledger
   and local official-package operation accounting are strong.
3. The local official-package anchor is exact at `137,600,326,656` FLOPs. The
   grading server charged an unexplained `+2**24`; include that observed offset
   near a multiplier boundary until it is explained.
4. Windows cannot enforce the 8-GiB virtual-address-space limit.
5. P01 is DESIGN, **168/600 consumed**, 432 free. No Phase-2 vault exists.

## 3. Candidate architecture hypothesis

The program tests a **mechanistic analytic prefix plus a task-specific
checkpoint and a higher-order final window**. Sampling and analytic propagation
may be complementary; each module must survive its own falsifier before integration.

```text
weights
  |
  +--> f32 analytic prefix
  |      K2p1 -> faithful augmented K2 if supported
  |
  +--> task-specific angular checkpoint
  |      kernel-screened MUB/orthogonal blocks + antipodes
  |      -> mean / diagonal variance / selected K3 queries
  |
  +--> final-window state engine
         sparse-aware exact K3 for the final m layers
         -> tiny rolled-forward response correction
         -> frozen output calibration
```

### A. Analytic prefix

Use f32 K2 as the stable reference. Port faithful recurrent augmented K2 only after
small-shape state parity with ARC's reference. Do not represent it with the old
marginal Edgeworth oracle.

### B. Exact sparse-aware K3 (W-EFF)

Keep fresh identity-supported blocks symbolic, replace `W @ I` with reuse/indexed
operations, fuse factor transforms with diagonal-slice reads, and separate sparse
fresh blocks from dense aged blocks. Exact parity comes before cost claims.

The first operating-point gate is a representative width-1024 layer billed by
flopscope 0.12 with zero mid-computation scalar materializations. If the integrated
full path cannot plausibly reach `<=0.35 B`, do not port the full tower blindly.

### C. Late-window or age-window K3

Run K2 through the prefix and start exact K3 only for the final `m` layers, or
retain complete layer-generated blocks by age. Candidate `m` is frozen to
`{2,4,6,8}` for the first truth-free screen.

Checkpoint initializations:

- zero K3 state;
- augmented-K2 trace/harmonic state;
- selected K3 diagonal queries estimated by the structured checkpoint.

Compare with full K3 on identical networks without target labels: prediction
drift, diagonal-slice drift, age contribution, retained rank, flopscope cost and
memory. The `m^2/L^2` cost argument is a hypothesis, not a result.

### D. Task-specific angular carrier

Positive homogeneity integrates the Gaussian radius exactly. Use kernel
discrepancy under the deep ReLU NNGP as a truth-free screening proxy for angular
point sets.

Frozen panel:

- current Kerdock prefixes;
- independently rotated small MUB blocks;
- one rigid MUB family;
- ordinary orthogonal controls;
- bounded scrambled-Sobol control after exact radial separation.

A single global rotation is a null intervention because it preserves the Gram
matrix. Multiple independent block rotations change cross-block correlations and
are a real design. Before using the proxy, test whether it ranks finite-width
carrier MSE on a frozen DESIGN panel. Kill the proxy if rank correlation is weak.

### E. Response and calibration

Only after the state engine works, test separately:

1. K3-chain projection/renormalisation coefficients;
2. diagonal/trace K4 readout correction;
3. a tiny shared permutation-equivariant per-neuron residual model.

Every fitted map uses disjoint DESIGN fit/evaluation networks and is evaluated on
its own rolled-forward trajectory. Teacher-forced one-step improvements do not
promote anything.

### F. Integration and safety

The deployable file stays self-contained and grader-legal. It must return all 16
rows, preserve exact layer 1, mutate no grader arrays, perform all numerical work
through flopscope, create no scalar synchronization inside `predict`, avoid budget
summaries, and fall back safely on instability.

## 4. Same-day critical path — 2026-08-28

The goal for today is an **evidence-backed submission candidate**, not a forced
submission. The clock does not relax the measurement law. “Done today” means one
of two honest outcomes:

1. a materially stronger frozen candidate reaches the approval gate below and is
   handed to Samuel with its exact SHA and full evidence packet; or
2. the highest-value path is killed by its preregistered falsifier, the result is
   recorded, and no weaker candidate is submitted merely to meet the date.

The same-day path deliberately narrows the full research program. It answers the
cost question first, then the layer-age question, and builds only the K3 module
those answers support. The carrier and augmented-K2 tracks remain part of the ideal
architecture, but they are not rushed into today's candidate without their own
evidence.

### Today A — orient, contract, and get Samuel's predictions

- Confirm HEAD, worktree cleanliness, official package versions, current public
  entries and the unspent seed ledger without touching data.
- Write and commit the W-EFF IMPORTANT contract before profiling or benchmarking.
- Ask Samuel for the three short predictions required by `AGENTS.md`: expected
  W-EFF cost, the smallest plausible K3 window, and the minimum improvement that
  would make a third public submission worthwhile.
- Do not combine truth-free engineering and a later truth-bearing decision under
  one contract. They are separate estimands and separate stop points.

### Today B — W-EFF go/no-go, then independent review

Use `local_sim/p2_pb_rank.py` and the ARC/upstream factorized-K3 implementation as
the reference, not a rewritten mathematical surrogate. Prove small-shape parity
for fresh symbolic blocks and bill one representative width-1024 layer through
flopscope 0.12. Include casts, assembly and slice extraction. Require zero
`__bool__`, `__float__` and `__int__` materializations inside the priced path.

Continue only if the measured spelling supports an integrated `<=0.35 B` path.
That is an affordability screen, not a claim that `0.35 B` wins. If it fails, stop
the full-tower port immediately. Run default preflight, portability where relevant,
and a fresh independent Skeptic before its result controls Today C.

### Today C — late-window imitation without labels

Freeze `m={2,4,6,8}`, checkpoint initializers and drift/cost thresholds before the
probe. Compare each arm with uncapped full K3 on common networks, report the
triangle-inequality error bracket, and bill each arm. Do not claim a seed-ledger
block and do not read labels in this stage.

Proceed only when a window meets the preregistered drift threshold and has a
credible official-stack cost below the full-tower option. If no window survives,
stop: neither scalar calibration nor arbitrary K3 rank capping is an acceptable
same-day fallback. Finish the IMPORTANT synthesis and fresh Skeptic review before
building a submission candidate.

### Today D — freeze one candidate, then spend truth once

Implement the narrowest supported design in a research arm first. Only after its
math, shape and cost tests pass, write and commit a STRATEGIC promotion contract,
create a new version archive, and promote it into `estimator.py` so the existing
preflight certifies the actual ship file. Do not expand candidate-selectable
preflight during this time-box unless the current route is genuinely blocked; that
would be a separate IMPORTANT instrument slice.

Then claim one fresh SINGLE_USE Phase-2 DECIDE block through the ledger and compare
the frozen candidate with the exact v2.7 control on common random networks. Use
only `competition_score.decide()` for promotion. Report the paired arithmetic
adjusted-score difference, valid interval, raw MSE with ground-truth noise, counted
FLOPs, multiplier, failures, residual headroom and memory accounting.

The recommended same-day **engineering bar to put to Samuel before data exists**
is at least a `2x` adjusted-score point improvement over v2.7 with the valid paired
interval excluding zero. It is not binding until Samuel accepts it in the
preregistered contract. The frontier milestone remains `<=1.2e-8`; a candidate
that merely clears `2x` is stronger, not yet a winning architecture.

### Today E — deployment packet and human-only submission decision

If the decision gate passes, obtain the required independent Skeptic verdict, run
the full official deployment sequence on the promoted file, archive the exact
candidate, and produce one approval packet containing:

- estimator version, commit and SHA-256;
- paired score result and GT-noise floor;
- full counted-FLOP ledger and observed server-offset treatment;
- subprocess residual p50/p99/max and failure counts;
- memory evidence and remaining uncertainty;
- `whest validate`, clean mini run, and `preflight.ps1 -Full` transcript with its
  own `GREEN 16/16` verdict line;
- Skeptic verdict and the exact existing submissions this candidate would join.

Stop and ask Samuel to approve or refuse **that exact SHA**. Never call
`whest submit` from the handoff prompt and never interpret today's implementation
request as submission permission.

### Same-day fallback tree

| observation | action today |
|---|---|
| W-EFF cannot preserve exact K3 or implies `>0.35 B` integrated | kill full-tower port; test only the preregistered late-window spelling |
| no late window clears its drift/cost screen | record the negative; do not spend truth or ship another calibrated-K2 variant |
| candidate loses its fresh paired DECIDE gate | freeze it as a research artifact; do not use public feedback to tune it |
| score passes but deployment/memory/residual gate fails | repair the operational defect under a new contract; do not submit |
| every gate passes | hand Samuel the exact-SHA approval packet; wait for explicit submission approval |

## 5. Full execution order

Each IMPORTANT or STRATEGIC slice ends in synthesis and an independent Skeptic
before the next implementation consumes its result.

### Track 0 — continuity repairs

**0A, ROUTINE:** reconcile current docs, board/submission state and P01 accounting.

**0B, IMPORTANT:** separate local official-package parity from grading-server
parity and add immutable graded-total regression fixtures. Preserve the exact
19-operation local anchor; do not invent an operation-level attribution for the
server offset.

**0C, IMPORTANT:** add candidate-selectable preflight certification or require
formal promotion into `estimator.py` before certification. If parameterized,
thread the selected estimator through validation, identity, lifecycle, anchor,
shape, deployment and smoke steps and attack it with focused mutations.

### Track 1 — W-EFF price and parity, no truth

1. Write the IMPORTANT prediction and smallest representative layer contract.
2. Build an exact small-shape dense reference.
3. Implement fresh-block symbolic/sparse operations in a research arm.
4. Prove layerwise and final bit/tolerance parity at small shapes.
5. Bill width 1024 with flopscope 0.12, including assembly and casts.
6. Count `__bool__`, `__float__`, `__int__` materializations and require zero.
7. Account for residual and memory without claiming Windows enforcement.

Exit only if the measured price supports an integrated `<=0.35 B` path and no
operational falsifier fires.

### Track 2 — late-window K3 imitation, no labels

1. Freeze `m={2,4,6,8}`, checkpoint initializers and the full-K3 control.
2. Attribute full-K3 output by factor-block age before truncating anything.
3. Run exact age-window and start-at-checkpoint variants on the same unregistered
   probe networks.
4. Compare prediction and slice drift to full K3; compare to arbitrary rank caps
   at matched retained rank.
5. Bill every arm under flopscope before requesting truth.

This no-label stage is an imitation diagnostic, not an accuracy estimate. Register
the RMS prediction drift `D` from full K3 and compare it with the full-K3 RMSE anchor
and its interval. The triangle inequality gives only a bracket on a same-population
candidate RMSE, `max(0, RMSE_full-D) <= RMSE_candidate <= RMSE_full+D`; distribution
shift and anchor uncertainty must be carried explicitly. Continue to a later DESIGN
contract only if the registered drift and cost thresholds pass. The thresholds must
be written with Samuel before data exists.

### Track 3 — kernel carrier objective

**IMPORTANT, truth-free:** implement the deep-ReLU kernel energy and validate
algebraic nulls, including invariance to a global rotation.

**STRATEGIC, DESIGN:** freeze the carrier panel and test whether the proxy predicts
finite-width MSE ranking. Basis indices and weights may not be fitted on the public
mini. Only after proxy validation may a new single-use decision compare adjusted
scores.

### Track 4 — faithful augmented K2

Port ARC's recurrence at small shape, compare every retained state component and
layer, then price the actual f32 spelling. Decide standalone and as checkpoint
initialization/reference; those are separate estimands.

### Track 5 — K3 calibration and response

Run projection/renormalisation, K4 readout and per-neuron response as separate
DESIGN arms. Advance only effects that reproduce on a locked replica with the
paired arithmetic score interval excluding zero and with total measured cost.

### Track 6 — integrated decision

On a new SINGLE_USE Phase-2 block compare:

- v2.7 control;
- exact W-EFF K3 if deployable;
- late-window K3;
- supported carrier + augmented K2;
- supported hybrid.

Freeze architecture, coefficients, checkpoints and fallbacks before the block.
Run 300–600 fresh networks for robustness after the decision. Promote two diverse
nominees: a conservative state/carrier system and the most accurate supported K3
hybrid.

### Track 7 — deployment, confirmation and nomination

1. Three 100-MLP official-subprocess soaks; report first-row warm-up, p99, max and
   zero failures.
2. Enforce zero scalar materializations during `predict`.
3. Run an actual Linux memory limit with headroom.
4. Use the public mini as CONFIRM only; fit nothing.
5. Create/touch a Phase-2 vault only with Samuel's explicit authorization and a
   preregistered milestone.
6. Certify exact candidate hashes with full preflight and archive identity.
7. Submit and designate up to two entries only after explicit hash-naming approval.

## 6. Validation contract

Every strategic experiment contains, before data exists:

- exact estimand and isolated paired score difference;
- confirmed/inferred/speculative evidence;
- mechanism and numbered magnitude predictions;
- Samuel's prediction;
- gate, falsifier, baseline and tier;
- seed-ledger claim, never a literal;
- frozen flopscope counted-cost model (`C=F` in Phase 2);
- residual headroom against the worst observed official-subprocess run;
- confounders and what must not change;
- commit proving the contract preceded data.

Every result reports the raw artifact, ground-truth noise, paired arithmetic
studentised/basic-bootstrap interval, total FLOPs including assembly, residual,
wall, memory accounting, failures, tails and the independent Skeptic verdict.

## 7. Kill and pivot rules

- More Price/Hermite orders inside the same Gaussian covariance map stay low value.
- Marginal Edgeworth readout stays closed as a replacement for recurrent state.
- Generic CP/Tucker/TT compression stays closed unless its objective preserves the
  future diagonal-slice queries.
- Rank-`n` capping in the current K3 basis is a strong negative, not a theorem.
- K3 compression is weak-open only when a cheap mechanism exists; do not ladder
  another expensive oracle.
- Kerdock/MUB remains a bounded carrier family, not a presumed winner.
- Generic degree-five cubature and activation-region enumeration are infeasible at
  this dimension absent new external evidence.
- Broad learned correctors wait until the mechanistic state path is fixed.

## 8. Immediate next action

Use the same-day critical path above. Preregister **one representative W-EFF layer
price/parity slice** and take Samuel's prediction. Prepare—but do not yet execute—
the no-label factor-block-age/late-window contract. Do not spend another truth block
until one path has an authoritative flopscope operating point and its fresh
independent review is complete.
