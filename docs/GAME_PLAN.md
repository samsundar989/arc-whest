# GAME_PLAN.md — Phase-2 frontier research program

**Authoritative from 2026-08-31.** This is the sole active research roadmap. Its
two nominee architectures are falsifiable candidates, not established winners.
`CURRENT_STATE.md` says what has happened; `VALIDATION.md` governs every
measurement; `PHASE2_FRONTIER_AUDIT_2026-08-31.md` is the dated competition,
repository and literature evidence behind this revision. Older Kerdock-first and
pre-launch plans are historical.

> ## STATUS 2026-08-31 — carrier first; analytic response in parallel
>
> **Nothing new was shipped or submitted. `estimator.py` remains v2.6.1 and the
> best graded entry remains #328664 (v2.7) at `1.373807110667258e-07`, now rank
> 53/83.** The public leader is `5.60e-9` adjusted / `2.25e-8` raw.
>
> Four leading reports occupy a dense-forward cost envelope near 16,384 rows.
> Eight antipodal MUB bases have that leading cost, but so do Sobol, lattice,
> orthogonal, and other 16k-point systems. The clue promotes the **nested carrier
> frontier**, not a private-method claim or a fixed basis count.
>
> Installed WhestBench 0.16 proves its 120-second participant wall default is
> **per MLP**, not shared across the 100-MLP suite. Public ledgers show up to
> `104.766 s/MLP` passing, consistent with that default but not exposing the exact
> graded ceiling. The separate installed 0.4-second residual cliff remains.
>
> Therefore Track 3 begins with the missing submission-legal Kerdock/MUB constructor
> and deployment closure, followed by the frozen carrier objective and nested
> basis/sample frontier. Track 4 runs faithful AUGMENT-K2 parity and a truth-free
> query-directed/adjoint K3 prototype in parallel. The live-shape `k3pre`
> confirmation is useful, but no longer blocks candidate-producing work.

> ## HISTORICAL STATUS 2026-08-28 — Tracks 1 and 2 CLOSED on measured negatives
>
> **Nothing was shipped. `estimator.py` is unchanged at v2.6.1; the best graded
> entry remains #328664 (v2.7) at `1.373807110667258e-07`.** Two IMPORTANT
> slices ran under Rule-1 contracts and both failed their own preregistered
> gates. Four independent Skeptic passes reviewed them; all four returned
> PROVISIONAL or REJECT and all four were accepted.
>
> - **Track 1 (W-EFF) — CLOSED, killed on price.** Exact sparse/equality-aware
>   factorized K3 over the full tower costs **`1.0732 B`** under flopscope 0.12,
>   against its own `<= 0.35 B` screen: **`3.0664x` over**. The saving over the
>   dense spelling is `27.8%` and is a bounded constant factor, not a change of
>   order. **The `5.49x` cross-meter "affordability target" that shaped §2 and
>   §3B is DEAD** — it compared two different meters and never existed as a
>   saving.
> - **Track 2 (late-window K3) — CLOSED, killed on accuracy, provably.** Every
>   window arm's *provable* score floor is `2.09x`–`4.65x` **worse** than graded
>   v2.7. The gate is not rigged: with `MSE_FULL -> 0` *and* the multiplier
>   pinned at the floor — both physically impossible — it still fails by
>   `2.20x`–`2.49x`.
> - **The reason the window failed was RETRACTED and replaced.** It is not that
>   `kappa_3` is non-local. **`89.2%` of the drift is the bad `k_max=2` prefix**,
>   measured by an arm that isolates the two effects. §3C's `m^2/L^2` cost
>   hypothesis is likewise superseded by a measured per-layer table.
> - **Consequence for this plan: the ordering in §3 is unchanged but the weights
>   are.** Track 4 (faithful augmented-K2 prefix) moves from "pending" to **the
>   binding question**, because the prefix is where the measured error is.
>
> **The one number that frames everything below:** full factorized K3 has the
> accuracy to lead this board — delivered at the `0.1` multiplier floor it would
> score `3.92e-09` against the leader's `7.4e-09` — and it costs **`10.7x` too
> much**. W-EFF, the best exact-spelling idea available, bought `1.38x` of that.
> **The whole game is price.**
>
> `docs/CURRENT_STATE.md` carries the full evidence; the contracts, results,
> corrections and four Skeptic verdicts are at the bottom of
> `experiments/LOG.md`.

## 1. Objective and current score geometry

Win Phase 2 while obeying the intended flopscope computation rules and preserve a
contribution-prize-quality mechanistic account.

For each random no-bias `1024 x 16` ReLU MLP, predict all post-ReLU means under
`x ~ N(0, I)`. The score is

```text
mean_MLP[final_layer_MSE * max(0.1, FLOPs / 2**41)]
```

WhestBench 0.16 / flopscope 0.12 is live. Lambda is zero. Installed/run-config
defaults are a hard `0.4 s/MLP` residual cliff, 120-second `predict()` wall per
MLP, 5-second setup and 8-GiB memory; graded config does not expose these exact
fields, while public wall telemetry proves `>=104.766 s/MLP` passes.

Live board snapshot at **2026-08-31 14:57 UTC** (display precision):

| system | adjusted | raw final MSE | multiplier | status |
|---|---:|---:|---:|---|
| J2W, rank 1 | `5.60e-9` | `2.25e-8` | `~0.248` | public feedback |
| Puffi, rank 2 | `5.90e-9` | `2.26e-8` | `~0.260` | public feedback |
| marius_binner, rank 4 / best raw among top four | `7.70e-9` | `1.83e-8` | `~0.420` | public feedback |
| neopok919 v2.7, rank 53/83 | `1.374e-7` | `1.374e-6` | `0.1` | graded, zero failures |
| exact factorized K3 | `~4.21e-8` projected | `3.9249e-8` | `1.0732` from `1.0732 × B` | organiser mini; GT-noise floor `7.478e-11`; priced oracle, not deployable |

Here and below, cost notation such as `1.0732 B` means `1.0732 ×` the competition
budget `B = 2**41`, not 1.0732 billion FLOPs. The public gap is `24.6x` adjusted
and about `61x` raw. v2.7 is already
floor-pinned; cheap calibration alone cannot win. The top group shares raw MSE
around `1.8e-8`–`3.4e-8` and trades it against cost.

### Outcome targets

- **Frontier milestone:** honest adjusted score `<=1.2e-8`, zero failures.
- **Winning target:** `<=3e-9` on fresh offline evidence, leaving private-rerun
  margin below the moving public board.
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
5. **MEASURED 2026-08-28, and this supersedes the hypothesis it replaces.**
   `95.35%` of upstream K3 cost is in two operations, and the exact structure
   available is **mode equality**, not sparsity — identity dies at the first
   `contract_W`, but `from_dstensor`'s `(A, I, I)` stays `(., M, M)` and the
   two-leg block's mode 2 stays exactly `3M` forever (verified in upstream's own
   tower: `||B_D - C_D|| = 0`, `||B_L - 3 B_D|| <= 3.24e-16` at every layer).
   That collapses six `(n,n)` matrices per lineage to four and six slice matmuls
   to five — a **bounded constant factor**. Measured: **`1.0723 B`** for the
   dominant lines, **`1.0732 B`** whole-path, `27.8%` under the dense spelling,
   **`3.0664x` over the `<= 0.35 B` screen**. Parity `5.6e-16`; zero scalar
   materializations; zero uninstrumented leaks; bin reconciliation against
   upstream closes **exactly**. **The `5.49x` figure is dead**: it compared
   flopscope against upstream's torch counter and was never a saving.
6. **MEASURED.** A final-`m`-layer exact-K3 window from a zero checkpoint costs
   exactly the first `m` layers of the tower. `m <= 3` stays pinned at the `0.1`
   multiplier floor — but at a `7.1%`–`12.9%` **margin**, not as a category.
   `m = 4` costs `1.20x`, `m = 6` `2.01x`, `m = 8` `3.18x`. This replaces the
   `m^2/L^2` argument with a measurement.
7. **MEASURED.** Late-window K3 does not work, and not for the reason first
   recorded. Exact K3 over the last half of the tower recovers `12%` of the
   drift gap to full K3; over the last two layers, `0.37%`. An arm that isolates
   the two interventions shows **`89.2%` of `late:2`'s squared drift is the
   `k_max=2` prefix**, not the discarded `kappa_3`. Every window arm's provable
   score floor is `2.09x`–`4.65x` worse than graded v2.7.
8. **MEASURED, instrument.** flopscope's residual is
   `wall - backend - overhead`, so the `0.4 s` cliff is about **op count and
   Python glue, not total FLOPs**; the constraint that scales with FLOPs is the
   installed **`120 s` per-MLP participant wall cap** (public grader evidence only
   proves `>=104.766 s`). And flopscope grants an **aliased-operand
   discount**: `fnp.einsum('ij,kj->ik', A, A)` is `0.5005x` of `A @ A.T`, and
   `W core W^T` via `cholesky` + aliased Gram is `0.8336x` (relerr `7.65e-07`,
   SPD core required). Worth nothing at a floor-pinned operating point; worth
   `~0.005 B` to anything that is not.

### Structured sampling

1. The width-1024 Kerdock/MUB construction is algebraically valid. Dense 16 bases
   cost exactly `2**40` before control overhead.
2. The fixed 16-basis carrier is not a spherical 4-design; a carrier-aligned fourth
   moment is `21.6881x` the spherical target.
3. Public Phase-1 evidence supports Kerdock/MUB, antipodes, exact radial
   integration and checkpoint control. Phase-2 leader code remains private.
4. Current Phase-2 leader telemetry clusters near the cost of 16,384 dense forward
   rows. Eight antipodal MUB bases are one matching construction, but the cost alone
   cannot distinguish MUB from Sobol, lattice, orthogonal, or another sampler.
5. Ordinary IID and random-orthogonal arms lose locally, but they do not bound a
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
5. P01 is DESIGN, **168/600 consumed**, 432 free. It is not an independent
   replication block. No Phase-2 SINGLE_USE decision block or vault exists.

## 3. Candidate architecture hypothesis

The program develops **two genuinely different nominees**. Sampling and analytic
propagation may later be complementary, but each module must survive its own
falsifier before integration.

```text
Nominee A: exact-radial structured carrier
  stdlib/fnp point construction in setup
    -> nested structured angular propagation
    -> per-layer sample means
    -> optional frozen low-correlation analytic blend

Nominee B: compact analytic response
  corrected f32 K2
    -> faithful unfactorized AUGMENT-K2
    -> query-directed/adjoint K3 response
    -> optional neuron-specific mixture / rolled-forward correction
```

### A. Structured angular carrier — primary candidate-producing track

Positive homogeneity integrates the Gaussian radius exactly. The current
width-1024 Kerdock/MUB object has valid defining Gram properties, but the only
materialization probe builds it with NumPy and wraps the finished array in `fnp`.
It is **not yet grader-legal**.

First close byte-identical stdlib/`fnp` construction, fresh-worker setup, scalar
materialization, memory-accounting, and dense propagation parity. Then freeze a
nested carrier panel including:

- Kerdock/MUB basis counts spanning at least `4, 6, 8, 10, 12`;
- randomized signed-permutation/Hadamard orientations;
- matched-row high-dimensional scrambled Sobol;
- matched-row random orthogonal and IID controls;
- antipodal and non-antipodal variants only where the contract isolates them.

Select by paired **adjusted score**, including all auxiliary work. Eight bases is a
live-telemetry clue, not the selected answer. A deep-ReLU kernel discrepancy may
screen the panel only after it demonstrably ranks finite-width MSE; a global
rotation is a Gram-preserving null for that proxy.

### B. Analytic state and query response

Use f32 K2 as the stable reference. Port faithful recurrent augmented K2 only after
small-shape state parity with ARC's reference. Do not represent it with the old
marginal Edgeworth oracle.

In parallel, prototype a **query-directed/adjoint K3 response**. Linearize around
the K2/AUGMENT trajectory, back-propagate adjoints for all final outputs or a fixed
hard-neuron subset, and contract each layer's locally born K3 source directly into
those queries. Do not transport the complete K3 tensor.

The first test is truth-free at small width: compare layerwise response with exact
full K3, measure explained squared drift and cosine, scale the injected source
toward zero to test first-order convergence, and price the actual flopscope
spelling. Kill it if the response is diffuse, the adjoint rank expands to the full
state, or the source-scaling falsifier fails.

### C. Exact sparse-aware K3 (W-EFF) — closed accuracy oracle

Keep fresh identity-supported blocks symbolic, replace `W @ I` with reuse/indexed
operations, fuse factor transforms with diagonal-slice reads, and separate sparse
fresh blocks from dense aged blocks. Exact parity comes before cost claims.

**ANSWERED AND CLOSED, 2026-08-28.** The gate ran: `1.0732 B` whole-path,
`3.0664x` over the `<= 0.35 B` screen. **Do not port the full tower.** The exact
operations survive as reusable code (`local_sim/p2_weff.py`) and the per-layer
ledger survives as the cost model for anything windowed; the *lane* does not.

### D. Late-window or age-window K3 — closed candidate lane

Run K2 through the prefix and start exact K3 only for the final `m` layers, or
retain complete layer-generated blocks by age. Candidate `m` is frozen to
`{2,4,6,8}` for the first truth-free screen.

Checkpoint initializations:

- zero K3 state;
- augmented-K2 trace/harmonic state;
- selected K3 diagonal queries estimated by the structured checkpoint.

**ANSWERED AND CLOSED, 2026-08-28.** The panel ran on six fresh unregistered
probes with common random numbers. Every window arm provably cannot clear the
`2x` bar, and the matched-retained-rank control beats the window by `2.2x`–`4.6x`.
The `m^2/L^2` argument is superseded by a measured per-layer table. **The
checkpoint initializer, not the window length, is what mattered**: only the zero
initializer was implementable, and it is responsible for `89.2%` of the error.
Read §D of the Track-2 entry in `experiments/LOG.md` before reopening anything
here.

### E. Mixture, response, and calibration

Only after the state engine works, test separately:

1. a two-component neuron-specific skew mixture driven by supported AUGMENT trace
   state—never a positive global-scale mixture, which homogeneity collapses;
2. diagonal/trace K4 readout correction;
3. a tiny shared permutation-equivariant per-neuron residual model;
4. a fixed low-correlation blend of a supported carrier and analytic state.

Every fitted map uses disjoint DESIGN fit/evaluation networks and is evaluated on
its own rolled-forward trajectory. Teacher-forced one-step improvements do not
promote anything.

### F. Integration and safety

The deployable file stays self-contained and grader-legal. It must return all 16
rows, preserve exact layer 1, mutate no grader arrays, perform all numerical work
through flopscope, create no scalar synchronization inside `predict`, avoid budget
summaries, and fall back safely on instability.

## 4. Same-day critical path — 2026-08-28 (HISTORICAL — it ran, and it ended in a kill)

> **This section is the record of what was attempted on 2026-08-28, not an
> instruction.** Outcome: honest outcome **2** of the two it named — "the
> highest-value path is killed by its preregistered falsifier, the result is
> recorded, and no weaker candidate is submitted merely to meet the date."
> Both Today B and Today C failed their gates, so Today D and Today E never
> began. No truth block was spent, no candidate was built, nothing was
> submitted. **Do not execute this section.**


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

Before that decision can exist, complete a separate **IMPORTANT seed-instrument
allocation slice**: freeze the power plan, obtain Samuel's prediction, register at
least one disjoint Phase-2 DESIGN-replication block and one distinct Phase-2
SINGLE_USE DECIDE block, extend the ledger tests, run preflight, and obtain a fresh
Skeptic review. P01's unused range may support preregistered DESIGN development,
but it cannot stand in for the disjoint replica or the final decision block.

Then claim the newly registered SINGLE_USE Phase-2 DECIDE block through the ledger
and compare the frozen candidate with the exact v2.7 control on common random
networks. Use
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

**0D, IMPORTANT, before any new truth:** after the truth-free carrier and analytic
gates fix the required power, register a disjoint Phase-2 DESIGN-replication block
and a separate SINGLE_USE DECIDE block in the seed registry/ledger. This is an
instrument change: prediction first, focused tests, default preflight, knowledge
delta, and an independent Skeptic are mandatory. Do not allocate arbitrary sizes
before the power analysis, and do not treat P01 as an independent replication.

### Track 1 — W-EFF price and parity, no truth — **CLOSED 2026-08-28, killed on price**

1. Write the IMPORTANT prediction and smallest representative layer contract.
2. Build an exact small-shape dense reference.
3. Implement fresh-block symbolic/sparse operations in a research arm.
4. Prove layerwise and final bit/tolerance parity at small shapes.
5. Bill width 1024 with flopscope 0.12, including assembly and casts.
6. Count `__bool__`, `__float__`, `__int__` materializations and require zero.
7. Account for residual and memory without claiming Windows enforcement.

**All seven steps ran. The measured price is `1.0732 B` — `3.0664x` over the
screen — so the exit condition was not met and the lane is closed.** Every
sub-result survives and is reusable: the exact sparse/equality-aware operations,
the width-1024 per-layer ledger, the commutation parity, the mutation panel, the
bin reconciliation, and the enforced hard-rule-3 leak check.

### Track 2 — late-window K3 imitation, no labels — **CLOSED 2026-08-28, killed on accuracy**

1. Freeze `m={2,4,6,8}`, checkpoint initializers and the full-K3 control.
2. Attribute full-K3 output by factor-block age before truncating anything.
3. Run exact age-window and start-at-checkpoint variants on the same unregistered
   probe networks.
4. Compare prediction and slice drift to full K3; compare to arbitrary rank caps
   at matched retained rank.
5. Bill every arm under flopscope before requesting truth.

**It ran and it failed.** Note the bracket above is written wrongly here and was
written wrongly in the code: the reverse triangle inequality gives
`|RMSE_full - D| <= RMSE_candidate <= RMSE_full + D`, and the `|.|` matters —
dropping it discards the lower bound whenever `D > RMSE_full`, which is every arm
in the panel. With it restored, the panel decisively **kills** rather than merely
failing to endorse: provable score floors `2.09x`–`4.65x` worse than graded v2.7.
No DESIGN contract follows.

### Track 3 — submission-legal structured carrier — **FIRST CANDIDATE PATH**

The public evidence supports the *operating envelope*, not a private method:
several leaders spend about `5.45e11`–`5.73e11` counted FLOPs, close to propagating
`16,384` dense rows (or eight antipodal 1,024-row bases). Sobol, lattice,
orthogonal and MUB/Kerdock carriers can all occupy that envelope. Do not turn a
FLOP row into a disclosure claim.

**IMPORTANT-A, truth-free deployment closure:** replace the NumPy-built P1A probe
with a stdlib/`fnp` constructor that is byte-identical to the frozen reference.
Verify deterministic setup, zero in-predict scalar materialisations, live-shape
propagation parity, flopscope billing, first-call behaviour and explicit 8-GiB
memory accounting. Until this passes, the current carrier is a research artifact,
not grader-legal evidence.

**IMPORTANT-B, truth-free objective:** implement the deep-ReLU kernel energy and
validate algebraic nulls, including global-rotation invariance. Freeze the nested
basis/sample panel and all controls before labels. The existing fixed 16-basis
family is a component specification, not the selected policy.

**STRATEGIC, DESIGN:** test whether the frozen proxy predicts finite-width paired
MSE ranking on a preregistered DESIGN selection sample, then lock and reproduce the
choice on the newly registered, disjoint DESIGN-replication block. Only after
synthesis, independent review, and a complete freeze may the work advance to the
separate SINGLE_USE decision block. Basis count, indices, weights and blending
coefficients may never be fitted on the public mini.

### Track 4 — faithful augmented K2 and query-directed K3 — **PARALLEL ANALYTIC PATH**

First port ARC's AUGMENT recurrence at small shape, compare every retained state
component and layer, then price the actual f32 spelling. Decide standalone and as
checkpoint initialization/reference; those are separate estimands.

In a separate truth-free prototype, linearise final-mean error with respect to
layer-local third-cumulant sources and propagate final-output adjoints backwards.
Contract each local K3 source directly into those queries instead of materialising
the full recurrent K3 tower. At small shape compare against exact K3 for explained
drift, cosine/sign agreement, source-scaling linearity and counted price. This is a
mechanism-changing route around the measured `Theta(n^3 L^2)` full-tower cost, not
another constant-factor rewrite of W-EFF.

**Why these are open.** Track 2 measured that `89.2%` of a late window's error is
its `k_max=2` prefix, while the full recurrent K3 oracle is accurate but priced out.
AUGMENT asks whether a better compact prefix exists; the adjoint asks whether only
the final-layer-relevant K3 contractions can be retained. **Caution before work:**
`factor=True` with `kind=AUGMENT` is *not* equivalent to the unfactored recurrence
(upstream drops diagrams it cannot factor, `factor_k3.py:290-299`), so the parity
target must be frozen before porting.

### Track 5 — bounded response and mixture corrections

Run per-neuron skew-mixture response, projection/renormalisation and K4 readout as
separate DESIGN arms only after their analytic prefix or carrier is supported.
A global scale mixture is useless in a bias-free degree-one homogeneous network;
any mixture must change neuron-specific shape, not merely radius. Advance only
effects that reproduce on a locked replica with the paired arithmetic score
interval excluding zero and with total measured cost.

### Track 6 — integrated decision

Only after Track 0D has registered it, and after the selected policy reproduced on
the disjoint DESIGN-replication block, claim the new SINGLE_USE Phase-2 block and
compare:

- v2.7 control;
- the best frozen, grader-legal carrier policy;
- supported AUGMENT or query-directed K3, if either cleared its own gates;
- a supported carrier/analytic hybrid, if its interaction was preregistered.

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
- **Exact sparse-aware K3 over the full tower is CLOSED on a measured price.**
  Reopening needs a mechanism that changes the `Theta(n^3 L^2)` scaling, not a
  better constant — mode equality is already exploited and is worth `1.38x`.
- **Late-window / age-window K3 is CLOSED on a measured negative.** Reopening
  needs a *better prefix*, not a different window; the window length was never
  the binding variable.
- **A drift panel can kill decisively.** Use `|RMSE_full - D|` for the lower
  bracket end. Never write `max(0, RMSE_full - D)` — it silently discards the
  only decisive direction the panel has.
- K3 compression is weak-open only when a cheap mechanism exists; do not ladder
  another expensive oracle.
- Kerdock/MUB remains a bounded carrier family, not a presumed winner.
- Generic degree-five cubature and activation-region enumeration are infeasible at
  this dimension absent new external evidence.
- Broad learned correctors wait until the mechanistic state path is fixed.

## 8. Immediate next action — revised 2026-08-31 after live-board and evaluator audit

Tracks 1 and 2 are closed. **Do not re-run W-EFF or late-window K3, and do not
treat §4's historical checkpoints as an active plan.**

1. **Close carrier deployment legality first (IMPORTANT, truth-free).** Write the
   contract and prediction, then build a stdlib/`fnp` Kerdock/MUB constructor that
   is byte-identical to the frozen NumPy reference. Prove setup determinism,
   live-shape propagation parity, zero scalar synchronisations, official billing,
   first-call safety and memory accounting before requesting any labels.
2. **Freeze the carrier objective and panel separately.** Implement the deep-ReLU
   kernel proxy and preregister nested basis/sample counts plus IID, orthogonal,
   scrambled-low-discrepancy and rotation controls.
3. **In parallel, run two bounded analytic proofs, each under its own IMPORTANT
   contract and independent review:** faithful AUGMENT small-shape parity/price,
   and a small-shape query-directed/adjoint K3 prototype against the exact oracle.
   Neither may consume truth until its mechanism and price are credible.
4. **Allocate the missing Phase-2 evidence blocks before labels (IMPORTANT).** Once
   the truth-free gates determine a power plan, register a disjoint DESIGN-replica
   and a distinct SINGLE_USE DECIDE block, test the ledger changes, run preflight,
   and require a fresh Skeptic. Only the DESIGN sequence may establish transfer;
   only the frozen later SINGLE_USE comparison may decide promotion.
5. **Treat the live-shape `k3pre` rerun as optional attribution cleanup, not a
   blocker.** It cannot reopen the killed window without a better prefix and it
   must not delay the two candidate-producing paths above.

**Do not spend a truth block, claim a seed block, touch a vault, or build a
submission candidate until the relevant truth-free legality/parity and cost gates
have passed with fresh independent review.** The Phase-2 deadline is
`2026-10-17 23:59 UTC`; there is time to select two genuinely different nominees
instead of forcing a third public submission.
