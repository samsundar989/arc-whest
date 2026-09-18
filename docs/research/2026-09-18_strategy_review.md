# Strategy and implementation plan — September 18, 2026 UTC

Prepared in the September17 evening US Central session. Research and planning only: no estimator changes, new accuracy measurements, seed claims, truth-array reads, vault touches, submissions or designations. This document supplements the active September17b slice; it does not silently replace its committed contract, gates or branch rules. BatchB was running during review, so no partial results are used to select a new method.

Companions: [live Chrome evidence](2026-09-18_competition_snapshot.md), [architecture audit](2026-09-18_architecture_audit.md), [validation audit](2026-09-18_validation_audit.md), [primary-source research](2026-09-18_frontier_research.md), and independent Skeptic verdict to be appended alongside these reports.

## Decision

**Keep the architecture family; repair how its experiments are implemented and judged; focus the next scientific work on preserving useful higher-order history cheaply.** The repository has found a credible route, not yet a demonstrated winning operating point. The active plan's combination of coherent nonlinear recurrence and query-directed storage is sensible. Its current implementation has concrete objective/gate mismatches, and the completed accuracy evidence is much smaller than the remaining gap.

A broad rewrite or more submissions cannot substitute for that missing gain. A smaller, source-bound experiment-to-release path can substantially reduce wasted iteration. We should also stop treating an arm that differs from its intended formula as evidence against that intended idea.

## What the live comparison says

The board showed rank62/170. Our latest graded v31 #331317 is8.752e-9 adjusted,2.907e-8 raw,30.10% compute. The clean leading #331393 is2.239e-9 adjusted,1.655e-8 raw,13.53% compute. A newer #331435 displays2.220e-9 but has a TIME_EXHAUSTED row and inconsistent summary/partition labels; record it, but do not use it as a clean deployment target. Public truth-noise floors are unavailable. These displayed mean ratios describe the gap, not a paired promotion test.

Against the clean leader, our adjusted gap is about3.91x: approximately1.76x raw error and2.22x compute multiplier. Both matter. Keeping v31's accuracy and reaching the0.1 floor would still score about2.907e-9. Keeping its30.1% compute and merely matching the leader's raw error would score about4.98e-9. This is why neither accuracy-only nor cost-only polishing is a satisfactory winning strategy.

The existing2x release goal means approximately4.376e-9 on the current public scale. That is around rank13 today, whereas the top-ten boundary is around3.6e-9. The win objective must remain distinct from this intermediate release milestone.

For a constant multiplier q, the required raw MSE is target_score/q:

| Compute multiplier q | Raw needed for2x release (4.376e-9) | Raw needed to match clean leader (2.239e-9) |
|---:|---:|---:|
| 0.100 | 4.376e-8 | 2.239e-8 |
| 0.1353 | 3.234e-8 | 1.655e-8 |
| 0.150 | 2.917e-8 | 1.493e-8 |
| 0.216 | 2.026e-8 | 1.037e-8 |
| 0.301 | 1.454e-8 | 7.438e-9 |

These are planning identities on the public scale, not forecasts for P12/P13 or candidate guarantees. Real evaluation averages per-network products. A plausible engineering target with some room beyond today's leader is roughly1.7–2.0e-8 raw at0.10–0.12 compute, but its attainability remains unproven. We must stop assuming the present0.22-cost setting only needs a little accuracy tuning to win.

## What the current repository actually establishes

- Root `estimator.py` remains archive-compatible legacy v2.6.1; the latest graded candidate is `estimator_k3_v31.py`. A green root-estimator check alone is not certification of the new candidate.
- Stage1 of September17b is complete, despite stale status headers. Its fresh24-network P12 results selected the registered interaction branch(b). The complete reference nonlinear program improves our raw error1.077x, while removing it damages the reference much more. Terminal-readout-only enrichment is essentially null. That points to recurrent coupling, not merely a missing final-output formula.
- The own PAIR+Q program's official-precision gain is1.0476x in its stage1 comparison; BUDGET4+Q is1.0150x. Neither supports a promise of the major release by itself. The reference's empirical mask remains diagnostic-only under the current provenance decision.
- P12's measured truth noise is4.4961e-9 per replica, about2.24805e-9 for the averaged truth used to score. Cross-cost comparisons must show observed and official-precision results.
- BatchB includes the lambda interaction cells, P3 birth feed, query-basis variants and composed own-program/storage arms. Finish that batch and synthesize it under the existing contract. P13 stays untouched until a candidate is frozen.
- P3 is a partial beta1+beta2 feed, not every third-cumulant/covariance source term. Its gamma term remains omitted. P3 plus shared storage currently fails deliberate patch-anchor checks and is not an available combined candidate. Integrate only if P3's isolated evidence warrants it.

## Target architecture, in implementable terms

The scientific state should expose mean, full covariance, persistent factored third-cumulant sources, and an explicit approximate fourth-order core. The third-order history matters because cross-neuron dependence created many layers earlier continues to affect later activation means. Keeping only current marginal moments or current D3/D21 slices loses information needed after the next weight transform.

Use an explicit layer transition:

1. Transport the represented state through the actual weight matrix.
2. Evaluate the contractions consumed by the ReLU update: D3, D21 and the required fourth-order slices.
3. Apply one coherent nonlinear moment program, update covariance/core, and create new sources with explicit overlap compensation.
4. Compress older history at a precisely documented boundary. The basis objective must refer to that same boundary, and validation must include downstream nonlinear feedback.
5. Record means; prune unused final-layer work exactly.

Recent sources stay exact relative to the represented model until a tested compression rule admits them. Older-source storage may be shared and compressed, but source updates, joins, materialization, queries and fallback all count toward cost. Preserve exact P4/Strassen savings. Approximate fourth-order regeneration is a model, not an exact representation of the true activation distribution.

For development, expose stable interfaces for `State`, closure evaluation, source birth/compensation, dense and compressed consumers, and export. Introduce these seams around existing working components incrementally. Keep frozen archives unchanged. The deliverable remains one deterministic, shape-generic, grader-legal file, with no external diagnostic implementation dependency. We do not need a new numerical framework to achieve this separation.

## Immediate fixes before accepting stage3 conclusions

**1. Correct the decision adapter, preserving the registered question.** `local_sim/s12_decide.py` says official-precision but calls the binding decision function on observed MSE*cost. It also stores but never compares paired seed lists. The validation agent produced synthetic end-to-end false-pass and mismatched-identity examples. Those demonstrate real software defects; they do not establish that the current real verdict reverses.

Construct canonical paired rows, refuse duplicate/missing/mismatched identities, convert each row to the registered precision through one shared score helper, retain failure fallback semantics, and call `competition_score.decide` on those arrays. Report each estimand explicitly. Recompute from saved rows, without rerunning estimators or spending a new block. This is an IMPORTANT instrument repair: prediction, targeted tests, preflight and independent Skeptic remain required. Keep the existing P13 dual release requirements intact.

**2. Make query validation test the actual object.** Current S3 checks omit explicit E and next-state conditions named in the contract. The cheaper-setting implementation calls output-vector displacement “retained query energy,” which is not the same mathematical quantity. Either prove a recorded equivalence/redundancy or implement the stated measurement; do not weaken the gate after results. Preserve old artifacts and scope their claims accurately.

**3. Verify and repair compression-objective mismatches before another truth run.** Source inspection finds pre-Wick left factors carried unchanged into a post-Wick join. The represented D21 query should acquire left row weight w1² and right row weight w1. The present qE objective includes the latter through the legs but omits the former. Separately qEd constructs A*M*w2 although the documented L-source D3 summand is A²*M*w2; coefficient indexing must follow the joining lineage. Tiny asymmetric, nonuniform-gate and multiple-lineage examples should compare objective action against the direct consumer. Full-rank reconstruction cannot detect a wrong rank-selection objective.

These are implementation diagnoses and bounded experiment opportunities. They are **not** evidence that correcting them will improve final MSE. Keep current label-free qE displacement improvements and qEd regressions valid for their executed programs, with narrower interpretation; batch B accuracy was still pending. A corrected program needs an explicit new amendment/contract and appropriate DESIGN evidence, not retrospective relabeling.

## Scientific priorities after the active slice

Run at most one new accuracy discriminator and one new storage discriminator before another strategy synthesis. No broad closure/rank grid.

### First: complete the intended consumed-query test

After the label-free boundary/D3 fixes, compare the corrected objective against the existing qE at fixed storage, full metered cost and complete recurrent continuation. The mechanism is specific: neuron-dependent activation gates change which query directions matter; a pre-gate norm can allocate rank to directions that are suppressed before use. Only proceed to the already contemplated cheaper setting when its actual defined gate permits it.

Stop if preserving the intended query improves neither full continuation nor score at equal cost. Do not add another collection of proxy weights to rescue the result. The purpose is to test the declared method once correctly.

### Second: resolve the remaining accuracy interaction

Use completed batchB to choose the question. If P3 pays, integrate it with shared storage at the three known collision points and require dense/off/full-rank parity, newborn beta delta and compensation identities before measuring the composition. Do not claim its gains multiply until measured.

If lambda does not explain the reference/native gap, find the earliest complete-state operator divergence with compatible source history. A two-way replacement of one independently defined operation can distinguish core construction, birth feedback and residual representation. Matching scalar norms on different trajectories does not identify causation. If no distinct operator remains, stop reference-code archaeology.

A more novel physical hypothesis is a **small transported residual around the regenerated fourth core**, rather than replacing the entire chain with a full K4 tensor. It is worth a contract only if existing permitted diagnostics identify an omitted residual direction that persists and affects the consumed fourth-to-third feed. Preserve the exact diagonal and compensated repeated-index rules; evolve only that identified residual in a thin channel. Exact full K4 and arbitrary extra covariance modes already have adverse evidence. The new channel must discriminate temporal memory from a larger one-step fit, and its complete transport/query bill must fit the score economics. No gain is assumed.

### Third, conditional research: residual-enriched or interpolated aggregate queries

The [literature report](2026-09-18_frontier_research.md) proposes two higher-risk extensions after the above:

- **Residual-driven basis enrichment:** use the omitted nonlinear update's actual query action to reserve a small number of useful directions, with a fixed total resource envelope. Test full covariance/birth feedback and eventual output, not only immediate query norm. The mathematical approach is established; an inexpensive oracle for this cumulant recurrence would be the contribution.
- **Deterministic cross interpolation of signed aggregate D21:** retain source history but evaluate selected exact query rows/columns and reconstruct a stable approximation. Prove the row/column oracle is cheap, including selection, D3 and fallback, before any accuracy job. This is distinct from raw polynomial TensorSketch and memoryless slice propagation, both of which have adverse evidence.

Neither is a demonstrated challenge improvement. New public E008/E009 results specifically argue against repeating raw TensorSketch or a one-step future-weight sketch. Finite-width GP-mixture/Edgeworth papers that average over weight randomness do not solve integration over inputs for a fixed known network. No public paper or released leader source found in this review supplies a ready-made winner.

## Reduce iteration cost without losing the experiment

The core infrastructure should stay. Canonical minting, immutable grader-style weights, failure-aware scoring, independent truth replicas, exact-package checks, Linux memory enforcement and paired arithmetic gates answer necessary questions. The latest local packages match the server; the known v31 forecast was8.7133e-9 versus8.7520e-9 graded (~0.45%). This is one successful validation, not a calibrated hidden-score confidence interval. Server tail timing and a new private network draw remain uncertain.

Use a small front door over existing tools:

| Stage | Work | Exit condition |
|---|---|---|
| Build | Import/contract, changed algebra/invariants, full-rank/reference identities | Intended mechanism is implemented; no labels needed |
| Price | One complete live-shape cost/resource probe when changed work requires it | No omitted joins/readout/fallback; feasible resource plan |
| DESIGN | One registered panel, shared control, streamed rows/predictions, correct precision | Select at most one finalist; ties stay ties |
| DECIDE | Freeze exact bytes and configuration; one untouched block | Existing registered arithmetic and noise-aware release conditions |
| Release | Candidate-specific certification, enforced Linux8GiB, one public CONFIRM, projection, Skeptic, required preflight | One reviewable evidence bundle for per-submission approval |

Implement **verified frozen-control reuse and atomic prediction manifests**, using existing evidence-journal patterns. A cache key must cover candidate/dependency bytes, ordered MLP identities, truth/mint identity, full configuration, lifecycle/scoring implementation and numerical environment. Cache numerical evidence, not transferable server timing. Do not add automatic reuse/recovery to single-use DECIDE without its contract defining it.

Keep a tiny candidate manifest listing legacy root, latest graded, designated fallback, experimental finalist, source hashes, contract and completed evidence. Generate the current operational summary from this manifest/receipts; keep historical LOG entries. This removes contradictory “not started” and “running” headers without rewriting history.

Count useful decisions per day, not submissions. Track time spent in truth creation, repeated prediction, queue wait, packaging failures and review rework. Today's9-arm stage1 batch took about1h53 uncontended according to the LOG; expensive mini/certification should not be multiplied by each arm. Avoid overlapping CPU-heavy jobs that slow each other; heavy work stays on workerq with the mandatory optimization review. Prefer measured resource envelopes and sequential arms with streamed output.

## Ordered implementation checklist

1. **Now:** preserve batchB data; repair stage3 scorer/pairing and establish query-gate coverage before interpreting its screen. Record changes separately from candidate evidence.
2. **After batchB completion:** synthesize the declared program/lambda/P3/storage results once; update candidate status. Do not spend P13 on a candidate lacking the registered release case.
3. **Next small coding slice:** prove the query boundary and D3-lineage identities; implement corrected objective in a new generated candidate. Independently review tiny tests, then register a bounded accuracy/cost test with Samuel's predictions before measuring.
4. **Conditional integration:** if P3 or the permitted own fit pays, compose through explicit interfaces and verify the complete recurrent program. Preserve a reference dense path to separate physics from storage error.
5. **One challenger:** choose one complete accuracy/cost configuration on DESIGN, freeze its bytes, run fresh DECIDE and the one release path. Next submission remains aimed at the requested major jump. A sub-gate candidate is retained as research, not quietly released by changing the target.
6. **If the major gate misses:** state the residual gap on the same panel/precision scale. Choose one identified core residual or residual-enrichment test, with a stopping rule; consider cross interpolation only after its whole-cycle price is defensible.
7. **Before October17:** ensure robust prize-candidate evidence and Samuel's explicit up-to-two selection. The contribution write-up remains a separate claim-reviewed deliverable due October24; method limitations and falsified conjectures are useful evidence, not reasons to claim state of the art prematurely.

The next major jump is a research objective, not something this review can honestly promise. We now have specific instrument and compression defects to remove, a clearer two-axis performance target, and a bounded path to test whether the intended architecture can reach it.
