# Implementation plan — from F21 to a competitive joint cost/error result

**Proposed strategy, September 14, 2026.** Companion to the [research assessment](RESEARCH_STRATEGY_2026-09-14.md). These stages are work proposals, not committed experiment contracts, new gate definitions, release exceptions, submission approval or designation. Preserve all existing spent-block restrictions and frozen files. Dates below are time boxes, conditional on proof and queue availability, not promises of successful algorithms.

## Objective and controls

Primary control: frozen F21, source SHA256 `d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`, latest graded #330950. Keep L4 for historical algebra comparisons and L2 for operational context. Root v2.6.1 is a historical shipped baseline, not the primary competitive control.

The performance objective is the private-distribution arithmetic score, including failures. A dated aspirational joint target is multiplier around 0.10–0.13 with raw error around 2e-8 or lower. For example 0.12 times 1.9e-8 gives 2.28e-9 against the September-14 leader's visible 2.540e-9. Official public truth noise is unavailable; these are planning coordinates, not a fitted target or a guaranteed rank.

Keep three different thresholds visibly separate:

1. **Exploratory screen:** proposed at least 20% lower complete FLOPs than F21 with at most 5% increased raw error on permitted DESIGN data. Above the cost floor this means at most 84% of the old score under those assumptions. It is an engineering screen, not a promotion decision.
2. **Ambitious cost milestone:** multiplier at most 0.20, about 52.6% below F21's current counted cost, while keeping acceptable error. This demonstrates substantial headroom, but does not win at current accuracy.
3. **Binding decision:** a fresh preregistered paired arithmetic comparison through `competition_score.decide()`. A 1.05 gain gate is a reasonable proposal consistent with the last intervention, but the isolated intervention and power plan must justify the final committed value before any data. None of the screens substitutes for this gate.

## Stage 0 — candidate identity and bounded validation repair

**Time box: September 14–17; approximately one to two working days of focused instrument work, alongside the tiny algebra diagnosis.** Avoid a framework rewrite.

Create one candidate release manifest joining source/package hashes, named baseline, selected recurrence, representation/backend settings, official package versions, numerical tolerance, counted ledgers, Linux cap, local official result and any server-measured record. Existing certification domains already distinguish MATH, LOCAL, SERVER_PREDICTED and SERVER_MEASURED; reuse them. Stored F21 evidence must produce `LOCAL REFUSED` and `SERVER_MEASURED PASS`, with provisional server-byte binding, not an invented combined local pass.

Use the existing tighten-only error-scale option for future permutation/numerical checks. Predeclare the intended raw-error scale and record it in every shard. A mutation exceeding the declared 1% allowance must fail on reusable fixtures, while a known equivalent implementation passes. Include the actual target branch and early/middle/late permutations. Check observed existing drift before deciding a costly rerun is necessary.

Add durable per-row result journaling and an atomic terminal manifest. Crash tests on reusable synthetic inputs must preserve completed rows and failure categories, reject duplicate/incomplete results and refuse promotion. A journal must not grant another protected read. Any resume semantics affecting seed access require their own preregistered protocol before use.

Pin the production truth function's math mode and record Torch/CUDA/device, backend flags, chunk size and producer hash. Compare shared inputs through that function and an independently specified official-precision reference at 1024x16. Bound systematic arithmetic error relative to the planned isolated score gain. No new protected labels are needed to test the instrument. Heavy parity work still goes through workerq.

These changes touch the instrument/enforcement and are IMPORTANT: written prediction before implementation, affected tests and preflight green, LOG delta, fresh independent Skeptic. Any new gate or local-timing exception is STRATEGIC and must be handled separately. Do not silently relax local clean-mini requirements because F21 graded successfully once.

## Stage 1 — settle full-rank mathematical parity

**First action; stop after two working days if the first mismatch cannot be localized.** Reuse the existing fixed B4 width8/depth6 fixture, no labels or fresh seed block.

1. Build a separately named diagnostic adapter evaluating the frozen reference recurrence in float64. Do not edit frozen archives.
2. Evaluate the compact full-rank interpreter in float64 on the identical float32-quantized weights. Freeze all term selection, normalization, centering and birth conventions.
3. Save each consumed Wick term, D3, D21, scalar K4, covariance, source birth and retained age, starting at layer0. Stop at the first mismatch and save its operands. The known `wick.(4,1)` failure at zero-based layer3 is a localization target, not permission to skip earlier state.
4. If float64 parity passes, record that as support for the precision explanation. Then establish actual float32 backend parity under a separately declared, scale-sensitive numerical test. It is not a pass for the original failed B4 record.
5. Force target-only branches in tiny tests, cover signed/repeated-index sources and final-layer readout, and verify full-rank identity across more than one reusable synthetic fixture.

Acceptance is operator equivalence under a declared numerical regime, not improved MSE. If parity fails, fix the first recurrence mismatch; do not change rank or add K4. If sensitivity is excessive, diagnose cancellation/ordering and include any stabilizing arithmetic cost. Do not widen a threshold after seeing its failing output.

## Stage 2 — implement one faithful all-age compression

**Target window: September 18–24, conditional on Stage 1.** Preserve the working operator, current scalar K4, all source ages and F21's compatible algebra. Change only representation.

Start with recent dense sources and one shared basis for older sources. Specify transport, aging/join, orthogonalization, basis rotation, residual transport and every D3/D21 read. The exact bridge uses `A=A_L, P=M, w2=w2b, s=e=0, Z=A_D, L=I`. For the generalized representation `A_D=P diag(s)+3A diag(e)+Z L.T`, reducing the residual rank is a separate approximation from reducing the common basis. Do not perform both in the first causal test.

Before labels, make a complete parameterized operation/lifetime ledger, then a counted implementation using supported flopscope primitives. Include old-source reconstruction, QR/joins, copies/casts, output materialization, full covariance work, scalar synchronizations and maximum payload/kernel shape. Stream old-source queries where exact equivalence permits it. Persistent state size alone is not peak memory.

A modular local lab can separate `reference_operator`, `source_representation`, `contractions` and `candidate_recipe`; names are illustrative, not a request for a large package migration. Export a deterministic, legal single-file candidate with only permitted imports. Hash the export and prove its conformance to the tested lab recipe. Reuse existing certification and score tools.

Use fixed unprotected mathematical fixtures for correctness and designated DESIGN data for approximation choices. Freeze one rank schedule and residual policy before fresh DECIDE. Evaluate the exploratory screen against F21, not just L2 or a different compact off arm. Stop if complete reconstruction/join cost erases the saving. A locally cheaper approximation with a different recurrence does not answer this experiment.

## Stage 3 — derive one coherent accuracy extension

**Derivation can begin while Stage 2 is implemented; large accuracy evaluation follows a stable baseline. Time box the initial tiny derivation to two working days.** Additional fourth-order structure is a hypothesis motivated by coupled local AUGMENT evidence and external author ablations. It is not a proven isolated cause or an implemented inexpensive fix.

Write an independent dense definition of the proposed extra raw moments, affected covariance contribution and induced K3 births. State precisely which scalar-K4 component is already represented and subtract overlap. Specify whether the new closure is regenerated or transported at each nonlinearity, with explicit normalization. A linearly carried synthetic bank attached to otherwise nonlinear state is not automatically the proposed closure.

Require tiny dense-versus-compact parity on the forced target branch; include the known negative-variance fixture and full-depth finite/variance diagnostics. Test assembled raw moments and covariance, not PSD of an arbitrary signed tensor core. If damping or covariance repair is necessary, it changes the intervention: predeclare it, count it and test its bias.

Do not rerun the failed scalar-core arms unchanged. If no distinct inexpensive coherent update can be written and independently checked within the time box, park that implementation and report the missing derivation. Do not turn a suggestive public ablation into an accepted formula.

For a viable extension, measure both increased arithmetic and raw-error reduction against the frozen compact baseline, with a separate control comparison to F21. An approximate 40–50% raw reduction is a useful ambition, but only net paired score matters. Do not assume the T1 factorable AUGMENT improvement survives compression or cost constraints. Combine individually verified changes only in a separately frozen candidate with its own paired decision.

## Stage 4 — reserve two bounded alternatives

These are optional feasibility tests if ordinary compression cannot reach useful ranks. They are not parallel unbounded sweeps or automatic reopenings of closed lanes.

**Query/output-sensitive basis.** On fixed tiny states, compare equal-rank ordinary leg-energy and query-sensitive bases. Include equal-norm perturbations with different D3/D21 effects and signed cancellation. Proposed screen: at least 2x lower squared final-mean perturbation at equal rank, with complete construction cost low enough to permit a lower rank in the deployed method. Reject a sensitivity pass that costs as much as the saved work. Full output testing remains necessary even if a few output directions guide the basis.

**Sketched quadratic query.** Verify the exact `Phi2(Q) B Q.T` identity independently, then freeze sketch widths/seeds and test births, repeated indices, cancellations and rotations. Price rebuilding from retained factors when the basis rotates. Require a supported flopscope update/read cycle and a plausible at least 2x old-tier saving before any labels. Do not assume FFT/scatter availability or use a dense projection while claiming TensorSketch's cost. Sketch unbiasedness for a fixed query does not guarantee unbiased nonlinear propagation.

Each receives at most two working days before a continue/stop decision on the no-label evidence. A specific reopening condition and full contract are required if it touches a closed strategic lane. See the [frontier derivation](research/2026-09-14_frontier_research.md) for scope and primary sources.

## Stage 5 — fresh evidence and release

**Target: first complete frozen candidate September 25–October 3; robustness and selection evidence October 4–10; submission buffer October 11–17.** If derivations fail, preserve existing validated entries rather than forcing a late speculative replacement.

For every strategic experiment: commit the question, mechanism, predictions, baseline, isolated score gate, counted-cost model, confounders and seed claim before touching data; record Samuel's three predictions first. Use a genuinely fresh block. P10 and earlier spent blocks remain spent. A nominal panel around 128 networks is a planning starting point, not a substitute for effect-size/power and truth-noise analysis. Set replica/chunk choices before results. Do not add observations or samples in response to an interim result; any staged precision or sequential design must be preregistered.

Pair candidates on common networks and truth; quote raw MSE with its noise floor and measured/noise-corrected/official-precision forecasts. Use the mandated arithmetic difference interval and canonical decision function. Predeclare resource-failure treatment and preserve every failed row. Require the independent Skeptic to inspect the contract, diff, protocol and raw paths without Builder reasoning before promotion.

Then run candidate-specific export/contract validation, numerical tests, full counted ledgers, Linux memory/lifetime checks and the official capped mini under the standing release rules. Keep no-cap forecasts separate from cap passes. Changed synchronization, larger batches or deeper Strassen need explicit residual/wall uncertainty and relevant new checks. Reuse accepted evidence only where the bytes, dependencies and tested behavior remain valid.

Every heavy job gets a recorded optimization review before workerq submission: duplicate loaders, workers/prefetch, cache, batch/chunk size, passes, recovery, device placement and reuse of valid artifacts. Prefer a defensible 16–24-GiB host envelope on the 5090/64-GiB machine; reserve higher if measured peak requires it. Declare conservative RAM/VRAM/CPU estimates for the actual program. The grader's 8-GiB participant cap is distinct from total worker reservation. Small one-thread tiny tests can run directly.

A submission requires Samuel's per-submission approval after concrete evidence. No agent designates. Final two-entry choice should consider private-distribution performance and operational robustness, not merely sorted public scores. Resolve source-authenticity eligibility for the actual lineage before relying on a derivative entry. Keep the contribution write-up tied to exactly one successfully graded submission, with independently reviewed claims and source attribution; deadline October 24.

## Definition of progress

Progress is a smaller unresolved question with durable evidence: a localized recurrence discrepancy, a parity-proven representation, a complete lower-cost program, or a fresh net score gain. A new diagram, an incomplete FLOP estimate, a tiny branch-disabled pass, or another interpretation of public rows is not an architecture milestone. Update the short current status and append the LOG after each accepted slice so the next driver can resume without this conversation.
