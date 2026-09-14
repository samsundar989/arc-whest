# Architecture audit — 14 September 2026

Read-only research for Samuel's requested competition review. This is a proposed architecture decision, not a promotion, registered experiment, independent Skeptic verdict, or write-up claim. I read the current snapshot, LOG tail, active plan and September-14 packet, actual F21 and baseline source, full-rank compact probes, previous architecture reviews, and selected existing result summaries. No candidate or instrument was edited, no experiment was run, and no protected truth archive, seed block or vault was opened. External literature/Chrome freshness belongs to the companion research agents; external numbers cited here are explicitly dated repository observations.

## Decision

The repository has identified a credible **architecture family**, not a demonstrated winning architecture: retain the nonlinear cumulant recurrence and all source ages; reduce its representation cost; then improve the missing fourth-order structure coherently. The strongest active candidate, F21, validates the usefulness of exact algebraic optimization, but does not validate compression or a better closure. Its accuracy is essentially unchanged from L4, and cost-only improvement has an algebraic ceiling above the latest recorded leader.

Proceed with the compact full-rank differential as a tightly bounded engineering prerequisite. Its first step is already specified and genuinely discriminating. In parallel at the planning level, require an explicit accuracy mechanism and joint error/cost target. Do not allow another long compression project to claim it is by itself the route to first place. Keep F21 as the primary comparator and L2 as the operational fallback; root estimator.py is historical.

## What actually executes

| Object | Actual behavior | Source anchors |
|---|---|---|
| Root `estimator.py` v2.6.1 | At width1024, the width limit bypasses its Sobol sampling path and selects Gaussian/K2 analytic propagation. It is archive-identical, not the latest graded algorithm. | `estimator.py:525`, `:530`, `_analytic_only` |
| F21 submission #330950 | Full retained SIMPLE factored third-cumulant history, scalar projected fourth cumulant, float32 state, batched Strassen depth4, fused D21 contraction. | `estimator_k3_strassen_L4_f21.py:419`, `:638`, `:767`, `:1005` |
| L4 #330290 | Own frozen parent of F21; same nonlinear method, one extra D21 matrix product. | `experiments/2026-09-12_f21/estimator_L4_parent.py`; parent-to-F21 diff |
| L2 #330566 | Same working recurrence with shallower Strassen. A robustness comparator, not the newest best public score. | `estimator_k3_strassen_L2.py` |
| Existing compact candidate template | All-age shared-basis approximation, but its original nonlinear recurrence is not equivalent to working L2/F21. Its old off arm is not a valid full-rank baseline. | `local_sim/p2_compact_candidate_template.py`; September-12 architecture review |
| B1–B4 compact differential | Independent interpretation of the working nonlinear table with full-rank source reconstruction. B4 has unresolved numerical parity failures. | `local_sim/p2_compact_fullrank_deep_b4.py:243` |

The current state is `(mu, covariance, tower, c4)`. At each layer F21 computes `mu_pre=W.T@mu`, `cov_pre=W.T@cov@W`, transports old factors, forms the needed third/fourth cumulant slices, evaluates the entire frozen Wick/power-moment table, converts raw moments to centered moments, and appends a new third-cumulant source after subtracting the contribution already carried by old sources. It retains the existing scalar fourth-cumulant projection. At the final layer only mean-relevant output computations remain.

“Exact tower” means exact representation/transport of the selected approximate SIMPLE cumulant program, modulo floating point. It does not mean exact integration of the ReLU network, all possible third-order diagrams, or a complete fourth-cumulant law. This distinction explains how preserving the tower exactly can coexist with substantial residual error.

For `h` live source ages, the tower uses wide factors `A_D,A_L,M` with about `n*h*n` entries each. The youngest `M=I` is symbolic. The fourth factor is reconstructed as `C=A_L*w2b`, because every later transform acts on its left while the fixed birth scaling acts on its right. D3 is the row reduction

```
sum(A_D*M*M) + 3 sum(A_L*M*C).
```

F21's D21 numerator is

```
[2(A_D*M)+3(A_L*C)] @ M.T
  + (M*M) @ A_D.T
  + 6(A_L*M) @ C.T.
```

The `/3` and off-diagonal mask follow. This is three wide products, independent of the number of Python lineage objects. `_advance` also uses three wide products. Wide batching primarily saves metered operations; the lost product in F21 saves arithmetic. Neither mechanism is a learned approximation.

F21 source comments still contain historical pre-optimization costs and product counts. The current executed ledger and version manifest must override those comments; changing frozen bytes just to tidy historical comments is unnecessary.

## Evidence and the score ceiling

**CONFIRMED in existing accepted artifacts:** F21's fresh P10 paired arithmetic gate passes at 1.150566x versus original L4; the useful effect is 13.08574% fewer counted FLOPs. P10 raw MSE is approximately `4.1821e-8`, with midpoint truth-noise floor `2.2363658e-9`. F21 graded 100/100 without failures and bills `928,457,637,478` FLOPs, multiplier `0.42221365`. Its observed worst graded residual is 0.15623225s (2.56029x headroom); worst predict wall is 93.1597153s (1.28811x headroom). These are measurements of that run, not a private-rerun guarantee. The local capped-mini refusal remains a separate true observation.

F21 public raw MSE is `3.77943617e-8`; the official public truth-noise floor is unavailable. Its adjusted score is `1.59572954e-8`. The September-14 stored board has a rounded leader at `2.5e-9`. During this audit the parent agent's fresh Chrome inspection of leader submission #330876 exposed displayed detail `2.540e-9`, raw MSE `1.891e-8`, and mean budget percentage 13.43%, with variable public-row FLOPs approximately `2.77e11`–`3.11e11`. Those displayed values refine the target used below; exact full-precision telemetry and both public truth floors remain unavailable. These public levels are descriptive, not a paired selection gate, and variable work does not identify the leader's method.

For a network-independent cost, the conditional raw error needed to match that dated score is:

| Cost multiplier | Raw MSE to tie displayed 2.540e-9 | Required reduction from F21's displayed raw error |
|---:|---:|---:|
| 0.42221365, current F21 | 6.016e-9 | about 84.1% |
| 0.20 | 1.270e-8 | about 66.4% |
| 0.15 | 1.693e-8 | about 55.2% |
| 0.10, minimum | 2.540e-8 | about 32.79% |

At unchanged reported error, arbitrarily cheap F21 would score `3.7794e-9`, above the displayed leader. The displayed overall gap is approximately 6.28x; the older rounded target gave 6.38x and 33.85% required floor-error reduction, which remain valid only for that dated rounded comparison. Reaching the cost floor requires reducing its present FLOPs by approximately 76.3%. Compression which preserves its error can be valuable without being sufficient for victory. No public truth-noise correction is implied by this algebra. For a data-dependent cost, the actual objective remains the mean of per-network products; a product of mean error and mean cost cannot replace it.

**CONFIRMED accuracy headroom in a different existing local panel:** the accepted T1 raw result summary has SIMPLE cross-estimated MSE `4.00468513e-8` and complete AUGMENT `1.91232321e-8`, 24 networks, paired difference `2.09236192e-8` with interval `[1.94413595e-8,2.24058790e-8]`. Recorded per-half truth noise is `1.1552e-9` in the September-10 audit. This is evidence that richer coupled cumulant structure can reduce error, not that the currently unaffordable AUGMENT program can be submitted. Tested deletions of its coupled channels were much worse and cannot identify a standalone cheap channel. Combining this accuracy level with a multiplier near 0.1–0.13 is a useful research target, not a demonstrated combination or cross-panel score prediction.

## First executable slice: remove the B4 precision confound

The raw B4 JSON independently reads 841 checks, 85 failures, tolerance `1e-5`, worst normalized difference `4.04472977e-4`. First failure is layer3 `wick.(4,1)` at `1.5996221e-5`. Its loop explicitly builds the frozen side as f32 and the independent side as f64 applied to f32-quantized weights (`p2_compact_fullrank_deep_b4.py:251` onward). Because fourth derivatives scale with inverse powers of sigma, a late failure may amplify an earlier small state difference. It does not prove that the representation itself is wrong.

Proposed new IMPORTANT probe, after a written prediction and normal contract handling:

```
Wq = existing_fixed_B4_weights.astype(float32).astype(float64)
reference = frozen_L2_layer_in_float64(Wq, full_history, scalar_K4)
independent = current_independent_B4_interpreter(Wq, full_ranks, scalar_K4)
compare_existing_checkpoint_inventory(reference, independent)
```

Preserve the existing fixed fixture, coefficient table, clipping, floor, centering, births, all source ages and final readout. Change only the reference precision and corresponding constants. Write a new artifact; do not overwrite the failed B4 evidence or retroactively relax its threshold. This is tiny CPU-only work and requires no truth labels.

If f64 fails, stop before ranks or labels; localize the first disagreement in input state, Wick coefficient, raw moment, centered moment, source subtraction or orientation. If it passes, it supports mixed-precision drift, not production f32 parity. The next required comparison puts both complete programs at the actual target precision/backend and assesses final-output drift and every consumed field under new, justified tolerances. Do not demand coordinate equality from QR factors whose basis gauge differs; compare reconstructed sources, contractions and transported state as well.

## Target representation and costs that must be priced

The faithful source bridge is

```
A=A_L; P=M; w2=w2b; s=e=0; Z=A_D; L=I.
```

The more general identity `A_D=P diag(s)+3 A diag(e)+Z L.T` is exact only with a complete residual. Reducing `Z,L` rank is an approximation. A common basis `Q` for older `A,P` factors is a second approximation, independently measurable. Preserve a fixed recent window densely and all older ages in common coordinates; old-age retention differentiates this from failed history deletion.

Let `q` be recent dense ages, `h` compressed ages, `r` shared rank, and `s` residual rank. The following is a planning inventory, not an official FLOP certificate:

| Component per layer | Leading conventional arithmetic | Important live shapes |
|---|---|---|
| Mean/covariance backbone | `O(n^2)+O(n^3)`; two dense covariance products | several `n x n` matrices |
| Recent-age transport/readout | `O(q*n^3)` | dense recent `n x qn` factors |
| Shared-basis transport | `O(n^2*r)` | `Q: n x r` plus output |
| Residual transport | `O(h*n^2*s)` | `h` pairs of `n x s` factors |
| Basis sketch, QR, joining | typically `O(n^2*r+n*r^2+h*r^2*n)` depending on representation | sketch, old/new Q, rotations and source coordinates |
| All-age factor reconstruction | potentially `O(h*n^2*r)` | an `n x n` reconstruction per active age if scheduled sequentially |
| Consumed D3/D21 contractions | must exploit shared coordinates; naive reconstruction followed by dense products can restore `O(h*n^3)` | D3 `n`; D21 `n x n`; temporary coordinate return `n x r` |
| Nonlinear table and births | `O(n^2)` per retained table term, plus any residual decomposition | covariance, moments and newborn matrices |

The uncompressed covariance backbone alone is about 68.7 billion conventional multiply/add FLOPs over 16 layers (before Strassen and full official tariffs), about 0.0312 of budget. Thus the dense covariance itself does not preclude the 0.1 multiplier; source transport, queries, basis maintenance and any richer closure are the decisive remaining bill. This is scalar shape arithmetic, not an observed flopscope total.

The persistent full dense tower at 15 ages is roughly 176 MiB in f32, much smaller than F21's 5,381.7 MiB measured process VmPeak. Batched Strassen temporaries, materializations and process overhead matter considerably. A storage-rank improvement does not automatically yield a process-memory or wall-time improvement. Count materialization and QR using installed supported primitives and meter the complete executed program. Do not reuse the old supplied-core 0.372208-budget number as the cost of this architecture.

## One useful refinement beyond generic tensor compression

**SPECULATIVE:** select the compression metric by its effect on downstream means, rather than by captured factor energy. The working recurrence consumes D3 and D21 with very unequal Wick weights; their effect also changes with depth and covariance. A source factor with little energy can matter when multiplied by a large derivative or when it perturbs future covariance. This mechanism is visible in `_layer`, not inferred from leaderboard rank.

After full-rank parity, use deterministic tiny trajectories to compare fixed-rank residuals under (a) the existing source-leg Gram and (b) a weighted metric on consumed D3/D21 and next-state changes. Freeze a single weighting construction analytically before any DESIGN selection. The first screen is whether lower proxy loss predicts lower final-output error relative to the full reference at equal rank and counted cost. That screen needs no truth labels. It is not a scientific accuracy result: closeness to F21 can preserve F21's bias.

For a later Jacobian-based version, compute sensitivities around the actual nonlinear trajectory, preserving the propagator. Do not use the old `p2_adjoint_k3.py` source-rescaling sweep as a valid linearity test: its original kill was independently retracted because multiplying accumulated state at every layer changes the propagator. No additive one-shot adjoint approximation is assumed here. Any such probe must first pass an exactly linear recurrence positive control. Pay for every sensitivity operation used online; offline tiny diagnostics cannot grant a free online basis selector.

Stop this refinement if the proxy cannot order errors on fixed synthetic controls or its basis/update overhead removes the saving. It remains an open hypothesis rather than a claimed novel theorem or demonstrated improvement.

## Accuracy extension with the strongest local support

**INFERRED priority:** a coherent approximate matrix-valued fourth-order component, with the current scalar component retained separately, has stronger local evidence than another learned final readout or a return to Gaussian-only closure. T1 establishes useful complete fourth-order-coupled structure; public work summarized on September12 supplies independent architectural motivation but not local validation or eligible implementation bytes.

Define the represented K4 tensor/core and its projection explicitly. Derive its contributions to every required raw moment from the same cumulant/Wick expansion, use the same corrected means in raw-to-centered conversion, and subtract precisely the old carried contribution before creating new K3 sources. A consumer-only or birth-only patch is not a coherent control. All linear transports must match the declared metric; a signed bank carried linearly while its surrounding state changes nonlinearly is not a physical closure.

The cheapest falsifier is a forced-active, tiny two-layer dense tensor oracle with a nonzero anisotropic K4 input: compare all consumers, centered moments and resulting births against the declared dense expansion. Then test covariance validity and derivative amplification on the existing adversarial tiny fixtures. A variance floor does not prove PSD; symmetrization does not repair negative eigenvalues. Damping or a PSD projection is a new approximation requiring an explicit intervention and a counted price. Tests must force the production mechanism at tiny shape rather than silently take an off-shape fallback.

Do not require an exact fixed-rank nonlinear signed probability law before evaluating a declared approximate closure. Four moments are not an exact sufficient statistic for ReLU pushforward; the repository already has explicit counterexamples. Equally, do not call an arbitrary projected K4 update physical just because a tensor contraction is exact. If the tiny coherent update fails, stop that implementation without generalizing to every fourth-order method.

## Implementation order and stop conditions

1. **Now:** retain F21 evidence and exact hashes; run only the newly contracted B4 precision discriminator once authorized. One compact audit, not a new framework.
2. **After parity:** implement one all-age compression approximation in a new candidate, preserving the full nonlinear operator. Meter an executed complete program before labels. A proposed meaningful engineering target is multiplier at most 0.20 with small declared deviation from F21; it is an intermediate cost target, not victory. Freeze a single choice before fresh DECIDE.
3. **Accuracy branch:** specify the coherent K4 tiny oracle independently of compression, then combine only after each control is interpretable. A useful aspirational joint target is multiplier about 0.10–0.13 with cross-estimated error around the locally observed AUGMENT level. This target is not a promised feasible configuration or a replacement for a registered paired gate.
4. **Promotion:** compare the frozen candidate primarily against F21 on a fresh, preregistered paired block with explicit truth floor, arithmetic score difference and the canonical executable gate. Keep L4 as history and L2 as robustness context. Public rows can only CONFIRM.
5. **Deployment:** candidate-specific official operation ledger, lifecycle failure retention, shape/immutability checks, actual-process Linux memory cap, and server-domain wall/residual uncertainty. The F21 local/server timing discrepancy requires explicit domain separation; its one-off exception is not a standing release waiver.
6. **Stop or fall back:** if B4 cannot be matched in a bounded audit, fix the exact recurrence discrepancy rather than sweep ranks. If complete pricing cannot approach a useful cost target, reject that spelling before minting truth. If the coherent K4 mechanism fails its tiny oracle or numerical domain, retain F21 and document the precise obstruction. Do not let another instrumentation project replace a score-directed next decision.

Minimal restructuring is enough: one current candidate-role manifest; one reference nonlinear operator; separate representation and arithmetic backends; one export path producing the legal single file and exact hashes; one candidate-specific evaluator. Preserve historical archives. Refresh CURRENT_STATE's replacement-state summary so superseded paragraphs do not again state that F21 is unsubmitted or unpromoted below the current graded result. The append-only LOG remains the historical authority, but a new driver should not have to reconcile multiple contradictory active instructions.

## Evidence anchors

- `experiments/2026-09-13_p10/decide_P10.json`, accepted independent P10 result and graded Skeptic reports.
- `experiments/run_reports/f21_diagnostic_330950_graded_report.json` and September-14 board snapshot; live refresh belongs to the parent audit.
- `experiments/2026-09-12_b4_fullrank_deep.json`: raw parity check inventory, independently inspected here.
- `experiments/2026-09-10_slice5/t1_augment_lite_v2.json`: result summary independently inspected here, with truth-noise context in `docs/research/2026-09-10_architecture_audit.md`.
- `docs/STRATEGIST_PACKET_2026-09-14.md`, `docs/GAME_PLAN.md`, `docs/research/2026-09-12_architecture_review.md`, `docs/research/2026-09-12_frontier_update.md`.
- `experiments/LOG.md`: retracted adjoint kill around 23651–23882; full AUGMENT price/coupling discussion around 37676–38369; B4 contract/results around 47796–47895; F21 terminal grade at the current tail.
