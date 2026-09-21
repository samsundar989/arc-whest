# Strategist response — September 17: repair closure equivalence before changing representation

## Decision

**Continue toward a major performance submission. Do not submit the approximately 10% bundle, restart with a different estimator family, or pivot the whole effort to the write-up. The next major slice is an independently derived nonlinear closure plus controlled source approximation, followed by shared-factor cost reduction.**

The worker's principal accuracy numbers reproduce. The no-submission decision is correct. Its conclusion that the gap has been isolated to the base third-order source chain is not established: the tested transplant left a large nonlinear-table difference and a different compression boundary. The September 16 audit also missed the larger table discrepancy; this review corrects that omission rather than attributing the entire miss to the worker.

Execution handoff: [CLAUDE_CODE_HANDOFF_2026-09-17.md](CLAUDE_CODE_HANDOFF_2026-09-17.md). This response establishes the next direction and prospective targets. It is not a new measured candidate, an experiment registration, or permission to submit unspecified bytes. The Builder must register the actual intervention, data and gate before execution and obtain the required new predictions.

## What the work actually established

| Finding | Review verdict |
|---|---|
| v32 improves v31 by 1.0318x adjusted on P11 | Reproduced. The 1.5x slice gate failed; the smaller secondary gain stands. |
| v32_i beats v32 in the paired comparison | Reproduced. Its diagnostic branch requires the promised new contract/different block before selection. |
| v34 and the joint recipe hurt the tested base | Reproduced. Reject those tested combinations, not every fourth-order/closure interaction. |
| Public v29 is 1.5449x better adjusted than v32 on P11 | Reproduced. Raw gain is about 1.294x; the remainder is compute. Do not call 1.545x the raw gain. |
| P3 failed | It failed its cost screen; useful-rank accuracy was never measured. |
| P4 cannot reach 0.27B | None of the tested spellings meets that gate. No general exact-arithmetic impossibility was established. |
| N1 is closed at affordable ranks | Deprioritize the tested tiny-width policies. The latest eight-network screen is exploratory and registration-deficient, not a live-shape exclusion. |
| Stage 0 is done | Useful repairs landed; two producer/consumer and shell-portability defects still block reliable release integration. |

P11's **per-replica** truth noise is about 4.7634e-9. The scored average of its two replicas has noise **2.3817e-9**. The observed MSE rows and noise-corrected score must remain separately identified. Numerical organizer truth-noise floors are not exposed for the public graded scores.

The new validation review reproduces the principal rows and confirms that the primary stage-1 contract predates its actual data read. Do not invalidate that result because its hand-entered timestamp was wrong. In contrast, N1 job 1526 started at 01:59:52Z; its registration commit was 02:00:27Z and was absent from the queue snapshot. Its literal seeds were unregistered. This proves that committed-before-sampling provenance failed, not that results were seen before writing the rule. Preserve it as exploratory; never retroactively manufacture registration. Its four networks per width and sharp width-dependent effects also limit extrapolation independently of that defect.

## The overlooked discrepancy: we are running different nonlinear closures

The actual v32 table has 25 terms. Public v17 has 62 base terms plus four `(3,1)` rows. Counts alone are not a mathematical comparison, but the semantic mismatch is direct: its **executed final mean** contains

```
D3^2 * w(6,1) / 72 + D3*D4 * w(7,1) / 144
```

while ours contains only `w(0,1) + D3*w(3,1)/6 + D4*w(4,1)/24`. The added coefficients follow from the quadratic expansion of the cumulant differential operator. There are also many additional mixed products affecting covariance, repeated third-order moments and regenerated fourth-order state. Therefore the assertion that “identical fourth-order terms help them and hurt us” describes only the copied rows, not an otherwise identical nonlinear program.

These terms are live in v17's compiled program and final readout, not disabled fitted calibration metadata. Its inherited table was empirically pruned on public data, so **do not import that pruning mask as our mathematical closure**. Independently enumerate a stated set of diagrams/truncation orders using our pinned upstream and derivations. Matching the label SIMPLE across versions does not establish matching diagrams. Our `pK` convention is connected; ordinary raw moments cannot be inserted into its centering table unchanged.

This gives two cheap first questions: how much do missing univariate terms change the final readout on an unchanged trajectory, and how much do the mixed terms change the recurrence? They require O(n) and O(n²) local work, respectively, before any source transport changes. Neither their size nor their sign is known on our chain.

Historical A5 is a scope constraint, not a reason to ignore this difference. It found raised diagrams harmful with a scalar fourth-order state, while a richer fourth-order state plus raised diagrams worked. The new hypothesis couples a derived closure to the **already regenerated per-neuron core**, and tests the interaction explicitly. It does not reopen “more terms always helps” or “discard fourth-order state.”

## The second discrepancy: what is protected from rank truncation

Public v17 removes analytically known diagonal/rank-one feed pieces from the residual before approximating that residual, then carries those pieces in exact extra factors. Our route-(i) recipe leaves their repeated-support equivalents inside the residual being approximated.

If `P_r` denotes rank-r approximation, the difference is of the form

```
P_r(R + E) - P_r(R) - E.
```

It vanishes at full rank but generally not at a restricted rank. This can alter future consumed slices even when two uncompressed births represent the same tensor. The public method also uses a different rank and sketch. All three details must be controlled before treating “the same recipe” as the same algorithm. Exact protection may help, hurt, or be too expensive; measure it rather than assuming the answer.

The architecture report includes a deterministic small algebra demonstration. It is not a performance measurement. The relevant new candidate preserves our uncompressed mathematical tensor while changing its approximation boundary, rather than copying public implementation bytes.

## A cheaper and more decisive shared-state test

Do not construct a dense 1024³ tensor to compare local nonlinear consumers. They read `mu_pre, C_pre, D3, D21, dG, lambda` and layer/configuration metadata: O(n²) state. Capture these boundaries from both chains, keep their identities, and feed each boundary to both local programs. Inspect Wick coefficients, individual connected `pK` terms, centered outputs, source births and regenerated core in order.

Then compare new-source factors before/after equal-rank compression and after one common dense transport. Tiny dense fixtures establish the adapter's algebra; live-width factorized slice queries establish that the diagnostic exercises the actual shape without cubic memory. Do not compare arbitrary unequal factor coordinates as if they were tensor entries.

Be explicit about activation convention, scaling, symmetrization, table order, reference shape gates, dtype, clipping and covariance stabilization. Public v17 disables regeneration away from 1024×16 unless the diagnostic adapter explicitly changes that gate. A naive tiny reference run would test the wrong mechanism.

Equal first four moments do not identify an underlying probability law, so this is an **implementation/closure equivalence test**, not an exact physical-truth oracle. A standard Gaussian and the three-point variable taking 0 with probability 2/3 and ±sqrt(3) each with probability 1/6 share their first four moments but have different ReLU expectations. Use a specified realizable distribution for any quadrature claim about exactness.

## Next slice and prospective decisions

Timeboxes below control unproductive exploration; they do not authorize skipping correctness or stopping useful implementation merely because a calendar estimate expired.

| Step | Work and concrete deliverable | Decision |
|---|---|---|
| 0, hours | Fix receipt filename discovery and child PowerShell/module compatibility; pass a cross-tool fake-run fixture with source hashing | Do this before costly confirmation. No new harness platform. |
| 1, about 1 day | Canonical diagram inventory; independently generated selected closure; tiny connected-moment oracle; live-width shared-boundary report | Identify actual differing terms and approximation boundaries. No truth is needed to establish algebra. |
| 2, about 1–2 days | Predeclared final-readout-only diagnostic; recurrent selected closure with coherent centering/regeneration; protected-feed comparison at equal rank | Score a bounded set of meaningful arms, retaining the original v32 control. New combined-base advancement target: `decide(..., gate=1.15, interval="t")` on adjusted score; existing failed gates remain failed. |
| 3, about 2–3 days | On a supported coherent base, retain recent sources accurately and older per-source factors in shared age-tier coordinates; reuse verified P4 transformations | Price the complete lifecycle before accuracy. Initial design target <=0.20B and <=5% raw loss versus the frozen new base; this is an intermediate objective, not a new N1 verdict. |
| Optional attribution | Reconsider thin feedback only if the new base changes its mechanism; one predeclared equal-policy comparison | The old +1% gate remains failed. Any new joint-base economics must be registered before data and decided on adjusted score. |
| Release | Freeze one supported bundle, one fresh decision where required, one public CONFIRM and all candidate-specific deployment checks | Major-release target: paired observed gain >=2.0x versus graded v31's frozen bytes and `decide(...,gate=2.0)` pass, corroborating projected public score <=4.4e-9, zero failures, sound deployment margins. No upload merely for a small gain. |

`decide(gate=2)` requires the observed arithmetic gain to reach two and the paired difference interval to exclude zero; it does **not** prove that the entire confidence interval lies beyond twofold. Report that distinction. A hidden-suite score forecast is not a guaranteed rank. The approximate 4.4e-9 goal carries forward the prior major-release objective and makes the user's “significantly better” requirement concrete; preregister it before the new frozen comparison.

Use existing P11 only for its permitted exploratory/diagnostic role. The earlier v32_i selection condition still requires its separate contract/different block; it cannot be silently promoted as the baseline. Allocate future corpora through the registry, commit before queue submission and before truth generation, use actual timestamps and maintain independent truth streams. Reuse valid truth and control predictions. Freeze fitted constants before a fresh SINGLE_USE DECIDE read. Do not spend a fresh corpus on every tiny algebra edit.

If a coherent base reaches a strong release early, do not delay it for an unrelated speculative tensor project. If it does not, finish the bounded shared-factor path and deliver exact negative results plus the next discriminating question. Do not repeatedly ask the strategist after every failed patch. A later aggregate-query sketch remains conditional on a complete affordable join/transport/query proof; the current data does not justify another broad dense-Tucker sweep.

## Architecture and winning target

The next architecture is: **mean/full covariance + explicit selected nonlinear closure + recent third-order source factors + protected analytic feed + regenerated fourth-order state + old per-source factors in shared coordinates + metered, memory-bounded contractions**. The closure and approximation boundary are now explicit components, not assumptions hidden behind “K3.” A dense merged old tensor is not required.

Public v29 provides a nearer reproducible benchmark, not the winning method. On September 17 Chrome showed us rank **59/166**, leader still rounded 2.4e-9 and detailed entry #331308 previously 2.448e-9, top-ten boundary 3.8e-9; 4.4e-9 is around today's rank 13. The public reference remains 5.4e-9. Matching it alone would not meet the requested release target. The underlying leader implementation remains unavailable.

At raw error 2.0e-8, compute 0.20B gives 4.0e-9; compute 0.11B gives 2.2e-9. These are score arithmetic, not predictions. The long-term winning question remains whether we can preserve sufficient closure accuracy while making old-state queries substantially cheaper. No current experiment proves that representation exists, but the newly identified closure differences make abandoning the family premature.

The targeted literature refresh found no new public code at the two pinned heads and no ready theorem that solves our ReLU problem. Goal-oriented reduction and implicit tensor sums remain conditional construction tools. The August width-independent compression theorem requires analytic activations and explicitly excludes ReLU; it does not justify a restart. See [frontier review](research/2026-09-17_frontier_review.md).

## Evidence and continuity

- [Architecture review](research/2026-09-17_architecture_review.md): executable-path table inventory, protected-feed algebra, reference shape gate, exact anchors.
- [Validation review](research/2026-09-17_validation_review.md): reproduced statistics, noise conventions, integration failures, N1 chronology and uncertainty, P4 limits.
- [Independent Skeptic](research/2026-09-17_independent_skeptic.md): fresh raw-source challenge to the equivalence claims and remaining limitations.
- [Live board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2) and [public method discussion](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218), refreshed through Samuel's Chrome. No new posted mechanism was identified.

The internal policy already authorizes independently implementing mechanisms while retaining provenance and keeping public source bytes out of submitted packages. This does not require a new organizer-question loop. Submission remains per-file approval after the package is concrete. No agent designates. All heavy work remains subject to the recorded optimization review and workerq resource reservation.
