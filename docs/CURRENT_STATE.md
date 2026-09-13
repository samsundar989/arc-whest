# CURRENT_STATE.md — read this first

**F21 DESIGN passed; fresh DECIDE pending, September 12.** The separate L4-derived
candidate [estimator_k3_strassen_L4_f21.py](../estimator_k3_strassen_L4_f21.py)
is committed at SHA256 `d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`.
Its [exact diff](../experiments/2026-09-12_f21/parent_to_f21.diff) changes only
the specified `_dslice21` contraction and its local explanation. The frozen
L4 parent is [pinned separately](../experiments/2026-09-12_f21/estimator_L4_parent.py).
The exact tiny oracle and installed-flopscope 1024x16 trajectory pass; all
125 sampled fields pass, maximum normalized error `6.09644e-5` against
`2e-3`, and final relative RMS `7.30922e-7` against `1e-2`. The corrected
f64-readonly counted ledger is **928,457,637,478 FLOPs** versus original L4
**1,068,245,507,686**, a **13.08574% measured saving** at one 1024x16
network. The parent exactly matches the stored graded L4 ledger; F21 still
needs its own official subprocess evidence. Independent
[cost](../experiments/2026-09-12_f21/cost_skeptic.md) and
[full-depth trajectory](../experiments/2026-09-12_f21/full16_skeptic.md)
Skeptics ACCEPT these narrow engineering claims. The F21 contract and
Samuel's three "I don't know" predictions were committed before results.
The exposed-P01 DESIGN screen passed: F21/L4 paired official-precision
forecast score ratio `0.8691817714` against the predeclared `1.05`
maximum, with 24/24 score pairs favoring F21. Raw MSE is essentially
unchanged/slightly worse, so the gain is the measured FLOP reduction. The
[fresh DESIGN Skeptic](../experiments/2026-09-12_f21/f21_design_skeptic.md)
ACCEPTED only that screen; it is not a promotion decision. The repaired
candidate [Linux cap report](../experiments/2026-09-12_f21/f21_linux_cap_v6.json)
passed 23/23 at 8,192 MiB, and its [independent review](../experiments/2026-09-12_f21/f21_cap_v6_skeptic.md)
accepted the three-repeat local memory gate. Candidate release checks,
fresh DECIDE and submission approval remain open.

**Current gate:** Full preflight 1092 passed 16/16, repaired default 1095
passed 12/12, and P09-registry default preflight 1097 passed 12/12. P09 is
registered and claimed exactly once (128 seeds); workerq **1098** is running
the two-`2**24`-replica truth mint. No P09 score read exists. The frozen selection
is solely F21 SHA `d77dab0e...` against original L4. The separate compact
B0/B1/B2/B3 probes establish tiny full-rank recurrence parity only, not
compression or score. P08 remains exhausted. No vault/public F21 labels,
fresh DECIDE result or submission approval exists.

**September 12, 2026: strategist review of worker d124234.** Then read LOG from the bottom. The preceding detailed chronology is [archived verbatim](archive/CURRENT_STATE_before_worker_review_2026-09-12.md); all original contracts and results remain in experiments/LOG.md.

## Next measured action: fresh P09 DECIDE

When workerq 1098 finishes, verify producer quality from its log, finalize
the P09 artifact digest on the host, then make the single protected score
read. `competition_score.decide(gate=1.05,
interval="t")` on paired arithmetic F21-versus-original-L4 scores is the
promotion gate. L2 is robustness context, not the primary comparator. Do
not use exposed P01 as promotion evidence.

Use a new candidate file. The best-public L4 parent is inside `submission-k3strassen-L4.tar.gz`: estimator.py SHA256 `052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58`. Its relevant functions match L2 by independent AST comparison. Preserve the conservative generation/budget chooser, scalar K4, precision, saturation and shape behavior. Do not edit frozen files or shipped estimator.py.

Read [STRATEGIST_RESPONSE](STRATEGIST_RESPONSE_2026-09-12.md) for equations and evidence, [GAME_PLAN](GAME_PLAN.md) for execution, and [CODEX_EXECUTION_HANDOFF](CODEX_EXECUTION_HANDOFF.md) for the task boundaries. The F21 cost and score thresholds were preregistered in LOG; engineering and exposed DESIGN passed, but no fresh DECIDE result exists.

## Scope changed explicitly

Samuel answered **“Include eligibility and candidate preparation.”** The old Stage-3 hold is lifted for those activities. Actual submission still needs individual approval; external messages and designation are not authorized. [Current eligibility review and unsent clarification](ELIGIBILITY_REVIEW_2026-09-12.md). Do not let an unresolved participant-derived reconstruction block preparation of an independently documented own-code change.

## Accepted evidence and actual blockers

| Area | Current status |
|---|---|
| Stage 0 | Scoped independent ACCEPT; Linux positive plus M1–M12 aggregate 13/13. This is instrument coverage, not a complete future candidate certificate. Reuse it unless an affected path or new failure requires retesting. |
| Preflight | Workerq 1092 passed Full 16/16; repaired default 1095 passed 12/12. The new P09 registry edit requires an affected-path default run before claim. Candidate-specific checks still required. |
| Compact supplied-core implementation | Algebra/cost evidence remains scoped: about 0.372208 B and about 1,014 MiB on its supplied-core program. Not the cost of a complete physical signed-source method. |
| Existing compact off control | Already evaluated on 96 exposed P01 DESIGN networks: raw mean MSE 2.11130e-5, averaged-truth noise 9.31914e-9, score 7.83730e-6. Saved arithmetic independently checks out. It is not a matched L2 comparison and is not the working L2 recurrence. |
| New K4 arms | Frozen P01 arms failed their promotion gate. The source and covariance defects explain why scientific causation remains incomplete; do not fit them again unchanged. |
| Integrated signed source | n4 93/172; n8 186/187 with negative variance. Later positive-feed optimization preserves that failure. This path cannot support a full-width readiness claim. |
| Memory inventory | 93/93 accepted sampled n8/n16 inventories; width1024 named-payload model 715,844,864 bytes. Not a process bound or 8-GiB proof. Further symbolic lifetime work is not the next score bottleneck. |

The [validation audit](../experiments/2026-09-12_worker_review/validation_skeptic.md) recomputes existing score rows and artifact hashes. [Architecture review](research/2026-09-12_architecture_review.md) identifies the contraction and compact-recurrence gaps. No protected labels, seed claims, vault evaluation, candidate edits or heavy jobs were used for this review.

## Architecture and priorities

1. **First better candidate:** preserve the working L2/L4 full Wick recurrence and scalar K4; remove redundant D21 work. Compare fused L4 against original L4, with L2 as a robustness control. A cheaper L2 alone may still lose to L4.
2. **Larger savings:** full-rank compact-state bridge to the working recurrence, then one declared compression approximation. The current compact template uses Gaussian-only moments in places where L2 includes incoming corrections; its self-oracles did not establish L2 parity.
3. **Research extension:** coherent approximate K4 after a competitive baseline. Exact causal signed-law compression is deferred as the next-candidate strategy, not scientifically killed. Do not require a solution to that larger problem before preparing an independent working-K3 improvement.

A winning architecture is not yet demonstrated. The immediate change is a modest improvement hypothesis; larger gains still require faithful compression and/or better controlled corrections. Latest [primary-source update](research/2026-09-12_frontier_update.md): 504aldo and ARC code pins unchanged; ARC paper v2 supersedes previously inspected v1. Public tiny-shape tests bypass regenerated K4 and do not validate its target branch.

## Submission and data state

| Role | Frozen evidence |
|---|---|
| Best public | L4 #330290, dated adjusted 1.836e-8, worst displayed wall 114.6798 s. Archive/hash above. |
| Robust standing fallback | L2 #330566, dated adjusted 2.284e-8, 0/50 public failures, worst displayed wall 33.0009 s; SHA 390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db. |
| Shipped baseline | estimator.py v2.6.1, archive-identical, unchanged. |

Displayed official noise floors are unavailable for those public scores; dated wall observations are not guarantees. Chrome September12 shows our best 1.84e-8 at rank 52 and two leaders at rounded 2.60e-9. These unpaired rounded means imply roughly a sevenfold gap, not a promotion statistic. [Current source check](research/2026-09-12_competition_update.md).

P01/P07 remain exposed reusable DESIGN. P01 has at least the seven documented registered reads plus prior history; never describe it as untouched. P08 is exhausted. New DECIDE needs a fresh claimed block and committed contract. Public CONFIRM is read-only. The sealed Phase-1 vault and HUMAN_MODEL remain untouched. Previous “I don't know” predictions belong to their contracts, not all future work.

Phase2 remains 1024x16, budget2^41, lambda0, residual0.4s, wall120s, memory8192MiB enforcement. Use the paired arithmetic gate and explicit truth floors. Local timing is not server authority; current counted-package parity does not remove the separately observed server offset. Registration/team freeze October 2, submissions October 17 23:59UTC, write-up October 24.

Before every heavy or >30-minute job, record the workload optimization review required by AGENTS. Use workerq with conservative measured RAM/VRAM/CPU; the machine is RTX5090/64GiB host. Prefer 16–24GiB when appropriate without weakening evidence, not as a minimum for small work. Do not change queue priority or underdeclare. Recheck live jobs before launch.

Contribution write-up remains stale/do-not-submit until claims receive their own review. New plan supersedes old execution ordering; historical experiments and scoped negatives remain intact.
