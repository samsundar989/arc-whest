# CURRENT_STATE.md — read this first

**September 13 P10 truth is finalized; one protected score read remains.**
Samuel answered "I don't know" to all three new predictions; that response
was committed at `ce853f7` before any P10 data. The fresh single-use claim was
committed at `883bd45`; workerq 1176 SUCCEEDED with two `2**24` truth
replicas on all 128 networks. The artifact SHA256 is
`9026b4ea66900695929cc9a32f1a801258733c176ae100b079e979907a04de30`.
The producer's disagreement ratio is `0.9760`; its printed per-replica
noise `4.4727e-9` gives a midpoint screen of `2.23635e-9`, or `5.50%` of
the exposed L4 MSE, below the predeclared 10% ceiling. The ledger finalized
the exact hash, and the truth-read register remains spent 0/1. A fresh
pre-score Skeptic must review the binding before scorer submission. The same exact F21 candidate SHA
`d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94`
is frozen against original L4, with L2 context. P09's only protected read
ended without a score and cannot be repeated. P10 allocates fresh
`40,913,010,000..127` as SINGLE_USE (registry commit `24e8b32`) and still has
one finalized claim and zero truth reads. The [P10 contract](../experiments/LOG.md) keeps
the paired arithmetic `competition_score.decide(gate=1.05, interval="t")`
on the official-precision forecast with the conservative `+2**24` FLOP
offset. It explicitly disables both **local scoring** wall/residual caps so
host sleep cannot zero accuracy rows; this is not a deployment test. The
pre-data [protocol Skeptic](../experiments/2026-09-13_p10_protocol_skeptic.md)
and [registry Skeptic](../experiments/2026-09-13_p10_registry_skeptic.md)
accepted their narrow scopes; affected default preflight 1166 passed 12/12.
The producer's printed per-replica truth floor must be divided by two for
the midpoint before the one protected score read. After a
valid fresh score, full candidate checks and a **clean capped** official
mini remain release gates; a no-cap mini cannot substitute.

**September 13 F21 lifecycle diagnostic closed; P09 still has no verdict.**
The preregistered, label-free stress on the first 128 exposed P07 DESIGN
identities used frozen F21 SHA `d77dab0e...`, the P09 scorer's Phase-2 spec
with only the residual cap disabled, canonical mint and `f64_readonly` weights.
Workerq 1150 completed 128/128. Its sole failure was row 27,
`time_exhausted=true`, local wall `17,914.856` s; the Windows System log
records a `17,892.563`-s host sleep during this run, so that row is strongly
environment-confounded. All other 127 rows succeeded, maximum local wall
`71.860` s, each with `928,457,637,478` counted FLOPs; no estimator exception
or budget flag reproduced. This does **not** reveal P09's discarded failure
category, establish graded-host wall or residual behavior, produce an accuracy
score, or promote F21. The [raw diagnostic](../experiments/2026-09-13_f21_labelfree_stress.json),
[host sleep events](../experiments/2026-09-13_f21_labelfree/host_sleep_events.json),
and [fresh Skeptic verdict](../experiments/2026-09-13_f21_labelfree/skeptic_verdict.md)
are preserved. The Skeptic ACCEPTED this narrow diagnostic only. The first
affected preflight was RED on an undeclared deliberate P07 overlap; an
append-only pairing declaration repaired the ledger refusal, and workerq
1152 passed the retry 12/12. Both transcripts are retained. F21 remains
**unpromoted and not submission-ready**.

**F21 engineering and DESIGN passed; P09 DECIDE produced no verdict, September 13.** The separate L4-derived
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
accepted the three-repeat local memory gate. Candidate release checks and
submission approval remain open; F21 is **not a ready submission**.

**Current gate:** Full preflight 1092 passed 16/16, repaired default 1095
passed 12/12, and P09-registry default preflight 1097 passed 12/12. P09 was
claimed once (128 seeds); workerq **1098** completed the two-`2**24`-replica
truth mint. Replica disagreement ratio was 1.0102 and predicted midpoint
truth floor about `2.2442e-9`. Workerq **1099 FAILED exit 1** after original
L4 and fused L4 each reached 128/128: at least one fused lifecycle row
failed, but the scorer discarded failed-row flags before writing the JSON.
L2 and `competition_score.decide(gate=1.05, interval="t")` never ran. The
failed seed and cause cannot be recovered; the P09 truth read is spent 1/1
and may not be repeated. The fresh
[result Skeptic](../experiments/2026-09-12_f21/p09_result_skeptic.md) REJECTED
promotion evidence, without calling this a statistical loss or scientific
kill. The read-only
[forensic note](../experiments/2026-09-12_f21/p09_failure_forensics.md)
records what the terminal artifacts can and cannot show. A minimal scorer
diagnostic now logs only failure flags/timing before the unchanged abort;
workerq **1141** passed its affected default preflight 12/12, and the
[diagnostic Skeptic](../experiments/2026-09-12_f21/p09_failure_diag_skeptic.md)
ACCEPTED only that repair. The frozen F21
selection remains SHA `d77dab0e...` against original L4. The separate compact
B0/B1/B2/B3 probes establish n4/depth4 tiny full-rank recurrence parity only,
not compression or score. A separately preregistered n8/depth6 B4 extension
failed its fixed `1e-5` parity gate: 85/841 checks failed, worst normalized
error `4.04473e-4`; all age/activity/rank checks passed. The raw artifact and
contract are in LOG. B4 affected-tree preflight passed 12/12 and its fresh
Skeptic accepted only that failed tiny gate; no wider or scientific closure
follows. P08 remains exhausted. No vault/public F21 labels,
fresh DECIDE result or submission approval exists.

**September 12, 2026: strategist review of worker d124234.** Then read LOG from the bottom. The preceding detailed chronology is [archived verbatim](archive/CURRENT_STATE_before_worker_review_2026-09-12.md); all original contracts and results remain in experiments/LOG.md.

## Next discriminator: fresh F21 decision after the scoped diagnostic

P07's single observed timeout was confounded by host sleep; the other 127
rows have no local lifecycle failure. Do not rerun P09, infer its category
from P07, or treat this exposed DESIGN diagnostic as a score. A new F21
DECIDE requires a newly claimed and preregistered single-use block, new
Samuel predictions, the binding paired arithmetic gate and an explicit
local-wall handling plan in its contract. Only a passed fresh gate can
justify candidate-specific full certification, official mini and submission
review. The optional one-row no-sleep P07 check suggested by the Skeptic
would remain diagnostic, never a DECIDE substitute.

Use a new candidate file. The best-public L4 parent is inside `submission-k3strassen-L4.tar.gz`: estimator.py SHA256 `052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58`. Its relevant functions match L2 by independent AST comparison. Preserve the conservative generation/budget chooser, scalar K4, precision, saturation and shape behavior. Do not edit frozen files or shipped estimator.py.

Read [STRATEGIST_RESPONSE](STRATEGIST_RESPONSE_2026-09-12.md) for equations and evidence, [GAME_PLAN](GAME_PLAN.md) for execution, and [CODEX_EXECUTION_HANDOFF](CODEX_EXECUTION_HANDOFF.md) for the task boundaries. The F21 cost and score thresholds were preregistered in LOG; engineering and exposed DESIGN passed, but no fresh DECIDE result exists.

## Scope changed explicitly

Samuel answered **“Include eligibility and candidate preparation.”** The old Stage-3 hold is lifted for those activities. Actual submission still needs individual approval; external messages and designation are not authorized. [Current eligibility review and unsent clarification](ELIGIBILITY_REVIEW_2026-09-12.md). Do not let an unresolved participant-derived reconstruction block preparation of an independently documented own-code change.

## Accepted evidence and actual blockers

| Area | Current status |
|---|---|
| Stage 0 | Scoped independent ACCEPT; Linux positive plus M1–M12 aggregate 13/13. This is instrument coverage, not a complete future candidate certificate. Reuse it unless an affected path or new failure requires retesting. |
| Preflight | Workerq 1092 passed Full 16/16; default 1095, P09 registry 1097, B4 1100, F21 scorer diagnostic 1141 and label-free stress retry 1152 passed 12/12. Initial label-free 1148 was RED on undeclared P07 pairing; raw failure and append-only repair remain in LOG. Candidate-specific full checks remain conditional on a valid fresh DECIDE. |
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
