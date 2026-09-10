# CURRENT_STATE.md — read this first

**Current as of 2026-09-10, after the repo/competition/literature audit.** This replaces the accumulated snapshot, preserved verbatim in [the pre-audit archive](archive/CURRENT_STATE_before_2026-09-10_audit.md). Historical contracts and verdicts remain binding. Then read [experiments/LOG.md](../experiments/LOG.md) from the bottom.

## Where we stand

- **Winning architecture not yet identified.** Persistent higher-cumulant propagation is the strongest concrete direction, but current SIMPLE-K3 is too expensive and leaves accuracy on the table.
- **New evidence:** 504aldo released compact all-history K3 with coherent K4 regeneration and shared old-source bases. Reported adjusted 5.40e-9/raw 2.13e-8/0.2526 budget. This supports specific new contracts, not blanket lane reopening. Public-fit constants and mutable-buffer assumptions are not our protocol. [Pinned code analysis](research/2026-09-10_literature_review.md).
- **Current public leader:** marius_binner #330500 adjusted 2.683e-9/raw 1.700e-8/0.1578 budget. **Our rank 44**, best #330290 adjusted 1.836e-8, a descriptive 6.84× gap. Official MSE noise floors unavailable; these ratios are not promotion statistics. [Chrome evidence](research/2026-09-10_competition_sources.md).
- **`estimator.py` remains v2.6.1**, matching its archive; it is the shipped baseline, not the active K3 candidate. No estimator/archive/instrument/gate/seed/vault changes in this audit. No promotion, submission or designation.
- **Next:** Stage 0 trust repairs and Stage 1 representation derivation in [GAME_PLAN](GAME_PLAN.md), then fresh preregistered research reconstruction. Before treating a participant-derived reconstruction as a submission candidate, resolve rules §6's restriction on derivation from another participant's submission. MIT attribution or a mathematical rewrite alone does not settle eligibility. No large training before a feasible representation and useful reachable-state target.

## Candidate state

| Role | File/submission | Evidence |
|---|---|---|
| Shipped baseline | `estimator.py`, v2.6.1 | Default preflight targets this file, not every nominee |
| Best public result | Strassen L4, #330290 | Public adjusted 1.836e-8/raw 3.779e-8, 0.4858 budget; worst displayed server wall 114.6798 s against 120 s |
| Prior compromise | Strassen L3, #330327 | Historical graded evidence/exact bytes in LOG/archive; no fresh certification here |
| More robust fallback | `estimator_k3_strassen_L2.py`, #330566 | **Now GRADED**; public adjusted 2.284e-8/raw 3.779e-8; certified multiplier 0.604217, 9,321 ops; worst displayed wall 33.0009 s |
| Earlier fast fallback | `estimator_k3_full.py`, #330212 | Historical evidence retained; pair rationale must use best eligible score, not average/worst-of-two |

L2 SHA256 `390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db`. Public failures 0/50; all 100 visible statuses completed, private accuracy sealed. Exposed public raw/adjusted means and worst wall meet the corresponding prediction intervals; the full hidden-suite mean was not revealed. Prediction 9,304 ops already missed; residual ≤0.15 s remains unverified in the inspected UI. **Two historical certification gate failures remain failures.** Good grading does not amend their contract.

Observed cap/worst-wall ratios: L2 **3.64×**, L4 **1.05×**. These are dated server observations, separate from contention-adjusted planning margins; no future guarantee. Frozen comments still quote obsolete 1,022 matmul pricing; current SIMPLE shape model is 797. Correct manifests/docs without silently editing frozen bytes. [Architecture audit](research/2026-09-10_architecture_audit.md).

## Current competition facts

Phase 2 opened August 22, announcement August 23: 1024 × 16, Gaussian-input He-initialized bias-free ReLU. Budget 2^41, lambda 0; score is mean of per-network final MSE×max(0.1, F/B). Failed predictions use zero-output error without discount. Installed/current graded WhestBench 0.16.1, flopscope 0.12.1.

Rules explicitly publish **400 ms residual, 120 s wall, 5 s import/setup**, 8 GB participant memory; installed enforcement 8192 MiB. Participant one physical core/two vCPU, backend seven physical cores/fourteen vCPU. Local absolute wall/residual seconds are not server authority. Public mini 100/full 1000 with 1e9-sample truth is separate from leaderboard 50 public/50 sealed and fresh prize rerun.

**Rules v14 §7 resolves best-of-two:** only the participant's best-scoring eligible submission is ranked. §6 permits up to two, defaults to two best valid public submissions if none selected. The old §6-only “checked negative” and drafted clarification request are obsolete. Selection interface/deadline remains unpublished. Only Samuel designates; every submission needs his specific approval.

Deadlines: registration/team freeze **October 2**; submissions **October 17, 23:59 UTC**; write-up **October 24**; tentative winners November 15. Ten submissions/team/UTC day. [Official evidence](research/2026-09-10_competition_sources.md).

## Slice 5: supported scope

| Mechanism | Current status |
|---|---|
| Packing/symmetric tags on existing operands | Saving ~1.0067× failed its gate; not a representation-independent0.5B floor |
| Partial AUGMENT deletions, T1 | Failed; dependencies coupled; does not reject jointly derived K4 uses/births |
| Smooth final-readout correction, T2 | Gain ~1.0163× too small; corrected controls/ridge support that scoped negative |
| T2 P08 provenance | Read three times during repairs; spent, not pristine single-use confirmation |
| In-flight sensitivity arm | Never implemented/run; not settled by final readout |
| Three-generation carry/selectors | Scoped negatives stand; all-age shared coordinates are different |
| Old adjoint kill | Retracted; instrument broken; speculative, not ready to rerun unchanged |

**Whole score-lane closure was already retracted in LOG and must not be repeated.** New public mechanisms change the priority, without erasing tested negatives. New contracts must name the external evidence and exact distinction from a closed intervention.

Local T1 cross-MSE SIMPLE 4.004685e-8 versus AUGMENT 1.912323e-8, per-replica noise 1.1552e-9; full reference not demonstrated affordable. T2 per-replica noise 4.4988e-9. These are not official leaderboard errors. Raw paths and qualifications: [architecture audit](research/2026-09-10_architecture_audit.md).

## Validation and remaining repairs

Verified: configuration **96/96**, certifier aggregation **19/19**, default preflight **GREEN 12/16, four skipped**, shipped-file FLOP anchor 19 categories / zero disagreements. Queued job 539 declared 8 GiB RAM, 0 VRAM, 2 CPU, no GPU. [Evidence](research/2026-09-10_validation_audit.md).

Outstanding: memory-proof acceptance not bound to shape/cap/budget/runtime; Linux/fresh-worker input representation mismatch; Phase 1 vault bypasses canonical lifecycle; claims do not count reads and lack atomic locking; pairing lacks network IDs; nondefault t-interval alpha ignored; local/runtime-forecast labels overlap; numerical tolerance/byte manifests need candidate specificity. **No instrument repair was implemented during this audit.**

Common-label noise cancels in raw MSE differences, not automatically in adjusted scores with unequal multipliers. Future reports need cross-MSE and official-precision noisy-score forecasts. This mathematical correction does not assert a particular past promotion changes outcome. [Repair details](research/2026-09-10_validation_audit.md).

## Durable operating state

AGENTS and VALIDATION remain binding. IMPORTANT instrumentation: written prediction, preflight, LOG delta, fresh-context Skeptic. STRATEGIC: committed preregistration, Samuel's predictions, fresh claimed blocks, binding arithmetic gate. This roadmap grants no seed/vault touch, gate amendment, submission or designation.

HUMAN_MODEL remains human-owned. Existing vault seal/touch counter intact, historical Phase 1. Heavy workloads through workerq with declared resources. No autonomous optimization loop, tracker or new orchestration framework needed. Parallel agents were user-requested for this audit; they do not replace the independent Skeptic.

Contribution write-up remains stale/do-not-submit pending claim review. Attribute prior public work and distinguish replication from new tested contributions. The audit is not a submitted novelty claim.

Current map: [GAME_PLAN](GAME_PLAN.md) → [research synthesis](PHASE2_RESEARCH_AUDIT_2026-09-10.md) → [VALIDATION](VALIDATION.md)/[GOTCHAS](GOTCHAS.md). September 5 strategy, old GAME_PLAN and future-named September 11 packet are historical; conflicting next-action/closure statements superseded. Original snapshot retained in [archive](archive/CURRENT_STATE_before_2026-09-10_audit.md); all contracts/results in [LOG](../experiments/LOG.md).
