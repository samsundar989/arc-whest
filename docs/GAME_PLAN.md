# GAME_PLAN.md — current Phase-2 roadmap

**Updated 2026-09-10 after the competition, architecture, literature and validation audit.** This is the sole current implementation roadmap. It supersedes task ordering in the September 5 strategy and [previous GAME_PLAN](archive/GAME_PLAN_before_2026-09-10_audit.md). Existing contracts, gate verdicts, seed restrictions and submission approvals retain their original scope. This plan grants no new gate, seed authorization, lane reopening or nomination.

Read [CURRENT_STATE](CURRENT_STATE.md), the [research synthesis](PHASE2_RESEARCH_AUDIT_2026-09-10.md), and [VALIDATION](VALIDATION.md) before execution.

## Target architecture

Win on the fresh private rerun, where rules §7 rank the participant's best eligible designated submission. The leading inspected public result is 2.683e-9, versus our best 1.836e-8. This moving reference is not an experimental gate. Official reference-noise floors are unavailable in the UI.

Retain means/full covariance, **persistent K3 history**, a **coherently derived cheap K4 core**, shared coordinates retaining all older sources, accurate recent sources, and a final mean-only readout. Separate mathematical approximations, exact algebraic reassociation and metered implementation. The submission stays one shape-generic legal file; no general framework rewrite is required.

The new reference is [504aldo commit 18c17e2](https://github.com/504aldo/whest-p2-cumulant-k3/tree/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45). Its reported 5.40e-9 score remains roughly twice the leading score. Use reconstruction as an **attributed research reference**; do not import public-fit constants or assume mutable-buffer legality. Rules §6 says a submission “is not derived from another Participant’s submission,” alongside public-resource/attribution provisions. A mathematical rewrite does not automatically resolve that restriction. Before advancing a public-reference derivative toward submission, obtain an explicit eligibility basis and written organizer clarification of the apparent conflict. A separately developed candidate also needs a documented provenance rationale; fresh fitting or fresh code alone is not proof of eligibility. Research/algebra can proceed without publishing or submitting a derivative. [Rules](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules), [technical analysis](research/2026-09-10_literature_review.md).

## Stage 0 — repair specific evidence gaps

**First priority. IMPORTANT instrument work, except changes to gates/data tiers, which are STRATEGIC.** Write predictions before changes, keep each diff bounded, verify it, record its knowledge delta, and obtain a fresh-context Skeptic. Do not measure an accuracy hypothesis while repairing its instrument.

| Work package | Existing location | Concrete completion evidence |
|---|---|---|
| Bind memory proof to workload | `certify_candidate.py`, `linux_memory_gate.py` | Exact SHA, 1024 × 16, budget, 8192-MiB cap/readback, runtime/instrument versions and weight mode; wrong-shape/cap/version/stale-SHA fixtures refuse |
| Match candidate lifecycle | `lifecycle_eval.py`, certifier/Linux worker | Read-only f64 grader inputs, setup once, official materialisation; repeated-worker and cold-worker tests; store both same-seed hashes and runtime manifest |
| Enforce real pairing | `competition_score.py`, result schema | Reject missing, swapped or duplicate MLP IDs; preserve arithmetic score difference and `decide(gate=...)`; handle nondefault alpha honestly |
| Audit truth access | seed ledger/shared loader | Atomic locked claims and pre-read reservations; aborted reads counted; direct protected loads audited; P08 stays spent |
| Separate reference noise from signal | `p2_truth.py`, reports | Cross-MSE and official-precision score forecast separately; per-network multipliers/noise; numerical precision/producer manifest |
| Separate runtime evidence | certifier/report schema | Math validity, local runtime result and predicted server risk visibly distinct; preserve historic refusals; policy changes separately preregistered |
| Bind exact candidate bytes | manifest/export | SHA, parent, term set, fitted-data provenance, shape policy, versions, counted ledger, memory/runtime report hashes; no archive-byte rewrite |

The sealed vault remains a **Phase-1 historical asset**. Repair its lifecycle consumer on sacrificial fixtures before any new sealed evaluation. A Phase-2 corpus requires its own strategic contract and authorized touch plan; never repurpose/reset the existing seal. Creating one is unnecessary merely to prove algebra or fix metadata consumers.

Audit baseline: configuration 96/96, certifier aggregation 19/19, default preflight GREEN 12/16 with four skipped, shipped-file FLOP anchor 19 categories / zero disagreements. These do not mean the repairs above are implemented or separate nominees certified. [Audit evidence](research/2026-09-10_validation_audit.md).

## Stage 1 — derive and price without labels

Deliver a transparent reference and complete operation-family ledger. Changes to probe math are IMPORTANT even on tiny deterministic fixtures. No decision block is needed for hand-constructed tensor identities.

1. Pin source provenance, orientations, term sets and repeated-index subtraction. Distinguish SIMPLE, factored AUGMENT and full K4.
2. Derive the two shared dense legs plus residual identity, initially retaining the residual densely. Compare D3/D21 slices with the trusted chain on tiny signed, cancellation, zero-variance and permutation fixtures.
3. Derive one coherent K4 core, including every consumer and induced birth, before toggling components. Use-only is not the proposed replacement.
4. Derive the all-age shared basis: transport, insertion, join, contraction and readout. A recent-generation cap or covariance eigenspace is a different mechanism.
5. Price the whole 1024 × 16 program: transport, contractions, nonlinear terms, QR/basis maintenance, residuals, casts, materialisation, call structure and scratch memory. Separate exact algebra from state approximations; no whole-chain “7→4 exact” claim.

Stop before labels if parity fails or fully counted cost leaves no credible route toward the current accuracy/cost frontier. A memory reduction without a cheaper complete contraction path is insufficient.

## Stage 2 — reconstruct public mechanisms cleanly

**STRATEGIC: one committed contract per discriminating experiment.** Name the pinned external mechanism and its difference from a closed intervention. Record Samuel's three short predictions before results. Allocate through the ledger, never literal seeds. Freeze counted costs and truth precision. Set gates on the isolated paired arithmetic score difference. This ordering is a proposal, not an approved gate or block spend.

| Order | Intervention | Control and attribution |
|---:|---|---|
| 1 | Coherent K4 regeneration, fixed structural schedule first | Trusted SIMPLE plus explicit complete closure; fresh design fits only; consistent uses and births |
| 2 | Thin residual rank | Dense residual with identical term set; omitted feedback/data-selected pruning are separate explicit approximations |
| 3 | One common old-source basis | Same chain/all ages; exact-old-source control and matched random basis; measure score, not captured energy |
| 4 | Nested older tier, if first tier earns it | Frozen first-tier baseline; count joins/reconstruction |
| 5 | Frozen combined candidate | Untouched DECIDE block after design selection; arithmetic gate and Skeptic; read-only public CONFIRM afterward |

Separate design selection from final decision networks. Never fit coefficients/ranks/thresholds from public feedback. Disclose exposure to published structural settings. Full AUGMENT remains a scoped accuracy reference, not an assumed deployable candidate.

## Stage 3 — export and certify exact bytes

**Eligibility checkpoint:** before treating a reconstruction as a submission candidate, resolve the participant-derivation restriction above. If organizer clarification is needed, prepare a neutral question for Samuel to send/authorize; no agent posts it without his explicit authorization. Keep the research reference distinct from the candidate until provenance and eligibility are established.

Export to immutable legal `fnp` operations with deterministic seed derivation and shape-generic fallbacks. Compare reference and exact candidate per layer/consumed slice before treating mirror accuracy as candidate evidence. Batch related operations; avoid frequent scalar synchronization. Strassen depth is a runtime tradeoff, not a monotonic improvement.

Investigate exact participant API semantics before any buffer-pool proposal; native `out=` support does not authorize it under the current contract. Keep a functional implementation unless the contract is explicitly resolved. Do not inherit another team's fitting policy or local-runner advice.

Candidate evidence: contract validation; candidate-specific invariants at a frozen current error scale; counted FLOPs and applicable server offset; cold/repeated-worker determinism; residual forecast with its evidence domain; local subprocess result separately labelled; observed server wall when available; configuration-bound Linux memory enforcement. Preflight on `estimator.py` is additional infrastructure evidence, not certification of a different file.

Then clean official mini with zero failures, independent Skeptic and Samuel's approval request on exact bytes. No agent designates. Never queue submission or vault evaluation. Preserve the robust L2 fallback throughout.

## Stage 4 — one bounded frontier extension

Choose from the compact baseline's measured bottleneck. A paper-supported hypothesis is not a measured improvement.

| Priority | Hypothesis | First discriminator |
|---:|---|---|
| A | Sketch quadratic old-source products before reconstruction | Rolling transport/join proof, protected D3/D21, full supported-primitive cost and signed-tensor variance; reject dense replay or replica costs erasing savings |
| B | Output-sensitive shared-basis selection | Equal-cost perturbation comparison showing lower necessary rank; count sensitivity construction; reversing unchanged contractions does not qualify |
| C | Stochastic residual tail | Conditional expectation proof for frozen coefficients, nonlinear bias/variance accounting and priced independent replicas |
| D | Relational in-flight K4/birth correction | Reachable-state oracle and gate-strength controls before training; count every online feature/channel |

[Literature and falsifiers](research/2026-09-10_literature_review.md). A and B deserve bounded feasibility work first; C/D are secondary. No automatic sweep, repeat of failed scalar readout fitting, or reopening of primary sampling. Novelty claims require tested evidence and prior-art review.

## Calendar and continuity

Planning estimate from September 10: first 2–3 working days for instrument fixes/algebra; roughly the following week for compact-baseline reconstruction; then one frontier extension. These estimates do not commit data or queue capacity. Aim to freeze candidate bytes with time for fresh verification/runtime reserve before **October 17, 23:59 UTC**. Registration/team freeze October 2; write-up October 24. Nomination deadline/interface remains unpublished.

Each slice ends with LOG contract/evidence paths, knowledge delta and independent Skeptic. Update concise CURRENT_STATE rather than add another competing plan. Credit imported identities, scope negative results, and keep historical gates intact. If evidence is exhausted, write a new self-contained Strategist Packet; the old future-named September 11 packet is no longer the active handoff.
