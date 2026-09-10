# CURRENT_STATE.md — read this first

**Current September 10, 2026, after independent review of worker HEAD `bcca354`.** Then read `experiments/LOG.md` from the bottom. The preceding snapshot is [archived verbatim](archive/CURRENT_STATE_before_worker_review_2026-09-10.md).

## Pick up here

**Do not fit the current K4 arms or execute the old P01 command.** The instrument and mathematical Skeptics both returned **REJECT**. Their review is completed, not owed; acceptance of the repairs is now owed. [Findings and reproducer](WORKER_REVIEW_2026-09-10.md).

1. Fix the real certification/reporting/data-access paths with synthetic fixtures. Preserve the worker's valid repairs. Distinguish missing evidence from applicable explicit failures; bind actual package/instrument versions; repair report domains, memory controls, lab writing, noise accounting and protected loaders.
2. Derive and independently check the coherent K4 core and all its consumers/births. Emit the registered four arms, parse them, and test zero-core/control parity. The current implementation is a different intervention and one arm has a syntax error.
3. Replace the compact cost screen with a ledger of the algorithm actually implemented: weighted Gram sketch, thin QR, factor rotations and carried Gram core, with two shared legs and actual initial/final schedules. The previous cost/deployability conclusion is unsupported.
4. Obtain required preflight and fresh-context Skeptic acceptance before resuming label-consuming research. Reconcile exposed data/chronology and amend the Stage-2 contract transparently; P01 is already inspected DESIGN data. Fit only after core conformance and an actual fitting/report interface exist.

The sole implementation sequence is [GAME_PLAN](GAME_PLAN.md). Copy-ready continuation: [CODEX_EXECUTION_HANDOFF](CODEX_EXECUTION_HANDOFF.md). Historical LOG pickup instructions are superseded by this review; historical evidence, gates and seed restrictions are not erased.

## Execution in flight

- **workerq 574:** default preflight, 8 GiB RAM / 0 VRAM / 2 CPU / no GPU; queued at 20:12 UTC, still QUEUED at 20:18 UTC. It is not Full, and uses no source snapshot. Inspect before duplicating; record actual source at execution or replace this owned queued job with a frozen-source run. Output target: experiments/2026-09-10_worker_review/preflight.log.
- The worker's full test_certify_candidate.py mutation process was present earlier in review and was no longer present by 20:30 UTC; no completion artifact or exit status was captured. Locate its result; do not equate process exit or the 26 unit checks with full-battery success.
- Earlier default preflight 539 preceded these repairs. There is **no verified post-worker preflight result** in this snapshot. Queue contention is not permission to launch directly.

## Standing decisions and candidates

**Samuel: rules §6 eligibility work is "hold off entirely for now."** Stage 3 is closed by his decision. Do not draft/send a new organizer question, or advance a participant-derived research reconstruction toward submission. Older clarification drafts are historical/inactive. Attribution or a fresh mathematical rewrite alone does not establish eligibility.

| Role | Current evidence |
|---|---|
| Shipped baseline | estimator.py v2.6.1, archive identity unchanged. Default preflight targets this file, not every nominee. |
| Standing candidate | estimator_k3_strassen_L2.py, graded #330566; SHA256 390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db; untouched. |
| Best public result | Strassen L4 #330290, adjusted 1.836e-8; worst displayed wall 114.6798 s. |
| More robust fallback | L2 #330566, adjusted 2.284e-8; worst displayed wall 33.0009 s; 0/50 public failures. |

Public score noise floors are unavailable in the inspected reports. L2/L4 wall measurements are dated server observations, not guarantees. Two historical certification gate failures remain failures; grading did not amend their contracts. No promotion, submission or designation in this review.

## Architecture and competitive position

**A winning architecture has not been demonstrated.** Retain weight-conditioned means/full covariance, persistent K3 history, a consistently derived cheap K4 core, common coordinates retaining all older sources, accurate recent sources and a final mean-only path. Separate exact algebra, approximations and metered implementation. The expensive SIMPLE tower is a reference and fallback, not an established optimum.

504aldo's pinned public implementation supplies concrete external evidence for mechanisms our failed deletions and generation caps did not test. Its reported adjusted 5.40e-9 is still above the leader; its constants and participant-derived code are not automatically usable candidates. [Literature and code review](research/2026-09-10_literature_review.md).

Chrome recheck September 10: our best still about 1.84e-8, rank 44; leader about 2.70e-9. Earlier exact leader report #330500 was 2.683e-9, a descriptive 6.84x gap from our best. Official reference-noise floors are unavailable. No verified leader method was discovered. [Competition sources](research/2026-09-10_competition_sources.md).

## Worker progress: accepted scope and open defects

| Stage | Honest status |
|---|---|
| 0: instrument | **Partial.** Identity pairing, ordinary-alpha correction, atomic claims, new loader reservation and canonical vault lifecycle survive bounded checks. Real domains/version extraction, legacy negatives, memory retention, lab/report noise and bypassing consumers remain defective. |
| 1: representation | **Identities supported; cost conclusion withdrawn.** Stored 28/28 tiny checks do not prove the complete closure. Ledger prices a different join from both executable and cited program. |
| 2: coherent K4 | **Not tested as registered.** Wrong projection/consumers/feed; missing mechanism arms; syntax and writer failures. Smoke is evidence about the placeholder only. Do not tune theta to conceal a derivation mismatch. |
| Data provenance | P01 has three recorded DESIGN reads; cannot claim unread evaluation. Manual read timestamps do not establish contract-before-data. P08 stays spent/exhausted. Preserve records and disclose uncertainty. |

Independent review reran memory 25/25, truth reads 27/27, scorer 129/129, certifier units 26/26 and vault **sacrificial subset 30/30**. These passing suites coexist with reproduced failures on untested production paths. Full worker battery/preflight remain unverified. [Detailed review](WORKER_REVIEW_2026-09-10.md).

## Competition contract and measurement

Phase 2: 1024 x 16, He-Gaussian bias-free ReLU, Gaussian input. Budget 2^41, lambda 0. Score averages per-network final MSE times max(0.1,F/B); failed predictions receive zero-output error without discount. WhestBench 0.16.1 / flopscope 0.12.1. Published caps: 400 ms residual, 120 s wall, 5 s import/setup; 8 GB participant memory, installed enforcement 8192 MiB. Windows accounting does not prove Linux enforcement. Local timing does not prove server timing.

Public mini 100/full 1000 with 1e9-sample truth differs from 50 public/50 sealed leaderboard networks and the fresh prize rerun. Rules v14 §7 ranks the team's best eligible submission; §6 allows up to two and selects two best valid public submissions by default. Only Samuel designates. Every submission needs his specific approval. Registration/team freeze October 2; submissions October 17, 23:59 UTC; write-up October 24. Selection interface/deadline remains unpublished. [Dated official evidence](research/2026-09-10_competition_sources.md).

VALIDATION.md remains binding: paired arithmetic score difference, common random numbers, fresh preregistered DECIDE blocks, public CONFIRM read-only, vault touch restrictions. Report observed score, noise-corrected estimate and official-precision forecast separately; common-label noise does not generally cancel with unequal multipliers. The average of two independent truth replicas has half a single replica's noise floor.

## Continuity and closed lanes

Packing, partial AUGMENT deletions, scalar final-readout fitting and three-generation caps retain their scoped negatives. They do not close coherent K4 or all-age shared coordinates. The whole score-lane closure was retracted; the broken adjoint kill remains retracted. In-flight sensitivity was not implemented/tested. [Architecture audit](research/2026-09-10_architecture_audit.md).

AGENTS and VALIDATION govern classification, written predictions, committed strategic contracts, human predictions and independent Skeptics. HUMAN_MODEL is human-owned. Frozen archives and the Phase-1 vault remain untouched. Heavy workloads use workerq with declared resources. No autonomous optimization loop or new framework is needed. Contribution write-up stays stale/do-not-submit pending claim review.

Older plans/handoffs and the future-named September 11 packet are historical. The [earlier research synthesis](PHASE2_RESEARCH_AUDIT_2026-09-10.md) remains background; this worker review supersedes its execution status, and Samuel's hold supersedes its eligibility-question task.
