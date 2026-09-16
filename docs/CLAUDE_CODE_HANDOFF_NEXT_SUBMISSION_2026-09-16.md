# Claude Code handoff — implement the audit through the next submission

Work in `C:\Users\samsu\Documents\arc-whest` as the Builder. Implement the September 16 audit recommendations, resolve their empirical questions, and carry the strongest supported candidate through a complete submission-ready evidence package. After Samuel approves that exact submission, submit it, inspect grading, and record the result. Do not stop at another plan or a list of suggested fixes. Do not claim that research must succeed, lower gates after seeing results, or submit a weak candidate merely to finish the task.

## Read and establish the current state

Read `AGENTS.md`, `docs/CURRENT_STATE.md`, and `experiments/LOG.md` from the bottom first. Then read:

- `docs/research/2026-09-16_strategy_and_implementation_plan.md` and its linked architecture, validation, literature, competition, and independent Skeptic reports.
- `docs/CLAUDE_CODE_HANDOFF_2026-09-16.md`, `docs/STRATEGIST_RESPONSE_2026-09-16.md`, and the September 14 handoff's authorized process changes.
- `docs/VALIDATION.md`, `docs/GOTCHAS.md`, `docs/PROVENANCE_504aldo.md`, and `docs/SKEPTIC.md` before the corresponding work.

Audit commit: `e772fac`. Check what has changed since it. Preserve unrelated working-tree changes, including `_snapshot_for_review.tgz`. Treat newer recorded evidence as authoritative over this dated snapshot. Reuse finished work instead of rerunning it.

At handoff, v31 `estimator_k3_v31.py` is the latest graded candidate, submission #331317, SHA256 `6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c`. It scored 8.7520303e-9 adjusted, 2.9074e-8 raw, at 0.3010253 budget, with 0/100 failures; worst graded wall/residual were 68.13 s/0.140244 s. It is not the old root `estimator.py`. F21 and v31 remain frozen controls. P11 DESIGN truth already exists: 24 networks, two independent 2^24 replicas, recorded noise 4.76e-9. Do not remint it. Verify identities through the registry and receipts, not literal seed strings.

The September 16 original handoff contains completed and obsolete instructions: v31 is already graded, P11 is already minted, the organizer question was withdrawn, ×1.60 wall projection is disproved, and the {0,2^24} FLOP offset band is stale. Do not repeat those actions. Never infer permission to designate from a clean graded row. No agent designates.

## Working agreement and scope

Treat this handoff as the requested implementation sequence for the audit refinements. Amend the execution contract before new measurements, preserving binding validation requirements and explicitly recording the changed D2/D3 discriminator and N1 economics. Do not quietly rewrite historical contracts or retroactively register experiments.

Consolidate already-authorized process simplifications into one clear active pointer. Do not invent approval requirements or silently weaken existing ones. Obtain and record Samuel's required predictions before dependent STRATEGIC measurements; ask concise bundled questions early, explain that AGENTS Rule 1 requires them, and continue independent authorized infrastructure/algebra work while awaiting answers. Never record the strategist's prediction as Samuel's answer or assume silence means “I don't know.”

Use bounded subagents for independent tasks when useful. Keep the Skeptic in a fresh context with the contract, diff, protocol and raw artifact paths, without Builder reasoning. Apply the binding review policy; do not replace an independent review with self-review. A failed hypothesis ends that branch, not the whole task. Synthesize each strategic result before implementing the next dependent hypothesis.

Develop new candidates under new names; never edit frozen archives or overwrite the standing submitted file. Preserve shape generality, deterministic provided-seed randomness, immutable grader inputs, setup lifecycle, allowed imports and single-array limits. Reference code under `external/504aldo/` remains a local diagnostic oracle only. Derive our implementation independently; package none of its bytes. Keep provenance current. Do not edit HUMAN_MODEL, touch the vault, contact organizers, or publish claims. Do not re-raise the withdrawn rules question.

Before every heavy workload or job expected to exceed 30 minutes, record the required optimization review: reuse, dataset duplication, workers, caches, batches/prefetch, unnecessary passes, recovery and device placement. Submit through workerq with conservative RAM/VRAM/CPU declarations; prefer a defensible 16–24 GiB RAM envelope. Never bypass the queue. Continue independent work while jobs run. Tiny CPU-only fixtures may run directly.

## 0. Repair the defects blocking reliable iteration

Timebox the first repair pass to roughly half a day. Implement and test the following using existing tools and `evidence_journal.py`; avoid building a new experiment platform:

1. Route local accuracy smoke/mini through the corrected wrapper with the local residual cap disabled. Preserve real server caps as deployment requirements. Fix affected certification/soak/preflight call sites so an invalid local cap cannot zero outputs before an “advisory” check runs. Other failures must still fail.
2. Stream and retain raw stdout/stderr from launch, including nonzero exits and malformed/truncated output; checkpoint completed rows. Never destroy the only crash evidence.
3. Bind reports and reusable checks to source, instrument and package hashes, dataset/truth identity, seeds/setup, numerical configuration, weight mode and meter path. Replace Git-status-string identity checks with dependency-content identity. Unrelated edits must not invalidate good evidence; already-dirty dependency edits must invalidate it.
4. Repair projection provenance and refusal behavior: reject mismatched control/candidate receipts, missing scored rows and failed anchors instead of silently calculating. Separate the arithmetic paired interval from uncertain transfer to hidden networks. Do not label an uncalibrated band as a 95% hidden-score prediction interval.
5. Make FLOP accounting runner-specific. Stored subprocess runs were 2^25 below graded counts; grader-twin counts already matched. Verify fixtures and never add a universal offset. Replace the disproved timing rule with clearly labelled empirical forecasts and limitations; 1.9 × standalone median is also family-specific, not a safety proof. Report per-network maxima and synchronizations, not just average operation counts.

Use real F21/v31 report fixtures plus negative cases. Run meaningful affected tests and required preflight/review for instrument changes. Keep required full checks off the inner edit loop by reusing valid, explicitly bound evidence. Finish remaining receipt/deployment-version work before release. If a wider refactor threatens the timebox, isolate the immediate fix and record the bounded follow-up.

## 1. Resolve the full fourth-order-to-third-order birth discrepancy

Start with the cheapest discriminator: a tiny float64 full-tensor oracle requiring no labels or decision block. Hold incoming state and coefficients equal. Include nonzero off-diagonal fourth-order structure. Compare repeated-index and all-distinct entries separately, then apply one arbitrary dense weight transport and compare the consumed D3/D21 slices.

v31's fourth-order-dependent repeated-support completion does not by itself establish the all-distinct feed in the public reference. The reference subtracts the full hub's repeated slices before constructing its residual. Derive the corresponding identity and compensation; do not naively add the full hub and double-count. A repeated-slice-only birth test is insufficient. The existing dG transport already has subtraction and addition; inspect actual algebra instead of assuming a missing subtraction.

Use this to refine D2/D3. Compare identical incoming states before interpreting whole-trajectory divergence. Retain D1's diagnostic-only lambda-profile comparison where it distinguishes a remaining uncertainty, clearly separating non-submittable reference constants from our own candidate. Zeroing only repeated-slice feed does not prove full-feed completeness.

If the oracle identifies a discrepancy, implement the independently derived correction in a new candidate; test tiny parity and a target-shape operation ledger before paired P11 scoring. If the oracle establishes equivalence, retract this diagnosis. If the correction does not improve score, do not promote it on mathematical plausibility alone.

Preserve the existing slice target: at least 1.5x raw improvement versus frozen v31, counted cost within 0.5%, and the required arithmetic paired interval/gate. Record outcomes even below target. Do not fit on public data. Any fitted lambda values need the separately required fresh DECIDE evidence after freezing.

## 2. Test thin feedback and exact cost reductions separately

After synthesizing stage 1, test the planned thin D21 feedback into births with an equivariant construction and a tiny recurrence oracle. Existing P3 target: at least 1.06x raw improvement for at most 1% additional counted cost. Keep an attributable control and test interactions with stage 1; do not multiply isolated gains into a forecast.

Then implement P4's exact savings incrementally: final-layer variance-only readout, safe pooled scratch reuse, supported deeper Strassen, and block-symmetric covariance. Preserve instrumented arithmetic and allocation lifetimes. Do not assume deeper recursion is faster. Existing target: <=0.27 budget at specified raw parity, <=20,000 operations, Linux memory pass, and no unsupported routing warnings. Check synchronizations, wall forecast and memory separately. Retain only supported gains.

## 3. Make the old-history compression experiment economically meaningful

Before a large truth/rank sweep, spend at most roughly two days on two prerequisites:

- Define exactly what state perturbation and output error the compression objective measures. Test the cheap mean-path sensitivity against the full remaining moment recurrence, including covariance and future births. Use signed perturbations and a forward-energy control. Define a valid positive-semidefinite objective/generalized eigenproblem; a nonsymmetric product of Gram matrices is not ordinary PCA.
- Build and meter a complete old-source join, basis update, transport and D3/D21 query cycle. Count every online selector, QR/sketch, rotation, reconstruction and temporary. Generic dense core rotation is O(r^4); readout is O(n*r^3+n^2*r). A rank-halving result alone does not establish savings.

Proceed to registered P11 rank/accuracy exploration only if the proxy and full operation graph remain plausible. Include economically interesting ranks around 128–150 alongside the existing grid. Initial end-to-end target: <=0.20 budget with <=5% raw loss relative to its frozen control. This is an intermediate improvement, not a winning score guarantee. Adaptive rank must remain safe on worst-case per-network cost, residual and memory.

If the proxy fails, reject that proxy, not every output-aware method. If dense merging is too expensive, retain a bounded alternative: shared old-source factors without dense merging, or implicit summation/quadratic queries informed by the literature review. Require a complete join/change-of-basis/query prototype before spending truth work. Do not resurrect closed source-deletion or memoryless slice-only methods without the required new evidence. Do not start an open-ended automatic optimization loop.

## 4. Freeze the best supported bundle and prepare the next submission

The desired major next release is approximately 4.0–4.4e-9 adjusted or better, with a longer-term research objective around 2.0–2.2e-9. These are dated planning targets, not promised gains or replacements for preregistered statistical gates. Refresh the board when assessing readiness. Do not wait for every speculative branch if a sufficiently strong, fully supported bundle is ready. Do not request a canary after every small improvement; justify any earlier canary by a concrete runtime uncertainty it resolves.

For every decision, use paired common-network arithmetic score differences via `competition_score.decide` with an explicit preregistered gate on the isolated change, and studentised-t or basic bootstrap. Report truth-noise floors and the appropriate unequal-cost noise correction. Score adaptive work by averaging per-network error-times-cost, never by multiplying average error and average cost. DESIGN reuse is not fresh confirmation. Use public data only for confirmation. Fresh DECIDE is single-use and required where the operative policy requires it, including fitted constants; a crash does not make a spent decision block fresh.

Freeze exact candidate bytes before final evidence. Complete `whest validate`, required affected checks, generic-shape and deterministic setup/predict tests, official metering, a clean full mini through the corrected wrapper, and real Linux participant memory/lifecycle checks. Match the current server package/Python/backend versions as closely as available and report any remaining mismatch. Keep local accuracy, distributional evidence, enforced deployment checks and server timing forecasts as separate verdicts. Do not call local accuracy with a disabled residual cap a server-cap pass.

Reuse valid checks by complete identity, not filename. Produce one release manifest that includes the exact candidate/package SHA, controls, experiment contracts, raw rows, noise estimates, paired results, counted cost, worst operation/synchronization behavior, residual headroom, wall uncertainty, enforced memory result and independent review where required. Resolve material review findings before requesting upload approval. Preserve a safe fallback; do not redesign the whole repository to create this manifest.

## Approval, grading and completion

Samuel's per-submission approval is required by AGENTS Rule 6. “Implement through the next submission” is not blanket approval to upload unreviewed future bytes. Ask only when the package is concrete and reviewable: show exact file/SHA, measured gain, projected score with honest uncertainty, failures, deployment evidence and remaining risks. Explain the rule requiring that final confirmation. Never treat elapsed time or silence as consent.

After approval for that exact package, run the authorized submission from PowerShell, record its ID and hash, inspect grading, and compare actual score/cost/timing with the forecast. A rejected or failed upload does not authorize repeated new submissions automatically. No agent designates. Append the result and knowledge delta to LOG and `experiments/submissions.jsonl`; update CURRENT_STATE and the active-plan pointer so another Builder can resume without chat history.

Keep progress updates concise and commit coherent changes with evidence paths. When a branch fails, record the falsifier and continue the next justified branch. If bounded research is exhausted before the major-release target, produce a Strategist Packet with the best supported candidate, complete negative results, unresolved bottleneck and exact next discriminating question; do not fabricate success or lower the target after seeing results.

The deliverable is implemented and tested improvements plus a complete, reviewable next-submission package; after explicit approval, a recorded graded result. Start the bounded infrastructure/algebra work now, asking only for genuinely missing predictions or other required inputs while continuing independent work.
