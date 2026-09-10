# Independent mathematical/protocol Skeptic — worker review

Reviewed source through `bcca354`, including Stage-1 commit `0104c25`, Stage-2 contract `08f787b`, and implementation/evidence commit `2b048eb`. This context did not inherit the Builder's reasoning. Sources inspected: AGENTS, SKEPTIC, VALIDATION, GOTCHAS, CURRENT_STATE, LOG contracts/evidence, local reference/generator/generated arms/lab, existing Stage-1 and Stage-2 artifacts, seed/read ledgers, commit timestamps, and the pinned public generators. No fresh truth labels, accuracy reruns, heavy jobs, or protected writes were performed. Read-only AST parsing and scalar/table extraction were the new local checks. Parent subsequently authorized this review-record file only.

## Skeptic verdict: REJECT

**Strongest challenge**

Stage 2 implements a different intervention from its registered coherent fourth-order core. Stage 1 proves useful matrix identities, but its cost screen prices a different basis-maintenance algorithm from the cited public implementation. The pickup instructions at `bcca354` are not valid or executable as written.

**Evidence for it**

### 1. The fourth-order state is not derived consistently

`local_sim/p2_build_k4_arms.py:218` obtains the new vector by applying the existing scalar projection coefficient to row sums and centering. The cited matrix-core projection is a different harmonic projection. A read-only AST/table check found both local Laplacian coefficients equal 24: at width 1024, the local centered-row coefficient is `2.8554459e-6`; the rank-one matrix-core centered-row coefficient is `6/(n+4) = 0.0058365759`, approximately 2,044 times larger. Fitting theta cannot establish the missing derivation; this ratio is not a prescription to rescale theta.

The local deviation enters D4 without the matrix-core metric conversion; D22 uses fitted lambda as its consumer coefficient. The cited derivation distinguishes matrix-core transport, metric-scaled D4/D22 consumers, an additional D31 consumer, and explicit K4-to-K3 feed factors. Those extra consumer/feed families are absent from the generator. Existing K3 outputs can change indirectly, but that does not prove the intended closure was implemented. See pinned [V17 generator](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v17.py), lines 101–103, 132–142, 178–239, and 263–284.

Correction: define the matrix core and its tensor interpretation explicitly; derive its projection, consumers, and induced births together. Check resulting slices against an independently derived dense, small-dimensional reference before fitting. P3 is a valid matrix identity; this review does not claim it must generically be replaced by fourth powers of weights.

### 2. Registered mechanism controls are missing; the substitute cannot import

LOG lines 40600–40617 register exact, coherent, consumers-only, and births-only arms. The generator's ARMS list at line 288 emits exact, coherent, and no-carry. No-carry clears the fourth-order vector after the layer; it does not remove induced K3 births. It cannot test predictions 2 or 3.

Read-only `ast.parse` fails on `local_sim/p2_arms/arm_k3_k4nobirth.py:797`: an inline comment consumes the closing parenthesis. Exact and coherent arms parse successfully. The no-carry arm still executes the ordinary `_append` at line 774 before clearing its vector.

Correction: implement the registered independent switches and four arms, or explicitly replace the experiment with a new contract. Parse every emitted arm. Make parity fail when `cost_same` is false; currently the test only prints that discrepancy.

### 3. P01's freeze condition has already been violated

The old CURRENT_STATE pickup says to freeze before P01 is read, while immediately using the P01 smoke to constrain theta. `experiments/truth_reads.jsonl` records three completed P01 reads. Supplied read timestamps are `18:00Z` and `18:05Z`, while `08f787b` is dated `19:47:03Z`. These manually supplied timestamps may be inaccurate; they cannot establish contract-before-measurement chronology.

P01 is reusable DESIGN data: this is not a spent-block violation. It violates the experiment's stronger fit-before-evaluation promise.

Correction: preserve and disclose smoke exposure, reconcile timestamps from independent job records if available, and stop describing P01 as unread. Define subsequent evaluation roles honestly before more labels. Samuel's recorded predictions concern the registered coherent-core experiment, not the substituted intervention.

### 4. The lab cannot save evidence or execute the old pickup reliably

`local_sim/p2_lab.py:246` changes the loaded archive to a dictionary. `_write:529` still uses `z.files`, causing AttributeError. After that repair, line 551 accesses `a`, which is local to `main`, causing NameError.

Pickup instructions omit `--arms`; the default list excludes `k3_exact`, so `results[a.control]` eventually fails after running unrelated arms. Generator documentation mentions `--fit`, but its parser provides no fitting command.

Correction: test report writing with fabricated rows and a plain dictionary, pass configuration explicitly, and reject absent controls/missing arms/existing output paths before reading truth. Provide a complete validated command only after an actual fitting interface and intended arm set exist.

### 5. Promised noise-aware comparisons and all gate outcomes are not implemented

The contract promises `noise_corrected_compare`, but `p2_lab.py` never calls it. The gate uses observed scores; comments and saved `noise_correction_identity` claim universal cancellation. Unequal multipliers prevent that cancellation. The ordinary gate function implements the registered point-gain-plus-positive-interval build rule; it does not implement the separately stated kill rule or half-arm mechanism conditions.

Correction: report observed, corrected, and official-precision comparisons distinctly and specify which enters the design gate. Preserve identities through every comparison. Implement and label build, kill, and mechanism outcomes separately. Failed build is not automatically a kill. The root's independent integration review additionally identifies the per-replica versus averaged-replica noise-floor mismatch; production wiring must address that too.

### 6. Stage-1 price conclusions do not follow for the cited algorithm

`local_sim/p2_compact_reference.py:477` sets new-column count to width and prices an `(r+n)` eigendecomposition: a 1408-dimensional core at rank 384. The pinned public implementation builds a weighted Gram sketch, applies thin QR, rotates old factors, and updates a carried Gram core; it does not require this eigendecomposition. It shares two dense legs rather than the local reference's three. See [V21 generator](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v21.py), lines 89–111.

The local `basis_join` uses dense SVD, so the ledger does not price its own executable reference join either. The baseline counts `l+1` generations at every layer and includes D21 at the final layer, unlike the actual chain's initial/final shortcuts. Its 1.1928 calibration discrepancy is unsurprising. Neither scalar calibration nor asymmetric omissions establish an upper bound on achievable savings.

Correction: retain the ledger as a labeled hypothetical algorithm estimate; retract its published-configuration/deployability inference. Price the actual join/update/readout algorithm, every relevant retained factor, and actual first/final-layer schedules before drawing a representation-cost conclusion.

**Evidence for the original claim**

P1's shared-basis transport and P2's reassociation are valid for the stated representation. P3 is a valid matrix-core diagonal identity; the defect is the unsupported transition from that identity to the implemented closure. Stored 28/28 algebra checks support these bounded claims. Stored three-seed `96x5` parity supports exact-control output and cost equality at that shape. The smoke clearly shows the placeholder worsens accuracy, with a noise floor reported. No promotion, new DECIDE block, vault touch, or submission is established by this slice.

**Unresolved ambiguity**

Existing artifacts cannot establish whether a correctly derived coherent core improves fresh-network performance, whether the true compact program is cheaper, or whether manually supplied timestamps reflect actual execution. The exact `145,977,344` live-shape added-FLOP claim is not recoverable from the rounded smoke log alone.

**Cheapest discriminating follow-up**

One label-free, tiny dense-core conformance experiment: inject a nonconstant signed matrix core into a small layer; compare independently derived dense projection, transport, D4/D22/D31 consumers, and K3 feed slices against each of four emitted arms. Include zero-core and separate consumer/feed switches. No seed block or truth labels are needed. This precedes fitting. Repair writer/configuration validation with synthetic rows; run required preflight before resuming label-consuming research.

**Checklist items triggered**

1, 4, 6, 9, 10, 13, 14, 17, 18.

## Documentation-only follow-up verdict: ACCEPT within mathematical/protocol scope

Read the revised `docs/WORKER_REVIEW_2026-09-10.md`, `docs/CURRENT_STATE.md`, `docs/GAME_PLAN.md`, and `docs/CODEX_EXECUTION_HANDOFF.md`. They accurately scope the REJECT above: repair-before-fit, independent core conformance, four actual mechanism arms, executable report paths, exposed P01 disclosure, transparent contract amendment, and no conversion of a failed build into a scientific kill. They withdraw the unsupported cost/unsupported-primitive/upper-bound conclusions without closing compact all-history propagation. They preserve the standing candidate and Samuel's Stage-3 eligibility hold, including no organizer question drafted/sent. This is acceptance of the corrected plan/documentation, not acceptance of unimplemented repairs or an accuracy/novelty claim. Independent instrument findings and queue/competition observations belong to their respective reviewers; they were not freshly revalidated in this documentation pass.

One factual scope sentence needs correction: WORKER_REVIEW's opening says no estimator/instrument/etc. was “changed/read.” Source files and existing result artifacts were in fact read. Say instead: no estimator/instrument/gate/registry/protected corpus was changed; no protected truth/vault labels were read; source, logs and existing result artifacts were inspected. This correction was sent to the parent; no other document was edited by this reviewer.

No material mathematical/protocol misstatement was found in the revised continuation order. The documentation acceptance does not erase the original implementation REJECT.
