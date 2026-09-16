# Validation and iteration audit — 2026-09-16

Read-only audit of the current source, existing evidence, and two lightweight scorer tests. No candidate or instrument was edited; no heavy workload, protected truth read, seed-block claim, vault touch, submission, or browser operation was performed by this auditor. This report is an implementation proposal, not an instrument certificate or promotion decision.

## Assessment

The core measurement architecture is sound enough to guide the next improvement: canonical minting, real flopscope-backed immutable weights, setup-aware lifecycle, paired arithmetic scoring, independent truth replicas, and a Linux address-space gate. The latest v31 result provides unusually useful end-to-end evidence: a public-score forecast of `8.7133e-9` became `8.7520e-9`, a 0.45% error. That is one successful forecast in this candidate family, not a universal accuracy guarantee.

The expensive friction is largely in orchestration and stale interfaces, not a need to replace this mathematical infrastructure. Default smoke execution still arms the known-invalid local residual cap; projection code retains a disproved timing model; reports lack a unified identity-bound evidence manifest; certification both over-invalidates unrelated edits and under-detects changed already-dirty instruments. Fix those interfaces, cache valid evidence by exact dependencies, and run release checks once per frozen candidate.

## What was independently checked

The installed lightweight tests passed: `local_sim/test_competition_score.py`, **134/134**, and `local_sim/test_predict_public.py`, **39/39**. The former checks scalar arithmetic against installed WhestBench 0.16.1/flopscope 0.12.1, randomized rows, tiny synthetic failure cases, graded-report replay, arithmetic versus geometric decisions, and noise corrections. The latter passes even while asserting the obsolete `+2**24` and `×1.60` models. Test green therefore establishes the implemented identities, not the truth of every empirical assumption.

Existing JSON was parsed without loading protected truth. Replaying the rounded public per-network rows with the canonical scorer yields:

| Existing evidence | Score / diagnostic | Interpretation |
|---|---:|---|
| F21 `#330950`, 50 visible score rows | `1.5957396e-8` | Recomputed rounded public rows; exact submission aggregate `1.5957295e-8` |
| v31 `#331317`, 50 visible score rows | `8.7519492e-9` | Recomputed rounded public rows; exact aggregate `8.7520303e-9` |
| v31 projection | `8.7132756e-9` | Predicted score; official label-noise floor not supplied by report |
| F21 local mini | 100 rows, zero failed, `928424083046` FLOPs each | Local CONFIRM run with residual cap disabled |
| v31 local mini, twice | 100 rows each, zero failed, `661928064562` FLOPs each | Two local CONFIRM reports, not fresh independent networks |
| v31 Linux gate | PASS, 23 checks; VmPeak `5698572288` bytes | Actual 8-GiB RLIMIT_AS, exact candidate hash, three repeated predicts |
| v31 certification | 60 checks, zero failed, one skipped, **INCOMPLETE** | Section 10 deliberately skipped; later mini/server evidence exists separately |

Sources: `experiments/run_reports/f21_diagnostic_330950_graded_report.json`; `experiments/run_reports/v31_canary_331317_331317_graded_report.json`; `experiments/2026-09-16_slice7_mini/{f21_mini,v31_mini,v31_mini_confirm,predict_public_v31}.json`; `experiments/2026-09-15_v31_certify/{certify_v31,linux_memory_gate_estimator_k3_v31}.json`. Exact current candidate hash independently recomputed as `6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c`, matching these certificates. A small difference between rounded-row replay and the official aggregate is not scorer disagreement.

These measured MSEs are report values against organizer labels; the organizer truth-noise floor is unavailable in these artifacts. Do not describe it as zero or infer it from the misleading historical `sampling_mse` field.

## Verified strengths worth retaining

1. **The scorer measures the competition objective.** `competition_score.py:134` onward implements compute, failure multiplier, and per-row score before averaging; `:1151` is the binding arithmetic decision. Installed `whestbench/scoring.py:709` starts the budget context and `:743` onward implements zero-prediction failures. The budget is also a hard execution ceiling, not simply a smooth price. The synthetic parity tests exercise this.
2. **Lifecycle and representation are explicit.** `lifecycle_eval.py:510` constructs flopscope-backed weights; `:507` makes `f64_readonly` default; `:561` evaluates one estimator with setup once, immutable evidence rows, and a per-row evidence sink. Bare numpy probes can bypass method billing and must not replace this path. Local wall/residual remain a different timing instrument from the server.
3. **Truth noise is handled in the correct units.** `p2_truth.py:164` creates independent replicas. `competition_score.py:1004` implements observed, corrected, and official-precision forecast comparisons. Common truth noise cancels in raw paired differences but **does not cancel when cost multipliers differ**: the bias is mean `(q_control-q_candidate)*truth_noise` (`:1077`). Negative corrected estimates are retained. This matters directly for compression improvements near the frontier.
4. **Evidence loss has an existing reusable solution.** `evidence_journal.py:1`, `:134`, `:210`, `:223`, and `:242` implement exclusive manifests, fsynced rows, identity checks, completeness, and atomic publication. `f21_score.py:295` onward uses it with dependency hashes and `lifecycle_eval`'s sink. Extend this existing mechanism; do not invent another bespoke JSON writer.
5. **The memory gate is a real gate with controls.** The v31 report records an enforced 8-GiB cap, over-limit rejection, positive no-cap allocation, retained-state control, three repeat predicts, exact candidate and instrument hashes. `linux_memory_gate.py:517` onward records the conditions; `certify_candidate.py:930` validates consumption. This is stronger than Windows RSS accounting.
6. **The single-network diagnostic already exists.** `p2_repro_mlp.py:94` onward can reproduce named failures with organizer MLP construction and setup; it is not a substitute for a score or deployment certificate. Use the names from a retained failed report before replaying 100 networks.

## Verified defects and bounded repairs

### P0 — removing a residual comparison did not remove the residual failure path

`certify_candidate.py:2647` constructs its official two-network smoke without `--no-residual-wall-time-limit`. `:2107` still requires zero failed rows. The CLI therefore zeroes predictions before the subsequently advisory local-seconds text can help. The optional soak likewise retains the cap (`:2501`) and directly requires the worst pooled residual below the server cap (`:2225`). `scripts/preflight.ps1:544` has the same capped command, despite its later advisory wording.

The current shipped K2 file may still pass preflight; this is not evidence that a K3 candidate can pass the default candidate smoke. Today's v31 certificate uses `--skip-smoke`, so its literal INCOMPLETE status is accurate. A separate manifest should consume the valid later mini and graded evidence instead of leaving humans to infer completion.

**Repair:** one canonical local official-run wrapper for accuracy/functional smoke, explicitly disabling only the local residual cap and retaining failure categories. Gate residual forecasting separately on counted operations and scalar materializations; keep actual server residual as server evidence. Test command construction and a synthetic residual-only failure case. Do not relabel a cap-disabled run as actual server-cap enforcement. Reconsider the optional local residual soak's purpose before spending hundreds of runs on it.

### P0 — public-score projection still uses a disproved wall model

`predict_public.py:271` computes `local_worst * GRADED_WALL_CONTENTION_FACTOR` (`1.60`). The stored standalone v31 mini has local median `36.1839 s`, worst `39.5305 s`; the official report has worst `68.13 s`. Its current forecast is `63.2488 s`, below the observed server result. The contended local run happened to predict `83.2283 s` and passed by luck of host load. These are diagnostic comparisons, not quoted local absolute times as deployment authority.

**Repair:** version the model, preserve host/load metadata, and separate measured server timing from predicted timing. The current-state interim `1.9 × standalone median` would give about `68.75 s` here, but is supported by very few candidates and must remain advisory. A new arithmetic family, different Strassen depth, or changed allocation/synchronization behavior needs a matching canary or conservative envelope. Operation count and FLOPs cannot prove wall safety.

### P1 — cost offsets are path-specific, not an arbitrary server increment

`predict_public.py:86` and `:239` still return offsets `{0,2**24}`. Stored graded minus local `whest run` is **exactly `2**25` for both F21 and v31**. v31 Linux `f64_readonly` already bills `661961618994`, exactly the server. Adding an offset there would double-count it. The current effect is tiny, but stale logic near budget/floor boundaries is unnecessary risk.

**Repair:** attach a meter-path identifier and weight representation to every result; canonicalize subprocess reports to the checked grader representation only with candidate-family evidence. Prefer directly metered grader-twin FLOPs for the release manifest. Keep raw counts and their provenance; never silently rewrite historical artifacts. This changes an empirical model and should have artifact-based regression fixtures.

### P1 — git-status equality is not source immutability

`certify_candidate.py:1275` snapshots `git status --porcelain` and `:2285` compares it after the run. A previously dirty instrument can change content while its status remains ` M path`; a clean unrelated generator becoming dirty invalidates an otherwise useful run. Candidate and shipped file hashes are separately checked (`:2270`), but other relevant sources are not protected by status equality.

**Repair:** execute an immutable candidate/instrument snapshot; bind the report to actual dependency hashes before and after. Hash package versions, numerical configuration and selected tests too. Keep arbitrary working-tree edits outside that execution snapshot. Candidate-specific edge receipts already show the desired hash-bound pattern. This simultaneously strengthens integrity and eliminates the documented unrelated-edit hour loss. Do not merely remove the current check.

### P1 — the projector does not enforce enough report provenance

`predict_public.py:103` reads rows and budget, `:177` pairs names, and `:507` accepts arbitrary candidate/control/graded paths. It does not bind the local control to the submitted control source; its `--candidate` hash is read from the current file for memory lookup, not tied to the source that produced candidate-run JSON. Dataset hash, package/runtime equality and complete expected row count are not checked before projection. Missing-MSE rows are silently skipped (`:115`). A failed graded anchor is counted but not refused. Identity-aligned names are necessary, not sufficient provenance.

The introductory claim that it refuses a control replay that does not reproduce its graded score overstates the implementation. `graded_anchor()` replays the graded rows themselves; it does not verify the local control's bytes or calibrate local-to-graded agreement. Local and graded datasets differ, so equality of means would also be the wrong repair.

**Repair:** source-hash-bound run receipts, dataset fingerprint, names/counts, package/config/meter-path binding, successful anchor requirement, and explicit handling of absent labels. Reuse baseline rows only when all relevant identity keys match. Add negative fixtures for wrong control source, wrong dataset with colliding names, truncated reports and failed anchors.

### P2 — forecast intervals are not calibrated prediction intervals for the hidden board

`predict_public.py:196` computes a paired difference interval; `:199` rescales its endpoints by the observed control mean and then by the server anchor. That is a convenient conditional display of the difference uncertainty. It does not account for random denominator covariance, the different 50 hidden board networks, or model-transfer error. The synthetic test even insists on a nonzero interval for an exactly constant candidate/control ratio (`test_predict_public.py:125` onward), demonstrating that it is not a ratio-estimator confidence interval.

**Repair:** keep the paired arithmetic difference as the promotion statistic. Relabel the current band as a transformed local paired-difference interval, and use a separately preregistered, validated ratio/transfer forecast procedure if calibrated board prediction coverage is required. Do not change the binding gate into a ratio test. One 0.45% successful forecast cannot establish 95% coverage.

### P2 — crash-safe persistence is incomplete outside the F21 scorer

`scripts/run_mini_json.ps1:70` captures all stdout in memory; `:104` validates JSON before persisting it. It now correctly saves a valid complete report despite a nonzero exit code, but a malformed/truncated report still raises without writing the captured raw output. An interrupted process before final report emission also loses the in-memory report. `p2_truth.py:489` retains all minted networks and writes the final archive only after the loop (`:523`); it has no per-network durable checkpoint.

**Repair:** stream stdout/stderr to unique durable raw files from launch, then atomically publish validated JSON alongside failure metadata. Reuse the evidence journal for row-wise candidate evaluation. Truth generation can persist completed per-network replicas with producer/config/seed hashes and resume only identical unfinished generation; do not authorize a second protected scoring read. A crashed DECIDE read remains spent under the current policy.

## A faster execution architecture

The following is a proposal for one small orchestration layer over existing modules, not a new experiment platform.

| Lane | What executes | Evidence and reuse |
|---|---|---|
| Development | Syntax/import contract, changed algebra's tiny oracle parity, targeted invariant; one truth-free live-shape grader-twin meter when the operation graph changes | Fast defects before labels; no 100-network confirmation or complete certification per edit |
| DESIGN screen | Frozen candidate against fixed paired control on current authorized 24-network design corpus, with noise-aware arithmetic result, cost and op count | Persist predictions/rows once. Reuse the exact baseline when candidate-independent context, setup, data and instrument hashes match; timing comparison needs equivalent host conditions |
| Fitted-candidate DECIDE | One preregistered fresh held-out read, complete paired row journal, binding `competition_score.decide()` | No repeated holdout peeks or retrospective threshold selection; exact algebra transformations follow the separately authorized process below |
| Release | Frozen bytes: affected/full required preflight, candidate invariants, determinism, immutable weights, real live-shape count, Linux memory, one official mini with local residual cap disabled, projection receipt and packaging hash | Consume already-valid candidate-specific receipts; rerun only stale dependencies or unresolved failures. Existing full mini can supersede its subset smoke via explicit receipt validation |
| Server calibration | Individually approved material-change canary, preserve all rows and environment | Validate wall/residual and transfer model; do not require three replays of identical bytes for ordinary arithmetic development |

A cache key should include candidate SHA, evaluator and dependency SHAs, package/runtime fingerprint, dataset/truth hashes, network identities, seed/setup identity, weight mode, shape/budget/caps, and numerical policy. A receipt should state coverage, skipped checks, stale reasons, and evidence domain. Static instrument mutation tests can be reused until their dependency closure changes; candidate execution cannot be reused across changed candidate bytes without an explicitly proven equivalence policy.

Before implementation, reconcile the operating contract in one place. `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md:5` explicitly records Samuel authorizing §0: single-use DECIDE only for fitted constants; exact FLOP transformations use ledger-diff + tiny parity + one 24-network DESIGN run; daily truth cadence; no infrastructure CONTRACT/SKEPTIC blocks; Skeptic for write-up numbers and designated files. `docs/CLAUDE_CODE_HANDOFF_2026-09-16.md:14` repeats these. The supplied/current AGENTS contract still has broader requirements. This ambiguity itself creates delay. Do not silently choose a weaker policy or present already-authorized simplification as a new discovery; consolidate the current instruction hierarchy and make tools display the applicable lane.

Prioritize the first five defects above, then add receipt reuse. Avoid a general refactor of hundreds of historical probes. `scripts/preflight.ps1:3` describes itself as cheap, but its tests are for shipped `estimator.py`; v31's own certification says this explicitly at `certify_candidate.py:1287`. Spending time rerunning unchanged K2/carrier checks after every K3 experiment does not increase K3 confidence. An affected-path manifest can preserve required coverage without recomputing unchanged evidence.

The stored local mini durations illustrate the potential saving, not a server-time claim: F21 mini took 8,023 seconds; v31's runs took 5,932 and 4,958 seconds. Reusing a still-valid F21 baseline avoids another 2.23 hours of local work. Two identical v31 full minis were useful to investigate an actual unexplained failure; they should not become a standing ritual after that incident. No prospective runtime or memory reduction was benchmarked by this audit.

## Remaining limits on offline prediction

- Local accuracy/cost forecasting is useful and the latest pairing worked; it does not reproduce the private fresh-network selection exactly.
- The Linux v31 gate used Python **3.12.3**, NumPy **2.4.6** (`linux_memory_gate_estimator_k3_v31.json:164`), whereas the graded report records Python **3.10.21**, NumPy **2.2.6**. It proves an actual cap on that Linux stack, not exact environment parity. Match the deployment Python/NumPy stack for the final designated-file certificate; retain the existing run as corroboration.
- A 13.4-microsecond-per-operation proxy and a 22,000-op threshold are conservative for measured members of this family, not a guarantee for arbitrary scalar synchronization, allocation or control flow. Certification's scalar-materialization check (`certify_candidate.py:1805`) is relevant; retain it and real graded telemetry.
- Three repeated predicts under a controlled memory cap test the implemented lifecycle but do not exhaust every possible input-dependent branch or all 100 private MLPs. A memory accounting model alone is weaker still.
- New fitted constants must be selected on DESIGN and checked on fresh DECIDE under the operative policy. Public confirmation is not a tuning oracle, regardless of submission quota.
- Improving iteration speed should mean more distinct, informative candidates with durable evidence, not more submissions of unsupported changes. The next major score improvement requires approximation/cost gains; repairing orchestration makes those gains faster to discover and safer to ship.
