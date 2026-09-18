# Deep validation and iteration audit — 18 September 2026

Scope: source inspection and existing scalar artifacts/transcripts only. No estimator execution, truth-array access, new networks, seed claims, vault access, heavy jobs, submissions, or instrument changes. The only new file is this report. This is an infrastructure review, not a blind Skeptic verdict on a new scientific result.

## Assessment

Keep the evaluator and simplify its orchestration. The repository already reproduces the important mathematical parts of the competition: canonical networks, immutable grader-style weights, final-layer float32 loss, counted FLOPs, mean of per-network adjusted scores, and failure penalties. It also has source-bound prediction artifacts and a public-score projector. Rebuilding those systems is not the next breakthrough.

It does **not** reproduce remote wall/residual timing exactly, prove all future inputs fit Linux memory, or provide a calibrated interval for a score on unseen networks. C10's accepted DESIGN score is promising evidence, not a deployable winner certificate. The principal infrastructure opportunity is fewer duplicate executions and fewer bespoke decision readers. The principal scientific obstacle remains prediction accuracy per counted FLOP.

## What the offline system actually matches

| Property | Assessment and evidence |
|---|---|
| Network distribution and identities | `mint_provenance.py` follows independent weight, truth and estimator seed streams. Existing bit-pattern parity and Phase-2 dataset-pin checks support matching actual networks, rather than merely drawing an approximately equivalent distribution. |
| Live objective | `official_config.py` verifies the installed live defaults; `competition_score.py` computes the mean of individual MSE × multiplier products with B=2^41, lambda=0 and multiplier floor 0.1. Failed predictions are zeroed and receive multiplier 1. A product of aggregate mean MSE and aggregate mean cost is not the generic objective. |
| Execution representation | `lifecycle_eval.py:507` defaults to `f64_readonly`, with flopscope-backed weights. `evaluate()` builds one estimator, calls setup once, includes prediction materialization inside the meter, and computes final-layer MSE in float32. This addresses the historical bare-NumPy/free-cast and float32-weight defects. |
| Counted cost | The local real-package grader-twin path is strongly supported by stored graded comparisons. `predict_public.py:133` distinguishes subprocess-to-server +2^25 FLOPs for the measured K3 family from grader-twin offset zero. The older maximum +2^24 constant in `official_config.py` is family/path-specific historical evidence; do not add it universally. |
| Truth noise | `p2_truth.py` pins IEEE float32 forward math, disables TF32/autocast, accumulates in float64, and uses independent replicas. `p2_design_score.py` averages the two replicas and halves their per-replica noise floor. `competition_score.noise_corrected_compare()` preserves negative unbiased corrected estimates rather than clipping them. |
| Paired decisions | `competition_score.decide(gate=...)` uses the arithmetic score difference and an allowed interval. The current s17 factorial/query readers bind source hashes, ordered identities, configuration, full evidence rows and sidecars, then apply the declared official-precision comparison and failure veto. |
| Remote score forecast | `predict_public.py` uses a compatible paired public-mini ratio and a measured graded control. The existing v31 forecast 8.713276e-9 versus graded 8.752030e-9 missed by about 0.45%. One success is useful evidence, not general half-percent accuracy. Its interval is correctly labeled conditional on local networks, not a hidden-board prediction interval. |
| Memory | `linux_memory_gate.py` enforces and reads back RLIMIT_AS, uses positive/negative allocation controls, and supports repeated predictions to expose retained-state growth. Windows queue peak/working-set figures are resource accounting, not this proof. |

The official-precision local conversion is an expectation forecast, not the official hidden labels:

`forecast_mse_i = observed_mse_i - local_truth_noise_i + local_truth_noise_i * N_local / N_official`.

It removes the expected excess Monte Carlo label noise. It cannot remove the realized cross-term between a particular candidate's error and a particular noisy local label. Repeated selection on P12 also makes a selected DESIGN interval optimistic for generalization. Untouched DECIDE evidence is still needed.

## Important limits and current risks

1. **DESIGN accuracy is intentionally not a resource certificate.** `p2_design_score.py --wall-caps off` disables local seconds caps; zero failures there means no observed numerical/budget failures under that mode. The scorer's `--wall-caps official` wording overstates its equivalence: `lifecycle_eval.evaluate()` is in-process; `predict_timeout_s` and `memory_limit_mb` are stored in its spec but are not process-kill or address-space enforcement in that function. It cannot faithfully exercise worker death. Use the real subprocess and Linux paths for those claims.

2. **Residual is forecast from operation calls, not reproduced locally.** The 13.4 microseconds/call law and 22,000-call working margin have useful family evidence but are not universal guarantees for different operation mixtures or server load. C10's 23,421 calls imply modeled 0.31384 s and 1.2745x hard-cliff headroom. It exceeds the conservative working margin. That is an unresolved release constraint, even with zero DESIGN failures. The empirical wall multiplier in `predict_public.py` is explicitly advisory and should stay so. The root review's live Chrome reading of [forum 18202, post 4](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202/4) finds an important correction: 166 scalar bool/float/int conversions caused graph flushes; removing them left about the same 21,000 calls and produced a clean 50/50 result. Call count alone therefore does not explain that failure. The repository already has the correct complementary mechanism check: `certify_candidate.py:1949` requires zero measured scalar materializations via `MaterializationCounter`. Retain and run that source-bound check for a finalist; a text search for `float()` is insufficient because many conversions are harmless Python configuration scalars.

3. **The generic scorer still gives an observed-score verdict.** `p2_design_score.py:405` passes observed suite scores to `decide`, even when it also prints official-precision forecasts. This is valid only for a contract explicitly gating observed scores. The s17 readers correct this for their own contracts; their results are not undermined. A future generic interface should require a named estimand and emit both results without forcing every slice to write another mathematical adapter.

4. **Snapshot-local registries are not one global lock.** `truth_reads.py:77` locates the register relative to its execution root; `reserve()` locks the ledger beside it. The LOG records reconciling completed reads from workerq snapshots back into the live registry. Existing DESIGN bookkeeping was repaired. Two independently copied snapshots would not share a lock on a future SINGLE_USE reservation; source inspection establishes that limitation, not that a forbidden double read occurred. Before any P13 execution, bind reservation to a single durable live authority, or reserve once before dispatch and pass an authenticated receipt to the snapshot. Never infer global single-use safety from an isolated copied register.

5. **Preflight green is scoped.** `preflight.ps1` validates shipped `estimator.py` (still v2.6.1), not whichever new K3 candidate is of interest. Candidate certification is a separate existing command. Standard green is 12/16 with four named omissions. Full smoke deliberately disables local residual and does not prove the server residual cliff. The terminal still prints an unconditional Phase-1-vault evaluation suggestion; `VALIDATION.md:879` correctly says the existing Phase-1 vault must not be used for a Phase-2 candidate absent a separate contract. The operational message should match that restriction.

These are scope/automation issues, not reasons to discard the accepted negative rank32 result or rerun the completed factorial.

## Measured iteration cost and avoidable duplication

The following are **job elapsed times for scheduling**, not claims about server estimator performance:

| Existing work | Recorded compute duration | Recorded peak host memory |
|---|---:|---:|
| #1882, six complete arms × 24 networks | 7,192.79 s = 119.88 min | 7,861.51 MiB = 7.68 GiB |
| #1885, exact C10 versus rank32, 48 evaluations | 2,132.15 s = 35.54 min | 4,685.30 MiB = 4.58 GiB |
| #1748, fixed own-fit schedule, 228 evaluations | 12,878.98 s = 214.65 min | 6,284.99 MiB = 6.14 GiB |
| #1883/#1884, separate query cost/continuation audits | 104.83 / 109.37 s | 2,575.11 / 3,471.36 MiB |

Evidence: LOG entries around lines 54194, 54676, 54840 and 54913; factorial and query `score.json` arm durations corroborate the distinction between evaluator work and whole-job duration. Queue waiting time, code development and human review time are not included.

**Repeated broad preflight is material.** Eighteen completed `experiments/2026-09-18*preflight*.log` transcripts have start/end stamps totaling **107.30 run-minutes**: sixteen GREEN, two RED. Two additional matching logs are interrupted. Completed green runs span approximately 5.13–6.98 minutes. This sum is not critical-path wall time because some executions overlapped. The source banner's “3–4 minutes” is not a good contemporary budget.

There is concrete duplicate work inside each preflight: step 7 invokes `test_enforcement_labels.py --labels-only`; its `SUITES`, `measure()` and `section_a()` rerun official configuration, shipped invariants, canonical scorer tests, seed-ledger refusal tests and lifecycle tests. Steps 1/3/8/9/11 execute those same suites again. They are rerun to validate prose counts. Capture a structured result from the first execution and validate labels against that result, bound to exact source/configuration identity. This preserves the checks and removes duplicate execution.

Concurrent preflights additionally collided. LOG lines 54625–54635 record RED runs caused by shared `_lifecycle_slowsetup_tmp.py`; `lifecycle_eval.py` still creates fixed temporary module names under `local_sim/` at lines 1173, 1242 and 1285. Use isolated per-run temporary locations and serialize one authoritative broad preflight after a coherent batch of instrument edits. Independent focused tests can still run in parallel.

**Control reuse exists now, but its identity is unnecessarily broad.** The earlier audit's “no cache / non-atomic manifest” observations are obsolete: `--reuse-control` exists, manifests use atomic replace, and strict source/full-row binding exists. However:

- `p2_design_score.py:237` hashes every top-level Python file in `local_sim` (347 files observed in this checkout), including unrelated experimental readers/tests. Any such edit invalidates reuse.
- Lines 279–282 remove only timestamp from the compared config. A change of gate, which cannot affect prediction or counted FLOPs, invalidates the numerical cache.
- Reuse only accepts an old arm literally labeled `control`; C10 was `cand0` in the factorial. A scientifically valid prior candidate is not accessible as a new control through this interface.
- Absolute executable/numerical-environment metadata and workerq-relative paths provide useful provenance but require one canonical resolver; several readers independently got snapshot or Windows separator handling wrong.

The LOG explicitly reran C10 because the new query gate/configuration differed. Its query-run control took **1,118.44 s = 18.64 min**, over half the 35.54-minute pair job. Its raw MSE and FLOPs exactly match the earlier factorial summary. This is evidence of an avoidable future recomputation opportunity; it is not permission to alter the historical contract or reuse uncertified timing.

Additional repeated repair cycles involved source-root resolution, Windows truth-path separators, sorted JSON parameter order and Git CRLF preservation. Those repairs caught real provenance problems and should not be mocked as unnecessary rigor. The architectural problem is reimplementing the same interfaces in each slice, then discovering mismatches only after expensive jobs finish.

## Recommended smaller execution architecture

Implement a thin manifest-driven layer over the existing primitives, not a new evaluator or tracker. Preserve historical adapters/artifacts as frozen evidence.

**One numerical run artifact.** Bind candidate bytes, actual evaluator dependency closure, weight mode, ordered networks, truth digest, numerical configuration, package/BLAS/thread identity, full rows and final predictions. Publish atomically with a final completion marker. Use canonical relative paths plus an explicit run root and stable byte-writing policy. A gate and comparison role belong in the analysis contract, not the numerical cache key. Allow a verified completed arm to become another run's control by hash, without relabeling historical evidence.

**One decision artifact.** Consume complete verified run artifacts and a precommitted contract naming control/challenger, estimand, gate, allowed branches, failure veto and tier. Use one shared official-precision row conversion and `competition_score.decide`; save observed, true-function and official-precision comparisons. The decision artifact carries contract identity and numerical-input hashes. Changing analysis presentation never reruns an estimator. Changing a gate after results still remains prohibited, regardless of cache reuse.

**One release artifact.** Bind exact finalist bytes to candidate-specific conformance, source-bound subprocess CONFIRM, enforced Linux memory, declared operation-call/residual model and independent review. Reuse compatible checks once. A two-row smoke strictly covered by a complete identical-source mini should not require another execution merely for another report title. Current explicit preflight obligations remain binding until prospectively consolidated; do not skip them ad hoc.

**One active runbook.** State the current shipped baseline, graded incumbent, strongest DESIGN candidate, source hashes, active closed lanes, exact command per stage, and remaining release conditions on one page. Archive historical narratives without deleting them. `CURRENT_STATE.md`, `VALIDATION.md` and `GOTCHAS.md` contain valuable history but also superseded operational sentences; a new driver should not rediscover their precedence by reading thousands of lines.

Suggested implementation order, all as IMPORTANT instrument work with prediction/tests/preflight/Skeptic:

1. Make broad tests concurrency-safe; generate one source-bound suite receipt and consume it for count-label validation. Preserve all refusal tests. This directly attacks the measured 107-minute repetition.
2. Centralize artifact path/byte identity and split numerical identity from decision identity. Add verified arm reuse and a declared dependency closure. Test snapshot/live output layouts, separators, CRLF, missing/incomplete rows, source mutation, role swaps and wrong truth identity before jobs start.
3. Replace future slice-specific scalar-score readers with one contract-configured reader; retain method-specific mathematical checks as independent modules. Make the estimand mandatory.
4. Make snapshot-safe truth reservation an explicit interface before the next SINGLE_USE run. Add read-only status to the run manifest; do not use post-hoc reconciliation as the first line of protection.
5. Consolidate candidate release receipts and bring console guidance into agreement with the current Phase-2 protocol. Do not launch a broader infrastructure rewrite while no new accuracy mechanism exists.

## Cadence and submission policy

The evidence supports running multiple **local candidate comparisons** in a day once a bounded mechanism is ready: a six-arm 24-network batch completed in about two hours of compute. It does not support promising a fixed number of remote submissions daily; queueing, truth creation, release checks, permission and result turnaround vary. Do not spend a submission slot just to repair local scalar arithmetic that existing rows already answer.

Use the leaderboard to verify hidden-suite transfer and actual server behavior for frozen qualified candidates. It is not the final prize evaluation: the private fresh-network re-run determines that. Treating every leaderboard feedback as a tuning label would undermine the intended generalization test and the repo's public-read-only rule.

A next major-jump candidate should earn one fresh DECIDE comparison, one candidate-specific release pass and one explicit submission approval as soon as it meets the current gate. Do not hold that candidate for unrelated infrastructure cleanup. Conversely, no simplification turns C10's accepted 1.366x DESIGN gain into the unchanged 2x release case. Changing that target would be a separate prospective strategic decision, not an interpretation of this audit.

## Verification and evidence status

This audit checked source and completed saved scalar artifacts. It did not execute a fresh test suite, independently recompute C10 statistical intervals, or certify C10. Existing s17 Skeptic verdicts support those narrow completed results. Confirmed current implementation facts above are distinguished from risks and proposed changes; the snapshot-global reservation limitation is a source finding, not a claim that P13 has been consumed. No timing/memory forecast here is represented as a measured server result.
