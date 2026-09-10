# Phase-2 offline validation audit — 2026-09-10

The repository has a strong Phase-2 accuracy and FLOP-accounting foundation. It does **not** have an exact offline replica of AIcrowd's deployment environment, and its sealed vault is still a Phase-1 instrument. The next infrastructure investment should close a few specific provenance and lifecycle holes, rather than replace the evaluator.

This began as a read-only source and stored-evidence audit and was followed by existing infrastructure checks. No research seed was claimed and no vault evaluation touch was taken. The default preflight used its established test inputs and verified vault integrity, without producing a new candidate accuracy result. `.venv/Scripts/python.exe local_sim/official_config.py` returned **96/96, exit 0** against installed WhestBench 0.16.1 / flopscope 0.12.1. The existing certifier's pure report-aggregation `unit_tests()` returned **19/19, exit 0**. Queued default preflight **passed 12/16, four explicitly skipped, exit 0**; no full preflight or fresh nominee certification is implied. Source references below are paths relative to `C:/Users/samsu/Documents/arc-whest`, with one-based line numbers at audit time.

## What the offline evaluator can establish

| Question | Assessment | Evidence and boundary |
|---|---|---|
| Is the random-network distribution correct? | Verified by the existing parity test | `mint_provenance.py:54–89` reproduces seed-stream splitting, float64 normal generation, He scaling, then float32 cast. `verify():142` compares with the installed official generator at the live 1024×16 shape. Default preflight reran the provenance test successfully. |
| Are seeds semantically correct? | Yes in the canonical mint | Weight, truth and estimator streams are separate; `MLP.seed` is the estimator stream, not the weight seed. Old legacy helpers remain and must stay historical. |
| Are the round and packages current locally? | Verified at audit time | `official_config.py:583` checks package pins, round defaults, stored run settings, and graded-report-derived calibration. `pyproject.toml` pins exact 0.16.1 / 0.12.1 versions. |
| Is the scoring identity correct? | Yes through the canonical scorer, given correctly formed rows | `competition_score.py:123–178,284–354`: score each MLP's own MSE and multiplier, then average. Budget exceedance is strict `C > B`; failed predictions must have zero-output MSE and multiplier 1.0. |
| Is the candidate lifecycle reproduced? | Substantially in `lifecycle_eval`, not in every legacy consumer | `lifecycle_eval.py:590–791` handles one setup, per-MLP budgets, weight representation, materialisation, failure order and float32 score arithmetic. `harness.run_estimator` and `vault._run_candidate` do not use this path. |
| Are counted FLOPs trustworthy? | Strong, candidate-dependent evidence | The local official-package anchor is op-by-op. Genuine f64→f32 cast candidates have exact aggregate matches to stored graded totals; the same-dtype cast family still needs an empirical offset. No grading-server operation ledger is available, so universal server op-by-op parity is unproved. |
| Are local residual seconds the grading server's seconds? | No | `VALIDATION.md:433–542` and `GOTCHAS.md:126` document large systematic differences even under local official subprocess runs. The operation-count model is the useful predictor. |
| Are local absolute wall seconds transferable? | No | Backend topology and contention differ. The 1.60 contention factor in `official_config.py:462–479` is a planning model, with explicit uncertainty about whether it scales to long candidates. |
| Is 8 GiB enforced locally? | In a separate Linux process; not on Windows | `linux_memory_gate.py:111–124` applies and reads back `RLIMIT_AS`; positive and negative controls run before the candidate. Its current report acceptance and weight-representation gaps are detailed below. |
| Can offline evaluation reproduce the exact hidden leaderboard score? | No | Fresh mints estimate performance on the distribution; they cannot reveal the hidden suite, its realised labels, server load, or exact private rerun outcome. Confidence intervals and deployment-risk estimates are appropriate outputs. |

The installed official source confirms Phase 2: 1024 wide × 16 deep, 1,000,000,000 truth samples, `B = 2**41`, lambda 0, 0.4-second residual cap, and 120-second wall cap (`.venv/Lib/site-packages/whestbench/budget.py:133–151`). `scoring.py:60–74` additionally carries setup 5 seconds and process memory 8192 MiB. Its `predict_timeout_s=30` is not a competing 30-second hard prediction deadline: `runner.py:82–90` uses the maximum of that value and wall limit plus response grace.

The driving agent's Chrome review is the authority for newly published website details. It reports Rules v14 §5.6 now explicitly states the 120-second cap, five-second import/setup limit, one physical core for the participant and seven for the backend. It also reports newly graded submission 330566 uses WhestBench 0.16.1 / client+server flopscope 0.12.1, participant shim 0.1.0, Python 3.10.21 and backend NumPy 2.2.6. That corroborates the current package pins. This audit did not independently browse those pages; use the driving agent's saved browser evidence for direct citation.

## Findings that should change the implementation plan

### 1. Bind Linux memory evidence to the entire tested configuration

**High-priority correctness gap, source-confirmed.** `certify_candidate.py:733–780` finds a Linux memory report by candidate SHA. It then checks `cap_applied`, negative-control failure, `predict_error == ""`, and `status == "PASS"`. It does **not** verify that the report's width, depth, budget, memory limit or runtime versions equal those of the candidate being certified.

Those fields are configurable in `linux_memory_gate.py:361–370`, and recorded in its report at lines 395–402. Consequently, a same-SHA report generated at a small shape or a looser memory cap can satisfy the consumer's current checks. This is a reachable acceptance-path defect, not evidence that an existing submitted candidate exceeds memory. I did not run a mutation or claim that any stored report was forged or misconfigured.

The Linux worker also builds writable float32 weights (`linux_memory_gate.py:153–157`), whereas the grader-twin contract uses float64 read-only weights. Its peak therefore does not directly include the same live input representation and conversion path. This matters especially for a file with substantial cached state or casts, and should be recorded as a scope limit on the existing report.

Repair as an IMPORTANT slice: require exact candidate hash, live shape, budget, 8192-MiB limit, read-back soft/hard limit, weight mode, official package versions and successful controls. Record the instrument hash as well. Run the candidate under `lifecycle_eval.build_weights(..., "f64_readonly", ...)`; include repeated predictions in the same worker to detect persistent caches or growing state. A single prediction cannot prove suite-lifetime memory stability. Add metadata-mutation tests: wrong shape, wrong cap, wrong version and stale SHA must each refuse. Do not silently reinterpret old reports as passing the stronger gate.

### 2. The vault is neither Phase-2-shaped nor lifecycle-equivalent

**High-priority capability gap, source-confirmed.** `vault.py:71` fixes width/depth to 256×32; the archive documented at lines 29–32 has 500 such networks. It cannot certify a 1024×16 milestone. Preserve that seal as historical evidence; never repurpose it in place.

There is also a separate lifecycle gap: `vault.py:244–291` uses bare NumPy weights, makes a new `Estimator` per network, never calls `setup`, and omits official output materialisation. Its budget context lacks the official wall limit, and there is no matching residual/error-to-zero path. `_score_block():294–342` excludes residual and invokes `score_suite(on_over_budget="fail")`, which explicitly retains candidate MSE when forcing the failure multiplier. That is labelled optimistic inside the canonical scorer and cannot model an actual failed submission.

This does not retrospectively invalidate every accuracy observation on the old vault. It means those observations have their actual old shape, candidate and lifecycle scope; they are not deployment or Phase-2 certification.

Repair the evaluator wiring first on sacrificial synthetic fixtures, before any new sealed data exists. A Phase-2 vault or fresh final single-use block requires the STRATEGIC protocol, Samuel's prediction, and an explicit authorized touch plan. Reuse `lifecycle_eval` rather than create another prediction loop. Keep statistical accuracy, locally observed failures and predicted server deployment status separately visible. Merely switching the old vault to live lambda while retaining its old shape is not round-aware evaluation.

### 3. Single-use claims do not enforce single-use reads

**High-priority research-integrity gap, already observed.** `seed_blocks.py:205–230` explicitly records that P08 was read three times while the instrument was repaired. `seed_ledger.claim_range():431–489` prevents another single-use claim; it cannot stop another `np.load` of the claimed truth. The LOG tail correctly withdraws “single held-out read.”

Add a read registry with block/artifact hash, purpose, consumer and instrument hash, UTC timestamp, result artifact and status. Reserve a read before exposing the labels, and record aborted attempts; otherwise a crash becomes an uncounted look. A cap should refuse a second decision read. Truth-consuming code should use one shared loader. Add an audit for direct protected-artifact loads, while being clear that a single-operator filesystem is an audit discipline rather than an adversarial security boundary.

Concurrent agents also make atomic reservation important: the current claim path reads the ledger, checks it, then appends without an interprocess lock (`seed_ledger.py:246–259,460–489`). Two simultaneous claims can both pass the pre-append check. The present task allocated no seeds, so this is a prospective concurrency defect, not a finding of overlapping claims today.

P08 should remain spent. A repaired instrument and plausible negative result do not restore a pristine held-out test. If a future promotion depends on that result, use genuinely untouched networks under a frozen contract; do not rerun P08 as confirmation.

### 4. Separate a local runtime result from a server forecast

**High-priority communication and decision-model issue.** `certify_candidate.py:532–564` correctly gates operations against `official_config.PHASE2_OP_COUNT_GATE = 22,000`, while treating local residual as a ratio. Section 10 at lines 786–823 still labels local subprocess residual “GRADED-PATH” and deployment authority. `scripts/preflight.ps1:516–560` uses similar wording. `VALIDATION.md:531–536` deliberately preserves the conservative refusal and requires Samuel to weigh both numbers per submission.

These facts can coexist, but one unqualified CERTIFIED/REFUSED label conceals the distinction. Keep the old failed gates on the historical record. Do not amend them after seeing candidate results and declare a retroactive pass.

For future contracts, expose at least three explicit states: mathematical/contract checks, local runtime checks, and predicted server deployment risk. The server forecast should show measured grader-twin operations, the versioned per-operation model, its evidence domain and headroom; local subprocess failures should retain their real flags. An exceptional submission remains Samuel's per-submission decision under repository policy.

`official_config.py:380–528` currently carries 13.4 microseconds per operation and a 22,000-operation planning gate. Its 96/96 check reproduces stored observed bounds. It does not establish an upper bound on every future candidate's server residual. Operation mix, scalar work, backend behavior and contention can change. “Within the calibrated envelope” is supportable; “guaranteed safe offline” is not.

### 5. Tighten deterministic and numerical verification to the winning accuracy scale

`certify_candidate.py:347–376` runs two fresh interpreters **for each** of two seeds and compares same-seed hashes. The two hashes stored in `sections.determinism.fresh_worker` are for different seeds, so their inequality is expected. The artifact does not store the second member of either comparison or the seed beside each hash. Store both hashes, both PIDs, seed, mode and comparison verdict to make the evidence self-contained.

The fresh worker currently uses float32 weights and no official materialisation (`certify_candidate.py:146–179`). Extend it to f64 read-only grader inputs and the actual returned scoring dtype. Also test repeatability across a multi-MLP sequence, not only a new process per network, so hidden mutable state cannot pass unnoticed.

The depth-2 candidate's existing certification records `arc_whest_raw_mse` as unset, hence the invariant uses the frozen `1.4e-6` signal (`experiments/2026-09-10_slice5/certify_k3_strassen_L2.json:23–30`). For a candidate near `4e-8`, a tolerance tied to 1% of `1.4e-6` is much looser relative to the target error. This is not proof of a numerical defect; it is a mismatch between sensitivity and current ambitions. For new candidates, freeze a conservative expected MSE scale before evaluation and pass it through the existing tightening mechanism. Verify GPU research mirror versus deployed flopscope file per layer, prediction dtype and live shape before using mirror accuracy as evidence for that file.

### 6. Make paired inference and label-noise accounting impossible to misread

The primary arithmetic statistic and `decide(gate=...)` are right. The gate explicitly refuses percentile and BCa intervals; legacy diagnostic APIs still expose them. `paired_compare_rows():670–685` checks only equal row counts and then pairs by order; it does not verify network identity. Equal length does not prove common random numbers. Add immutable MLP identifiers and reject mismatched or duplicate IDs before pairing; report deliberate truth/randomisation sharing too.

A smaller real API defect: `paired_diff_ci(method="t", alpha=...)` at lines 456–489 uses `_t975` for **every** alpha. The documented/current 95% default is unaffected, but a requested 99% interval silently remains 95%. Either reject unsupported alpha or implement the requested quantile. Do not claim any published default interval was changed by this latent issue.

The two-replica truth design (`p2_truth.py:10–27,77–87`) is useful. For a frozen prediction `p`, independent labels `y_A=t+e_A`, `y_B=t+e_B` give the unbiased signal estimator `mean((p-y_A)*(p-y_B))`. The common-label squared-error noise bias cancels between two **raw MSEs**. It does **not** cancel from an adjusted-score comparison if their multipliers differ:

`E[q_c MSE_c - q_k MSE_k] = q_c MSE_true,c - q_k MSE_true,k + (q_c-q_k) v`,

where `v` is that network's label-noise floor. Pairing removes much variance, not this unequal-cost noise term. This is a mathematical scope correction to `p2_truth.py:14–16`, not evidence that a particular prior comparison is wrong.

Report two outputs: noise-corrected true-function performance, and predicted competition score including the official label-noise contribution at one billion samples. When multipliers vary by network, add each network's noise contribution before averaging. Preserve negative cross-product estimates as noisy estimates; do not clip them to zero or drop those networks. The geometric diagnostic's positive subset has no promotion authority.

Truth generation uses float32 forward arithmetic and float64 accumulation (`p2_truth.py:122–167`), but does not explicitly freeze TF32/matmul precision or save a complete numerical-runtime manifest in its output (`p2_truth.py:402–414`). Record Torch/CUDA/device versions, precision settings, chunk size and producer hash. A cross-platform numerical-bias check is distinct from replica independence: two replicas can agree while sharing the same arithmetic bias. Replica-disagreement checks should produce stored pass/refuse evidence, not only a printed ratio.

### 7. Make preflight scope and byte identity explicit

`scripts/preflight.ps1:321–325` runs the historical default invariant command; its candidate-specific live counterpart is in `certify_candidate.py:303–335`. The preflight primarily certifies `estimator.py`, which remains v2.6.1, while the active nominees are separate files. A green preflight on the shipped baseline does not certify a separate nominee's behavior. Preserve both requirements but name their targets clearly in the evidence packet.

Step 5 compares normalized LF bytes (`scripts/preflight.ps1:347–363`). That is appropriate for checking semantic archive identity but does not establish the exact SHA submitted to AIcrowd. `.gitattributes` currently contains only a raw-cache rule, with no `*.py eol=lf`. Add deterministic Python line endings as an IMPORTANT provenance change, preserve historical raw hashes, and verify the packaged inner file against the exact certified byte SHA. Do not rewrite frozen candidate archives for documentation cleanup.

The certification consumer's regression suite is not named in preflight. Add the lightweight metadata-consumer tests to an existing instrument-check step, so the memory binding and report schema guards are exercised automatically, while leaving expensive candidate execution queued.

### 8. Native `out=` support is not proof of a valid participant buffer pool

The installed native flopscope implementation contains metered `out=` handling (`_counting_ops.py:59–128`, `_dtype_billing.py`, `_ndarray.py:224`), but explicitly rejects indexed assignment, fill and other array mutations (`_ndarray.py:943–987`). Its errors say these match the client API. The grading participant uses a separate shim/client, so a native operation accepting an output destination does not establish that a mutable buffer pool is available in the actual sandbox. The repository's current immutable-array contract also has to be respected. No buffer-pooling implementation or remote API experiment was performed here. Treat it as an API question requiring exact official participant semantics and a narrow contract clarification, not as a deployable cost optimization established by native NumPy behavior.

## Verification executed during this audit

| Check | Result | Scope |
|---|---|---|
| `official_config.py` | 96/96, exit 0 | Installed and stored graded configuration, scalar score behavior and existing telemetry bounds; no new accuracy data. |
| `test_certify_candidate.unit_tests()` | 19/19, exit 0 | Pure synthetic report/failure aggregation and traceback attribution. The expensive mutation candidates were not run. |
| Default `scripts/preflight.ps1` | **GREEN 12/16, four SKIPPED; job 539 SUCCEEDED, exit 0** | Existing default gate, targeting shipped v2.6.1, not a new nominee. |

Job 539 was submitted with 8 GiB RAM, zero VRAM, zero GPUs and two CPUs. `--no-snapshot` keeps the ignored vault integrity asset accessible; preflight records the HEAD and dirty state at start. The source HEAD was `0fe164cbee81bb329be0672a232554f4faf05459`; only new audit/research paths were dirty at startup. Command:

```powershell
workerq submit --project arc-whest --ram 8 --vram 0 --gpus 0 --cpus 2 --no-snapshot -- powershell -NoProfile -ExecutionPolicy Bypass -File C:/Users/samsu/Documents/arc-whest/scripts/preflight.ps1 -Log C:/Users/samsu/Documents/arc-whest/experiments/2026-09-10_validation_audit/preflight-default.log
```

The transcript is `experiments/2026-09-10_validation_audit/preflight-default.log`; line 81 carries the final verdict. The complete native child output is copied to `preflight-worker-539.log` in that directory, SHA-256 `a65687e165dc72ba71f519a33396fdb6345d154f16ac9becaa891996791158fd`. `workerq-539.json` records the successful job. No instrument repair was applied during these checks.

The completed default run checked live-shape mint provenance, dataset-pin regressions (74/74), shipped-file invariants and their mutation battery, `whest validate`, archive identity, vault seal/integrity, seed-ledger consistency, documentation/enforcement labels (39/39), the canonical scorer (98/98), ledger refusals (66/66), lifecycle evaluator (56/56), and the official-package FLOP anchor. The anchor compared 19 operation categories with zero disagreements and exact total equality at 137,600,326,656 counted FLOPs. This is the shipped-file local official-package anchor, not universal server parity. The ledger reported zero hard collisions and retained its one historical vault touch.

The four unexecuted full-tier steps are shape parity, carrier strength, deployment envelope and official Phase-2 subprocess smoke. Linux memory enforcement, fresh nominee certification, a full public confirmation and a Phase-2 sealed milestone were not run. Passing the existing tests does not refute the newly identified gaps: those are properties the current tests do not assert.

## The practical offline testing architecture

Keep one candidate file, one round configuration, one weight adapter, one lifecycle evaluator and one scorer as the authoritative path. Research mirrors remain fast design tools whose outputs must be reconciled with that path. The recommended sequence is:

1. **Truth-free contract checks:** imports, setup, live shape, read-only input, exact layer 1, finite/nonnegative output, same-seed determinism, permutation equivariance and bounded numerical drift. Use positive and deliberately broken controls.
2. **Truth-free cost/deployment profile:** genuine f64 read-only inputs, output materialisation, counted FLOPs, operation census, scalar materialisations, Linux capped memory, repeated-worker state, and explicit server forecast. Freeze the file and model before labels.
3. **DESIGN data:** explore only on registered reusable design networks, with paired replicas, per-layer diagnostics and numerical-runtime provenance. Positive controls must be sized at the proposed gate, as the T2 repair demonstrated.
4. **Fresh DECIDE read:** preregister one hypothesis, exact baseline and file hashes, score/noise model, threshold, paired statistic and failure handling. Commit before the protected read. Store per-MLP IDs, costs, raw MSE, noise-corrected estimate, official-noise prediction and failures.
5. **Independent Skeptic:** supply contract, diff and raw evidence, without builder reasoning. Require explicit checks of measurement scope, cost convention, pair alignment, gate isolation and artifact reads.
6. **CONFIRM:** the public dataset checks a frozen decision. Its labels may not choose constants or rescue a failed decision. Pin the dataset revision and loader provenance; the current Arrow/parquet functional comparison covers one shard, not every published row (`p2_tag_drift.py:37–45`).
7. **Final untouched milestone:** a separately authorized Phase-2 sealed set or equivalent fresh single-use decision set, evaluated once under a frozen nominee plan. The Phase-1 vault is not a substitute.

All heavy executions go through `workerq`, with declared RAM/VRAM/CPU and `--gpus 0` for CPU-only work. For the proposed repairs, run cheap metadata and synthetic regression tests first. Queue full preflight and live-shape certification only after the code and written prediction are complete. The existing queue snapshot rules require exact candidate-byte verification, absolute output paths and explicit handling of ignored truth assets.

No amount of local replication reveals the grading server's hidden MLPs or load. The correct deliverable is a calibrated estimate with uncertainty, a measured cost envelope and a clearly bounded failure risk. The repository already has most of that architecture; its weak point is the few alternate paths and evidence consumers that can silently certify a different experiment from the one the report appears to describe.
