# Validation infrastructure audit — 14 September 2026

This is a read-only audit of current code, installed official-package source, and already exposed evidence. No estimator or instrument was changed; no experiment, seed claim, truth-file read, vault operation, or heavy workload was run. Existing result JSONs were inspected and simple aggregate arithmetic recomputed. Findings below are review conclusions, not newly measured candidate performance.

## Decision

**Keep the existing validation core, but do not claim it can reproduce the complete private grading service offline.** It now provides a credible paired accuracy-and-counted-cost forecast for frozen candidates. It does not reliably reproduce graded residual or wall time, and it needs tighter numerical sensitivity, durable failure evidence, and production-truth precision provenance before ambitious accuracy work.

The strongest calibration result is F21: the fresh P10 arithmetic gain forecast, 1.150566x, agrees closely with the public paired descriptive gain, 1.150559x. Its f64-readonly counted cost exactly matches all 100 graded rows: 928,457,637,478 FLOPs. Its absolute fresh forecast, however, is 1.6744893e-8 against the displayed public 1.5957295e-8, about 4.94% higher. Those are different networks and differently precise labels; this is neither a scorer defect nor a paired accuracy comparison. It demonstrates why relative paired validation is stronger than promising an exact future absolute score.

The P10 raw F21 error is 4.1821088e-8 beside midpoint label-noise floor 2.2363658e-9. The public truth-noise floor is unavailable. Sources: `experiments/2026-09-13_p10/decide_P10.json`, `experiments/run_reports/f21_diagnostic_330950_graded_report.json`, and the independent `experiments/2026-09-13_p10/f21_graded_skeptic.md`.

## What the implementation gets right

| Question | Code evidence | Assessment |
|---|---|---|
| Are networks drawn correctly? | `local_sim/mint_provenance.py:53–88,131–187` | Canonical float64 Gaussian draws, He scaling, one cast to float32, independent weight/sample/estimator seed streams. Verification includes live 1024x16 shape against installed package primitives. |
| Is the scored quantity right? | `local_sim/competition_score.py:181,290`; installed `.venv/Lib/site-packages/whestbench/scoring.py:943–978,1078` | Final-layer float32 MSE multiplied per network by `max(0.1,F/2**41)` at lambda zero, then arithmetic mean. It does not substitute a geometric mean or multiply separately averaged quantities. |
| Are failures scored correctly? | `local_sim/lifecycle_eval.py:698–794`; `competition_score.py:181` | Exceptions and exhausted budgets/timers zero predictions; failed multiplier is one. It reproduces exception-cost reporting and ordering of post-predict checks. Setup failure normally aborts instead of yielding a misleading clean score. |
| Is the lifecycle realistic? | `local_sim/lifecycle_eval.py:628–670,713–721` | One estimator with setup once outside the budget; output materialization inside it. Immutable, source-bound evidence rows retain round deviations and measurement path. |
| Is pairing enforced? | `competition_score.py:800,896,1004`; `f21_score.py:324–338` | Identity alignment, common truth and seeds, required explicit promotion gate; arithmetic difference with t/basic interval. |
| Is truth noise recognized? | `p2_truth.py:76–87,125–165,394–415`; `competition_score.py:1004–1110` | Independent replicas, midpoint noise, subtraction of label variance and addition of official-precision variance. Unequal multipliers' label-noise bias is handled. Negative unbiased estimates are retained rather than silently clipped. |
| Are cost representation traps covered? | `lifecycle_eval.py:504–553`; `p2_flop_anchor.py:20–57,157–180`; `certify_candidate.py:1823–1933` | Explicit f32/f64/read-only arms, metered weight methods, and output cast cost. The shipped-file anchor checks operations against a stored official local subprocess report. |
| Does the memory gate actually enforce a limit? | `linux_memory_gate.py:113–139,170–329,630–775`; `certify_candidate.py:930–1110` | Linux/WSL address-space cap with uncapped positive and capped negative controls, repeated-worker retention controls, candidate/instrument/config bindings, and same-worker repeated predictions. This is a real improvement over Windows accounting. |
| Are held-out reads protected? | `seed_ledger.py:305–343`; `truth_reads.py:1–48,83–90,237,323` | Exclusive filesystem lock for claims and reserve-before-load read accounting. Aborts still spend a single-use read. It explicitly does not pretend to be a security boundary. |

The latest accepted memory artifact, despite its filename `f21_linux_cap_v6.json`, identifies artifact schema version 5. It records a real applied 8,192-MiB limit, 23/23 checks, approximately 5,381.7 MiB peak virtual size, and three completed same-worker predictions. Its filename is not a schema assertion. Generic preflight still checks the shipped v2.6.1 file; candidate certification separately binds F21. These scopes must remain distinct.

## Priority findings and acceptance criteria

### P0 — Timing evidence needs an explicit forecast-versus-measurement release decision

**Confirmed limitation.** The capped local F21 mini failed 100/100 on residual time, with local residuals 7.312–9.633 seconds. The actual graded run completed 100/100 with worst residual 0.15623225 seconds and worst predict wall 93.1597153 seconds. The local refusal is real, but it did not predict failure on the server. Source: `experiments/2026-09-13_p10/mini_F21_capped_official.json` and the graded report above.

Conversely, the server can be slower in total wall time even while showing much less residual. `official_config.py:431–478` records a prior Strassen FLOP saving accompanied by a large wall increase and server contention. A single local/server scalar conversion is therefore invalid. Lower FLOPs can mean more small kernels, communications and overhead, and less arithmetic throughput.

The repo already separates MATH, LOCAL, SERVER_PREDICTED and SERVER_MEASURED in `certify_candidate.py:167–213,254–294`. Keep those domains. The 22,000-op policy and 13.4 microseconds/op model (`official_config.py:378–504`) are conservative planning rules from limited observations, not hard guarantees. F21 now adds an own-code observed point near 14,091 operations, materially wider than the original two direct anchors. Do not silently refit historical gates after observing it.

Fresh Chrome evidence supplied by the coordinating reviewer strengthens this caution. The participant's August-28 correction in [discussion 18202, post 4](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202/4) attributes residual pressure to scalar `bool`/`float`/`int` coercions that flush the remote computation graph: removing 166 such points reportedly changed a run with 5/50 failures to 50/50 success while retaining roughly 21,000 calls. This is participant-reported evidence, not our independently reproduced result or an organizer guarantee. It directly warns that equal operation counts need not imply equal residual cost. The later August-29 report of crashes with roughly 1.1-GB batched three-dimensional payloads is unresolved participant evidence. Track scalar synchronization/graph flushes, serialization payloads, and kernel structure in the candidate model; do not replace this with a fixed microseconds-per-call promise.

**Proposed work:** a concise candidate release record that joins exact source/package hash, numerical decision, counted cost, Linux cap, local official status, graded measured status if available, and the forecast assumptions. Reconcile misleading labels such as `GRADED PATH` for local seconds in `scripts/preflight.ps1:582`. A future policy change concerning local false negatives is STRATEGIC and needs its own explicit contract; this audit does not waive any existing gate.

**Acceptance:** replay stored F21 local and graded artifacts into the record and obtain LOCAL REFUSED plus SERVER_MEASURED PASS, never a combined fictional local pass. A capped run, no-cap accuracy forecast and op-count model must remain distinguishable. Unknown/missing telemetry must remain unknown. Before any changed arithmetic implementation, report largest kernel shapes, operation counts, memory lifetime and explicit server-wall uncertainty, not FLOPs alone.

### P0 — Current numerical tolerance is too loose for the intended accuracy scale

**Confirmed coverage gap, not an observed F21 accuracy defect.** `cert_F21_full.json:16328–16333` records `ARC_WHEST_RAW_MSE` unset and denominator 1.4e-6. Its permutation bar is therefore 1.4e-8, approximately one third of P10 F21 raw error, whereas the intended rule is 1% of an appropriate error scale. The same old scale was identified in the September-10 audit and remains in the latest report.

The tighten-only mechanism already exists (`edge_cases.py:658–720`, `certify_candidate.py:320–339`). No new framework is needed. Predeclare a defensible expected raw-error scale for the next candidate, use it consistently across subprocess shards, and save it in the certificate. Check actual observed drift before deciding whether any expensive rerun is warranted; a loose threshold is not proof that the candidate approaches it.

**Acceptance:** a deliberate numerical perturbation exceeding 1% of the declared candidate error must fail; a known equivalent implementation with smaller observed drift must pass. Verify the production 1024x16 branch and early/middle/late hidden-neuron permutations. Thresholds are committed before candidate measurement, not inferred from whichever result passes.

### P1 — Failure metadata is improved, but expensive single-use results remain fragile

**Confirmed implementation gap.** P09 spent its read and lost its failure category before the scorer aborted. Current `f21_score.py:214–235` now prints failing row flags and timing. However `f21_score.py:304–320` retains whole-arm results in memory, and the only durable result is written at `406` after all arms finish. A process death or interruption can still lose completed observations and prevent precise recovery. Printed diagnostics are a useful repair, not a crash-safe evidence journal.

**Proposed work:** append-only per-row diagnostic/evidence records, with an atomic final manifest that says COMPLETE, FAILED or INTERRUPTED, source/instrument/truth hashes, row identities, expected row count and completed rows. Preserve original observed outcomes on every failure. Under the read protocol, such a journal does not grant another truth read, permission to change a candidate, or permission to infer a promotion from partial data. Any permitted resume behavior must be designed and preregistered before a single-use experiment.

**Acceptance:** crash-inject after one complete arm, mid-arm and during final publication using synthetic or reusable unprotected inputs. Persist completed rows and the terminal failure marker; refuse promotion on incomplete data; prove duplicate rows cannot be silently accepted; do not consume or forgive a second protected read. A successful process exit must not be sufficient to call a run accurate or deployment-ready.

### P1 — Production truth needs numerical-runtime provenance and a matching parity check

**Confirmed provenance gap; bias remains unverified.** `p2_truth._truth_one` executes float32 GPU matrix multiplication and float64 accumulation (`p2_truth.py:125–165`). The producer does not explicitly pin TF32/matmul precision and its saved NPZ metadata (`402–414`) lacks Torch/CUDA/device versions, numerical backend flags, chunk size and producer hash. `gpu_parity.py:57–68` explicitly disables TF32 but defaults to 256x32; its standalone forward implementation is not the production truth function.

Two replica streams can share a numerical bias. Their disagreement ratio near one tests random noise behavior, not GPU-versus-official arithmetic bias. This does not show P10 labels are wrong: present code plus receipts simply cannot close that question at increasingly small errors. The concern is especially relevant when pursuing an accuracy gain beyond F21's nearly unchanged predictions.

**Proposed work:** pin and record the actual production numeric mode, add a sidecar truth-producer manifest, and compare shared-input forwards at 1024x16 using the same production function and an independently specified reference. Use the installed official baker's documented precision as reference, not merely an old parity-script default. Preserve float32 label export separately from the accumulator's precision.

**Acceptance:** run a preregistered lightweight pilot through workerq when needed, with exact common inputs, compare per-layer means and variance and isolate arithmetic bias from sampling error. The conservative bound on resulting final score error must be small relative to the planned intervention's gain. A changed backend flag, library version or producer bytes must be visible in provenance. Verify chunk-size changes against the same requirements before optimizing a large mint. No new protected labels are required for this instrument work.

### P1 — Cost and package parity must remain candidate-specific

**Confirmed distinction.** Installed `.venv/Lib/site-packages/whestbench/domain.py:126–137` and `subprocess_worker.py:18–26` explicitly reconstruct f32 weights. Existing graded service evidence supports the f64-readonly representation model for the current submissions. F21's local f32 mini count is 928,424,083,046, lower than the matching graded-twin count by 33,554,432: the known opening weight-cast effect. This is already explained in `docs/GOTCHAS.md:68–103`; it must not be rediscovered or mislabeled as an unexplained new gap.

F21's graded count needs **zero additional offset** beyond its twin count. `official_config.py:303–339` retains maximum observed offset `2**24` as a conservative sensitivity bound for other cast behavior; `f21_score.py:186–202` evaluates both. AGENTS' blanket unexplained-offset sentence is stale relative to these details. Do not add the maximum twice or claim the conservative forecast is exact.

`grader_stack.py:49–67` pins official package versions and records but does not equate Python/NumPy versions. The F21 server reports NumPy 2.2.6/Python 3.10.21, whereas accepted Linux cap artifact reports NumPy 2.4.6/Python 3.12.3. Version matching is valuable but not proof of all remote protocol behavior or timing equality.

**Acceptance:** per candidate, reconcile local f32 and f64-readonly ledgers, in-window output conversion, and any candidate-specific observed server offset. Keep the existing op-by-op shipped anchor, but do not transfer it into a universal cost certificate for new primitives. A deliberately altered dtype or omitted materialization must break the parity test. Attach latest runtime version observations without automatically changing a frozen experiment's assumptions.

### P2 — Tighten scope and reduce contradictory status prose

CURRENT_STATE has an authoritative newest paragraph followed by historical statements that F21 remains unpromoted, unsubmitted or has no new DECIDE. They are chronology, but phrased as current instructions. `scripts/preflight.ps1:633–634` still says nothing here tests the memory cap, while the actual candidate Linux gate does. Native Windows cannot enforce RLIMIT_AS; this is different from the repository having no Linux proof. The default `gpu_parity.py` command likewise remains Phase-1 shaped.

**Acceptance:** one short current status matrix sourced to immutable evidence, then dated chronology; one candidate-specific release command/record; no wholesale deletion of historical refusals or implicit weakening of controls. Preserve the old shipped baseline separately from the latest competitive candidate. Keep normal preflight as infrastructure/ship-file maintenance, not the default expensive answer to every unrelated read-only review.

## Statistical limits that matter for winning

1. **A paired score decision is not a guaranteed private result.** Fresh blocks, committed choices and common random numbers make the inference credible; they do not make 128 networks identical to the private population. Changes to the seed protocol also need scoped handling: canonical mint implements the independent spawn-based protocol, while installed `domain.py:96–116,155–176` supports a salted protocol 4.0 estimator seed. This is not evidence of a distributional mismatch for the current deterministic analytic F21 method. Exact private seed identity is unavailable and should never be promised.
2. **Truth noise becomes a research cost.** P10's midpoint noise is about 5.35% of its raw F21 error. That is reasonable for an almost algebraically equivalent cost intervention. A small accuracy intervention needs a power/precision plan based on the isolated paired score delta, not merely a blanket floor percentage of total MSE. Report measured, noise-corrected and official-precision forecast scores together and retain the uncertainty from per-network sampling and label noise.
3. **A failure overwhelms a frontier score.** Existing P10 F21 rows have mean zero-prediction MSE 0.9089196. One representative failure in a 100-network evaluation would contribute about 0.00909 to its mean score, versus a successful score near 1.6e-8. Zero failures in one 100-row graded run is valuable evidence, but under an IID binomial model alone the one-sided 95% upper bound is still about 2.95%. Reliability must come from structural resource headroom, exact contract behavior and branch/lifetime coverage, not claiming that a modest zero-failure sample statistically proves essentially zero failure probability. These arithmetic illustrations do not predict the private suite size or its failure rate.
4. **The winner target is not just the cost gate.** Under F21's current displayed raw error, even reaching the 0.1 cost floor leaves an adjusted score about 3.7794e-9, above the dated 2.5e-9 leader. Infrastructure should therefore discriminate both faithful cost reductions and real error reductions, while retaining the current full recurrence as a numerical reference. A cheap approximate recurrence that does not match that reference when compression is disabled is not a valid compression experiment.

## Recommended implementation order

First consolidate the candidate release evidence and freeze future numerical sensitivity. Next implement crash-safe result recording and production-truth precision provenance using synthetic/reusable inputs and mutation tests. Then build one faithful full-rank reference comparison for the selected architectural intervention before any compressed approximation or fresh DECIDE block. Run the already appropriate candidate-specific Linux/official checks only when relevant code paths change or unresolved failures justify them. Reuse accepted unchanged evidence rather than repeatedly spending compute on unrelated old baselines.

These instrument changes are IMPORTANT under AGENTS: record a written prediction, complete affected checks and preflight, append the knowledge delta to LOG, and obtain fresh independent Skeptic review. Any change to a gate, release exception or data protocol is STRATEGIC. This audit changes none of those rules and authorizes no submission or designation.
