# Independent validation review — September 17, 2026

Scope: source and git review of the worker's September 16–17 changes, existing JSON/JSONL recomputation, workerq job metadata, and lightweight fixture tests. No truth NPZ was opened, no network was minted, no estimator evaluated, and no protected record changed. This reviewer read the worker's contracts and narrative as well as its artifacts; this is an independent verification report, not a claim to have been blind to the narrative.

## Verdict

**The principal live-shape accuracy measurements reproduce, and there is no major release candidate. The instrument repairs are useful but stage 0 is not integration-complete. N1 may reasonably be deprioritized, but its latest result is an exploratory tiny-shape screen with registration defects, not a clean pre-registered closure of compression at width 1024.**

Keep the improved evidence foundation. Repair two concrete release-path defects before expensive confirmation; do not launch a broad infrastructure rewrite. Correct the packet's surviving overclaims before using it as strategist input.

## 1. Recomputed accuracy evidence

Recomputed each network's `MSE * max(0.1, FLOPs / 2**41)` from the saved `arms.*.rows`, paired by identical ordered seeds, then calculated arithmetic differences and t intervals. These results agree with the raw report summaries:

| Candidate / control | Arithmetic score gain | Paired score-difference t95 | Better networks |
|---|---:|---:|---:|
| v32 / v31 | 1.0318263× | [2.35593e-10, 4.04233e-10] | 23/24 |
| v32_i / v31 | 1.0373225× | [2.75003e-10, 4.71341e-10] | 22/24 |
| v32 with reference λ / v32 | 1.0240355× | [9.46946e-11, 3.77167e-10] | 20/24 |
| public v29 / v32 | 1.5448558× | [3.16397e-9, 3.92641e-9] | 24/24 |
| v34 / v32 | 0.9238577× | [−9.29146e-10, −7.27755e-10] | 0/24 |

Sources: `experiments/2026-09-16_stage1_k4feed/score.json`, `experiments/2026-09-16_diag_gap/score.json`, `experiments/2026-09-16_stage1b_rep/score.json`.

The v32 primary slice fails its 1.5× raw-error improvement target. Its positive secondary result is real within this DESIGN block, but does not license selecting the diagnostic v32_i contrary to the explicit branch rule. The prior Skeptic's paired v32-versus-v32_i correction stands.

**Noise clarification:** `4.7634e-9` is the average estimated noise of one P11 truth replica. The scorer averages the two replicas, so the average truth noise in these actual MSE rows is **2.381716943e-9**. The code correctly halves it (`local_sim/p2_design_score.py:105–109`) and applies unequal-cost noise correction; prose should identify which floor it quotes. The observed DESIGN MSE and its noise-corrected estimate are different estimands.

The primary stage-1 contract commit `909f3d2` is 17:38:45Z, well before the actual truth reservation/read at 19:29:06Z. It is therefore not invalidated by the separately wrong hand-supplied `ts=23:30Z`. The stage-1b contract likewise predates its actual read. Preserve those valid measurements while correcting timestamp provenance.

## 2. Concrete integration defect: receipt producer and consumer disagree

**Release-blocking, introduced by these repairs.** `scripts/run_mini_json.ps1:246` writes `"$Out.receipt.json"`: for `report.json`, this is `report.json.receipt.json`. But `local_sim/predict_public.py:325–331` strips the report extension and auto-discovers `report.receipt.json`.

Thus a valid new wrapper output is not automatically found by the projector. Its default source-binding checks then refuse the evidence; users may be tempted to weaken provenance or manually reconstruct receipts. The existing tests cover each half separately and miss this interface.

Repair one canonical convention, keep compatibility with existing receipts, and add a tiny cross-tool test that feeds the wrapper-produced report to `find_receipt`/`load_receipt` and source verification without explicit receipt overrides. This uses the fake launcher and takes seconds. Do not repeat a mini run to discover it.

The LOG's deferred note that the wrapper still needs runtime receipts is stale: commit `ba19866` added runtime receipt emission. Update status to the actual remaining filename/interface issue.

## 3. Driver portability: wrapper fixture failure under the current Codex environment

Fresh lightweight checks:

- `uv run python local_sim/test_certify_instrument_repairs.py`: **14/14 PASS**.
- `uv run python local_sim/test_predict_public.py`: **87/87 PASS**.
- `uv run python local_sim/test_run_mini_json.py`: **1/4 PASS, 3 FAIL** with inherited environment.

The third suite launches Windows PowerShell 5.1 (`test_run_mini_json.py:63–66`). Its child inherits a `PSModulePath` containing Codex's bundled PowerShell 7 modules before the Windows PowerShell modules. In that child, `Get-Command Get-FileHash` fails: it discovers the incompatible Utility module. Consequently `run_mini_json.ps1:115,136,235` cannot hash artifacts, and failed-run metadata or the success receipt is not completed. Raw stdout/stderr still survive.

**Controlled diagnosis:** rerunning the same four tests with only the child process's `PSModulePath` restricted to `%SystemRoot%/System32/WindowsPowerShell/v1.0/Modules` passes **4/4**. No system environment or repository file was changed. A direct Python-to-PowerShell `Get-Command` reproduction confirms the mismatch independently of the wrapper.

This does not show that the worker's earlier 4/4 report was false; its driver environment differed. Nor is `Get-FileHash` entirely new to the wrapper. It is an exposed cross-driver dependency that now affects additional receipt-writing paths. Fix or preflight the explicit shell/module pairing locally; do not change machine-wide module settings. Include a real, existing estimator fixture so hashing is exercised before launch, since the current fake suite names a nonexistent estimator.

## 4. Stage-0 identity improvements and remaining limits

The replacement of git-porcelain equality with source hashing is a material improvement. Tests confirm both intended cases: unrelated work does not invalidate a run, and editing an already-dirty dependency does. Residual-cap-disabled functional routing now keeps non-timing failures binding; the fixtures verify this. The projector now rejects missing rows, failed anchors, mismatched sources, and inconsistent meter paths.

However `dependency_identity()` is not immutable execution isolation (`local_sim/certify_candidate.py:1236–1330`):

- Static tracing follows plain `.py` imports rooted under `local_sim`; dynamic imports and subprocess dependencies need an explicit maintained root.
- Installed-package identity hashes the distribution's **RECORD metadata**, not all installed file bytes (`:1293–1302`). A patched package file without an updated RECORD would not be detected.
- Before/after hashing misses a dependency changed and then restored during execution.

These are limits to the new identity claim, not evidence that existing results were contaminated. For the next frozen release, use a content-preserving execution snapshot and explicit dependency manifest. Avoid invalidating cached results merely because unrelated files changed. Full certification and the four `-Full` preflight checks remain owed on the actual candidate.

The scorer's hand-supplied `--ts` requirement (`p2_design_score.py:64–67`) is preexisting. Multiple fresh reads—not just job 1463—carry invented schedule times marked `supplied_historical`, although their machine-created receipt timestamps preserve the real chronology. Use actual receipt timestamps for audit and future automatic timestamps for fresh reads; reserve historical timestamps for explicitly backfilled records. Do not rewrite historical lines to make them appear contemporaneous.

## 5. N1: exact numbers, limited inference, new registration defect

The saved cross-replica score formula `mean((p-r1)*(p-r2))` is an unbiased estimate of true-function MSE given independent replicas. The pooled arithmetic losses in `experiments/2026-09-17_n1_cycle/cycle_accuracy.json` reproduce:

| Affordable configuration | Width 64 | Width 96 |
|---|---:|---:|
| join_b, age 2, rank n/8 | −23.1079% | +98.3793% |
| join_a, age 2, rank n/8 | +7.2442% | +320.7246% |
| join_a, age 3, rank n/8 | −6.8575% | +86.4958% |

There are **four networks per width**, depth 12, full-rank P2 residual (`v32_dense`) and float64 dense-lineage calculations. This differs from the actual 1024×16 candidate. “Exact” here means uncompressed within that moment closure, not exact Gaussian-input expectation. Its mean true-function MSE is `1.83416e-4` at width 64 and `2.60969e-5` at width 96.

Uncertainty is large. Recomputed t95 intervals on the paired absolute MSE difference, expressed as a percentage of the fixed observed baseline mean for readability, are:

- width 96 join_b age2 rank12: **[−33.2%, +229.9%]**;
- width 96 join_a age2 rank12: **[+132.1%, +509.4%]**;
- width 96 join_a age3 rank12: **[−40.6%, +213.5%]**.

These are descriptive normalized difference intervals, not ratio confidence intervals and not live-shape predictions. A conservative stop after such an unfavorable exploratory screen is sensible. A claim that all affordable old-history compression is proven inaccurate at 1024 is not established. The sharp width dependence and sign reversals also make extrapolating `r/n` alone unsafe.

**Truth precision:** job 1526's log records replica disagreement MSE from `4.961e-10` to `2.380e-8` (width64) and `1.114e-9` to `6.368e-9` (width96). Half of disagreement estimates a single replica's noise; one quarter estimates averaged-truth noise. These are far below the tiny-shape closure errors here. The JSON should carry this metadata rather than require queue-log archaeology. This was a truth-bearing MC accuracy screen, despite `n1_tiny_truth.py:4` and parts of the LOG calling it label-free.

**Registration chronology is not as claimed:**

1. `workerq show 1526 --json`: submitted **01:59:49Z**, snapshot `d660239…` at **01:59:50Z**, started **01:59:52Z** on September 17.
2. The snapshot's `experiments/LOG.md` has no continuous-cycle registration entry, and the snapshot does not contain `local_sim/n1_cycle_accuracy.py`.
3. Commit `fb981a1`, which adds the N1 registration and accuracy script, is **02:00:27Z**, after truth sampling began. Job completion is **02:00:46Z**.

This establishes that the repository's claimed committed-before-truth ordering did not hold. It does not establish that the worker had inspected results before writing its rule. Keep the result as **exploratory, registration deficient**, and retract the stronger provenance claim rather than retrospectively manufacturing registration.

`n1_tiny_truth.py:35–40` also constructs literal seeds `90_000_019_000 + 97*n + net`, explicitly outside all registered blocks. There is no corresponding registry/ledger entry. Avoiding an existing block does not satisfy the seed-allocation requirement. Any future method-selection corpus must be allocated and registered before minting, even if small; do not retroactively relabel these eight networks as fresh evidence.

Resource discipline also needs correction: job 1526 declared **4 GiB RAM**, measured **5.139 GiB** peak, with 2 GiB VRAM and two CPUs. No job-specific pre-submission optimization review for this GPU workload was found in the LOG. Future jobs should use measured peak plus margin, with the required recorded review.

## 6. P4: useful counted savings, no general impossibility result

The nine rows in `meter_strassen_depth.json` confirm v32_p4 at depth5 costs **0.2802036389 B**, **17,483 operations**, with **6.0838 GiB Windows peak commit**. Depth6 costs **0.2749110745 B**, **20,519 operations**, with **9.7197 GiB Windows peak commit**. The decimal-GB labels in the original table are internally consistent; do not confuse them with GiB.

The measured depth6 spelling fails the 20,000-operation target and carries a strong memory warning. It still does not reach 0.27 B. Retain depth5 as the provisional exact-cost candidate and require its paired raw-parity check and actual Linux memory cap before release.

The metered spelling set does **not** establish that no exact transformation, alternative batching/liveness schedule, or combination with a changed architecture can reach 0.27. State “none of the tested P4 spellings meets the full gate.” Windows commit is accounting evidence, not an enforced Linux failure or pass. The existing meter compares every prediction to the first v31 depth4 arm, so its tiny relative differences are not a direct v32-versus-v32_p4 parity certificate either.

## 7. Required corrections to strategy inputs

At review start, the current packet's header retracts the 0.17 N1 gate and n64 proxy statement, but its body still repeats them (`STRATEGIST_PACKET_2026-09-17.md:62–69`). `relevance_n64.json` explicitly reports `proxy_informative=true`, `proxy_pays=false`. Correct the operative narrative rather than leaving contradictory instructions under a top-line disclaimer. Keep historical artifacts untouched.

The recipe-transplant negatives justify deprioritizing those tested combinations. They do not uniquely identify the K3 representation as the cause: closure interactions, prior-state divergence, precision and correlated truncation errors remain alternatives. A controlled same-state transition comparison is the appropriate next discriminator. A major-release gate should remain separate from this diagnosis; nothing in these reports justifies promising a 2× next submission.

## Recommended bounded follow-up

Before expensive work: repair receipt discovery and explicit PowerShell/module compatibility; run the cross-tool fixture; correct the packet's provenance/closure language. For scientific work: register a same-state transition and representation audit, preserving the public reference solely as the already-authorized diagnostic. Screen only a concrete mechanism once an observable discrepancy is isolated. If a supported bundle reaches the major-release target, freeze exact bytes once and perform the full release suite with enforceable Linux memory, public CONFIRM, and independent review. If it does not, report the failed target honestly and continue justified alternatives; do not change thresholds or spend a submission just to finish.
