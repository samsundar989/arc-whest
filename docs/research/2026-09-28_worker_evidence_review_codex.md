# Worker evidence review, 2026-09-28

Independent review of worker changes after `5138388`, through `3b7a74a`, and the existing uncommitted v72 ledger entry. PLAN rev 15 was read first, then current AGENTS and RESULTS. This review did not train, evaluate a network, launch/cancel a queue job, submit, or designate anything. Small CPU checks read saved predictions/artifacts and tested acceptance logic with mocks. The root agent independently inspected Chrome; its observations are attributed below. Existing experiment artifacts were preserved.

## Assessment

The worker delivered real, modest improvements and useful new structural evidence. v70 and v71 have complete saved graded reports; v72's precise ledger result is corroborated by the root agent's live Chrome observation. The current best is **v72 #332766: adjusted 4.620737758949154e-9, raw final-layer MSE 2.1518525734620654e-8**. The final-layer label floor is approximately **7.7e-11**, only 0.36% of this raw error. v72 improves the v68 aggregate by 1.767%, including a 0.346% improvement over v71. These aggregate comparisons are corroborated by paired DEV evidence, not offered as standalone ratios of means. Sources: `experiments/submissions.jsonl:34`, `:36`, `:37`, `:38`; `RESULTS.md:223`–`:227`; floor contract `AGENTS.md:89`.

This is not yet a winning architecture. The root agent's Sep 28 Chrome result-page read put J2W #332700 at **1.748e-9**, raw 1.616e-8, cost 0.1082B, and us at rank 36. The leaderboard's 1.70e-9 display is too coarsely rounded for this comparison. Even reducing v72's cost to the 0.1 scoring floor at unchanged raw accuracy yields approximately **2.152e-9**. At unchanged cost 0.214733, matching 1.748e-9 requires raw MSE approximately **8.14e-9**, a 62% reduction. Both accuracy and cost still matter. The new all-distinct third-cumulant evidence identifies a meaningful research direction, but its oracle gains are not executable gains.

The root also paired 50 public names in Chrome: J2W wins on adjusted score, raw error, and cost on every one. Mean paired score difference (ours minus leader) is 2.8742e-9, t95 [2.75639e-9, 2.99201e-9]; raw difference 5.36e-9, t95 [4.88691e-9, 5.83309e-9]. Mean per-network leader/ours ratios are 0.37975 adjusted, 0.75376 raw, 0.50424 cost. These are rounded DOM values; the raw final-layer floor remains approximately 7.7e-11. Root owns the saved paired CSV and summary.

## What was independently checked

- Current source bytes, release `estimator.py`, archive inner `estimator.py`, ledger inner-source hash, and WSL artifact source hash match for v70, v71, and v72. Archive hashes also match the ledger. Each archive contains only `estimator.py`, `LICENSE`, and `manifest.json`.
- AST comparison against current v68 proves v70/v71 change only nine parameter-table assignments: `_LAM_GAIN`, `_THETA_GAIN`, `_FEED_GAIN`, `_TT_GAIN`, `_T1_GAIN`, `_CU_GAIN`, `_CB_GAIN`, `_Q31_COEF`, `_CORR_B64`. Function bodies are unchanged. This supports inherited code-path cost/wall evidence.
- All three stored DEV runs have 100 predictions, no failed nets, empty cost/setup error strings, zero materializations, and WSL PASS under f64 read-only weights, 8,192 MiB address-space cap, three lifecycle repeats. Their final source hashes match, despite the evaluator defect described below.
- Metadata-only panel check passed all 11 checks; the historical `holdout` route raises the intended retirement error. No large datasets were opened or hashes recomputed.
- Recomputed twin/export MSE parity from the saved 100-net predictions and saved per-net twin MSEs. This loaded small prediction arrays only; no inference.
- Computed paired score differences on the common 50 public networks in the v68/v70/v71 saved graded reports. Public values are rounded by the site, so precision beyond the shown intervals is not meaningful.

These checks establish consistency of the retained evidence. They do not replace a private re-run or prove that internal numerical repairs never fire.

## Submission evidence

| candidate | tested source SHA256 | package SHA256 |
|---|---|---|
| v70 | `45f26986a950b995381f9f5f44869fc474a0cf4af895884f998f962795815e8d` | `073b58f47439ac9e066ccb41ae47f05c5d6c6b2eacda287eb88eff0882c9084a` |
| v71 | `dde9a8631c87fa8513656e38e042836cfbbe748f9ec0a6beba2c8e482d650b68` | `ece5f82d9d070d02c2780e53342d9dae0319c78fa24b19ed182f6c5dfaf96cbb` |
| v72 | `3b6cc39790295d3be60a88f9dd9053915f695bdf0a0da2cf2e99587cce692bac` | `1da5ca8f81b4ef834cad2b6b673f2cdfc26120ecbb898199ff2f1324c8d7e8cb` |

Packages: `experiments/2026-09-27_v70/v70-canary.tar.gz`, `experiments/2026-09-27_v71/v71-canary.tar.gz`, `experiments/2026-09-28_v72/v72-canary.tar.gz`. Source/manifest validations are recorded as passed in ledger lines 36–38; this review verified archive contents/hashes but did not rerun the official validator.

| evidence | v70, step 368 | v71, step 736 | v72, transport gate |
|---|---:|---:|---:|
| DEV raw final MSE; floor ~7.7e-11 | 2.263189e-8 | 2.244217e-8 | 2.234916e-8 |
| DEV projected public | 4.6819e-9 | 4.6426e-9 | 4.6234e-9 |
| graded adjusted | 4.665633e-9 | 4.636794e-9 | 4.620738e-9 |
| graded raw final MSE; floor ~7.7e-11 | 2.172776e-8 | 2.159346e-8 | 2.151853e-8 |
| meter FLOPs | 472,199,315,449 | 472,199,315,449 | 472,202,879,721 |
| operation calls | 22,701 | 22,701 | 22,992 |
| WSL VmPeak, MiB | 6,711.99 | 6,714.00 | 6,702.67 |
| paired DEV wins vs immediate parent | 74/100 | 79/100 | 70/100 |
| paired adjusted difference, t95 | [1.728e-11, 4.173e-11] | [3.101e-11, 5.046e-11] | [8.898e-12, 3.098e-11] |

Sources: `experiments/2026-09-27_v70/v70_dev.json:8`, `:22`, `:33`; `experiments/2026-09-27_v71/v71_dev.json:8`, `:33`; immediate-parent pair in `experiments/2026-09-27_v71/v71_vs_v70.json:22`; `experiments/2026-09-28_v72/v72_dev.json:8`, `:22`, `:33`, `:183`, `:202`. The v71 secondary pair artifact omits a fresh memory run; its candidate hash and score match the main v71 artifact that contains the memory evidence. This is combined evidence, not a second complete acceptance run.

v70/v71 saved reports have 100 completed and zero failed, with adjusted/raw aggregates at lines 6–7 and completion counters at lines 38–40:

- `experiments/run_reports/v70_canary_332737_graded_report.json`
- `experiments/run_reports/v71_canary_332753_graded_report.json`

The 50-network public paired check yields v70 vs v68: 34/50 wins, mean adjusted improvement 3.816e-11, t95 [2.212e-11, 5.420e-11], 0.811% relative improvement. v71 vs v70: 37/50 wins, 2.884e-11, t95 [1.819e-11, 3.949e-11], 0.618%. The full 100 raw per-net scores are not public; aggregate and public-pair claims must remain distinct.

Server wall readings for v70/v71 are recorded in `RESULTS.md:224`, `:226`: medians 80.65/83.25 s and maxima 84.73/94.45 s; residual maxima 0.2372/0.2479 s. Meter/server FLOPs differ by only 619 operations of approximately 4.72e11, not a practically relevant calibration offset; the prose “exact” is approximate here.

For v72, the root agent independently observed Chrome submission #332766: rounded adjusted 4.621e-9, raw 2.152e-8, 100 COMPLETED, zero failed; wall median **83.30295 s**, maximum **86.2848 s**, p90 as ordered item 89 **84.6671 s**. The implied noisy maximum is **103.295658 s** using median ×1.24. The precise ledger score agrees with the rounded site read. No full v72 graded report was saved when this review began; the root owns its closeout. This observation is not represented as a report file that this subagent read.

The ledger has three Sep 28 canaries, safely inside the five-total-per-UTC-day limit (`experiments/submissions.jsonl:36`–`:38`, `docs/PLAN.md:75`). No separate backup slot was assumed.

## Acceptance defects and repair

### Fixed: stale memory evidence could pass

At worker HEAD, `memory_gate` reused `experiments/_ev/mem_<candidate>.json`, ignored subprocess return code, and loaded any existing JSON. A failed or unavailable new run could therefore return a previous PASS. `gate_table` checked only status, not candidate hash. A tiny mutation check reproduced acceptance of an all-zero wrong source hash. This was a future acceptance defect, not evidence that any of v70/v71/v72 failed its memory gate: all their stored hashes independently match.

At the root agent's request, this review changed **only `tools/ev.py` and new `tools/test_ev_memory_gate.py`** as executable files:

- Each memory invocation uses a private temporary output path. A stale destination cannot satisfy it.
- Nonzero subprocess return, launch failure, missing or malformed fresh JSON, wrong source SHA256, or a candidate modified during the memory run fails closed.
- Fresh matching results are saved to the normal destination; old artifact bytes are preserved on launch/result failure.
- Evaluations carry the full `candidate_sha256`, and memory acceptance requires that exact hash as well as PASS. Older saved reports without the top-level full hash must not be treated as freshly accepted by replaying `report`; their evidence can be reviewed against the source, as done above.
- The best-graded default was advanced to exact v72 #332766 after the root's Chrome verification.

Implementation anchors: `tools/ev.py:453` (`memory_gate`), `:560` (full hash comparison), `:600` (memory gate verdict), `:607` (evaluation source hash), `:654` (stored full hash), `:740` (latest graded default). The new test file covers stale PASS, nonzero exit with/without a fresh PASS, missing/malformed JSON, wrong/missing hash, changed source, valid fresh evidence, launch error, and incomplete DEV/pair/memory gates.

Validation: **13 CPU-only unittest tests passed**, `scripts/verify_portability.py` **102 checks / 0 failures**, and `git diff --check` passed. No costly memory gate was rerun. No existing experiment artifact was rewritten.

### Still manual: measured-parent wall acceptance

The evaluator's generic absolute model reports a noisy wall of 114.22 s for v72 and treats it as advisory against 115 s. PLAN requires **105 s for a new A code path**, using `(measured parent median + wall-law delta) ×1.24`; data-only A candidates inherit the approximately 115 s policy and B is 95 s (`docs/PLAN.md:73`). The stored v72 record correctly applies the new-code-path formula: 83.25 s plus about 0.1 s, then ×1.24 ≈103.4 s (`RESULTS.md:227`). Live v72 confirms 103.30 s implied noisy. Thus v72 meets the actual policy, but the evaluator alone does not certify it.

Rather than introduce an untested wall-classification/calibration interface, the report now explicitly says **manual PLAN §5.8 wall evidence remains required** (`tools/ev.py:573`, `:713`). A numeric advisory PASS must not replace that check. A mocked old row with a 1000 s advisory wall still had `accepted=True`; this is why that flag means only the named checkable gates, not full permission to submit.

### Inherited finite-value repair limits the “zero failures” claim

The twin `_step` replaces nonfinite means by zeros (`tools/twin/twin.py:381`–`:385`). The shipped estimator does likewise within the layer loop and on the final output (`estimator_k3_v72_gt.py:4365`, `:4393`, `:4420`), and sanitizes several intermediate arrays. `_predict_worker.py:177` checks finiteness only after `predict` returns; WSL also checks final finiteness (`local_sim/linux_memory_gate.py:285`). These repairs predate this worker and are shared by twin/export, so this is not a newly introduced parity discrepancy.

No saved evidence shows a repair firing on these networks. However, “0 failed” proves the external contract passed; it does not prove every internal state remained finite. A future bounded diagnostic should count nonfinite values immediately before the existing repairs and verify finite training loss/gradients, without changing the candidate's mathematics during attribution. Do not silently reinterpret internal zeros as a validated predictor.

## Parity threshold change

PLAN changed its DEV-mean export tolerance from 1e-4 to 1e-3 after v71 measured 1.9e-4 (`docs/PLAN.md:67`, commit `2801783`). This is a post-observation threshold change and should remain explicit. Its numerical justification is plausible: independent recalculation from the saved small arrays gives:

| candidate | relative difference of panel means | p95 absolute per-net MSE difference | maximum absolute per-net MSE difference |
|---|---:|---:|---:|
| v70 | 5.785e-5 | 0.2293% | 0.3218% |
| v71 | 1.877e-4 | 0.2145% | 0.3101% |
| v72 | 1.190e-4 | 0.1891% | 0.5063% |

All MSEs here refer to the final layer with floor ~7.7e-11. These differences are compatible with the documented float32 backend effects and are much smaller in the panel mean than the observed gains. The revised panel-mean bar is not a per-neuron or per-network equality proof: averaging can conceal compensating errors. Maintain identity parity and same-backend structural checks for new mathematics, and keep the measured per-net distribution alongside the mean.

Inputs: `experiments/2026-09-27_v70/twin_dev_step 368.json:1`, `twin_dev_step736.json:1`, `experiments/2026-09-28_v72/twin_dev.json:1`; predictions under `cache/preds/<full source SHA256>/dev/n100.npz`. AST data-only checks support v70/v71; gate identity/full-rank algebra checks support v72 (`RESULTS.md:307`–`:316`).

## Training attribution and stopped work

### S2t3 versus continued TRAIN-only fitting

The evidence supports “additional optimization explains the observed gain; there is no observed advantage for adding d8b at the tested checkpoints.” It does **not** cleanly isolate a zero effect from d8b:

- The mixed run uses a 736-step cosine; TRAIN-only control uses 784 steps. The control's learning rate is approximately 10% hotter at step 368, acknowledged before it ran (`experiments/2026-09-27_S2t3ctl/PREDICTIONS.md:8`–`:10`).
- At steps 230/368/506, TRAIN-only has lower VAL MSE than mixed. At 368 it is 2.240008e-8 versus 2.246946e-8; at 506,2.234042e-8 versus2.236140e-8. Final mixed 736 reaches 2.232939e-8; there is no matched final control (`RESULTS.md:232`–`:243`). All are final-layer MSE with floor ~7.7e-11, on the repeatedly selected VAL panel.
- The control died at approximately 530 from ENOSPC. The written stop was after 368, so it consumed extra work beyond its own stop condition (`PREDICTIONS.md:18`, `RESULTS.md:234`). A disk failure is not scientific convergence.

`RESULTS.md:245`'s “d8b adds nothing measurable” is stronger than this incomplete, schedule-mismatched experiment. No need to rerun the entire expensive control just to defend prose: qualify the attribution. The actual v70/v71 DEV and graded gains remain valid regardless of whether data or steps caused them.

### Gates

The gate-only stages freeze all old leaves and initialize gates at identity (`tools/twin/twin.py:95`–`:99`, `:533`–`:550`). G1tg transport improves VAL from 2.232939e-8 to 2.215950e-8 at step 84 (0.76%), then stops at 168 (`experiments/2026-09-27_gates/G1tg/train.log:1`, `:88`, `:176`). S3c, existing leaves from the same initial checkpoint, is worse than initialization at every recorded validation point, best 2.235038e-8 at 84 (`experiments/2026-09-28_S3c/train.log:1`, `:30`, `:59`, `:88`). All these final-layer MSEs have floor ~7.7e-11. This is useful specificity evidence for the tested optimizer recipe, not proof that no further old-leaf optimization can help.

The birth-only G1bg job also completed. Its best checkpoint is **step 112, VAL 2.213009e-8**, approximately 0.13% below transport's best; its last 196 value 2.213729e-8 is not the best (`experiments/2026-09-27_gates/G1bg/train.log:117`, `:204`, `:205`). At review start it had no exported DEV/package/graded evidence, so it is an inexpensive candidate to check later, not an established next winner or a significant leap.

Full-rank algebra proves compensation preserves targeted marginal slices; shipped P2 rank 4 leaks 5–30% of transport-gate effect into K3/K21 (`RESULTS.md:313`–`:315`). The graded gate gain is real, but do not claim it is a perfectly isolated change to all-distinct information at the shipped rank.

## Where the significant remaining opportunity lies

The new B1/B3 evidence is stronger than another nearby parameter sweep, but it must be interpreted narrowly:

1. On 20 FIT/TRAIN networks with the older S3/v68 accuracy path, all-distinct (AD) projected third-cumulant relative error grows from approximately 0.043 at the label floor in the first layer to approximately 0.25 at layer 14, versus approximately 0.003 noise. Exact-input mode does not remove it. It therefore reflects the carried third-order approximation, not only drift in its input means/covariances (`RESULTS.md:253`–`:269`).
2. Replacing the AD projection by labels at every layer cuts median final MSE 33.1%; replacing full D3+D21 cuts 46.9%. Repeated-only replacement worsens 4.8%; correcting compression mismatch alone improves only 3.4%. Scalar/per-neuron shrink captures considerable tensor-error energy yet worsens final MSE 13.6–15.8% unless the already-fitted compensation is revisited. The component outside the present model's span gives 15.9% oracle improvement (`RESULTS.md:271`–`:284`). Base raw final MSE is approximately 2.15e-8; its floor is~7.7e-11.
3. The D21 error matrix has concentrated singular energy at depth: rank 64 captures approximately 81%, 88%, 92% at layers 8, 11, 14. Yet these are **label-oracle singular vectors**, not features available to the submission. Low matrix rank of D21 does not prove the whole symmetric third-order tensor has a cheap propagatable representation (`RESULTS.md:286`–`:305`).
4. The reported rank 32 oracle's 18.9% median gain includes a **full D3 AD oracle fix in every arm**. Rank 0 already gains 9.5%; rank 32 supplies approximately 9.4 additional percentage points in this specific combined intervention. Do not advertise 18.9% as a standalone rank 32 D21 gain or add it to another ceiling. The full combined oracle gains 35.3% (`RESULTS.md:300`–`:305`).

The right next discriminator is whether the missing, score-relevant projection can be produced from available state/weight/covariance features with affordable contractions and maintained through layers. A compact label SVD is diagnostic only. The root's proposed bounded feature-basis probe addresses that gap before committing to a new online compressor or large fitting campaign. Even a 33% accuracy improvement at current cost projects around 3.1e-9, still above the current 1.748e-9 leader; the eventual design also needs a cost improvement or a larger independent accuracy mechanism.

## Panels, reproducibility, and resource state

The panel repair is correct and important: DEV mini 100 is never fitted but repeatedly selected; TRAIN full 0–899 has 900 nets; VAL full 900–999 has 100 and is used for checkpoint selection; the historical full 100–999 panel contains 800 TRAIN nets. There is no untouched local test. Code refuses ambiguous `holdout`; metadata checks pass (`docs/PLAN.md:36`–`:46`, `tools/panels.py:13`, `:201`–`:221`). A current RESULTS header still described HOLDOUT as never fitted (`RESULTS.md:14`) at review start; PLAN and the repaired route take precedence. Older historical results should not be relabeled as independent tests.

Current docs were not fully closed out at worker HEAD: PLAN/CURRENT_STATE still showed v68, RESULTS lacked the v72 graded row, calibration prose still said eight observations despite later entries, DESIGNATION was not updated, and v72's full graded report was missing. These are continuity gaps; root owns reconciliation. No stale strategic pre-Sep21 document was used to choose the research direction.

There remains a workload-policy contradiction: current AGENTS rule 6 and its command notes exempt evaluations from workerq, while the user's machine policy requires **all substantial GPU/RAM/CPU workloads** through the queue. Follow the explicit machine instruction and record optimization/resource analysis before submitting heavy work; do not bypass the queue because a GPU looks idle.

Read-only state at initial inspection: no arc-whest job was running/queued; G1tg 2261, rank 2262, S3c 2267, G1bg 2268 had succeeded. Later during root integration, new arc-whest 2292/2293 were queued; this subagent did not launch either. Completed measured footprints:

| job | peak host RAM | peak VRAM | runtime |
|---|---:|---:|---:|
|2261 transport gates|10,822.8MiB|7,682MiB|5,824s|
|2267 S3c|10,828.8MiB|7,494MiB|3,069s|
|2268 birth gates|10,835.2MiB|7,575MiB|6,059s|
|2262 rank oracle screen|10,051.7MiB|13,287MiB|677s|

These are workerq measurements, not hard runtime limits. In particular the old rank diagnostic reached 12.98 GiB VRAM, so a new 12 GiB request needs a defensible streaming analysis rather than inheriting that envelope blindly. The root's new probe is designed to avoid the old state cache.

At the initial disk check C: had 199,706,091,520 bytes free (approximately 186 GiB). Workerq's snapshot tree held 42 directories and 68,808 files totaling 32,370,675,460 logical bytes (approximately 30.15 GiB); this is logical length, not a hardlink-aware allocated-size measurement. No snapshots or artifacts were removed. Earlier ENOSPC failures make checkpoint/snapshot footprint relevant even when current free space is ample.

## Ownership and remaining work

This subagent owns the new review document, the focused `tools/ev.py` repair, and new `tools/test_ev_memory_gate.py`; no commit was created. Root owns integrated documentation, live v72 closeout, and any new diagnostic jobs. Before another candidate submission, retain full-hash memory evidence, use v72 as the paired incumbent/best graded threshold, record the manual PLAN wall comparison, validate/package/hash-match the final bytes, and check the day ledger. The reviewed worker results warrant continued investigation; they do not yet establish a winning solution.
