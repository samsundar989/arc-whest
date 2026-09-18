# Validation and iteration audit — September 18, 2026

Scope: read-only source inspection and small CPU-only fixture checks. No truth arrays opened, no networks minted, no estimator predictions run, no seed claims, no heavy jobs, no submissions. The active September 17b batch was writing artifacts during this review; its partial results were not evaluated. This document is an infrastructure review, not an independent blind Skeptic verdict on that experiment.

## Assessment

The core instrument is substantially sound and should be retained. The repository can reproduce the competition's network generation, scoring arithmetic, weight representation, counted costs and functional lifecycle. It has a useful public-score forecasting method with one successful held-out submission comparison. It cannot reproduce server timing exactly or predict the next hidden suite's score with a calibrated confidence interval. Windows accounting alone cannot certify the Linux memory cap.

The main iteration problem is the accumulation of partially overlapping entry points and contract-specific verdict scripts, rather than a missing evaluator. A new concrete instance of that problem is present in the active stage-3 selector: it describes the required noise-corrected objective but gates the uncorrected one. Fix this bounded defect and consolidate adapters; do not restart the entire infrastructure project.

## Findings requiring action

### 1. P0 before relying on stage 3: `s12_decide.py` gates the wrong estimand

The active contract at `experiments/LOG.md:53002` requires an OFFICIAL-PRECISION adjusted score comparison for stage 3. `local_sim/s12_decide.py:12` repeats that requirement. However:

- Lines 88–91 construct `score = observed_mse * max(.1, FLOPs/B)`.
- Lines 98–109 read existing forecast summaries only into an output/display object. Those summaries are also comparisons against batch B's external control, not the required qoff control.
- Line 126 passes the observed score arrays to `competition_score.decide`.

With different costs, common labels do not cancel the bias: it is `(q_control-q_candidate)*truth_noise`. This is already correctly explained and implemented by `competition_score.noise_corrected_compare`. The official-precision selector must operate on corrected per-network scores, not observed scores or ratios copied from comparisons against another control.

**Synthetic end-to-end proof, run during this audit:** 24 fabricated rows, control MSE .011 and multiplier .2, candidate MSE .020 and multiplier .1, common averaged-label noise .01, 2^25 local samples and 1e9 official samples. The actual `s12_decide.py` reports screen PASS and gain 1.10. Applying the specified precision conversion gives gain **0.258437**, FAIL. The query-error fixture passed the independent S3 conditions. These deliberately exaggerated numbers prove the routing defect; they are not a claim that the real stage-3 verdict reverses.

**Minimal correction:** parse canonical `MlpRow` objects; validate complete, unique paired identities; obtain per-row official-precision MSE `mse - gt_noise + gt_noise*N_local/N_official` through one shared helper used by `noise_corrected_compare`; preserve failed-row scoring through `score_suite`; pass these arrays to `decide(gate=1.0)`. Save observed, true-function and official-precision intervals separately. Test unequal-cost reversal, equal-cost agreement, missing noise, failed rows and a comparison whose control differs from the batch's control. Recompute the verdict from already saved rows after review; no estimator rerun or new truth read is needed.

The generic `p2_design_score.py:323–325` likewise applies `--gate` to observed scores even while printing a forecast. That is not intrinsically wrong for an observed-score contract, but it is unsafe as an implicit default for new official-precision DESIGN selection. Make the estimand explicit in the small adapter and artifact. Keep the registered P13 release requirement intact: its observed gate AND noise-aware major-gain support are both required.

### 2. P1: stage-3 pairing and query-gate coverage are incomplete

`s12_decide.py` saves seed lists but never compares them. In the same synthetic fixture, changing every qE seed by +100 still produced PASS. `s9_decide.py:123–125` at least checks equal sorted seed lists. The canonical `competition_score.align_rows_by_identity` provides the stronger reusable check; use it instead of another handwritten scorer. Missing rows, duplicate rows, mismatched candidate identities and wrong shape should refuse before inference.

The S3 code at `s12_decide.py:127–129` checks final-row displacement and mean D3/D21 errors. It does not check the full E query or next-layer state displacement promised in the contract. `s10_parity.py:402` records D3, D21, mu_out and cov_out, but no explicit E error. Its cheaper-setting selector (`s12_decide.py:163–165`) calls `1 - all_rows_rel_vs_uncompressed` “retained query energy”; this is relative prediction displacement, not generally retained singular-space query energy. The existing numbers remain useful label-free diagnostics, but the full gate claim is not supported by those fields alone.

Correct the implementation to the existing contract, or explicitly resolve the measurement definition before further branching. Do not retroactively weaken the gate after seeing accuracy. At minimum document E's precise relationship to the measured contractions if it is algebraically redundant. Also choose qEd, rather than qoff, when qEd alone passes and the cheaper-setting criterion is meant to follow the successful arm.

### 3. P2: the new prediction sidecars are valuable evidence, not yet a reusable execution cache

`design_sidecar.py` stores final predictions and verifies row hashes. This is exactly the data needed for error-change decomposition and avoids repeated predictions for later analysis. But `p2_design_score.py:223–252` always executes every requested arm, including the control. There is no verified cache lookup or partial-run resume. `write_manifest` directly truncates and rewrites the manifest rather than publishing an atomic replacement. A crash can leave durable rows but a broken manifest.

Before adding reuse, bind candidate bytes, ordered network identities, truth identity, lifecycle/score/dependency hashes, weight mode, complete configuration, package versions and numerical environment. Current sidecars bind the scorer and sidecar module but do not include the complete evaluator/scorer dependency closure or numerical environment. `verify` also skips source checks when a named source is missing, so a verifier must distinguish “stored hash known” from “source independently verified.” These are limits of a future reuse claim, not evidence that current rows are wrong.

Use the existing `evidence_journal.py` durability patterns. Add a small cache adapter for complete DESIGN runs and frozen controls, with an explicit no-reuse/no-resume policy for single-use DECIDE unless its contract authorizes recovery. Do not create another scorer. Never reuse wall-time observations across incompatible load conditions.

## What already matches the competition

| Component | Evidence and practical limit |
|---|---|
| Live shape and score | `official_config.py` and `competition_score.py` carry Phase 2 1024×16, B=2^41, lambda=0, multiplier floor .1 and failure handling. `score_suite` preserves mean-of-per-network-products; factorized mean MSE × mean cost is not a substitute. |
| Canonical networks | `mint_provenance.py` follows the independent weight/truth/estimator seed streams. `phase2_mint_parity.py` also compares against published Phase-2 weight bytes. No new parity run was needed for this audit. |
| Truth precision | `p2_truth.py` records IEEE float32 forward math with TF32/autocast disabled, float64 accumulation, two independent replicas and measured disagreement. The scoring path halves per-replica noise when averaging replicas. The local Monte Carlo draws need not equal official draws; the distribution and arithmetic policy must be documented. |
| Grader representation | `lifecycle_eval.build_weights` defaults to immutable, flopscope-backed float64 weights, and evaluation includes setup and prediction materialization. This closes the previously costly bare-numpy and float32-only gaps. |
| Package stack | Fresh metadata read: local whestbench **0.16.1**, flopscope **0.12.1**, NumPy **2.2.6**. These match the current server versions reported by the main review agent's Chrome inspection. Local Python is **3.10.19** versus server **3.10.21**. OS, BLAS and process timing still differ. |
| Counted FLOPs | `predict_public.py:128–146` distinguishes subprocess (+2^25 for these K3 controls) from grader-twin (zero increment). F21 and v31 paired local/server observations support this. Do not add a universal offset to all paths; `official_config.py` still carries a historical +2^24 model used by older callers. |
| Public forecast | Existing v31 replay projects **8.713276e-9** against graded **8.752030e-9**, about **0.45%** error. `predict_public.py` now labels the interval as a transformed conditional local difference interval, not a hidden-board prediction interval. One successful forecast is encouraging, not general half-percent calibration. |
| Memory | Candidate-specific `linux_memory_gate.py` is the relevant enforced 8-GiB check. Windows peak commit is useful accounting only. One tested input is not a proof for all inputs when state size or control flow is data-dependent. |
| Failure cliffs | Cap-disabled local accuracy is separate from resource feasibility. Operation count <=22,000 is a family-calibrated residual planning guard, not a server execution measurement. Wall ×1.9 from an uncontended median remains advisory. |

The fresh lightweight regression checks in this audit passed: **87/87** public projector checks, **10/10** mini wrapper tests, **14/14** certification repair tests. The prior receipt-name mismatch and PowerShell module-path failure are resolved, including cross-tool tests. The old demand that all of `local_sim/` remain quiescent is also superseded: certification now compares the candidate's dependency content identity, so unrelated edits do not invalidate a run. Do not repeat those repaired defects as current blockers.

## A shorter evidence path without weaker decisions

1. **Build loop:** candidate import/contract, independent tiny algebra oracle, targeted affected invariants; one live-shape label-free cost/resource observation if the implementation's cost is uncertain. No public mini, full certification or truth mint per spelling.
2. **DESIGN batch:** one committed mechanism contract with explicit conditional branches; one new DESIGN panel per scientific slice; control evaluated once; candidate rows and predictions streamed. Use the official-precision estimand for cross-cost selection, report uncertainty and ties, and select at most one finalist. Do not require a 2× gain from each isolated component when the release target is a composition.
3. **Freeze and DECIDE:** exact candidate bytes, one unused block, registered paired arithmetic gate and declared noise-aware condition. A failure is recorded, not retuned on that block. Existing P13 contract already has this structure.
4. **Release:** candidate-specific conformance, frozen-source certification, enforced Linux memory, one source-bound mini CONFIRM, public projection using compatible cached v31 control, independent Skeptic and the existing full preflight obligations. Reuse compatible raw evidence; do not rerun a smoke already strictly covered by a complete source-bound mini solely for a second report title.
5. **Submission:** present the exact package, projected gain, uncertainty and remaining resource risks for Samuel's per-submission approval. A major improvement can be tested locally; exact hidden-suite score and server tail timing cannot be guaranteed before submission. There is no scientific reason to consume submissions on every intermediate arm, but no reason to defer a qualified frozen major candidate for an unrelated infrastructure cleanup.

Implement this as a small command/manifest over existing primitives, not a framework. The release artifact should say which exact bytes were evaluated, the primary question and tier, observed and official-precision paired results, counted cost, noise, failures, Linux memory, timing evidence domain, and which obligations remain. Historical ledgers stay append-only.

## Priority and stop rules

**Before active stage-3 conclusions:** fix and independently review the estimand/pairing adapter; audit S3 coverage. Preserve current raw batch data. This is an IMPORTANT instrument change requiring the repository's written prediction, affected tests, preflight and Skeptic; no new model evidence needs to be generated to repair arithmetic from saved rows.

**Before the next release:** freeze exact source bytes once and verify source-bound receipts, authoritative Linux memory and candidate-specific checks. Keep the already working projector and wrapper. Resolve historical-vs-current cost offset naming only where a release caller actually uses it.

**After the slice:** add verified frozen-control reuse and atomic manifests; consolidate the active entry points in a short current runbook. Keep historical material available, but remove superseded instructions from the operational path. The present repository's long historical documents should not make an agent repeat an old failure or obsolete gate.

The infrastructure can support a major jump. It cannot manufacture the missing raw-accuracy improvement, and additional certification layers are not the next scientific breakthrough.

## Reproduction appendix: stage-3 adapter defects, without real data

From the repository root, run the following in PowerShell. It creates only temporary synthetic JSON files, invokes the existing verdict adapter, and removes its temporary directory on exit. It neither executes an estimator nor opens a dataset. The intentionally exaggerated noise makes the routing error obvious. The expected output for the source reviewed here is `observed_screen_pass=True`, `official_pass=False`, official gain about `0.258437`, and `mismatched_seeds_screen_pass=True`. After repair, the mismatch case should refuse and the official screen should fail. No real-stage verdict reversal is implied.

```powershell
@'
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.append('local_sim')
import competition_score as cs

names = {
    'qoff': 'estimator_k3_v39_v37_a4r256_p4d5_qoff.py',
    'qE': 'estimator_k3_v39_v37_a4r256_p4d5_qE.py',
    'qEd': 'estimator_k3_v39_v37_a4r256_p4d5_qEd.py',
}
with tempfile.TemporaryDirectory(prefix='s12_synthetic_') as temporary:
    root = pathlib.Path(temporary)
    arms = {}
    for name, candidate in names.items():
        mse, multiplier = (.011, .2) if name == 'qoff' else (.020, .1)
        arms[name] = {
            'path': candidate,
            'rows': [dict(seed=i, mse=mse, flops_used=multiplier * 2**41,
                          failed=False, nonfinite=False, gt_noise=.01)
                     for i in range(24)],
        }
    score = dict(arms=arms, comparisons={}, n_samples_per_replica=2**24)
    query = {'arms': {}}
    for name in names:
        error = 2.0 if name == 'qoff' else 1.0
        query['arms'][name] = {
            'final_row_rel_vs_uncompressed': error,
            'all_rows_rel_vs_uncompressed': error,
            'c_over_b': .2,
            'per_layer': [{'D3': error, 'D21': error}],
        }
    score_path, query_path, result_path = [
        root / name for name in ('score.json', 'query.json', 'result.json')]
    query_path.write_text(json.dumps(query), encoding='utf-8')
    command = [sys.executable, 'local_sim/s12_decide.py',
               '--score', str(score_path), '--qe-a4', str(query_path),
               '--out', str(result_path)]

    def run():
        score_path.write_text(json.dumps(score), encoding='utf-8')
        subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(result_path.read_text(encoding='utf-8'))

    observed = run()['comparisons']['qE']['SCREEN_PASSED']
    # gt_noise is for the averaged truth: two replicas of 2^24 samples.
    official_noise = .01 * (2**25 / 1e9)
    control_scores = [(.011 - .01 + official_noise) * .2] * 24
    candidate_scores = [(.020 - .01 + official_noise) * .1] * 24
    official = cs.decide(control_scores, candidate_scores, gate=1.0)
    for row in score['arms']['qE']['rows']:
        row['seed'] += 100
    mismatch = run()['comparisons']['qE']['SCREEN_PASSED']
    print(json.dumps({
        'synthetic_only': True,
        'observed_screen_pass': observed,
        'official_pass': bool(official.passed),
        'official_gain': float(official.gain),
        'mismatched_seeds_screen_pass': mismatch,
    }, indent=2))
'@ | .venv/Scripts/python.exe -
```

The fixture deliberately leaves `comparisons` empty because the current adapter permits it and never consumes official-precision summaries for its S1 decision. The correction should reconstruct and validate the actual paired rows, not merely require a display field or transplant a forecast ratio from the batch's different control.
