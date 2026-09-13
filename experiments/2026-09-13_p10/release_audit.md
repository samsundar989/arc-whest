# F21 release-gate audit — 2026-09-13

Read-only, independent release audit. This is **not** a P10 result review or promotion Skeptic verdict. No P10 truth, seed, public label, or vault data was opened. The candidate remains SHA-256 `d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94` (`docs/CURRENT_STATE.md:8-24,46-72`). Release work below is conditional on a valid fresh P10 DECIDE and an independent result Skeptic (`experiments/LOG.md:48293-48305`; `AGENTS.md`, Rule 2).

## Post-DECIDE checks on the exact candidate

1. Verify the source and prepared tar member still match the frozen SHA; there is no separate `release/estimator.py` path in the current tree. Preserve the parent SHA `052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58`, the one-hunk `_dslice21` diff, package manifest, and ARC attribution. The existing source/package audit covers the narrow own-parent delta; it does not prove the parent's full ancestry or organizer eligibility. The package receipt still says `validated_for_submission: false` (`experiments/2026-09-12_f21/f21_provenance_audit.md:9-15`; `package_receipt.json:2-23`; `parent_to_f21.diff`). A separate participant license/grant is needed if this tar is also intended as the later public prize release; that issue does not itself block technical validation of the current submission tar.
2. Run full candidate certification, with no `--quick`, `--skip-smoke`, or `--memory-only`, consuming the accepted exact-byte Linux 8,192-MiB report. The certifier checks validator, shape/lifecycle, grader-twin f64-readonly FLOPs and operation count, memory-report bindings, and a **two-row capped official subprocess smoke** (`local_sim/certify_candidate.py:8-48,1211-1241,1737-1773,2090-2115`; `experiments/LOG.md:47737-47755`). Suggested workerq command after recording/rechecking the workload optimization review and live capacity:

   ```powershell
   workerq submit --project arc-whest --ram 20 --vram 0 --cpus 6 --gpus 0 --live-worktree -- `
     C:/Users/samsu/Documents/arc-whest/.venv/Scripts/python.exe local_sim/certify_candidate.py `
     --candidate estimator_k3_strassen_L4_f21.py `
     --memory-report experiments/2026-09-12_f21/f21_linux_cap_v6.json `
     --json C:/Users/samsu/Documents/arc-whest/experiments/2026-09-13_p10/cert_F21_full.json
   ```

3. Run a **separate, full 100-row capped** official Phase-2 mini, with the default 0.4-s residual and 120-s wall caps. Record the complete JSON/workerq receipt and require `n_failed_mlps: 0`; a no-cap run is CONFIRM-only diagnostic and cannot replace this release gate (`AGENTS.md`, hard rule 6; `experiments/LOG.md:47756-47771,48297-48305`; `docs/VALIDATION.md:0-35`). The preregistered 20-GiB/6-CPU envelope is grounded in the observed full-mini harness peak (`experiments/LOG.md:47768-47771`; `docs/SETUP_WINDOWS.md:218-221`).

   ```powershell
   workerq submit --project arc-whest --ram 20 --vram 0 --cpus 6 --gpus 0 --live-worktree -- `
     C:/Users/samsu/Documents/arc-whest/.venv/Scripts/whest.exe run `
     --estimator estimator_k3_strassen_L4_f21.py `
     --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 `
     --split mini --runner subprocess --detail full --json
   ```

4. Retain the existing `preflight.ps1 -Full` 1092 green 16/16 for the unchanged shipped-file/instrument scope, and rerun affected steps if source/instrument changes warrant it. **It does not certify F21**: preflight hardcodes `estimator.py` in candidate-relevant steps; its `-Archive` only changes the shipped-file identity comparison (`scripts/preflight.ps1:8-16,85-107,129-175`; `local_sim/certify_candidate.py:8-16`). The already accepted F21 Linux cap v6 memory-only consumer is one section, not a full certificate (`experiments/2026-09-12_f21/cert_F21_memory_only_v6.json:4-18,163`).

## Cost evidence and open refusal

The exact f64-readonly local ledger records **928,457,637,478 counted FLOPs and 14,091 operations** (sum of `result.ops[*].calls`) for F21 at 1024×16. This is below the working 22,000-op residual planning gate; at the conservative 13.4 µs/op it predicts 0.1888 s and 2.12× headroom against 0.4 s. It is a planning model, not a graded-server observation (`experiments/2026-09-12_f21/meter_F21_f64.json:3-23`; `local_sim/official_config.py:376-380,499-525`; `docs/VALIDATION.md:445-469,512-533`). The stored generic `p2_flop_anchor.py --gate` proves op-by-op meter parity only for its pinned official-report SHA. Running `--gate --candidate estimator_k3_strassen_L4_f21.py` would **refuse** because that report belongs to another file; it is not a F21 certification command (`local_sim/p2_flop_anchor.py:737-795`). Include the separately observed `+2**24` graded FLOP offset in score-boundary forecasts (`AGENTS.md`, Measurement law; `experiments/LOG.md:48236-48253`).

The previous quick F21 certificate was formally **REFUSED**: all four local wall measurements were 123.66–125.92 s against 120 s, though each recorded 14,091 operations and the frozen FLOP count (`experiments/2026-09-12_f21/cert_F21_quick_remote1074_refused.json:16398-16518,16657,17052`). The full certificate may reproduce this refusal. Preserve its actual verdict and the capped mini's failure breakdown; do not relabel a local residual reading, op model, or uncapped mini as a clean deployment pass. Local residual seconds are ratio-only under the binding protocol (`docs/VALIDATION.md:445-450,528-533`). If the capped mini fails, the normal submission-approval path remains blocked unless Samuel separately decides on the exact documented exception; per-submission approval is required in every case, and no agent designates (`AGENTS.md`, hard rule 6; `experiments/LOG.md:48293-48305`).
