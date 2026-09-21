# RESULTS — ARC White-Box Estimation Challenge, Phase 2

One row per candidate evaluation or graded run. This file replaces
`experiments/LOG.md` (archived 2026-09-21). Append; never rewrite history.
The plan is `docs/PLAN.md`; the submission gates are its §5.

## Column meaning

| column | meaning |
|---|---|
| **date** | UTC date of the evaluation or of the submission |
| **file** | candidate source in the repo root |
| **sha8** | first 8 hex of sha256 of the *estimator source bytes* (not the package) |
| **panel** | `DEV` = the 100 organizer mini nets · `FIT` = 100 full-split nets disjoint from mini · `HOLDOUT` = the remaining 800 (≤2 reads/week, never fitted) · `GRADED` = AIcrowd server, 100 nets (50 public + 50 sealed) |
| **raw MSE** | final-layer MSE, unweighted |
| **F/B** | FLOPs per MLP ÷ 2^41, i.e. the score multiplier before the 0.1 floor |
| **ops** | flopscope operation calls per MLP (drives residual wall time) |
| **wall** | worst per-MLP wall seconds. Local rows are marked `(local)`; the server figure is `local × 1.3097` (§ Calibration) |
| **RSS** | VmPeak under the WSL `RLIMIT_AS` 8,192 MB gate |
| **projected** | predicted public score = DEV score ÷ 1.037 |
| **graded** | the AIcrowd adjusted score actually returned |
| **paired gain** | paired arithmetic gain vs the stated baseline, common random numbers |
| **wins/N** | nets on which the candidate beat that baseline |

## Calibration constants (recheck after every graded row)

- **DEV → graded**: divide the 100-net mini adjusted score by **1.037**. Observed
  1.0386 / 1.0385 / 1.0339 on three submissions (±0.3%); v31's own row landed
  +0.45% off `predict_public.py`. If a graded row deviates >2% from its
  projection, stop submitting and recalibrate before the next slot (PLAN §5).
- **local wall → server wall**: multiply by **1.3097** (v31: worst local 52.02 s
  on the 100-net subprocess mini → worst graded 68.1319 s).
- **ops → server residual**: **10.8–11.2 µs/op** (F21 13,936 ops → 0.156232 s;
  v31 13,037 ops → 0.140244 s). The cliff is 0.4 s/MLP and it FAILS the MLP.
  The in-process local `residual_wall_time_s` is an instrument artefact
  (C10 reads 5.9–9.1 s locally) and must never be quoted — project from ops.
- **server FLOP offset**: the grading server bills an unexplained **+2^24**
  (+0.0122%) over the local meter. Include it only near a score boundary.

## Rows

| date | file | sha8 | panel | raw MSE | F/B | ops | wall | RSS | projected | graded | paired gain | wins/N | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-14 | `estimator_k3_strassen_L4_f21.py` | `d77dab0e` | GRADED | 3.779e-08 | 0.42221 | 13,936 (local meter) | 93.160 s | — | — | **1.5957e-08** | — | — | sub #330950, 100/100, 0 failed. Worst residual 0.156232 s (2.56× headroom). The designated fallback and the anchor for `predict_public.py`. |
| 2026-09-16 | `estimator_k3_v31.py` | `6dde5fd4` | GRADED | 2.9074e-08 | 0.30103 | 13,037 | 68.132 s | 5,434.6 MB | 8.713e-09 | **8.7520e-09** | 1.8233× vs F21 | 50/50 | sub #331317, 100/100, 0 failed, smoke 8/8. Worst residual 0.140244 s (2.85× headroom, 10.76 µs/op). Projection landed +0.45% — the best graded score to date and the §5 baseline. |
| 2026-09-18 | `estimator_k3_v31.py` | `6dde5fd4` | DEV | 3.006065e-08 | 0.30101 | 13,037 | 52.02 s (local) | 5,434.6 MB | 8.713e-09 | see above | baseline | — | The DEV control. `experiments/2026-09-16_slice7_canary/`. Reproduce this row in D1: DEV adjusted 9.0486e-09 ± 1%. |
| 2026-09-18 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | DEV | 3.094966e-08 | 0.21865 | 23,405 | 60.259 s (local) → 78.9 s proj | 5,754.8 MiB | **6.526e-09** | PENDING | 1.3371× vs v31 DEV (score ratio, unpaired) | — | C10. DEV adjusted 6.767156e-09, 100/100, 0 failed, official `v2-phase2` mini, subprocess runner (`experiments/2026-09-18_c10_diagnostic/c10_mini.json`). Raw MSE is 3.0% WORSE than v31; the whole gain is cost (0.30103 → 0.21865 F/B). All §5 gates pass — see below. |

## Open items

### C10 canary — PACKAGED AND GATED, NOT YET SUBMITTED (2026-09-21)

Package `experiments/2026-09-21_c10_canary/c10-canary.tar.gz`
(sha256 `1d476ab2d3527eae4ec4729ce012f8e2399bdee7c9d54ccbdc751cca24d3ec00`;
inner `estimator.py` = `fd8d497d…6152d1bf`, byte-identical to
`estimator_k3_v43_complete_c10.py`; LICENSE = the ARC MIT notice shipped with
#330950 and #331317). `whest validate-package` passes.

PLAN §5 gate readings, all green:

| gate | threshold | C10 | verdict |
|---|---|---|---|
| DEV projected vs best graded | ≥3% better than 8.7520e-09 | 6.526e-09 = **25.4% better** | PASS |
| `whest validate` | passes | passes, 96 ms | PASS |
| WSL `linux_memory_gate.py`, RLIMIT_AS 8 GiB | PASS | PASS, VmPeak 5,754.8 MiB, peak RSS 3,890.2 MiB | PASS |
| projected server wall | < 100 s | 60.259 × 1.3097 = **78.9 s** | PASS |
| op calls | ≤ 24,500 | **23,405** | PASS |
| setup | < 2 s | 0.00 s | PASS |
| no bare `except` returning a constant | — | one `except Exception` in `setup`, best-effort, returns nothing; `predict` has no broad handler | PASS |

Advisory (not a §5 gate): projected server residual = 23,405 ops × 10.8–11.2 µs
= **0.252–0.262 s** against the 0.4 s cliff, i.e. 1.53–1.59× headroom. Thinner
than v31's 2.85× and F21's 2.56×. Measuring it is half the point of this canary
(PLAN §6 D0: "the first server residual measurement at 23k ops").

**Blocked:** the submit call was refused by the agent session's own permission
classifier, not by any project gate. Samuel runs it, or grants the permission:

```powershell
.\.venv\Scripts\whest.exe submit experiments\2026-09-21_c10_canary\c10-canary.tar.gz `
  --description "C10 canary" --watch --format plain
```

Slot ledger at the time of packaging: 0/10 used on 2026-09-21 UTC.
