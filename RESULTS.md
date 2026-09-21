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

All four are implemented in `tools/ev.py`; change them there and here together.

- **DEV → graded**: divide the 100-net DEV adjusted score by **1.037**. Observed
  1.0386 / 1.0385 / 1.0339 on three submissions (±0.3%); v31's own row landed
  +0.45% off `predict_public.py`. If a graded row deviates >2% from its
  projection, stop submitting and recalibrate before the next slot (PLAN §5).

- **FLOPs: there is no offset to add.** Take the number from the **f64 read-only
  grader twin** (`tools/_cost_worker.py`, one net) and use it as-is. Verified
  2026-09-21 against both graded rows we hold:

  | candidate | twin FLOPs/MLP | twin multiplier | graded multiplier | delta |
  |---|---:|---:|---:|---|
  | F21 #330950 | 928,457,637,478 | 0.42221365105 | 0.4222136511 | **0** |
  | v31 #331317 | 661,961,618,994 | 0.30102529263 | 0.3010252926 | **0** |

  (A third confirmation is on record: the twin predicted #330212's `flops_used`
  to delta 0 out of sample, `docs/GOTCHAS.md` 2026-09-07.) What *does* read low
  is `whest run --split mini`, by exactly **2^25 = 33,554,432** — its local
  dataset hands out float32 weights so the candidate's opening cast never runs.
  That is the "+2^25" PLAN §4 mentions; taking the number from the twin makes
  the correction unnecessary rather than approximate. C10 measured: twin
  480,850,933,121 vs `whest run` 480,817,378,689, difference 2^25 exactly.

- **one-net twin wall → server WORST-of-100 wall**: multiply by **1.5981**.
  Two anchors, both measured with `tools/_cost_worker.py`:

  | candidate | twin wall (1 net) | graded worst wall | factor |
  |---|---:|---:|---:|
  | F21 #330950 | 58.295 s | 93.1597 s | 1.5981 |
  | v31 #331317 | 43.547 s | 68.1319 s | 1.5646 |

  The factor deliberately bundles *server slowness* with *worst-of-100 vs our
  one net*, because the gate asks whether the worst net finishes. The larger
  anchor is used: over-projecting wall costs nothing, under-projecting costs the
  submission.

- **ops → server residual**: **10.74–11.09 µs/op** (v31 13,053 twin ops →
  0.140244 s; F21 14,091 → 0.156232 s). The cliff is 0.4 s/MLP and it FAILS the
  MLP. The in-process local `residual_wall_time_s` is an instrument artefact
  (C10 reads 5.9–9.1 s locally) and must never be quoted — project from ops.

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
