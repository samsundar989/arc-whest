# Phase-2 frontier submission telemetry — 2026-08-31 14:57 UTC

Manual transcription from the public per-submission report pages in Samuel's
Chrome window. Each report exposed 100 completed telemetry rows, 50 public score
rows, 50 sealed score rows, and zero failures. Error values below are public-50
means; FLOPs and wall summaries cover the 100 visible telemetry rows. The pages do
not expose code or an operation ledger.

| submission | adjusted | final MSE | all-layer MSE | mean FLOPs | budget | wall min / mean / max | failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| [J2W #329139](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329139) | `5.580e-9` | `2.251e-8` | `1.00e-8` | `5.45e11` | `24.78%` | `64.742 / 66.586 / 69.539 s` | `0` |
| [Puffi #329181](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329181) | `5.892e-9` | `2.262e-8` | `8.72e-9` | `5.73e11` | `26.05%` | `80.207 / 85.436 / 90.615 s` | `0` |
| [Activation-Regions #328901](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/328901) | `7.404e-9` | `2.933e-8` | `3.41e-4` | `5.55e11` | `25.25%` | `60.932 / 66.370 / 75.564 s` | `0` |
| [marius_binner #329126](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/329126) | `7.685e-9` | `1.833e-8` | `7.31e-9` | `9.22e11` | `41.92%` | `94.453 / 98.172 / 104.766 s` | `0` |

Interpretation boundary: the first three mean-FLOP totals cluster around the
leading cost of 16,384 dense forward directions (`2**39 = 5.497558e11`), while
marius buys lower raw error at higher cost. This constrains an operating envelope;
it does **not** identify row count, sampler, control variate, or private code.
Activation-Regions' `3.41e-4` all-layer MSE alongside `2.933e-8` final MSE shows
that at least one strong entry explicitly optimizes the scored final row rather
than maintaining accurate intermediate rows.
