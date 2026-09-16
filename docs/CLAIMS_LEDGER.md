# CLAIMS_LEDGER.md — every measured number that might reach the write-up

*Opened 2026-09-16 (Slice 7, handoff `CLAUDE_CODE_HANDOFF_2026-09-16.md`
§ Throughout). **One line per measured number**, with what it is, where it was
read, which block it came from, and the date. Budget: ≤1 hour a day. **No
write-up drafting before Oct 3** — this file is the raw material, not the draft.*

## Why this file exists

The contribution-prize write-up is a co-equal objective and it will be written
from numbers that were measured weeks earlier, by a session whose chat history
no longer exists. A number that cannot be traced back to a block, an artifact
and a date is not evidence — it is a recollection. Everything here is a
**pointer**, so the claim can be re-derived rather than re-remembered.

## Status vocabulary

| status | meaning |
|---|---|
| **MEASURED** | read off an artifact in this repository, on a named block |
| **GRADED** | read off an official AIcrowd graded report |
| **PROXY** | measured, but at a shape or on a chain that is not 1024×16 ours |
| **RETRACTED** | was claimed, then withdrawn. Kept, because the write-up's negative results are part of the contribution |
| **THEIRS** | 504aldo's published finding, reproduced or not. Never claimed as ours |

## The ledger

| # | claim | value | status | block | artifact | date |
|---|---|---|---|---|---|---|
| 1 | F21 (exact factored K3 tower) public score | adjusted `1.596e-8`, raw `3.779e-8`, C/B `0.42221` | GRADED | public 100 | `experiments/run_reports/f21_diagnostic_330950_graded_report.json` | 2026-09-14 |
| 2 | F21 worst graded wall / residual | `93.16 s` of 120; `0.156232 s` of 0.4 | GRADED | public 100 | same | 2026-09-14 |
| 3 | P1 (κ4 regeneration + label-free λ) over F21 | `1.2371×` on 24/24, t95 `[+3.36e-9, +4.39e-9]`, sign p `0.0000` | MEASURED | P01 | `experiments/2026-09-14_slice6a_p1/` | 2026-09-15 |
| 4 | P1's cost | `+0.000044 B` — unchanged to five decimals | MEASURED | P01 | same | 2026-09-15 |
| 5 | λ-scale sweep: bare core / 0.5 / **1.0** / 2.0 | `1.0696` / `1.1964` / **`1.2371`** / `1.0344`× | MEASURED | P01 | λ-sweep artifacts | 2026-09-15 |
| 6 | Our label-free λ sits within 1.1% of the sweep optimum (0.794, peak `1.2506×`) | 1.1% | MEASURED | P01 | same | 2026-09-15 |
| 7 | λ sensitivity is asymmetric: halving costs 3.3%, doubling 16.4% | — | MEASURED | P01 | same | 2026-09-15 |
| 8 | λ's own fitted value at 1024 is undetermined: leave-one-out window spread | `8.89×` | MEASURED | P03 | `local_sim/p2_lambda_extrapolate.py` | 2026-09-14 |
| 9 | E14's Gaussian-tree λ is a *different object* from the one that pays: opposite sign from layer 4, R² ≤ 0.2, depth-profile Pearson r = `−0.01` | — | MEASURED | — | `experiments/2026-09-14_exp_lambda/` | 2026-09-14 |
| 10 | P2 (7→4 identity) cost ratio, metered at 1024×16 in one BudgetContext | `0.7128×`, C/B `0.42221 → 0.30097`, saving 28.7% | MEASURED | — | Skeptic metering, `--parity` claim 6 | 2026-09-15 |
| 11 | P2 dense arm reproduces v30 end to end | raw `0.99994×`; final-row MSE `4.23e-13` / `3.25e-13` against a `~3.9e-8` signal | MEASURED | P01 | `experiments/2026-09-15_slice6b_p2/` | 2026-09-15 |
| 12 | Rank-32 M-residual truncation costs | `+0.90%` raw | MEASURED | P01 | same | 2026-09-15 |
| 13 | — and 504aldo publish `+0.9%` for the same rank on *their* chain | `+0.9%` | THEIRS (F63) | — | `docs/PROVENANCE_504aldo.md` | 2026-09-10 |
| 14 | v31 (P1+P2) over F21, read off raw and measured C/B, **not multiplied** | `1.7202×` = raw `1.2262×` × cost `1.4028×` | MEASURED | P01 | `experiments/2026-09-15_slice6b_p2/` | 2026-09-15 |
| 15 | v31 op count against the 22,000 gate | `12,665` ops on P01; **`13,037` on the public mini** (+2.9%) — 7.4% fewer than F21's `14,075` there | MEASURED | P01 / mini 100 | `--parity`; `experiments/2026-09-16_slice7_mini/v31_mini.json` | 2026-09-15 / 09-16 |
| 16 | v31 under a real `RLIMIT_AS = 8,192 MB` on Linux | 23/23, VmPeak `5,434.6 MB`, 1.51× headroom, peak RSS `3,563.1 MB` | MEASURED | — | job 1374, `local_sim/linux_memory_gate.py` | 2026-09-15 |
| 17 | **A deterministic sketch is not automatically a well-posed one.** `W[:, :r]` (504aldo's published sketch) fails permutation-equivariance at `1.07%` of raw MSE at 128×8; norm-selected columns restore it to `1.3×` the float32 floor | `5.4e5×` / `1.8e5×` the floor vs a 100× bar | MEASURED | D06 | `experiments/2026-09-15_slice6b_p2/edge_cases_*.log` | 2026-09-15 |
| 18 | — and the selection is **not** a better approximation. At 256×16 strongest columns are the worst of three | `1.121e-4` vs `9.940e-5` vs `1.025e-4` | RETRACTED (as an accuracy claim), MEASURED (as an ablation) | — | Skeptic ablation | 2026-09-15 |
| 19 | Server FLOPs vs local: `+2**24` on early same-dtype-cast rows; **exactly `+2**25` against the `whest run` subprocess meter** for both #330950 and #331317, while F21's grader-twin (`lifecycle_eval f64_readonly`) matched the server exactly — the gap is between the two local paths | `+0.005%` of v31 | MEASURED + GRADED | — | `official_config.py`; LOG 2026-09-16 16:09 | earlier / 2026-09-16 |
| 20 | ~85–90% of 504aldo's κ4 gain rides on one n² term (λ inside the dG transport), consumed through the (4)/(2,2) rows | `×2.32` at 128×8, `×3.95` at 64×8 | PROXY | 8 He-init nets/shape | `experiments/2026-09-16_exp_q31/results_all.json` | 2026-09-16 |
| 21 | — and the (3,1) consumer alone is worth | `×1.00` at 128×8, `×1.09` at 64×8 | PROXY | same | same | 2026-09-16 |
| 22 | — while F68 reports the κ4→κ3 feed is worth `~4×` at 1024×16, against `≤1.14×` on the proxy | — | THEIRS (F68) + PROXY | — | same | 2026-09-16 |
| 23 | v31 (P1+P2) public score | adjusted `8.752e-9`, raw `2.9074e-8`, C/B `0.30103`, 100/100 | GRADED | public 100 | `experiments/run_reports/v31_canary_331317_331317_graded_report.json` | 2026-09-16 |
| 24 | v31 vs F21, **graded against graded**, paired per network | `1.8233×`, better 50/50, t95 `[+6.88e-9, +7.53e-9]`; raw `1.300×` | GRADED | public-board 50 | same + `f21_diagnostic_330950_graded_report.json` | 2026-09-16 |
| 25 | v31 worst graded wall / residual | `68.13 s` of 120; `0.140244 s` of 0.4 (`10.76 µs/op`) | GRADED | public 100 | same | 2026-09-16 |
| 26 | `predict_public.py` projection vs the graded row | projected `8.713e-9`, graded `8.752e-9`: **+0.45%**, inside the band | GRADED vs MEASURED | mini 100 | `experiments/2026-09-16_slice7_mini/predict_public_v31.json` | 2026-09-16 |
| 27 | v31 is deterministic to the bit on the public mini | 100/100 networks identical across two runs, every layer and every FLOP | MEASURED | mini 100 | `experiments/2026-09-16_slice7_mini/v31_mini{,_confirm}.json` | 2026-09-16 |
| 28 | ×1.60 on local worst wall **under-predicts** the server when the local run is uncontended | `63.2 s` projected vs `68.1 s` graded; graded/local median `1.82` alone, `1.46–1.49` contended | GRADED vs MEASURED | mini 100 | LOG 2026-09-16 16:09 | 2026-09-16 |

## Claims this repository has RETRACTED, and must not quietly re-acquire

| what was claimed | what replaced it | where |
|---|---|---|
| F21's `93.16/87.72 = 1.062` is the local→graded wall anchor | Worst-over-worst is not transferable; median-to-median is, and it moves 1.46→1.82 with host load | LOG 2026-09-16 16:09 |
| "κ4 pays only in an exact tower" (Sep 10 "theorem") | Falsified: memoryless λ·C_off regeneration retains ~90% of AUGMENT's gain at zero n³ cost. It generalised from one ablation | strategist 09-14 §4 |
| P2 priced at 27.6% / 41.5% / 32.3% / 31.3% of the tower | **Metered** at 28.7%. Four hand models, three wrong, and the one that was close was right for the wrong reason | LOG 2026-09-15 |
| The identity suite is `11/11` | It was `8/11`: three checks could not fail, one of them a literal `True`. Now `10/10` and mutation-tested | LOG 2026-09-15 |
| "The P2 algebra is ours; the oracle confirmed only the sketch and the rank" | The reconstruction identity is ours (committed 1h47m before the clone). The `(2,1)` regrouping and the range finder are **504aldo's**, matched deliberately | `docs/PROVENANCE_504aldo.md` § The P2 timeline |
| The norm-selected sketch is a better approximation | It is not; it is *equivariant*, which is the real and sufficient justification | LOG 2026-09-15 |

## What the write-up may claim as ours, as of today

Stated conservatively, because `PROVENANCE_504aldo.md` is the file that has to
survive a §6 eligibility question:

1. The **reconstruction identity** `A_D = M·d(s) + 3·A_L·d(e) + Z·Lᵀ` — committed before the clone existed.
2. The **2-D Strassen-wide layout** and its shuttles: a genuinely different cost regime from their 3-D tower.
3. The **equivariant sketch**, which corrects a real defect in their published estimator — found by an instrument, not by a closer reading.
4. The **label-free λ** and its **flatness result** (claims 5–8): the value is ill-determined and it does not matter.
5. The **negative results** (claim 9, claim 18, the four mispricings, the unfalsifiable checks).
6. The **measurement protocol** itself — pre-registration, single-use blocks, the paired arithmetic gate, read-counted artifacts.
7. **Pending, and the strongest of them:** an "anatomy of the κ4 gain" at 1024×16 against F21 as an *exact* K3 reference. We are the only team known to run one at this shape. D1–D3 produce these numbers as a by-product.

**Not ours, and to be credited on sight:** the κ4 regeneration law (F68), the
7→4 identity (F63), the D21 feedback (F69), the age-gated basis (F72/F73), the
rank 32, and the `(2,1)` regrouping.
