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
| 29 | C10 (factored K3 + regenerated κ4, rank 256 age 4) public score | adjusted `6.4914e-9`, raw `2.9686e-8`, C/B `0.21867`, 100/100 | GRADED | public 100 | `experiments/run_reports/c10_canary_331816_graded_report.json` | 2026-09-22 |
| 30 | **The DEV→graded factor is stable and slightly pessimistic.** Six observations of (100-net mini score)/(graded score) | `1.0386 / 1.0385 / 1.0339 / 1.0426 / 1.0471 / 1.0458`, mean `1.0411` | GRADED vs MEASURED | mini 100 vs public 100 | `RESULTS.md` § Calibration | 2026-09-22 → 09-24 |
| 31 | The DEV→graded gap is entirely a RAW-MSE effect, not a cost effect | C10's graded multiplier matched its DEV multiplier to `2e-10` while raw moved `3.0950e-8 → 2.9686e-8` | GRADED | public 100 | same | 2026-09-22 |
| 32 | **The f64 read-only grader twin predicts server FLOPs exactly.** Three out-of-sample confirmations | delta `0`, `0`, and `491` FLOPs of `4.8e11` (`1e-9` relative) | GRADED vs MEASURED | F21, v31, C10 | `RESULTS.md` § Calibration | 2026-09-21 |
| 33 | **D3 billing polish is worth −2.26%, not PLAN's −6–10%.** Of five proposed items two are infeasible, one is worth ~0.05%, and the two that work only work together | `−2.26%` score at flat ops | MEASURED | DEV 100 | `estimator_k3_v46_d3.py`; `RESULTS.md` § D3 | 2026-09-22 |
| 34 | **`out=` buys nothing under flopscope.** `fnp.add(a,b,out=buf)` bills exactly what `a+b` bills; slice assignment raises (arrays immutable); `fnp.empty` itself costs an op | — | MEASURED | — | `RESULTS.md` § D3 item 3 | 2026-09-22 |
| 35 | **Symmetry tagging discounts elementwise ops, never matmuls.** `X@Y` and `einsum` bill identically tagged or not; `X+Y` halves; `as_symmetric` itself costs ~7n² | `33,488,896` → `33,488,896` | MEASURED | n=256 | `RESULTS.md` § D3 item 5 | 2026-09-22 |
| 36 | **63% of C10's op budget is Strassen glue.** Depth 0 → 5 buys a 33.6% FLOP reduction with 14,697 extra op calls | ops `8,724 → 23,421` | MEASURED | — | `tools/d3_strassen_table.py` | 2026-09-22 |
| 37 | **K4 is unaffordable, and worsens with width.** Its 3.4–6.5× accuracy costs 21–42× the FLOPs | score `6.22×` worse at n=128, `6.48×` at n=256 | MEASURED (PROXY shape) | ARC's own repo | `tools/d4b_price_k4.py`; `RESULTS.md` § D4b | 2026-09-22 |
| 38 | **Restoring the dropped same-order diagrams is score-negative**, so ARC's `TODO.md` note is real but unaffordable | `1.15×` worse at n=128, `2.18×` at n=256 | MEASURED (PROXY) | same | same | 2026-09-22 |
| 39 | **Factored SIMPLE K3 beats factored AUGMENT on SCORE by ~3×**: the extra Edgeworth diagrams cost 8× and buy 2.5–2.7× | `0.343` / `0.312` score ratio | MEASURED (PROXY) | same | same | 2026-09-22 |
| 40 | **PLAN D5's premise is backwards — the old tier is UNDER-funded.** Even the mildest truncation (rank 192, −6.8% cost) costs 43.9% of raw MSE | score `+34.1%` | MEASURED | DEV 100 | `RESULTS.md` § D5 | 2026-09-22 |
| 41 | **Spending more on age wins, and age beats rank.** Compression age 4→5 costs +6.13% FLOPs and returns 15.50% of raw | score **`−10.32%`** — the largest single lever found | MEASURED | DEV 100 | same | 2026-09-22 |
| 42 | — but age 5→6 adds only a further `−0.66%` of score (+5.14% FLOPs for −5.53% raw): **the age curve is nearly flat past 5** | `−10.91%` vs age 4 | MEASURED | DEV 100 | same | 2026-09-22 |
| 43 | **Fitting is saturated on this chain.** Refitting all profiles plus a newly wired old-tier readout gain, 302 further CMA-ES evaluations | `0.00%` beyond D2's shipped coefficients | MEASURED | FIT | `RESULTS.md` § D6 | 2026-09-22 |
| 44 | — and the first run of that reported 0.00% **for the wrong reason** (the search seeded at identity, discarding the shipped coefficients, and restarted 4.5% behind) | — | RETRACTED then re-measured | FIT | same | 2026-09-22 |
| 45 | **Structure beats fitting on this chain by a wide margin**: one integer (age 4→5) is worth −10.32% where the entire fitted-coefficient apparatus is worth −2.7% and will not go further | — | MEASURED | DEV 100 | `RESULTS.md` § D5 + § D6 | 2026-09-22 |
| 46 | **THE WALL LAW: server wall is set by Strassen LEAF SIZE, not by FLOPs.** Kernel time per FLOP is ~1.9× higher with 32-wide leaves than 64-wide | `176.3` vs `92.8` ps/FLOP | GRADED | 10 graded runs | `RESULTS.md` § Calibration (wall refit) | 2026-09-23 |
| 47 | — so depth 5's ~7% FLOP saving **costs ~35 s of wall**, and per-op overhead is flat across every run at | `~0.36 ms/op` | GRADED | same | same | 2026-09-23 |
| 48 | — and a mixed-leaf chain reads **between** the two endpoints, confirming the mechanism rather than a per-candidate constant | `141.0` ps/FLOP (tr64) | GRADED | public 100 | `experiments/run_reports/tr64_canary_332088_graded_report.json` | 2026-09-24 |
| 49 | **The wall TAIL is the day, not the candidate.** Identical v50 bytes graded worst `87.7 s` at 03:00 UTC and `104.1 s` at ~10:00 UTC the same day | worst/median `1.03–1.055` quiet, `1.20–1.235` noisy | GRADED | public 100 ×3 | `RESULTS.md` rows #331939 / #331989 | 2026-09-23 |
| 50 | — and the slow nets are **random, not net-specific**: per-net wall correlation between two identical-bytes runs | `0.03` | GRADED | public 100 ×2 | rows #331967 / #331980 | 2026-09-23 |
| 51 | **Strassen depth 6 (leaf 16) is DEARER in FLOPs** — the Winograd glue outruns the multiply saving, so leaf 32 is the FLOP minimum in our spelling | C/B `0.22014` vs v55's `0.21618` | MEASURED | f64 twin | `experiments/2026-09-24_depth6/` | 2026-09-23 |
| 52 | **The wall-safe twin: −54 s of wall for +7.28% of FLOPs.** v56b = v55 at depth 4, raw MSE IDENTICAL | worst `60.7 s` vs v55's `114.5 s` | GRADED | public 100 | `experiments/run_reports/v56b_canary_332063_graded_report.json` | 2026-09-23 |
| 53 | Negative, measured: single-leg tower (T0–T3), N1 sensitivity-weighted basis, tiered old-tier ranks, P3 D21-feedback at any rank, per-network adaptive λ, exact Gaussian bivariate covariance tail, joint refit | each `≤0.4%` or score-negative | MEASURED | DEV 100 / FIT 100 | `RESULTS.md` §§ T0–T3, Step 1–4e | 2026-09-23 |
| 54 | — in particular, **energy ranks a lineage's downstream sensitivity almost perfectly**, which is why balanced truncation has nothing to add | Spearman `0.994` / `0.958` | MEASURED | 2 nets, 512 directions | `RESULTS.md` § Step 3 (N1) | 2026-09-23 |
| 55 | — and **per-network λ adaptation is not where per-net variation lives**: the ratio it keys on varies only | `1.8–9.0%` CV over 100 FIT nets | MEASURED | FIT 100 | `RESULTS.md` § Step 4a | 2026-09-23 |
| 56 | **THE LABEL NOISE FLOOR IS 6.8e-10, NOT 9e-11.** Layer 1 is our exact closed form in float64, so its MSE measures the organizer labels' own Monte-Carlo noise directly | measured `6.8815e-10` vs analytic `Var(ReLU)/1e9 = 6.817e-10` (1%) | MEASURED | DEV 100 | `RESULTS.md` § Step 1 (09-24) 1a; `AGENTS.md` corrected | 2026-09-24 |
| 57 | **THE ERROR IS PROPAGATED, NOT LOCAL.** One-step residual (chain stepped from the LABELLED previous mean) as a fraction of the accumulated residual falls monotonically with depth | `0.570` at layer 4 → **`0.206`** at layer 16 | MEASURED | DEV 100, reproduced on FIT 100 (`0.205`) | `experiments/2026-09-24_anatomy/anatomy_dev100.json` | 2026-09-24 |
| 58 | **A ReLU layer CONTRACTS incoming mean error.** Writing `M_l = A_l M_{l-1} + S_l`, the amplification is below 1 at every layer — all growth in MSE is newly created error | `A_l = 0.60–0.88`; the decomposition reconstructs `M_16` to **`−0.00%`** | MEASURED | DEV 100 | same | 2026-09-24 |
| 59 | **No local feature correlates with the residual.** Max \|Pearson\| over all 16 layers and 7 per-neuron chain features | **`0.097`** (σ); α and D3/σ³ read `0.002` | MEASURED | DEV 100 × 1024 neurons | same | 2026-09-24 |
| 60 | — and the one-step residual's concentration on α (top-2 of 10 bins = `0.664` at layer 16) sits at **α≈0, the ReLU kink**, with per-bin bias only `0.1–2.4%` of RMS and the SAME sign either side of zero: symmetric spread, not offset | no closure term implicated | MEASURED | DEV 100 | same | 2026-09-24 |
| 61 | **A linear corrector cannot see the signal; a nonlinear one can.** Every single-feature correlation < 0.10 and every single-axis nonparametric ceiling < 0.7%, yet a 33-feature MLP reaches net-wise held-out R² of | ridge `0.016–0.024` vs **MLP `0.057–0.119`** | MEASURED | FIT 100, folds split BY NET | `experiments/2026-09-24_anatomy/{ceiling,mlp_ceiling}.json` | 2026-09-24 |
| 62 | — implying an end-to-end raw gain, through the measured propagation weights, of | ridge `2.13%`; **MLP `6.16–7.32%`** (bracket over two targets) | MEASURED (projection from measured R²) | FIT 100 | same | 2026-09-24 |
| 63 | — and capacity is already saturated: hidden 64 is UNIFORMLY worse than 32 at all 15 layers, while the ridge's train/held-out gap of `~0.0005` says the RIDGE is not data-limited but the MLP is | `−0.005` to `−0.011` R² | MEASURED | FIT 100 | `experiments/2026-09-24_anatomy/mlp_ceiling_h64.json` | 2026-09-24 |
| 64 | **Per-lineage adaptive age is a FUZZY FIXED AGE in this chain.** Lineage energy ratios `e_old/e_new` are too tightly distributed for a tolerance to allocate age per net | median `0.380` at the operating point, p5–p95 `0.276–0.524` (factor 1.9) against a factor 1.7 between consecutive ages | MEASURED | 3 DEV nets, all live layers | `tools/adage_calib.py`; `RESULTS.md` § Step 3 (09-24) | 2026-09-24 |
| 65 | — confirmed by the fold schedules: `tau = 0.28` produces **ONE schedule across every net** (= fixed age 6), and its `−5.40%` raw is fixed age 6's `−5.53%` | 1 distinct schedule / 20 nets | MEASURED | DEV 20 | `experiments/2026-09-24_adage/schedules.json` | 2026-09-24 |
| 66 | — while v55 also produces one schedule, independently re-confirming the property every graded row reports | "FLOPs spread 0 across all 100 nets" | MEASURED + GRADED | DEV 20 / public 100 | same | 2026-09-24 |
| 67 | **A scalar materialization is cheap in ops and unmeasurable in wall locally.** 16 comparisons bill 32 FLOPs and 32 op calls, but the grader runs arithmetic SERVER-SIDE (`flopscope_server_comms_wall_s` median `0.78 s`), so a forced round trip is invisible here | ops `+0.14%` | MEASURED | — | `tools/build_v60.py` docstring; tr64 report | 2026-09-24 |
| 68 | **tr64 (tower transports at leaf 64) public score and wall** — the A designate | adjusted `5.4341e-9`, raw `2.4610e-8`, C/B `0.22081`; worst `80.30 s` (median `76.90`), 0 nets >100 s, 0 failed | GRADED | public 100 | `experiments/run_reports/tr64_canary_332088_graded_report.json` | 2026-09-24 |

*Rows 29-68 added 2026-09-24. The ledger had stopped at 2026-09-16, before the
Sep 21 reset; everything from C10 onward -- the wall law, the D5/D6 inversion,
the all-layer anatomy and the corrector ceiling -- was unrecorded. The write-up
will be written by a session that cannot read this one.*

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
