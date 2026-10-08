# Lane M3 report — κ4 as memory, the value of fourth order, which piece carries it (2026-10-08)

Branch `lane-m`, `experiments/2026-10-08_laneM3/`. Predictions recorded before any run:
`predictions.md`. GPU work ran on the RunPod box (RTX PRO 5000): the local queue could not admit
M8 (28 GiB host RAM wanted, ~13 GiB free).

## Summary for Lane P and the strategist

1. **M8 — the κ4 core is not a memory of old κ3. Dropping old blocks: NO-GO. Folding them into a
   rank-256 basis: GO-shaped.** In ARC's b+E chain, dropping κ3 blocks older than A = 3 / 5 / 8
   layers costs ×16.9 / ×6.7 / ×2.1 (16/16 nets). At short memory b+E and SIMPLE end equal
   (a / b+E = 1.01–1.05): AUGMENT's whole 2.5× gain over SIMPLE needs the long κ3 history. But
   keeping the old blocks in a shared rank-256 basis (re-based every layer, energy-ordered — our
   old tier's rule) with A = 5 exact recent layers costs only **×1.09 [1.06, 1.12]**, inside the
   1.3× bar, and keeps a / b+E = 2.37. A complete-K3 chain becomes cheap through *our* lineage
   representation, not through the core.
2. **M9 — fourth order is worth ~3–5× at n = 1024 (ARC's sweep, 8 nets per width).** K=4
   SIMPLE / unfactored K=3 AUGMENT = 0.56 / 0.50 / 0.43 / 0.36 at n = 96 / 128 / 192 / 256 (8/8
   nets at every n ≥ 128). The ratio trend extrapolates to **0.18–0.36 at n = 1024 (2.8–5.5×
   headroom)**; per-method slope extrapolation is not usable (it puts K=3 at 1.9e-9 at 1024 vs
   ~1.5e-8 measured). [n = 320 point pending.]
3. **M10.1 — the K=4 gain is NOT in the star / path / 3+2 births.** With every top-slice birth
   switched off ("none": κ4 = the exactly transported repeated-index blocks only), K=4 SIMPLE keeps
   **all** of its gain (none / full = 1.01 [0.67, 1.53] at n = 128; none / K3 = 0.50, 8/8). Star
   and star+path are *worse* than none; 3+2 alone is harmful (×2.5 vs full); removing κ4 transport
   memory loses the gain (×2.3 vs full, 0/8). **The lever is exact transport of the repeated-slice
   (2,1,1)/(2,2)/(3,1)/(4) blocks**, which K=3 AUGMENT projects to the r = 1 trace core. The
   M10.2 shared-factor arms (star D4/D31/D22, path D4 read from κ3 legs) target pieces that carry
   nothing, so they were not built. **Direction needed** (below).

## M8 — Is κ4 a memory? (16 DEV nets, n = 1024, organizer labels, floor-subtracted)

Driver `m8_drop.py`. Every κ3 factor column carries its block's birth step; at the input of step t
columns with τ < t − A are removed by building a new `FactoredTensor` from the kept columns. The
correction block K3 += from_dstensor(K_ds[3]) is split exactly by tag: from_dstensor is linear and
its legs 2, 3 are I, so its first leg is D_new + Σ_g D_g with D_g = −[rep_g(3,)·I + 3·rep_g(2,1)ᵀ]
for pK_111's columns in group g; corrections die with their block. Calibration (`m8_calibrate.json`,
n = 64, float64): split-all A = ∞ equals the unmodified chain to 5e-14 (both chains); every dropped
group has zero repeated-index slices at birth to 2e-16 (no −Rep ghosts). **Gate:** A = ∞
reproduces b+E: 1.583e-8 (Lane M2: 1.583e-8; per net 8e-5 relative, a different GPU/torch build).

| chain | A | final MSE | ratio to A = ∞ (gm, 95 %) | worse | per-layer ratio L4 / L8 / L12 / L16 |
|---|---|---|---|---|---|
| b+E | ∞ | 1.583e-08 | 1 | — | — |
| b+E | 8 | 3.338e-08 | ×2.10 [2.01, 2.20] | 16/16 | 1.00 / 1.00 / 1.25 / 2.11 |
| b+E | 5 | 1.069e-07 | ×6.74 [6.43, 7.08] | 16/16 | 1.00 / 1.86 / 4.53 / 6.75 |
| b+E | 3 | 2.672e-07 | **×16.9 [15.8, 18.1]** | 16/16 | 1.00 / 9.99 / 14.65 / 16.89 |
| b+E | 2 | 4.187e-07 | ×26.5 [24.7, 28.4] | 16/16 | 7.22 / 24.64 / 26.29 / 26.45 |
| b+E | 1 | 6.712e-07 | ×42.5 [39.7, 45.6] | 16/16 | 44.51 / 57.60 / 48.06 / 42.41 |
| a | ∞ | 3.930e-08 | 1 | — | — |
| a | 8 | 5.501e-08 | ×1.40 [1.34, 1.46] | 16/16 | 1.00 / 1.00 / 1.08 / 1.40 |
| a | 5 | 1.236e-07 | ×3.14 [2.93, 3.36] | 16/16 | 1.00 / 1.30 / 2.24 / 3.15 |
| a | 3 | 2.798e-07 | ×7.11 [6.59, 7.66] | 16/16 | 1.00 / 4.30 / 6.01 / 7.12 |
| a | 2 | 4.268e-07 | ×10.9 [10.1, 11.7] | 16/16 | 3.59 / 9.80 / 10.46 / 10.86 |
| a | 1 | 6.775e-07 | ×17.3 [16.1, 18.6] | 16/16 | 19.05 / 22.32 / 19.00 / 17.24 |

**Compression arm** (old blocks folded, not dropped: every leg of the columns older than A
projected onto the top-r left singular subspace of their stacked legs, re-based each step):

| chain | A, r | final MSE | ratio to A = ∞ | worse | per-layer L4 / L8 / L12 / L16 | captured energy (first / last step) |
|---|---|---|---|---|---|---|
| b+E | 5, 256 | 1.724e-08 | **×1.09 [1.06, 1.12]** | 16/16 | 1.00 / 1.02 / 1.07 / 1.09 | .992 / .999 |
| b+E | 5, 128 | 3.700e-08 | ×2.33 [2.20, 2.48] | 16/16 | 1.00 / 1.30 / 1.98 / 2.34 | .917 / .987 |
| b+E | 3, 256 | 3.229e-08 | ×2.03 [1.95, 2.12] | 16/16 | 1.00 / 1.93 / 2.08 / 2.04 | .965 / .996 |
| b+E | 3, 128 | 1.323e-07 | ×8.32 [7.83, 8.83] | 16/16 | 1.00 / 6.52 / 8.59 / 8.36 | .82 / .972 |
| a | 5, 256 | 4.087e-08 | ×1.04 [1.02, 1.06] | 16/16 | | .993 / .999 |
| a | 5, 128 | 7.324e-08 | ×1.86 [1.80, 1.92] | 16/16 | | |
| a | 3, 256 | 6.717e-08 | ×1.70 [1.60, 1.80] | 16/16 | | |
| a | 3, 128 | 2.097e-07 | ×5.29 [4.86, 5.75] | 16/16 | | |

a / b+E (ratio of means) at equal treatment: dropped A = 1 / 2 / 3 / 5 / 8: 1.01 / 1.02 / 1.05 /
1.16 / 1.65; folded A5r256 2.37, A5r128 1.98, A3r256 2.08; nothing dropped 2.48.

**Readings.**
- The handoff's test ("b+E with A = 3 dropped ≤ 1.3× ⇒ the core carries old κ3") fails by 13×:
  the core does not stand in for old κ3. Old κ3 is irreplaceable *as information*.
- It is, however, **compressible**: 99.2–99.9 % of the old blocks' stacked-leg energy sits in
  256 directions, and A = 5 + rank 256 loses 9 % of final MSE. The fold costs the most at the
  *first* fold step (energy .82–.97) and with r = 128 (×2.3) — the rank matters more than A.
- The fold hurts b+E more than SIMPLE (×1.09 vs ×1.04): AUGMENT's heavy legs live partly in the
  directions the rank-256 basis drops. Lane P should expect the fold, not the core, to set the
  cost of a complete-K3 chain.
- Unit price in our representation: the rank-256 old tier is what v77/v80 already carry; the
  increment of a complete-K3 (AUGMENT) chain over ours is its heavy-leg births and the r = 1
  core with its K211 / 211→111 loop (Lane M2: none of these is separable).

**Predictions vs outcome.** Strategist: b+E A = 5 dropped ×2–5 [60 %] → ×6.7 (miss); a A = 5
×10–20 [70 %] → ×3.1 (miss, the other side: SIMPLE needs *less* history than AUGMENT). Worker:
b+E A = 5 ×3 → ×6.7 (miss); A = 3 ×6, NO-GO [70 %] → ×16.9, NO-GO (direction right); A = 1 ×20+
→ ×42 (right); a A = 5 ×8 → ×3.1 (miss); "a degrades faster than b+E at every A" → **wrong**:
b+E degrades faster in ratio terms because it has more to lose.

## M9 — How much is fourth order worth? (ARC's sweep, RunPod)

Pipeline: ARC's own `scripts/kprop_by_width.py` (`m9_sweep.py`): 16 ReLU layers act0..act15, all
W ~ N(0, 2/n) (`num_layers = 17, init 'manual', w_var 2`), `use_avg_metric = True`, grad off,
fp32, TF32 off, `trials = 8`, `sample_k_max = output_d_max = 1`, N = 2³⁰ samples per net with 64
bootstrap blocks. One shared sample cache, so all methods are paired by (n, trial). Deviation:
the sampler is a means-only re-implementation of ARC's (`fast_sample_means`) — bit-identical
output on the smoke seeds, ~10× faster; pre-activation layers are not sampled (not scored).
Label noise at act15 ≈ 1e-10, ≤ 0.03 % of every MSE quoted.

Final-layer MSE (err − se2), mean of 8 nets:

| method | n = 64 | 96 | 128 | 192 | 256 | slope n ≥ 128 |
|---|---|---|---|---|---|---|
| K=3 AUGMENT unfactored (baseline) | 1.65e-03 | 3.14e-04 | 2.61e-05 | 3.90e-06 | 1.10e-06 | −4.58 |
| K=3 AUGMENT factored (ARC) | 3.33e+00 | 6.63e-04 | 2.67e-05 | 5.20e-06 | 1.61e-06 | −4.05 |
| K=3 SIMPLE factored | 5.69e-04 | 1.82e-04 | 2.00e-05 | 6.51e-06 | 2.62e-06 | −2.92 |
| **K=4 SIMPLE factored** | 2.05e-02 | 2.35e-04 | 1.35e-05 | 1.67e-06 | 3.97e-07 | −5.09 |
| K=4 AUGMENT factored | NaN | 4.28e-04 (4/8 finite) | 1.37e-04 | — | — | |

Paired ratio to the baseline (geometric mean, 95 %, nets better):

| method | n = 96 | 128 | 192 | 256 |
|---|---|---|---|---|
| K=3 AUG factored | 1.28 [0.87, 1.87] 3/8 | 1.09 [0.85, 1.39] 3/8 | 1.41 [1.12, 1.79] 1/8 | 1.45 [1.25, 1.68] 0/8 |
| **K=4 SIMPLE** | 0.56 [0.37, 0.84] 7/8 | 0.50 [0.36, 0.69] 8/8 | 0.43 [0.27, 0.68] 8/8 | **0.36 [0.28, 0.45] 8/8** |
| K=4 AUGMENT | 6.3 [3.3, 12.3] 0/4 | 3.0 [1.5, 5.9] 0/8 | | |

**Cross-check against the Oct 6 table (our nets, MC labels) at n = 256:** unfactored AUG 1.10e-6
here vs 1.21e-6; factored AUG 1.61e-6 vs 1.65e-6. Factored/unfactored = 1.45 here vs 1.36 (Oct 6:
G_drop 26.6 %) — consistent.

**Extrapolation to n = 1024.** Per-method fitted slopes are pre-asymptotic (K=3 −4.6 here vs
≈ −3.2 measured from 256 to 768 on Oct 6), so extrapolating each method separately puts K=3 at
1.9e-9 at n = 1024 against ~1.5e-8 measured — 8× wrong; the strategist's "naive per-order
extrapolation overstates" is confirmed. The ratio is better behaved: log-slope of K4/K3 from 128
to 256 is −0.48; continuing it gives **0.18 at n = 1024**, holding it flat from 256 gives 0.36.
**Headroom 2.8–5.5× over unfactored K=3 AUGMENT**, i.e. ~3.5–7× over ARC's factored K=3 level
(the 1.46–1.52e-8 cluster). [n = 320 point pending; it decides where in that range we sit.]

K=4 AUGMENT (factored) is unstable at depth 16 at these widths (NaN at 64, 4/8 finite at 96,
3× *worse* than K=3 at 128) — ARC's factored AUGMENT at K = 4 is not a usable reference here.

**Outcome rule.** ≥ 3× ⇒ M10 continues; 1.5–3× ⇒ M10 only with a cheap shared-factor arm. The
estimate straddles 3×; M10.1 was run (cheap), M10.2 was not built — see M10.

**Predictions vs outcome.** Strategist: K4S/K3 0.3–0.6 at 256 [I] → 0.36 (hit); 0.05–0.2 at
1024 [50 %] → 0.18–0.36 (edge); naive overstates → yes. Worker: 0.5 at 256 → 0.36 (miss low);
0.25 at 1024 → in range.

## M10.1 — which fourth-order pieces carry it (n = 96, 128; 8 nets each; same nets/labels as M9)

Driver `m10_ablate.py`: `factor_k4.factored_nonlin_kprop_k4` re-compiled with each birth class
scaled by a flag (`factor_k4.py:487/497/506` alphas, `:434` memory); "full" equals unpatched K=4
SIMPLE exactly.

| arm | n = 96 vs full | n = 128 vs full | n = 128 vs K3 baseline | share of K3→K4 gap (median, 96 / 128) |
|---|---|---|---|---|
| full (path + star + 3+2) | 1 | 1 | 0.50 [0.36, 0.69] 8/8 | 1 / 1 |
| **none** (no top-slice births) | 0.77 [0.56, 1.06] | **1.01 [0.67, 1.53]** | **0.50 [0.41, 0.63] 8/8** | 1.06 / 1.03 |
| path only | 0.78 [0.58, 1.06] | 0.94 [0.62, 1.40] | 0.47 [0.38, 0.58] 8/8 | 0.99 / 1.16 |
| star only | 1.26 [0.74, 2.14] | 1.35 [0.95, 1.91] | 0.67 [0.56, 0.81] 8/8 | 0.79 / 0.69 |
| star + path | 1.24 [0.60, 2.53] | 1.22 [0.85, 1.75] | 0.61 [0.49, 0.75] 8/8 | 0.83 / 0.84 |
| 3+2 only | 4.10 [1.91, 8.81] | 2.48 [1.86, 3.30] | 1.24 [0.76, 2.02] 4/8 | −0.29 / −0.34 |
| no κ4 transport memory | 1.53 [0.53, 4.42] | 2.33 [1.46, 3.72] | 1.16 [0.72, 1.87] 4/8 | 0.37 / −0.62 |

(Means of the gap share are dominated by single-net blow-ups at n = 96; medians shown.)

**Readings.**
- The K=4 SIMPLE gain over K=3 is **entirely** in what remains with the top-slice births off: the
  `from_dstensor` correction blocks — the exact repeated-index slices (4), (3,1), (2,2), (2,1,1)
  born each layer, carried as full n × n per-column factors and transported exactly. K=3 AUGMENT
  keeps only their r = 1 trace projection (R3 §3.4.5). Their memory matters (no-memory ×2.3).
- Star and star+path *cost* 20–35 % relative to "none"; 3+2 alone is harmful. The birth classes
  are not additive goods: their sum (full) ≈ none, so they roughly cancel.
- Consequently the M10.2 shared-factor arms (star D4/D31/D22 and path D4 read off the κ3 legs,
  R3 §3.1–3.2) target pieces that carry ≤ 0 of the gain. Not built.

**Predictions vs outcome.** Strategist: star + path ≥ 60 % of the gain [40 %] → 0.83 median share
but *worse than none*: carries nothing net (miss in kind); D4-star-only ≥ 20 % [30 %] → not built,
star-only arm is worse than no births. Worker: 45 % / 20 % — same miss.

## Direction needed (strategist)

The decision rule (GO if an arm captures ≥ 40 % of the gap at ≤ +10 % F) cannot be met by any
shared-factor arm. The only piece that carries the gain is the exact transport of the
repeated-slice κ4 blocks, which ARC stores as one n × n matrix per column (R3 §4 item 1: ~2 B per
block-layer, Σ ≈ 240 B at n = 1024). Open questions, cheapest first, all at n = 96–256 on the M9
cache (minutes per arm on the pod):

1. **Rank of the transported correction blocks.** Replace each block's B factor by its top-k
   eigen/SVD part (k = 4, 16, 64) — does the gain survive? If k ≲ 16 suffices, the transport is
   k × (2 n³) per block-layer, still ~n× too dear unless k is tiny — but tells us what the object is.
2. **Which slice carries it.** Keep only the (2,1,1) part of B (κ4 → κ3 feedback path) vs only
   (2,2) / (3,1) / (4) — R3 §2.3 says the (2,1,1) slice feeds the dense κ3 top slice; that is the
   one channel K=3 AUGMENT's r = 1 core reproduces only at trace level.
3. **Gain-mode / template forms** (Lane M2 E21b-style) for the transported repeated-slice
   content, measured as in M6 (one-step capture, then end-to-end).

Recommendation: do not open a K = 4 prototype for Lane P on this evidence. The M8 fold result
(A = 5 + rank 256, ×1.09) is the actionable item for the Oct 10 decision.

## Artifacts

`experiments/2026-10-08_laneM3/`: `m8_drop.py`, `m8_calibrate.json`, `m8_all.json`,
`m8c_all.json`, `m8_table.md`, `m8_summary.json`, `m9_sweep.py`, `m10_ablate.py`,
`analyze_m8.py`, `analyze_m9.py`, `m9_table_relu_w2_m9.md`, `pod/` (raw ARC outputs and logs).
Pod cost: ~$10 (8.5 h of it idle after the runs finished — the follow-on runs had not been chained).
