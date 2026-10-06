# Strategist packet — 2026-10-06

## Lane V+M (session 2, branch `lane-m`)

**V (fresh panels).** V1 done: `tools/fresh/gen.py` mints through whestbench's own
`derive_seed_streams` + `sample_mlp`; on 5 organizer `mini` nets both that path and a
numpy mirror reproduce the published weights **bit for bit**. FRESH-A (100, seeds
9,100,000+) and FRESH-LOCK (200, 9,100,100+) are pinned with weight hashes
(`experiments/2026-10-06_laneV/seeds.json`); `tools/panels.py --check` asserts disjointness
from every organizer seed and from the DEV/FIT hashes. V2 done: `tools/fresh/mc.py` (TF32 off,
float32 matmuls, float64 across-chunk sums, per-net variances → exact floor) runs 66 s/net at
N = 1e8. Validation on 10 mini nets: MSE vs the organizer labels = 0.95 ± 0.05 × the predicted
floor sum (8.2e-10 vs 8.7e-10), and against the EXACT layer-1 closed form 0.992 ± 0.016 × our
floor — unbiased. A third panel, FRESH-D (d8b corpus extension nets the twin never read,
other generator, other labeller), is built and verified. `tools/ev.py` gains `--panel
fresh_a|fresh_lock|fresh_d` (floor-subtracted absolutes, raw paired differences, FRESH-LOCK read
log refusing a 4th read without `--i-am-designating`). FRESH-A/LOCK labels are on the queue
behind Lane P; V4 (v73 vs v68_ws) is running on FRESH-D now, FRESH-A next.

**M (dropped diagrams) — measured, GO by rule, CLOSED on cost.** Factored AUGMENT drops exactly
4 of 91 terms: the covariance triangle T1 and three κ3 × edge terms (E3, E21a, E21b). Unfactored
restricted to the kept terms reproduces factored to float32 rounding. On 8 nets per width,
**G_drop (final layer) = 26.6 / 18.6 / 15.6 / 16.8 % at n = 256 / 384 / 512 / 768**, 8/8 wins
everywhere — flat from 384 on. T1 ≈ 0; the κ3 × edge set carries all of it (E21b alone ~80 %;
E3 or E21a alone make it worse). Strategist's 5–25 % / κ3×edge call was right; mine (3 %) wrong.
Two findings decide it: (1) on 16 DEV nets at n = 1024, ARC factored AUGMENT beats v73 raw 16/16
(1.79e-8 vs 2.32e-8 — known in kind, CLAIMS #39: AUGMENT costs ~8×); (2) E21b lands in the
OFF-DIAGONAL κ4 core, which v73 does not carry (memoryless λ·C_off). I wrote E21b in ARC's factored
form (exact vs unfactored to 2.3e-13) and restricted it to recent κ3 births: last 1 / 2 / 3 layers
keep 2.8 / 6.5 / 8.7 % (n = 512) and 2.5 / 5.6 / 8.0 % (n = 768) of the 13 %. In v73 any form also
needs an n×n κ4-core transport (≈ +14 % FLOPs), so the best case is ≈ +14 % FLOPs for ≤ 2.8 % or
≈ +27 % for ≤ 9 % — far below the ≥ 5 % at ≤ +5 % bar. **M3 = CLOSE**; write-up negative result
(`docs/research/2026-10-06_laneM_report.md`). Moving to M4 (neighbor-feature corrector screen).
