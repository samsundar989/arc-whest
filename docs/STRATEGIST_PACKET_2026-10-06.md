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

**M (dropped diagrams).** Factored AUGMENT drops exactly 4 of 91 terms: the covariance
triangle T1 and three κ3 × edge terms (E3 top slice, E21a/E21b (2,1,1) analogues). Unfactored
restricted to the kept terms reproduces factored to float32 rounding. On 8 nets per width,
**G_drop = +26.6 % [16.5, 36.6] at n = 256 and +15.6 % [7.8, 23.5] at n = 512**, 8/8 wins each.
T1 contributes ≈ 0; the κ3 × edge family carries all of it, but only as a set (E3 or E21a alone
make it worse; E21b alone gives most). Strategist predicted 5–25 % with κ3 × edge dominant —
right; I predicted 3 % — wrong. The trend is down (×0.59 per doubling), so the GO rule's "flat
or growing" clause currently fails; n = 384 and 768 (n = 1024 does not fit in VRAM unfactored)
are queued to settle it before the Oct 8 decision.
