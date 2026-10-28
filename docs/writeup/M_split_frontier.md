# M — SPLIT family error-vs-cost frontier (exact κ3 hub-star lineages)

Script `/home/claude/sandbox/split_frontier.py` (imports pooled_carrier; legs built once per (n, seed), SVD of the band once per (E, layer), r swept on top). Grid: E ∈ {2..8}, r/n ∈ {0.14, 0.21, 0.28, 0.375, 0.5, 0.56, 0.75}; n ∈ {128, 256}; depth 16; seeds 1, 2; errors = rms over layers 9–15 of the sensitivity-weighted D3 (e3w) and D21 (e21w) errors vs the full lineage sum, seeds pooled (rms). Cost in n³ units/layer, mean over layers 9–15. Full tables: `/home/claude/sandbox/results/split_frontier_n{128,256}.{json,log}`. Runtime 9 s (n=128) + 31 s (n=256).

Cost models (both divided by n to give n³ units): **cost1** = 4·nex + 6r (pooled_carrier); **cost2** = 4·nex + 2r + 4r + 0.5·r²/n·(n_old/n), nex = KAS-kept exact columns (ages ≤ E), n_old = KAS-kept columns in the band (ages > E).

## Error floor
Every (E, r) converges to the same floor, which is exactly the KAS truncation (|α| ≥ 2.5 hubs dropped from both tiers, verified with all ages exact): **n=128: e3w 0.0245 / e21w 0.0217; n=256: e3w 0.0259 / e21w 0.0237.** The frontier question is therefore about the *excess* over that floor.

## Pareto frontier at n = 256 (1 % tolerance on error)
Under **cost1** (e3w and e21w frontiers are identical):

| E | r/n | cost1 | e3w | e21w |
|---|-----|-------|-----|------|
| 2 | 0.14 | 6.33 | 0.2149 | 0.2697 |
| 2 | 0.21 | 6.75 | 0.1318 | 0.1638 |
| 2 | 0.28 | 7.17 | 0.0812 | 0.0978 |
| 2 | 0.375 | 7.73 | 0.0430 | 0.0498 |
| 2 | 0.50 | 8.48 | 0.0273 | 0.0261 |
| 2 | 0.56 | 8.83 | 0.0262 | 0.0242 |
| 2 | 0.75 | 9.98 | 0.0259 (floor) | 0.0237 (floor) |

Under **cost2**: same seven E=2 points (cost2 = 6.41, 6.94, 7.52, 8.35, 9.58, 10.21, 12.46); for e21w the last point is replaced by (E=3, r=0.5n, cost2 12.45, e21w 0.0239) — i.e. once the r² readout term bites, r=0.75n at E=2 and r=0.5n at E=3 cost the same and reach the same floor. No E ≥ 4 point is on either frontier under either cost model.

## Where (E=6, r=0.28n) sits
- n=256: cost1 19.80 / cost2 20.03, e3w 0.0266, e21w 0.0250 (excess over floor 7e-4 / 13e-4). Cheapest point matching or beating that error: **E=2, r=0.56n at cost1 8.83 (45 % of the cost) / cost2 10.21 (51 %)**, e3w 0.0262, e21w 0.0242. Best point at ≤ its cost: E=5, r=0.5n (cost1 17.75) sits on the floor. Same on both seeds individually.
- n=128: cost1 21.30, e3w 0.0248, e21w 0.0226; E=2, r=0.56n matches it at cost1 9.58 (45 %), cost2 11.02 (51 %).
- Sandbox analogue of the zero-shot fact: E=6, r 0.28n→0.375n costs +2.8 % (cost1) and lowers e3w by 2.4 %, e21w by 5.0 % at n=256 (+2.6 %, −1.1 %, −3.6 % at n=128) — same sign and magnitude class as the real +1.86 % cost / −1.2 % MSE.

## Excess over floor ×1e4 (e3w / e21w), n = 256
| r/n | E2 | E3 | E4 | E5 | E6 | E7 | E8 |
|-----|----|----|----|----|----|----|----|
| 0.28 | 553/741 | 230/334 | 85/126 | 31/46 | 7/13 | 2/5 | 1/1 |
| 0.375 | 171/261 | 39/67 | 6/16 | 3/3 | 1/1 | 0/0 | 0/0 |
| 0.50 | 14/24 | 1/2 | 0/1 | 0/0 | 0/0 | 0/0 | 0/0 |
| 0.75 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

Each extra exact age costs ≈ 2.6 (n=256) to 3.1 (n=128) cost units; each +0.1n of rank costs 0.6 (cost1) to ≈ 1.0 (cost2 at r≈0.5n). At r ≥ 0.5n the rank buys down the excess ~4–10× faster per cost unit than an exact age does.

## n = 128 → 256 trend
The frontier is the same seven E=2 points at both widths, the floor-matching point is E=2, r=0.56n at both, and the cost ratio vs (E=6, 0.28n) is 45 % / 51 % at both. The excess at fixed (E, r/n) grows mildly with n (E=2, r=0.5n: 9→14e-4; E=6, r=0.28n: 3→7e-4), so if anything a larger n needs slightly more rank at fixed r/n, but it does not change which direction the frontier runs.

## Reading
At equal cost, fewer exact ages with a higher-rank band wins decisively under both cost models: the whole frontier is E=2 with r climbing from 0.14n to 0.56–0.75n, and (E=6, r=0.28n) is dominated by a point at about half its cost (E=2, r=0.56n), with E=3, r=0.5n as the alternative once the r² readout term is charged. The direction to move from today's point is toward more rank and fewer exact ages, not more rank at fixed E — the +0.1n rank step at E=6 is a 2–5 % error gain for 3 % cost, whereas trading four exact ages for +0.28n of rank is a ~55 % cost cut at equal error. Two caveats temper the transfer to the real chain: the sandbox band basis is the oracle SVD of the stacked old legs at each readout layer (an upper bound on what a transported basis achieves — a worse basis pushes the sweet spot toward E=3–4 rather than back to E=6), and the sensitivity-weighted errors are floor-dominated by the KAS cut, so a decision on the real chain should be gated on the excess-over-floor numbers (E=3, r=0.5n and E=4, r=0.375n are the conservative candidates at 58 % and 70 % of today's cost1).
