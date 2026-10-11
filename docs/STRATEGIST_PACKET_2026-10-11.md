# Strategist packet — Lane P6, 2026-10-11

Raw artifacts: `experiments/2026-10-11_p6/` (PREDICTIONS.md, rhocheck/, r2parity/, screen/, r3/). RESULTS § "P6".

## Daily paragraph (Oct 11)

**The kink-active H lever fails as specified, and the cause is identified.** Through the shipped `_kh` gather,
every count-table H variant is catastrophic unrefit on VAL-100: fold+Habs +65.7 %, Hasym2030 +44.5 %,
Hasym2025 +71.4 %, Habs20 +140 % (0/100 nets better in each). Twin arm and file agree to ≤ 0.016 %, so this is
real and not a gather bug. Decomposition on the R2 + fold arm: M5's threshold "H2.5 literal" (A_L/w2b/hubs off on
|α| > 2.5, residual rows and S3 off on dead columns only) costs **+1.22 %**. The same arm with `Hfull=1` costs
**+65.1 %**. **The P2 residual rows and S3 of the always-on columns carry the chain.** The estimator's lineage
holds one column set (`hcols`) for A_L, w2b, sb, e_b, cA, cB *and* Lb, so it cannot drop one without the other.
Deeper dead-only cuts are not free either: by count, τ2.0 is +5.95 % (−3.4 % F), τ1.75 +17 % and τ1.5 +44 % vs
fold, all unrefit. **v83 therefore = v82c_r2 + fold**, refitting on the pod (R3: `fold` vs a matched `ctl`, both
from R2). Separately, **R2 beat R1** by −0.46 % on VAL and −0.46 % on FRESH-A (96/100). It is submitted as
v82c_r2, **#335298**, projected 3.0275e-9. Its ws twin goes in the A/B slot instead of v82c_r1ws, which it
dominates. Housekeeping: P6.0a/b are done (lane-m merged, ρ reproduces exactly). Twin support is in:
`Hrank`, `Hfull`, shim `linalg.svd(k=)`/`eigh` (basis detached). `tools/build_v83.py` reproduces the metered
variant byte for byte.

## Ledger vs v81d (#334948: F/B 0.17629, raw 1.8833e-8 graded, 25,741 ops)

| build | ΔF cumulative | Δraw (best evidence) | ops | status |
|---|---:|---:|---:|---|
| v82c_r1 | −6.36 % (0.16508) | graded −2.01 % | 22,375 | graded 3.0464e-9 (#335253) |
| v82c_r2 | −6.36 % (0.16508) | FRESH-A −0.46 % vs v82c_r1 → ≈ −2.47 % | 22,375 | submitted #335298, proj. 3.0275e-9 |
| v82c_r2ws (B) | −4.16 % (0.16895) | ≈ v82c_r2 | 20,269 | FRESH-A running (A/B slot) |
| v83 = r2 + fold | −6.11 % (0.16554) | unrefit VAL −1.03 % vs r2; refit pending (M5: −2.58 % vs ctl) | 22,618 | R3 on pod, ETA ≈ 07:00 UTC |

## What the evidence changes in P6

- **P6.1 as written is closed:** habs and every hasym variant (τ2.5, τ2.0, −2.0/+3.0, −2.0/+2.5). The target
  "v83 ≈ 2.8e-9 by Oct 12" was ~6–7 % F from H. Without it, v83 is roughly v82c_r2 × (1 − 2 % refit fold gain +
  0.27 % F) ≈ 2.97–2.99e-9.
- **The only H route left is a two-column-set lineage.** Drop A_L/w2b/hubs on |α|-ranked columns, keep Lb/sb on
  every non-dead column. Unrefit cost +1.22 %; the F saving is unpriced, and it needs `_append` and every
  consumer of `hcols` split. That is strategist territory: please price it, or say no, before I build it.
- Dead-only H deeper than τ2.5 is closed (τ2.0 +5.95 % for −3.4 % F).
- Still on the P6 queue, unchanged: the P6.2 spend items from M6 (age 6, SVD join, J=3), P6.3 per-shape depth
  (−1.3 % F at leaf ≥ 12, which no longer stacks on H), and refit polish.

**Spend items and depth re-priced on the v83 stack (fold, no H; vs fold F/B 0.16554), exact interpreter:**
od12 −1.21 % F (24,976 ops) · od16 −0.68 % (24,688) · J3 +0.33 % (22,327) · age 6 +4.83 % (22,113) · SVD join
+1.07 % (21,754). These are within 0.1 % of the fold+Habs prices, so the break-evens in the P6 handoff table hold.

## Oct 11 afternoon/evening addendum

- **Graded today:** v82c_r2 **3.0323e-9** (#335298); v82c_r2ws **3.1018e-9** (#335319, new B, A/B slot);
  **v83_fold 2.9483e-9** (#335365; fold + R3 refit, −2.77 %); **v83_r4od16 2.9191e-9** (#335407, new best; R4 refit +
  per-shape Strassen depth leaf ≥ 16, −0.99 %). All four landed within +0.07 to +0.16 % of their FRESH-A projections.
  Four canaries used (3 + the A/B slot); target "v83 ≈ 2.8e-9 by Oct 12" is missed without H, but we are at −4.2 %
  vs this morning's best.
- **Refit polish keeps paying:** fold R3 −2.20 % vs matched ctl; continuations R4 −0.43 %, R5 −0.33 % (VAL). R6 running;
  by the restart rule the recipe stops after two consecutive restarts < 0.3 %.
- **P6.3:** od12 closed on wall (1.25×, ≈ 112.5 s); od16 shipped (1.133×, ≈ 101.9 s noisy-implied, raw +0.075 %).
  The A wall margin is now 3 % under the 105 s gate, so no further wall-costing item stacks on A without a wall cut.
- **B line:** v83_r5ws (fold, block 40, wall 1.003× v82c_r2ws ≈ 78 s) on FRESH-A for the Oct 12 A/B slot.
- **Ledger vs v81d:** v83_r4od16 ΔF −6.74 % (F/B 0.16441), graded Δraw −5.73 %, ops 24,688 (−1,053).
- **Still needed from the strategist:** go/no-go on the two-column-set H lineage (see above); the M6.2 spend items
  (age 6 +4.83 % F, SVD join +1.07 %, J3 +0.33 %) need their twin evidence before P6.2 can integrate them.
