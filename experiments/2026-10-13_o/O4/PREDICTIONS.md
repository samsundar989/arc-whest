# O4 predictions (recorded 2026-10-13, before the run)

Base: v87 candidate = `fold+age6` + `sb2+sbtrim+depth8+k0:unrefit+k1:2.5_3.0+k2:3.0:unrefit+k3` (Lane K stack B
on Lane S's sb2/sbtrim/depth8). Atoms are uncommitted copies from lane-k @ ed62f4f and lane-s @ 408d409, plus K's
tables via a junction. R8 params, unrefit. Prices are from Lane K's out=-aware meter (`variants/prices.txt`); the
base reproduces K's 0.14250 / 25,211 ops exactly. Accuracy: unrefit VAL-100, twin `--raw` on the built files,
paired against the v87 file. Ops gate (P7.1): ≤ 26,100.

| arm | F/B | ΔF | break-even Δraw | ops (Δ) |
|---|---:|---:|---:|---:|
| v87 | 0.14250 | — | — | 25,211 |
| age 7 (`--base fold+age7`) | 0.14784 | +3.75 % | −3.61 % | 24,476 (−735) |
| J3 (`fold+age6+J3`) | 0.14256 | +0.04 % | −0.04 % | 24,780 (−431) |
| J4 (`fold+age6+J4`) | 0.14375 | +0.88 % | −0.87 % | 24,309 (−902) |
| SVD join (`+o1`) | 0.14465 | +1.51 % | −1.49 % | 23,782 (−1,429) |

On the pruned stack J3 is nearly free in F (K pruned the late joins' inputs) and every arm lowers ops.

**Strategist (handoff):** age 7 −3 to −4.5 % raw for ≈ +3.5 % F; marginal.

**Lane O (mine):**
- age 7: −2.5 to −4.0 % raw; P(clears −3.61 %) ≈ 0.35.
- J3: −0.15 to −0.4 % (R2 on age 6: −0.26 %); P(clears −0.04 %) ≈ 0.85. This is the likely winner: net −0.1 to
  −0.35 %, with 431 fewer ops.
- J4: −0.3 to −0.7 %; P(clears −0.87 %) ≈ 0.2.
- SVD join: −0.1 to −0.3 % (O1: −0.24 % on v84); P(clears −1.49 %) ≈ 0.02.
