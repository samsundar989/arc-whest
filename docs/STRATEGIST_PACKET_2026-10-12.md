# Strategist packet — Lane P6, 2026-10-12

Raw artifacts: `experiments/2026-10-11_p6/` (r3/ … r8/, screen/, *_fresh_a.*, mem_*, *_wall_ab.txt). RESULTS § "P6".

## Daily paragraph (Oct 12)

**Refit polish is exhausted, and age 6 is in.**
- **The fold refit chain is done.** It ran R3 → R7: R4 −0.43 %, R5 −0.33 %, R6 −0.26 %, R7 −0.22 % on VAL. The restart rule
  stopped it after R7, and R8's control moved only +0.02 % from R7.
- **R7's constants graded 2.8942e-9 (#335484, v83_r7od16, best graded).**
- **Age 6 (M6's hand-over) refit from R7:** −6.15 % raw vs the matched control (R8). FRESH-A vs v83_r7od16: −5.84 % raw,
  1.0062× score, projected 2.8648e-9. **Only 53/100 nets win:** the raw gain barely covers age 6's +5.55 % F, so its net is
  small but positive on average (t95 just excludes 0). Submitted as #335562.
- **Age 6 and od16 do not stack on A.** On wall, age6 alone is 0.931× r7od16 (≈ 94.9 s noisy-implied); age6 + od16 is 1.033×
  (≈ 105.3 s), which fails A ≤ 105 s. So v84 drops od16 and gains back ~7 s of wall margin.
- **Memory:** age 6 on A passes (VmPeak 6,773 MB). The block-40 ws twin fails the settled-growth check (+476 MB over 5
  predicts, +172 MB over 15: non-monotone allocator steps), so **B stays age-free**. B is v83_r5ws (2.9952e-9, #335461), and
  v83_r7ws (1.0052× vs r5ws, 90/100) is queued for the Oct 13 A/B slot.

## Ledger vs v81d (#334948: F/B 0.17629, raw 1.8833e-8 graded, 25,741 ops)

| build | ΔF | Δraw (graded unless noted) | ops | graded |
|---|---:|---:|---:|---:|
| v83_fold | −6.10 % | −5.43 % | 22,618 | 2.9483e-9 |
| v83_r4od16 | −6.74 % | −5.73 % | 24,688 | 2.9191e-9 |
| v83_r5od16 | −6.74 % | −6.08 % | 24,688 | 2.9082e-9 |
| v83_r7od16 | −6.74 % | −6.53 % | 24,688 | **2.8942e-9** |
| v84_age6 | −1.56 % | FRESH-A ≈ −12.0 % | 22,113 | pending #335562 (proj. 2.8648e-9) |
| B v83_r5ws | −3.90 % | −6.12 % | 20,512 | 2.9952e-9 |

## Needs a strategist decision

1. **Two-column-set H lineage** (Oct 11 packet). This is the only H route left; −6 % F was the original prize. Go or no-go?
2. **The A wall margin is now the binding constraint for stacking.** On v84 (≈ 94.9 s) there is about 10 s for
   wall-costing items: od16 (+7 %), or jsvd as an age-6 substitute (M6: substitutes, +0.74 % marginal). Which item
   should spend it?
3. **P6.5 adaptive keep counts** (about −1 % raw at about 0 F): worth building in the 4 days left before the Oct 17 freeze?
