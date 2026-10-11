# Hand-over to Lane P: exact SVD join basis (M6.2), 2026-10-11

**Lever.** The old-tier join basis becomes the exact top-r eigenspace of J·D·Jᵀ + Q·Sg·Qᵀ, via a
top-k SVD. Today's basis is one warm-started subspace-iteration pass.
- price.py name: `fold+jsvd` (the `join_svd` patch in `docs/research/2026-10-11_p6price/price.py`).
- flopscope bills the top-k SVD at 4mnk, its randomized-SVD price, while computing it exactly.
- Twin candidate: `experiments/2026-10-11_m6/refit/cand_jsvd.py`.

## Evidence (VAL-100, paired per net)

| step | arm | control | Δraw | per-net Δ ± se | wins |
|---|---:|---:|---:|---:|---:|
| unrefit, R2 files | 1.85469e-8 | fold 1.88375e-8 | −1.54 % | −1.53 ± 0.19 % | 79/100 |
| twin arm, R2 params (parity) | 1.854913e-8 | — | file +0.012 % | — | — |
| unrefit, R3 fold params | 1.822418e-8 | ctl 1.850971e-8 | −1.54 % | — | — |
| **refit, 1 epoch from R3 fold vs a matched control** | **1.820761e-8** | **1.848205e-8** | **−1.48 %** | **−1.47 ± 0.19 %** | **77/100** |

Same recipe and control as the age 6 hand-over (`HANDOVER_M6_age6.md`). In the refit the basis is
detached; only the singular values carry a gradient.

## Cost (exact interpreter, on fold)

| | F/B | ΔF | ops |
|---|---:|---:|---:|
| fold (v83) | 0.16554 | — | 22,618 |
| fold + jsvd | 0.16731 | +1.07 % | 21,754 (−864) |

- Break-even Δraw: −1.06 %. **Net score: −0.43 %.**
- **Wall time is not measured.** The strategist estimated +1–2 s per MLP for the exact SVD. Ops
  fall by 864, which offsets some of that. `tools/wall_ab.py` decides.

## Stacking with age 6: measured; do NOT stack

Joint unrefit screen (job 2647): fold+age6+jsvd gives 1.77087e-8 at F/B 0.17507. Against fold+age6
that is −0.14 % raw for +0.88 % F, a **+0.74 % marginal net (worse)**, with 54/100 wins. Both
levers repair the old-tier compression error at join time, so they are substitutes.

**Use jsvd only as the fallback if age 6 fails its wall-time or memory gate.**

## Twin note

The refit first stalled (job 2626): cuSOLVER's default SVD driver failed to converge. The twin
shim now decomposes outside autograd with the `gesvd` driver (commit 280e2e8, lane-m). Parity is
unchanged to every digit. The estimator side uses flopscope's own SVD, so this affects only the
twin.
