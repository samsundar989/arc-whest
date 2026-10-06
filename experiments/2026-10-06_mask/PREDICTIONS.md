# P1 mask test — predictions (written 2026-10-06, before the VAL-100 grid ran)

Code: `tools/twin/twin.py` arm `mask` on `estimator_k3_v73_gb.py` (`--raw`), driver
`tools/twin/mask_grid.py`. Panel VAL-100 (`hold100` = full 900–999), free-running, raw
final-layer MSE vs the unmasked control on the same code. Control with the arm loaded and no
mask is bit-identical to plain v73 (net 900: 2.480706895155988e-08 both).

**Disclosure.** Before writing these numbers, one smoke net (full 900, CPU) was run with the
literal aggressive set R2.0/H2.5/D2.5: **+12.8 % raw**. That datum informs the H and aggressive
lines below.

**Second disclosure (smoke, same net 900, after the table below was written, table unchanged):**
R2.0 +0.75 %, Hk3.0 +1.3 %, D3.0 **+10.1 %**, safeK +11.8 %. D's cost is the mean corrector:
its `d21e` feature is `rowsum(D21²)/σ⁴` over ALL columns, and a dead column j does not make the
pre-activation κ(i,i,j) small, so masking shifts the feature for live neurons. Diagnostic `Dc`
(D21 masked everywhere except the corrector feature): D3.0 → **+0.24 %**. So the pre-refit D
number measures the corrector's input shift, and the decisive D number is after the refit (the
compact chain can only serve the kept-column feature). Grid gains `Dc2.5`, `Dc3.0`, `safeKc`;
worker predictions for those: +0.5 %, +0.3 %, +1.5 %.

**H has two readings, both run.** The handoff zeros `A_L[:, j]` for every decided column but
keeps `eb_j` for always-on ones; `eb` is carried on `A_L[:, j]`. `H` = literal: `A_L[:, j]`
dropped for all `|α| > τ`, `e_b`'s separable S21 row folded into the rank-4 P2 residual.
`Hk` = `A_L[:, j]` kept for always-on columns (carrying `e_b` only), two-leg `w2b_j` and hubs
zeroed. Hk saves less transport (always-on A_L columns stay) but should cost less accuracy.

| config | strategist | worker (mine) |
|---|---|---|
| R2.0 | ≤ +0.5 % | +0.3 % |
| R2.5 | — | +0.15 % |
| H2.5 | — | +7 % |
| H3.0 | ≤ +0.5 % | +3 % |
| H3.5 | — | +1.5 % |
| Hk2.5 | — | +1.0 % |
| Hk3.0 | — | +0.4 % |
| Hk3.5 | — | +0.2 % |
| D2.5 | — | +0.6 % |
| D3.0 | ≤ +0.5 % | +0.3 % |
| safe R2.0/H3.0/D3.0 | ≤ +1.5 %, FLOPs −20 to −30 % | +4 % |
| safeK R2.0/Hk3.0/D3.0 | — | +1.2 %, FLOPs −15 to −22 % |
| aggr R2.0/H2.5/D2.5 | +3 to +8 % | +12 % |
| aggrK R2.0/Hk2.5/D2.5 | — | +2.5 % |

**Stop rule (handoff).** If the better of safe/safeK is > +3 % raw after the short refit, the
lane closes: record the table and go to P3 with the remaining time.
