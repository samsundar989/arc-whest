# Hand-over to Lane P — M5.4: kink-active H columns WITHOUT moving the hub, 2026-10-10

**The hub move is closed** (Step 1 fidelity: age 1 R² 0.95–0.99 with γ₁C̃ + γ₂Wᵀdiag(v)W, but ages ≥ 2
0.5–0.8 and falling with age → fails "R² ≥ 0.8 at every age"; `hubfid_report.md`).

**But the lever it was meant to unlock is available directly.** On today's chain, dropping the always-on
birth columns outright costs far less than P1 measured on v73:

| twin v81d, VAL-100 | unrefit | refit (S2ts 1 ep, lr 1e-4, from starall2) vs matched control |
|---|---:|---:|
| Hd2.5 by threshold (reference) | −0.09 % | — |
| Hkh2.5 (hubs kept on always-on columns) | +0.09 % | — |
| Hk2.5 (hubs dropped on always-on, A_L kept) | +1.37 % | — |
| **H2.5 literal** (all of A_L, w2b, hubs, slice dropped on |α| > 2.5) | +1.59 % | **+1.35 %** (1.967743e-08 vs 1.941552e-08) |

With the interpreter's −6.03 % F alone (−5.65 % on top of the bundle), the net is ≈ **−4.8 % score**
(1.0135 × 0.9397). The kill bar for the hub item was +2 % after refit.

Caveats:
- The twin test uses the per-net THRESHOLD |α| > 2.5 (`--mask R=2.0,H=2.5,Dz=2.5,K=ktab_RD.json`); the
  −6.03 % F is for the KAS count table (1024, 1024, 1024, 992, 928, 896, 832, 800, 768, 736, 704, 640,
  608, 576, 576, 544 = FIT q50 kink-active at τ 2.5). The twin's K-table mechanism ranks by α only (it
  can keep the top-K but cannot drop always-on columns by count), so the count version's accuracy is
  unmeasured; P's build must rank columns by |α| distance from the kink.
- Interaction with the fold (M5.1) is not measured: the fold reads the star over the recent columns, which
  H drops. Expect a little of the fold's gain to go with the dropped columns.
- 1 epoch only; refits keep paying ~0.5 %/epoch on both arms.

Twin command: `experiments/2026-10-10_m5/batch.py refit hlit25 S2ts --mask "R=2.0,H=2.5,Dz=2.5,K=<ktab_RD.json>"`;
params `refit_hlit25/params_best.npz`.

## Update — joint refit (fold rec + H2.5 literal, S2tsf 1 ep from starall2)
VAL 1.916160e-08: −1.31 % vs the control (1.941552e-08), −2.62 % vs H2.5 literal alone (1.967743e-08).
The fold keeps its whole gain when the always-on columns are dropped. Stacked estimate ≈ −7 % score
(raw −1.3 %, F −6.03 % + 0.25 %). Params: `refit_foldhlit/params_best.npz`.
