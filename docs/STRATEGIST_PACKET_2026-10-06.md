# STRATEGIST PACKET — 2026-10-06 (Lane P, session 1)

**P0 done.** v74_ws (= v73_gb at Strassen leaf 64, nothing else) passed every rev 18 §5 hard
gate and graded #334434 at **4.976094e-9** (raw 2.151578e-8, +0.006 % vs v73; F/B 0.23128;
projected 4.9632e-9, +0.26 %). It is the new wall-safe B; A stays v73_gb #332893 4.6199e-9.
PLAN rev 18 is committed as `docs/PLAN.md`, and main is pushed to origin. Samuel designates.

**P1 mask table** (twin, VAL-100 free-running, unmasked control 2.21336e-8 on the same code;
modeled F from the job2309 census × FIT-100 kept fractions; full table in `RESULTS.md`
§ "PLAN rev 18 — Lane P"):

| config | Δ raw | modeled F | strategist prediction |
|---|---:|---:|---|
| R2.0 / R2.5 | +0.90 / +0.06 % | 0.926 / 0.946 | R2.0 ≤ +0.5 % |
| H3.0 literal / Hk3.0 | +2.01 / +2.00 % | 0.936 / 0.959 | H3.0 ≤ +0.5 % |
| **Hd3.0 / Hd2.5** (dead columns only, every part) | **+0.03 / +0.15 %** | 0.959 / 0.936 | — |
| D3.0 / D2.5 | +7.09 / +10.55 % | 0.934 / 0.910 | D3.0 ≤ +0.5 % |
| Dc3.0 / Dc2.5 (corrector reads unmasked D21) | +0.03 / +0.56 % | — | — |
| safe R2.0/H3.0/D3.0 | +9.91 % | 0.812 (F/B 0.174) | ≤ +1.5 %, F −20 to −30 % |
| aggressive R2.0/H2.5/D2.5 | +15.12 % | 0.763 (F/B 0.164) | +3 to +8 % |

Three findings change the P2 design:
1. **Only dead columns can be dropped at birth.** The literal H rule's whole cost is the κ4
   hubs on *always-on* columns (`cA = w1 (m/8) λ` with `w1 ≈ 1`). Hkh3.0 (hubs kept there)
   costs 0.00 %; the two-leg block alone (Hw) and the slice alone (Hs) cost nothing. Dropping a
   dead column entirely (A_L, M, two-leg, hubs, slice) is the clean compact form and is nearly
   free. Always-on columns cannot be dropped without losing the hubs.
2. **D21's cost is the mean corrector, not the closure.** The corrector's `d21e = rowsum(D21²)/σ⁴`
   runs over all columns, and a dead column does not make the pre-activation κ(i,i,j) small.
   With the feature unmasked, D3.0 costs +0.03 %. The refit has to absorb it, because the
   compact chain can only serve the kept-column sum.
3. **The modeled cost floor of these masks is F/B ≈ 0.17**, not 0.14–0.16. The model credits
   only the matmul sites (transport, D21 readout, old tier). Mixed elementwise reads and the
   full-row old-leg reconstruction (D3 reads every row) stay at full price.

Also checked for P2: real flopscope selects (`argsort`/`argpartition`) and gathers (`take`,
fancy indexing) with **0 materializations**, and a gather bills **8 FLOPs per output element**
(the handoff assumed 4). Scatter-back is a zero-pad plus a double `take` (17 M FLOPs at n×n).

**Refits** (S3 stage = corrector + all gains, init G1bg = v73, lr 3e-4, 2 epochs TRAIN-900,
VAL selection), running: safe-d R2.0/Hd3.0/D3.0 (init +8.2 %), aggr-d R2.0/Hd2.5/D2.5, and an
unmasked control with the same recipe. Kill rule: safe-d > +3 % raw vs the control.
R decomposition (inherited vs newborn rows) on CPU in parallel.

## Update, evening — P1 closed, P2 shipped (v75 graded new best)

**Two corrections to the morning table.**
1. The first row-only D runs were measured with the corrector reading the *unmasked* D21 (a
   twin bug, found by v75's parity check). True row-only D costs +2.5 % at τ 2.5. The fix is
   **Dz**: no corrector delta on the dropped rows (+0.62 % at 2.5, +0.02 % at 3.0).
2. **Refit:** the prescribed S3 refit (lr 3e-4) spikes +7 % at step 28 and returns to init by step
   84 on converged params, the same as Sep 28's S3c. It cannot beat its own control, so no P1
   refit was run to completion. With Dz the mask shift is small anyway.

**Frontier** (thresholds, VAL-100, score = raw × modeled F): R2.0/Dz2.5 is the optimum (0.885).
Both roles have a cliff below τ 2 (R1.5 +12 %, Dz2.0 +10 %).
**Fixed K** (variant A, zero materializations) at any quantile q50–q90 scores the same, about
0.907: keeping more neurons buys back what it costs.

**Graded and submitted (all PLAN §5 hard gates; acceptance on DEV-100 + VAL-100 pending FRESH-A):**

| | F/B | DEV raw | projected | graded |
|---|---:|---:|---:|---:|
| v75_kas (R + D rows + Dz, block 20) | 0.19020 | 2.2769e-8 | 4.1722e-9 | **4.1871e-9 (#334464), new best, −9.4 %** |
| v76_kasws (v75 at block 40, the B) | 0.20416 | 2.2764e-8 | 4.4774e-9 | pending (#334474) |
| v77_kash (v75 + Hd dead birth columns) | 0.17981 | running | ≈ 3.95e-9 expected | not yet submitted |

v77's VAL-100 paired vs v75: score ratio 0.944, 100/100. Its wall is a measured-parent projection
(local 0.897 × v73's server median 83.3 s ≈ 74.7 s; noisy 92.7 s).

**Against the strategist's P2 prediction** (F/B 0.145–0.165 at raw ≤ +1.5 %, graded 3.2–3.6e-9),
reality is F/B ≈ 0.180 at +2.6 %, about 3.95e-9. Three reasons, all measured:
- always-on columns can't be dropped (their hubs);
- the corrector's D21 feature limits D;
- fixed counts and the Strassen leaf floor (block 20) eat part of the proportional saving.

The next lever, which the masks cannot reach, is structural: the transport and D21 readout
still run at ~0.8 n on their inner dimension.
