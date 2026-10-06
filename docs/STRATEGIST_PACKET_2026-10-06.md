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

*Refit results: appended below when read.*
