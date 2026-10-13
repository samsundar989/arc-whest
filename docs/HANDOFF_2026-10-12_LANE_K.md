# HANDOFF — Lane K (new session, branch `lane-k`), 2026-10-12: pruning v2 — kink-aware index sets with analytic gain replacement

**Read first:** `docs/research/2026-10-12_deep_review.md` §0, §5.2, §5.3, §6. Data and tools: `docs/research/2026-10-12_gain/` (the label analysis: `gain.py`, `extra.py`, `tables_*.txt`) and `docs/research/2026-10-12_costfloor/` (`x3lib.py` = exact-anchor text patches on v84 that give every scenario below the right *shapes*; `x3_tables.txt` = all prices; `alpha_stats.npz` = per-net α for FIT-100).

**Setup.** `git worktree add ../arc-whest-lane-k -b lane-k main`. You build arms in the twin on the v84 arm (R8 parameters) and hand each to Lane P as a patch + twin parity + VAL-100 numbers (unrefit, and refit vs a matched control) + interpreter F and ops. GPU: the local 5090 through gpuq; pod 2 for refits (Lane P's queue has pod 1).

**Standing rules.** Predictions on record before every run. Select on VAL-100; acceptance is FRESH-A by Lane P. No fitting or selection on FRESH panels. Every refit comparison uses a same-recipe control with matched epochs. No external messages. No write-up work. Kept counts stay on the Strassen grid (multiples of 32 now; agree multiples of 64 with Lane S, handoff S4).

## Why
Three prunings were closed on unrefit cliffs: always-on D21 rows (+34 %), dead leg rows (+66 %), kink-active H by count (+65 %). The exact labels now say what was lost each time:
- The gain mode is **one scalar v per layer for all of κ3** (D3 and D21 agree to ≤ 1.5 %): D3_a ≈ 6v·μ_a σ_a², D21_ac ≈ 2v·(2 μ_a C_ac + μ_c σ_a²). It is 77–84 % of D21's energy at depth and 95 % of D3 on decided neurons.
- Zeroing always-on + dead D21 rows at τ = 2.5 moves next-layer variances by 1.2–1.8e-4 (tolerance ≈ 1e-4); filling those rows with the gain formula leaves 5–8e-5. At τ = 3.0: 4.3e-5 zeroed, 1.7e-5 filled. At τ = 2.0 it fails either way (2.7e-4).
- Zeroing dead neurons' D3 at τ = 2.5 costs 3.3–4.4e-5 rms in the mean (the old "+66 %"); 6vμσ² leaves 1.0–1.4e-5; at τ = 3.0, ≤ 3.5e-6.
- Dead D21 columns at τ = 2.5 are free (1.2e-5 zeroed). **Always-on columns are essential** (9.2e-3): never prune them.
- v fitted on kept rows only gives the same result (within 2 %), so the chain can compute it from its own D3.

These are single-layer numbers with everything else exact. The chain has 37k fitted numbers tuned to today's masks, so every arm needs a matched-control refit before it is judged.

## In-chain v
Per layer, after the D3 readout: v = Σ_a γ3_a α_a / (6 Σ_a α_a²) over kept kink neurons, γ3 = D3/σ³ (O(n)). Give it a per-layer learned gain initialised at 1. Normalised forms: γ3 = 6vα; D21_ac/(σ_a² σ_c) = 2v(2 α_a ρ_ac + α_c).

## K0 — gain-fill the D21 rows we already drop (free; Oct 13 first)
Today dead D21 rows (D set, τ 2.5) are zero and the corrector delta is skipped there. Fill them with the gain formula (O(n²) elementwise, no product), compute `d21e` for those rows from the formula, and keep the corrector delta on. Labels: 8–12e-5 → 4–5e-5 variance error at L8–L14.
**Prediction (strategist):** unrefit −0.3 to −1.5 % raw; after refit −0.5 to −2 %. Kill: not better than the matched control.

## K1 — D21 rows = kink-active, gain-filled elsewhere
Rows a with |α_a| ≤ τ_r keep the lineage readout; all other rows get the gain formula. Columns stay the non-dead D set. Arms: τ_r = 3.0, the asymmetric −2.5 ≤ α ≤ 3.0, and 2.5.

| arm | ΔF (exact) | label error (max over layers, filled) |
|---|---:|---:|
| \|α\| ≤ 3.0 | −2.5 % | 1.7e-5 |
| −2.5 ≤ α ≤ 3.0 | −4.2 % | ≈ 5e-5 (dead rows 5.2e-5, always-on rows 1.2e-5) |
| \|α\| ≤ 2.5 | −5.9 % | 7.6e-5 |

`d21e` and the ρ calibration (`_D21E_RHO`) must be recomputed for the new row set (`experiments/2026-10-10_p5/d2/compute_rho.py`). D21 has five other consumers (dressed leg H, births, κ4 core, closure, fold): the filled rows feed them the gain part, which is the right first-order content for a near-linear neuron.
**Prediction:** τ 3.0 unrefit ≤ +1 %; τ 2.5 unrefit +1 to +4 %, after refit 0 to +1.5 %. Accept on the marginal rule after refit.

## K2 — leg output rows α > −τ_o, gain D3/D4 on the dropped rows
Today `_advance` produces all 1,024 rows. Compute μ_pre and C_pre first (they do not depend on the tower), take α, and transport only output rows with α > −τ_o; the old rebuild uses the same rows. On dropped rows set D3 = 6vμσ² and D4 = 12·v₄·σ⁴, with v₄ fitted on kept rows from the chain's own D4 (D4 needs its own scalar: 0.5 v at L1 rising to 1.0 v at L15). Arms τ_o = 3.5, 3.0, 2.5: ΔF −3.3 %, −4.6 %, −6.2 % (the last layer included).
**Prediction:** τ_o 3.0 unrefit ≤ +1.5 %; 2.5 unrefit +3 to +6 % (labels: +3–5 % MSE from the mean channel alone).

## K3 — two-column-set H
As specified in the Oct 11 packet: A_L, w2b and hubs only on birth columns with |α_b| ≤ 2.5; M, sb, e_b and the residual rows L on every non-dead column. Measured +1.22 % unrefit (P6.1). Exact price: **−3.13 %** F (`x3lib.py` scenario S2b; τ 2.0 would be −6.4 %, τ 3.0 −0.9 %). Split `_append` and every consumer of `hcols`.
**Prediction:** after refit 0 to +0.8 %.

## K4 — last layer: transport only the youngest lineages
L15 costs 8.4 u for D3 and star-D4 alone. Transport only the K youngest lineages into L15 (K = 4, then 2) and supply the rest of D3 by a gain term: at L14, where both are available, fit v_rest from D3_full − D3_youngest-K; at L15 use g·v_rest·6μσ² with a learned scalar g (init 1.02). ΔF −3.0 % (K = 4) to −3.8 % (K = 2) with the old tier; the saving is mostly the 2.2 u old rebuild.
**Prediction:** unrefit +2 to +8 % (kink neurons' D3 is only 66–73 % gain-mode, and the last layer is the scored one); P(passes the marginal rule after refit) ≈ 0.35. Screen it early; do not let it block K0–K3.

## K5 — optional: per-network counts
Fixed q50 tables cut live neurons on half the nets (fixed counts cost ~1.1 points vs thresholds in P1; the kink count's CV is 6.8 % at L15). One sync per layer returning all counts, rounded to the Strassen grid. Mean F is unchanged; the gain is accuracy. Only after K0–K3 are handed over, and only if the residual budget allows 16 syncs.

## Order and hand-over
1. Oct 13: K0, K1 (three arms), K2 (three arms), K3 as unrefit VAL-100 screens (`twin.py eval --raw`; use the `x3lib.py` patches for the shape plumbing and add the numerics).
2. Oct 13–14: one joint refit of the best stack vs a matched control on pod 2 (corrector + gains + the new v gains; cosine, lr ≤ 1e-4, 2 epochs).
3. Hand to Lane P per arm and as a stack: patch atoms, twin parity, VAL (unrefit; refit vs control), interpreter F and ops, VmPeak.

**Stack targets (exact prices, age 6, old tier kept):** K1(3.0 asym) + K2(3.0) + K3 ≈ −11 % F; K1(2.5) + K2(2.5) + K3 ≈ −14 to −15 %.

## Do not
- Prune always-on D21 **columns**, or any τ < 2.5 on rows (labels: fails with or without the gain fill).
- Use the gain formula as the whole D21 (2–4e-3 error) or in place of kink neurons' D3.
- Reopen covariance pruning (×3–13) or `Hfull` (+65 %).

## Report
A daily paragraph in `docs/STRATEGIST_PACKET_2026-10-1X_laneK.md`; RESULTS-style tables under `experiments/2026-10-13_k/`.
