# Strategist response — 2026-09-22: the single-leg tower

*Answers `docs/STRATEGIST_PACKET_2026-09-22.md`. Amends `docs/PLAN.md` rev 12: D5/D6 are replaced by the program in §5 below; everything else in PLAN (gates, loop, standing rules, D7/D8) stands. Labels: **[M]** measured (packet, RESULTS.md, code read today), **[I]** inference, **[?]** unknown until measured.*

## 0. Verdict on the day

Good day. 8.75e-9 → 5.56e-9 with three clean graded rows, and four theses of mine falsified by clean measurement: fitting is a 2.7% lever, billing polish 2.3%, K4 and unfactored AUGMENT lose on score, and *lossy* compression of old sources is catastrophic while *exact* old sources are worth −15–21% raw. I accept all of it.

One correction to the packet's reasoning, because the whole next step turns on it. The rank/age table proves that **lossy** treatment of old sources is fatal. It does not say anything about **cheaper exact** treatment of them; H1 and H4 were struck by inference from that table, not measured. The right reading of §4d is the opposite of "the old tier is under-funded": old sources carry a large share of the final-layer accuracy, so the winning representation is the one that keeps them *exact* at the lowest possible price. That representation exists, and its price lands on the leaders' number.

## 1. Cost anatomy: what a source-layer costs in our chain [M, code read]

Per lineage per layer in `estimator_k3_v53_age5.py` (1 unit = one n³ matmul before Strassen; ×0.51 at depth 5):

| step | where | cost |
|---|---|---|
| transport of the two dense legs `A_L`, `M` (`_advance`, l.1107) | `W^T A_L`, `W^T M` | 2 units |
| transport of `Z` (P2 residual) | thin | ~0.03 |
| D3 slice (`_dslice3`, l.1305) | elementwise, O(kn²) | ~0 |
| **D21 slice (`_dslice21`, l.1330)** | **three** `(n,kn)@(kn,n)` products | **3 units** |
| old-tier shared basis (age ≥5, rank 256) | `Q`, `FA/FM`, joins | ~0.7/layer, lossy |

So an exact lineage costs 5 units nominal ≈ 2.6 Strassen'd per layer, and **60% of it is D21**, whose three products exist only because the L-block (`A_L`, the correlation-born κ3) has a *different* leg from the D-block. The D-block alone — `A_D = M·d(sb)`, the neuron-diagonal κ3 born as Σ_j sb_j e_j⊗e_j⊗e_j — is a **single-leg** object: all three legs are `M = T_{l←s} = W_l^T … W_{s+1}^T` (with Wick scalings). For it, D3 = Σ_j sb_j M_ij³ (elementwise), D21 = (M∘M·d(sb)) Mᵀ (**one** product), and transport is **one** product.

Now the leaders' bill [M]: 0.131 B = 134 units for ~120 source-layers (16 layers, one birth per layer) = **1.1 units per source-layer all-in**. Our exact tier is 2.6 + glue. Two dense legs plus three-product D21 cannot reach 1.1 at any age gate or Strassen depth (depth 7 would need ~800k ops). A **single dense leg per source, D21 as one product, D3 free** costs 2 units nominal ≈ 1.02 Strassen'd per source-layer — and 120 × 1.02 + covariance 14 + closure/regen ~20 ≈ **156 units = 0.152 B with D21 for every source**, or **~117 units = 0.114 B with D21 only for sources of age ≤3**. The leaders sit at 0.126–0.146, between those two [I]. 504aldo's two unexplained observations — "old-source content carried at young-tier cost" and "κ4 content beyond a memoryless core at no visible cost" — both fall out of the same representation (§2). I know of no other design whose arithmetic lands there.

## 2. The hypothesis: single-leg diagonal-born tower, correlation-born content regenerated

**Carry exactly, for every source s and every layer, one n×n matrix M_s (the transport map since birth) and per-source birth vectors** sb_s (κ3 diagonal), q_s (κ4 diagonal), and optionally the κ5 diagonal. Everything diagonal-born rides on M_s at zero extra matmuls:

- κ3: D3_i += Σ_j sb_j M_ij³; D21 += (M∘M d(sb)) Mᵀ [1 product]
- κ4: (4)-slice_i += Σ_j q_j M_ij⁴ (free); (3,1)-slice = (M∘M∘M d(q)) Mᵀ [1 product, optional, young only]; (2,2) diag pieces elementwise
- κ5: (5)-slice_i += Σ_j q5_j M_ij⁵ (free)

This makes the fourth cumulant *carried and exact* for its diagonal-born part — which is what 504aldo's ablation and our own say carries ~85–90% of the κ4 gain — instead of regenerated memoryless, and adds the fifth for nothing. It also makes the final layer exact in every diagonal-born order, which is where the score is read.

**Drop the second dense leg.** The correlation-born κ3 (`A_L = w1∘C_off`, plus `3A_L d(eb)` and the `Z Lᵀ` residual inside `A_D`) is what costs the second transport and two of the three D21 products. Treat it the way the chain already treats correlation-born κ4 (`G_off = λ·C_off`): **regenerated memoryless from the current C_off with a fitted per-age/per-layer scalar**, exact only for the newborn (age 0) and possibly age 1. The precedent is strong: memoryless regeneration retained ~90% of AUGMENT's κ4 gain at ~zero cost; the SIMPLE-beats-AUGMENT result in §4c says the same thing at the level of term sets — correlation-born diagrams are the expensive, low-value-per-FLOP part.

**What goes away:** `A_L`, `Z/L`, `eb`, the shared basis `Q/FA/FM/Sg` and both join routines, P2, P3, the hubs `Y` (subsumed by q_s on M), and the age gate. State per layer: `mu, cov, {M_s, sb_s, q_s}`. Memory ≈ 15 × 4 MB for the M's; wall a fraction of today's; ops dominated by ≤ 2 Strassen products per source-layer.

**What is unknown [?]:** how much raw MSE the correlation-born κ3 of *old* sources is worth, exact vs memoryless. That is the single number the next two days must produce. My prior: the loss is 10–40% raw before regeneration and ≤15% after it. I do not know; record it as "I don't know" if you prefer.

If the loss after regeneration is ≤15%: raw ≈ 2.1e-8 × 1.15 ≈ 2.4e-8 at 0.12–0.15 B → **2.9–3.6e-9, top 8–12**, before the free κ4/κ5 diagonal channel, which is the first thing on this chain that could push raw below 504aldo's 1.94e-8 ceiling → 2.3–2.8e-9, top 3–5. If the loss is >40% and regeneration does not recover it, the hypothesis is dead in two days and we fall back to §6.

## 3. Answers to the five questions

**Q1 (a representation that is cheaper *and* more accurate).** Yes, §2: single-leg diagonal-born tower. Cheaper because the diagonal-born block needs one leg, one product for D21, and nothing for D3; more accurate because it never compresses (the rank curve says full rank is mandatory — `M_s` is a product of a random matrices and has no usable low-rank structure, which is also why every CP/Tucker/basis attempt here and at 504aldo hit cliffs) and because κ4/κ5 diagonal content becomes exact and carried instead of memoryless.

**Q2 (is "spend more on the exact tier" a local optimum?).** Yes, as a direction it ends at age 6 — but only because each exact source costs 5 units nominal in the two-leg spelling. At 2 units nominal, *every* source is exact at age ∞ for less than we pay today. Stop raising the age; change the spelling.

**Q3 (cheaper term set?).** Do not migrate to ARC's SIMPLE table — different counter, unoptimised code, small shapes, and our chain is not that arm. Take the *principle*: correlation-born (AUGMENT-type) content costs n³ per source-layer and buys little per FLOP; diagonal-born content is cheap and dominant. §2 applies that principle inside our own chain, where the meter is exact. Fitting is not being asked to rescue accuracy; it is being asked to set ~16 regeneration scalars, which is the one job it has done well here (504aldo's λ table).

**Q4 (only the final layer is scored — adjoint formulation?).** The pure backward scheme (read old sources out only at layer L through R_s = W_L M_s) is exact for their *direct* contribution but drops their intermediate D3, which shifts every later mean at first order and is transported to the final layer at ~unit gain; I expect that loss to be large and do not recommend building it. The final-layer benefit the adjoint view promises — exact high-order marginals from every source — comes for free in §2 because every diagonal-born order rides on M_s. Keep the last-layer prune (`{(1,),(3,),(4,)}`), add (5,).

**Q5 (push or bank?).** Push, and bank as we go. v53 is designable only if the private rerun's wall cap matches the public one and its 96.5 s holds there; v50 (88.8 s, 6.19e-9) is the safer bank today. Do not spend a slot on age 6 (87% memory, wall >100 s). The §2 chain, if it works, is both the score entry and the write-up: "single-leg cumulant towers: exact diagonal-born content of all orders at one product per source-layer" is a real algorithmic contribution and is not in 504aldo's write-up or ARC's paper. Write-up time stays at zero until Oct 10.

## 4. Kill tests — two days, all on `tools/ev.py`, DEV 100, unshared chain as the reference (`_SHARED_AGE=99`, `_SHARED_ON=False`)

Each is an *accuracy* probe first (cost of the probe arm is irrelevant; the real cost is computed from §1 arithmetic and then measured once on the rebuilt chain). Record a prediction for each before running.

- **T0** unshared exact chain: raw, cost, memory. (Reference. If it does not fit in 8 GiB on DEV, run 24 nets.)
- **T1 — D21 gate.** For lineages of age > a, drop their contribution to D21 (they keep transport and D3). a ∈ {2, 3, 4}. *Measures what old-source (2,1) content is worth.* Prediction: ≤8% raw loss at a=3.
- **T2 — L-block gate.** For lineages of age > a, zero `A_L` (and `eb`, and the `Z Lᵀ` residual) so `A_D = M·d(sb)` — i.e. old sources become single-leg. a ∈ {0, 1, 2, 3}. *Measures what old correlation-born κ3 is worth.* This is the decisive test. Prediction: I don't know; 10–40% at a=1.
- **T3 — regeneration.** On the T2 arm at a=1, add back the dropped content memoryless: for each layer l, the old-lineage L-block D3/D21 contributions replaced by ρ_{l} × (the age-0 lineage's L-block D3/D21 at layer l), ρ a 16-vector fitted on FIT with the D2 optimizer (also try ρ per age with a 2-parameter exponential). Gate: T3 within 15% raw of T0.
- **T4 — κ4 on M.** On the T2/T3 arm, replace the memoryless diagonal κ4 core with the carried diagonal-born κ4: per-lineage q_s at birth, (4)-slice Σ_j q_j M_ij⁴ summed over lineages, (3,1) feed = (M∘M∘M d(q)) Mᵀ for age ≤ 2 only, λ·C_off kept for the correlation-born part. Add the (5,)-slice at the last layer. Prediction: ≥10% raw gain over T3 (this is the channel 504aldo could not afford).
- **T5 — cost.** Rebuild the chain as §2 (new file, `estimator_k3_v60_single_leg.py`, generator-based like the others; no `A_L`, `Z`, `Q`, joins, P2, P3, hubs), bit-parity against the T2/T3/T4 arms on 2 nets, then one metered run: FLOPs, ops, wall, RSS. Expected 0.11–0.16 B, < 60 s wall, < 2 GiB.

Decision after T3: if T3 ≤ 15% loss → build T5 and canary it with §5 gates (expected 3–4e-9). If T3 loses 15–40% → build T5 anyway but keep exact two-leg treatment for age ≤ 2 (cost ~0.17 B) and canary. If T2 loses > 40% and T3 does not recover to < 40% → hypothesis dead; go to §6.

## 5. PLAN amendments (rev 12.1)

- D5 and D6 are replaced by §4 (T0–T5) and, on success, by D5′ = "ship the single-leg chain, then fit its ~16–32 regeneration scalars on FIT, then add (3,1)-feed and κ5 if they pay". Everything else in PLAN stands, including the §5 submission gates as amended in RESULTS.md (ops ≤ 26,000, residual a hard gate).
- Nothing is struck by inference again. H1/H4-type ideas that were dismissed from the rank table get one line each in RESULTS.md saying "not measured; superseded by T1–T3".
- Bank: v50 stays the safe designate-B candidate; v53 is designate-A candidate pending a second clean graded run (submit identical bytes once to check wall variance, as marius does).
- Cadence: T0–T2 tomorrow (Sep 23), T3–T4 Sep 24, T5 + canary Sep 25–26.

## 6. Fallback if §2 dies

Ship v53 (or age 5 with Winograd 15-add if memory allows a shallower Strassen depth to buy wall), and run the one remaining measured lever: the **elasticity table** — ablate each n³ component of the exact tier one at a time (P2 residual, hubs, P3, the (3,1)/(2,2) consumers, κ4 feed) on DEV, record Δcost and Δraw, and re-spend every component with score elasticity < 1 on age. Expect ~5–10%; that would leave us ~5e-9, rank ~40, and the write-up becomes the objective.

## 7. What I am and am not claiming

Measured: the cost anatomy in §1 (from the v53 source and the packet's ops/FLOP rows); the leaders' unit price; the rank/age table; that D21 is three products and D3 is elementwise. Inferred: that the leaders use a single-leg diagonal-born representation with regenerated correlation-born content (it fits their bill to within 10% and explains both of 504aldo's anomalies; nothing else I can construct does). Unknown: what the old correlation-born κ3 is worth (T2/T3) — the entire hypothesis rests on that one measurement.
