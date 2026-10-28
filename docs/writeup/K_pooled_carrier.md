# K — Pooled-column κ3 carrier vs exact-age/band split (sandbox, Oct 21 04:40 UTC) — VERDICT: dead; L5 stays closed

**Question.** Is a κ3 carrier that pools hub columns from all ages under an importance cap (the only disclosed structural fact about a leader's carrier: AndreasHad04 "cap 3776" = 3.7n columns; hyojun "rank cap 7,680") more efficient than our exact-age tier + rank-r band? If yes by ≥ 20 % at equal cost, build it today (HANDOFF L5).

**Method** (`/home/claude/sandbox/pooled_carrier.py`, results `results/pc/pc_n{128,256}_s{1,2}.json`). Random He nets, x ~ N(0, I), n = 128 and 256, depth 16, 2 seeds each. Exact Gaussian (μ, C) chain; every first-order hub-star lineage built explicitly: hub j born at layer s has u_j = W_{s+1}e_j (Jacobian column), v_j = W_{s+1}Φ_sC_se_j/σ_j (cross-covariance column), weight w_j = σ_jφ(α_j); both legs transported by G_l = W_{l+1}diag(Φ_l). Readouts D3_i = 3Σ_j w_j u_ij v_ij², D21_ik = Σ_j w_j(2u_ij v_ij v_kj + u_kj v_ij²). Reference = all ages, all hubs. Compared at every layer 4–15, sensitivity-weighted (D3 by (αφ/σ²)², D21 by (φ_iφ_k)²) and unweighted:
- SPLIT(E, r): ages ≤ E exact with KAS hubs (|α| < 2.5); older ages projected on the best rank-r basis of their stacked legs at that layer (an *optimistic* band: optimal basis, no transport error). Cost 4·E·n_hub + 6r (units n²/layer).
- POOLED(N, rule): one column set, newborn kink hubs appended each layer, the lowest-importance columns deleted sequentially to keep N. Rules: w‖v‖²‖u‖, w‖v‖², sensitivity-weighted Σ_i s_i|u_ij|v_ij². Cost 4N.

**Result** (rms over layers 9–15 of the weighted D3 error; the floor 0.025 is the KAS hub pruning, common to all):

| cost (n³/layer) | split | e3w | pooled | e3w |
|---|---|---|---|---|
| ≈ 8 | E2 r0.56n (n=128) / E2 r0.28n (256) | 0.025 / 0.081 | N = 2n | 0.54 / 0.49 |
| ≈ 14–15 | E4 r0.28n | 0.030 / 0.034 | N = 3.7n | 0.32 / 0.32 |
| ≈ 20–24 | E6 r0.28n | 0.025 / 0.027 | N = 6n | 0.14 / 0.14 |
| 32 | — | — | N = 8n | 0.066 / 0.060 |

- **At equal cost the pooled cap is 5–10× worse than the split, at both widths, under every importance rule.** The gap does not close from n = 128 to 256 (pooled errors are identical at equal N/n). Deleting whole hub columns loses their content outright; in a random He net the kink hubs carry similar content, so no sparse subset exists — a shared basis captures the old content's dominant directions and the cap cannot.
- The split needs r ∝ n to sit on the floor (r = 0.56n at both widths), consistent with the real-label leg-compressibility table (rank 512 = 0.5n ⇒ ≤ 3 % readout error for ages ≥ 3 at n = 1024). The sandbox cost model understates the real band (its readout from thousands of old columns scales as n²·cols or n·r³, which is why the Oct 17 census found "r = 512 is neutral but saves nothing" and settled on E7 r288); it does not change the pooled-vs-split ordering, which is a representation-error result.
- 504aldo's dead-end #5 ("capped hub columns … 8n needed for 2 % D21 error") is reproduced: N = 8n reaches 0.06.

**Consequences.** (1) L5 is closed; no build. (2) Andreas's "cap 3776" cannot be a plain importance cap on hub columns — his "atoms" are something else (merged/rotated columns, i.e. a basis), which is our band family. (3) For the write-up: a clean negative with a mechanism — hub content in He-init nets is spread, not sparse; compression must be by basis, not by selection.
