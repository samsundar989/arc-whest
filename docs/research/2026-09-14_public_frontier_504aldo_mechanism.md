# Public frontier mechanism — 504aldo `whest-p2-cumulant-k3` (V29), extracted 2026-09-14

*Ideas and published measurements only, for independent reimplementation with attribution (MIT, `github.com/504aldo/whest-p2-cumulant-k3`; forum post discourse.aicrowd.com/t/…/18218, 2026-09-09). No code was copied. Sources: README.md, docs/community_post.md, docs/findings_index.md, docs/derivation_code_map.md, docs/claims_evidence.md, docs/ablation_table.md, docs/audit_v29_ns_d0.log, docs/figures/fig2_cost_anatomy_v29.csv, estimators/estimator_v29.py (constants), generators/lam_obs_probe.py, harness/mem_v29.py. Unit: 1 unit = 2n³ FLOPs = 2.147e9 at n=1024; B = 2^41 = 1024 units.*

## 1. κ₄ memoryless regeneration (V17: raw 4.46e-8 → 2.26e-8 at −1.3% cost)

Core `G = d(g) + λ_ℓ·C_off` (g = exact κ₄ diagonal; C_off = preactivation covariance, diagonal zeroed). One scalar per layer; R² of oracle off-diagonal core on C_off = 0.66–0.97 rising with depth (0.51 at layer 0). Adding eight more candidate modes (C×C, symmetric derivatives, variance products, K22): 2.230e-8 → 2.205e-8 (nothing).

Diagonal transport (affine in λ): `dG = (W∘W) g_prev + λ (diag(C_pre) − (W∘W) var_prev) = t_g + λ t_v`. Post-ReLU harmonic projection: `g_prev ← cA·(K4v + K22·1) + cI·(sum(K4v) + sum(K22·1))`, `cA = 6/(n+4)`, `cI = −3/((n+2)(n+4))`. Whole κ₄ machinery O(n²); ledger 0.49 units (0.2% of the bill).

Use-side slices (METRIC_C = 2.0): `g4row = 2·dG` (4,), `wk4m = (2/6)(dG_i + dG_j)` off-diagonal (2,2), `wk431 = (2/2)·λ·C_off` (3,1). κ₄→κ₃ feed pair: `X3 = A·d(c1) + P·d(c2)` with `c1 = λ·w1` (static at birth), `c2 = w1²·dG` (transported); `Y3 = y·1ᵀ` with y the last thin-Z column; `R = (A∘P)c1 + (P∘P)c2`; `D3 += y∘R`; `D21 += (1/3) R yᵀ + (2/3)(y∘X3) Pᵀ`. Rides existing A/P contractions — zero new n³.

Binding ablations: regenerating the diagonal too → 3.27e-7 (15× worse); use-side slices without the feed → 9.2e-8 (harmful). The gain flows through the feed.

λ_ℓ table (shipped, ×0.95 of raw LS fit on 8 public dumps against ~1e9-sample oracle κ₄ cores): `[4.9895e-3, 8.0876e-3, 9.8291e-3, 1.0549e-2, 1.0851e-2, 1.0828e-2, 1.0589e-2, 1.0048e-2, 9.6483e-3, 9.1720e-3, 8.7730e-3, 8.3938e-3, 8.0555e-3, 7.6770e-3, 7.2588e-3, 7.2588e-3]`. `REF_R = [6.588e-3, 8.184e-3, 8.531e-3, 8.389e-3, 8.102e-3, 7.743e-3, 7.360e-3, 6.911e-3, 6.558e-3, 6.179e-3, 5.840e-3, 5.568e-3, 5.325e-3, 5.052e-3, 4.772e-3]`. Global scale 0.9/1.1 → −0.2%/+2.3% score. Label-free per-MLP adaptation (V25): `rr = mean(dG0)/mean(var)`, `λ ← λ·clip(rr/REF_R[ℓ−1], 0.5, 2.0)`; weight-space observables only; effect −0.6…−1.3%.

## 2. D21→birth feedback, rank 16 (V18: −10% raw, zero n³)

κ₃→κ₃ feedback: the current (2,1) slice D21 enters the newly born source's legs: `X1 = 3 a_b + 1.5 d(w2) D21`, `Y1 = a_b d(w2) + 0.5 d(w1) D21ᵀ d(w3)`, `a_b = d(w1) C_off`. D21 compressed at birth by a two-pass range finder with the free sketch `Om_f = W[:, :16]`: `Qf = qr(D21 Om_f)`, `Qf = qr(D21 (D21ᵀ Qf))`, `Bf = Qfᵀ D21`; carried as a thin (n,n,32) leg; folded into LA/LP. Rank ladder 8/16/32 → 2.17/2.14/2.14e-8; 32 costs +6.2% for −3.4% raw (score-negative). Ledger: 11.45 units (4.4%).

## 3. The 7→4 restructure (V16b: 0.833 → 0.50 B, bit-parity)

Redundancy A: the paper's legs X = 3A and Y = A·d(w2) are diagonal scalings of A — never stored/transported; `M = P d(s) + 3A d(e) + Z Lᵀ` never formed densely (rank-18 thin correction). Only A and P move.
Redundancy B: the four (2,1) products `H = (A∘P∘w2)Aᵀ`, `T2 = (A∘A∘w2)Pᵀ`, `T3 = (P∘P)(A d(w1) c_off)ᵀ`, `T4 = (M∘P)Pᵀ` regrouped by right operand: `LA = 2(A∘P∘w2) + (P∘P)∘e`, `LP = (A∘A∘w2) + (P∘P)∘s/3 + (2/3)(M∘P)`; `D21 = Σ_b [LA_b A_bᵀ + LP_b P_bᵀ]` batched.
Per young source per layer: A transport, P transport, A contraction, P contraction = 4 nominal units, priced 2.17 after SW-5; age-1 sources 3. Young tier flat at 8.70 units/layer from layer 5.

## 4. Old-source confinement and ledger

Constants: AGE_OLD = 4, R_OLD = 384 (one-pass); AGE_OLD2 = 7, R_OLD2 = 224 (two-pass, nested). Join: weighted Gram `A d(dA) Aᵀ + P d(dP) Pᵀ + Qp S Qpᵀ`, `dA = 9 + w2² + 9e²`, `dP = 1 + s²`; range finder with the free sketch `W[:, :384]`; rotate old factors by `Qnᵀ Qp`; transport `Qc ← W Qn`. Rank law ≈3n/8 at age 4, n/4 at age 6, 7n/32 at age 7; age 3 needs 512; age 2 at 128 is a cliff (+96%, 3×). Rank 192 looked best on 2 dumps and failed on 8.

V29 ledger (260.1 units = 0.254 B): young_transport 60.71, hub 54.90, shared 27.91, old_legs 27.89, j_rf 20.76, fb 11.45, j_rot 9.14, j_proj 7.50, cpre 7.10, j_tier2 6.70, thin_transport 5.86, j_qc 3.75, thin 3.70, elem 3.27, nonlin 2.92, j_core 2.82, birth 2.47, feed 0.49. Grouped: young K3 115.6 (44.5%), old K3 106.8 (41.1%), thin/elem 24.8, covariance 7.1, birth/closure 5.7. **Without the old tier: 153 units = 0.150 B — the leaders' cost.**

Per-layer units (young/old/thin/cov/closure/total): ℓ0 0/0/0/0.50/0.10/0.61; ℓ1 2.33/0/0.22/0.47/0.39/3.42; ℓ4 9.10/0/0.89/0.47/0.39/10.86; ℓ5 8.70/4.56/1.10/0.47/0.39/15.24; ℓ8 8.70/9.50/1.76/0.47/0.39/20.84; ℓ12 8.70/12.74/2.64/0.47/0.39/24.96; ℓ14 8.70/14.30/3.08/0.47/0.39/26.96; ℓ15 5.82/4.36/1.64/0/0/11.84.

Last-layer trim (V19, −8%, bit-identical): no D21, `D3 = 3·einsum(A,A,P∘w2)`, `var = colsum(W ∘ (C W))`. Setting D3 := 0 at the last layer is a 10× penalty.

## 5. The wall and the dead ends

Khatri–Rao wall: with legs `A = Q F_A`, `P = Q F_P` (rank r), `A∘P` has rank r²; the dense leg must be re-formed at n²r per use, ≈ its transport cost. Confinement buys ≈2× per source (old ≈1.5 units vs young 2.17), never 10×. Closed exits: Tucker core in the shared basis (r³/n² units: 10.7 at 224, 54 at 384); multiplicatively closed basis; hub-side merging (CP cap 4n → D21 error 0.39; 8n → 0.03–0.14; need 0.02).

Dead ends (measured): drop old sources W=4 → 1.02e-6; memoryless closure of (2,1)/(3,) → 9.1e-7; Puffi A716 closure → 9.8e-7; Gaussian scale-mixture → 2.33e-6; symmetric Tucker R=512/256 → 14×/31×; per-source rank ladder → 17× loss for 1.8× compute; per-source κ₄ legs ≥0.95 B; learned output corrections corr <0.09; cheap covariance closures 4–5e-6; MC control variates best 1.5×; symmetric-contraction FLOP tricks don't apply; no dtype lever; f32→f64 +0.004%; higher feedback/residual ranks score-negative; exact (2,1,1) κ₄ core −9% raw at 0.5–1.0 B; adjoint sensitivity of the final mean to D21 gave no cost drop (F88); pricing polish ≤4% on paper, 0% shippable (+74–118 ms residual); pure sampling ≈1.2e-6 adjusted; literature (~25 papers) zero methods.

Top-3 conjecture (unvalidated): leaders' C/B (0.149–0.164) = this chain minus the old tier; marius raw 1.71e-8 < their f64 memoryless-κ₄ floor 1.94e-8 ⇒ leaders carry κ₄ beyond a memoryless core and old-source content at young-tier cost — "not a per-source hub-tensor expansion".

## 6. Engineering

f32 with one `astype(float32)` per MLP; SW-5 within ~1e-6 relative of dense. Memory ≈5.5 GB after two predicts (8 GB cap). Residual: base chain 0.331 → 0.170 s via pre-allocated `out=` buffers (≈0.02 ms/op); dominant cost was result-buffer allocation churn, not op count; mini reports ≈0.29 s/MLP. Setup cap 5 s kills the whole submission. Over-budget → zero fallback, no discount.
