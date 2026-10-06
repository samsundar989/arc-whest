# HANDOFF — Lane M2 (session 2, branch lane-m), 2026-10-07: what makes AUGMENT better, the κ3×edge oracle, and coherent early errors

Worktree `../arc-whest-lane-m` (branch `lane-m`; rebase on main after Lane P merges your Lane V work). No estimator edits on main, no submissions. GPU via workerq below Lane P's twin jobs; RunPod approved if the local GPU is busy > 2 h. The rev 19 docs named below (and this handoff) live in the main checkout, `C:\Users\samsu\Documents\arc-whest\docs\` — read them there by absolute path until Lane P2 commits them and you rebase. Read: `docs/PLAN_rev19_2026-10-07.md`, your own `docs/research/2026-10-06_laneM_report.md`, and `docs/research/2026-10-07_R1_efficient_higher_order.md` (§5.1 dressed legs, §5.3 κ3×edge) and `docs/research/2026-10-07_R2_structure_and_subspace.md` (§1.5 gain mode, §3 early-layer transfer).

**Why.** Your M2/transfer result is the most important finding of the week: ARC factored AUGMENT (b) beats v73 raw 16/16 at n = 1024 (1.79e-8 vs 2.32e-8) with no learned parts, and unfactored AUGMENT (c) is ≈ 1.50e-8 — the leaders' level. We now need to know *which* AUGMENT ingredients carry that gap, and whether cheap forms reproduce them.

## M5 — ARC-b ablation at n = 1024 (Oct 7, 16 organizer DEV nets, organizer labels)
Run factored AUGMENT with one ingredient changed at a time (read each from factor_k3.py `_nonlin_kprop` / kprop_harmonic):
1. **no heavy legs**: `legs = [(1,1,WK_11,1)]` only (SIMPLE births) but everything else AUGMENT.
2. **dressed legs instead of heavy legs** (R1 §5.1, all four heavy edges): one leg X̃ = w(1)∘C + H with H[a,j] = ½ w2_a D21[a,j] + (w3_j/w2_j)(½ w1_a D21[j,a] + ¼ w2_a K22[a,j] + ½ w2_a C_aj²); births Sym(X̃ ⊗ 3 w2 e_j ⊗ X̃). (D21 = `diagslice(WK[3],(2,1))`, K22 = `diagslice(WK[4],(2,2))`, both zero-repeated as in ARC.) This is the transfer test for Lane P's P2.1.
3. **no κ4 memory**: rebuild the κ4 core each layer from the current state only (drop the transported WK[4] contribution) — and, as a second arm, replace the core by the gain-mode form: κ4_ijkl ≈ 4βc²(C_ij C_kl + C_ik C_jl + C_il C_jk) (sum of the three pairings, not their average; c² = m̂ᵀCm̂/‖m‖² from the post-ReLU state, β = 0.7) projected onto ARC's r = 1 core with `DS_harmonic_proj` (derive the normalization numerically on a width-64 dense check rather than by hand).
4. **no 211→111** (κ4 core → κ3 births) and 5. **no K211 contributions** (pK_111/pK_211 → κ4 core).
Report final-layer MSE per arm (floor-subtracted, paired vs b and vs v73's cached DEV preds), plus all-layer MSE.
- Predictions (strategist): (1) loses 35–60 % of b's gain over SIMPLE; (2) recovers ≥ 70 % of what (1) loses; (3, memoryless) loses 20–40 %, the gain-mode core recovers half of that; (4)/(5) small (< 10 % each). Record yours.

## M6 — Dense-κ3 oracle for the κ3 × covariance-edge family at n = 1024 (Oct 7–8)
At n = 1024 a dense κ3 is 4 GiB in fp32 — it fits on the 5090 (unfactored AUGMENT ran out of memory because of κ4/other buffers, not κ3 alone). Build a torch chain at n = 1024 on 4–8 DEV nets that carries dense κ3 with SIMPLE + dressed births and the E-family (E3 + E21a + E21b) as a set, only to measure:
- the size of the E-family's contribution to D21 and D3 at layers 4/8/12/16 (relative to D21's norm);
- **rank-r spike truncation** of C inside the E terms (top-r eigenpairs of C, r ∈ {1, 4, 16, 64}): capture fraction;
- **gain-mode form of E21b** (R1/R2 cross-over): with κ3_ijk ≈ 2βc²(μ_i C_jk + μ_j C_ik + μ_k C_ij) (the three placements summed), the E21b core term Σ_i W22_i C_ik κ3_ijk reduces to O(n²) pieces plus one n³ product C diag(W22) C — measure what fraction of the exact E21b it reproduces;
- R² of a learned proxy (ridge on D21∘C row sums, the top-4-mode projections, w2-weighted D21) for the E-family's ΔD21.
Decision: the family earns a v-chain prototype only if (size ≥ 2 % of D21) AND (r ≤ 16 captures ≥ 70 %, or the gain-mode form ≥ 60 %, or proxy R² ≥ 0.6). Then hand Lane P the form with its unit price.

## M7 — Coherent early errors (Oct 8, ≤ ½ day; twin read-only)
R2 measured how errors injected at layer l₀ survive to layer 16 in a real net: random-direction mean errors keep only 2–11 % of their energy from l₀ ≤ 4, but errors along the mean/gain direction keep 27–64 %; a coherent relative covariance error η at layers 2–4 costs 0.04–0.13·η² of final MSE. On FRESH-A or FIT nets with exact labels, measure v77's per-layer error at layers 1–6 projected on m̂ (gain direction) and on the top-4 covariance directions, plus tr C and mean σ² vs labels. If the coherent part at layers 2–4 exceeds the budgets (coherent covariance error > 1e-4, mean scale error > 5e-5), prototype the fix: exact layer-2 covariance via the arc-cosine kernel (layer 1 is exactly Gaussian) and an exact layer-2 κ3 (Price/Plackett, O(n²) per pair). Otherwise record "early layers are not the lever" for the write-up.

## Do not
Submit, edit estimator files on main, fit or select on FRESH panels, read FRESH-LOCK, contact organizers.

## Report
`docs/research/2026-10-0X_laneM2_report.md` with tables; daily paragraph in the strategist packet; hand any GO form to Lane P with its price and expected gain.
