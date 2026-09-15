# Experiment — E14 `_tree_core` label-free λ vs 504aldo's fitted λ table (2026-09-14)

*Run tonight in the Cowork cloud container (float64 numpy, no flopscope). Scripts/CSVs: scratchpad `exp_lambda/` (`lambda_compare.py`, `lam_compare_summary.csv`, `lam_post.csv`, `lam_pre_off.csv`, `mc_check_tree_core*.py/.log`). This doc preserves the result; re-run in the repo with `mint_provenance.canonical_mint` on public mini MLPs 0–7 to make it per-MLP.*

## Question

504aldo's V17 (findings F68) regenerates κ4 each layer as G = diag(g) + λ_l·C_off with a 16-entry λ table fitted by least squares of the *dense augmented K3 chain's* off-diagonal κ4 core on C_pre_off (8 public MLPs; shipped 0.95×mean; adaptive per-MLP scaling V25). It halves raw MSE (4.46e-8 → 2.26e-8). Our `estimator_e14_tree.py::_tree_core` produces the same retained core, same 6/(n+4) projection constant, same consumers (`q4`, `q31`, `q22`, metric 2.0 = v29's METRIC_C), but λ is *derived* label-free from a connected-tree Gaussian diagram in C_pre with Wick weights b=w(1,1), a=w(2,1), d=w(3,1). Are the two λ the same object?

## Index alignment (from code)

v29: end of layer li sets `g_prev`, `var_prev`, `lam_prev = LAM[li]`; consumed at li+1 as `dG = WW@(g_prev − var_prev·λ) + var·λ`, `wk431 = λ·C_off(li+1)·0.5·METRIC_C` → model G_pre_off(li+1) = LAM[li]·C_pre_off(li+1). `lam_prefit_perdump.npy` is 8×15, column li = LAM[li].
E14: `_tree_core(cov_pre(li), cov_out(li), …)` returns λ = LS of (6/(n+4))H_off on the **post-ReLU** C_out_off(li); consumed at li+1 identically in form (`dG = (Wᵀ∘Wᵀ)@(g − λ·diag Cout) + λ·diag cov_pre`, `Goff = λ·sym(cov_pre(li+1))·off_mask`). So E14 λ (call li) ↔ LAM[li]. Both definitions were computed (post-ReLU as coded; and 504aldo's pre-activation LS applied to E14's transported core); they agree except at li=0, so the definition is not what separates them.

## Method

8 He-init 1024×16 MLPs (seeds 0–7, `harness.py` legacy recipe; HF and PyPI are blocked from the container, so same distribution as the public mint, not the same realisations — the chain's per-layer mean pre-activation variance matches 504aldo's `lam_obs_d0-7.npz` public-MLP column to 1–3%). Float64 Gaussian covariance chain using E14's own frozen Wick table (`_TABLES_B64`) and degree-2 off-diagonal closure (`K[(1,1)]_off = C·w11ᵢw11ⱼ + C²/2·w21ᵢw21ⱼ`, checked = arccos expansion, max rel err 4e-4 at ρ≤0.25); `_tree_core` ported line by line and checked symbol-for-symbol against `local_sim/e14_tree_oracle.py::fast_h`; K[(1,)], K[(2,)], K[(4,)] unit-checked to 1e-15. 7.5 s/MLP.

## Result

| li | 504aldo λ mean (sd%) | 0.95× shipped | E14 λ post (as coded) | E14 λ pre (504aldo's def.) | ratio pre/504aldo | R² of E14's own core on C_out_off |
|---|---|---|---|---|---|---|
| 0 | 4.99e-3 (4.2) | 4.74e-3 | +2.61e-2 | +9.14e-3 | 1.83 | 0.94 |
| 1 | 8.09e-3 (3.9) | 7.68e-3 | +7.25e-3 | +3.08e-3 | 0.38 | 0.48 |
| 2 | 9.83e-3 (2.9) | 9.34e-3 | +2.54e-3 | +1.35e-3 | 0.14 | 0.15 |
| 3 | 1.055e-2 (3.4) | 1.002e-2 | +6.6e-4 | +4.2e-4 | 0.04 | 0.02 |
| 4 | 1.085e-2 (2.7) | 1.031e-2 | −8.5e-5 | −7.0e-5 | −0.006 | 0.00 |
| 5 | 1.083e-2 (2.8) | 1.029e-2 | −7.0e-4 | −4.8e-4 | −0.045 | 0.04 |
| 6 | 1.059e-2 (2.3) | 1.006e-2 | −6.9e-4 | −5.2e-4 | −0.049 | 0.05 |
| 7 | 1.005e-2 (3.9) | 9.55e-3 | −8.4e-4 | −6.4e-4 | −0.064 | 0.09 |
| 8 | 9.65e-3 (4.3) | 9.17e-3 | −1.01e-3 | −8.0e-4 | −0.083 | 0.14 |
| 9 | 9.17e-3 (3.7) | 8.71e-3 | −9.3e-4 | −7.4e-4 | −0.081 | 0.14 |
| 10 | 8.77e-3 (4.3) | 8.33e-3 | −9.5e-4 | −7.9e-4 | −0.090 | 0.17 |
| 11 | 8.39e-3 (5.7) | 7.97e-3 | −9.4e-4 | −7.8e-4 | −0.093 | 0.19 |
| 12 | 8.06e-3 (7.0) | 7.65e-3 | −8.1e-4 | −6.9e-4 | −0.086 | 0.17 |
| 13 | 7.68e-3 (8.2) | 7.29e-3 | −8.6e-4 | −7.4e-4 | −0.097 | 0.21 |
| 14 | 7.26e-3 (9.5) | 6.90e-3 | −7.7e-4 | −6.6e-4 | −0.092 | 0.20 |

Depth-profile Pearson r (l=1–14): −0.01 (pre) / −0.12 (post). Best single scale k in E14 = k·LAM: 0.02 with 49% rms residual. **Not within 10%, not a constant factor, not the same shape, opposite sign from layer 4 on.**

Monte-Carlo check of what `_tree_core`'s H is (n=12, 4e7 samples; H_MC,ij = Σ_k κ4(y_i,y_j,y_k,y_k)): diagonal anchors agree <1%; H_off direction agrees (cosine 0.964 / 0.997 / 0.9996 as ρ falls 0.17→0.03) but the magnitude overshoots 1.34× / 1.66× / 1.71× and λ overshoots 1.54× / 1.70× / 1.71×. Mechanism: the (3,1) repeated-index pattern (k∈{i,j}) is evaluated with the 4-vertex tree formula rather than the exact Wick expansion; only the (4) and (2,2) slices are anchored. That 1.71× is the 1.83× excess at li=0, the one layer where both λ are pure Gaussian-order production.

## Interpretation

504aldo's λ is the projection onto C_off of the augmented chain's off-diagonal κ4 core, which is dominated by **κ3-sourced κ4 production** (ARC's k211/hub channel) — R² 0.66–0.97 rising with depth. E14 regenerates only the **Gaussian-order** tree from C_pre and derivative Wick coefficients; that piece is +2.6e-2 at layer 0, decays through zero at layer 4 and settles at a small negative plateau (the w(3,1) = −αφ/σ² star terms win as |α| grows), with R² ≤ 0.2. The two are different objects. E14's consumers and κ4→κ3 feed plumbing are correct and reusable; its λ is not, and E14 as specified should not be expected to deliver the 2× κ4 gain.

## What to do instead (label-free and clean)

Fit λ_l = ⟨G_pre_off(l+1), C_pre_off(l+1)⟩/‖C_pre_off(l+1)‖² from **our own torch `kind=AUGMENT` chain** (`local_sim/p2_augment.py`, already measured at 1.912e-8 on 24 nets) on **our own minted networks** — no competition labels involved. Prediction: within ~10–15% of 504aldo's table per layer. Also record the diagonal core g and the (3,1) slice to check E14's diagonal anchors. Then a second fit with target = the *consumed mean correction* (Puffi's unshipped −13.6% refit) as lever N3.

## Caveats

Synthetic MLPs (distributional comparison; per-layer spreads 1–12% on our side, 2–10% on 504aldo's, so the sign/magnitude conclusion does not depend on realisation); Gaussian-only chain (E14's live chain feeds κ3/κ4 corrections into C — a few % of C_off, cannot flip sign or order of magnitude); degree-2 closure (CLOSURE=deg4 implemented, not run); MC check at n=12 with a ρ-sweep converging to 1.7× (extrapolated to n=1024).
