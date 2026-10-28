# J — Do a few coherent, zero-FLOP covariance "faces" explain the 8be chain's per-layer variance error? (real data, 100 FIT nets)

Measured 2026-10-20 on sam-mega-pc (device_bash VM, 2 CPU, no GPU). Scripts and result JSONs: `_claude_snap/x8face/` in the repo (`collect.py`, `analyze.py`, `analyze2.py`, `analyze3.py`, `analyze4.py`, `mdtables.py`; `feat/net000..099.npz` (188 MB, per-net projections), `res_L01..15.json`, `res2.json`, `res4.json`, `tables.md`). Nothing else in the repo was touched. Weights were regenerated with `tools/fresh/gen.py mint_numpy(seed)` from `data/panels/fit.json` (sha256 verified on all 100 nets; 0.2 s per net).

## 0. Verdict

**The sandbox's structural claim is confirmed, its consequence is not.** The real 8be chain's variance error does have one dominant, coherent, per-net-stable, zero-FLOP covariance face — the scale mode `Cov(h_l) ⊃ ρ_l m_l m_lᵀ`, whose sandwich image is exactly `μ_{l+1}²` — and it explains **40–64 % of the unweighted variance error at layers 9–15** (participation ratio of the best quadratic correction 1.2–1.3, top direction cos 0.98–0.99 with `W m`). But that error lives entirely on decided neurons (|α| > 2, where `μ²` is large and `∂E/∂σ = φ(α) ≈ 0`). **Sensitivity-weighted, the face explains 0.4–0.6 % of the variance-error energy at layers 9–15, every face set together ≤ 3 %, and the generous upper bound (all 66 products of 11 per-neuron direction vectors) 1–4 %.** On the kink neurons (|α| < 1), which carry 84 % of the sensitivity-weighted energy, the chain's relative variance error has rms 0.03 % (L6) → 0.12 % (L15) and is ≥ 96 % incoherent — no function of the local layer state or of any rank-few face built from it.

**Expected effect of a fitted rank-1 / rank-few zero-FLOP face after refit: −38 % × (0.005–0.03) × 0.5 ≈ −0.1 … −0.6 % raw, central ≈ −0.3 %.** The unweighted numbers would wrongly suggest −11 … −13 %. For X6 this means: a learned rank-k state transported as `Φ∘(W U)` *will* find the scale mode immediately (it is the top direction, R² 0.6 unweighted), and that will not move the mean. The part of the covariance error that reaches the final MSE is high-rank at the neuron level.

## 1. Data, alignment, definitions

| item | source | check |
|---|---|---|
| Chain state | `cache/x1/train/net0000..0099.npz`: `feats` (16,7,1024) = [mu_pre, sigma, D3, D4, mu_out_u, wnorm, d21e], `pred` (16,1024) post-ReLU chain mean, `truth` | index l = pre-activation z_l of 1-indexed layer l+1 (`mu_pre[0] = 0`, `sigma[0]² = colsum W[0]²` to 6e-6) |
| Exact labels | `cache/twin_labels/fit_000..099.npz` via `_claude_snap/x4ceil/npzmap.py` (memmap; only `mu, var, D3, dG` read) | `mu` vs chain `mu_pre` rel. err ≤ 1.1e-4 at every layer |
| Weights | `mint_numpy(fit.json seeds[i])`, (16,1024,1024) f32 stored (input, output): `z_{l+1} = h_l @ W[l+1]` | sha256 = `fit.json weight_sha256[i]` for all 100; **`pred[l] @ W[l+1] == mu_pre[l+1]` to 3e-7–3e-6 on all 100 nets** — this fixes both the layer convention and the matvec direction |
| Noise floor | labels N = 1e9 ⇒ rel. std of `var` ≈ 4.5e-5 | share of the kink-neuron δvar energy that is label noise: 12 % (L3), 1.7 % (L6), < 0.6 % (L ≥ 9); layers 1–2 are at the floor and are not interpreted |

Definitions (chain index t = 1..15 = target layer; source l = t−1): `δvar_t = sigma_t² − var_t^label` (chain minus truth); `α = mu_pre/sigma`, `g = Φ(α)`, `φ = φ(α)`; post-ReLU chain mean `m_l = pred[l]`; sandwich image of a source vector `u`: `(W u)_i := (u @ W[t])_i`; faces `(W u)²`, crosses `(W u)(W v)`. Diagonal transport `f0 = (g_l² δvar_l) @ (W∘W)` (X1-type), `f0x` the same with the exact `∂Var[relu]/∂σ² = Φ − φ(αΦ + φ)`. Local at t: `[s2_t, mu_t, mu_t², g_t, phi_t]`. Sensitivity weight `w = φ(α_t)²/(4 sigma_t²)` (squared mean-channel sensitivity to a variance error). All fits: OLS with intercept, pooled over the 100 FIT nets per layer (102 400 rows); "per-net" = the same model fitted on each net's 1024 neurons, median R²; "transfer" = pooled coefficients applied per net. Note that **f1 = (W m_l)² ≡ mu_t²** exactly (verified), so the scale-mode face is already a *local* feature of the chain at layer t; f2 = (W m∘g)² is nearly collinear with it (single-face R² within 0.01–0.03 of f1 at L9–15).

## 2. Step 1 + 2: diagonal baseline, faces, upper bound (T1)

| t | rms δvar | rms δvar / rms var | mean δvar/var | loc | f0 | base = loc+f0+f0x | f1 alone (= mu_t²) | f2 alone | f6 alone | faces7 alone | base+faces7 (incr.) | +extra faces (incr.) | base+gram12 (incr.) | **w: base** | **w: base+faces7** | **w: base+gram12** | per-net median R²: base / +faces7 / +gram12 / transfer(+faces7) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.36e-05 | 0.00 % | +0.000 % | 0.026 | 0.004 | 0.030 | 0.011 | 0.011 | 0.000 | 0.011 | 0.030 (+0.000) | 0.030 (+0.000) | 0.030 (+0.000) | 0.021 | 0.021 | 0.022 | 0.032 / 0.035 / 0.046 / 0.025 |
| 2 | 7.88e-05 | 0.01 % | −0.003 % | 0.055 | 0.003 | 0.057 | 0.019 | 0.018 | 0.014 | 0.019 | 0.057 (+0.000) | 0.081 (+0.024) | 0.083 (+0.026) | 0.042 | 0.042 | 0.058 | 0.063 / 0.069 / 0.158 / 0.054 |
| 3 | 1.13e-04 | 0.01 % | −0.006 % | 0.046 | 0.002 | 0.050 | 0.002 | 0.000 | 0.019 | 0.041 | 0.087 (+0.037) | 0.101 (+0.014) | 0.104 (+0.054) | 0.049 | 0.064 | 0.073 | 0.063 / 0.109 / 0.185 / 0.084 |
| 4 | 1.46e-04 | 0.02 % | −0.009 % | 0.086 | 0.003 | 0.091 | 0.039 | 0.023 | 0.080 | 0.096 | 0.146 (+0.055) | 0.153 (+0.007) | 0.156 (+0.065) | 0.055 | 0.071 | 0.077 | 0.109 / 0.168 / 0.238 / 0.142 |
| 5 | 1.78e-04 | 0.03 % | −0.013 % | 0.152 | 0.004 | 0.159 | 0.120 | 0.089 | 0.183 | 0.200 | 0.238 (+0.079) | 0.244 (+0.006) | 0.247 (+0.088) | 0.046 | 0.063 | 0.069 | 0.173 / 0.259 / 0.324 / 0.239 |
| 6 | 2.10e-04 | 0.05 % | −0.019 % | 0.236 | 0.005 | 0.243 | 0.212 | 0.178 | 0.239 | 0.276 | 0.307 (+0.063) | 0.314 (+0.007) | 0.316 (+0.073) | 0.040 | 0.054 | 0.060 | 0.253 / 0.320 / 0.381 / 0.299 |
| 7 | 2.39e-04 | 0.06 % | −0.027 % | 0.303 | 0.007 | 0.309 | 0.291 | 0.263 | 0.267 | 0.333 | 0.351 (+0.042) | 0.361 (+0.010) | 0.364 (+0.054) | 0.026 | 0.034 | 0.039 | 0.317 / 0.363 / 0.423 / 0.347 |
| 8 | 2.61e-04 | 0.08 % | −0.033 % | 0.352 | 0.007 | 0.358 | 0.341 | 0.314 | 0.288 | 0.378 | 0.395 (+0.037) | 0.404 (+0.008) | 0.407 (+0.048) | 0.028 | 0.035 | 0.039 | 0.371 / 0.406 / 0.462 / 0.391 |
| 9 | 2.87e-04 | 0.10 % | −0.040 % | 0.407 | 0.008 | 0.413 | 0.398 | 0.373 | 0.322 | 0.434 | 0.448 (+0.036) | 0.459 (+0.011) | 0.462 (+0.050) | 0.023 | 0.031 | 0.036 | 0.413 / 0.452 / 0.508 / 0.438 |
| 10 | 3.11e-04 | 0.12 % | −0.048 % | 0.447 | 0.009 | 0.451 | 0.439 | 0.415 | 0.350 | 0.477 | 0.489 (+0.038) | 0.499 (+0.010) | 0.503 (+0.052) | 0.020 | 0.027 | 0.031 | 0.451 / 0.494 / 0.545 / 0.476 |
| 11 | 3.37e-04 | 0.14 % | −0.055 % | 0.500 | 0.010 | 0.503 | 0.495 | 0.474 | 0.377 | 0.525 | 0.533 (+0.030) | 0.544 (+0.011) | 0.548 (+0.045) | 0.014 | 0.020 | 0.024 | 0.505 / 0.541 / 0.589 / 0.523 |
| 12 | 3.63e-04 | 0.17 % | −0.064 % | 0.547 | 0.011 | 0.549 | 0.544 | 0.526 | 0.397 | 0.568 | 0.574 (+0.024) | 0.584 (+0.010) | 0.588 (+0.039) | 0.013 | 0.018 | 0.022 | 0.542 / 0.577 / 0.615 / 0.554 |
| 13 | 3.87e-04 | 0.19 % | −0.074 % | 0.568 | 0.013 | 0.570 | 0.565 | 0.549 | 0.404 | 0.587 | 0.592 (+0.022) | 0.602 (+0.010) | 0.607 (+0.037) | 0.009 | 0.013 | 0.017 | 0.560 / 0.590 / 0.637 / 0.573 |
| 14 | 4.16e-04 | 0.22 % | −0.084 % | 0.612 | 0.013 | 0.613 | 0.610 | 0.597 | 0.430 | 0.629 | 0.631 (+0.018) | 0.641 (+0.009) | 0.645 (+0.032) | 0.006 | 0.009 | 0.013 | 0.615 / 0.642 / 0.676 / 0.626 |
| 15 | 4.35e-04 | 0.26 % | −0.094 % | 0.638 | 0.015 | 0.639 | 0.638 | 0.625 | 0.443 | 0.654 | 0.656 (+0.017) | 0.665 (+0.009) | 0.671 (+0.031) | 0.004 | 0.007 | 0.012 | 0.643 / 0.661 / 0.696 / 0.641 |

`faces7` = f1…f7 of the brief; `extra` = (Wg)², (W D4)², (W sd_relu)², (W var_relu)², (W 1)², (Wm)(W D3), (Wm)(W σ), (Wm)(W g), (Wm)(W sd_relu), (W d21e)²; `gram12` = all 78 products of 12 projections (upper bound for any covariance correction `Σ A_ab b_a b_bᵀ` in the span of 12 per-neuron source vectors). Adding the oracle faces (W δm)², (Wm)(W δm) with δm = pred − truth changes nothing (≤ +0.001 at every layer).

Reading:
- **Step 1 as expected:** the X1-type diagonal message f0 has R² 0.003–0.015 (the W∘W transport of a per-neuron variance error is a nearly uniform shift — it sits in the intercept, not in the neuron-to-neuron variation). The local features carry everything the baseline has, and of those it is `mu_t²` alone (0.40 → 0.64 at L9–15); `s2_t` alone is 0.02–0.21.
- **Step 2:** the seven faces add +0.017…+0.079 unweighted over the baseline (peak at L5–6, decaying to +0.017 at L15), the extra faces +0.01, and the 78-product upper bound only +0.03…+0.09 over the baseline — the baseline's `mu_t²` *is* the dominant face (f1 ≡ mu_t²). Per-net medians equal the pooled values and pooled coefficients transfer (per-net transfer R² within 0.02 of per-net fits): the structure is stable across nets.
- **Weighted:** 0.004–0.06 for every model at every layer; at L9–15 the baseline is 0.004–0.023, faces add +0.003…+0.008, the gram upper bound reaches 0.012–0.036.

## 3. The scale-mode face: `δvar_t = −ρ_t mu_t² + c_t` (T3) and where the error lives (T5)

| t | ρ_t pooled | ρ_t per-net median [IQR] | CV across nets | c_t pooled | R² pooled | R² per-net median | R²_w per-net median | ρ_t − ρ_{t−1} | in-layer 2b [mu², (mu g)², D3·mu, s2] R² u / w |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2.9e-06 | 3.1e-06 [0.8e-06, 4.9e-06] | 0.88 | −3.8e-05 | 0.002 | 0.003 | 0.000 | +1.0e-05 | 0.048 / 0.046 |
| 4 | 1.39e-05 | 1.40e-05 [1.19e-05, 1.64e-05] | 0.26 | −3.8e-05 | 0.039 | 0.040 | 0.001 | +1.1e-05 | 0.090 / 0.052 |
| 5 | 2.75e-05 | 2.77e-05 [2.52e-05, 3.10e-05] | 0.19 | −2.8e-05 | 0.120 | 0.121 | 0.003 | +1.4e-05 | 0.161 / 0.042 |
| 6 | 4.04e-05 | 4.06e-05 [3.81e-05, 4.54e-05] | 0.14 | −2.2e-05 | 0.212 | 0.212 | 0.003 | +1.3e-05 | 0.244 / 0.036 |
| 7 | 5.06e-05 | 5.24e-05 [4.74e-05, 5.64e-05] | 0.14 | −2.1e-05 | 0.291 | 0.297 | 0.006 | +1.0e-05 | 0.310 / 0.022 |
| 8 | 5.84e-05 | 6.03e-05 [5.58e-05, 6.48e-05] | 0.13 | −1.6e-05 | 0.341 | 0.350 | 0.005 | +7.8e-06 | 0.357 / 0.023 |
| 9 | 6.66e-05 | 6.80e-05 [6.31e-05, 7.33e-05] | 0.13 | −9.2e-06 | 0.398 | 0.393 | 0.005 | +8.2e-06 | 0.411 / 0.018 |
| 10 | 7.46e-05 | 7.64e-05 [7.05e-05, 8.34e-05] | 0.12 | −3.6e-06 | 0.439 | 0.440 | 0.004 | +7.9e-06 | 0.450 / 0.018 |
| 11 | 8.35e-05 | 8.71e-05 [7.99e-05, 9.28e-05] | 0.12 | +7.2e-06 | 0.495 | 0.497 | 0.004 | +8.9e-06 | 0.502 / 0.013 |
| 12 | 9.13e-05 | 9.48e-05 [8.58e-05, 1.01e-04] | 0.11 | +1.5e-05 | 0.544 | 0.536 | 0.005 | +7.8e-06 | 0.549 / 0.013 |
| 13 | 9.84e-05 | 1.00e-04 [9.17e-05, 1.08e-04] | 0.12 | +2.1e-05 | 0.565 | 0.556 | 0.005 | +7.2e-06 | 0.569 / 0.010 |
| 14 | 1.06e-04 | 1.09e-04 [9.95e-05, 1.17e-04] | 0.11 | +2.9e-05 | 0.610 | 0.610 | 0.005 | +7.9e-06 | 0.613 / 0.009 |
| 15 | 1.13e-04 | 1.15e-04 [1.08e-04, 1.24e-04] | 0.11 | +3.3e-05 | 0.638 | 0.636 | 0.004 | +6.3e-06 | 0.638 / 0.007 |

(L1–2: ρ ≈ −7e-6, R² 0.01–0.02, at the label-noise floor.) ρ_t > 0 means the chain's variance is **too small** on large-|μ| neurons, i.e. the exact `Cov(h_l)` has a scale-mode component `ρ m mᵀ` beyond what the chain's pairwise closure produces; ρ grows ≈ linearly with a per-layer innovation of ≈ 1.3e-5 (L4–6) falling to ≈ 7e-6 (L9–15), consistent with a self-transporting rank-1 state (`Φ∘(W m)` ≈ m_{t} on decided neurons) plus a small innovation each layer. The per-net CV of ρ_t is **0.11–0.14 at L6–15** — by the brief's criterion (CV < 50 %) this is a usable face, and the only one: of the other faces in the joint model (T4 in `tables.md`) f2 (CV 0.43–0.69), f6 (0.37–0.59) and f4 (0.50–0.68) are borderline and f3/f5/f7 are not stable (CV 0.8–5.6). For scale (order of magnitude only): ρ_15 = 1.1e-4 against the full scale-mode strength of a depth-15 He-ReLU layer, `Var(‖h‖)/‖h‖² ≈ 5l/(4n) ≈ 0.018` (Hanin–Nica growth of the norm's fourth moment) — the chain's dense covariance already carries ≈ 99 % of the scale mode; this is the closure residual.

Where the error lives (relative δvar by |α_t| bin, pooled 100 nets; "coherent frac" = (Σwy)²/(Σw·Σwy²) within the bin):

| t | bin | n share | mean δvar/var | std δvar/var | share of w-energy | coherent frac |
|---|---|---|---|---|---|---|
| 6 | 0–0.5 | 0.21 | −0.006 % | 0.034 % | 0.52 | 0.03 |
| 6 | 0.5–1 | 0.20 | −0.008 % | 0.034 % | 0.32 | 0.04 |
| 6 | 1–2 | 0.31 | −0.014 % | 0.037 % | 0.15 | 0.10 |
| 6 | 2–3 | 0.18 | −0.031 % | 0.042 % | 0.00 | 0.29 |
| 6 | > 3 | 0.10 | −0.064 % | 0.054 % | 0.00 | 0.52 |
| 9 | 0–0.5 | 0.16 | −0.004 % | 0.058 % | 0.51 | 0.00 |
| 9 | 0.5–1 | 0.16 | −0.007 % | 0.059 % | 0.33 | 0.01 |
| 9 | 1–2 | 0.27 | −0.019 % | 0.061 % | 0.15 | 0.06 |
| 9 | 2–3 | 0.20 | −0.045 % | 0.068 % | 0.01 | 0.23 |
| 9 | > 3 | 0.21 | −0.119 % | 0.097 % | 0.00 | 0.48 |
| 12 | 0–0.5 | 0.13 | +0.008 % | 0.085 % | 0.51 | 0.01 |
| 12 | 0.5–1 | 0.13 | +0.003 % | 0.088 % | 0.33 | 0.00 |
| 12 | 1–2 | 0.24 | −0.013 % | 0.091 % | 0.15 | 0.01 |
| 12 | 2–3 | 0.20 | −0.051 % | 0.099 % | 0.01 | 0.14 |
| 12 | > 3 | 0.30 | −0.176 % | 0.152 % | 0.00 | 0.40 |
| 15 | 0–0.5 | 0.11 | +0.023 % | 0.117 % | 0.51 | 0.04 |
| 15 | 0.5–1 | 0.11 | +0.019 % | 0.121 % | 0.33 | 0.02 |
| 15 | 1–2 | 0.21 | −0.001 % | 0.124 % | 0.15 | 0.00 |
| 15 | 2–3 | 0.19 | −0.050 % | 0.132 % | 0.01 | 0.06 |
| 15 | > 3 | 0.38 | −0.235 % | 0.211 % | 0.00 | 0.28 |

The coherent part (mean −0.12 … −0.24 % at |α| > 3, 28–52 % of that bin's energy) is the `−ρ μ²` face; it carries **0.00–0.01 of the sensitivity-weighted energy**. The kink neurons (|α| < 1; 84 % of the weighted energy at every layer ≥ 5) have a relative error that is zero-mean to ±0.02 % with a neuron-specific spread of 0.03–0.12 % — coherent fraction 0.00–0.04 — and this spread is 60–700× the label-noise floor. It is not a function of α (bins), not of σ², D3, g, φ, m (T1 local), not of any transported diagonal (f0), and not of any rank-few face in the 12-vector span (gram12).

## 4. Rank structure of the best covariance correction (T6) — the sandbox hypothesis check

Best quadratic correction `Σ_ab A_ab (W b_a)(W b_b)` in an orthonormalised 11-vector basis [m, m∘g, D3, σ, d21e, m∘φ, g, D4, sd_relu, var_relu, 1] (base here = local without mu_t², + f0, f0x); eigen-decomposition of A is basis-invariant; "rank-r refit" = base + the top-r eigen-faces `sign(λ_j)(W u_j)²` refitted.

| t | u: base | u: base+quad | PR | λ₁² share | cos(u₁, W m) | u: rank-1 / 2 / 4 refit | w: base | w: base+quad | PR_w | cos_w(u₁, W m) | w: rank-1 / 2 / 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.072 | 0.155 | 1.85 | 0.98 | 0.845 | 0.153 / 0.154 / 0.155 | 0.055 | 0.077 | 2.02 | 0.829 | 0.074 / 0.076 / 0.076 |
| 6 | 0.125 | 0.316 | 1.45 | 0.99 | 0.947 | 0.313 / 0.315 / 0.316 | 0.040 | 0.060 | 1.51 | 0.945 | 0.054 / 0.058 / 0.059 |
| 8 | 0.160 | 0.406 | 1.26 | 1.00 | 0.974 | 0.405 / 0.406 / 0.406 | 0.028 | 0.039 | 1.31 | 0.972 | 0.036 / 0.037 / 0.038 |
| 9 | 0.169 | 0.462 | 1.27 | 1.00 | 0.976 | 0.460 / 0.462 / 0.462 | 0.023 | 0.036 | 1.35 | 0.971 | 0.031 / 0.034 / 0.035 |
| 10 | 0.180 | 0.503 | 1.23 | 1.00 | 0.978 | 0.501 / 0.502 / 0.502 | 0.020 | 0.031 | 1.21 | 0.980 | 0.028 / 0.030 / 0.031 |
| 11 | 0.205 | 0.547 | 1.20 | 1.00 | 0.982 | 0.546 / 0.547 / 0.547 | 0.014 | 0.024 | 1.25 | 0.978 | 0.021 / 0.023 / 0.024 |
| 12 | 0.224 | 0.588 | 1.19 | 1.00 | 0.985 | 0.586 / 0.587 / 0.588 | 0.012 | 0.022 | 1.22 | 0.985 | 0.018 / 0.021 / 0.022 |
| 13 | 0.242 | 0.607 | 1.21 | 1.00 | 0.987 | 0.605 / 0.606 / 0.607 | 0.008 | 0.016 | 1.20 | 0.986 | 0.013 / 0.015 / 0.016 |
| 14 | 0.271 | 0.645 | 1.17 | 1.00 | 0.989 | 0.644 / 0.644 / 0.645 | 0.006 | 0.013 | 1.23 | 0.986 | 0.010 / 0.011 / 0.012 |
| 15 | 0.301 | 0.670 | 1.19 | 1.00 | 0.989 | 0.669 / 0.670 / 0.670 | 0.003 | 0.012 | 1.21 | 0.989 | 0.007 / 0.010 / 0.011 |

Against the sandbox (PR 2–4, dominant direction scale-mode-like, top-8 directions carrying 80–95 % of the induced next-layer variance error): PR is 1.2–1.5 (more rank-1 than predicted), the dominant direction is `m` (cos 0.95–0.99), and the rank-1 face alone recovers all of the quadratic model's gain (rank-1 = rank-4 to ≤ 0.003). But the share of the induced variance error it carries is +19 … +37 pp of R² over a base without mu² at L6–15 (31 → 67 % in total), not 80–95 %, and sensitivity-weighted the whole quadratic family adds 0.01–0.02. The remaining unweighted 33–68 % and weighted 96–99 % are outside the span of per-neuron-generated faces.

## 5. Step 3 — translation to final MSE

Pooled over layers 9–15 (T7): fraction of δvar energy explained, centered (R²) and including the per-layer intercept (a uniform variance shift, itself zero-FLOP):

| model | R² unweighted | **R² weighted** | energy incl. intercept (u) | **(w)** |
|---|---|---|---|---|
| intercept only | — | — | 0.176 | 0.010 |
| base = loc + f0 + f0x | 0.555 | 0.012 | 0.633 | 0.022 |
| base + faces7 | 0.579 | 0.017 | 0.653 | 0.027 |
| base + faces7 + extra | 0.589 | 0.019 | 0.662 | 0.029 |
| base + gram12 (upper bound) | 0.594 | 0.021 | 0.665 | 0.031 |
| scale-mode face alone (per layer, T3) | 0.40–0.64 | 0.004–0.006 | | |

Scale of the whole covariance channel: oracle exact pre-activation C at every layer ⇒ −38 % final MSE (job 3476). The share of the *sensitivity-weighted* variance-error energy at L9–15 is 0.5 % for the scale-mode face alone, 1.2 % for the baseline (which contains it), 1.7 % for baseline + all seven faces, 2.1 % for any face in the 12-vector span, 3.1 % with the per-layer intercept counted as well. With the 0.5 refit-absorption discount:

**expected −38 % × (0.005 … 0.03) × 0.5 = −0.1 … −0.6 % raw, central ≈ −0.3 %.** (Using the unweighted 0.58–0.66 would give −11 … −13 %; that credits the decided-neuron error, which the mean closure does not read: on a decided neuron E[relu] = μ or 0 independent of σ, and its variance error transports to the next layer only as the uniform shift that the intercept — 1 % of the weighted energy — already represents.)

## 6. Step 4 — insertion (given that f1 has R² ≥ 0.3 unweighted at L ≥ 8; flagged mean-neutral)

Diagonal form (n FLOPs per layer, the exact zero-FLOP version since `mu_t` is already in the chain): after the sandwich at layer t,

    var_t += rho_t * mu_t**2          # chain is low by rho_t mu_t^2 on large-|mu| neurons

with the per-layer ρ_t starting table (T3, pooled; per-net IQR ± 8 %): L3 0.3e-5, L4 1.4e-5, L5 2.8e-5, L6 4.0e-5, L7 5.1e-5, L8 5.8e-5, L9 6.7e-5, L10 7.5e-5, L11 8.4e-5, L12 9.1e-5, L13 9.8e-5, L14 1.06e-4, L15 1.13e-4, plus the intercepts c_t (−3.8e-5 at L3–4 → +3.3e-5 at L15, i.e. a per-layer uniform variance gain). As a gain group in the second-stage refit: one ρ per layer (13 scalars) or the two-parameter law ρ_t ≈ 1.3e-5·(t−3) for t ≤ 7, 5.1e-5 + 7.8e-6·(t−7) for t ≥ 7.

Full rank-1 form (one n² outer product per layer; transports by itself afterwards): before the sandwich of layer t, `Cov(h_{t−1}) += Δρ_t m_{t−1} m_{t−1}ᵀ`, equivalently `C_t += Δρ_t (W m_{t−1})(W m_{t−1})ᵀ = Δρ_t mu_t mu_tᵀ`, with the **innovation** Δρ_t = ρ_t − ρ_{t−1} ≈ 1.0–1.4e-5 (L3–6), 1.0e-5 (L7), 6–9e-6 (L8–15) — not the cumulative ρ_t, since the inserted face is carried forward by the chain's own sandwich-plus-closure (gain ≈ 1 on decided neurons). The off-diagonal part changes the next layer's D22/Q31 inputs by O(Δρ · μ_i μ_j), which is 1e-5 relative — nothing. Either form is predicted mean-neutral (−0.1 … −0.6 % raw); run it only as a free rider in a refit that is happening anyway, and read its fitted ρ_t against the table above (a refit that drives ρ to compensate the corrector rather than to 0.7–1.1e-4 at depth is fitting noise).

## 7. What this says for X6

- The unknown X6 §0a named — "whether the real 1024×16 chain's covariance error is as coherent as the towers'" — is now measured: in the direction a learned per-neuron-emitted rank-k state can represent, it is *perfectly* coherent (one direction, cos 0.99 with m, PR 1.2) and worth 0.4–0.6 % of the sensitivity-weighted variance error. The 96–99 % that reaches the final MSE through kink neurons is not in the span of faces generated from the per-neuron layer state (T6 upper bound, weighted, 0.012–0.036 at L9–15), so a k = 16 or 32 state whose innovations come from a per-neuron MLP has the same ceiling; X6's quadratic-readout arm should be expected to beat X1-A by < 0.3 pp, which is its own pre-registered stop line.
- The kink-neuron variance error is pair-structured (it needs δC_jk that is not `a_j b_k`): the candidates are the chain's bivariate closure residual per pair and the hub / dressing terms. Those are n²-FLOP objects, not zero-FLOP faces, and the X6 spec's own sandbox number for a missing-hub error (PR 10–40, rank-16 R² 0.65–0.93) is the optimistic case for them; this measurement cannot separate them.

## 8. Caveats

- `δvar` is measured against the chain's own tap (`sigma` in `cache/x1/train`), which is the consumed state after 8be's in-loop corrections; a face could look different against an un-corrected variance, but it is the consumed variance that reaches the mean closure.
- Faces were tested through the sandwich of the exact regenerated weights; the gram upper bound spans 12 (or 11 non-collinear) per-neuron source vectors — a per-neuron MLP could emit other vectors, but the α-bin coherent fractions (≤ 0.04 on kink neurons) and the failure of every local feature say the kink-neuron error is not a function of the per-neuron state at all.
- Layers 1–2 are at the N = 1e9 label-noise floor (chain exact there) and are shown but not read; layer 3 has a 12 % noise share on kink neurons.
- The −38 % oracle is the whole covariance channel (all layers, all neurons, incl. its κ3/κ4 side effects); attributing the face's share linearly in sensitivity-weighted energy and applying the 0.5 refit discount is the same convention as H §6.
