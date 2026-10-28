# H — Ceiling of worker spec X4's "SSC / scale-mixture κ4 faces", measured on the 8be chain before any refit

Measured 2026-10-20 on sam-mega-pc (CPU only, ≤ 8 threads, no GPU). Scripts, per-net caches and result JSONs are under `_claude_snap/x4ceil/` in the repo (`npzmap.py`, `collect_diag.py`, `analyze_diag{,2,3}.py`, `collect_dense.py`, `analyze_dense.py`, `mdtables.py`; `diag_results.json`, `diag_results2.json`, `dense_results.json`). Nothing else in the repo was touched.

## 0. Verdict

**The SSC face is not where the κ4 error lives. Expect ≈ −0.2 % raw (range −0.05 … −0.5 %) from the SSC item of X4 after a matched refit, against the strategist's −1.5 %.**

- **Diagonal (D4) channel, real chain residual (100 FIT nets):** the var² face alone removes 43 % of Σ r² unweighted but only **14 % when weighted by the mean-closure sensitivity** to κ4, and **≈ 0 % at layers 12–16**. On top of the chain's existing re-scalable terms (a per-layer gain on D4, the D3²/var and μ·D3 laws — all of which a matched CONTROL refit also retunes), the face adds **+2.9 % unweighted / +2.3 % weighted**. What the face does explain (55–63 % at layers 2–4) is a *coherent ≈ 10 % under-delivery of D4* that the chain's own gains could already remove and that the K9.4b refit chose to leave in place (8be's c₀ at layers 2–4 is 0.60/0.73/0.80 vs the label fit's 0.99).
- **(3,1) channel:** the SSC Q31 face `3 v var_i C_ij` (= `3 v ρ_ij` in Q31n) **is identically the kai-law b₁ term** that 8be already carries per layer (`_K4T1_B1`). Its ceiling as a *new* gain is zero by construction; it only re-parametrises b₁. Exact Q31 has cos² 0.48–0.59 on that direction; the kai law with 8be's (b₁, b₂) leaves a residual of which 35–43 % is on the face at layers 2–4 but < 5 % at layers 9–15, and at layers 2–6 the Tier-1.5 fold births (c31/c211/cs ≈ 0.6–1.2 in 8be) already fill the early-layer slot that a larger b₁ would fill (k4 REPORT §8: `tier1_5_nodiag` b₁ → 0.0002 at l = 1 once births are in).
- **(2,2) channel:** the shipped form `(dG_i + dG_j)/3` with exact dG explains 92–99 % of exact D22; the residual has **0.00 %** on the face `v (var_i var_j + 2 C_ij²)` at every layer (fitted v = 0.0000). The face is collinear with the form itself (cos² 0.92–0.99). Dead.
- **Fitted v_l (labels, chain residual, face alone):** 0.00034 (L2) → 0.0006 (L6) → 0.0016–0.0039 (L9–16) unweighted, i.e. 0.4–0.65 × the sandbox's v_eff(l) = 1/(2n) + 0.4(l−1)/n; sensitivity-weighted: 0.0003–0.0005 at L2–6, ≤ 0.0012 at L9–11, ≈ 0 at L12–16. The spec's sanity range "v_l ≈ 0.002–0.01" is failed on exact labels at layers 2–8 even before compensation.
- The graded sibling (Susupro, ×0.946) got its gain from *decoupling λ's three uses*; 8be already has them decoupled (Q31 = kai law, D4 = c₀·dG + D3²/var + μD3 + lineage readouts, D22 = dG form). That mechanism is not available to us a second time.

## 1. What data exists (paths, keys, shapes)

| data | path | keys / shapes | notes |
|---|---|---|---|
| Exact N = 1e9 labels, FIT = `full` rows 0–99 | `cache/twin_labels/fit_000.npz … fit_099.npz` (100 files, 268.7 MB each) | `mu, var, D3 (= κ3_iii), dG (= κ4_iiii/2)` f32 (16, 1024); `name` <U; `C11` (diag 0), `D21 = κ3_iij`, `D22 = κ4_iijj`, `Q31 = κ4_iiij` f32 (16, 1024, 1024) | Uncompressed zip members: `npzmap.py` memmaps each member at its byte offset (row blocks read in ms; no 1 GB member reads). Layer index l = pre-activation z_l of 1-indexed layer l+1; index 0 is the Gaussian input layer (dG rms 3e-4 = noise). |
| Chain's own per-layer diagonal state, exact 8be source (cupy shim, `_CORR_HOOK` tap, written by `experiments/2026-10-20_x1/dump.py`) | `cache/x1/train/net0000.npz … net0899.npz` (TRAIN = `full` 0–899; 656 KB each); also `cache/x1/val/` (100, real numpy chain) and `cache/x1/fresh_a/` | `feats` f32 (16, 7, 1024) = [`mu_pre, sigma, D3, D4, mu_out_u, wnorm, d21e`]; `pred` f32 (16, 1024); `truth` f64 (16, 1024) | `D4` is `_D4c` = the consumed D4 (`_K4T1_OLDD4 = False`), i.e. metric·(c₀ dG + c_D D3²/var + c_m μD3 + lineage readouts) + the K9.1b mix, after `_k2_fill` on the last layer. net i ↔ fit_i for i < 100. **Alignment verified:** chain `mu_pre` vs label `mu` relative error ≤ 1e-4 at every layer ≥ 1; chain σ² vs label var 0.03 % (L2) → 0.3 % (L16). |
| Chain Q31 / D22 / dG dumps | **none in MAIN.** The k4 lane's follow-up used Lane K8's v90a dump at `~/mnt/arc-whest-lane-k/experiments/2026-10-15_k8/dump/dump_NNN.npz` (lane-k worktree, not in the connected folder), and that dump held the consumed `mu, var, D3, D21` plus towers, not Q31/D22. `experiments/2026-10-16_k9/` holds pod scripts and `coef_tables_v93r_o9.json` only. | | (3,1) and (2,2) are therefore measured label-side: fraction of exact energy on the face, and the residual of the 8be *forms* evaluated on exact inputs. |
| 8be coefficients used | `experiments/2026-10-20_x1/estimator_8be.py` lines 2888–2898 (`_K4T1_B1/_B2/_C0/_CD/_CM`, `_K4T1_GD = 0.5`, `_K4M_DS`), D22 form at line 3253 `((metric/6)(dG_i + dG_j)) * off_mask`, last-layer fill `_k2_fill` line 2917 | | Note `_k2_fill` fills the 320 non-O rows of the last layer with q̄·var² — itself a var² face — so the L16 diagonal row below is mildly contaminated in the face's favour. |

Compute note: `device_bash` runs in a 2-CPU / 3 GiB Linux VM over a ≈ 230 MB/s mount, not on the 64 GiB host; everything below is row-block / sufficient-statistics work and ran in < 1 min per call.

## 2. Definitions

- Diagonal: `D4_exact = 2·dG_label = κ4_iiii`; `r_i = D4_exact,i − D4_chain,i`; face `f_i = 3·var_chain,i²` (exact var gives identical numbers to 3 decimals). Fits are no-intercept least squares pooled over nets; "frac" = 1 − Σ(r − v f)²/Σ r² = fraction of residual energy removed. Weights: `mean` = |α²−1| φ(α)/σ³ (= 24·∂μ_out/∂κ4, the Edgeworth mean-closure sensitivity at the layer itself), `cov` = |α³−3α| φ(α)/σ⁴ (k = 1 covariance-closure path), `kink` = 1[|α| < 2]. Under neutral error transport (X1 measured gain 1.00 ± 0.03 per layer) the weighted energy is the right proxy for what reaches layer 16.
- (3,1): target exact `Q31[i,j] = κ(z_i,z_i,z_i,z_j)`, face `f1 = var_i C_ij`, kai term `f2 = D3_i D21_ij / var_i` (so `Q31n = b₁ ρ + b₂ D3n·D21n`). "kai law (8be)" = `b₁ f1 + b₂ f2` with 8be's lists. 512-row blocks × 1023 columns per layer per net, 40 nets (21 M pairs per layer). Weight `w31 = |α_i| φ_i/σ_i² · Φ_j`.
- (2,2): target exact `D22[i,j] = κ4_iijj`, i ≠ j; form `g1 = (dG_i + dG_j)/3` with exact dG; face `g2 = var_i var_j + 2 C_ij²`, also split into `g3 = var_i var_j`, `g4 = C_ij²`. Weight `w22 = φ_i φ_j/(σ_i σ_j)`.
- v_pred = 1/(2n) + 0.4 (l−1)/n at n = 1024 (sandbox emergent-mode prediction; l 1-indexed).

## 3. Diagonal channel (100 FIT nets)

### T1 — residual size, face alone, fitted v_l

| L | rms D4_ex | rms r | r/D4 | slope D4_ex/D4_ch | v_l (face alone) | frac Σr² removed | v_l per-net median [IQR] | v_w (mean-sens.) | frac_w | v_pred (sandbox) | v_l / v_pred |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.83e-02 | 2.43e-03 | 0.132 | 1.118 | 0.00034 | 0.635 | 0.00034 [0.00032, 0.00036] | 0.00034 | 0.628 | 0.00088 | 0.39 |
| 3 | 1.69e-02 | 2.21e-03 | 0.131 | 1.110 | 0.00054 | 0.586 | 0.00053 [0.00050, 0.00057] | 0.00051 | 0.551 | 0.00127 | 0.43 |
| 4 | 1.38e-02 | 1.52e-03 | 0.110 | 1.061 | 0.00043 | 0.297 | 0.00043 [0.00037, 0.00049] | 0.00050 | 0.369 | 0.00166 | 0.26 |
| 5 | 1.10e-02 | 1.21e-03 | 0.109 | 1.051 | 0.00043 | 0.203 | 0.00042 [0.00034, 0.00049] | 0.00035 | 0.148 | 0.00205 | 0.21 |
| 6 | 8.91e-03 | 1.07e-03 | 0.120 | 1.060 | 0.00059 | 0.238 | 0.00057 [0.00046, 0.00066] | 0.00046 | 0.172 | 0.00244 | 0.24 |
| 7 | 7.25e-03 | 8.57e-04 | 0.118 | 1.032 | 0.00038 | 0.077 | 0.00035 [0.00025, 0.00046] | 0.00031 | 0.061 | 0.00283 | 0.13 |
| 8 | 6.00e-03 | 9.33e-04 | 0.156 | 1.057 | 0.00075 | 0.145 | 0.00070 [0.00052, 0.00088] | 0.00008 | 0.003 | 0.00322 | 0.23 |
| 9 | 5.01e-03 | 9.58e-04 | 0.191 | 1.124 | 0.00159 | 0.361 | 0.00154 [0.00132, 0.00173] | 0.00081 | 0.192 | 0.00361 | 0.44 |
| 10 | 4.28e-03 | 8.98e-04 | 0.210 | 1.128 | 0.00180 | 0.334 | 0.00175 [0.00153, 0.00190] | 0.00069 | 0.119 | 0.00400 | 0.45 |
| 11 | 3.68e-03 | 1.05e-03 | 0.287 | 1.197 | 0.00298 | 0.425 | 0.00283 [0.00254, 0.00314] | 0.00122 | 0.241 | 0.00439 | 0.68 |
| 12 | 3.22e-03 | 9.50e-04 | 0.295 | 1.153 | 0.00272 | 0.292 | 0.00248 [0.00215, 0.00286] | 0.00034 | 0.018 | 0.00479 | 0.57 |
| 13 | 2.83e-03 | 8.98e-04 | 0.317 | 1.182 | 0.00332 | 0.334 | 0.00306 [0.00271, 0.00357] | 0.00066 | 0.058 | 0.00518 | 0.64 |
| 14 | 2.52e-03 | 7.84e-04 | 0.311 | 1.167 | 0.00325 | 0.299 | 0.00302 [0.00255, 0.00351] | 0.00029 | 0.010 | 0.00557 | 0.58 |
| 15 | 2.28e-03 | 8.16e-04 | 0.358 | 1.172 | 0.00390 | 0.291 | 0.00352 [0.00302, 0.00411] | −0.00003 | 0.000 | 0.00596 | 0.65 |
| 16 | 2.05e-03 | 5.57e-04 | 0.272 | 1.147 | 0.00276 | 0.231 | 0.00258 [0.00214, 0.00297] | −0.00014 | 0.001 | 0.00635 | 0.44 |

- One shared v across layers 2–16: **v = 0.00041**; per-layer fraction removed under it 0.61, 0.55, 0.30, 0.20, 0.22, 0.08, 0.12, 0.16, 0.14, 0.11, 0.08, 0.08, 0.07, 0.06, 0.06; total Σr² removed **0.354** (per-layer v_l: **0.431**).
- Unweighted residual energy share by layer: 0.25, 0.21, 0.10, 0.06, 0.05, 0.03, 0.04, 0.04, 0.03, 0.05, 0.04, 0.03, 0.03, 0.03, 0.01 — layers 2–3 hold 46 %. Mean-sensitivity-weighted shares: 0.04, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08, 0.10 — flat to rising with depth. Weighted energy removed by the face per layer, as a fraction of the total weighted residual energy: 0.023, 0.029, 0.019, 0.007, 0.009, 0.003, 0.000, 0.013, 0.007, 0.019, 0.001, 0.004, 0.000, 0.000, 0.001 → **sum 0.135**.
- Sensitivity-weighted residual energy / weighted D4_exact energy: 1.7 % (L2) → 1.1 % (L5) → 3.9 % (L16), i.e. the chain's D4 is 10–20 % off in rms where the closure reads it; the coherent part (slope) is 3–20 % by layer.
- Per-net v_l is tight (IQR ≈ ±10 %), so the unweighted v_l is a stable per-layer constant: a real coherent under-scaling, not noise.

### T2 — two-term faces, and the increment of var² over what the refit can already rescale

Columns 6–13: fraction of Σ r² removed by (A) a single per-layer gain on the chain's own D4, (B) A plus free coefficients on D3²/var and μ·D3 (the existing Tier-1 laws), (C) B plus the var² face; the increment C − B is the face's ceiling in a matched refit. Unweighted and mean-sensitivity-weighted.

| L | [var², D3²/var] coefs | frac | [var², var·dG] coefs | frac | rescale D4_ch alone (unw) | +[D3²/v, μD3] | +var² | increment (unw) | rescale (mean-w) | 3-term (w) | +var² (w) | increment (w) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.00035, −0.131 | 0.635 | 0.00025, 0.045 | 0.635 | 0.627 | 0.627 | 0.638 | +0.011 | 0.621 | 0.621 | 0.632 | +0.011 |
| 3 | 0.00052, 0.251 | 0.588 | 0.00034, 0.082 | 0.587 | 0.574 | 0.580 | 0.594 | +0.015 | 0.539 | 0.541 | 0.558 | +0.017 |
| 4 | 0.00057, −0.821 | 0.355 | 0.00146, −0.377 | 0.316 | 0.271 | 0.339 | 0.364 | +0.025 | 0.351 | 0.375 | 0.399 | +0.024 |
| 5 | 0.00035, 0.286 | 0.217 | 0.00051, −0.027 | 0.203 | 0.192 | 0.223 | 0.240 | +0.018 | 0.138 | 0.144 | 0.165 | +0.021 |
| 6 | 0.00044, 0.323 | 0.264 | 0.00051, 0.033 | 0.238 | 0.224 | 0.265 | 0.288 | +0.023 | 0.161 | 0.164 | 0.186 | +0.022 |
| 7 | 0.00031, 0.095 | 0.081 | 0.00052, −0.055 | 0.077 | 0.067 | 0.087 | 0.111 | +0.023 | 0.054 | 0.065 | 0.076 | +0.012 |
| 8 | 0.00006, 0.736 | 0.334 | 0.00134, −0.246 | 0.146 | 0.118 | 0.361 | 0.394 | +0.033 | 0.001 | 0.038 | 0.054 | +0.016 |
| 9 | 0.00076, 0.659 | 0.500 | 0.00121, 0.173 | 0.361 | 0.325 | 0.505 | 0.533 | +0.028 | 0.178 | 0.198 | 0.217 | +0.018 |
| 10 | 0.00075, 0.652 | 0.489 | 0.00182, −0.010 | 0.334 | 0.284 | 0.502 | 0.541 | +0.039 | 0.103 | 0.128 | 0.152 | +0.024 |
| 11 | 0.00135, 0.791 | 0.589 | 0.00731, −2.357 | 0.470 | 0.311 | 0.590 | 0.679 | +0.090 | 0.207 | 0.251 | 0.297 | +0.046 |
| 12 | 0.00049, 0.861 | 0.534 | 0.00813, −2.923 | 0.356 | 0.187 | 0.571 | 0.641 | +0.070 | 0.015 | 0.068 | 0.072 | +0.004 |
| 13 | 0.00084, 0.775 | 0.547 | 0.00752, −2.408 | 0.373 | 0.218 | 0.573 | 0.651 | +0.077 | 0.046 | 0.086 | 0.106 | +0.019 |
| 14 | 0.00057, 0.707 | 0.512 | 0.00652, −1.893 | 0.320 | 0.196 | 0.565 | 0.620 | +0.056 | 0.005 | 0.032 | 0.048 | +0.016 |
| 15 | 0.00060, 0.731 | 0.498 | 0.00951, −3.416 | 0.357 | 0.149 | 0.574 | 0.662 | +0.089 | 0.003 | 0.038 | 0.074 | +0.036 |
| 16 | 0.00063, 0.367 | 0.365 | −0.00025, 1.660 | 0.258 | 0.210 | 0.360 | 0.389 | +0.029 | 0.006 | 0.047 | 0.085 | +0.038 |

Totals over layers 2–16 (fraction of Σ_l Σ r² removed):

| weight | rescale D4 only | + D3²/v, μD3 | + var² | **increment of var²** | var² alone |
|---|---|---|---|---|---|
| unweighted | 0.398 | 0.498 | 0.527 | **+0.029** | 0.431 |
| mean-closure sensitivity | 0.134 | 0.162 | 0.185 | **+0.023** | 0.144 |
| covariance path | 0.069 | 0.092 | 0.114 | **+0.021** | 0.074 |
| kink neurons (|α| < 2) | 0.466 | 0.474 | 0.491 | **+0.017** | 0.478 |

The two-term faces: `[var², D3²/var]` lifts the unweighted fraction to 0.50–0.59 at layers 9–15, but that is the D3²/var law (coefficient 0.65–0.86 — the chain's c_D is 0.86–1.0 and the refit can move it); the var² coefficient inside it drops to 0.0005–0.0013. `[var², var·dG]` with the chain's own dG is unstable (coefficients −2…−3.4 at depth) and adds nothing over the rescale.

### Why the face fails at depth — the normalised view (q = D4/var², i.e. κ4 in units of var²)

| L | mean q_exact | mean q_chain | mean (r/var²) = 3v-offset | std (r/var²) | corr(r/var², |α|) | offset on |α|<1 (share) | 1–2 | 2–3 | >3 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.0133 | 0.0119 | +0.0014 | 0.0011 | −0.01 | +0.0014 (0.86) | +0.0014 (0.14) | — | — |
| 3 | 0.0164 | 0.0148 | +0.0016 | 0.0014 | +0.09 | +0.0015 (0.69) | +0.0017 (0.27) | +0.0020 (0.04) | — |
| 4 | 0.0171 | 0.0161 | +0.0010 | 0.0016 | −0.22 | +0.0012 (0.58) | +0.0009 (0.31) | +0.0002 (0.09) | −0.0005 (0.01) |
| 6 | 0.0162 | 0.0153 | +0.0009 | 0.0017 | +0.18 | +0.0007 (0.45) | +0.0008 (0.32) | +0.0010 (0.16) | +0.0021 (0.07) |
| 8 | 0.0147 | 0.0141 | +0.0007 | 0.0021 | +0.46 | −0.0000 (0.37) | +0.0004 (0.30) | +0.0012 (0.19) | +0.0027 (0.14) |
| 10 | 0.0133 | 0.0120 | +0.0013 | 0.0023 | +0.48 | +0.0005 (0.32) | +0.0008 (0.27) | +0.0014 (0.20) | +0.0032 (0.21) |
| 12 | 0.0121 | 0.0107 | +0.0014 | 0.0029 | +0.60 | +0.0001 (0.28) | +0.0005 (0.25) | +0.0014 (0.20) | +0.0038 (0.27) |
| 14 | 0.0111 | 0.0098 | +0.0013 | 0.0028 | +0.57 | +0.0001 (0.25) | +0.0003 (0.23) | +0.0009 (0.19) | +0.0031 (0.33) |
| 16 | 0.0102 | 0.0092 | +0.0009 | 0.0025 | +0.46 | −0.0002 (0.22) | +0.0001 (0.21) | +0.0007 (0.19) | +0.0022 (0.38) |

At layers 2–3 every neuron is a kink neuron and the chain's normalised κ4 is uniformly ≈ 10 % low — a coherent offset, which is what the var² face (or any D4 gain) fixes. From layer 8 on, the offset lives on the decided neurons (|α| > 2, whose κ4 does not move E[relu]) and is ≈ 0 on |α| < 1; the std of the normalised residual (0.002–0.003) is twice its mean, i.e. the sensitivity-relevant error at depth is incoherent, neuron-specific, and not a function of var.

## 4. (3,1) channel (40 FIT nets, exact labels; no chain Q31 dump exists)

| L | cos²(Q31_ex, var_i C_ij) | proj. coef a | kai law (8be b₁,b₂) R² | kai residual: frac on face | δb₁ = 3v | v | refit (b₁,b₂) R² | b₁* | b₂* | weighted: cos² / kai R² / resid-frac / refit R² | normalised space: cos² / resid-frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.477 | 0.0101 | 0.109 | 0.428 | 0.0091 | 0.0030 | 0.491 | 0.0100 | 1.620 | 0.477 / 0.114 / 0.426 / 0.493 | 0.476 / 0.427 |
| 3 | 0.472 | 0.0146 | 0.215 | 0.353 | 0.0112 | 0.0037 | 0.492 | 0.0141 | 1.443 | 0.472 / 0.217 / 0.353 / 0.493 | 0.472 / 0.354 |
| 4 | 0.496 | 0.0184 | 0.256 | 0.366 | 0.0137 | 0.0046 | 0.529 | 0.0174 | 1.354 | 0.497 / 0.248 / 0.372 / 0.528 | 0.498 / 0.370 |
| 5 | 0.504 | 0.0220 | 0.434 | 0.209 | 0.0107 | 0.0036 | 0.553 | 0.0202 | 1.303 | 0.503 / 0.413 / 0.222 / 0.543 | 0.507 / 0.218 |
| 6 | 0.500 | 0.0253 | 0.468 | 0.188 | 0.0113 | 0.0038 | 0.569 | 0.0222 | 1.277 | 0.494 / 0.425 / 0.211 / 0.547 | 0.504 / 0.200 |
| 7 | 0.492 | 0.0284 | 0.522 | 0.123 | 0.0098 | 0.0033 | 0.583 | 0.0237 | 1.262 | 0.479 / 0.457 / 0.157 / 0.543 | 0.497 / 0.141 |
| 8 | 0.486 | 0.0315 | 0.559 | 0.090 | 0.0090 | 0.0030 | 0.601 | 0.0247 | 1.273 | 0.461 / 0.470 / 0.131 / 0.540 | 0.489 / 0.109 |
| 9 | 0.481 | 0.0348 | 0.597 | 0.049 | 0.0070 | 0.0023 | 0.620 | 0.0253 | 1.274 | 0.440 / 0.481 / 0.096 / 0.531 | 0.483 / 0.069 |
| 10 | 0.482 | 0.0382 | 0.628 | 0.026 | 0.0054 | 0.0018 | 0.640 | 0.0256 | 1.281 | 0.420 / 0.486 / 0.074 / 0.524 | 0.479 / 0.046 |
| 11 | 0.491 | 0.0422 | 0.655 | 0.014 | 0.0042 | 0.0014 | 0.664 | 0.0257 | 1.280 | 0.401 / 0.485 / 0.067 / 0.520 | 0.481 / 0.035 |
| 12 | 0.503 | 0.0461 | 0.684 | 0.001 | −0.0012 | −0.0004 | 0.689 | 0.0252 | 1.285 | 0.384 / 0.500 / 0.026 / 0.513 | 0.483 / 0.004 |
| 13 | 0.527 | 0.0510 | 0.718 | 0.002 | −0.0017 | −0.0006 | 0.722 | 0.0243 | 1.294 | 0.364 / 0.493 / 0.029 / 0.508 | 0.493 / 0.004 |
| 14 | 0.538 | 0.0545 | 0.736 | 0.004 | −0.0025 | −0.0008 | 0.740 | 0.0232 | 1.319 | 0.348 / 0.488 / 0.033 / 0.505 | 0.496 / 0.004 |
| 15 | 0.561 | 0.0587 | 0.728 | 0.109 | −0.0135 | −0.0045 | 0.761 | 0.0221 | 1.333 | 0.331 / 0.502 / 0.000 / 0.502 | 0.506 / 0.028 |
| 16 | 0.593 | 0.0631 | n/a (8be b₁ = b₂ = 0 at the last layer; Q31 unused there) | | | | | | | 0.321 / n/a | 0.519 / n/a |

Reading: half of exact Q31's energy is on the `var_i C_ij` direction at every layer (projection coefficient 0.010 → 0.063, rising with depth — the kai b₁ that saturates). The SSC face adds `3v` to that same coefficient, nothing else. Against the kai law with 8be's lists, the residual's share on the face is 43 % at L2 falling to ≈ 0 at L12–14 (8be's b₁ 0.0232–0.0309 at l = 12–14 vs exact-input optimum 0.022–0.025); at L2–8 the exact-input optimum b₁* is 0.009–0.010 above 8be's, **but** 8be carries the Tier-1.5 fold births in the same slot (k4 REPORT §8: with births the label-fitted b₁ is 0.0002–0.013 at l = 1–5, i.e. the births supply the early-layer `ρ` content). Without a chain Q31 dump this cannot be closed; what can be said is that no *new* direction exists here — a trainable per-layer b₁ (or `_K4T1_GQ`-type gain) already spans it. The normalised-space numbers (Q31n on ρ, the form the spec names) are the same.

## 5. (2,2) channel (40 FIT nets, exact labels)

| L | rms D22_ex | rms form | form R² (gain 1) | form R² (best gain g*) | g* | form residual: frac on g2 | v | frac on free [var_i var_j, C_ij²] | v3 | v4 | cos²(D22_ex, g2) | R² on [g1, g2] | weighted: form R² / resid-frac g2 / resid-frac [g3,g4] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6.10e-03 | 6.08e-03 | 0.988 | 0.988 | 1.00 | 0.0000 | 0.00000 | 0.002 | −0.00001 | 0.0067 | 0.990 | 0.990 | 0.988 / 0.0000 / 0.002 |
| 3 | 5.65e-03 | 5.63e-03 | 0.987 | 0.987 | 1.00 | 0.0000 | 0.00000 | 0.002 | −0.00002 | 0.0082 | 0.988 | 0.989 | 0.987 / 0.0000 / 0.002 |
| 4 | 4.61e-03 | 4.59e-03 | 0.985 | 0.985 | 1.00 | 0.0000 | −0.00000 | 0.002 | −0.00003 | 0.0093 | 0.986 | 0.987 | 0.986 / 0.0000 / 0.002 |
| 5 | 3.65e-03 | 3.63e-03 | 0.983 | 0.983 | 1.00 | 0.0000 | 0.00001 | 0.003 | −0.00004 | 0.0103 | 0.983 | 0.985 | 0.984 / 0.0000 / 0.002 |
| 6 | 2.91e-03 | 2.89e-03 | 0.980 | 0.980 | 0.99 | 0.0000 | 0.00000 | 0.003 | −0.00006 | 0.0116 | 0.980 | 0.982 | 0.980 / 0.0000 / 0.002 |
| 7 | 2.37e-03 | 2.35e-03 | 0.976 | 0.976 | 0.99 | 0.0000 | −0.00000 | 0.004 | −0.00009 | 0.0130 | 0.975 | 0.979 | 0.977 / 0.0001 / 0.002 |
| 8 | 1.94e-03 | 1.93e-03 | 0.971 | 0.971 | 0.99 | 0.0000 | 0.00000 | 0.004 | −0.00011 | 0.0146 | 0.970 | 0.975 | 0.972 / 0.0000 / 0.001 |
| 9 | 1.62e-03 | 1.61e-03 | 0.966 | 0.966 | 0.99 | 0.0001 | 0.00003 | 0.006 | −0.00011 | 0.0168 | 0.965 | 0.970 | 0.968 / 0.0000 / 0.001 |
| 10 | 1.40e-03 | 1.38e-03 | 0.960 | 0.960 | 0.99 | 0.0000 | −0.00002 | 0.008 | −0.00021 | 0.0199 | 0.959 | 0.965 | 0.962 / 0.0004 / 0.001 |
| 11 | 1.19e-03 | 1.17e-03 | 0.955 | 0.955 | 0.99 | 0.0000 | 0.00002 | 0.010 | −0.00024 | 0.0228 | 0.953 | 0.960 | 0.956 / 0.0001 / 0.001 |
| 12 | 1.03e-03 | 1.02e-03 | 0.949 | 0.949 | 0.99 | 0.0000 | 0.00002 | 0.014 | −0.00032 | 0.0262 | 0.947 | 0.955 | 0.951 / 0.0000 / 0.000 |
| 13 | 9.09e-04 | 8.95e-04 | 0.942 | 0.943 | 0.99 | 0.0001 | 0.00005 | 0.019 | −0.00041 | 0.0303 | 0.942 | 0.950 | 0.945 / 0.0000 / 0.000 |
| 14 | 8.07e-04 | 7.92e-04 | 0.936 | 0.936 | 0.99 | 0.0004 | 0.00010 | 0.024 | −0.00047 | 0.0329 | 0.934 | 0.945 | 0.938 / 0.0001 / 0.000 |
| 15 | 7.23e-04 | 7.09e-04 | 0.929 | 0.930 | 0.98 | 0.0004 | 0.00010 | 0.031 | −0.00062 | 0.0359 | 0.928 | 0.940 | 0.931 / 0.0001 / 0.000 |
| 16 | 6.49e-04 | 6.37e-04 | 0.923 | 0.923 | 0.98 | 0.0003 | 0.00011 | 0.039 | −0.00079 | 0.0379 | 0.923 | 0.935 | 0.924 / 0.0001 / 0.000 |

Reading: `κ4_iijj ≈ (κ4_iiii + κ4_jjjj)/6` is the i.i.d.-weight leading order and holds at R² 0.92–0.99 with gain 0.98–1.00; the SSC face is itself 92–99 % collinear with it and holds **none** of the residual (v = 0.0000 ± 0.0001; the only residual structure is a weak `C_ij²` term worth 0.2–3.9 % of a residual that is 1–8 % of a slice that is 4 % of the κ4 cost). In the live chain D22's error is the propagated chain-dG error, i.e. the diagonal channel above.

## 6. Translation to final MSE

Inputs: raw 8be MSE 1.45e-8 (VAL 1.535e-8); κ4-induced isolated cost 1.48e-8 on v93r with dG/D4 55 %, Q31 29 %, D22 4 % (k4, Oct 16); the Tier-1/1.5/2 laws now in 8be cut the isolated dG cost to 0.48–0.68× and the Q31 cost to ≈ 0.62× (k4 §1, §3), so the 8be D4-channel isolated cost is ≈ 3–8e-9 (20–55 % of raw MSE; consistent with the measured sensitivity-weighted residual energy of 1.1–3.9 % of D4's, which under the quadratic "±10 % ⇒ 14–22 %" law is 15–85 % of MSE if coherent, less if not). Realisation of an isolated κ4 improvement in the refit chain: 0.1–0.3 (κ4 refit ceiling realised ≈ 31 % of isolated cost; K9.4b realised −4.2 % raw from a ≈ 37 %-of-MSE isolated reduction ⇒ 0.11; O9.3: exact κ4 is neutral corrector-on).

| channel | share of the measured residual energy the face removes | isolated effect (exact chain, corrector-off) | after matched refit (×0.1–0.3; CONTROL also retunes existing gains) |
|---|---|---|---|
| D4, **incremental over existing re-scalable terms** (the matched-refit quantity) | 2.3 % weighted / 2.9 % unweighted | 0.5–1.3 % of MSE (0.07–0.23e-9) | **−0.05 … −0.4 % raw** |
| D4, face alone (credits the face with the coherent 3–20 % D4 rescale the control's own gains can do) | 13.5 % weighted / 43 % unweighted | 2.7–7.4 % of MSE (weighted) | −0.3 … −2.2 % raw (not a matched-refit number) |
| Q31 face | 0 as a new direction (≡ b₁); label-side kai residual 35–43 % on it at L2–4, < 5 % at L9–15, births already fill L2–6 | ≤ 29 % × 0.62 × 1.48e-8 × (resid share ≤ 0.1 realistic) ≈ ≤ 1.8 % of MSE | ≈ 0 … −0.3 % raw, only if b₁ is not already trainable |
| D22 face | 0.00 % | 0 | 0 |

**Range for the SSC item of X4 after the matched refit: −0.05 … −0.5 % raw, central ≈ −0.2 %**; the upper bound −2 % requires crediting the face with a coherent D4 rescale that the CONTROL arm's own D4 gains reach equally well — and which the K9.4b refit, given exactly that freedom, declined (it set c₀ to 0.60–0.80 at layers 2–4, leaving D4 ≈ 10 % under the exact value). Assumptions: neutral error transport (X1: gain 1.00 ± 0.03), energy-proportional cost within a channel, the k4 shares for 8be scaled by its tier gains, no cross-channel super-additivity (O9.3 κ3+κ4 is super-additive, but X4 touches κ4 only).

Expected fitted v_l if the worker runs it anyway: on the label side 0.0003–0.0006 at L2–7 and ≤ 0.0012 (weighted) at depth, i.e. below the spec's sanity band 0.002–0.01; in the refit the sign and size will be whatever compensates the corrector, not the physical value. The forecast of "v_eff(l) ≈ 1/(2n) + 0.4(l−1)/n" over-predicts the unweighted label-side v_l by 1.5–4× and the sensitivity-weighted one by 3–∞×.

## 7. One-line verdict for the X4 worker

**Expect ≈ −0.2 % raw (−0.05 … −0.5 %) from the SSC faces after the matched refit; the SSC face is not where the κ4 error lives — in D4 it is 2–3 % of the residual beyond gains the control already has (the rest of the D4 error at depth sits on decided neurons and is incoherent on kink neurons), in Q31 it is kai's b₁ by another name, in D22 it is exactly zero.** If X4 runs, run the SSC-only arm so the other three items are not credited to it, and read the D4 var² gain's fitted v_l against 0.0003–0.0006 (L2–7), not against 0.002–0.01.
