# L4 — is the κ3 slice (D21) error pair-local?

Spec: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-20_L4_PAIRLOCAL_K3_SLICE.md` (strategist D1: L1 takes it in the A1 gap).

## 0. Prediction and kill line (on record before any read)
- P(sensitivity-weighted pair-local R² ≥ 0.3 at l ≥ 9) ≈ 0.10. If it fires: per-layer gain group on the D21 read, −3 … −10 % raw after refit.
- **Kill:** weighted R² < 0.15 at every layer ≥ 8 with the full feature set ⇒ write the table, close, stop. No feature tuning beyond §3, no per-net fits.

## 1. Method
- Dump (`dump_l4.py`): served 8be chain (`experiments/2026-10-20_x1/estimator_8be.py`, 8be40031), numpy shim, FIT nets 0–9 (= full rows 0–9,
  labels `cache/twin_labels/fit_00N.npz`, N = 1e9). anatomy2's inert taps (renamed `_l4tap`: 8be has locals called `_a2`/`_A2`) for
  var, D3, dG, D4, C11, D22, Q31; the shipped `_CORR_HOOK` for the D21 the closure and `_corr_raw` actually read (after the KAS rescale,
  which anatomy2's D21 tap precedes). Layers 2–14 (layer 15 forms no D21: last-layer pruning). `--verify` on net 0: recorded run ==
  unrecorded run bit for bit. Dumps in `cache/l4/` (≈ 210 MB/net, not committed).
- Regression (`regress_l4.py`): per layer, target E_ij = D21_chain − D21_label (off-diagonal); 13 pair features from chain quantities
  only (handoff §3 list; x12 omitted — not in the chain); fit on nets 0–7 (250 k random pairs/net), R² out of sample on all pairs of
  nets 8–9. Weights: closure sensitivity w = (½ φ_i/σ_i · Φ_j)² (first-order coefficient of κ3_iij in Cov(relu z_i, relu z_j)),
  the handoff's (φ_iφ_j)² alternative, and unweighted. Models: ridge-13, the 2-feature law (D21_ij, C_ij(D3_i/σ_i² + D3_j/σ_j²)),
  3×32 MLP ceiling. Pair classes by |α| < 1.

## 2. Dump checks (jobs 3593/3594; diagnostic 3596)
- `--verify` net 0: recorded == unrecorded predictions **bit-identical**. Alignment `pred[l−1] @ W_l` vs `mu_pre[l]`: max |Δ| ≤ 5.7e-6 on all 10 nets.
- anatomy2's D3 tap holds only the K1-kept rows on layers 4–14 (992 → 832); the closure's D3 is the hook's (full n), which is what is used.
- **Anatomy does NOT reproduce the handoff's "≈ 6 % at l = 3 → 10–14 % at l = 14".** The served 8be D21 (the one the closure and `_corr_raw`
  read; no reassignment between line 3156 and the hook) is 3.6 % off at l = 2, then **35 % at l = 3 → 68 % at l = 14** (net 8, all ten nets
  agree). Orientation is right (transposed: 125–141 %), and l = 2 matches, so this is the object, not an indexing slip. Split (net 8): the
  KAS-dropped rows (32 → 448 of 1024) carry 13 % → 60 % of the error; the kept rows alone are still 33 % → 66 % off. The pre-k0 tap is worse
  (38 % → 86 %); the k0 gain-formula fill improves it. The 6–14 % figures belong to an older, untruncated tower; on 8be the D21 slice is
  far coarser than the plan assumed.

## 3. Regression (job 3595, `regress.json`; scale-only fit `scale_only.json`)
OOS = nets 8–9, all off-diagonal pairs; fit on nets 0–7, 250 k pairs each. w = closure sensitivity (½ φ_i/σ_i Φ_j)². "scale" = best single
per-layer factor g with D21_label ≈ g · D21_chain (weighted, nets 0–7); "energy removed" = 1 − Σw(E − (1−g)·D)²/Σw E² on nets 8–9.
Classes: kink×kink / kink×decided / decided×decided (|α| < 1).

| l | D21 rel err | R² unw ridge13 | R² (φφ)² ridge13 | **R² w ridge13** | R² w law-2 | R² w MLP 3×32 | R² w scale only | scale g | w-energy removed by g | w-err share kk / kd / dd | R² w per class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3.6 % | 0.124 | 0.080 | **0.088** | 0.072 | 0.140 | 0.059 | 1.0086 | 0.08 | 0.48 / 0.45 / 0.07 | 0.07 / 0.09 / 0.13 |
| 3 | 34.9 % | 0.018 | 0.101 | **0.157** | 0.143 | 0.189 | 0.132 | 1.0183 | 0.23 | 0.32 / 0.57 / 0.11 | 0.09 / 0.14 / 0.15 |
| 4 | 38.8 % | 0.022 | 0.116 | **0.190** | 0.177 | 0.213 | 0.165 | 1.0266 | 0.32 | 0.24 / 0.63 / 0.13 | 0.08 / 0.17 / 0.14 |
| 5 | 48.5 % | 0.043 | 0.119 | **0.216** | 0.206 | 0.238 | 0.194 | 1.0324 | 0.38 | 0.18 / 0.68 / 0.14 | 0.09 / 0.18 / 0.14 |
| 6 | 55.5 % | 0.074 | 0.115 | **0.232** | 0.220 | 0.245 | 0.211 | 1.0356 | 0.41 | 0.14 / 0.69 / 0.17 | 0.08 / 0.21 / 0.17 |
| 7 | 54.2 % | 0.071 | 0.107 | **0.225** | 0.215 | 0.239 | 0.207 | 1.0382 | 0.42 | 0.12 / 0.71 / 0.18 | 0.07 / 0.20 / 0.15 |
| 8 | 59.7 % | 0.094 | 0.097 | **0.239** | 0.228 | 0.252 | 0.218 | 1.0427 | 0.44 | 0.11 / 0.72 / 0.18 | 0.06 / 0.22 / 0.15 |
| 9 | 63.4 % | 0.136 | 0.077 | **0.242** | 0.229 | 0.258 | 0.219 | 1.0449 | 0.44 | 0.10 / 0.72 / 0.18 | 0.03 / 0.22 / 0.16 |
| 10 | 63.1 % | 0.143 | 0.073 | **0.226** | 0.213 | 0.232 | 0.201 | 1.0461 | 0.43 | 0.09 / 0.72 / 0.19 | 0.03 / 0.21 / 0.15 |
| 11 | 64.5 % | 0.157 | 0.070 | **0.234** | 0.223 | 0.240 | 0.215 | 1.0495 | 0.45 | 0.07 / 0.74 / 0.19 | 0.02 / 0.23 / 0.14 |
| 12 | 67.4 % | 0.209 | 0.069 | **0.230** | 0.220 | 0.231 | 0.209 | 1.0495 | 0.44 | 0.07 / 0.74 / 0.19 | 0.02 / 0.22 / 0.16 |
| 13 | 68.1 % | 0.224 | 0.060 | **0.228** | 0.213 | 0.234 | 0.207 | 1.0517 | 0.45 | 0.06 / 0.74 / 0.19 | 0.01 / 0.22 / 0.15 |
| 14 | 68.1 % | 0.267 | 0.053 | **0.210** | 0.196 | 0.221 | 0.187 | 1.0494 | 0.43 | 0.07 / 0.74 / 0.19 | 0.01 / 0.20 / 0.15 |

## 4. Verdict (prediction vs outcome)
- Prediction: P(weighted R² ≥ 0.3 at l ≥ 9) ≈ 0.10. Outcome: **max 0.242 (ridge13) / 0.258 (MLP ceiling) at l = 9 — does not fire.**
- Kill line (< 0.15 at every l ≥ 8): **not met either** (0.21–0.24 at every l ≥ 8). Middle band; the handoff has no rule for it ⇒ strategist.
- **What the 0.2 is:** a single per-layer scalar. The served D21 is uniformly *too small*: g = 1.018 (l3) → 1.043 (l8) → 1.049–1.052
  (l11–14). Scale alone gives weighted OOS R² 0.19–0.22 and removes ≈ 44 % of the sensitivity-weighted D21 error energy at l ≥ 8; the other 12
  pair features add ≤ 0.03, the MLP ceiling ≤ 0.02 more. So there is **no pair-local law beyond a coherent per-layer gain**; the old "a_l − 1 ≈ 0"
  finding does not hold on 8be (truncation/KAS shrink the slice).
- **The damage is in kink × decided pairs** (72–74 % of the weighted error at l ≥ 8, R² 0.20–0.23 there), not kink × kink (6–11 %, R² ≤ 0.06).
- Actionable form (if the strategist wants it): a per-layer gain on the D21 read after line 3156, `D21 = D21 * _G21[l]`, default 1.0 ⇒
  byte-identical zero control, n² FLOPs/layer (≈ +13 MF, +0.005 %). Measured init: l2 1.0086, l3 1.0183, l4 1.0266, l5 1.0324, l6 1.0356,
  l7 1.0382, l8 1.0427, l9 1.0449, l10 1.0461, l11 1.0495, l12 1.0495, l13 1.0517, l14 1.0494. Its effect on the final layer is **unmeasured**:
  the in-loop corrector and gains were fitted against the shrunk slice, so an unrefit read may understate or reverse it; the honest test is
  as a gain group in a refit (L2), or an unrefit VAL screen once the GPU is free after R3. Not run now: any GPU/WSL job would block R3.

## 5. Unrefit screen of the measured D21 gains (jobs 3601–3603, `screen/`, `screen_val.json`)
`screen/est_8be_g21.py` = 8be + `D21 = D21 * _G21[l]` after the k0 fill (gains of §4); `est_8be_g21_zero.py` (all 1.0) reproduces 8be's VAL900
predictions bit for bit. Same numpy chain as the X1 VAL dumps (control = `cache/x1/val`). VAL 900–928, final layer, paired:
**+34.7 % raw MSE (t95 [+30.5, +38.9] %), 0/29 wins** — stopped there, the sign is not in doubt.
Reading: the chain's corrector, gates and dressing gains were all fitted against the shrunk slice; restoring D21 alone breaks that balance.
The uniform shrink is a compensated feature of the fitted chain, not a free correction. A `_G21` group could only enter a full refit,
started at 1.0, and nothing here suggests it would pay. **Recommendation: close L4 (park `_G21`); do not add it to the second stage.**
