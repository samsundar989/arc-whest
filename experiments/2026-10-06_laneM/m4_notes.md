# M4 groundwork — where neighbor features would plug in (mapped 2026-10-06)

E = estimator_k3_v73_gb.py, T = tools/twin/twin.py.

- Corrector: E:1651-1791 (`_CORR_ON`, `_CORR_NF=9`, `_CORR_PI/_CORR_PJ`, `_CORR_B64` pages:
  fmu/fisd (16,33), W1 (16,33,32), b1, W2 (16,32,32), b2, W3 (16,32,1), b3).
  `_corr_raw` E:3735-3767 builds the 9 raw features (mu_pre, sigma, mu_pre/sigma, D3/σ³, D4/σ⁴,
  ‖W[:,i]‖, Σ_j D21²/σ⁴, mu_out, mu_out − mean(mu_prev)); `_corr_delta` E:3770-3791 expands
  them to 33 (raw + squares + 15 pair products of the first 6), standardises, then MLP
  tanh32-tanh32-1. ADDITIVE correction to mu_out at every layer (page 0 zero), applied in
  `_layer` E:4078-4081. Twin runs the same file via a torch shim with text patches.
- Available at the call site: mu, cov (dense n×n), mu_pre, var/sigma, alpha (clipped),
  cov_pre = WᵀcovW dense (None at the LAST layer under _P4_FINAL_VAR), C11 = off-diag cov_pre
  (not at last layer), D3, D21 (n×n, not last layer), D22, D4. ρ = C11/(σσᵀ) must be formed;
  Φ, φ are locals of `_wick` — recompute via norm.cdf/pdf(alpha) (as the q31 patch, T:208).
- Training: `cmd_train` T:648-781, TRAINABLE stages T:70-99 (S1 = corr only); Adam with
  per-tensor lr ∝ RMS, cosine; loss = final MSE + 0.1 × mean rows 1..14; train panel
  `train` (900) or `train+d8b`; select on hold100 (= VAL 900-999); params_best.npz.
  ~25 s/step, 224 steps ≈ 1.5 h.
- HARD PARTS: 33-wide layout is a contract (build_twin.py:58 rejects shape changes;
  anatomy/corr_ceiling/fit_corr/fit_corr_seq share the order); fmu/fisd frozen and no tool
  computes new feature statistics; twin text patches anchor on the exact `_corr_delta(`
  call text (T:343-346, count==1 at T:407); last layer lacks C11 → needs fallback.
  Zero-init new W1 rows to start exactly at v73 (`--zero-scale` only acts on all-zero tensors).
