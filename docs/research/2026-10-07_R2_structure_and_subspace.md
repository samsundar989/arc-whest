# R2: Structure of deep random ReLU nets at fixed weights, and subspace-plus-bulk estimation

*Prepared 2026-10-06 for the ARC White-Box Estimation Challenge (1024×16, He weights, no biases, x ~ N(0, I)).*

**Method.** I combined a literature and forum search with new numpy measurements on simulated 1024×16 He-ReLU networks:
- **Main network (seed 0).** 2^19 = 524k inputs for spectra and readout oracles, 131k for the cumulant decompositions, 32k for the error-transfer runs.
- **Second network (seed 1).** Used to replicate the gain-mode test.

The simulated network matches our 100 FIT nets at layer 16:

| statistic (layer 16) | simulated | FIT nets |
|---|---|---|
| std(α) | 3.26 | 3.19 |
| dead (α < −3) | 19.8% | 19% |
| decided (\|α\| > 3) | 38.4% | 38% |
| pre-activation skew, rms | 0.105 | 0.103 |
| excess kurtosis, mean | 0.062 | 0.054 |

**Conventions.**
- Unit **u** = one 1024³ matmul = 2.15e9 FLOPs ≈ 0.1% of the budget.
- Every claim is tagged **[lit]** (literature), **[meas]** (my measurement) or **[inf]** (my inference).
- Scripts are listed in the Appendix.

---

## 0. Bottom line

1. **At depth, the top covariance mode is the network's input-dependent gain.** [meas]
   - The top principal direction u₁ of Cov_x(h_l) aligns with the mean vector m_l: |cos| = 0.94 at layer 8 and 0.98 at layer 16.
   - Its coordinate correlates with the representation norm ‖h_l(x)‖ at |r| = 0.99.
   - It carries 9.5% of the variance at layer 16.
   - The norm fluctuates by 8.4% across inputs (coefficient of variation), roughly log-normally.
   - Its quenched variance follows a mean-field law that I checked to within 15%: Var_x log‖h_l‖² ≈ [2 + 5·Σ_{j<l}(1−ρ_j)]/n. This is the fixed-weights counterpart of Hanin–Nica's annealed 5L/n, and at layer 16 it is about 2.9× smaller. The law saturates with depth.

2. **New: a zero-parameter "super-sample" (gain-mode) law explains most of the non-Gaussianity at depth.** [meas]
   - Model each pre-activation as z = g·ẑ: a scalar gain g with Var g = c² (c² measured directly as Var‖h‖/(E‖h‖)²) times a Gaussian ẑ. Positive homogeneity of a bias-free ReLU net makes this natural.
   - The model predicts:
     - D3_i = 6c²μ_iσ_i²
     - κ4_i = 12c²σ_i⁴
     - D21_ij = 2c²(μ_jσ_i² + 2μ_iC_ij)
     - K22_ij = 4c²(σ_i²σ_j² + 2C_ij²)
   - With no fitting at layer 16, these explain:
     - D3: 52–60% (weighted by each neuron's effect on its mean); 77–80% unweighted;
     - κ4 (weighted): 57–66%;
     - D21: 68–71%;
     - K22: 63–77%.
   - With one fitted scalar per slice (β ≈ 0.7), the unweighted D3, κ4, D21 and K22 fits reach 83–92%. The Edgeworth-weighted D3 fit reaches only 55–62%.
   - This plausibly explains why 504aldo's κ4 regeneration G_off = λ·C_off works [inf]:
     - the gain mode makes κ4 ⊇ 4βc²·Sym(C⊗C), whose r = 1 core is ∝ C_off;
     - its predicted depth profile λ_l ∝ β_l c_l² σ̄_l² matches the ratio in their fitted table (layers 4–5 vs 15: 1.6–1.8 predicted vs 1.57 fitted);
     - it explains why AndreasHad04's regression of the (2,2) slice on C collapses with depth: the right template is σ_i²σ_j² + 2C_ij², not C_ij.

3. **The effective dimension shrinks roughly linearly in depth.** [meas]
   - The participation ratio (PR) of Cov_x(h_l) falls 664 → 424 → 247 → 127 → 70 → 49 at layers 1, 2, 4, 8, 12, 16. Fit: n/PR − 1 ≈ 0.6·l + 0.04·l².
   - Directions needed:
     - for 90% of the variance: 708 (layer 1), 264 (layer 8), 148 (layer 16);
     - for 99%: 985 (layer 1), 674 (layer 8), 511 (layer 16).
   - So the bulk never becomes low-rank, but the head becomes sharp:
     - layer 16 needs only 21 directions for 50% and 75 for 80%;
     - the median eigenvalue is 0.09× the mean.
   - The per-input Jacobian's free-probability law, PR = n/(1+2L) [lit], has the same scaling.

4. **Where the mean-relevant non-Gaussianity lives.** [meas]
   - Write z_i = m_i + b_iᵀa + e_i, with a the top-k principal-component coordinates of h_{l−1}.
   - At layer 16, the latent's own third cumulant plus its first-order coupling to the bulk explains this share of the per-neuron skew energy that moves the means:
     - k = 1 (the gain mode alone): 43%;
     - k = 4: 66%;
     - k = 16: 91%;
     - k = 64: 100%.
   - The latent's own κ3 alone explains only 40% at k = 16.
   - The pure-bulk κ3 is ≤ 9% at k ≥ 16.
   - D21 is 93% rank-16 as a matrix, and 65% (k = 16) / 88% (k = 64) of its off-diagonal is the latent's own D21.
   - At layer 4 everything is more spread out: 59% (k = 16) and 88% (k = 64).

5. **Subspace readouts do not beat Edgeworth; subspaces matter only for propagation.** [meas]
   - With exact per-neuron cumulants, the Edgeworth readout to κ4 is already essentially exact at layer 16: bias² 7.6e-11. Stopping at κ3 gives 2.0e-8; a Gaussian readout gives 1.6e-7.
   - Rao–Blackwellized (RB) / conditional-Gaussian readouts on the top-k latent:
     - with no bulk response: 5.1e-8 at k = 64, worse than useless;
     - with a first-order response model: 1.2e-8 at k = 64;
     - adding bulk Edgeworth: 0.4e-9 at k = 4, 3.1e-9 at k = 16, 7.4e-9 at k = 64. Accuracy gets *worse* as k grows, because coupling a large latent to the bulk is hard to model.
   - So build subspace machinery only to get **κ3 and κ4** right, never to read out the mean.

6. **Early-layer exactness pays only through coherent errors.** [meas]
   - Measured transfer of a mean error injected at layer l₀ to the layer-16 means, as an energy gain (it matches the mean-field product Π_{l>l₀} 2E[Φ(α_l)²] within about 20%):

     | l₀ | random direction | along the mean/gain direction |
     |---|---|---|
     | 1 | 0.022 | 0.27 |
     | 2 | 0.035 | 0.45 |
     | 4 | 0.11 | 0.64 |
     | 8 | 0.30 | 0.98 |
     | 12 | 0.57 | 0.82 |

   - A coherent relative covariance error η at layer l₀ costs 0.13η², 0.076η², 0.040η², 0.0096η² or 0.0044η² of final MSE for l₀ = 1, 2, 4, 8, 12.
   - So:
     - generic early errors are damped 9–45×;
     - coherent ones (overall scale, variance level) survive;
     - layers 2–4 need coherent relative errors ≤ ~1e-4 to stay under 1e-9 of final MSE.
   - Exact κ3 at layers 2–4 is worth building only if our chain's *coherent* early errors exceed that; check this before building anything.

7. **News since Oct 1.**
   - The organizers ruled that building on another participant's MIT code is allowed with credit, licence compliance and "substantial changes of your own" (Oct 5).
   - The submission-selection UI opens one week before the deadline, after a flopscope accounting fix and re-evaluation (Oct 4).
   - No top team has published a method.
   - Today's leaderboard: J2W 1.5e-9 (raw 1.47e-8, utilisation 0.105); marius_binner 1.6e-9 (raw 1.14e-8); Luna 1.6e-9; mliston 1.7e-9; and a cluster of nine teams at raw 1.37–1.57e-8.

---

## 1. Geometry of a fixed deep random ReLU net

### 1.1 The mean-field skeleton [lit + meas]

**Correlation map.** For two independent inputs, the post-activation correlation follows the ReLU arc-cosine map:

ρ_{l+1} = (√(1−ρ_l²) + (π − arccos ρ_l)·ρ_l)/π, with ρ_0 = 0.

- It gives 0.318, 0.494, 0.681 (l = 4), 0.834 (l = 8), 0.897 (l = 12), 0.930 (l = 16).
- Expanding near ρ = 1 gives ε' = ε − (2√2/(3π))·ε^{3/2}, so 1−ρ_l → 9π²/(2l²). The convergence is polynomial, not exponential (Hayou–Doucet–Rousseau, [arXiv:1902.06853](https://arxiv.org/abs/1902.06853); Poole et al., [arXiv:1606.05340](https://arxiv.org/abs/1606.05340); Schoenholz et al., [arXiv:1611.01232](https://arxiv.org/abs/1611.01232)).
- At l ≤ 16 the asymptote is poor because of a large offset; 1−ρ_l ≈ 44.4/(l+9)² fits l ≥ 8 to within 10%.
- So 93% of each neuron's second moment at layer 16 is input-independent: the representation is "a mean function plus a 7% fluctuation". This is the finite-depth version of the degeneracy that Li–Nica–Roy prove for unshaped nets: the network becomes "a constant function sending all inputs to the same output" ([arXiv:2206.02768](https://arxiv.org/abs/2206.02768)).

**Decidedness.**
- Per-neuron α_i = μ_i/σ_i has std(α_l) ≈ √(ρ_{l−1}/(1−ρ_{l−1})).
- Predicted: 3.46 at layer 16 (measured 3.26; FIT nets 3.19). Predicted dead fraction Φ(−3/s): 19.3% (measured 19.8%). Decided fraction 2Φ(−3/s): 38.6% predicted vs 38.4% measured. The Gaussian-α formula slightly overstates s, and heavier-than-Gaussian α tails make up for it.
- This is a deterministic, weight-independent profile. Lane P can use it to plan pruning.

### 1.2 Spectrum and effective dimension of Cov_x(h_l) for one network [meas]

| l | std α | decided | PR | k50 | k80 | k90 | k99 | λ₁/tr | \|cos(u₁, m̂)\| | gain CV | degree-1 share (PR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 0 | 664 | 239 | 527 | 708 | 985 | 0.003 | 0.04 | 0.041 | 0.74 (510) |
| 2 | 0.69 | 0 | 424 | 147 | 377 | 560 | 913 | 0.006 | 0.10 | 0.052 | 0.60 (284) |
| 4 | 1.16 | 1.2% | 247 | 84 | 246 | 413 | 816 | 0.012 | 0.60 | 0.062 | 0.45 (142) |
| 8 | 1.95 | 12% | 127 | 45 | 145 | 264 | 674 | 0.032 | 0.94 | 0.076 | 0.30 (65) |
| 12 | 2.89 | 32% | 70 | 28 | 95 | 179 | 545 | 0.070 | 0.97 | 0.081 | 0.25 (34) |
| 16 | 3.26 | 38% | 49 | 21 | 75 | 148 | 511 | 0.095 | 0.98 | 0.084 | 0.21 (21) |

**Shape of the layer-16 spectrum.**
- λ₁ = 9.5% of the trace (the gain mode). λ₂…λ₁₀ fall smoothly from 3.6% to 2.0%.
- The tail is heavy: the median eigenvalue is 0.09× the mean.
- Pre-activation spectra (W·C·Wᵀ) are almost identical: PR 51, k90 = 131, k99 = 415 at layer 16.
- The "degree-1 share" is the Stein/Hermite linear part J̄J̄ᵀ, with J̄ = E[∂h/∂x]. It carries 21% of the variance at layer 16 but is very low-rank (PR 21). The other 79% sits in higher Wiener chaoses, which are spread far more evenly.

**Theory that predicts the PR curve.**
- **Per-input Jacobian** J = Π D_l W_l [lit + derivation]. Free multiplicative convolution makes the normalized variances add, so the squared singular values of J have m₂ = 1 + Σ_l(v_W + v_D):
  - Gaussian W at ratio 1: v_W = 1;
  - ReLU Bernoulli(½) gate: v_D = 1;
  - so m₂ = 1 + 2L and PR = n/(1 + 2L) = 31 at L = 16.
  - Sources: Pennington–Schoenholz–Ganguli ([arXiv:1711.04735](https://arxiv.org/abs/1711.04735), [arXiv:1802.09979](https://arxiv.org/abs/1802.09979)).
- **x-averaged propagator.** The gate is Φ(α), whose spread grows with depth (v_D goes from 0 to about 0.7). The covariance is a superposition of innovations of different ages transported by these propagators. That is why n/PR grows a little faster than linearly: 0.6l + 0.04l².
- **The C review's age law** for old-source propagators, PR ≈ n/(2·age), is this same law.
- **Edge of the spectrum.** The largest squared singular value grows only polynomially, roughly as e·(L+1) (Fuss–Catalan edge [lit]).
- **Opposite regime.** Exponential Lyapunov separation, the "triangle law" h(λ) = 2λ, and rank-one collapse apply when L ≳ n (Hanin–Nica [arXiv:1812.05994](https://arxiv.org/abs/1812.05994); Daneshmand et al. [arXiv:2003.01652](https://arxiv.org/abs/2003.01652); Feng et al., [Rank diminishing, NeurIPS 2022](https://neurips.cc/virtual/2022/poster/55444)). At L/n = 1/64 we are firmly in the polynomial regime.

**Predictions for n = 1024, L = 16** (formulas confirmed by the table above):
- PR_l ≈ n/(1 + 0.6l + 0.04l²);
- k90 ≈ 150 and k99 ≈ 500 at the last layer;
- the effective coupling 1/PR ≈ 0.02 at layer 16 vs 0.0015 at layer 1. This is why ARC's power counting runs in ℓ/n.

### 1.3 The gain (mean) mode [meas + lit]

**What it is.**
- At every layer ≥ 6, u₁ ≈ m̂ (|cos| 0.89–0.98), and u₁'s coordinate is the norm: corr(‖h_{l−1}‖, a₁) = 0.93 at layer 8 and 0.99 at layer 16.
- The rank-1 share predicted by the norm fluctuation, CV²·‖m‖²/tr C, matches the variance along m̂: 9.1% vs 9.2% at layer 16 (2.7% vs 3.0% at layer 8).
- So the dominant direction of variation is the network's **input-dependent overall gain** g(x) = ‖h_l(x)‖/√n multiplying a near-constant direction. A bias-free ReLU net is positively homogeneous: h(αx) = αh(x).

**Statistics.**
- CV(g) = 4.1% at layer 1 (the χ-distributed input radius plus layer-1 gating), 6.2% at layer 4, 7.6% at layer 8, 8.4% at layer 16.
- log g is nearly symmetric (skew −0.05), so g is close to log-normal, as Hanin–Nica predict for the annealed problem: (n₀/n_d)‖Ju‖² ≈ exp N(−β/2, β), β = 5Σ1/n_j ([arXiv:1812.05994](https://arxiv.org/abs/1812.05994); "the future is log-Gaussian", Li–Nica–Roy [arXiv:2106.04013](https://arxiv.org/abs/2106.04013)).
- The latent coordinate has skew +0.20 and excess kurtosis 0.05–0.075 at depth. Principal components 2–16 have skew rms of about 0.04; components 17–64 about 0.03.

**Quenched law (new, checked).** For a fixed network, randomness comes only from x.
- Each layer's annealed increment 5/n is reduced by the fraction of fresh variability (1−ρ_j):

  Var_x log‖h_l‖² ≈ [2 + 5·Σ_{j<l}(1−ρ_j)]/n.

- Predicted vs measured:

  | layer | predicted | measured | annealed (2+5l)/n |
  |---|---|---|---|
  | 1 | 0.0068 | 0.0068 | 0.0068 |
  | 4 | 0.0146 | 0.0155 | 0.0215 |
  | 8 | 0.0194 | 0.0228 | 0.041 |
  | 16 | 0.0239 | 0.0280 | 0.080 |

- The quenched gain variance **saturates**: c² = CV² goes 0.0039 (layer 4) → 0.0057 (layer 8) → 0.0070 (layer 16).

### 1.4 Where the per-neuron non-Gaussianity lives [meas]

**Decomposition.** z_i − μ_i = s_i + e_i, with s_i = b_iᵀa the projection on the top-k principal components of h_{l−1} and e_i the bulk. Exactly:

κ3(z_i) = T1 + T2 + T3 + T4, with T1 = E[s³], T2 = 3E[s²e], T3 = 3E[se²], T4 = E[e³].

- T2 is the bulk mean responding quadratically to the latent.
- T3 is heteroscedasticity: the bulk variance responding linearly to the latent.

**Metric.** "R²w" is the fraction of Σ_i(w3_i·κ3_i)² explained, where w3_i = −α_iφ(α_i)/(6σ_i²) is the Edgeworth weight that turns κ3 into a mean correction. In other words, it is the skew energy that moves the means.

| layer | k | latent share of variance | R²w(T1) | R²w(T1+T2+T3) | D21 off-diagonal R² from the latent's own D21 |
|---|---|---|---|---|---|
| 4 | 16 | 0.12 | 0.14 | 0.59 | 0.10 |
| 4 | 64 | 0.36 | 0.40 | 0.88 | 0.31 |
| 8 | 16 | 0.23 | 0.29 | 0.82 | 0.35 |
| 8 | 64 | 0.56 | 0.65 | 0.98 | 0.68 |
| 16 | 1 (gain) | 0.08 | 0.02 | **0.43** | – |
| 16 | 4 | 0.17 | 0.14 | 0.66 | – |
| 16 | 16 | 0.41 | 0.40 | **0.91** | 0.65 |
| 16 | 64 | 0.75 | 0.72 | **1.00** | 0.88 |
| 16 | 256 | 0.95 | 0.93 | 1.00 | – |

What the table shows:
- For k = 1, the dominant term is **T3**: rms 0.82 of κ3's rms at layer 16. The gain mode skews neurons by modulating their bulk variance.
- As a matrix, D21 has 93% (layer 16), 83% (layer 8) and 59% (layer 4) of its energy in its top-16 singular vectors.
- Per-neuron κ4 is only 37% the latent's own κ4 even at k = 64. The rest comes from mixed (s, s, e, e) terms, the variance-of-variance, which §1.5 explains.
- **Reading for us** [inf]:
  - at depth, the D3/D21 we must get right are generated by about 16 latent directions *and their coupling to the bulk*;
  - this is consistent with our error sitting 81–92% in a 64-dimensional subspace aligned with the top covariance modes;
  - early layers (≤ 4) are genuinely high-rank, which matches the measured failure of compact κ3 cores.

### 1.5 The super-sample (gain-mode) law [meas, 2 networks]

**Model.** z = g·ẑ, with g independent of ẑ, E g = 1, Var g = c², and ẑ ~ N(μ, C). By the law of total cumulance, to O(c²):
- κ3 = 2c²·Sym(μ⊗C) (all orders in μ);
- κ4 = Var(g²)·Sym₃(C⊗C) ≈ 4c²·Sym₃(C⊗C).

The slices are the formulas in §0.2, plus Q31_ij = 12c²σ_i²C_ij.

c² is **measured, not fitted**: it is Var‖h_{l−1}‖/(E‖h_{l−1}‖)². It also equals m̂ᵀCm̂/‖m‖² to within 3% at depth (0.0070 vs 0.0072), so **a cumulant chain can compute it from its own mean and covariance**.

R² across neurons or off-diagonal pairs, given as β = 1 / best scalar fit (fitted β):

| slice | layer 8 (seed 0) | layer 16 (seed 0) | layer 16 (seed 1) |
|---|---|---|---|
| D3, Edgeworth-weighted | 0.64 / 0.68 (0.80) | 0.52 / 0.55 (0.80) | 0.60 / 0.62 (0.85) |
| D3, unweighted | 0.78 / 0.86 (0.77) | 0.80 / 0.92 (0.73) | 0.77 / 0.92 (0.72) |
| κ4 per neuron, weighted | 0.46 / 0.85 (0.60) | 0.66 / 0.88 (0.67) | 0.57 / 0.86 (0.63) |
| D21 off-diagonal | 0.65 / 0.72 (0.76) | 0.71 / 0.84 (0.72) | 0.68 / 0.83 (0.70) |
| K22 off-diagonal | 0.47 / 0.81 (0.61) | 0.77 / 0.86 (0.76) | 0.63 / 0.84 (0.67) |
| Q31 off-diagonal | 0.02 / 0.11 (0.52) | 0.39 / 0.42 (0.79) | 0.24 / 0.34 (0.65) |

Further measurements:
- Mean excess kurtosis vs 12c²: 0.039 vs 0.064 at layer 8; 0.062 vs 0.082 at layer 16. So per-neuron kurtosis at depth is about 75% of a pure scale mixture.
- At layers 2–4 the law explains less (D21: 0.16–0.45; κ4 needs β ≈ 0.5).
- For K22 I compared the product template σ_i²σ_j² + 2C_ij² with the additive r = 1 form (κ4_ii + κ4_jj) used by the harmonic core, where the additive form is fed the *exact* diagonal κ4:
  - product: R² 0.59 / 0.81 / 0.86 / 0.86 at layers 4 / 8 / 12 / 16;
  - additive: 0.49 / 0.76 / 0.84 / 0.83.
  - The product form is better and needs no κ4 diagonal.
- For Q31: λ·C_ij gives 0.38 and σ_i²C_ij gives 0.42 at layer 16. Q31 has substantial non-gain physics, consistent with the E review's finding that Q31's R² is only 0.46–0.50.

**Mechanistic reading** [inf]:
- The κ4 the chain needs at depth is mostly the "super-sample covariance of the covariance". This is exactly Takada–Hu's SSC term C_SSC = σ_b²(∂P/∂δ_b)(∂P/∂δ_b) ([PRD 87, 123504](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevD.87.123504/fulltext); separate-universe calibration, Li–Hu–Takada [arXiv:1401.0385](https://arxiv.org/abs/1401.0385)), with:
  - the "background mode" = the gain;
  - the response ∂C/∂δ_b = 2C supplied exactly by homogeneity.
- Puffi's Phase-1 SSC closure with mode B₁ = C is this term, with Λ₁₁ = Var(g²) ≈ 4βc².
- In the K = 3 AUGMENT r = 1 core, the off-diagonal is G_off ≈ 6Λσ̄²·C_off (under my reading of the normalization), hence λ_l ∝ β_l c_l² σ̄_l².
  - Predicted λ(4–5)/λ(15) ratio: 1.6–1.8. 504aldo's fitted ratio: 1.1e-2/7e-3 = 1.57.
  - The absolute constant comes out about 2.7× theirs, which is probably a normalization convention.
- AndreasHad04's collapse of "(2,2) on C_off" with depth (R² 0.998 → 0.045) is predicted: the gain term σ_i²σ_j² is not a function of C_ij.

### 1.6 Predictions for 1024×16 (summary)

| quantity | value |
|---|---|
| PR(h_l) | ≈ n/(1 + 0.6l + 0.04l²): 49 at layer 16 |
| k50 / k90 / k99 at layer 16 | 21 / 148 / 511 |
| top mode | the gain, ≈ m̂, 9.5% of the variance |
| gain c² | ≈ [2 + 5Σ(1−ρ_j)]/(4n) ≈ 0.007, saturating |
| decided neurons | ≈ 2Φ(−3/√(ρ_{l−1}/(1−ρ_{l−1}))), 38.6% predicted vs 38.4% measured at layer 16 |
| mean-relevant κ3 explained (k = 16) | 91% (layer 16), 82% (layer 8), 59% (layer 4) |
| mean-relevant κ3 explained (k = 64) | 100%, 98%, 88% |
| κ3/κ4/D21/K22 from the zero-parameter gain law at depth | 55–80% |

---

## 2. Low-dimensional subspace + near-Gaussian bulk: methods

### 2.1 The setting and the exact identity

Write z_i = m_i + b_iᵀa + e_i, with a ∈ ℝ^k the latent (non-Gaussian) and e_i | a ≈ N(γ_i(a), v_i(a)), the **responsive bulk**. Then exactly:

E[relu z_i] = E_a[G(m_i + b_iᵀa + γ_i(a), v_i(a))], where G(μ, v) = √v·φ(μ/√v) + μ·Φ(μ/√v).

Two consequences:
- **Single index.** Each neuron sees the latent only through the scalar s_i = b_iᵀa. Its cumulants are κ_r(s_i) = κ_r(a)[b_i^{⊗r}].
- **Smoothing.** G is smoothed by the bulk (scale √v_i). That is why moment-matched point sets converge fast where per-neuron Edgeworth around the kink does not.

§1.4 shows that γ and v (the T2 and T3 terms) are *not* optional.

### 2.2 The method families, mapped onto this setting

1. **Rao–Blackwellized point sets / conditional-Gaussian (mixture Kalman).**
   - Formula: Σ_s w_s·G(m_i + b_iᵀa^{(s)} + γ_i, v_i(a^{(s)})).
   - Cost per layer: n·S·(k + 25) FLOPs, about 0.01 u even at S ≈ 500.
   - Accuracy: limited by how well γ and v are modelled (§2.3).
   - Sources: Doucet et al. [arXiv:1301.3853](https://arxiv.org/abs/1301.3853).
2. **Separate universe / SSC response.**
   - Formula: expand in the background mode, E ≈ G(m_i, V_i) + response × latent cumulants. For the gain mode the response is exact by homogeneity: ∂μ/∂δ = μ, ∂C/∂δ = 2C.
   - Cost: O(n²) per layer for all slices, about 0.005 u.
   - Accuracy at depth: §1.5 (55–80% zero-parameter, 83–92% with β).
   - Separate-universe simulations calibrate responses to "a few percent or better" in cosmology ([arXiv:1401.0385](https://arxiv.org/abs/1401.0385)).
3. **Gaussian sum with shared covariance (blended particle filter).**
   - Representation: latent ~ Σ_c π_c N(a_c, Q). The readout is closed-form per component: G(m_i + b_iᵀa_c, v_i + b_iᵀQb_i).
   - Propagation: centres move deterministically, a'_c = U'ᵀ(Ḡ_c − m'); fresh within-component variance goes into Q'.
   - Cost: 2nkC + 2n²k ≈ 0.02–0.1 u per layer.
   - Sources: Majda–Qi–Sapsis ([PNAS 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4040593)). They factor p(u) = p(u₁)·p^G(u₂|u₁), with particle-dependent conditional means and a shared covariance R₂, using a 5-dimensional adaptive subspace of the 40-dimensional Lorenz-96 system. GMM-DO (Sondergaard–Lermusiaux, [MIT DSpace](https://dspace.mit.edu/handle/1721.1/78912)) is the ocean data-assimilation version.
   - Splitting along the most nonlinear direction while preserving mean and covariance: Kulik–LeGrand ([arXiv:2412.00343](https://arxiv.org/abs/2412.00343)).
4. **Sigma points in the subspace.**
   - Variants:
     - UT: 2k+1 points, exact for degree-3 polynomials (Julier–Uhlmann);
     - higher-order UT: matches the skewness and kurtosis tensors via rank-1 tensor decompositions and is exact for quartics (Easley–Berry, [arXiv:2006.13429](https://arxiv.org/abs/2006.13429));
     - conjugate UT: degree 5–7 with O(k²) points.
   - Readout as in method 1.
   - Matching moments up to order p in k dimensions matches every 1-D projection to order p, so the per-neuron error is a p-th-order Edgeworth error of the *smoothed* G.
5. **Sparse grids / cubature in k dimensions.**
   - Smolyak degree 5 needs 2k²+2k+1 points (545 at k = 16; 8,321 at k = 64). Degree 7 needs about (4/3)k³ points.
   - Readout cost: 2nkS, which is 0.02 u at k = 16 and 0.5 u at k = 64 (degree 5).
   - Weights must be Edgeworth-reweighted for a non-Gaussian latent. Only worth it if a distribution, not just moments, must be integrated.
6. **Multivariate Edgeworth in the subspace.**
   - Carry κ3(a) (k³ entries) and κ4(a) (k⁴); contract per neuron.
   - Cost: n·k⁴ = 0.03 u at k = 16, but 8 u at k = 64 unless κ4(a) is CP-factored (then n·k·r).
   - §1.4 shows the latent's *own* κ4 is a minority of per-neuron κ4. The mixed and gain terms dominate.
7. **Blended reduced-subspace UQ (QG-DO / MQG-DO)** (Sapsis–Majda, [Physica D](https://math.nyu.edu/faculty/majda/pdfFiles/Blended%20reduced%20subspace%20algorithms%20for%20uncertainty%20quantification%20of%20quadratic%20systems%20with%20a%20stable%20mean%20state.pdf)).
   - Mean + s non-Gaussian dynamically-orthogonal modes + a Gaussian complement whose covariance equation receives third-order subspace statistics as a flux. Even s = 1–2 modes fixed the energy-cascade regimes there.
   - For us: our covariance/κ3 chain is the "complement", and the latent's κ3 is the flux term.
   - This is the closest published analogue of "low-rank refinement".
8. **ARC's "low-rank refinement"** (townhall: "spending more compute on higher-variance directions").
   - Carry κ4/κ5 only in the top-k subspace.
   - Transport costs k^{p+1}, which is negligible. Births need n²k projections, about 0.03 u each.
   - Nothing new has been published since the townhall: the ARC blog index lists no posts after June 9, 2026.

### 2.3 Oracle readout accuracy vs k [meas]

Bias² of each estimator against the true mean, from split halves of 262k inputs each. The latent distribution is exact (the empirical s_i); responses are fitted in-sample. This is the best any subspace readout could do.

| estimator | layer 8 | layer 12 | layer 16 |
|---|---|---|---|
| Gaussian (exact μ, σ²) | 2.9e-7 | 2.1e-7 | 1.6e-7 |
| Edgeworth κ3 (exact) | 3.2e-8 | 2.4e-8 | 2.0e-8 |
| **Edgeworth κ3 + κ4 (exact)** | **6e-11** | **≈ 0** | **7.6e-11** |
| RB, k = 64, constant bulk | 9.7e-8 | 6.7e-8 | 5.1e-8 |
| RB, k = 64 + response (γ, ρ) | 1.9e-8 | 1.4e-8 | 1.2e-8 |
| RB, k = 1 / 4 / 16 / 64 + response + bulk Edgeworth | ≈0 / ≈0 / ≈0 / 3.6e-9 | ≈0 / ≈0 / 1.7e-9 / 6.6e-9 | 0.1e-9 / 0.4e-9 / 3.1e-9 / 7.4e-9 |
| RB, k = 256 + response | 1.9e-9 | 1.6e-9 | 1.2e-9 |

The noise floor is about ±4e-9 at layer 8 (the negative entries are noise) and about 1e-9 or less at layer 16.

**Conclusions.**
- (a) Given correct per-neuron κ3 and κ4, the readout is solved. κ4 is worth 2e-8 at layer 16: our whole error budget.
- (b) Any subspace or conditional-Gaussian scheme must model the bulk's response to the latent, and gets *harder* as k grows.
- (c) So the latent's value is in **propagation**: computing D3, D21 and κ4 slices. That is where our error is [inf].

### 2.4 A design that fits these facts

Call it **"SSC + latent correction"**, layered on the existing K3 chain.

**Step (i): gain-mode SSC terms** (≈ 0 FLOPs).
- Compute c_l² = m̂ᵀC m̂/‖m‖² from the chain's post-activation state (one matrix-vector product).
- Use κ4 = 4β_l c_l²·Sym₃(C⊗C) for all consumed κ4 slices:
  - D4_i = 12βc²σ_i⁴;
  - K22_ij = 4βc²(σ_i²σ_j² + 2C_ij²);
  - Q31_ij = 12βc²σ_i²C_ij.
- Do this in place of, or alongside, the λ-table and its additive r = 1 (2,2) slice.
- Optionally audit the chain's D3/D21 against 2βc²Sym(μ⊗C).
- About 16 β values, or a smooth β(l) running from about 0.5 to 0.75; these are the only constants.

**Step (ii): k = 16 latent correction** (≈ 0.5–1 u per layer).
1. Get top-16 eigenvectors by warm-started subspace iteration (0.07 u).
2. Project the chain's factored κ3 legs onto U to get the latent core (about 0.05 u per lineage).
3. Propagate a Gaussian-sum latent (M ≈ 33–65 centres from HOUT/UT) through the exact single-index map with responses.
4. Correct D3/D21 by (latent-exact − latent-chain). This is a deterministic two-level control variate, so nothing is double-counted.

**Step (iii):** if Lane M finds the dropped κ3×edge family matters, evaluate it with κ3 and C restricted to rank k = 16. With C ≈ UΛUᵀ, the next-layer D3 term needs the k² vectors W(ŵ₂∘u_q∘u_s), about 0.25 u per layer.

---

## 3. Early-layer exactness

### 3.1 Exact tools [lit]

**Layer 1.** z₁ = W₁x is exactly zero-mean Gaussian. So:
- E[h₁] and Cov(h₁) are closed-form arc-cosine kernels (Cho–Saul, [NeurIPS 2009](https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning));
- the layer-2 pre-activation mean and covariance are exact.

**Bivariate quantities.** Any bivariate quantity with nonzero means reduces to a 1-D integral by Price's theorem plus Plackett's identity, ∂E[f(z₁)g(z₂)]/∂C₁₂ = E[f'g']. This gives:
- E[relu z_a·relu z_b] = E relu_a·E relu_b + σ_aσ_b∫₀^ρ Φ₂(α_a, α_b; t)dt;
- the same trick gives D21(h₁)_ab = κ3(h_a, h_a, h_b) for all pairs at O(n²·quadrature) cost. Used in the forum by venture_agent_ai (#18214) and zacharymartinot (#18188).
- Kuang & Lin ([arXiv:2601.22307](https://arxiv.org/abs/2601.22307)) give exact Gaussian moment matching with error bounds. Kan & Robotti give truncated multivariate-normal recursions ([JCGS 2017](https://www.tandfonline.com/doi/full/10.1080/10618600.2017.1322092)).

**Trivariate quantities.** The all-distinct κ3(h₁)_abc has no useful closed form beyond the zero-mean orthant identity (1/8 + Σ arcsin ρ/(4π)). Its diagram series in ρ_ab ~ O(n^{-1/2}) converges fast at layer 1:
- path ∝ ρ²;
- triangle and double-edge path ∝ ρ³.

So "exact κ3 at layer 1" means adding the ρ³ diagrams. The C review measured the triangle at 2–5% of D3/D21 at layers 5–9.

### 3.2 How early errors propagate [meas + lit]

**Mean errors.**
- For a mean error δ at layer l₀, the final error is E_x[J_{l₀→16}]·δ. I measured this by forward-mode averaging over 32k inputs.
- For generic directions the energy gain is ≈ Π_{l>l₀} 2E[Φ(α_l)²], with E[Φ(α)²] = ¼ + arcsin(s²/(1+s²))/(2π) for α ~ N(0, s²):
  - predicted 0.025 / 0.27 / 0.55 vs measured 0.022 / 0.30 / 0.57 from l₀ = 1 / 8 / 12.
- **Zero modes.** The gain/mean direction keeps gain 0.27–0.98 (§0.6). By homogeneity, a pure rescaling persists exactly.
- The other top principal components (u₂, u₈) are not amplified: they get random-direction gains. Top-subspace errors therefore matter because errors are *created* there, not because the network amplifies them.

**Covariance errors** (coherent relative error η added as independent Gaussian noise, antithetic, 32k inputs). Final MSE per η²:

| l₀ | full covariance | variance only | bulk only (outside top 64) | uniform mean shift per η |
|---|---|---|---|---|
| 1 | 0.128 | 0.128 | 0.089 | 0.21 |
| 2 | 0.076 | 0.078 | 0.040 | 0.16 |
| 4 | 0.040 | 0.040 | 0.013 | 0.115 |
| 8 | 0.0096 | 0.010 | 0.0016 | 0.047 |
| 12 | 0.0044 | 0.0051 | 0.0004 | 0.031 |

**Budgets for ≤ 1e-9 of final MSE:**
- coherent covariance error ≤ 1.1e-4 at layer 2, ≤ 1.6e-4 at layer 4, ≤ 3e-4 at layer 8;
- coherent mean-only error along m̂: ≤ 5e-5 to 7e-5 relative at layers 2–4 (there E_i[m²] = 0.49–0.65 and the gain is 0.45–0.64);
- a full rescaling of the layer state (mean and covariance together): ≤ 3e-5. By homogeneity it passes one-to-one to the final means, and E_i[m_16²] ≈ 1.0.

This agrees with forum lore:
- marius (Phase 1): a 0.195% layer-2 variance error raised final MSE 4.2×;
- jamesrahenry: "coherent bias amplifies ~16:1";
- trim_qewas (Phase 1): "answer born early, error made late".

It also matches Boyen–Koller's result that stochastic transitions contract approximation errors ([arXiv:1301.7362](https://arxiv.org/abs/1301.7362)), except along the zero modes.

### 3.3 Verdict

- Our per-layer error profile is 2.3e-9 at layer 5, including about 0.7e-9 of label noise. With gain ≤ 0.11 from layer 4, generic early errors contribute ≤ ~2e-10 (about 1%) at layer 16.
- **Exact κ3 at layers 2–4 pays only if our chain has *coherent* early errors** (scale or variance level) of ≳ 1e-4 relative. These are cheap to measure:
  - project the layer-l₀ mean error on m̂;
  - compute the relative error of tr C and of mean(σ²) at layers 2–5 against labels;
  - multiply by the table above.
- If they are below budget, spend nothing on early layers.

---

## 4. News since Oct 1 (forum, GitHub, web)

**Forum.**
- **#18240 (Oct 5), organizer ruling (mohanty).**
  - You may base your entry on another participant's public code if its licence permits it, you credit it clearly, you conform to the licence, and you "add substantial changes of your own".
  - Minor modifications may be excluded from prizes at the organizers' discretion.
  - Keeping the MIT notice in the shipped files plus disclosing in the write-up what you started from and changed is sufficient attribution.
  - The same principles apply to the write-up prize.
  - *Relevant to our 504aldo-derived mechanisms. Our PROVENANCE file should be write-up ready.*
- **#18118 (Oct 4), organizers.**
  - The submission-selection UI activates one week before the Phase 2 deadline, with forum and email notice.
  - First they will fix flopscope discrepancies, release a new version and re-evaluate affected submissions. This responds to AndreasHad04's questions about flopscope issues #264–#278.
- **flopscope issues (AndreasHad04 and others, from Oct 2).** Issues #263–#284 report symmetry-tag under-billing (random.shuffle, cumsum axes, einsum(symmetry=…)) and #275: ufuncs or batched matmul whose `out=`/`where=` is larger than the inputs are billed at input size (33–513× under-billed).
  - *Action:* check that none of our `out=` writes broadcast. Expect leaderboard multipliers to move for anyone relying on these.
- **#18238 (Oct 4–5).** Questions on how the hidden-set run invokes `whest run`, whether the 8 GB limit is RLIMIT_AS or RSS, and whether one process persists across MLPs. No answer yet. Our memory margin should hold under either reading.
- **#18218 (Oct 4, ngmitam).** Strassen leaf 32→16→8 gave −3.5% on a 504aldo-derived chain (5.23e-9 at 0.245 B), with peak memory 5.08 GB.
- **#18237.** The overview page states the leader is "×722" more efficient than plain MC; the leaderboard says ×707. Unanswered.

**Leaderboard (Oct 6).**

| rank | team | score | raw | utilisation |
|---|---|---|---|---|
| 1 | J2W | 1.5e-9 | 1.47e-8 | 0.105 |
| 2= | marius_binner | 1.6e-9 | 1.14e-8 | 0.144 |
| 2= | Luna | 1.6e-9 | 1.40e-8 | 0.118 |
| 4 | mliston | 1.7e-9 | 1.46e-8 | 0.117 |
| 5= | suliman_tadros | 1.9e-9 | 1.57e-8 | 0.123 |
| 5= | mk1981 | 1.9e-9 | 1.82e-8 | 0.106 |
| 5= | oliver_chen | 1.9e-9 | 1.37e-8 | 0.141 |
| 8 | thegamers | 2.1e-9 | 1.51e-8 | 0.143 |
| 9= | AndreasHad04 | 2.2e-9 | 1.50e-8 | 0.150 |
| 9= | shiv_m | 2.2e-9 | 1.52e-8 | 0.145 |
| 11 | lode_dockx | 2.3e-9 | 2.29e-8 | 0.102 |
| 12= | MeatProxy | 2.4e-9 | 1.52e-8 | 0.159 |
| 12= | ben3 | 2.4e-9 | 1.57e-8 | 0.156 |
| 12= | emanuel_ruzak | 2.4e-9 | 1.63e-8 | 0.149 |

- **Nine teams share raw ≈ 1.37–1.57e-8.** This suggests a common accuracy step beyond 504aldo's ~2.0e-8 floor [inf]. A κ4/SSC-type term is a candidate.
- marius (1.14e-8) remains the raw outlier.

**Elsewhere.**
- No method disclosures from J2W, marius_binner, Luna, mliston, mk1981, oliver_chen, suliman_tadros, thegamers, ben3, AndreasHad04, shiv_m, lode_dockx or MeatProxy on the forum, GitHub, LessWrong/AF or arXiv.
- Public repos are mid-table: corpaci/ARCwhitebox (5.05e-9, release Sep 29), tim8es and Hisernberg reproductions.

---

## 5. Ranked ideas (impact × feasibility, ~10 days)

1. **Latent/SSC error audit of our chain** (diagnostic; 1 day; ~0 FLOPs). *Highest value per day.*
   - On 8 fresh nets with MC labels, at layers 4, 8, 12 and 16, regress the twin's D3, D21, D4, K22 and Q31 errors on:
     - the zero-parameter SSC templates, with c² from the chain's own m̂ᵀCm̂/‖m‖²;
     - the MC latent terms T1–T4 at k = 1 and k = 16.
   - Also project the final mean error on m̂ and the top-4 principal components.
   - Go to idea 2 if ≥ 30% of the weighted κ4/D21 error is explained by the SSC templates. Go to idea 3 if the error aligns with T2/T3 at k = 16.
2. **Gain-mode SSC κ4 closure** (accuracy; ~0.005 u per layer; 2–3 days).
   - Replace or augment the λ-table with 4β_l c_l²·Sym₃(C⊗C) slices: product-form K22, σ²-scaled Q31, D4 = 12βc²σ⁴.
   - c² comes from the chain's own state; only β(l) is fitted.
   - Expected −2…−10% raw [inf]. The K22 product form beats the additive r = 1 form by +3–10 points of R², and the κ4 channel ceiling is 28%.
   - Very high write-up value: a mechanistic, nearly fit-free explanation of the κ4 law, the λ depth profile, and the (2,2)-slice paradox.
3. **k = 16 low-rank refinement as a two-level correction** (accuracy; 0.5–1 u per layer; 4–6 days; risky).
   - Gaussian-sum latent with HOUT/UT centres and responses, plus (if Lane M is go) a rank-16 κ3×edge term.
   - Justification: ≥ 90% of the mean-relevant skew at depth lives in 16 latent directions plus their bulk coupling, and our error is concentrated there.
   - Kill test: if a twin prototype with one fitted gain gains < 5% on VAL by day 3, stop.
4. **Coherent early-error check** (half a day). Budgets: covariance η ≤ 1e-4 (layers 2–4); mean-only scale ≤ 5e-5; full-state rescaling ≤ 3e-5.
   - Fix (layer-1 ρ³ diagrams; exact bivariate D21(h₁)) only if exceeded.
   - Otherwise close "early exactness" as a negative result for the write-up.
5. **Keep the per-neuron Edgeworth readout to κ4; do not build RB/subspace readouts.** The oracle shows they cannot beat 7.6e-11. Put effort into κ4 accuracy at layer 16, worth 2e-8.
6. **Planning formulas for Lane P.** decided(l) ≈ 2Φ(−3/√(ρ_{l−1}/(1−ρ_{l−1}))) and dead ≈ half that: 38.6% and 19.3% predicted at layer 16 vs 38.4% and 19.8% measured. These give deterministic per-layer pruning budgets.
7. **Write-up section, "Geometry of a fixed random ReLU net".** Assemble it from the quenched gain law, the gain-mode SSC law, the latent decomposition and the transfer law. All are new, measured, and mechanistic, which is what ARC's rubric favors.
8. **Hardening.** Audit `out=` broadcasting before the flopscope fix and re-evaluation. Watch the selection-UI announcement (about Oct 10).

---

## 6. Sources

**Geometry and signal propagation**
- Hanin & Nica, *Products of many large random matrices and gradients in deep neural networks* — https://arxiv.org/abs/1812.05994
- Li, Nica & Roy, *The future is log-Gaussian* — https://arxiv.org/abs/2106.04013
- Li, Nica & Roy, *The neural covariance SDE* — https://arxiv.org/abs/2206.02768
- Pennington, Schoenholz & Ganguli, *Resurrecting the sigmoid* — https://arxiv.org/abs/1711.04735
- Pennington, Schoenholz & Ganguli, *The emergence of spectral universality* — https://arxiv.org/abs/1802.09979
- Poole et al. — https://arxiv.org/abs/1606.05340
- Schoenholz et al. — https://arxiv.org/abs/1611.01232
- Hayou, Doucet & Rousseau — https://arxiv.org/abs/1902.06853
- Daneshmand et al. — https://arxiv.org/abs/2003.01652
- Feng et al., *Rank diminishing in deep neural networks* — https://neurips.cc/virtual/2022/poster/55444
- Cowsik et al. (geometric signal propagation, transformers) — https://arxiv.org/abs/2403.02579
- Xiao, Pennington & Schoenholz — https://arxiv.org/abs/1912.13053

**Subspace and bulk methods**
- Majda, Qi & Sapsis, *Blended particle filters* — https://pmc.ncbi.nlm.nih.gov/articles/PMC4040593
- Sapsis & Majda, *Blended reduced subspace algorithms* — https://math.nyu.edu/faculty/majda/pdfFiles/Blended%20reduced%20subspace%20algorithms%20for%20uncertainty%20quantification%20of%20quadratic%20systems%20with%20a%20stable%20mean%20state.pdf
- Sondergaard & Lermusiaux, GMM-DO — https://dspace.mit.edu/handle/1721.1/78912
- Takada & Hu, super-sample covariance — https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevD.87.123504/fulltext
- Li, Hu & Takada, *Super-sample covariance in simulations* — https://arxiv.org/abs/1401.0385
- Easley & Berry, *A higher order unscented transform* — https://arxiv.org/abs/2006.13429
- Conjugate unscented transform — https://pure.psu.edu/en/publications/conjugate-unscented-transform-and-its-application-to-filtering-an/
- Kulik & LeGrand, Gaussian-mixture splitting — https://arxiv.org/abs/2412.00343
- Doucet et al., Rao–Blackwellised particle filtering — https://arxiv.org/abs/1301.3853
- Boyen & Koller — https://arxiv.org/abs/1301.7362

**ARC**
- ARC blog index — https://www.alignment.org/blog/
- *Mechanistic estimation for expectations of random products* (L² sketching) — https://www.lesswrong.com/posts/7RyAefESvb6BQ3tMz/mechanistic-estimation-for-expectations-of-random-products
- Wu et al. paper — https://arxiv.org/abs/2605.05179
- Gabriel Wu, deduction–projection thesis — https://gabrieldwu.com/assets/thesis.pdf

**Exact moments**
- Cho & Saul, arc-cosine kernels — https://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning
- Kuang & Lin — https://arxiv.org/abs/2601.22307
- Wright & Nakahira — https://arxiv.org/abs/2403.16163
- Kan & Robotti — https://www.tandfonline.com/doi/full/10.1080/10618600.2017.1322092

**Forum, leaderboard and code**
- #18240 — https://discourse.aicrowd.com/t/18240
- #18118 — https://discourse.aicrowd.com/t/18118
- #18238 — https://discourse.aicrowd.com/t/18238
- #18218 — https://discourse.aicrowd.com/t/18218
- #18237 — https://discourse.aicrowd.com/t/leaderboard-vs-overview-leading-scores/18237
- #18197 — https://discourse.aicrowd.com/t/18197
- Leaderboard — https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards
- flopscope #275 — https://github.com/AIcrowd/flopscope/issues/275
- flopscope #264 — https://github.com/AIcrowd/flopscope/issues/264
- flopscope #284 — https://github.com/AIcrowd/flopscope/issues/284
- corpaci release — https://github.com/corpaci/ARCwhitebox/releases/tag/Phase2-submission-332675

---

## Appendix: reproduction

Scratchpad `…/scratchpad/sim2/`:

| script | arguments | what it does |
|---|---|---|
| `sim_struct.py` | run_s0: seed 0, NB = 128 batches of 4096, NBS = 32, NBT = 8; about 30 min on 2 CPUs | Pass 1: spectra, gain, linear share. Pass 2: T1–T4 decomposition, principal-component skew, D21 shares. Pass 3: split-half readout oracles. Pass 4: mean-shift transfer and antithetic covariance perturbations |
| `sim_ssc.py` | seeds 0 and 1, 131k inputs | Gain-mode/SSC template tests |
| `sim_k22.py` | — | K22 product vs additive comparison; Q31 regressors |

Results are in `run_s0/res.json`, `run_ssc0/ssc.json`, `run_ssc1/ssc.json` and `run_k22/ssc.json`.

**Caveats.**
- One network for the main run; two for the SSC test. The decompositions are neuron-averaged and self-averaging, but a twin replication on organizer nets is the real check.
- The readout oracles use in-sample responses, so they are upper bounds on any subspace readout.
- The transfer runs measure first-order (mean) and second-order (covariance) responses of the true network, not of our chain.
