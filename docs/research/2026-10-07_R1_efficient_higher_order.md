# R1: Carrying AUGMENT-level κ3/κ4 at a fraction of the cost

*Literature review plus toy experiments, 2026-10-06.*

**Units.** 1 u = one 1024³ matmul = n³ MACs ≈ 0.1% of B. Costs are given before Strassen and before pruning.

**Toy evidence.** I ran a toy chain to size the leading ideas or rule them out. The scripts are in `scratchpad/exp/`.

- **Model.** Random He-init ReLU network, depth 16, x ~ N(0, I). The covariance is closed exactly in the Gaussian sense (Mehler series to order 10).
- **κ3.** SIMPLE births (Φ_a C_ab w2_b C_bc Φ_c), carried as exact hub-factored lineages at n = 512–1024, or as a dense tensor at n = 128–384.
- **Limitations.** The toy has no κ3/κ4 feedback into μ or C, no learned parts, and uses 1–3 seeds. Read its numbers as structural sizes, not predictions.

---

## 0. Bottom line

**1. No tensor format beats the readout wall for κ3 content.** The ReLU step consumes D3_i = κ3_iii and D21_ic = κ3_iic. For any representation whose leg index has rank r, the cost of producing these is:

- n·r³ for Tucker (via face-splitting);
- r₁r₂n² per layer to transport a tensor-train or hierarchical-Tucker core;
- (CP rank)·n² for CP.

In the toy, the D21 error equals 0.9·√(1 − f), where f is the fraction of leg energy the compression keeps. A 2% D21 target therefore needs f ≥ 99.96%. That requires rank > 256 even for lineages aged ≥ 9, and rank ≈ n for ages ≤ 3 (§1). This reproduces every compression failure in the project record.

DLRA and DO theory are quasi-optimal: they track the best low-rank approximation and cannot beat it (§2). They do not move this wall.

**2. The large cheap win is algebraic: Dyson-dressed legs.** AUGMENT's three extra κ3 birth families share SIMPLE's hub leg:

- the doubled covariance edge;
- the κ3 (2,1) heavy edge;
- the κ4 (2,2) heavy edge.

To first order in the heavy part, all three fold exactly into one dressed covariance leg X̃ = Φ∘C + H. That gives AUGMENT's κ3 births at SIMPLE's column count (~8n → ~2n columns per lineage). In the toy, the spurious second-order term this introduces is only 2.4–3.0% of the correction (§5.1).

**3. Partial inclusion can hurt.** In the toy, the doubled-edge and (2,1) families are each 2.3–3.2% of D21, but anti-correlated (cos ≈ −0.8). Their sum is only 1.5–2.1%. Carrying one family without the other can be worse than carrying neither. Check which families v73 carries before anything else.

**4. The dropped κ3×edge diagram is large and does not shrink with width.** It is 2.4% / 2.9% / 3.2% of D21 at layer 16 for n = 128 / 256 / 384. That is consistent with the ~16% gain the team measured.

- It has no exact factorization cheaper than n⁴.
- Truncating C to its top-r spikes needs r ≈ 1.8×PR to capture 90% of the effect. The toy's PR ∝ n, which implies r ~ 100 at n = 1024.
- Measure it on a dense-κ3 GPU oracle first. At n = 1024 a dense κ3 is 4 GiB in fp32 and fits on the 5090 (§5.3).

**5. Subspace refinement suits only error that is genuinely concentrated.** This is the MQG-DO / "low-rank refinement" template.

- Corrections born recently are as high-rank as the content. At n = 1024, a k = 64 subspace captures only 32% of the D21 energy of the AUGMENT correction (55% at k = 128).
- So use dressing (exact, full-rank, free) for births. Reserve a moving top-k core (0.3–1 u per layer) for old-origin residuals and κ4, after confirming the concentration at the slice level (§3).

---

## 1. Anatomy: why every compression of content fails

### Lineage and readouts

A SIMPLE lineage born at layer s has two legs:

- the covariance leg X = diag(Φ_s)·C_s;
- the hub leg J = diag(w2_s).

Both are transported to layer L by the same propagator P_{s→L} = W_Lᵀ Φ_{L−1} ⋯ Φ_{s+1} W_{s+1}ᵀ. To first order:

- X_{L,s} = Cov(z_L, z_s), the cross-covariance between layers;
- J_{L,s} = E[∂z_L/∂h_s]·diag(w2), the mean Jacobian.

The readouts are:

- D3_i = 3 Σ_j J_ij X_ij²
- D21 = 2 (J∘X)Xᵀ + (X∘X)Jᵀ

### The face-splitting identity

(UA)∘(UB) = (U⊠U)(A⊙B), where ⊠ is the row-wise Kronecker product and ⊙ the Khatri–Rao product ([Khatri–Rao/face-splitting](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product)).

So for a Tucker core S of rank k with V = WᵀU:

D21 = (V⊠V)·S₍₁₂;₃₎·Vᵀ, costing n·k³ + n²·k MACs.

| k | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| cost (u) | 0.06 | 0.31 | 2.1 | 16.3 |

Tensor trains move the cost into a middle-core transport of r₁r₂n² per layer, with r ~ n for this content. CP needs ≥ 8n columns, as measured. Hadamard-product recompression ([Kressner–Periša](https://infoscience.epfl.ch/record/233913?ln=en)) uses structured matvecs costing about (n+m)r² each. That only pays when r² < n, i.e. r < 32 here.

### Toy at n = 1024, layer 16

PR = 60.5. The covariance trace beyond rank 64 is 22.8%; beyond rank 256 it is 3.6%. Legs are projected onto the top-k eigenvectors of C₁₆.

- fX, fJ = fraction of each leg's energy kept by the projection.
- eD21 = relative D21 error after projecting all legs.

| age | fX (k64) | fJ (k64) | eD21 (k64) | fX (k256) | fJ (k256) | eD21 (k256) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | .978 | .105 | .881 | 1.000 | .534 | .546 |
| 5 | .965 | .593 | .583 | .999 | .972 | .139 |
| 10 | .950 | .867 | .338 | .999 | .996 | .055 |
| 15 | .937 | .935 | .218 | .999 | .998 | .034 |

Summed over all lineages, eD21 is 0.35 at k = 64, 0.26 at k = 128 and 0.15 at k = 256.

### What the toy shows

- **Hub legs carry the rank, not covariance legs.** The covariance legs are covariance-smooth (they are Cov(z_L, ·), so their components are bounded by √λ_k). The Jacobian hub legs are not.
- **Old lineages still need rank above 256.** The 2% D21 target needs more than k = 256 even for old ages. This matches the "cliff below the age law".
- **Projecting only the single (c) index of D21 also fails.** I tested whether D21's c-index is covariance-smooth enough to read out through a top-k basis. It gives 9.4% error at k = 256, so I dropped the idea.
- **Frobenius energy and slice capture move together.** Tensor-energy capture and slice capture track each other (e.g. age 8 at k = 64: tensor 0.77, D21 0.86, D3 0.91). A claim like "X% of the κ3 error energy lies in the subspace" therefore transfers roughly to the slices.

---

## 2. Dynamical low-rank approximation: what transfers

### 2.1 The methods

**Koch–Lubich DLRA** ([2007](https://doi.org/10.1137/050639703), Tucker [2010](https://doi.org/10.1137/09076578X)).

- Evolve Y ≈ USVᵀ by Ẏ = P_T(Y)·F(Y), the projection of F onto the tangent space at Y.
- Quasi-optimality: ‖Y − A‖ ≲ ∫ ε, where ε is the distance of F from the tangent space. The method tracks the best low-rank approximation; it cannot beat it.

**Projector-splitting** ([Lubich–Oseledets 2014](https://arxiv.org/abs/1301.1058)).

- Splits the projector into K-, S⁻- and L-substeps.
- Exact when the solution has rank r.
- Its error bound O(h + ε) is independent of small singular values ([Kieri–Lubich–Walach](https://na.uni-tuebingen.de/~walach/Publikationen/lowrank-siam.pdf)).

**BUG integrators.**

- BUG ([Ceruti–Lubich](https://arxiv.org/abs/2010.02022)): K-step and L-step update the bases; a Galerkin S-step works in the new bases. No backward substep; symmetry-preserving.
- Rank-adaptive BUG ([Ceruti–Kusch–Lubich](https://arxiv.org/abs/2104.05247)): augment Û = orth[K(t₁), U₀] (rank 2r), do the Galerkin step there, then truncate at tolerance ϑ. The error is c₁ε + c₂h + c₃·nϑ, again robust to small singular values.
- Parallel BUG ([2304.05660](https://arxiv.org/abs/2304.05660)) solves all substeps simultaneously. There is also a midpoint second-order variant ([2402.08607](https://arxiv.org/abs/2402.08607)).
- Tensor-train and hierarchical-Tucker versions: [Lubich–Oseledets–Vandereycken](https://arxiv.org/abs/1407.2042) and [Lubich–Rohwedder–Schneider–Vandereycken](https://infoscience.epfl.ch/record/189617?ln=en). Lubich's 2026 slides survey BUG on tree tensor networks ([slides](https://cermics-lab.enpc.fr/wp-content/uploads/2016/12/slides_lubich_2026.pdf)).

**Step-truncation** ([Rodgers–Dektor–Venturi](https://arxiv.org/abs/2008.00155)): take an exact step, then truncate.

### 2.2 Applying it to our discrete map

Our map is T ← M^{⊗3}T + births, with M = WᵀΦ, plus a few Hadamard terms.

- **The linear part is already exact.** M^{⊗3} applied to a Tucker tensor is exact, so the linear part needs no DLRA.
- **DLRA reduces to step-truncation.** BUG on this map is exactly what the shared-tier "join" already does: augment with new birth directions, take a Galerkin step, truncate.
- **The error is a transported sum of truncations.** The global error is Σ_l ‖P_{l→L}‖·τ_l.
- **Truncations in the top modes persist; the rest decay.** For gated products of k layers, σ²_max(P) exceeds the mean squared singular value by about e(k+1), a Fuss–Catalan-type spread ([Hanin–Nica](https://arxiv.org/abs/1812.05994); [Pennington–Schoenholz–Ganguli](https://arxiv.org/abs/1802.09979)). Errors left in the top modes survive, while errors in the bulk contract. This is why "energy ≈ sensitivity" (N1) came out the way it did.
- **What DLRA cannot do.** It cannot make content of rank > 256 cheap. Section 1 shows the best approximation is already too poor, and DLRA is quasi-optimal.

**Hadamard terms.** For elementwise nonlinear terms, DLRA codes use DEIM or cross interpolation: evaluate the nonlinearity only at O(r) selected rows and fibres, F(V) ≈ F[q,:]·V[q,:]⁻¹·V ([Naderi–Babaee](https://ar5iv.labs.arxiv.org/html/2207.10656); [DEIM-Tucker cross](https://ar5iv.labs.arxiv.org/html/2401.04249); [Ghahremani–Babaee](https://arxiv.org/abs/2403.12826)). That requires the result to be low-rank in the sampled index. Here D21 is low-rank in neither index for young content (c-index projection fails, §1), so DEIM does not apply to the readout.

**Lyapunov / Stein analogies.** Low-rank Smith or ADI methods need contractive M. Ours is critical (He init) and amplifies its top modes. This is the same reason old content stays high-rank.

### 2.3 What to borrow

**(a) Conservation by moment-preserving truncation.**

- *The idea from kinetic DLRA:* carry the conserved low-order moments exactly and constrain the low-rank remainder so it does not alter them. Examples: macro–micro decomposition ([Coughlin–Hu–Shumlak](https://arxiv.org/abs/2311.09425)), quasi-conservative DLRA ([Einkemmer–Lubich](https://arxiv.org/abs/1807.02338)), HOLO ([Peng–McClarren](https://arxiv.org/abs/2011.06072)).
- *Our analog:* any truncation must leave the (3)- and (2,1)-slices it is supposed to keep unchanged at the compression layer. The A1 review's P2 diagonal leak (`sb = S3 − rowsum(Z∘L)`) is exactly this failure mode.
- *Cost and gain:* O(n·r). Small gain, but free.

**(b) BUG robustness for any moving subspace.** If a basis is evolved (§3), use the augment–Galerkin–truncate update. DO-style mode equations contain C_YY⁻¹ and are ill-conditioned when modes carry little energy ([Babaee–Sapsis](https://arxiv.org/abs/1509.04629); [DO ≡ DLRA geometry, Feppon–Lermusiaux](https://dspace.mit.edu/handle/1721.1/120002)).

---

## 3. DO / DyBO and blended non-Gaussian subspace closures

### 3.1 The literature template

**DO and DyBO.** DO ([Sapsis–Lermusiaux 2009](https://mseas.mit.edu/?p=898)) writes u = ū + Σᵢ Yᵢ(ω,t)·eᵢ(t), with orthonormal modes and non-Gaussian coefficients. DyBO ([Cheng–Hou–Zhang](https://archive.ymsc.tsinghua.edu.cn/pacm_download/477/11600-Zhiwen_Zhang_1.pdf)) is bi-orthogonal. The two are equivalent to each other ([Choi–Sapsis–Karniadakis](https://dspace.mit.edu/entities/publication/fa4ecd0f-21ae-4c3b-9cd4-5aff6067c1da)) and to DLRA.

**MQG.** [Sapsis–Majda's MQG closure (PNAS 2013)](https://math.nyu.edu/faculty/majda/pdfFiles/Statistically%20Accurate%20Low%20Order%20Models%20for%20Uncertainty%20Quantification%20in%20Turbulent%20Dynamical%20Systems.pdf):

- evolves the mean and full covariance;
- replaces the third-order flux Q_F by extra damping plus additive noise, Q_F⁻ = NR + RN* and Q_F⁺ ∝ Q_F∞⁺;
- calibrates them by statistical energy balance.

**Blended MQG-DO.** [Blending modified Gaussian closure with non-Gaussian reduced subspaces](https://math.nyu.edu/faculty/majda/pdfFiles/Blending%20Modified%20Gaussian%20Closure%20and%20Non-Gaussian%20Reduced%20Subspace%20Methods%20for%20Turbulent%20Dynamical%20Systems.pdf); also [blended reduced subspace](https://math.nyu.edu/faculty/majda/pdfFiles/Blended%20reduced%20subspace%20algorithms%20for%20uncertainty%20quantification%20of%20quadratic%20systems%20with%20a%20stable%20mean%20state.pdf).

- **Full space (cheap):** a Gaussian closure evolves R.
- **Subspace (expensive):** an s-dimensional DO subspace carries the non-Gaussian coefficients.
- **The coupling:** the subspace third moments ⟨Y_m Y_n Y_k⟩ supply the flux,

  dR/dt = L_v R + R L_v* + Q_σ + Q_MQG + (Q_VS − Q_VS,∞)

  where Q_VS − Q_VS,∞ is the subspace flux minus its double-counted calibrated part.
- **Feedback the other way:** the subspace dynamics receive a correction N_C for interactions with the unresolved modes.
- **Results:** monotone improvement in s; no tuned parameters.

**Conditional Gaussian.** [Chen–Majda](https://arxiv.org/abs/1709.05562) use a conditional-Gaussian mixture in the high-dimensional part and a kernel density in the low-dimensional part. That is a sampling method, so it falls in the class the team has already ruled out.

### 3.2 Our version: subspace-blended correction core (SB-core)

**State.** Keep the full chain (μ, C, lineages, κ4 core) as the "MQG" layer. Add:

- an orthonormal U_l (n×k);
- a symmetric correction core ΔS3 (k³);
- optionally ΔS4 (k⁴ at k = 16).

ΔS holds the fine-minus-coarse difference inside span(U), mirroring MQG-DO's "subtract Q_VS,∞" so nothing is counted twice.

**Readout at layer l.** With V = Wᵀ_{l+1} U_l:

- D3_i += ΔS3[v_i, v_i, v_i]
- D21 += (V⊠V)·ΔS3₍₁₂;₃₎·Vᵀ, costing nk³ + n²k
- D22 += (V⊠V)·ΔS4₍₁₂;₃₄₎·(V⊠V)ᵀ, costing nk⁴ + n²k²
- D4 and D31 follow the same pattern.

**Update.**

ΔS3_{l+1} = ΔS3_l ×₁R ×₂R ×₃R + Πₖ[births_fine − births_coarse] + Πₖ[inheritance corrections]

- R = U_{l+1}ᵀ·M_l·U_l.
- Πₖ projects each hub-factored family (P, Q, R) onto the subspace: Σⱼ (UᵀPⱼ)⊗(UᵀQⱼ)⊗(UᵀRⱼ).
- Anything that leaks out of span(U) is dropped. That is the assumption the error-concentration claim must justify.

**Basis.** Choose one:

- **Discrete OTD:** U_{l+1} = qr(M_l·U_l), the Benettin/QR algorithm. It tracks the most amplified directions, which is where transported errors concentrate ([OTD](https://arxiv.org/abs/1509.04629)).
- **BUG augmentation:** orth[M_l·U_l, top-k range of C_{l+1}], then truncate by the singular values of the Galerkin core.

**Cost per layer.**

| item | k = 32 | k = 64 |
|---|---:|---:|
| basis | 0.03 u | 0.06 u |
| projected family (each) | 0.12 u | 0.44 u |
| κ3 readout | 0.06 u | 0.31 u |
| core rotation | < 0.01 u | 0.05 u |

The κ4 core at k = 16 costs 0.31 u for its readout and 0.06 u per projected family. Unlike lineages, the core accumulates every age, so its cost does not grow with depth.

### 3.3 Toy evidence: what the subspace can and cannot hold

| correction (D21 at layer 16) | k = 64 | k = 128 | note |
|---|---:|---:|---|
| AUGMENT one-heavy-edge net correction, n = 1024: energy captured | 0.32 | 0.55 | young-dominated; ages ≤ 3 give 38% of its norm |
| same, ages ≤ 2 exact + older in subspace: relative error | 0.60 | 0.41 | |
| κ3×edge, dense, n = 256 (k = 64 is ~5×PR): tensor energy captured / D21 relative error | 0.81 / 0.47 | — | at k = 16 / 32: 0.29 / 0.83 and 0.50 / 0.71 |
| standardized skew, top-16 modes vs random directions vs single neurons | 0.085 / 0.092 / 0.091 | — | top mode alone: 0.29 |

**Verdict.**

- Diagram corrections are as high-rank as the content. A k ≤ 64 core at n = 1024 would recover roughly a third of them.
- The top covariance modes are not systematically more non-Gaussian than random directions; the leading mode is the exception. So "higher cumulants only in the top modes" has weak support, apart from k ≈ 1–4.
- The team's measured concentration (81–92% of κ3 error in 64 dimensions) must come mostly from old-origin error. If so, SB-core can capture it, but only if a better in-subspace source term is known. The core does not invent physics.

**Falsification test (½ day).** Take v73's D21 error against labels (or against the AUGMENT chain) at layers 8, 12 and 16. Fit the best (V⊠V)·Sᵀ·Vᵀ-structured correction for k = 32 and 64 by least squares; this is linear in S. If it removes less than 50% of the ‖ΔD21‖² energy, drop SB-κ3. Separately, the oracle "top-32 fix → −19%" says the ceiling is modest.

---

## 4. Symmetric-tensor formats, Hadamard recompression, streaming: verdict

**Symmetric CP.** A generic symmetric r³ tensor has CP rank about r²/6 (Alexander–Hirschowitz; [secant-variety survey](https://arxiv.org/abs/1812.10267)). Expanding a Tucker core into CP therefore costs more, not less.

**Fewer multiplications by symmetry.** [Solomonik–Demmel](https://www.research-collection.ethz.ch/items/680c0221-c81a-4173-adc6-a5a5a70d2591) save multiplications at the price of extra additions. Under the flopscope meter the best available saving is 2× on the (p,q)-symmetric part of n·k³ (the team's #13 "do not apply").

**Sketches.**

- TensorSketch and Khatri–Rao projections ([Ahle et al.](https://arxiv.org/abs/1909.01410); [KRP analysis](https://arxiv.org/abs/2507.23207)) are unbiased with variance ∝ ‖A∘B‖²‖C‖²/s. That needs s ~ m/ε², which is useless at 2%.
- Streaming Tucker ([Sun–Guo–Luo–Tropp–Udell](https://arxiv.org/abs/1904.10951)) and online CP ([OnlineCP](https://doi.org/10.1145/2939672.2939763)) inherit the same rank floor as §1.

**The one positive use.** A Tucker core is depth-independent. Every age folds into one k³ object read out at n·k³. So it is the right container for small corrections that tolerate ~30–50% error (§3), never for content.

---

## 5. Closure stability, resummation, and the cheap AUGMENT

### 5.1 Dyson-dressed legs: AUGMENT κ3 births at SIMPLE cost

**The expansion.** Expand relu around the Gaussian reference with Wick coefficients w(k) = E[relu⁽ᵏ⁾]. In particular w1 = Φ, w2 = φ/σ, w3 = −αφ/σ². The post-ReLU κ3 is a sum over connected graphs whose vertices carry k legs. The hub-factored terms up to one heavy edge are:

| family | Hermite orders | term | hub weight | heavy leg on a (column j = hub) |
|---|---|---|---|---|
| SIMPLE | (1,2,1) | w1_a w2_b w1_c C_ab C_bc | w2 | — (light leg X = Φ∘C) |
| F1 doubled edge | (2,3,1) | ½ w2_a w3_b w1_c C_ab² C_bc | w3 | ½ w2_a C_aj² |
| F2 κ3 (2,1) edge | (2,2,1) | ½ w2_a w2_b w1_c κ3_aab C_bc | w2 | ½ w2_a κ3_aaj |
| F3 κ4 (2,2) edge | (2,3,1) | ¼ w2_a w3_b w1_c κ4_aabb C_bc | w3 | ¼ w2_a κ4_aajj |

These are exactly ARC's AUGMENT families. Each keeps the hub leg e_j and differs from SIMPLE in one leg only.

**The dressing.** Absorb the hub-weight ratio per column; note w3/w2 = −α/σ, which stays finite for decided hubs. Define

X̃[:, j] = Φ∘C[:, j] + H[:, j], with

H[a, j] = (w3_j/w2_j)·½ w2_a C_aj² + ½ w2_a D21[a, j] + (w3_j/w2_j)·¼ w2_a D22[a, j]

for a ≠ j, with the diagonal handled as the chain's closure handles it.

Then

Sym Σⱼ X̃ⱼ ⊗ (w2_j eⱼ) ⊗ X̃ⱼ = SIMPLE + F1 + F2 + F3 + Sym Σⱼ Hⱼ ⊗ w2_j eⱼ ⊗ Hⱼ.

**The spurious term is benign.** The last term is second order in the heavy part. For the doubled-edge piece it equals the true next-order (2,4,2) diagram except that w3²/w2 = α²φ/σ³ replaces w4 = (α²−1)φ/σ³. So the dressing even approximates the next order. This is the Dyson / 2PI "dressed propagator" idea: resum insertions into the line instead of carrying them as separate families ([Berges](https://arxiv.org/abs/hep-ph/0105311); [Berges review](https://arxiv.org/abs/hep-ph/0109170)).

**Cost.** Forming H takes a few elementwise n² passes, about 0.003 u. Transport and readout stay at the SIMPLE count: two legs and two readout contractions per lineage. That is AUGMENT's ~8n columns cut to ~2n.

**Generalization.** Any two families that share the same hub leg up to per-column scalars, and in which one leg is a small perturbation, merge the same way: (H, P, P) + γ(H, P, Q) → (H, P + γQ/2, ·). The error is (γ²/4)(H, Q, Q). If v73 carries a separate heavy family (the "(2,1) leg" or the A_L-family) whose hub leg is column-proportional to A_D, merging removes a whole family. That is up to ~5 u per layer if it is the second of two equal families. Check the proportionality first.

**Toy numbers** (D21 at layer 16; F1 + F2 only, since the toy has no κ4):

| run | net AUG / SIMPLE | F1 / SIMPLE | F2 / SIMPLE | cos(F1, F2) | spurious / net |
|---|---:|---:|---:|---:|---:|
| n = 1024, seed 0 | 0.015 | 0.026 | 0.029 | — | 0.024 |
| n = 512, seed 1 | 0.018 | 0.023 | 0.031 | −0.82 | 0.029 |
| n = 512, seed 2 | 0.021 | 0.028 | 0.032 | −0.75 | 0.030 |

**Expected value.**

- By 504aldo's ε² law (extra MSE ≈ 4.2e-6·ε²), the toy's net correction of 1.5–2% of D21 is worth 1–1.7e-9, about 5–8% raw. The remainder of AUGMENT's 23% would come from its κ4 core (O(n³) per layer, cheap) and the F3 family.
- If v73 carries F2 but not F1, the toy says its D21 is off by |F1| ≈ 2.3–2.8%. Completing the set would then be worth about 10–15% raw.

**Falsification test (1–2 days).**

1. In the torch AUGMENT chain at n = 1024, replace the extra families by X̃.
2. Compare D3/D21 at layers 4, 8, 12, 16 and the final MSE against full AUGMENT and against SIMPLE.
3. Kill if the dressed version recovers less than 50% of AUGMENT's gain. Otherwise port it to v73, together with the κ4 core, and refit the gains in the twin.

The refit is required: the learned knobs currently compensate for the missing families.

### 5.2 Truncation pathologies and their known fixes

**Quasi-normal closures.** The quasi-normal approximation (κ4 set to 0) produced negative energy spectra (Ogura). EDQNM fixed this by damping the third cumulant; realizable Markovian closures enforce realizability ([Bowman–Krommes–Ottaviani](https://www.math.ualberta.ca/~bowman/publications/rmc1.pdf)).

**Moment closures in chemical kinetics.** These can give unphysical moments or diverge, and higher order is not monotonically better ([Schnoerr et al. 2014](https://arxiv.org/abs/1407.8316), [2015](https://arxiv.org/abs/1508.01737)).

**Secular growth.** In time-dependent field theory, fixed-order truncations grow secularly. 2PI (self-consistently dressed) truncations cure this ([Berges](https://arxiv.org/abs/hep-ph/0105311)).

**Our analogs.**

1. **Dress, don't append (§5.1).** Evaluate Φ and w2 under the corrected marginal, i.e. the kink density from Edgeworth or saddlepoint. This is probably already done; verify.
2. **Add families in complete sets.** The cancellations make partial sets risky.
3. **Damping as gains.** Fitted per-age damping on carried content is the EDQNM analog (`g_u` already exists).
4. **Realizability guards.** p_z(0) > 0; κ4 ≥ −2σ⁴ for the marginals.
5. **Final readout.** A saddlepoint / Lugannani–Rice readout of E[(z)₊], the option-pricing formula with strike 0 ([Kwok–Zheng saddlepoint survey](https://www.math.hkust.edu.hk/~maykwok/piblications/Saddlepoint.pdf)). Worth ≤ 5% raw, at low risk.

### 5.3 κ3×covariance-edge: the large term with no cheap form

**The term.** Δκ3_abc = w2_a w2_b w1_c·C_ab·κ3_abc + 2 permutations, from Hermite orders (2,2,1) with a κ3 hyperedge.

**Why it is expensive.** Per lineage hub j, the (a,b) block is diag(w2∘αⱼ)·C·diag(w2∘βⱼ), which has rank n. Even the next layer's D3 alone,

Σⱼ [(Wᵢ∘w2∘αⱼ)ᵀ C (Wᵢ∘w2∘βⱼ)]·[(Wᵢ∘w1)ᵀγⱼ],

is a genuine four-index contraction: n⁴ ≈ 1024 u per lineage.

**Toy measurements.**

- Size (dense, Gaussian closure, layer 16): 2.4% / 2.9% / 3.2% of D21 and 1.7–2.2% of D3 for n = 128 / 256 / 384. It does not shrink with width.
- The reason it persists: PR grows ∝ n at fixed depth (12 / 24 / 62 at n = 256 / 512 / 1024, layer 16), so the correlations inside the dominant modes stay O(1).
- So the (L/n)^K error conjecture of ARC's paper ([arXiv 2605.05179](https://arxiv.org/abs/2605.05179)) is not yet in force at 16 layers.
- The triangle diagram (2,2,2) is only 0.3–0.6% of D21: ignore it.

**Options.**

- **(a) Spike truncation.** Replace C by its top-r spikes: Δ ≈ Σ_{ρ≤r} λ_ρ Σⱼ (w2∘αⱼ∘v_ρ)⊗(w2∘βⱼ∘v_ρ)⊗(w1∘γⱼ). That is r extra hub families per lineage.
  - Toy at n = 192 (PR ≈ 9): relative error 0.79 / 0.36 / 0.11 / 0.03 at r = 1 / 4 / 16 / 48.
  - At n = 1024 that suggests r ~ 64–128. That is affordable only for the youngest one or two lineages with an immediate fold, about 4r u per layer.
- **(b) Learned proxy.** Regress the dense-oracle ΔD21 on cheap structures (D21∘C row sums, projections onto the top modes, w2-weighted D21) and fit the gains in the twin.
- **(c) SB-core.** Cheap, but the toy capture is low (§3.3).

**Decision gate.** Build the dense-κ3 GPU oracle at n = 1024: 4 GiB per tensor in fp32, about 6.6e12 FLOPs per layer for transport, roughly 0.1 s per layer on the 5090. Only κ3 needs to be dense, not κ4 — the "≥133 GiB" in the Sep 25 review cannot be for κ3 alone. Measure:

- the true size of ΔD21 at n = 1024;
- the spike-rank curve;
- the R² of a proxy.

Proceed only if the size is above 2% and either r ≤ 16 captures more than 70% or a proxy reaches R² > 0.6.

---

## 6. Other fields: short verdicts

- **Finite-width ML theory** ([Yaida](https://ar5iv.arxiv.org/html/1910.00019); [Dyer–Gur-Ari](https://arxiv.org/abs/1909.11304); [Roberts–Yaida–Hanin](https://arxiv.org/abs/2106.10165)). These are recursions over weight ensembles, not per network. Their use here is explanatory: the per-network propagator spectra (Hanin–Nica log-normality; Pennington's Fuss–Catalan-type laws) predict the age law, rank ≈ n/(2·age). That is useful for the write-up, not for cost.
- **Loop corrections and Onsager terms** ([Montanari–Rizzo](https://arxiv.org/abs/cond-mat/0506769)). On dense random graphs, loop terms reduce to Onsager self-energies only after averaging over the couplings. Here each missing diagram has zero mean over the next layer's W but is O(1) per instance, so there is no deterministic shortcut. Not recommended.
- **Non-Gaussian component analysis** ([Blanchard et al.](https://jmlr.csail.mit.edu/papers/v7/blanchard06a.html)). The "Gaussian plus low-dimensional non-Gaussian" model is contradicted by the toy's skew isotropy (§3.3), apart from the leading mode.
- **Hybrids.** Conditional-Gaussian or sigma-point mixtures cost one covariance propagation (1–2 u) per node per layer. With ≥ 5 nodes that is more than 5 u per layer for a single mode. Not recommended.

---

## 7. Ranked list (expected impact × feasibility, Oct 7–17)

| # | Idea | Cost per layer | Expected effect | Days | Kill test |
|---|---|---|---|---|---|
| 1 | Dyson-dressed legs for AUGMENT F1–F3; audit v73 for partial families | ≈0 u (≤ −5 u if a family merges) | −5 to −15% raw, plus possible cost cut | 2 | dressed recovers < 50% of AUGMENT's gain in the torch chain |
| 2 | AUGMENT κ4 memory core plus κ3→κ4 births (if not already in v73) | ~1.2–2 u | part of the 28% κ4 ceiling | 2 | κ4 slices at layers 4/8/12 improve < 10% |
| 3 | Dense-κ3 GPU oracle at 1024×16 (enabler) | offline | decides #4 and #5 | 1 | — |
| 4 | κ3×edge: learned proxy, or r ≤ 16 spikes on the youngest lineage plus fold | 0.1–4 u | up to −10 to −16% raw | 3–4 | size < 2% of D21, or capture < 70% |
| 5 | SB-core (BUG/OTD basis, k = 32–64) for concentrated residuals, plus a κ4 core at k = 16 | 0.3–1 u | ≤ −19% (team oracle), likely less | 4–5 | structured S removes < 50% of ‖ΔD21‖² |
| 6 | Slice-preserving truncation (P2 diagonal) | ~0 u | ≤ 2% | 0.5 | twin A/B |
| 7 | Saddlepoint final readout | ~0 u | ≤ 5% | 1 | fresh panel |

**Not recommended (literature plus toy):**

- Tucker, TT, HT or CP compression of content (2% D21 needs 99.96% of leg energy);
- c-index-only readout projection (9.4% error at k = 256);
- TensorSketch or Khatri–Rao sketches; DEIM on the readout;
- streaming CP or Tucker;
- the triangle diagram (< 0.6%);
- κ5/κ6 in the top modes beyond k ≈ 2;
- conditional-Gaussian or sampling mixtures.

**Not verified here:**

- v73's exact family list and the proportionality of its hub legs;
- the real (non-Gaussian, fitted) sizes of F3 and of the κ4 core;
- whether the team's 81–92% concentration figure is tensor energy or slice error.

Ideas #1 and #3 resolve the first two within two days.

**Scripts** (scratchpad `exp/`):

- `lineage_subspace.py`: §1 table
- `dressed_legs.py`: §5.1 and §3.3
- `tensor_vs_slice.py`
- `dense_k3edge.py`, `dense_k3edge_lean.py`, `dense_k3edge_spikes.py`: §5.3
- `mode_skew.py`
- `pr_scaling.py`
