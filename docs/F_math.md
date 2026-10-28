# F_math — Cumulant tower for the ReLU-net mean: derivations, sandbox numerics, cost structure

Sandbox code: `/home/claude/sandbox/` (numpy/scipy only; port-ready).  
`relu_hermite.py` (all 1-D / 2-D / 3-D closed forms, verified to 1e-14 against quadrature in `test_hermite.py`),
`tower.py` (full-tensor first-order cumulant tower: κ3 as n³ tensors per birth layer, κ4 as n⁴ tensor, pattern-aware ReLU transport, indirect shifts, Edgeworth readout, sphere/scale-mixture variants),
`mc.py` (Monte-Carlo truth with per-layer cumulant accumulators), `adjoint.py` (forward lineages vs backward adjoint, exact to 1e-16),
`experiments.py` (Q2 ablations), `radial.py` (§7 template projections, Q4 self-averaging), `sketch.py` (Q5), `costs.py` (Q3 GF table).
Raw results: `/home/claude/sandbox/results/*.json|npz|log`.

Notation. z_l ∈ R^n pre-activation of layer l, h_l = ReLU(z_l), z_{l+1} = W_{l+1} h_l, h_0 = x ~ N(0,I).
Per neuron: σ_i² = C_ii, α_i = μ_i/σ_i, ẑ = (z−μ)/σ, φ, Φ standard normal pdf/cdf, He_p probabilists' Hermite.
g_i = Φ(α_i) is the gate. "Sym" = average over index permutations.

---

## 0. Executive summary

1. **Indirect beats direct.** At n = 128, L = 8 the first-order κ3/κ4 tower removes 91 % of the K2-only MSE (1.18e-4 → 1.07e-5). The direct final-layer κ3 readout alone removes only 21 % of that; the κ3-induced *intermediate* mean shifts alone remove 58 %; both together 91 %. The adjoint (exact direct term for all ages) is worth ≤ 12 % / 7 % / 4 % of the remaining MSE once indirect ages ≤ 3 / 4 / 5 are exact, because ≈ 80 % of the value of an old lineage is its coupling into intermediate μ_l (§2).
2. **Backward adjoint = forward lineages exactly** (4e-16), 59 vs 150 n³-matmuls for the direct term at L = 16; triangle diagrams are the only non-contractible born terms and are < 1 % at n = 1024 (§3).
3. **Sketching is dead**: every expensive contraction is a random-sign sum, so any subset/sketch estimator has relative variance ≈ n/k (§5).
4. **Radial / scale mode (§7).** (a) Exact identities: E_G = e₁(n)·E_sphere; the Gaussian problem = sphere problem ⊗ χ_n/√n scale mixture, which adds Cov += v m mᵀ, κ3 += 6v Sym(m⊗C), κ4 += 12v Sym(C⊗C) + 12v Sym(κ3⊗m) at *every* layer with the same v = 1/(2n) (no Sym(m⊗m⊗C) term). (b) A second, *emergent* scale mode is injected by every layer (‖h_l‖² fluctuations) and dominates κ4 at depth: it is a first-order κ3→κ4 response (κ4(h)_iijj ⊃ 2X2_iT2_j κ3(z)_ijj) that gate-transported κ4 legs miss entirely; adding it to the full tower cut the n = 64 residual MSE by 38 % (8.2e-5 → 5.0e-5) and explains the fitted λ-law v_eff ≈ 0.006–0.0095 ≫ input radius 1/(2n). (c) The sphere formulation (negative input κ4 face −6/(n+2)·Sym(C₁⊗C₁), carried with √T2 gates, final ×e₁) beats the Gaussian formulation of the *same* chain by 26 % (exact κ4) / 17.5 % (κ3-only) at n = 64 and 12 % at n = 128: the advantage shrinks roughly like n^{-1/2}, so ≈ 4–6 % is the realistic expectation at n = 1024 unless the chain is truncated (then more).

---

## 1. Edgeworth readout: E[ReLU(z)] for a weakly non-Gaussian z

Let ẑ have cumulants γ3 = κ3/σ³, γ4 = κ4/σ⁴. Gram–Charlier/Edgeworth: p(x) = φ(x)[1 + γ3 He_3(x)/6 + γ4 He_4(x)/24 + γ3² He_6(x)/72 + …].
With f(x) = ReLU(μ+σx) and Stein's identity E_φ[f He_p] = E_φ[f^{(p)}], f'' = σ δ(x+α):

    E_φ[f He_p] = σ He_{p−2}(−α) φ(α)   (p ≥ 2),   E_φ[f He_1] = σ Φ(α),  E_φ[f] = σφ(α)+μΦ(α).

Hence, exactly to first order in each cumulant,

    E[ReLU(z)] = σφ(α) + μΦ(α) + c3(α)·κ3/σ² + c4(α)·κ4/σ³ + c33(α)·κ3²/σ⁵ + O(κ5, κ3κ4, …)

    c3(α)  = −α φ(α)/6
    c4(α)  = (α²−1) φ(α)/24
    c33(α) = He_4(α) φ(α)/72 = (α⁴−6α²+3) φ(α)/72        (second order, needed in the sandbox at n ≤ 128)

Numerical check (`test_hermite.py`, standardized-Gamma family, exact quadrature): first-order residual is O(γ3²) and vanishes to 1e-8–1e-7 once c33 is added (table in the test output).

Sensitivities (for the error budget): ∂E/∂μ = Φ(α) = g, ∂E/∂σ = φ(α), ∂E/∂κ3 = c3/σ², ∂E/∂κ4 = c4/σ³. With independent errors the final MSE splits as

    MSE ≈ ⟨g² δμ²⟩ + ⟨φ² δσ²⟩ + ⟨(c3/σ²)² δκ3²⟩ + ⟨(c4/σ³)² δκ4²⟩ + cross terms.

Magnitudes at the last layer of a 16-deep He net (c ≈ 0.9 of E[z²] is in the mean): |α| is typically 1–4, so g ≈ 0.5–1, φ ≈ 0.05–0.3, |c3| ≤ 0.05 (max at |α| = 1), |c4| ≤ 0.017.
A δκ3 of 1 % of σ³ (γ3 = 0.01) moves E by ≤ 5e-4·σ; a δμ of 1e-4 moves E by ≈ 1e-4. So μ (and hence the indirect channel) is the most sensitive input, κ4 the least.

Important first-order *responses of all ReLU output cumulants* (all verified by finite differences in `test_hermite.py`; they are what "transport through the ReLU" really is). With A_p = E[h He_p(ẑ)], B_p = E[(h−m)² He_p(ẑ)], C_p = E[(h−m)³ He_p(ẑ)] (closed forms in `relu_hermite.py`):

| source → target | exact first-order factor | gate linearization |
|---|---|---|
| κ3(z)_ijk → κ3(h)_ijk, i,j,k distinct | g_i g_j g_k (+O(ρ)) | g_i g_j g_k |
| κ3(z)_iik → κ3(h)_iik | T2_i g_k,  T2 = Φ − (m/σ)φ  (= B_2/2σ²) | g_i² g_k |
| κ3(z)_ikk → κ3(h)_iik (cross term!) | X2_i φ_k/σ_k,  X2 = m(1−Φ) | 0 |
| κ3(z)_iii → κ3(h)_iii | T3 = (C_3 − 3V A_3)/(6σ³) | g_i³ |
| κ3(z)_iii → m_i | c3/σ² | — |
| κ3(z)_iii → Var h_i | v3 = φ(σ+mα)/(3σ²) | — |
| κ3(z)_iij → Cov(h)_ij | ½ φ_iΦ_j/σ_i  (+ symmetric) | — |
| κ4(z)_iiii → κ4(h)_iiii | T4 (closed form, `transport_k4`) | g⁴ |
| κ4(z)_iijj → κ4(h)_iijj | T2_i T2_j | g_i² g_j² |
| κ4(z)_iijk → κ4(h)_iijk | T2_i g_j g_k | g_i² g_j g_k |
| κ4(z)_iiii → m_i, Var h_i | c4/σ³, v4 = −φ(σα+m(α²−1))/(12σ³) | — |
| κ4(z)_iijj → Cov(h)_ij | ¼ φ_iφ_j/(σ_iσ_j) | — |
| κ4(z)_iijj → κ3(h)_iij | ½ T2_i φ_j/σ_j | — |

Example at α = 0: T3 = 0.341 vs g³ = 0.125; T2 = 0.341 vs g² = 0.25; T4 = 0.340 vs g⁴ = 0.0625.
Gate linearization is *asymptotically exact for the generic (all-distinct) entries only*. For a generic transported κ3 tensor the pair/diagonal entries are an O(n^{-1/2}) / O(n^{-1}) fraction of the next-layer diagonal contraction, so gating κ3 legs is fine at n = 1024 (3 % of the κ3 term). For κ4 the *dominant* channel into the next diagonal is the coherent pair sum Σ_{i≠j} W_ai²W_aj² κ4_iijj, whose transport factor is T2_iT2_j, not g_i²g_j² — gate-linearized κ4 legs are wrong by a factor ≈ 2 per layer (§2, §7).

---

## 2. Size of terms (sandbox ablations)

Setup: W_l i.i.d. N(0, 2/n), x ~ N(0, I), MC truth 1e8 samples (n = 64) / 5e7 (n = 128) / 2e7 (n = 256), MC floor 3.9e-9 / 4.8e-9 / 8.6e-9. K2-only MSE: 8.00e-4 / 1.18e-4 / 4.16e-5; κ3-only full tower: 9.56e-5 / 1.07e-5 / 4.72e-6 (the n = 256 residual is 11 % of K2 and is dominated by the missing κ4 — κ4 could only be run as a full tensor at n = 64, where it is 40 % of the residual once the κ3→κ4 channel is included). "Full" = full-tensor first-order tower (κ3: all born patterns to order 6, pattern-aware transport, indirect mean + cov shifts, Edgeworth readout with c3, c4, c33; κ4 full tensor at n = 64 only). Numbers = MSE vs MC of the final-layer mean. (Files: results/abl_n64_L8_s0.log, abl_n128_L8_s0.log.)

**Caveat on regime.** At n = 64 the last-layer skewness is γ3 ≈ 0.6 and the first-order tower leaves a residual (8e-5) that is 10 % of the K2 error and is dominated by second-order terms (even Edgeworth with the *exact MC* κ3, κ4 leaves 2e-6). The K2 error falls like n^{-2.76} and the full-tower residual like n^{-2.9} between n = 64 and 128; extrapolated to n = 1024, L = 8: K2-only MSE ≈ 4e-7, first-order residual ≈ 2–3e-8. Fractions (not absolute values) are what transfers to n = 1024; they are stable between n = 64 and n = 128.

### 2.1 Channel ablations

| variant | n = 64 MSE | n = 128 MSE | share of κ3 effect removed (n=128) |
|---|---:|---:|---:|
| K2 only | 8.00e-4 | 1.18e-4 | 0 |
| **full tower** | 8.16e-5 (5.0e-5 with κ3→κ4) | 1.07e-5 | 100 % (= 1.073e-4) |
| direct readout only (no intermediate shifts) | 6.40e-4 | 9.35e-5 | 21 % |
| indirect mean shifts only (+ direct) | 1.47e-4 | 4.99e-5 | 58 % |
| indirect cov shifts only (+ direct) | 8.56e-4 | 1.83e-4 | < 0 (cov shift without mean shift is unbalanced) |
| transport: gate (g³ on all entries) instead of pattern-aware | 9.25e-5 | 1.115e-5 | −0.4 % |
| born: diagonal only | 2.68e-4 | 5.62e-5 | 58 % |
| born: diagonal + pair | 1.19e-4 | 2.20e-5 | 90 % |
| born: + hub (order-2 star) | 8.80e-5 | 1.089e-5 | 99.8 % |
| born: + order-3 stars | 8.17e-5 | 1.072e-5 | 100 % |
| no κ3² readout term | 7.17e-5 | 1.039e-5 | (second order; irrelevant at 1024) |
| no κ4 (n = 64, κ4 = full tensor) | 9.56e-5 | — | κ4 ≈ 2 % of the K2 error at n = 64 (see §7.3 for the emergent part) |

Reading: (a) ignoring κ3 entirely costs the whole 1.07e-4 (n = 128) → scaled n^{-2.76}: ≈ 3.5e-7 at n = 1024, L = 8, more at L = 16; (b) ignoring κ4 costs ≈ 2 % of that *before* the κ3→κ4 emergent channel is counted (with it, 40 % of the n = 64 residual); (c) ignoring the indirect shifts while keeping the exact direct D3 keeps only 21 % of the κ3 value — **an estimator that is exact for the final layer but drops intermediate shifts cannot work**; (d) pair-born terms are worth 32 % and hub-born 10 % of the κ3 effect; everything beyond the order-2 star is < 0.2 %.

### 2.2 Age truncation (n = 128, L = 8; age a = L − birth layer, so age 1 = κ3 born at h_{L−1})

MSE vs MC (share of the κ3 effect kept in parentheses):

| a | ages ≤ a in both channels | direct ALL ages (adjoint) + indirect ages ≤ a | direct ages ≤ a + indirect ALL ages |
|---|---|---|---|
| 1 | 8.55e-5 (30 %) | 6.28e-5 (51 %) | 2.17e-5 (90 %) |
| 2 | 3.07e-5 (81 %) | 2.42e-5 (87 %) | 1.45e-5 (96.5 %) |
| 3 | 2.00e-5 (91 %) | 1.77e-5 (93.5 %) | 1.27e-5 (98.2 %) |
| 4 | 1.40e-5 (97 %) | 1.30e-5 (98 %) | 1.115e-5 (99.6 %) |
| 5 | 1.148e-5 (99.3 %) | 1.104e-5 (99.7 %) | 1.091e-5 (99.8 %) |
| all | 1.073e-5 | | |

Same at n = 64 (both channels ≤ a: 3.87e-4, 2.92e-4, 1.52e-4, 8.3e-5, 7.6e-5 for a = 1..5; direct-all+indirect≤a: 2.99e-4, 2.30e-4, 1.18e-4, 7.3e-5; direct≤a+indirect-all: 1.17e-4, 1.01e-4, 1.06e-4, 9.1e-5).

**Fraction of old-age value that is intermediate coupling**: going from a = 2 to all ages removes 2.0e-5; making only the direct term exact for all ages removes 0.65e-5 (32 %), making only the indirect term exact removes 1.62e-5 (81 %). So ≈ 70–80 % of the value of lineages older than 2 layers is their coupling into intermediate μ_l, not the final D3. The exact direct term (adjoint) on top of indirect-exact-to-age-a buys 27 % / 21 % / 12 % / 7 % / 4 % of the remaining MSE for a = 1..5 — ≤ 12 % at a = 3–4 and nothing once ≥ 5 ages are exact in the indirect channel. At L = 16 the gates are more saturated (c ≈ 0.9), lineages decay more slowly (per-layer leg norm factor 2E[g²] ≈ 0.9 vs ≈ 0.6–0.8 at L = 8), so these age thresholds move up by roughly a factor 1.5–2 (not measured; the L = 16 MC did not finish before the container restart).

Direct per-birth contributions to the final MSE (n = 128, c3·d3/σ² norm²): recorded in results/abl_n128_L8_s0.log run; the direct term of the oldest births is ≈ 1/10 of the youngest.

---

## 3. Forward lineages vs backward adjoint

### 3.1 Born κ3 of h = ReLU(z), z ~ N(μ, C), by index pattern (exact)

Expand h_i = m_i + Σ_{p≥1} A_p(i) He_p(ẑ_i)/p!. Using E[He_p(x)He_q(y)He_r(w)] = Σ_{a+b=p,a+c=q,b+c=r} p!q!r!/(a!b!c!) ρ_xy^a ρ_xw^b ρ_yw^c (verified by MC):

    κ3(h)_ijk  (all distinct) = Σ_{a,b,c≥0; a+b,a+c,b+c ≥1} A_{a+b}(i) A_{a+c}(j) A_{b+c}(k) ρ_ij^a ρ_ik^b ρ_jk^c /(a! b! c!)
                              = [hub/star, O(ρ²)]  A_2(i)A_1(j)A_1(k) ρ_ij ρ_ik + (two permutations)
                                + [triangle, O(ρ³)] A_2A_2A_2 ρ_ij ρ_jk ρ_ki
                                + [order-3 stars] ½ A_3(i)A_2(j)A_1(k) ρ_ij² ρ_ik + …  (6 perms) + O(ρ⁴)
    κ3(h)_iik  (pair)         = Σ_{p≥1} B_p(i) A_p(k) ρ_ik^p / p!   = B_1(i)A_1(k) ρ_ik + O(ρ²),   B_1 = 2σm(1−Φ), A_1 = σΦ
    κ3(h)_iii  (diagonal)     = closed form (third central moment of ReLU(N(μ,σ²)))

(A_1 = σΦ = "a1", A_2 = σφ = "a2", A_3 = −σαφ: the task's a2_i a1_j C_ij / (a1 a1 a1 C C C) language maps onto this with ρ = C/(σσ); the "hub" term is the (2,1,1) star A_2(i)A_1(j)A_1(k)ρ_ijρ_ik.)
There is no pure "(1,1,1) A_1A_1A_1 C C C" term: a term linear in all three A_1 would need p=q=r=1, and E[He_1He_1He_1] = 0.

The brute-force n³ tensor (`born_kappa3_full`, triple series to total order 6, pair series to p = 40, exact diagonal) reproduces MC κ3(z_2) at n = 64 to MC noise (5e-4 on entries of size 0.3).

### 3.2 Transport and the adjoint identity

Under gate linearization M_l = diag(g_l) W_l, κ3(z_L) = Σ_l (P_l ⊗ P_l ⊗ P_l) κ3born(h_l), P_l = W_L G_{L−1} W_{L−1} ⋯ G_{l+1} W_{l+1}, and the readout needs only the diagonal
D3_a = Σ_l Σ_{ijk} P_l[a,i]P_l[a,j]P_l[a,k] κ3born(h_l)_ijk.
Let V^{(l)} = P_lᵀ (column a = legs of all h_l neurons into output a). Then

    V^{(L−1)} = W_Lᵀ,     V^{(l)} = W_{l+1}ᵀ G_{l+1} V^{(l+1)}        (one n³ matmul per layer, L−2 in total)

and per birth layer, with V = V^{(l)}:

    diag born   D3_a += Σ_i d_i V_ia³                                              O(n²)
    pair born   D3_a += 3 Σ_{i≠k} P_ik V_ia² V_ka  = 3 colsum_a[(V∘V) ∘ (P V)]      1 matmul (P V)
    hub born    D3_a += 3 Σ_i a2_i V_ia [ (ρ̃ (a1∘V))_ia² − ((ρ̃∘ρ̃)(a1²∘V∘V))_ia ]   2 matmuls  (ρ̃ = ρ with zero diagonal; second term removes j=k)
    order-3 stars: +2 matmuls (ρ̃^{∘2}(a2∘V), ρ̃^{∘3}(a1a2∘V²))
    triangle    Σ_{ijk} w_iw_jw_k ρ_ijρ_jkρ_ki = tr((D_a ρ̃)³) per a                 O(n⁴) — NOT contractible in n³

Rule: every *tree* (star) diagram is contractible with one matmul per edge-multiplicity class; every diagram with a *cycle* (triangle) is not.
Verification (`adjoint.py`, n = 48, L = 6): forward lineages = backward adjoint = full n³ tensor tower, max |Δ| = 4e-16 on D3 of size 0.14. The triangle term is 1/30 of the hub term at n = 48 and scales as n^{-1/2} relative → < 1 % of the hub (≈0.3 % of D3) at n = 1024: drop it.
Order-3 stars: 3.5 % of D3 at n = 48 (→ ≈ 0.8 % at n = 1024, i.e. ≈1e-6 RMS: keep only if cheap; 2 matmuls per birth).

### 3.3 Matmul / GF counts, n = 1024, L = 16  (unit = one n³-class matmul = 2n³ = 2.147 GF; Strassen d = 7 → ×0.39 = 0.843 GF; floor 220 GF = 102 units at d = 0, 261 units at d = 7)

| scheme | units | GF (d=0) | GF (d=7) |
|---|---:|---:|---:|
| (i) exact K2 chain, gemm (C₁ = W₁W₁ᵀ, then W Cov Wᵀ ×15) | 31 | 66.6 | 26.1 |
| (i') K2 chain with chol+trmm+syrk (if the counter honours triangular/symmetric kernels) | 18 | 38.7 | 15.2 |
| (ii) forward lineages, diag legs, all ages (gives direct + intermediate diagonals) | 105 | 225.5 | 88.5 |
| (ii') + pair & hub readout at the end | 150 | 322.1 | 126.5 |
| (ii'') + pair matrices at every layer (indirect Cov shifts, diag-born only) | 255 | 547.6 | 215.0 |
| (iii) backward adjoint, direct: diag+pair+hub | 59 | 126.7 | 49.8 |
| (iii') backward adjoint, direct: diag+pair only | 29 | 62.3 | 24.5 |
| (iii'') + order-3 stars | 89 | 191.1 | 75.1 |
| (iv) hybrid(a): adjoint direct + young legs (indirect mean, age ≤ a) | 59 + Σ_l min(a, L−1−l) | | |
| a = 1 / 2 / 3 / 4 / 5 | 73 / 86 / 98 / 109 / 119 | 157 / 185 / 211 / 234 / 256 | 62 / 73 / 83 / 92 / 100 |
| (iv') hybrid(a) + pair matrices for indirect Cov (diag-born), age ≤ a | 87 / 113 / 137 / 159 / 179 | 187 / 243 / 294 / 341 / 384 | 73 / 95 / 116 / 134 / 151 |

All numbers exclude the K2 chain (add 31 / 18 units). Which hybrid is needed is decided by the age tables in §2.

---

## 4. Intermediate shifts (indirect channel)

What is needed at layer l: δm_i = c3(α_i) κ3(z_l)_iii/σ_i² (+ c4 κ4_iiii/σ³) — the **diagonal** of the total κ3 — and δCov(h)_ij = ½[κ3_iij φ_iΦ_j/σ_i + (i↔j)] + ¼κ4_iijj φ_iφ_j/(σ_iσ_j), δVar_i = v3 κ3_iii + v4 κ4_iiii — the **pair matrix**. The mean channel carries ≈ 60 % of the κ3 effect, the cov channel ≈ 30 % (only in combination with the mean channel: §2.1).

Cost: the diagonal of a transported lineage at layer l needs its legs P^{(l)}_b (n² per birth per layer given the legs, but the legs themselves are the L²/2 matmuls); the pair matrix needs one extra matmul per (birth, layer) for diag-born ((U∘U)diag(d)Uᵀ) and ≈ 3 for pair-born lineages. There is no exact shortcut: the cubic diagonal Σ_m d_m (P^{(l)}_b)_am³ is not obtainable from the backward adjoint (which only gives P^{(L)}_b), not sketchable (§5), and not mean-field — it is a zero-mean random-sign sum, so its regression on (μ_a, σ_a, μ_aσ_a², …) has R² ≈ 0 except for the coherent template part (§7). What *is* cheap: the κ4 diagonal and the κ4 coherent pair level (mean-field, O(n²), §7.3), and the scale-template part of κ3 (§7).

Practical rule from §2.2: carry exact young legs for ages ≤ 3–4 (indirect mean + cov), drop older lineages from the indirect channel, and do **not** pay for the adjoint direct term of old ages unless the young-leg budget is already ≥ 4 ages.

## 5. Sketching (randomized estimators)

Every contraction in this problem that is expensive is a *random-sign* sum: D3_a = Σ_m d_m V_ma³ is a sum of n terms of size n^{-3/2} whose total is only √n·n^{-3/2}; the same holds for (ρ̃x)_i inside the hub term and for the intermediate cubic diagonals Σ_m d_m (P^{(l)}_b)_am³.
For such sums an unbiased estimator from a random subset / sketch of size k has

    Var(estimate) / value² ≈ (n/k − 1) · O(1)      (subset sampling),      ≈ (n/k)·O(1) (Gaussian two-sketch products)

because the estimator's noise is set by the un-cancelled magnitude √n·term·√(n/k) while the signal is √n·term. Measured (`sketch.py`, oldest birth, direct term):

| n | k | subset sampling, rel. MSE | two-sketch hub, rel. MSE |
|---|---|---|---|
| 64 | 16 / 32 / 64 | 0.77 / 0.30 / 0 | 5.1 / 1.5 / 0.69 |
| 128 | 32 / 64 / 128 | 0.74 / 0.24 / 0 | 2.6 / 0.86 / 0.32 |

To bring a 5e-5-RMS term's error below 1e-5 (MSE 1e-10) needs relative MSE ≲ 0.04, i.e. k ≳ 25 n for subsets and k ≫ n for sketches — more expensive than the exact matmul. **There is no sketch size with variance < 1e-9 at < 20 GF; sketching is a dead end for κ3 contractions.** (Sketches only pay for *coherent* quantities — traces of positive matrices, κ4 pair sums — but those are already cheap by mean-field/template formulas, §4/§7.)
Hutchinson-type trace estimates are likewise not useful: the K2 chain needs the full C_{l+1} (all entries enter the next Cov(h)), not a trace.

---

## 6. Other exact restructurings

* **syrk/trmm for the K2 chain**: C_{l+1} = (W L)(W L)ᵀ with L = chol(Cov h): n³/3 + n³ + n³ flops = 1.17 units instead of 2 per layer → 31 → 18 units (−28 GF at d = 0, −11 GF at d = 7). Only if the FLOP counter honours the triangular/symmetric kernels (as it honours Strassen); the lineage/adjoint matmuls have no such symmetry.
* **Hermite split of Cov(h)**: Cov(h) = diag(V − A₁²) + diag(A₁/σ) C diag(A₁/σ) + R, R = O(ρ²) entries. The R term is needed (its contraction into the next diagonal is a coherent O(1/n) term), so no matmul is saved; but it shows the K2 chain's exact off-diagonal content is just C scaled by the gate (A₁/σ = g) plus the ρ² Hadamard term — useful for the worker's implementation (no bivariate normal CDF needed: pair series to p ≈ 6 is exact to 1e-12 for |ρ| ≤ 0.2).
* **Hutchinson / trace tricks**: nothing in the chain is a trace; not applicable.
* **Low-rank / Cholesky corrections**: Cov(h) − diag is full rank with O(n^{-1/2}) entries; no.
* **Mean-field priors with net-specific corrections**: κ4 diagonal is coherent and self-averaging (Σ_i W_ai⁴ κ_i = 12κ̄/n ± 10 % at n = 1024, plus the coherent pair sum); the κ4 channel can be run as *scalar level P_l + exact diagonal* (1 matmul per layer or O(n²) mean-field, implemented as `k4mode='scalar'` in `tower.py`) — but it must include the κ3→κ4 response (§7.3) or it is useless (n = 64: scalar κ4 without it gives no gain; full κ4 with it gives −38 %).
* **Scale-mode projection (§7)**: zero-FLOP removal of the radial templates at every layer is exact for the final mean (E[s] = 1 invariance); it only helps a truncated chain, and the test with a per-layer fitted template was score-negative in the sandbox (see §7.2) — the clean way is the sphere formulation.

## 7. Radial mode / homogeneity

### 7.1 Exact statements

(a) Degree-1 homogeneity (no biases): h_L(r u) = r h_L(u) for r > 0. With x = r u, r = ‖x‖ ~ χ_n independent of u uniform on S^{n−1}:

    E_G[h_L(x)] = E[r] · E_S[h_L(u)] = e₁(n) · E_S[h_L(√n u)],     e₁(n) = E[χ_n]/√n = √(2/n) Γ((n+1)/2)/Γ(n/2) = 1 − 1/(4n) − 1/(32n²) + …

(n = 1024: e₁ = 1 − 2.44e-4; n = 64: 1 − 3.93e-3.) The sphere problem (input g̃ = √n u, ‖g̃‖² = n) has exact first-layer cumulants
μ_1 = 0, C_1 = W₁W₁ᵀ, κ3(z_1) = 0 (all odd cumulants vanish), κ4(z_1) = −(2/(n+2))·3·Sym(C₁⊗C₁) = −(2/(n+2))(C_ijC_kl + C_ikC_jl + C_ilC_jk), κ6 = O(n^{-2}).

(b) Scale mixture. Write x = s·g̃ with s = r/√n ⟂ g̃. Then h_l(x) = s h_l(g̃) for every l and the raw moments factorize, E[(s h)^{⊗k}] = E[s^k] E[h^{⊗k}]. Converting to cumulants (exact algebra, verified numerically on a toy in the sandbox):

    m_G   = e₁ m_S
    Cov_G = E[s²](C_S + m_Sm_Sᵀ) − e₁² m_Sm_Sᵀ
    κ3_G  = E[s³] κ3_S + 3(E[s³] − e₁E[s²]) Sym(m_S⊗C_S) + (E[s³] − 3e₁E[s²] + 2e₁³) m_S⊗m_S⊗m_S
    κ4_G  = E[s⁴]R4 − 4Sym(E[s³]R3 ⊗ e₁m) − 3Sym(E[s²]²R2⊗R2) + 12Sym(E[s²]R2⊗e₁²mm) − 6e₁⁴ m⊗⁴   (R_k = raw moments of h_S)

For the χ_n radius: E[s²] = 1, E[s³] = e₁(1+1/n), E[s⁴] = 1+2/n, hence v := Var s = 1 − e₁² = 1/(2n) + O(n^{-2}), κ3(s) = κ4(s) = O(n^{-2}). To first order in v with E[s] = 1:

    Cov_G = (1+v) C_S + v m mᵀ
    κ3_G  = (1+3v) κ3_S + 6v · Sym(m⊗C)                       [coefficient 6v = 3/n, not 3v]
    κ4_G  = (1+6v) κ4_S + 12v · Sym(C⊗C) + 12v · Sym(κ3⊗m)   [NO Sym(m⊗m⊗C) term at O(v); toy-MC fit of the coefficients: 12.5, 12.7, 1.1±1 (×v) for (CC, κ3m, mmC)]

Consistency: at layer 1, κ3_G(h_1) has zero template content (its pair entries are O(ρ) random-sign Hermite terms), so κ3_S(h_1) = −6v Sym(m⊗C)/(1+3v): the sphere's negative κ4 face generates exactly this negative coherent κ3 through the κ4→κ3 response ½T2_i(φ_j/σ_j)κ4_iijj (checked analytically at α = 0: T2σ² = V, σφ = m). Likewise κ4_G(z_1) = κ4_S(z_1) + 12v Sym(C⊗C) = −(6/(n+2))Sym(CC) + (6/n)Sym(CC) = O(n^{-2}) ✓.

So the *exact coherent radial content of the Gaussian problem at every layer* is, in closed form from the layer's own (m_l, C_l):

    Δκ3_l = 6v Sym(m_l⊗C_l) = (3/n) Sym(m_l⊗C_l),     Δκ4_l = (6/n) Sym(C_l⊗C_l) + (6/n) Sym(κ3_l⊗m_l),     ΔCov_l = (1/2n) m_l m_lᵀ,

with the SAME v at every layer and e₁ carrying the whole effect on the final mean (E_G = e₁E_S). These are identities between the two formulations, not extra terms: an exact Gaussian-formulation tower already contains them (they are hidden inside the transported Hermite-born terms). They become a *correction* only for a truncated chain that loses them.

(c) The readout footprint of the κ3 template is coherent with α: D3_a ⊃ (3/n) μ_a σ_a², so δE_a = c3·(3/n)μ_a = −α_a²φ(α_a)σ_a/(2n), RMS ≈ 0.05σ/n ≈ 3e-5 at n = 1024 (MSE ≈ 1e-9 direct alone, before the indirect channel).

### 7.2 What "sphere formulation" means operationally (as implemented: `run_tower(..., sphere=True, k4=True)` = full face; `sphere=True, k4face=True` = cheap carrier)

1. Input: u uniform on the sphere of radius √n ⇒ μ_1 = 0, **C_1 = W₁W₁ᵀ unchanged** (same covariance scale as Gaussian), κ3(z_1) = 0, and the initial κ4 face
   κ4(z_1) = λ₁·Sym(C₁⊗C₁),  **λ₁ = −6/(n+2)** (negative; i.e. κ4_iiii = −6σ_i⁴/(n+2), κ4_iijj = −(2/(n+2))(σ_i²σ_j² + 2C_ij²), κ4_iiij = −(6/(n+2))σ_i²C_ij).
2. At every ReLU the face is treated like any κ4 (all first-order responses of §1): mean shift c4κ4_iiii/σ³, δVar = v4κ4_iiii, δCov_ij = ¼κ4_iijj φ_iφ_j/(σ_iσ_j) − (1/6)(κ4_iiij α_iφ_iΦ_j/σ_i² + sym), and the κ4→κ3 births κ3(h)_iij += ½T2_i(φ_j/σ_j)κ4_iijj + (B₃(i)/6σ_i³)g_jκ4_iiij − (B₁(i)/6σ_i)(α_jφ_j/σ_j²)κ4_ijjj, κ3(h)_iii += (C₄ − 3VA₄)/(24σ⁴)·κ4_iiii, κ3(h)_ijk += ½(φ_i/σ_i)g_jg_k κ4_iijk + perms.
3. Transport of the face: full-tensor version = pattern-aware factors (T4 on the diagonal, T2_iT2_j on (iijj), T2_ig_jg_k on (iijk), T31_ig_j on (iiij), g⁴ generic) then W⊗4. Cheap carrier: keep one matrix C̃_l with C̃_1 = C_1 and
   **C̃_{l+1} = W_{l+1} diag(√T2_l) C̃_l diag(√T2_l) W_{l+1}ᵀ**, face_l = λ₁ Sym(C̃_l⊗C̃_l) (2 matmuls/layer; the √T2 — not g — scaling is what reproduces the dominant (iijj) channel; with g it reproduces the gate-linearized full tensor to 1e-16 but under-propagates the face by ≈ 2× per layer).
4. Final: **y = e₁(n) · (Edgeworth readout of the sphere chain)**, e₁ = E[χ_n]/√n = 1 − 1/(4n) − … (n = 1024: 1 − 2.44e-4 — this factor alone is 1.2e-4 absolute on y ≈ 0.5, i.e. the whole error budget: a sphere chain without it, or a Gaussian chain with it, is ruined).
Everything else (exact K2 chain, κ3 lineages/adjoint, Gaussian-born terms computed from the chain's own μ_l, C_l which now include the face-induced shifts) is unchanged.

What a chain that already fits a per-layer scale-mixture κ4 face (λ-law) must change: (i) the *input* face is negative and known, λ₁ = −6/(n+2) ≈ −0.00585 at n = 1024 — in the Gaussian formulation the corresponding content is +12v Sym(CC) = +6/n hidden in the Hermite-born κ4; a fitted positive v_eff ≈ 0.006–0.0095 is therefore NOT the input radius but the **emergent** mode of §7.3 plus whatever the chain's diag κ4 legs miss; (ii) in the sphere formulation the fitted λ must be re-fitted (the emergent part stays, the input part flips sign and is exact); (iii) the face must be transported with √T2 gates and must feed the κ4→κ3 births and δCov (otherwise the ½T2(φ/σ)κ4_iijj coherent κ3 and the ¼φφκ4_iijj covariance pieces — which are the same size as the κ4 readout — are lost); (iv) double counting: a λ-face projected onto Sym(C⊗C) over pairs picks up the *diagonal* born κ4 of earlier layers (projection coefficient ≈ 12κ̄4/(nV̄²) ≈ 6/n from the (iiii) entries alone) — if the chain also carries diag κ4 legs, fit the face on pairs with the diagonal excluded, or subtract Σ_i W_ai⁴κ_i-type content before fitting.

### 7.3 The emergent scale mode (κ3→κ4 coupling) — the large effect

MC at n = 64, L = 8 shows κ4(z_l)_aaaa growing with depth (rms 0.37, 0.42, 0.63, 0.94, 1.70, 0.87, 0.99 for l = 2..8) and a coherent pair level ⟨κ4_aabb⟩ growing from 0.11 to 0.23, while the κ4 tower *without* κ3→κ4 coupling decays (0.35 → 0.18; pair level → 0.003). Adding the first-order response of κ4(h) to κ3(z),

    κ4(h)_iijj += 2 X2_i T2_j κ3(z)_ijj + 2 T2_i X2_j κ3(z)_iij,      κ4(h)_iiii += T43 κ3(z)_iii,   T43 = (D₃ − 4κ3(h)A₃ − 6V B₃)/(6σ³),

(X2 = m(1−Φ), T2 = Φ − (m/σ)φ; verified: they are exactly the pieces that make the scale-mixture identities Cov_G = C_S + v mmᵀ, κ4 ⊃ 12v Sym(CC) close at every α) reproduces the growth (pair level 0.16 at l = 8; diag rms 0.64 vs 0.99 — the rest is second order at n = 64) and **cuts the full-tower residual from 8.16e-5 to 5.03e-5 (−38 %)**. Interpretation: every layer injects relative variance of ‖h_l‖² of order Var(h²)/(n E²[h²]) ≈ 5/n (early) … 1/n (deep); by homogeneity this acts on the next layer as a global scale with gain exactly 1, so the effective scale variance grows linearly with depth, v_eff(l) ≈ 1/(2n) + ≈0.4(l−1)/n (sandbox n = 64: 0.05 at l = 8 vs input 0.008; at n = 1024, l = 16 this predicts v_eff ≈ 0.006 — the coordinator's fitted 0.006–0.0095). In cumulant language the injected scale variance sits in κ4 pairs *and* in κ3 pairs coherent with μ (κ3_ijj ∝ μ_i σ_j²) and in C; the κ3→κ4 response is how the κ3 part is converted. Gate-transported κ4 legs (W⊗4 of diag-born κ4 with g⁴) contain none of this.

Cheap carrier for this mode: because it is coherent, the κ4 pair level is a scalar per layer, P_{l+1} = (4/n²)[Σ_i ê_i + Σ_{i≠j} p̂_ij] with ê = T4 q + κ4_born_diag + T43 d3, p̂_ij = T2_iT2_j P_l + b_ij + 2X2_iT2_j κ3(z)_ijj + sym, and the diagonal q_{l+1,a} = Σ_i W_ai⁴ ê_i + 3 Σ_{i≠j} W_ai²W_aj² p̂_ij (one (W∘W)-matmul, or mean-field O(n²)). The bilinear κ3 term Σ_{i≠j} X2_iT2_j κ3(z)_ijj = Σ_births Σ_m d_m (X2·u_m)(T2·u_m²) is O(n²) per birth given legs. (`k4mode='scalar'`; at n = 64 the scalar carrier alone did not help the final MSE — the fluctuating part of the κ4 diagonal matters at n = 64 — so it must be validated at n ≥ 256 before use.)

### 7.4 Sphere vs Gaussian formulation: n-scaling of the gain (same chain, same MC truth, L = 8, seed 0)

| chain | n = 64 | n = 128 | n = 256 |
|---|---|---|---|
| full tower, κ3 only (no κ4): gauss → sphere(face carrier √T2) | 9.56e-5 → 7.89e-5 (**0.825**) | 1.073e-5 → 9.44e-6 (**0.880**) | 4.72e-6 → 4.29e-6 (**0.909**) |
| variant C (diag+pair+hub, gate), κ3 only: gauss → sphere(carrier) | 9.42e-5 → 7.75e-5 (**0.823**) | 1.136e-5 → 1.023e-5 (**0.900**) | 5.84e-6 → 5.21e-6 (**0.892**) |
| full tower WITH exact κ4 tensor (+κ3→κ4): gauss → sphere(full face) | 5.03e-5 → 3.71e-5 (**0.736**); gauss+carrier 4.27e-5 (0.85) | n/a (n⁴ tensor) | n/a |
| variant C WITH exact κ4: gauss → sphere(full face) | 5.72e-5 → 4.71e-5 (0.824) | n/a | n/a |
| scalar-κ4 chain: gauss → sphere(carrier) | 9.26e-5 → 8.44e-5 (0.911) | | |
| (earlier run, κ4 without κ3→κ4) full: 8.16e-5 → 6.14e-5 full face (0.75), 6.39e-5 carrier (0.78); A/B/C/D variants 0.90/0.84/0.78/0.83 | | | |

The sphere gain for the κ3-only chains falls from 17.5 % (n = 64) to 12 % (n = 128) to 9 % (n = 256) for the full κ3 tower (17.7 % → 10 % → 10.8 % for variant C); per doubling of n the gain shrinks by ≈ 1.3–1.5×, i.e. ≈ n^{-0.45}. Both formulations are first-order exact; the difference is second-order (radial × κ) terms that the Gaussian formulation must represent perturbatively and the sphere formulation does not have, so the gain tracks the second-order share of the residual, ≈ n^{-1/2}. **Extrapolation to n = 1024: ≈ 4–6 % of the residual MSE for a complete first-order chain, zero FLOPs** (the e₁ factor and λ₁ are closed form, the carrier is 2 matmuls/layer and can share the K2 chain's W-products). For a *truncated* chain (lossy old lineages, κ4 by gated legs) the gain is larger because the sphere formulation removes the input part of the scale mode that such a chain mis-propagates, but the emergent mode (§7.3, 3–10× larger at L = 16) is not removed by the sphere formulation — it has to be carried (scalar P_l + κ3→κ4 response) or absorbed in a λ-law face fitted *after* switching to sphere inputs.

### 7.5 Template content of the exact cumulants (7c) — partial

The per-layer projection of the tower's total κ3(z_l) onto Sym(μ_l⊗C_l) is large and not radial: n·ε = 15.8 with cos² = 0.68 already at z_2 for n = 24 (and n·ε_fit of the content missing from a diag-only chain: 3.6, 6.3, 9.7, 12.1, 12.7, 12.6, 14.0 for z_2..z_8 at n = 64, vs the radial 6v·n = 3). This coherent content is produced by the diag-born lineages themselves (κ3(z)_aab = Σ_i d_i W_ai²W_bi correlates with μ_b = Σ_j W_bj m_j through W_bi² m_i: analytic estimate n·ε ≈ 18 V̄⟨dm⟩/(⟨m²⟩⟨V²⟩) ≈ 10–25) and exists on the sphere too, so a fitted ε_l Sym(m⊗C) inserted as a born term double-counts it: in the sandbox the oracle-fitted template made every chain worse (A: 2.9e-4 → 3.0e-3; B: 1.4e-4 → 5.4e-4) and the pure radius template (6v) helped only the diag-only chain by 12 % and hurt all others. The MC-based projections (`radial.py`, needs the per-layer MC pair cumulants, which exist in the npz files) were not completed before the container restart; the script is ready.

---

## 8. Implications for a 1024×16 estimator at the 220 GF floor — falsifiable predictions

Budget: 102 n³-matmul units at d = 0, 261 at d = 7 (unit = 2.147 / 0.843 GF). K2 chain costs 31 (18 with syrk/trmm).

P1. **Direct-only is not enough.** An estimator with the exact K2 chain + an exact adjoint for the final-layer D3 (all ages, diag+pair+hub, 59 units) but no intermediate shifts recovers only ≈ 20 % of the κ3 MSE (sandbox 20 % at n = 64, 21 % at n = 128, L = 8). Prediction: at 1024×16 its MSE stays within 20–25 % of the K2-only MSE.

P2. **Hybrid(a) value table** (fractions of the κ3 effect recovered, from §2.2; L = 16 thresholds ≈ 1.5–2× higher in age): indirect+direct ages ≤ 2: 81 %; ≤ 3: 91 %; ≤ 4: 97 %; ≤ 5: 99 %. Adding the adjoint direct term for all older ages on top: +6 % / +2.5 % / +1 % / +0.4 %. Cost (units, excl. K2): young legs mean-only 15a − a(a+1)/2 (a = 3: 39; a = 4: 50; a = 5: 60); + pair matrices for cov shifts ≈ ×2; adjoint direct 59 (29 without hub). Prediction: hybrid with a = 3 young legs (mean + cov) and NO adjoint keeps ≈ 91 % (a = 4: 97 %) of the forward-lineage accuracy at 109 units = 234 GF (d = 0) / 92 GF (d = 7) including the K2 chain (a = 4: 131 units = 281 / 110 GF); the adjoint on top of a ≥ 4 changes the MSE by < 2 %, so at d = 0 spend the budget on young legs, not on the adjoint. The forward-lineage all-ages scheme (255 units with cov shifts) is affordable only at d = 7 (215 GF).

P3. **Born patterns**: pair-born lineages are worth ≈ 30 % and hub-born ≈ 10 % of the κ3 effect; order-3 stars and the triangle are < 1 % at n = 1024 (triangle ≈ n^{-1/2} × hub). Prediction: dropping hub costs ≈ 10 % of the κ3 MSE (≈ 1e-9 if the κ3 effect is ≈ 1e-8 at 1024×16).

P4. **Gate transport of κ3 legs is fine** (0.4 % of the κ3 effect at n = 128, → ≈ 0.05 % at 1024); **gate transport of κ4 is not**: the coherent (iijj) channel needs T2_iT2_j (≈ 2× g_i²g_j²) and κ4 legs must be √T2-scaled. Prediction: a κ4 lineage carried with g⁴ underestimates κ4(z_L) by ≈ 2^{L−l} and the fitted λ-law face is compensating for exactly this plus the κ3→κ4 emergent mode.

P5. **κ3→κ4 emergent scale mode** (§7.3): including κ4(h)_iijj += 2X2_iT2_jκ3(z)_ijj (+sym) and the diagonal T43 term in the κ4 channel reduced the n = 64 residual by 38 %. Prediction at 1024×16: the κ4 coherent level v_eff(l) ≈ 1/(2n) + 0.4(l−1)/n (≈ 0.006 at l = 16), i.e. the fitted λ-law values; a chain that builds this from the carried κ3 pair content (O(n²) per birth given legs, plus one (W∘W) matmul per layer for the exact diagonal) should need no fitted λ, and its κ4 readout+shift contribution is ≈ 2e-4·(1/16) ≈ 1–2e-5 RMS at n = 1024 (MSE 1–4e-10, i.e. at the floor level — but its *shift* channel through μ is larger, untested at n ≥ 128).

P6. **Sphere formulation** (§7.2/7.4): same chain, zero extra FLOPs beyond the 2-matmul/layer carrier (or none if the face is folded into the κ4 scalar channel): −17.5 % / −12 % / −9 % of the residual at n = 64/128/256 → predicted **−4 to −6 % at n = 1024** for a complete first-order chain, more for a truncated chain. Mandatory ingredients: C₁ = W₁W₁ᵀ unchanged, λ₁ = −6/(n+2), √T2 transport, κ4→κ3 and κ4→Cov responses, final ×e₁(n) = 1 − 2.44e-4. Prediction: omitting e₁ costs ≈ 1.5e-8 MSE (y ≈ 0.5 × 2.4e-4)², i.e. more than the whole leader gap — this is the easiest thing to get wrong.

P7. **Sketching**: no randomized estimator of any κ3 contraction reaches relative variance < 0.05 below k ≈ 25n; prediction: any sketch-based lineage compression will show variance ≥ its systematic gain at any k < 20 GF (§5).

P8. **Second order matters at the 1e-8 level?** The complete first-order tower's residual falls like n^{-2.9} from 64 to 128 (K2 error n^{-2.76}); extrapolated, the first-order residual at 1024×8 is ≈ 2–3e-8 — above the leader's 1.1e-8. If this holds at L = 16, the leader's error floor is set by second-order terms (κ3², κ3κ4 responses, O(ρ) transport corrections), not by lineage truncation — the sphere formulation removes part of it for free. Falsifiable by running the float64 reference with all first-order channels at n = 1024.

## 9. Files / how to port

`relu_hermite.py`: all closed forms (A_p, B_p, C_p, T2, X2, T3, T4, T43, c3, c4, c33, v3, v4, pair series, triple-Hermite born κ3, κ4 patterns, e₁). `tower.py`: `run_tower(W, opts)` — full-tensor reference (options: k4, transport='gate'|'exact', born_* flags, distinct_maxorder, indirect_mean/cov, direct_max_age, indirect_max_age, sphere, k4face/face_gate/face_lam, k4mode='scalar', k3_to_k4, template_eps, merge_births). `adjoint.py`: `k2_chain`, `backward_adjoint`, `forward_lineages`, `contract_birth` — the production kernels to port to torch (V recursion + 3 contractions per birth). `mc.py`: MC truth with per-layer κ3/κ4 pair accumulators (npz files in results/). `experiments.py`, `decompose.py`, `template_exp.py`, `sphere_scaling.py`, `radial.py`, `sketch.py`, `costs.py`: the tables. Not finished before the deadline: L = 16 decompositions (`decompose.py 128 16 0`, ≈ 10 min), MC-based template projections (`radial.py`), n = 256 ablations (`experiments.py 256 8 0 0`, ≈ 30 min), sphere MC (`mc.py 64 8 0 1e8 sphere`).
