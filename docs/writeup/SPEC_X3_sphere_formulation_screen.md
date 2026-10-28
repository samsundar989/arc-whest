# Worker spec X3 — sphere (radial-homogeneity) formulation: one matched-refit screen

**Status:** GO as a single matched-refit screen. Sandbox n-scaling (F_math.md §7.4): gauss→sphere MSE ratio 0.825 / 0.880 / 0.909 at n = 64 / 128 / 256 (≈ n^−0.45) ⇒ ≈ 4–6 % at n = 1024 for a complete first-order chain; expect 2–4 % on ours after the λ refit. Run it in whichever lane frees a GPU first (after X1's first VAL read, or after X2's billing), before ⟨4,4,4;48⟩.
**Owner:** lane 2 after the X2 ablations are billed, or lane 1 after X1's first VAL read; ≈ 2 h GPU (one 112-update matched refit pair) + 1 h build.
**Why:** with no biases the network is positively homogeneous, so E_G[h_L(x)] = E[‖x‖]·E_S[h_L(u)] exactly (u uniform on the sphere). On the sphere there is no radial mode, so every coherent "scale-mixture" cumulant the Gaussian formulation generates at every layer (κ4 ∝ Sym(C⊗C), κ3 ∝ Sym(m⊗C), κ6, …) is absent; the only price is a small, exactly known negative κ4 face at the input (λ₁ = −6/(n+2)) and the exact factor e₁(n) = E[‖x‖]/√n on the final mean. In the sandbox (L = 8) this cut the κ3 towers' MSE by 17.5 / 12 / 9 % at n = 64 / 128 / 256 and the full-K4 tower's by 26 % at n = 64; our own full-support f64 reference at n = 1024 (3386) measured −33 % correctors-off / −10.9 % vs the corrected chain for a data-driven extraction of the same mode; the pure χ treatment (3394) gave −10 % vs untreated. The fitted λ-law κ4 face absorbs part of the mode today, which is why unrefit tests looked bad: **this screen must be a matched refit.**

## 0. Pre-registered prediction and kill line
- Prediction (strategist): VAL-100 paired after matched refit: **−2 % raw (80 %: +1 … −5 %)**; P(≤ −2 %) 0.4.
- Keep if ≤ −2.0 % raw with CI excluding 0; then FRESH-A. Otherwise record the number and close the item.

## 1. The change (three constants + one fixed term; zero control byte-identical)
1. **Input unchanged:** C₁ = W₁W₁ᵀ (the sphere of radius √n has the same second moment); μ₁ = 0; κ3(z₁) = 0.
2. **Initial κ4 face, fixed and negative:** κ4(z₁) = λ₁·Sym(C₁⊗C₁) with **λ₁ = −6/(n+2)** (n = 1024: −5.848e-3), i.e. κ4_iiii = λ₁σ_i⁴, κ4_iijj = (λ₁/3)(σ_i²σ_j² + 2C_ij²), κ4_iiij = λ₁σ_i²C_ij. In the chain's λ-law language this is a scale-mixture face on C_off with a *known negative input coefficient*; the fitted per-layer λ must be refit on top (the emergent κ3→κ4 part stays; the input part is now exact). The face must be transported with √T2 gates (T2 = Φ − αφ, the (2)-Hermite gate), not g, and must feed the κ4→κ3 births (κ3(h)_iij += ½T2_i(φ_j/σ_j)κ4_iijj) and δCov_ij = ¼κ4_iijj φ_iφ_j/(σ_iσ_j) — see F_math.md §7.2 items 2–3. Cheapest implementation: fold λ₁ into the existing λ-law face at layer 1 and let the chain's own κ4 machinery propagate it; if the chain's face rides on C_l with g-gates, add a per-layer scalar √T2/g correction factor (zero FLOPs).
3. **Final readout × e₁(n)**, e₁ = E[χ_n]/√n = √(2/n)·Γ((n+1)/2)/Γ(n/2) = 1 − 1/(4n) − 1/(32n²) + … (n = 1024: **0.99975586**). This is mandatory: omitting it is a coherent 2.4e-4 relative bias ≈ 1.5e-8 MSE.
4. Arm S1 = items 2+3 with the fitted κ4 laws (λ table, `_K4T1_*`, `_K4M_*`) and the corrector **free** in the refit; arm S0 = control with the same refit recipe; both start from the 8be export. Zero control: λ₁ = 0 and e₁ = 1 reproduces the current chain byte-for-byte.

## 2. Refit and read
- Matched CONTROL ‖ ARM, young recipe (112 updates, TRAIN900, corrector LR ×10, VAL every 28), best-VAL export. VAL-100 paired; report Δraw, wins, CI; also the refit's movement of the λ table (expected: λ falls by roughly the input-mode share, ≈ 2/n of the 0.006–0.0095 v_eff).
- Bill: unchanged to within 0.01 % (elementwise terms only).

## 3. Deliverable
`experiments/2026-10-20_x3/RESULT.md` with predictions vs outcome; one `RESULTS.md` line. If kept, fold into the R3 composition before launch, or into the Oct 21 second refit.
