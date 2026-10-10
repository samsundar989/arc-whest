# HANDOFF — Lane M5 (session 2, branch lane-m), 2026-10-10: the κ₄ D31 channel and the hub move — measure first, build the winners, hand them to P

**Read first:** `docs/research/2026-10-10_pivot_rev22.md` §1.2, §1.3, §2.3 and §6.

This handoff replaces `HANDOFF_2026-10-10_LANE_M4.md`, which was not started:
- M4.1/M4.2 (Ward test, full radial-split closure) are dropped;
- M4.4 (write-up science) is dropped, because there is no write-up until we are in the top 10;
- M4.3 becomes M5.3.

**Setup:** merge main into lane-m (merge, do not rebase) once Lane P has committed the strategist's files.

**How you work:**
- Build arms in the twin (`tools/twin/twin.py`) on lane-m.
- Hand each one to Lane P as a patch, with twin parity, VAL-100 numbers (unrefit and refit vs a matched control), interpreter F and ops.
- GPU:
  - local 5090 through gpuq for short jobs;
  - the RunPod pod belongs to Lane P's refit queue; ask P to queue your refits, which fill its gaps.

**Standing rules:**
- Predictions (yours and the strategist's) go on record before every run.
- Select on VAL-100; acceptance is FRESH-A by Lane P.
- No fitting or selection on FRESH panels.
- No external messages.
- Lane P owns main and the estimator files.
- Every refit comparison uses a same-recipe control with matched epochs; refits alone gain about 0.5 % per epoch.

**Tools:**
- `docs/research/2026-10-10_flopmodel/`: exact flopscope cost model. Price F **and ops**; each n-sized `_mm` costs about 100 ops.
- `docs/research/2026-10-10_k4x/`:
  - `verify_d31.py`: star, path and (2,2) D31 formulas against explicit 4-tensors, with Sym = average over permutations; the code's star convention is 4× that average;
  - `ward_check.py`: the radial shifts below, checked to O(v²);
  - `k4_anatomy.py`.

**Labels:**
- `cache/twin_labels/fit_000–099.npz` (built by `tools/twin/labels_cache.py`) hold **pre-activation cumulants already**:
  - D3 = κ_iii; D21 = κ_iij; D22 = κ_iijj; Q31 = κ_iiij (not normalized); dG = κ_iiii/2 under metric 2, so the true D4 = 2·dG;
  - plus μ, var and C11.
  - Do not convert them again.
- Only `cache/hf/arc-whestbench-p2-full1000-N1e9/moments/` (pre_M11/M21/M31/M22, N = 1e9) holds raw moments; for those, κ_aaab = c31 − 3·C_aa·C_ab on central moments (`tools/labels/cumulants.py`).
- Dense post_K22 for nets 0–99 is in the bakev2-bench-supp set (N = 1e8).
- FIT rows lie inside the twin's TRAIN split. That is fine for regressions and gains, but select on VAL-100.

**Notation.**
- m (metric) = 2, as in the code.
- μ_pre = Wμ, the pre-activation mean.
- w2_a, w3_a: the reading neuron's Wick coefficients, w2 = φ/σ and w3 = −α·φ/σ²; r_a = w3_a/w2_a = −α_a/σ_a (est:3963).
- c4_j: the per-lineage star weight (w3 at birth) held in `c4` (est:4457). Star-D4 also scales it by `_STAR_GAIN[ℓ]` (est:4111).

## Why

κ₄'s value in our chain sits in Q31 = κ₄(a,a,a,b):
- exact Q31 gave −15.5 % unrefit (Anatomy 2, v59);
- all κ₄ slices together gave 7.6 % unrefit and 27.9 % after refit (twin Step 2);
- the λ·C surrogate explains about R² 0.47–0.50 of the true Q31.

For blocks riding our κ₃ legs, D4 and D31 are O(n³) per block-layer. Two pieces are free and nothing in v81d carries them. Your job:
- find out which pieces explain the rest of the true Q31 (and D4), and build those;
- move the κ₄→κ₃ hub out of the lineage legs so that Lane P can drop always-on birth columns (−6.03 % F).

The Oct 11 regression read-out is the decision point on whether κ₄ can carry a push beyond the top 20.

## M5.1 — masked star-Q31 fold (today, 4–6 h)

**What it is.** In the code's convention (star-D4 = 4·Σ c4·A³·M, the centre summed over its 4 placements), the D31 readout of each lineage's star κ₄ is

Q★_ab = Σ_j c4_j·(A_aj³·M_bj + 3·M_aj·A_aj²·A_bj)

The covariance needs ½·w2_a·w1_b·[D21 + (r_a/3)·Q★]_ab.

**Arm.** Add to the D21 readout's left operands:
- LP += g★·ω_a·(r_a/3)·(A∘A∘A)·diag(c4)
- LA += g★·ω_a·r_a·(M∘A∘A)·diag(c4)

with:
- ω_a = 1{|α_a| ≤ 2};
- g★ a learned per-layer gain, initialized at `_STAR_GAIN[ℓ]` so that the star κ₄'s D4 and D31 start consistent.

**Where to put it.** Inside `_shared_contract_lean`, **after** `tw._wts`/`tw._qU` (est:755–759) and `tw._qold` (est:773) are stored. The old-tier join's range finder (est:786–826) uses those, so the fold must not steer old-tier compression.
- LA/LP span the old and recent columns, in the same column order as star-D4's concat(c4).
- The fold adds zero products, about 5kn² elementwise work and a few hundred ops.

**What else it touches.** D21 has five other consumers (pivot §2.3).
- The mask keeps 96–97 % of the intended covariance term and cuts contamination of H and d21e to 2–5 %.
- **Births still receive 0.3–0.5× of the term, about 80 % of it mismatched.** The Q13 orientation and −2·pK1·pK11 are correct physics; the (2,1) orientation gets ⅓·r_a·Φ_a·(1−Φ_a) where ⅓·φ_a/σ_a is correct.

**Measure:**
1. VAL-100 unrefit, plus the contamination diagnostics (|ΔH|, |Δd21e|, the births' share).
2. A short refit (corrector + gains + g★, lr 1e-4, 1 epoch) against a matched control.

**Predictions** (strategist): unrefit ±1 %; after the refit, −0.5 to −3 % vs the control.

**Kill:** not better than the control after the refit.

**Variant to try only if births leak badly:** ω_a = 1{|α_a| ≤ 1.5}.

## M5.2 — radial Sym(μ_pre⊗κ₃) κ₄ terms (today–Oct 11, free)

**Physics.** A random overall gain G (E G = 1, Var G = v), independent of the rest, adds to first order:
- Δκ₃ = v·[3κ₃ + 6·Sym(μ_pre⊗C)]
- Δκ₄ = v·[6κ₄ + 12·Sym(C⊗C) + 12·Sym(μ_pre⊗κ₃)]

This is checked to O(v²) by `ward_check.py`.

Our λ·C is the Sym(C⊗C) part. Nothing in v81d carries Sym(μ_pre⊗κ₃):
- the Q31 surrogate (est:598–603), D22 (est:4118) and D4 (est:4106–4111) have no κ₃ coupling;
- the closure's D21 monomials are all second order;
- only the fitted q3·α_a·α_b + q4 can absorb part of the rank-1 piece.

Our Anatomy 2 found the chain's λ term within 5–40 % of the label fit, so there is no direct sign of a missing term. The arm and M5.3 decide.

**Readouts** (pre-activation; D21_ab = κ₃(a,a,b), D21_ba = κ₃(b,b,a), D3 = κ₃(a,a,a); all from matrices the chain already computes):
- Q31_ab += 3v·(3·μ_pre,a·D21_ab + μ_pre,b·D3_a)
- D22_ab += 6v·(μ_pre,a·D21_ba + μ_pre,b·D21_ab)
- D4_a += 12v·μ_pre,a·D3_a

In the normalized units the closure uses, with α = μ_pre/σ, D21n_ab = D21_ab/(σ_a²·σ_b), γ_a = D3_a/σ_a³ and v′ = 3v:
- Q̂31 += v′·(3·α_a·D21n_ab + α_b·γ_a)
- D22n += 2v′·(α_a·D21n_ba + α_b·D21n_ab)
- D4n += 4v′·α_a·γ_a

In the covariance, the Q31 part acts as a row factor (1 − 3v·α_a²) on the first-order D21 term, plus the rank-1 term (v/2)·w3_a·D3_a·w1_b·μ_pre,b.

**Arm.**
- One learned per-layer coefficient v′_ℓ per slice (Q31, D22, D4), initialized at the physical value. The quenched gain law gives v_ℓ ≈ 0.0025 (L4), 0.0037 (L8), 0.0045 (L16), and v′ = 3v.
- Keep λ·C as it is; the refit re-balances it.

**Measure:** VAL unrefit, then a refit against the matched control (combine with M5.1 if both are alive).

**Predictions** (strategist): −0.5 to −3 % after the refit.

**Kill:** not better than the control by ≥ 0.3 %.

## M5.3 — the Q31 / D4 channel regression (today–Oct 11, GPU)

Use `tools/anatomy2.py` on v81d; all 9 patch anchors match v81d once. Add recorders at:
- est:4105: adv, c4, σ and **cov_pre** (C11 itself is formed only at est:4115);
- est:4451: K[(2,2)], hcols.

**Setup:**
- Layers ℓ = 2…15. Fit on nets 0–9, test on nets 10–19 (as Anatomy 2 did).
- Target: the true pre-activation Q31, normalized by σ_a³·σ_b, off-diagonal pairs only.
- Regress the **residual** y − S0, where S0 is the current surrogate (the Q31n hook).
- **Weight each pair by |w3_a·w1_b|·σ_a³·σ_b**, which is how Q31 enters the covariance. Unweighted fits chase dead and always-on pairs.

**Candidates, each per lineage age where applicable:**
- (a) Radial Sym(μ_pre⊗κ₃) (free): 3·μ_pre,a·D21_ab + μ_pre,b·D3_a, using the chain's D21/D3, and again using the TRUE D21/D3 to separate representation error.
- (b) Star-Q31 per age (free once folded): one GEMM per age, [A³·c4 | 3·M·A²·c4]·[M | A]ᵀ, with legs from `_full_legs_lean` split by `_lin_widths`.
- (c) Path-Q31: J = M∘w2b, V = (A∘J)·C̃⁽ˢ⁾ on that lineage's hcols. Then [(J∘V)·Aᵀ + (A∘V)·Jᵀ] times the code-convention factor; check it against `verify_d31.py` at n = 32 first.
- (d) (2,2)-block: P = M of the youngest ages, K = K22 from the birth layer, giving 3·((P∘P)·K∘P)·Pᵀ. Optionally use the true post_K22 to separate K error.
- (e) Others:
  - C̃_aa·C̃_ab (the shape of radial Sym(C⊗C));
  - (Wᵀ·diag(v)·W)_ab;
  - Puffi's modes P_off[v̄·C, C∘C, ½(D_m·D21ᵀ + D21·D_m), D3·mᵀ + m·D3ᵀ, v·vᵀ], where m here is μ_pre;
  - Sym₆(M⊗C)-type k31 = 3(M_aa·C_ab + C_aa·M_ab), with M ∈ {diag(dG), C̃, Wᵀ·diag(v)·W}.

**Repeat for D4:** target the true D4 = 2·dG residual after m·dG + star-D4. Candidates:
- 12·μ_pre,a·D3_a;
- path-D4;
- (2,2)-block D4 = 3·rowsum[((P∘P)·K)∘(P∘P)].
Weight by |w4_a|·σ_a⁴.

**Report:**
- For each candidate, and greedily combined, the out-of-sample weighted ΔR² over S0 per layer.
- A predicted Δraw from Δraw ≈ −0.155·ΔR²/(1 − R²_S0), unrefit. In the twin, refit multiplied the all-κ₄ gain by 3.7×, but treat that as an upper guide.
- The F and **ops** of each paid piece, from the interpreter.

**Build rule:**
- Free pieces: build if ΔR² ≥ 0.02 at layers ≥ 6.
- Paid pieces: build only if the predicted Δraw ×1.5 beats ΔF by ≥ 1.5× **and** the ops fit under 26,000. Lane P keeps about 3k ops of headroom.
  - One paid product on the youngest lineage at layers 6–15 costs about 1–2k ops.
  - 1 u ≈ 0.55 % F.

**Predictions** (strategist):
- (a) and (b) each explain 5–25 % of the weighted residual at layers ≥ 8;
- (c) and (d) explain less per u;
- Puffi's D21 mode overlaps (a).

## M5.4 — move the hub out of the legs and drop always-on columns (Oct 10–12)

**Why.** Route (i) of the κ₄→κ₃ hub (y = w2, c = (m/8)·λ·w1; est:4322–4397 and 1310–1344) rides each lineage's own legs. At birth it is Sym3(w2 ⊗ (mλ/4)·D1·X̃·D1). It is the only thing that needs always-on birth columns:
- 42 / 55 / 70 % of its column energy sits there at L9 / L12 / L15;
- P1: H2.5 literal (dead and always-on columns dropped) +3.41 %, Hd2.5 (dead only) +0.15 %, Hkh3.0 (hubs kept on always-on columns) 0.00 %.

**Step 0 — minutes.** Run **Hkh2.5** in the existing mask arm: kink-active columns at τ 2.5, hubs kept on always-on columns.
- That is the accuracy ceiling of the −6.03 % lever.
- If it costs > +1 %, stop M5.4.

**Step 1 — fidelity regression** (5090; FIT nets 0–4, layers 4–15). For each lineage age:
1. Compute route (i)'s exact contribution to D21 and D3: the Sym3(y⊗S_h) readouts with S_h = A·diag(c)·Mᵀ + M·diag(c)·Aᵀ.
2. Regress it on the same readouts with S_h replaced by γ·C̃_pre, and by γ₁·C̃_pre + γ₂·Wᵀ·diag(v)·W.
3. Report R² per age and layer, weighted as in M5.3.

The concern: the transported hub matrix differs from C̃ by inherited diagonal renewals of the same order.

**If age 1 fails** (R² < 0.8), the move is dead. Keeping age 1 explicit as (A·diag(cA))·W costs about 1 u per layer, i.e. +8 % F, which is more than the −6.03 % it frees.

**Step 2 — arm, if R² ≥ 0.8 at every age.**
- Read route (i) as Sym3(U⊗C̃_pre), with U = Σ_age γ_age·y_age (the transported thin y legs Y_A, with learned per-layer/age gains):
  - D21_ab += ⅓(2·U_a·C̃_ab + U_b·var_a);
  - D3_a += U_a·var_a.
  - These two formulas use the averaged-Sym convention. Match the code's Sym3 normalization (e.g. D3 += 2·Σ Y∘RY); the γ gains absorb any constant.
- Drop route (i) from the columns; route (ii) and route B stay on them.
- This is O(n²), about 0.002 u per layer.

**Step 3.** H = kink-active columns, KAS H table with 16 entries: 1024, 1024, 1024, 992, 928, 896, 832, 800, 768, 736, 704, 640, 608, 576, 576, 544 (FIT q50 at τ 2.5; `kink_tables.json`). Interpreter: −6.03 % F alone.

**Measure:** VAL unrefit, then a refit against the matched control.

**Predictions** (strategist): unrefit +0.5 to +3 %; after the refit, 0 to +1.5 %.

**Kill:** > +2 % after the refit. Hand over with interpreter F and ops.

## M5.5 — build the regression winners (Oct 12–14)

Build twin arms with learned per-layer gains for the pieces M5.3 selects; paid pieces go on the youngest ages first, within the op budget. Refit (pod via P) against a matched control, measure VAL, and hand over with interpreter F and ops and the marginal-rule check: −Δraw > ΔF/(1+ΔF).

## M5.6 — optional, only after M5.1–M5.5

- An α-dependent row factor on the covariance's D21 and D22 terms: g14 → g14·(1 + β_ℓ·α_a²), which generalizes M5.2's (1 − 3v·α_a²).
- Cheap corrector features:
  - row energy of C̃/σσ;
  - D21 row sums and energies;
  - K22 row sums;
  - Q̂31 row energy;
  - dG and the star parts of D4, separately;
  - the previous layer's δv.

## Do not

- Full K=4 transport, low-rank B_r, the κ₄-core loop.
- Coherent κ₃ tiers (Sym(v⊗C), C-powers).
- Learned pair closures (Step 3.3: ≤ 0.75 %).
- The full Ward/radial-split closure (dropped for time).
- Any write-up work.

## Report

- A daily paragraph to the strategist packet.
- RESULTS-style tables on lane-m.
- Hand-overs to P as: patch + twin parity + VAL (unrefit, and refit vs control) + interpreter F and ops.
