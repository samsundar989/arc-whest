# PLAN — ARC White-Box Estimation Challenge, Phase 2 (rev 14, 2026-09-27)

**This is the only plan.** It supersedes rev 13 (Sep 25) and answers `docs/STRATEGIST_PACKET_2026-09-26.md`. Rev 13's levers are executed: seven graded steps 4.9756e-9 → **4.7039e-9** (v68_wino #332478), the twin works and has plateaued, every size/age/rank knob is measured, and the anatomy against exact labels is in (`RESULTS.md` § Anatomy 2). Labels: **[M]** measured / primary source, **[I]** inference, **[?]** unknown. Close Oct 17 23:59 UTC (20 days); write-up Oct 24.

Samuel's decisions (Sep 26–27): one Claude Code session for now, more later. Defaults adopted by the strategist until he overrides: (a) the cost experiment (§3 Step 1) runs first because it is the shorter test; (b) decision point **Oct 5** on graded evidence, write-up starts **Oct 8** regardless, hardening/designation Oct 12–16; (c) designation stays **A = best score (leaf 32), B = wall-safe twin** under Rules v14 §7 (only the better designate is ranked). Standing rules from rev 13 (submission gate = projected score beats the best graded + the hard gates; ≤ 5 canaries/day + 1 A/B slot; external MC data approved; probe-before-inference; every run has a written prediction) stand.

---

## 1. Where we stand [M]

Board Sep 26 evening: #1 J2W 1.90e-9 (raw 1.60e-8, **0.119 B**, 341 entries), #2 suliman_tadros 2.10e-9 (raw 1.59e-8), **#3 Luna 2.10e-9 — raw 1.54e-8 (best on the board), all-layers 6.24e-9, 0.139 B, walls 53–56 s, per-net FLOPs ±2%; two days ago 1.67e-8 / 0.155 B** — improving both axes, with ~65 s of wall unused. #4 marius 2.30e-9, #5 mliston 2.30e-9 (raw 1.68e-8), #6 a_s6 2.50e-9, #7 Puffi 2.70e-9 (raw 2.00e-8 — they cut cost), #9 emanuel_ruzak 3.0e-9. **Us #33 at 4.70e-9** (raw 2.19e-8, 0.215 B, A = v68_wino 93.9 s worst quiet, B = v64_ws 63.6 s measured on a noisy day).

Gap to #1: accuracy 1.37× (2.19 vs 1.60e-8), cost 1.8× (0.215 vs 0.119 B). To win at our cost we would need raw 9e-9 — better than anyone's; to win at our accuracy we would need 0.087 B — below the floor. **Both gaps must close; neither closes with knobs.**

## 2. What the anatomy says, and what it changes

Anatomy 2 (v59 vs exact 1e9-sample labels, 20 FIT nets) [M]:
- The carried objects are accurate: at layer 16 var 0.19%, C_off 0.9%, D3 7%, D21 10.6% relative error. The score is that small because ∂E[ReLU]/∂σ ≈ φ(α) is large.
- **Every object's error grows linearly with depth** (var 4.5e-5 → 1.9e-3; C_off 8e-4 → 9.1e-3; D21 1.2% → 10.6%). Random accumulation grows as √L; linear growth is a systematic per-layer bias — the ReLU closure's output cumulants are biased by a deterministic function of the state. This is the cumulant-neglect signature (Wu & Lin 1996) and the Nica–Ortmann para-Gaussian recursion says the leading 1/n correction is layer-recursive and additive (`docs/research/2026-09-27_G_closure_learning_lit.md`).
- Rolled exact substitution removes: cov 42% (var 42, C_off 32), κ3 51% (D3 22, D21 12.5), κ4 15% (all Q31; exact D4/D22 make it *worse*), all-but-μ 80%, all 97%. Two-step: exact C_off at l−1 recovers 35% of the mean error at l; exact var at l−1 recovers 0% — the variance error is inherited from the *off-diagonal* covariance, which is why a per-neuron σ corrector (S4) found nothing and why the μ corrector caps at 7%.
- D21 error is birth/closure (T0 unshared = 99% of v59's); old-tier compression owns 1–2% of D21 error but 6–10% of final MSE through other channels.
- "Exact κ4 makes it worse" means λ/θ were fitted to compensate the wrong κ4 model: **substitution without refit underestimates a channel's value.** The right instrument for an error-compensating system is substitute-and-refit (§3 Step 2).

Consequences: (i) accuracy lives in the **closure bias** (calibrable — we hold both sides of the closure equation for 100 nets: exact inputs and exact post-ReLU K1/K2/K11/K3/K21/K22 with pre 5th/6th moments); (ii) cost lives in the **old tier's representation**, and the one representation never tried is ARC's own harmonic split (§3 Step 1); (iii) the wall glue is memory-bound (Winograd bought 6.7 s where the law said 1.5 s), so layout work is worth more than priced and is what makes leaf 16 designable.

## 3. The plan (one session; ordered by expected value per day)

### Step 1 — the harmonic old tier (Sep 27–28, ≤ 1.5 days; the cost jump)
Hypothesis [I]: an old source's κ3 splits, with respect to the metric C = C_pre(l), into a radial part **Sym(u ⊗ C)** (vector core u, ARC `harmonic.py` r = 1 for d = 3) plus a traceless remainder; the radial part transports for free (C is already transported; u ← Wᵀ(Φ∘u)), its slices are elementwise (D3_i = u_i C_ii; D21_ij = (u_i C_jj + 2 u_j C_ij)/3), and it is linear in u so **all old sources collapse into one n-vector**. If the radial part carries most of the old tier's *value to the final mean*, the old tier costs ~0 instead of ~10 units/layer at depth (25% of FLOPs), the leaders' bill is explained, and age 6–7 exact tiers become affordable. It can fail (the traceless part may carry the value — T1's ×8.85 is the size of what must be recovered). This was rev-12 H1, never run.
Test: fold at age 5 into u (metric C, and the cheap metric diag(var)); arms at fold age 4 and 3; DEV raw vs v68; units saved from the f64 twin. Prediction (strategist): metric-C fold at age 5 within +10% raw of v68 with −20% FLOPs [I, 35% confidence]; worker: record own. Decision: ≤ +10% raw ⇒ ship path (age sweep 5/6/7 with the free tier, then canary); +10–50% ⇒ one day on a rank-16 traceless remainder per source (Sym(a⊗b⊗b) terms from the source's leading singular columns) before closing; > +50% ⇒ closed, record as a write-up result.

### Step 2 — refit ceilings (Sep 29–30; the instrument for the accuracy jump)
In the twin on FIT nets (labels for FIT rows 0–99 exist: full1000 + bench-supp), substitute one channel exact at every layer 2..16 **and refit** the twin's parameters (corrector pages, λ/θ/feed, term gains; S2t-length stage): X ∈ {cov, κ3, κ4, all}. Report HOLDOUT-20 final MSE vs the refit control; the gap is X's true value. Prediction (strategist): cov ≥ 45%, κ3 ≥ 45%, **κ4 30–45% (vs 15% unrefit)**; worker: record own. The largest ceiling picks Step 3's target; a κ4 ceiling ≥ 30% also opens Step 4b.

### Step 3 — closure calibration (Sep 30 – Oct 3; the accuracy jump)
Bench: run our closure on **exact inputs** (μ, C, D3, D21, D4, Q31, D22 from labels) and compare its outputs (K1, K2, K11, K3, K21, K22) with exact post-ReLU moments; report per-layer residuals next to the anatomy's per-layer error increments. Then regress the residual on the closure's own normalised inputs (α, γ3, γ4, ρ, d21, d12, d22, q31, and, as a diagnostic only, the exact pre 5th/6th moments to see whether the residual is truncation or state-predictable). Embed what is predictable as (i) analytic terms if the residual has a clean form, else (ii) tiny elementwise nets (univariate on n; bivariate on n² pairs, ≤ 0.2% FLOPs, ~10 ops/layer), trained on exact pairs, then **fine-tuned through the twin** (solver-in-the-loop; offline-accurate closures are unstable online without it). Prediction (strategist): ≥ 20% raw with 35% probability, 5–20% with 35%, < 5% with 30%. Kill: exact-input residual < 20% of the per-layer increment for every channel ⇒ the drift is amplification, not closure bias — go to Step 4.

### Step 4 — exploit and cost (Oct 4–7)
(a) **Layout / wall** (CPU): no-`concatenate` block layout, fewer Strassen temporaries, merged thin calls; refit the wall law with the Winograd point; price leaf 16 on the two widest contractions; canary if noisy-implied worst ≤ 105 s (A) — B stays leaf 64. (b) **κ4 with memory in harmonic r = 1 form** (only if Step 2's κ4 ceiling ≥ 30%): carry the matrix core G (n×n) with memory, G ← WᵀGW (2 units) + births from the closure's K22/K31, κ4 slices from Sym(G⊗C); fit gains in the twin. (c) If Step 1 landed: age 6/7 sweep with the free tier; P3 rank-8 re-priced.
Background GPU whenever idle: twin S2t3 on TRAIN 800 + the d8b corpus (2,048 nets' final means, noise 9e-10); expected −1–2%.

### Step 5 — decide, harden, write (Oct 5 decision; write-up from Oct 8; hardening Oct 12–16)
Oct 5: graded evidence decides whether score work continues to Oct 15 (a jump ≥ 10% in hand) or narrows to one canary/day. Write-up leads with the mechanism: the channel-drift anatomy, the linear-in-depth closure bias and its calibration in tensor-basis form, the harmonic old tier (positive or negative), the elasticity map; the corrector is a component. Designation A/B with three clean graded runs each, one noisy-day reading of B, residual ≤ 0.33 s, RSS ≤ 7 GiB.

## 4. Not doing (closed by measurement; do not reopen)
Everything in rev 13 §5 plus: adaptive age/λ, fold-age-by-birth-layer, shared rank ≠ 256, tiered ranks, σ corrector on var_pre from local features, cov_out from corrected pK1, recursive corrector feature, hidden 64, memoryless old-tier regeneration (T3), per-source dense κ4 legs (≥ 0.95 B), sampling of any kind.

## 5. Odds (strategist, honest)
#1: 5%. Top-3: 10% (needs Step 1 and Step 3 both). Top-10: 35% (Step 3 alone at ≥ 15%, or Step 1 alone). Contribution prize: the strongest lever we hold — Step 3's calibrated closure and Step 1's harmonic test are both mechanistic results whether they win or lose.

## 6. Evidence
`docs/research/2026-09-25_A..F_*.md` (deep review), `2026-09-27_G_closure_learning_lit.md` (closure-learning literature: Huang–Hu–Ma, Maddu–Weady–Shelley, Ling TBNN, Gupta–Lermusiaux, Um solver-in-the-loop, Brandstetter pushforward, Nica–Ortmann para-Gaussians, Sherman–Kolda symmetric CP), `RESULTS.md` § Anatomy 2 and § twin stages.
