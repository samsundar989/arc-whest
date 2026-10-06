# HANDOFF — Lane P2 (session 1, main), 2026-10-07: AUGMENT-lite physics + cost hygiene

Repo `C:\Users\samsu\Documents\arc-whest`, branch main (you own main, RESULTS, ledger, DESIGNATION, CURRENT_STATE, estimator files). Read first: `docs/PLAN_rev19_2026-10-07.md`, `docs/research/2026-10-07_deep_review_strategist.md` (why), `docs/research/2026-10-07_R1_efficient_higher_order.md` §5.1 (dressed legs) and `docs/research/2026-10-07_R2_structure_and_subspace.md` §1.5 (gain-mode law) (also in the claude.ai project as `claude/deep-review-2026-10-07.md`, `claude/review-2026-10-07-R1/R2-*.md`). Standing rules: PLAN rev 18 §5 hard gates; ≤ 5 canaries/UTC day + 1 A/B slot; every run has your prediction and the strategist's written down first; Samuel designates; no external messages.

**Why.** On 16 DEV nets at n = 1024, ARC's factored AUGMENT — no learned parts — beats v73 raw 16/16 (1.79e-8 vs 2.32e-8, −23 %); unfactored AUGMENT is ≈ −35 %. Nine teams now sit at raw 1.37–1.57e-8: AUGMENT-level accuracy is the field's common step and we don't have it. Two cheap ways to get most of AUGMENT's physics into our chain at ~0 FLOPs were found this session: dressed legs (R1) and the gain-mode κ4 closure (R2).

## P2.0 — housekeeping (Oct 7 morning, ≤ 3 h)
0. First, commit the strategist's rev 19 docs unchanged in one commit (`docs/PLAN_rev19_2026-10-07.md`, `docs/HANDOFF_2026-10-07_LANE_P2.md`, `docs/HANDOFF_2026-10-07_LANE_M2.md`, `docs/research/2026-10-07_*.md`; they are untracked on main now) so Lane M2 can rebase onto them.
1. Submit v78_kashws on today's A/B slot (gated already). Expected ≈ 4.226e-9.
2. Merge Lane V's panel/evaluator changes from `lane-m` (tools/fresh, panels.py, ev.py `--panel fresh_a|fresh_lock|fresh_d`). From now on, acceptance is **FRESH-A paired vs the incumbent**, per PLAN rev 18 §2.
3. Run v77 on FRESH-A (calibration of the current A; expect ≈ 4.03e-9 = graded × 1.019).
4. **Exact cost census of v77** with `tools/cost_site_probe.py` (update the anchor constants for v77; real meter, f64 weights, one DEV net). Report the site table by category and by layer. We need to know whether the recent tier is still ~60 %.
5. flopscope re-evaluation exposure: organizers will release a fixed flopscope and re-evaluate affected submissions before selection opens (~Oct 10; forum #18118). Issue #275 under-bills `out=`/`where=` broadcasts and batched-`out=` matmuls. v77 has no `out=`, `where=` or symmetry tags (strategist grep) — confirm for v77/v78 and for anything you build; never introduce them.

## P2.1 — Dressed legs: AUGMENT's κ3 births at SIMPLE cost (Oct 7, twin first)
ARC's AUGMENT two-leg births (factor_k3.py `legs`) add four "heavy" edges to SIMPLE's light edge C: D21/2 (κ3 (2,1), near multiplicity 2), D21ᵀ/2 (κ3 repeated at the hub), K22/4 (κ4 (2,2)) and C∘C/2 (doubled edge). Every resulting family shares the SIMPLE hub leg and differs from it in one outer leg only, so to first order they fold into one **dressed covariance leg**:

  X̃[a, j] = w1_a C_aj + H[a, j],
  H[a, j] = ½ w2_a D21[a, j] + (w3_j / w2_j) · ( ½ w1_a D21[j, a] + ¼ w2_a D22[a, j] + ½ w2_a C_aj² ),  a ≠ j, diagonal 0

with w1 = Φ(α), w2 = φ(α)/σ, w3 = −αφ(α)/σ² (Wick table w(1,1), w(2,1), w(3,1); note w3/w2 = −α/σ, finite), D21[a, j] = κ3_aaj of the current pre-activation (the slice the closure already reads), D22 = our κ4 (2,2) slice, C = C11. Then Sym Σ_j X̃_j ⊗ (3 w2_j e_j) ⊗ X̃_j = SIMPLE + all eight AUGMENT heavy-edge families + a spurious H⊗H term (second order; R1 toy: 2.4–3 % of the correction). The strategist checked each family's coefficient and hub weight against ARC's `fac1/fac3` construction.
- **Implementation in v77:** the newborn's A_L = (w1 g_e)∘C11 becomes (w1 g_e)∘C11 + H (apply the birth gate to the light part only, or to both — test), and C_L = A_L diag(w2b) follows automatically. Recompute every repeated-slice compensation that reads fac1 (p111_21's two-leg term, S_sep, hub compensation) with the dressed leg. No new matmuls: H costs a few n² elementwise passes (≈ 0.003 units/layer).
- **Twin test first** (VAL-100, free-running): v77 vs v77+dressing, (a) unrefit, (b) after a gentle refit (S2t-stage gains + corrector, lr 1e-4, 1 epoch TRAIN-900). The learned gains currently compensate for the missing families, so an unrefit loss is not a verdict; the refit result is.
- Also test the subsets {C²}, {D21, D21ᵀ}, {D22} separately, because families cancel each other (R1 toy: cos(F1, F2) ≈ −0.8): include them as a set unless a subset is clearly better.
- Predictions (strategist): full dressing after refit −4 to −12 % raw [I, 60 % that it is ≥ 4 %]; subsets alone can be worse than none. Record yours.
- Kill: full set after refit < −2 % raw on VAL ⇒ record and stop.

## P2.2 — Gain-mode κ4 closure (Oct 7–8, twin)
R2 found that at depth the top covariance direction is the network's input-dependent overall gain (a bias-free ReLU net is positively homogeneous; |cos(u₁, m̂)| ≈ 0.98 at layer 16). Modelling z = g·ẑ (g a scalar random gain, Var g = c²) gives zero-parameter slice templates that explain 57–77 % of κ4 slices at layer 16 (83–92 % with one fitted β per layer), and it explains why 504aldo's λ·C_off law works and why the (2,2)-slice regression on C collapses with depth. Templates (pre-activation):
  c_l² = m̂ᵀ C_post m̂ / ‖m‖²  (m = post-ReLU mean, from the chain's own state, O(n²))
  D4_i = 12 β c² σ_i⁴ ;  D22_ij = 4 β c² (σ_i² σ_j² + 2 C_ij²) ;  Q31_ij = 12 β c² σ_i² C_ij
- Twin arms (VAL-100): (a) replace D22's additive form (dG_i + dG_j)/3 by the product template (keep D4 from dG); (b) also replace Q31's λ C_ij by the σ²-scaled template (keep or drop the 5 learned q31 modes — test both); (c) blend: κ4 slice = existing + γ_l·template with γ_l fitted. β_l and γ_l are per-layer twin parameters (init β = 0.7).
- Predictions (strategist): best arm −2 to −8 % raw after refit. Kill: < −1 %.
- Cost ≈ 0 (O(n²)).

## P2.3 — Combine and ship (Oct 8–9)
Build `estimator_k3_v79_aug.py` = v77 + whichever of P2.1/P2.2 passed, refit in the twin (lr 1e-4, select on VAL), then FRESH-A paired vs v77, all §5 gates, canary (A) + its block-40 twin (B). Prediction: graded 3.4–3.8e-9 if both pass.

## P2.4 — Cost hygiene on the v79 chain (Oct 9–11; each a measured, separate commit)
1. **Always-on D21 rows (twin mask first).** The closure multiplies row i of D21 by w2_i = φ(α_i)/σ_i, which vanishes for always-on neurons too, not just dead ones; K11 for an always-on row only needs D21's *column* i on kink rows. Extend Dz to rows with α > +τ (τ ∈ {2.5, 3.0}); compute S21 for those rows without the inherited D21 term (it cancels against p111 for a linear neuron); no corrector delta on them first, then test serving their corrector feature from kink columns. Expected −4 to −7 % FLOPs at ≤ +1 % raw.
2. **Covariance sandwich on alive rows**: WᵀC W with C restricted to the alive block (dead rows/cols of the post-ReLU C ≈ 0). Expected −2 to −3 %.
3. **Join**: route the join's big products through `_mm`; compose the old-tier re-rotation lazily instead of re-rotating every old coordinate at every join (A1 review §12.3.2). Expected −1 to −4 %.
4. Layer-0 `C_pre = WᵀW` directly if `_mm(Wᵀ, I)` is still there (≈ −0.3 %).
5. Strassen leaf for compact products (block 20 today): test 12/16 with ops and residual measured (v77's worst residual is already 0.306 s — keep worst ≤ 0.33 s).
Each item ships only if FRESH-A says score improves.

## Do not
Reopen PLAN §6 items; use `out=`/`where=`/symmetry tags; fit or select on FRESH; designate; contact organizers.

## Report
Daily paragraph in `docs/STRATEGIST_PACKET_2026-10-0X.md`; RESULTS rows; after P2.1/P2.2 the twin tables with both predictions; after P2.3 the FRESH-A and graded rows; the v77 census as soon as it exists.
