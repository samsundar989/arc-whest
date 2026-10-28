# Worker spec X4 — zero-FLOP κ4/closure faces never tested in our chain (SSC slices, marg2, per-neuron metric, C0³/C0·K31), one matched refit

**Owner:** one Claude Code session; GPU = RunPod 5090 pod (≈ 2–3 h, ≈ $5–8) so it runs beside the local R3; local box only for the final VAL-100 read if the pod lacks the labels.
**Clock:** gains + zero control by +2 h; matched 112-update refit by +4 h; VAL read by +5 h. If it passes, fold the gains into R3's selected export as a second refit stage (≈ 2 h) before the canary.
**Why:** these four items are the only public mechanisms the round-4 forum audit found *absent* from b636/8be, all cost ≈ 0 FLOPs, and none has a RESULTS.md row. The SSC item has a graded sibling result: Susupro ported Puffi's Sym(C⊗C) slices beside V29's λ law, decoupled λ's three uses, refit 9 coefficients on 12 nets → **×0.946 adjusted, raw −5.4 %** (forum 18254; code github.com/logxio/fourth-cumulant, MIT). Our λ law is the same family. The round-4 physics agent argued the gain should be small for us from a census; that is an argument, not a chain measurement.

## 0. Pre-registered prediction and kill line
- Prediction (strategist): matched refit, VAL-100 paired, corrector free: **−2.5 % raw (80 %: −0.5 … −6 %)**; P(≤ −2 %) 0.45. Per item if measured separately: SSC −1.5 %, marg2 −0.5 %, metric −0.3 %, C0³/K31 −0.2 %.
- Keep if ≤ −1.5 % raw with CI excluding 0 (zero FLOPs ⇒ no cost break-even), then FRESH-A100 ≤ −1 %.

## 1. The gains (all default 0 ⇒ byte-identical zero control; top-level literal lists per layer, `_YC`-style gate, read by index at use time — same pattern as the U1 gains)
1. **SSC faces** (scale-mixture κ4 beside the λ law, per-layer coefficient v_l): D4_i += 3 v_l var_i²; Q31_ij += 3 v_l var_i C_ij (in Q31n: 3 v_l ρ_ij); D22_ij += v_l (var_i var_j + 2 C_ij²). Decouple λ's uses as Susupro did: separate gains for the λ term in D4/dG, in Q31, and in D22 (`_K4M_QS`, `_K4T1_GQ`-type knobs already exist — expose each as its own trainable per-layer list). Also add the var² D4 face alone as its own gain (`_K4T1_CV`, already specified in the round-4 U1 spec §3.2 item 4).
2. **marg2**: apply the k = 2 Wick-coefficient corrections (a2 → w(2) + w(5)t/6 + w(6)κ4/24) only on rows with α ≤ −2, in Q11/Q21/Q22 together (forum 18249 §3.5(iv): applying it to one power cumulant only is ×1.09–1.11 worse). Implement as an α-gate on the existing k = 2 monomials with a per-layer gain.
3. **Per-neuron metric**: replace the scalar metric 2.0 by M_ii = Σ_a W_ai² (= `_WT2.sum(axis=1)`, already formed) in the κ4 slices through a blend gain: metric_i = 2.0·(1−g_l) + g_l·M_ii.
4. **Closure pair diagrams**: ⅙ a3 b3 C0³ and C0·K31 as biv `["C","Q31"], (4,2), 1/6` plus its transpose, each with a per-layer gain (closure JSON `s6_build_closure`).

## 2. Refit and read
- Matched CONTROL ‖ ARM from the 8be export: young recipe, 112 updates, TRAIN900, corrector LR ×10, VAL every 28, best-VAL export, step-0 fallback. If the pod has time, a third arm = SSC only (to attribute).
- VAL-100 paired on the real numpy chain; report Δraw, wins, CI, the fitted per-layer v_l (sanity: v_l ≈ 0.002–0.01, cf. the sandbox's emergent-mode prediction v_eff(l) ≈ 1/(2n) + 0.4(l−1)/n) and the λ movement.
- Bill: real meter on 2 nets (expected +0.0x %; elementwise only), ops (+≤ 60 calls).

## 3. Deliverable
`experiments/2026-10-20_x4/RESULT.md` with predictions vs outcome; one `RESULTS.md` line; if kept, the gain tables as constants for the R3 second stage.
