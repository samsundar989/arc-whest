# HANDOFF — Lane M6 (session 2, branch lane-m), 2026-10-11: anatomy v2, the spend frontier, cheap gains — measure first, hand winners to P

**Read first:** `docs/research/2026-10-10_evening_review.md` §2, §4 and §5. This handoff follows M5, which is complete.

**Setup:** merge main into lane-m (merge, do not rebase) once Lane P6.0 has merged lane-m into main and committed the strategist's files.

**How you work** (as in M5):
- Build arms in the twin on lane-m.
- Hand each one to Lane P as a patch, with twin parity, VAL-100 numbers (unrefit, and refit vs a matched control), and F and ops.
- GPU: the local 5090 through gpuq; the pod belongs to P's refit queue, and your refits fill its gaps.

**Standing rules:**
- Predictions (yours and the strategist's) go on record before every run.
- Select on VAL-100; acceptance is FRESH-A by Lane P.
- No fitting or selection on FRESH panels.
- No external messages.
- Lane P owns main and the estimator files.
- Every refit comparison uses a same-recipe control with matched epochs.
- No write-up work.

**Tools:**
- `docs/research/2026-10-11_p6price/price.py` prices every lever below on v82c_r1 and writes the patched estimator files. Use `--write-only` for `twin.py eval --raw` screens.
- The F/B and ops below are exact (README there).
- Labels are as in M5: `cache/twin_labels/fit_*.npz` hold pre-activation cumulants already.

## Why

The κ₄ D31 channel is harvested (the fold). The marginal path ends near 2.6–2.7e-9 (review §2). Two jobs remain:
1. find out whether any attackable chunk of ≥ 10 % of today's error is left (anatomy v2, Oct 11);
2. measure the remaining spend levers so P can integrate them by Oct 14.

Our extra error against the leaders has been a roughly constant ~5e-9 per network since September, uncorrelated with how hard a net is for them. That points to a missing mechanism, with the all-distinct κ₃ the standing suspect (B1 oracle −33 %). The anatomy tells us whether it is still the largest chunk on today's chain.

## M6.1 — anatomy v2 (Oct 11, time-box one day; GPU)

**Setup:**
- On v82c_r1's twin arm: P5's v82b arm with R1 params. Add fold + H once P6.1 step 2 has picked the H variant.
- `tools/anatomy2.py`, with anchors re-checked on the v82c layout.
- Nets 0–9 for fitting, 10–19 for testing, as in Anatomy 2.

**Unrefit exact substitution**, at every layer and then rolled forward, of:
- (a) var only;
- (b) C_off only;
- (c) D3;
- (d) D21;
- (e) the κ₄ slices (Q31, D4 = 2·dG, D22);
- (f) the old tier made exact (`_SHARED_ON = False`; cost ignored);
- (g) all but μ.

Repeat (a)–(e) restricted to the layer bands 2–6, 7–11 and 12–15.

**Report:**
- the share of final MSE removed per arm and band;
- the same table for Anatomy 2 (v59: covariance 42 %, κ₃ 51 % with D3 22 % and D21 12.5 %, κ₄ 15 %, all but μ 80 %) side by side.

**Decision rule:** a chunk of ≥ 10 % that maps to a mechanism we can build and refit in three days goes to M6.4. Otherwise M6.4 is skipped and the lane spends its time on M6.2–M6.3.

**Prediction (strategist):**
- κ₃ 35–50 %, with D3 ≥ D21 and most of it in band 12–15;
- covariance 35–45 %;
- κ₄ 5–10 % (the star and fold took the rest);
- old tier made exact 3–8 %.

## M6.2 — the spend/join frontier (Oct 11–12; twin, VAL-100)

Use P6.1's chosen fold + H chain as the base; until it exists, use `fold+Habs`.

| arm | price.py name | ΔF on fold+H | break-even Δraw | strategist prediction |
|---|---|---:|---:|---|
| age 6 | `fold+Habs+age6` | +4.88 % | −4.65 % | −4 to −6 % (v53 measured −5.53 %) |
| exact SVD join basis | `fold+Habs+jsvd` | +1.09 % | −1.08 % | −0.3 to −2 % |
| J = 3 | `fold+Habs+J3` | +0.24 % | −0.24 % | −0.2 to −0.6 % |
| J = 4, only if J = 3 clears its break-even by ≥ 0.5 % | `fold+Habs+J4` | +1.09 % | −1.08 % | — |
| P2 by exact top-4 SVD | `fold+Habs+p2svd` | −0.08 % | < +0.08 % | 0 to −0.3 %; it costs +1.5–3 s wall, so hand it over only if Δraw ≤ −0.3 % |

1. **Unrefit screen:** `twin.py eval --raw --cand …/variants/est_<name>.py --panel hold100` against `est_fold_Habs.py`.
   - The SVD arms need the shim's `linalg.svd`/`eigh` (P6.1 step 1(iii)); if P has not added them yet, add them on lane-m.
2. **Refit** the arms that pass break-even unrefit, via P's pod queue, against a matched control. Detach the SVD basis in refits.

**Why the SVD join is worth a test:**
- The join's basis today is one subspace-iteration pass warm-started from the previous Q. That captures the joiner's new directions only through their overlap with the old basis.
- flopscope bills a top-k SVD at 4mnk, its randomized-SVD price, while computing the exact SVD. This is documented in `flopscope/numpy/linalg/_svd.py` and is not a bug.
- So the exact top-256 eigenspace of G = J·D·Jᵀ + Q·Sg·Qᵀ costs +1.09 % F, whereas a second pass costs +3.65 %.
- Optional diagnostic before refitting: the fraction of G's top-256 energy that the 1-pass basis captures, per join layer, on 2 FIT nets. If it is ≥ 99 % everywhere, skip the refit.

**If the SVD join beats its break-even by ≥ 1 % raw**, the 1-pass basis is far from optimal, and the exact basis may let the old tier shrink. Screen these unrefit:

| arm | F/B | ΔF vs fold+H+jsvd | break-even Δraw vs fold+H+jsvd | 1-pass history |
|---|---:|---:|---:|---|
| `fold+Habs+jsvd+rank224` | 0.15350 | −2.5 % | +2.6 % | rank 224 cost +6.8 % raw (v67) |
| `fold+Habs+jsvd+rank192` | 0.14773 | −6.2 % | +6.6 % | rank 192 cost +21 % raw (v67) |
| `fold+Habs+jsvd+age4` | 0.14910 | −5.3 % | +5.6 % | age 4 cost +18 % raw (v53) |

This is the only cost lever left that could approach the leaders' F (0.107). Odds are low, but each screen is an hour.

**Hand-over** to P: price.py name, twin parity, VAL (unrefit, and refit vs control), F/ops, plus the VmPeak of age 6 (it added ~700 MiB on v53).

## M6.3 — α-dependent covariance gains and cheap corrector features (Oct 12–14)

- **A row factor on the covariance's D21 and D22 terms:** g14 → g14·(1 + β_ℓ·α_a²), with β_ℓ learned per layer and initialized at 0. This generalizes M5.2's (1 − 3v·α_a²), whose physical v the refit absorbed.
- **Cheap corrector features** (each O(n²) or less, from arrays the layer already holds):
  - row energy of C̃/σσ;
  - D21 row sums and row energies;
  - K22 row sums;
  - Q̂31 row energy;
  - dG and the star part of D4, separately;
  - the previous layer's δv.
- **Measure:** VAL-100 refit vs a matched control (corrector + gains, lr ≤ 1e-4, 2 epochs).
- **Prediction:** −0.5 to −2 %.
- **Kill:** < 0.3 % better than the control.
- History: local per-neuron features explained only 6–8 % of the orthogonal error (B2), and hidden 64 lost to 32. Keep the corrector's width and add features only.

## M6.4 — conditional on M6.1 (Oct 13–14)

Pick from this menu by the anatomy's largest attackable chunk:

| anatomy result | lever | P owner |
|---|---|---|
| κ₃, band 12–15 | keep late lineages exact longer: age 6 only for lineages born at layer ≥ 8 (the `tools/build_fold.py` mechanism in reverse; Step 4(b) tested only earlier folding, which failed; price it first), or J = 4 under M6.2's condition | M arm → P |
| old tier made exact ≥ 8 % | SVD join (M6.2), then rank 288 only with the exact basis | M → P |
| κ₄ ≥ 10 % | per-band refit gains for the fold (g★) and the star (sg), as in M6.5. Paid Q31 pieces stay closed (M5.3: ΔR² ≤ 0.02 each) | M |
| covariance var, one band | an α-dependent var gain for that band (M6.3 form) | M |

Do not start a new representation for the all-distinct κ₃ this week. B2–B5 and rev 22 closed the cheap forms, and low-rank D21 readouts failed on Oct 10: 2.5e-3 variance error at rank 64 against a ~1e-4 tolerance.

## M6.5 — optional, only if M6.1–M6.3 finish early

- **Deeper dead-side H cut:** asymmetric H with the dead side at −1.5 (alive at 3.0). Use per-net thresholds in the twin; P prices the count table from the twin's FIT statistics.
- **Per-band refit gains** for the fold (g★) and the star (sg) if the anatomy shows κ₄ concentrated in one band.

## Do not

- Full K=4 transport, low-rank B_r, the κ₄-core loop, coherent κ₃ tiers, learned pair closures.
- Radial Sym(μ⊗κ₃) (killed M5.2), the hub move (fidelity failed M5.4), paid Q31 extras (M5.3).
- Low-rank D21 readouts, spherical-design cubature (Oct 10, review §4).
- "Spend more on hard nets": per-net MSE CV is 12 %, so there is no leverage.
- Any write-up work.

## Report

- A daily paragraph to the strategist packet.
- RESULTS-style tables on lane-m.
- Hand-overs to P as: price.py name or patch, twin parity, VAL (unrefit, and refit vs control), interpreter F and ops.
