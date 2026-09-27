# Step 3.1 (HANDOFF 09-27) — closure bench on exact inputs: predictions before the run (2026-09-26 ~21:45 UTC)

`tools/closure_bench.py`: the shipped closure (v62_s1 + S3 parameter set) fed EXACT pre-activation
objects at every layer; K vs exact post moments on FIT 0–99. Kill rule (handoff): residual < 20% of
the per-layer state-error increment for EVERY output ⇒ drift is amplification, not closure bias.
Increments (Anatomy 2): var ~1.3e-4/layer, C_off ~6e-4/layer, D3 0.45%/layer, D21 0.7%/layer.
The like-for-like columns are `var_next` / `Coff_next` (W^T ΔC W vs the exact pre objects at l+1).

- Strategist (for the whole of Step 3): ≥ 20% raw (35%), 5–20% (35%), < 5% (30%).
- Mine: the kill rule does NOT fire. var_next 1e-4 … 5e-4 and Coff_next 3e-4 … 2e-3 per layer
  (i.e. ≥ the increments: the closure's exact-input bias is the same size as the observed drift);
  K3 relative 1–3%, K21 2–5% (both above the D3/D21 increments). K21 orientation: the chain's
  K[(2,1)] matches bench `post_K21` untransposed. Regression (3.2): R² of the univariate K1/K2
  residual on (α, γ3, γ4, layer) 0.3–0.6 with the ridge, higher with the 16-hidden MLP; the pre
  κ5/κ6 add < 0.1 R² (truncation is not the dominant term). Step 3 end to end: 3–8% raw.

## Step 3.3 — embedded closure nets: predictions before any run (2026-09-27 ~10:10 UTC)
Build: `tools/build_clos.py` on v62_s1 (twin base) with the offline fit `clos_init.npz`
(`tools/closure_fit.py`, exact-input pairs FIT 0–79). Checks first: (a) the all-zero build reproduces
the base in the twin exactly; (b) the offline-initialised nets, UNTUNED, on HOLDOUT-100 with the S3
parameter set; then (c) twin stage S5 = S3 leaves + closure nets, TRAIN 900, lr 3e-4, S2t length,
select on HOLDOUT-100; export → bit-parity → f64 cost → canary as (A, B).
- Strategist: ≥ 20% raw (35%), 5–20% (35%), < 5% (30%).
- Mine: (b) untuned: anywhere from −4% to +10% (offline-accurate closures are often unstable
  in-loop; the chain's λ/θ/q31/corrector were fitted to compensate the uncorrected closure, so a
  corrected closure is initially double-counted). (c) after S5: **−3% to −8% raw** on HOLDOUT-100
  vs the S3 set (P(≥ 20%) = 5%, P(5–20%) = 45%, P(< 5%) = 50%). Cost +0.2% FLOPs, +~250 ops.

### S5 so far, and S5z added (2026-09-27 ~11:40 UTC, before S5z runs)
Zero build == S3 on HOLDOUT-100 (2.264823e-08, exact). Offline init untuned: 2.379587e-08 (+5.1%).
S5 fine-tune: 2.3819 (step 28) → 2.3006 (56) → 2.2765 (84) → 2.2775 (112): converging toward, not
below, the base. **S5z**: identity start — the offline fit's hidden layers kept, output layers zeroed,
so the chain starts exactly at S3 and the gradient adds only the correction that survives in-loop.
Mine: S5z −0.3% to −2% on HOLDOUT-100 vs S3 (the closure residual is real and predictable, but the
chain's λ/θ/q31/corrector already absorb much of it).
- S5 final 2.247849e-08 (−0.75% vs S3, −0.58% vs S3b). Real-meter cost of the embedded nets:
  **+1.68% FLOPs** (480.12 G vs 472.20 G; +969 ops) — the per-PAIR net; per-neuron only (--no-biv)
  is +0.004% (472.22 G, +444 ops). As built, S5 is score-NEGATIVE (~+0.9%).
- S5z (identity start, zero output layers): **diverged** — 2.264823 → 4.614227e-08 at step 28 (×2.04):
  Adam drives every zero-initialised output weight at ~lr·zero_scale per step whatever the gradient,
  so all corrections switch on together and overshoot. Cancelled (worker-q 2224). Prediction
  (−0.3 to −2%) wrong: the identity start is not stable at this step size.
- v69_clos export parity: 2.247738e-08 vs twin S5 2.247849e-08 (5e-5, Winograd rounding) ✓.
  v69u (S5 params, pair net removed): 2.387342e-08 (+5.4%) — S5's parameters co-adapted with it.
- **S5u** (per-neuron net only, trained from the offline init; +0.004% FLOPs): mine −0.2% to −0.6%
  on HOLDOUT-100 vs S3 (most of S5's −0.75% came with the pair net).
