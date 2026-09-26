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
