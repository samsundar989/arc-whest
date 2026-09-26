# S2t3 — twin continued on TRAIN 900 + d8b 2,048: predictions before the run (2026-09-26 ~20:00 UTC)

Init: S3/params_best.npz (the v67/v68 parameter set). Stage S3 leaves (corrector pages, λ/θ/feed,
term gains, q31), lr 1.5e-4, 2 epochs over TRAIN 900 (full rows 0–899) + d8b 0–2047 (per-layer
MC means, N = 1e8; loss = final MSE + 0.1 × all-layer MSE as in every stage). Select on HOLDOUT-100.
Note: the handoff says "TRAIN 800"; the twin's TRAIN has been full rows 0–899 since S1, so it is kept.
Control already on record: S3b = S3 continued at 1.5e-4 on TRAIN 900 alone → HOLDOUT-100 −0.17%
(2.264823 → 2.260955e-08). The d8b effect is S2t3 minus S3b.

- Strategist: −1 to −2% raw.
- Mine: HOLDOUT-100 −0.3% to −0.8% vs S3 (i.e. d8b worth −0.1 to −0.6 pp beyond S3b); the stages
  have plateaued (S3b −0.17%) and the train/HOLDOUT gap is small, so more nets of the same
  distribution buys little. P(≥ −1%) = 20%.
