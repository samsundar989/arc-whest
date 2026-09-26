# Twin stages — predictions, recorded 2026-09-25 ~05:45 UTC, BEFORE any training step

Acceptance (done first): twin DEV 100 raw 2.385123e-08 vs v59 2.384984e-08 (+0.006%, bar 0.1%).
Per-net parity on FIT 0-2: +7.5e-4 / +6.3e-5 / -8.4e-5 relative -- inside the cupy-vs-numpy per-net
envelope (worst 1.9e-3, RESULTS "Two accuracy backends"); the handoff's per-net 1e-4 bar is below
what any two float32 backends achieve, so the panel criterion is the binding one.

Order after Anatomy 2 (Samuel acknowledged the S4 promotion, 2026-09-25): S1 -> S4 -> S2 -> S3.

## Strategist (PLAN rev 13 §3 Step 2)
- S1: -3 to -6% raw over v59; S2-S4 together a further -5 to -15%.

## Mine
- S1 (corrector pages end to end, 900 TRAIN nets): **-1.5 to -3.5% raw** on DEV. The sequential
  teacher-forced fit already removed exposure bias layer by layer; what end-to-end adds is
  credit assignment to the final layer, which I expect to be modest. HOLDOUT-100 within 1 pp of DEV.
- S4 (sigma page on var_pre, on top of S1): **a further -3 to -8%**. Exact var recovers ~47% of
  one-step error, but a 27-feature page can only learn the predictable part; the mean corrector's
  analogous ceiling was ~7% of a 21%-local channel.
- S2 (lambda/theta/feed gains + S1): -0.5 to -2% (D6 found the profiles already at their optimum
  under greedy fitting; end to end may unlock a little).
- S3: not run unless S1+S4+S2 leave time (Anatomy 2 caps it at ~3%).

## Scored so far (2026-09-25 ~17:50 UTC)
- S1: DEV -0.62% / HOLDOUT-100 -0.76% (strategist -3..-6, mine -1.5..-3.5: both too high).
- S4 (on S1): HOLDOUT-100 -0.10% (mine -3..-8: wrong). The output page moved ~2 Adam steps' worth in
  448 -- the gradient has no consistent sign, i.e. the 27 local features do not predict the var error.
- S2 (on S1, gains lam/theta/feed + corr pages): my -0.5..-2% stands.

## S2b / S2t (2026-09-26 ~01:10 UTC)
- S2b (S2 continued 2 epochs): HOLDOUT-100 2.358810 -> 2.355795e-08, -0.13%. Converging.
- S2t prediction (mine, before running): per-term gains (13 table + 8 one-D + 12 uni + 27 biv
  per layer, 960 scalars) on top of S2b: **-0.3 to -1.0%** HOLDOUT-100. More capacity than the three
  profiles, but D6 found the closure constants already near their greedy optimum.
