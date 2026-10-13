# O5 results: anatomy on v84 (2026-10-13, workerq job 2661, 22.5 min)

Setup:
- `tools/anatomy2.py` (lane-m tree, for its gpu_shim `block`) on `estimator_k3_v84_age6.py`, with `--set _KAS_ON=False`.
- 20 FIT nets (full rows 0–19); truth = keenanpepper N = 1e9 cumulants; MSE = the noise-free cross-estimate at
  layer 16.
- Verify: the patched loop and the recording are bit-identical, and self-substitution gives 0.0 (`verify_v84.json`).
- Raw outputs: `fit20_v84_kasoff.{json,md}`. Predictions: `PREDICTIONS.md`. M6.1 comparison base:
  `experiments/2026-10-11_m6_anatomy/RESULTS.md` (v82c_r1, KAS off).

## Share of final-layer MSE removed (rolled exact substitution, layers 2–16)

| arm | M6.1 v82c_r1 | **v84** | absolute error left by this channel (share × chain): M6.1 → v84 |
|---|---:|---:|---|
| chain final MSE (x-est) | 1.8022e-8 | **1.6179e-8** (−10.2 %) | |
| var | 41.5 % | 41.6 % | 7.48e-9 → 6.73e-9 |
| C_off | 26.6 % | 22.5 % | |
| **cov** | 42.8 % | **42.7 %** | 7.71e-9 → 6.91e-9 (−10 %) |
| D3 | 12.5 % | 6.1 % | |
| D21 | 4.8 % | −3.7 % | |
| **κ3** | 35.2 % | **22.0 %** | 6.34e-9 → 3.56e-9 (**−44 %**) |
| Q31 / D4 / D22 | 15.5 / −6.2 / −19.3 % | 12.3 / −4.5 / −18.8 % | |
| κ4 | 9.3 % | 8.7 % | 1.68e-9 → 1.41e-9 |
| **old tier exact (T0)** | 7.4 % | **1.9 %** | 1.33e-9 → 0.30e-9 (**−77 %**) |
| T0 + exact P2 (T0x) | 8.6 % | 3.2 % | |
| all but μ | 75.0 % | 63.9 % | |
| all | 97.2 % | 95.9 % | |

## Bands (substituted only inside the band)

| object | 2–6 | 7–11 | 12–15 | M6.1 12–15 |
|---|---:|---:|---:|---:|
| var | −2.0 % | +3.5 % | **+18.1 %** | +19.4 % |
| C_off | +2.5 % | **+12.2 %** | +9.3 % | +14.4 % |
| D3 | −1.1 % | −3.0 % | +7.6 % | +11.4 % |
| D21 | −1.4 % | −0.1 % | +3.8 % | +6.8 % |
| κ4 | −5.3 % | −6.9 % | +9.3 % | +8.2 % |

κ3 prefixes: layers 2..13 remove 10.8 % and 2..16 remove 22.0 %, so layers 14–16 hold 11.2 points (M6.1: 15.5).

## Reading

1. **Where age 6's gain came from:** κ3 and the old tier. The κ3 error the chain leaves fell 44 % in absolute
   terms, and the old-tier compression error fell 77 %. The covariance error fell only by the chain's own 10 %.
   This is the mechanism M6 named: the 6th exact generation removes the join's compression error, which reaches the
   final layer through late D3/D21.
2. **The old tier is now almost exact:** making it exact removes only 1.9 % (3.2 % with exact P2). With O1–O3 that
   closes the old tier from both sides. It cannot be made cheaper (O1–O3) and making it better is worth ≤ 2 %.
3. **The covariance is now the dominant channel: 42.7 %, and it is late.**
   - var in band 12–15 carries 18.1 points; the answer to the handoff's question ("is it still 19 points?") is yes.
   - C_off peaks in band 7–11 (12.2 points).
   - It is 2× κ3 (22.0 %) and 5× κ4 (8.7 %).
4. **κ4 is unchanged at 8.7 %, almost all Q31 (12.3 %).** D22's −18.8 % is still learned compensation, not a
   missing mechanism.

Versus predictions:
- chain ✓; cov 42.7 % (predicted 43–47 %: the share held, because cov fell with the chain);
- var 12–15 ✓; κ4 ✓;
- κ3 ✗: 22.0 % against 28–34 %, so age 6 took more out of κ3 than predicted;
- T0 ✗: 1.9 % against 3–6 %, so the old tier is closer to exact than predicted;
- all but μ 63.9 % against 70–78 % ✗, consistent with the κ3 drop.

**Implication for the plan (strategist's call):** the remaining attackable chunk is the late covariance, var in
layers 12–15 (≈ 18 points) and C_off in 7–15. M6.3's α-dependent var gain (acov) was killed at +0.001 %, so a
learned rescale is not it. The structural question is which approximation in cov_pre's propagation (the C_off
recursion, or the W^T C W sandwich's use of the gained closure) drifts late. That is outside Lane O's handoff.
