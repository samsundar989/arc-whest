# M6.1 anatomy v2: results (2026-10-11, jobs 2595 / 2598 / 2602)

Run setup:
- `tools/anatomy2.py` on `estimator_k3_v82c_r1.py`, 20 FIT nets (full rows 0–19).
- Truth = keenanpepper N = 1e9 cumulants. MSE = the noise-free cross-estimate at layer 16.
- Exact substitution is unrefit.
- Verify: the patched loop is bit-identical to `predict`, and self-substitution gives 0.0
  (`verify_*.json`).
- Predictions: `PREDICTIONS.md`.
- Raw outputs: `fit20_v82c_r1{,_kasoff}.{json,md}`, `loc_*.json`.

## The base: KAS off (same chain, same MSE)

On the shipped chain (KAS on), exact D21 makes things WORSE: −42.3 %, and k3 −1.4 %.

A 4-net localisation found the cause (`loc_*.json`), the KAS D-mask:
- the mask leaves D21 rows unformed for dropped neurons;
- the corrector is fitted with d21e = 0 on those rows;
- so the full truth feeds rows the downstream path never sees.

KAS off costs F but not accuracy: chain MSE 1.8022e-8 vs 1.8199e-8 (−1.0 %). That makes it the
interpretable base. The KAS-on table is kept in `fit20_v82c_r1.md`.

## Share of final-layer MSE removed (rolled exact substitution, layers 2–16)

| arm | Anatomy 2 (v59) | v82c_r1, KAS on | **v82c_r1, KAS off** | strategist |
|---|---:|---:|---:|---|
| chain final MSE (x-est) | 2.2943e-8 | 1.8199e-8 | **1.8022e-8** | |
| (a) var | 41.8 % | 41.9 % | **41.5 %** | |
| (b) C_off | 31.7 % | 26.5 % | **26.6 %** | |
| cov (a + b) | 42.4 % | 43.2 % | **42.8 %** | 35–45 % ✓ |
| (c) D3 | 21.6 % | 12.7 % | **12.5 %** | |
| (d) D21 | 12.5 % | −42.3 % | **4.8 %** | D3 ≥ D21 ✓ |
| κ3 (c + d) | 50.5 % | −1.4 % | **35.2 %** | 35–50 % ✓ (low edge) |
| (e) κ4: Q31 / D4 / D22 | 15.5 / −1.3 / −6.6 % | 14.9 / −6.4 / −19.0 % | **15.5 / −6.2 / −19.3 %** | |
| κ4 (all three) | 15.4 % | 8.7 % | **9.3 %** | 5–10 % ✓ |
| (f) old tier exact (T0) | — | 7.3 % | **7.4 %** | 3–8 % ✓ |
| (f′) T0 + exact P2 (T0x) | — | 8.5 % | **8.6 %** | |
| (g) all but μ | 80.3 % | 40.6 % | **75.0 %** | |
| all | 96.8 % | 97.2 % | **97.2 %** | |

## Bands (KAS off; substituted only inside the band, free elsewhere)

| object | 2–6 | 7–11 | 12–15 | full 2–16 |
|---|---:|---:|---:|---:|
| var | −2.5 % | +0.9 % | **+19.4 %** | 41.5 % |
| C_off | +2.5 % | **+13.6 %** | **+14.4 %** | 26.6 % |
| D3 | −1.2 % | −0.2 % | **+11.4 %** | 12.5 % |
| D21 | −0.6 % | +3.3 % | +6.8 % | 4.8 % |
| κ4 | −4.7 % | −4.7 % | +8.2 % | 9.3 % |

The κ3 prefixes show the same concentration:
- substituting at layers 2..13 removes 19.7 %;
- substituting at 2..16 removes 35.2 %;
- so layers 14–16 hold 15.5 points by themselves.

## Reading

1. **The covariance is the largest chunk, and it is late.**
   - var removes 19.4 % in band 12–15 and nothing earlier.
   - C_off removes 13.6 % in 7–11 and 14.4 % in 12–15.
2. **κ3 is 35 %, and it is late and interacting.**
   - D3 and D21 alone sum to 17 %, but together they remove 35 %, so the two errors compound.
   - Almost all of it sits in layers ≥ 12: D3 11.4 % and D21 6.8 % in band 12–15.
   - The v59 → v82 drop (50 → 35 %) is what the star, the gates, the dressing and the refits
     harvested.
3. **κ4 is 9 %, and it is all Q31 (15.5 %).**
   - The chain's D4 and D22 *forms* are worse than truth only in the sense that the learned
     gains are fitted around them. Exact D22 hurts by 19 %, which is learned compensation, not
     a missing mechanism.
   - Below the 10 % bar.
4. **Old tier made exact: 7.4 %** (8.6 % with exact P2). This is below the M6.4 bar of 8 %, and
   M6.2's jsvd margin agrees: the 1-pass basis is close to exact.
5. **Unrefit substitution is now a lower bound on what a refit mechanism could take.**
   - The learned parts (corrector, dressing) are fitted to the chain's own errors: with the
     corrector off, nomu reaches 96 %.
   - So "share removed" understates attackable chunks that the corrector currently
     half-patches.

## Decision (handoff rule: a chunk of ≥ 10 % that maps to a 3-day mechanism goes to M6.4)

- **κ3, band 12–15 → M6.4 GO: keep late lineages exact longer.**
  - The arm is age 6 only for lineages born at layer ≥ 8.
  - M6.2 already measured age 6 everywhere at −5.86 % raw unrefit (net −1.31 % score) on fold.
  - A late-only age 6 should keep most of that gain at a fraction of the +4.83 % F.
  - Price it first in price.py.
- **Covariance var, band 12–15 (19.4 %) → M6.3's α-dependent var gain, restricted to layers
  12–15.**
  - This is the menu's "covariance var, one band" lever, and it is cheap.
  - It folds into the M6.3 refit instead of being a separate arm.
- κ4 (9.3 %) and the old tier (7.4 %): no M6.4 arm.
