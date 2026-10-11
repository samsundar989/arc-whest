# M6.2 unrefit screen: results (2026-10-11 ~02:50 UTC, job 2601)

Setup:
- VAL-100 (`twin.py eval --raw --panel hold100`), paired per net against P6.1's control files on
  the same R2 base: `experiments/2026-10-11_p6/screen/val_raw_{fold,fold_Habs}.json` (main tree).
- Prices are exact (price.py on v82c_r2; `variants/prices_m6.json`).
- break-even Δraw = 1/(1+ΔF) − 1; net = (1+Δraw)(1+ΔF) − 1, the score change at λ = 0.
- Predictions: `PREDICTIONS.md` (recorded before the run).

| arm | raw MSE | Δraw | break-even | ΔF | **net score** | per-net Δ ± se | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold+J3 | 1.86999e-8 | −0.73 % | −0.33 % | +0.33 % | **−0.40 %** | −0.72 ± 0.10 % | 76/100 |
| fold+age6 | 1.77337e-8 | −5.86 % | −4.61 % | +4.83 % | **−1.31 %** | −5.82 ± 0.27 % | 99/100 |
| fold+jsvd | 1.85469e-8 | −1.54 % | −1.06 % | +1.07 % | **−0.49 %** | −1.53 ± 0.19 % | 79/100 |
| fold+p2svd | 1.88398e-8 | +0.01 % | +0.08 % | −0.08 % | −0.07 % | +0.01 ± 0.06 % | 51/100 |
| fold+Habs+J3 | 3.13545e-8 | −0.55 % | −0.24 % | +0.24 % | −0.31 % | −0.54 ± 0.08 % | 75/100 |
| fold+Habs+age6 | 3.04294e-8 | −3.49 % | −4.65 % | +4.88 % | +1.22 % | −3.48 ± 0.22 % | 93/100 |
| fold+Habs+jsvd | 3.13113e-8 | −0.69 % | −1.08 % | +1.09 % | +0.39 % | −0.68 ± 0.17 % | 64/100 |
| fold+Habs+p2svd | 3.15344e-8 | +0.02 % | +0.08 % | −0.08 % | −0.06 % | +0.01 ± 0.04 % | 48/100 |

Controls: fold 1.88375e-8 (F/B 0.16554); fold+Habs 3.15284e-8 (F/B 0.15577).

## Reading

- **On the near-equilibrium base (fold), three levers clear break-even unrefit:**
  - age6: net −1.31 %, 99/100 wins. v53 measured −5.53 % raw and we now see −5.86 %.
  - jsvd: net −0.49 %.
  - J3: net −0.40 %.
- **On fold+Habs, age6 and jsvd fail.** That base is +65.6 % raw off equilibrium unrefit, so I
  trust the fold base, as recorded beforehand, and the refits decide. The fold+Habs numbers are
  the hand-over caveat: a lever's gain may shrink once H drops columns.
- **p2svd is flat** (needs ≤ −0.3 %), so it is not handed over.
- **Conditional follow-ups, both not triggered:**
  - J4: J3 clears its break-even by 0.40 %, under the 0.5 % condition.
  - jsvd + rank224 / rank192 / age4: jsvd beats break-even by 0.48 % raw, under the 1 %
    condition. The 1-pass basis is close to the exact one, so the old tier cannot shrink.
- Versus the strategist's predictions:
  - age6 −5.86 % (predicted −4 to −6) ✓;
  - jsvd −1.54 % (−0.3 to −2) ✓;
  - J3 −0.73 % (−0.2 to −0.6), slightly better;
  - p2svd 0 (0 to −0.3) ✓.
- My own predictions: age6 ✓; jsvd better than my −0.2 to −1.0; J3 slightly better; p2svd ✓.

## Next

The refits for age6, jsvd and J3 each need a same-recipe matched control. On which base, fold
or the H chain P6.1 picks, is P's call given R3. The jsvd refit detaches the SVD basis.
Hand-over: price.py names `fold+Habs+age6`, `fold+Habs+jsvd`, `fold+Habs+J3`.

## M6.4 late-only age 6 (job 2623, unrefit VAL-100 vs fold, R2)

| arm | raw MSE | Δraw | break-even | ΔF | net score | per-net Δ ± se | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold+lateage9 (born ≥ 4) | 1.83045e-8 | −2.83 % | −2.05 % | +2.09 % | −0.80 % | −2.82 ± 0.21 % | 89/100 |
| fold+lateage11 (born ≥ 6) | 1.85585e-8 | −1.48 % | −0.88 % | +0.89 % | −0.61 % | −1.47 ± 0.14 % | 87/100 |
| fold+age6 (all), for reference | 1.77337e-8 | −5.86 % | −4.61 % | +4.83 % | **−1.31 %** | −5.82 ± 0.27 % | 99/100 |

- lateage13 (born ≥ 8) is a no-op: `J=2` already skips the joins at layers 13–14.
- Gain per unit of F (|Δraw| / ΔF) rises toward the late layers: 1.21 for all, 1.35 from born ≥ 4,
  1.66 from born ≥ 6. This agrees with the anatomy.
- The variants are nested and age 6 everywhere has the largest net gain, so **age 6 (all) is the
  pick**. lateage9 and lateage11 are fallbacks if age 6's wall time or VmPeak (~+700 MiB on
  v53) fails a gate.
- Both predictions were in range: lateage9 −2.83 % (predicted −2.5 to −4), lateage11 −1.48 %
  (−1 to −2).
