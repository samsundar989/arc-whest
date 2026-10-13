# O1 results: exact-SVD join and smaller old-tier ranks: CLOSED (2026-10-12)

Setup:
- Base: v84 = `fold+age6` with R8 params (`build_v85.py --check-v84` passes, `c1b098c0`).
- Arms: `tools/atoms/o1.py` (`o1` = SVD join at rank 256, `o1:<r>` = SVD join at rank r), built with `build_v85.py`.
- Accuracy: unrefit VAL-100 (`twin.py eval --raw --panel hold100` on each built file), paired per net against the v84
  file run the same way. The control reads 1.716117e-8, +0.005 % against R8's recorded 1.716023e-8.
- Prices: exact interpreter (`variants/prices.json`).
- Predictions: `PREDICTIONS.md`. Table: `../compare.py O1 ...`. Raw outputs: `val_raw_*.json`, `energy_v84.json`.

| arm | raw MSE | Δraw | break-even | ΔF | **net score** | per-net Δ ± se | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| o1 (SVD, r 256) | 1.71204e-08 | −0.24 % | −0.88 % | +0.88 % | **+0.64 %** | −0.23 ± 0.12 % | 56/100 |
| o1:224 | 1.75834e-08 | +2.46 % | +1.08 % | −1.07 % | **+1.37 %** | +2.46 ± 0.20 % | 7/100 |
| o1:192 | 1.85408e-08 | +8.04 % | +4.02 % | −3.87 % | **+3.86 %** | +8.08 ± 0.38 % | 0/100 |
| o1:128 | 2.42806e-08 | +41.49 % | +9.86 % | −8.97 % | **+28.79 %** | +41.60 ± 1.03 % | 0/100 |

Every arm is net worse. None is inside 1.5× its break-even (r 224 needed ≤ +1.6 %, r 192 ≤ +6.0 %,
r 128 ≤ +14.8 %), so none is refit (handoff rule).

## Diagnostic: energy captured by the one-pass basis (FIT nets 0, 1; `o1_energy.py`)

| join layer | ko | n | cap_256 (1-pass / exact top-256) | top-128 / tr G | top-192 | top-224 | top-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 1024 | 0.9990 | 0.937 | 0.983 | 0.991 | 0.996 |
| 8 | 2 | 1024 | 0.9999 | 0.958 | 0.991 | 0.996 | 0.9985 |
| 10 | 4 | 992 | 0.9999 | 0.968 | 0.994 | 0.998 | 0.9990 |
| 12 | 6 | 928 | 1.0000 | 0.977 | 0.996 | 0.998 | 0.9994 |

(both nets agree to ±0.003; every layer is in `energy_v84.json`)

## Reading

1. **The one-pass basis is already the exact top-256 eigenspace in energy terms:** 99.90–100.00 % at every join.
   The exact SVD has almost nothing left to fix. Its −0.24 % raw comes from the direction set within that energy,
   and it is smaller on age 6 than at age 5 (M6: −1.54 %), because age 6 already repairs the same join error
   (the M6 substitution finding).
2. **The final MSE is very sensitive to G's small tail.** Rank 224 loses only 0.2–0.5 % of G's energy and costs
   +2.5 % raw. Rank 192 loses 0.4–1.7 % and costs +8 %; rank 128 loses 2–6 % and costs +41 %. The raw cost grows
   roughly 5–10× faster than the lost energy. The old tier's low-energy directions carry the cross-lineage
   structure the late D21 and D3 depend on.
3. **The old tier cannot shrink on this objective.** Removing 0.04 B through rank alone is not available.

Versus predictions:
- strategist: r 224 +2 to +5 % ✓; r 192 +10 to +18 % ✗ (+8.0 %, better than predicted, still failing);
  P(any smaller rank passes) 0.25 → none passed.
- mine: o1 −0.1 to −0.3 % ✓; r 224 +1.5 to +4 % ✓; r 192 +7 to +15 % ✓; r 128 +25 to +50 % ✓;
  cap ≥ 0.98 ✓; top-224 / top-256 ≥ 0.97 ✓.

**No O1 hand-over to Lane P.** The `o1` atom stays in `tools/atoms/` as the documented fallback for an age-6
rejection only (M6.2: jsvd −1.48 % refit on fold, without age 6).
