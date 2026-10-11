# M6.2 refits: predictions, recorded 2026-10-11 ~03:40 UTC, BEFORE any run

Base: v83 = v82c_r2 + fold. The H lever is closed (P6 packet Oct 11), so `fold` is the live base.

Recipe:
- P6's R4: the R2 mask with J=2, the fold arm (tau 2, rec), S2tsf, 1 epoch over TRAIN-900,
  lr 1e-4 cosine, accum 8.
- Every arm starts from R3 fold's `params_best.npz`.
- The control is the same recipe with no lever, run locally, so the comparison has matched
  epochs on the same hardware.

Arms: `ctl`; `J3` (mask J=3); `age6` (candidate `_SHARED_AGE = 6`); `jsvd` (candidate
price.py join_svd, basis detached by the shim).

## Parity (before any refit)

With R2 params, each twin arm must match its R2 price.py file on VAL-100 to within 0.05 %:
- `ctl` vs P6's val_raw_fold (1.88375e-8);
- each lever vs `../screen/val_raw_fold_<arm>.json`.

The P6 arm-vs-file agreement was ≤ 0.016 %.

## Refit: Δ VAL-100 vs the refit control, and the score after ΔF

| arm | unrefit Δraw on R3 | refit Δraw vs ctl | net score |
|---|---:|---:|---:|
| age6 | −5 to −6.5 % | −4.5 to −6.5 % | −0 to −1.6 % |
| jsvd | −1.2 to −1.8 % | −0.8 to −1.8 % | +0.3 to −0.7 % |
| J3 | −0.5 to −0.9 % | −0.4 to −0.9 % | −0.1 to −0.6 % |

The refit should keep most of the unrefit gain: these levers add exact information instead of
re-weighting what the learned parts already absorb.

Kill rule per arm: the refit Δraw fails to beat break-even (age6 −4.61 %, jsvd −1.06 %, J3 −0.33 %).

## Parity result (job 2621), recorded before the refits start

Arm vs file, R2 params, VAL-100 — all PASS (< 0.05 %):

| arm | arm | file | diff |
|---|---:|---:|---:|
| ctl | 1.884030e-8 | 1.88375e-8 | 0.015 % |
| J3 | 1.870311e-8 | 1.86999e-8 | 0.017 % |
| age6 | 1.773564e-8 | 1.77337e-8 | 0.011 % |
| jsvd | 1.854913e-8 | 1.85469e-8 | 0.012 % |

Unrefit on R3 params, against ctl 1.850971e-8:
- J3: 1.836513e-8 (−0.78 %);
- age6: 1.738215e-8 (−6.09 %);
- jsvd: 1.822418e-8 (−1.54 %).

## M6.3a `acov`: α-dependent covariance and variance gains (recorded BEFORE any run)

Twin bank keys `acov` and `avar`:
- `acov`: S-block D21/D22 row factor (1 + ab_ℓ α²);
- `avar`: var × (1 + va_ℓ α²).

Both start at zero gain. Stage S2tsfa = S2tsf + gain/ab + gain/va. The recipe and the control are
the same as the M6.2 refits (`refit_ctl`). ΔF = 0, because the gains fold into existing products
when built.

- Inertness: with zero gains, the first 10 VAL nets must equal `ctl` bit for bit (`inert` batch).
- Refit prediction vs ctl: −0.3 to −1.2 % raw. Handoff: −0.5 to −2 % (with features). Kill if
  it beats ctl by less than 0.3 %.
- Learned va_ℓ should be largest in |value| for layers 12–15 (anatomy band).
