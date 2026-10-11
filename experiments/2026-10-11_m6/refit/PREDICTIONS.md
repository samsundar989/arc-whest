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

- Step size: zero-init leaves step at lr × zero_scale = 1e-6, so ab and va get `--lr-mult 1000` (about 1e-3 per step). Job 2633 was cancelled before it started for this reason.
- Inertness: with zero gains, the first 10 VAL nets must equal `ctl` bit for bit (`inert` batch).
- Refit prediction vs ctl: −0.3 to −1.2 % raw. Handoff: −0.5 to −2 % (with features). Kill if
  it beats ctl by less than 0.3 %.
- Learned va_ℓ should be largest in |value| for layers 12–15 (anatomy band).

## M6.3b `cfeat`: linear feature head (recorded BEFORE any run)

Bank key `cfeat` adds a zero-init head after the corrector, active on non-final layers only:
mu_out += σ · (Z @ cf_ℓ). Z holds the per-net z-scores over neurons of five features:
- C̃ row energy;
- normalised D21 row sums;
- K22 row sums;
- dG/σ⁴;
- the star part (D4 − m·dG)/σ⁴.

Settings: stage S2tsfc, `--lr-mult gain/cf=10` (about 1e-5 per step). ΔF ≈ +5 O(n²) reductions
per layer, about 0.
- Not built: Q̂31 row energy (needs the closure's nb) and the previous layer's δv (needs a state
  carry).

Inertness: zero head = ctl bit for bit (10 nets).

Prediction vs ctl: −0.1 to −0.6 % raw. B2: local features explain only 6–8 % of the orthogonal
error, and the MLP already reads d21e. Kill if < 0.3 % better.

## jsvd refit, restarted (~07:10 local)

Job 2626 stalled: cuSOLVER's default driver failed to converge on the join's X, and there was no
training step in 45 min. It was cancelled.

Shim fix: the SVD runs outside autograd with the `gesvd` driver. The singular values keep their
exact first-order gradient through diag(Uᵀ a V). The jsvd patch uses only the detached basis.

Parity re-check: `r2_jsvd_gesvd` must equal r2_jsvd (1.854913e-8) to within 0.01 %. Then the
refit is requeued. The prediction is unchanged.

## Patience fix (~08:20 local)

The batch used M5's `--patience 3`. J3 (job 2627) early-stopped at step 84: the step 28 dip meant
no eval beat the unrefit start. At matched step 84 it reads 1.836519e-8 vs ctl 1.850907e-8
(−0.78 %); the log is kept in `refit_J3_p3/`.

ctl ran the full 112 steps (its step 84 eval beat its start by 6e-13). From now on every arm runs
`--patience 99`, as R4 does, so the epochs match. acov (2634) was cancelled one minute in and
requeued, and J3 is rerun.

## cfeat diverged (job 2637), retry (~09:00 local)

Train MSE went 1.75e-8 at step 1 → 0.98 at step 5, and VAL was 0.906 at step 28; the job was
cancelled. The unbounded z-scores (near-constant features turn float noise into huge values) plus
the chain feedback blew up within the first few Adam steps. The log is kept in `refit_cfeat_div/`.

Retry:
- z → 3·tanh(z/3), denominator rms + 1e-6;
- `--lr-mult gain/cf=1` (about 1e-6 per step);
- inertness is unaffected (zero head ⇒ identity).

Prediction unchanged (−0.1 to −0.6 %); kill if < 0.3 %.

## J3 full refit cancelled (job 2642, unstarted)

age6+J3 is neutral (+0.10 % marginal), so J3 matters only if age 6 is rejected. Its unrefit
evidence (net −0.40 %) and the p3 refit read (−0.78 % at matched step 84) cover that case. The GPU
time goes to M6.3.

## acov at lr-mult 1000 FAILED to train (job 2641); rerun at 100

VAL by step: 1.8510 (init) → 60.77 (step 28) → 2.0613 → 1.9799 → 1.9321e-8 (step 112).

That is +4.54 % vs ctl at step 112. The spike at 1e-3 steps on α²-multiplied gains (α² ≲ 9) never
recovered. Log kept in `refit_acov_lr1000/`. This is a step-size failure, not a verdict on the
lever.

Rerun at `--lr-mult 100` (1e-4 per step, up to about 0.01 of movement over the epoch). Same
prediction and kill rule.

## cfeat root cause (~11:45 local): a NaN gradient, not instability

The retry (2644) hit the same VAL, 9.061023e-01, to 7 digits: that is the MSE of a zeroed (NaN)
prediction. A zero-spread feature (the star part where sg = 0, D21 sums at layer 0) gives
d√0 = ∞, so the first backward step produces a NaN gradient and a NaN head.

Unit test: the old normaliser's gradient is non-finite and the new one, √(mean + 1e-12), is
finite. Fixed in twin.py and requeued. The tanh bound and lr-mult 1 stay.

## acov root cause (~12:10 local): unbounded α²

Rerun 2650 at lr-mult 100 still read 2.0976e-8 at step 28 (+13 % vs ctl at step 28); cancelled.
ALPHA_CAP is 40, so α² ≤ 1600 in both gains: a 1e-4 step moves a dead neuron's variance by 16 %,
and α depends on var (feedback).

Fix: both gains read min(α², 9) (|α| ≤ 3, the kink region). Shim `minimum` added. Rerun at
lr-mult 100, after an inertness re-check. Same prediction.

## cfeat at lr-mult 1 (job 2651): underpowered, rerun at 10

VAL 1.846790e-8 vs ctl 1.848205e-8 (−0.077 %). But the learned |cf| max is 5.3e-6 (rms 2.3e-6),
an effect of ≈ 1.5e-5 on the mean against an error rms of ≈ 1.3e-4, with signs flipping by layer.
The head barely moved, so this is not a verdict. The earlier lr-mult 10 blow-up was the √0 NaN
gradient, now fixed. Rerun at lr-mult 10 (1e-5 per step); log kept in `refit_cfeat_lr1/`. The
kill rule stands.
