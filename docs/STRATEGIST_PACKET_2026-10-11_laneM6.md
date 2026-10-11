# Strategist packet: Lane M6, 2026-10-11 (daily paragraph)

Raw tables:
- `experiments/2026-10-11_m6_anatomy/RESULTS.md` (M6.1);
- `experiments/2026-10-11_m6/screen/RESULTS.md` (M6.2).

Predictions were recorded before each run in the `PREDICTIONS.md` beside each table.

## Oct 11 paragraph

**Tool fixes.** `tools/anatomy2.py` is re-anchored on v82c_r1. All 10 anchors occur exactly once,
and the patched loop is bit-identical to `predict` on v82c_r1 and fold+Habs. Two fixes were
needed:
- a post-star D4 hook. The star is added after `dG`, so the old κ4 arm would have counted it
  twice.
- `block` in the cupy shim, for Strassen.

**Anatomy v2 needed a different base.** On the shipped chain, exact D21 makes the final MSE
*worse* (−42 %), and "all but μ" removes only 41 % (v59: 80 %).
- A 4-net knockout localised the cause to the **KAS D-mask**, not the dressing (my prediction
  was wrong). The mask leaves D21 rows unformed and the corrector reads d21e = 0 there.
- KAS off leaves the chain MSE unchanged (1.802e-8 vs 1.820e-8), so it is the interpretable base.

**Shares of final MSE removed on that base** (your prediction in brackets):

| arm | share | prediction |
|---|---:|---|
| covariance | 42.8 % | 35–45 ✓ |
| κ3 | 35.2 % | 35–50 ✓ |
| κ4 | 9.3 % | 5–10 ✓ |
| old tier exact | 7.4 % | 3–8 ✓ |
| all but μ | 75 % | |

- **κ3 is late and interacting.** D3 alone removes 12.5 % and D21 alone 4.8 %, but together
  they remove 35 %. Layers 14–16 alone hold 15.5 points, and band 12–15 has D3 11.4 % and
  D21 6.8 %.
- **The covariance is late as well.** var removes 19.4 % in band 12–15 and nothing earlier.
  C_off removes about 14 % in each of 7–11 and 12–15.
- **κ4 is all Q31** (15.5 %). Exact D4 and D22 *hurt*, because the gains are fitted around the
  chain's forms.

**M6.1 decision: M6.4 GO.**
- κ3, band 12–15 → keep late lineages exact longer (age 6 for lineages born at layer ≥ 8;
  price first).
- Covariance var, band 12–15 → M6.3's α-dependent var gain, restricted to layers 12–15.
- No arm for κ4 or the old tier.

**M6.2 unrefit screen** (VAL-100, R2, paired against P6.1's controls; net = score change):

| arm | Δraw on fold | net score on fold | wins | on fold+Habs |
|---|---:|---:|---:|---|
| age6 | −5.86 % | **−1.31 %** | 99/100 | fails break-even |
| jsvd | −1.54 % | **−0.49 %** | 79/100 | fails break-even |
| J3 | −0.73 % | **−0.40 %** | 76/100 | −0.31 % |
| p2svd | flat | — | — | flat; dropped |

- Unrefit fold+Habs is +65.6 % raw, so it is a poor screening base. I trust fold, as recorded
  beforehand.
- Neither conditional follow-up is triggered:
  - J4 needs J3 to clear break-even by ≥ 0.5 %; it cleared by 0.40 %.
  - jsvd + rank/age4 needs jsvd to clear by ≥ 1 %; it cleared by 0.48 %.

**Next.** Refit the three winners against a matched control. Which base to refit on (fold, or
P6.1's H chain once R3 picks it) is P's call. Then price the late-only age 6, and set up the
M6.3 var-gain and feature refit.

## Oct 11, later: M6.2–M6.4 complete

All refits share one recipe:
- 1 epoch from R3 fold, R4's recipe, VAL-100;
- a matched local control: 1.848205e-8.

**age 6: hand-over** (`experiments/2026-10-11_m6/HANDOVER_M6_age6.md`).
- Refit −6.07 % vs the control (−6.04 ± 0.27 % per net, 99/100), at +4.83 % F: **net −1.54 %**.
- Twin parity 0.011 %.
- WSL VmPeak +400 to +670 MB (1.15–1.20× headroom). The lifecycle check passed with 5 repeats and
  failed once with 3 (a one-off allocator step).
- Wall time not measured.

**jsvd and J3 substitute for age 6.**
- jsvd's refit gives −1.48 % (net −0.43 %), but on top of age 6 it is +0.74 % marginal.
- J3 is neutral on top of age 6 (+0.10 %).
- Both are fallbacks only, in case age 6 fails a gate. jsvd's hand-over is written.

**M6.4.**
- The handoff's "age 6 for lineages born ≥ 8" is already in the chain: J=2 skips those joins.
- lateage9 and lateage11 are nested and lose to age 6 everywhere (net −0.80 % / −0.61 %).
- Age 7 late on top of age 6: +0.46 %.

**M6.3: both arms killed.**
- α-dependent cov/var gains: +0.001 % vs the control. The learned ab ≈ −2.5e-4 has a consistent
  sign, but averaged physics does not transfer.
- Linear corrector-feature head: +0.046 %.
- Before the verdicts, three training bugs were fixed: unbounded α² (ALPHA_CAP 40), a √0 NaN
  gradient in the z-score, and frozen zero-init gains.

Twin fixes on lane-m (P should merge them):
- the shim SVD runs outside autograd with the `gesvd` driver (cuSOLVER non-convergence stalled
  the jsvd refit);
- shim `minimum`;
- anatomy2: the post-star D4 hook, band arms, `--only` / `--set`;
- cupy shim `block`.

**Lane status.** The M6 plan is complete. Nothing on it is left to attack:
- the anatomy's late κ3 chunk is what age 6 buys;
- the late variance chunk resisted a parametric gain;
- κ4 and the old tier are below the bar.

Strategist direction is needed for anything further. Candidates: an age-6 + od16 joint price and
the wall-time A/B (P); or a network-specific variance mechanism, which this lane has no 3-day
design for.
