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
