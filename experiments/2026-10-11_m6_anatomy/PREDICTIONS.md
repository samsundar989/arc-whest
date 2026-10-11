# Anatomy v2 (M6.1): predictions, recorded 2026-10-11 ~01:40 UTC, BEFORE any chain run

Handoff: `docs/HANDOFF_2026-10-11_LANE_M6.md` § M6.1. Tool: `tools/anatomy2.py` on lane-m.
The panel is the same as Anatomy 2: 20 FIT nets (full rows 0–19), with nets 0–9 used to fit and
10–19 to test (that split matters only for the κ4 regressions). Nothing is selected on these nets.

## Tool changes since Anatomy 2 (v59)

- **Post-star D4 hook.** v82 adds the star, `D4 = m·dG + D4_star`, AFTER the `dG` hook. A `dG`
  substitution alone would therefore count the star twice. Every arm that substitutes `dG`
  now also substitutes `D4` (truth κ_iiii). On a star-free chain like v59 this is the
  identity, so the shares stay comparable with Anatomy 2.
- **Band arms.** var, C11, D3, D21 and k4 are substituted only inside 1-based layers 2–6, 7–11
  and 12–15, and the chain runs free outside the band.
- **Anchors on v82c_r1 and fold+Habs.** All 10 occur exactly once in both files. The fold
  enters through the tower recompression (`_shared_contract_lean`), i.e. the next layer's κ3
  lineages, downstream of the D3/D21 hooks. The H cut masks `fac1` *after* C11 is read, so
  it is a structural truncation that substitution does not undo. This is by design: we
  substitute objects, not mechanisms.

## Strategist (handoff)

- κ3: 35–50 %, with D3 ≥ D21 and most of it in band 12–15
- covariance: 35–45 %
- κ4: 5–10 % (the star and the fold took the rest)
- old tier made exact: 3–8 %

## Mine (Lane M)

These are shares of the chain's final-layer MSE (cross-estimate) removed by rolled
substitution at layers 2–16.

| arm | Anatomy 2 (v59) | v82c_r1 prediction |
|---|---:|---:|
| (a) var | 41.8 % | 30–42 % |
| (b) C_off | 31.7 % | 22–32 % |
| cov | 42.4 % | 35–45 % |
| (c) D3 | 21.6 % | 18–28 % |
| (d) D21 | 12.5 % | 12–22 % |
| k3 (D3 + D21) | 50.5 % | 40–55 % |
| (e) k4 (Q31, D4, D22) | 15.4 % | 3–10 % |
| (f) old tier exact (T0) | — | 2–8 % |
| (g) all but μ | 80.3 % | 75–85 % |

Reasoning:
- The corrector, the gates and the refits since v59 mostly absorbed *average* drift. The
  network-specific κ3 and covariance error remains, and the relative κ3 share rises.
- The star and the fold took the κ4 D31 channel, so the k4 share falls.
- D21 rises relative to D3, because the all-distinct κ3 is the standing suspect.

Bands (share removed, out of the full-range value for each object):
- band 12–15 carries ≥ 50 % of the k3 share;
- band 2–6 carries ≤ 15 % of every object;
- the band sum is less than the full-range share, because the substitutions interact
  sub-additively (Anatomy 2's prefixes showed this).

Decision rule (handoff): a chunk of ≥ 10 % that maps to a mechanism we can build and refit
in three days goes to M6.4. My prior is that κ3 band 12–15 will clear 10 %, and that the
mechanism is "keep late lineages exact longer" (age 6 for late-born lineages / J = 4).

## Addendum, ~02:55 UTC: compensation localisation (recorded BEFORE the run)

Interim on 4 nets (job 2595, nets 0–3). Rolled D21 @2..16 gives −43.7 % (it HURTS), k3 −2.6 %,
nomu +36.6 % (v59: +80.3 %), all +97.1 %. Reading: the learned parts the chain has acquired since
v59 are fitted to the chain's biased D21, so exact D21 breaks the compensation.

Run: nets 0–3, rolled arms D21, k3, nomu and all only, with one component switched off at a
time. Each knockout is its own base (the chain MSE changes too), and the share is measured
against that base.
- `corr` = `_CORR_ON=False`: the corrector reads d21e.
- `dress` = `_DRESS_C2=_DRESS_D21=_DRESS_D22=0`: the dressed leg reads D21.
- `kas` = `_KAS_ON=False`: the D-mask truncates the D21 rows formed.
- `alloff` = all three.

Prediction:
- dress is the main culprit: D21 turns positive (≥ +5 %) with dress off.
- corr is second.
- kas alone changes the D21 share by < 10 points.
- alloff restores nomu to ≥ 70 %.

## Addendum, ~02:40 UTC: localisation result, and the KAS-off anatomy (predicted BEFORE the run)

Localisation on nets 0–3, rolled D21 / k3 / nomu shares:
- none: −43.7 / −2.6 / 36.6;
- corr off: +8.2 / +44.3 / 96.2;
- dress off: −29.7 / +19.4 / 51.5;
- KAS off: +7.5 / +34.5 / 74.1;
- all off: +14.4 / +59.7 / 97.3.

**My dress-first prediction was wrong. KAS is the culprit:**
- with KAS off, the chain MSE barely moves (1.6652e-8 vs 1.6685e-8);
- with KAS on, the D-mask leaves the chain's D21 rows unformed for dropped neurons, and the
  corrector is fitted with d21e = 0 there (`_kz`);
- substituting the full true D21 then feeds rows the downstream path never sees.

The corrector is the second compensator: with corr off, nomu reaches 96 %.

So the interpretable anatomy is the full 20-net run with `_KAS_ON=False`, at the same base MSE.

Prediction for that run (rolled 2..16 shares):
- var 40–44 %; C11 25–29 %; cov 42–46 % (KAS barely touches covariance);
- D3 15–25 %; D21 +3 to +12 %; k3 30–40 %;
- k4 8–14 %;
- nomu 70–78 %;
- band 12–15 carries ≥ 50 % of k3.
