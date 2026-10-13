# O3 results: more exact ages, no old tier: CLOSED, (ii) closed on (i)'s numbers (2026-10-12)

Setup as O1: v84 R8 base, unrefit VAL-100, twin `--raw` on each built file, paired against the v84 file
(1.716117e-8). Arms are `tools/atoms/o3.py` `o3:<A>`: no old tier, the A youngest lineages exact, older ones dropped.
Exact prices are in `variants/prices.json`; predictions are in `PREDICTIONS.md`.

## (i) older content dropped

| arm | raw MSE | × v84 | break-even | ΔF | **net score** | per-net Δ ± se | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| o3:9 | 6.89507e-08 | ×4.02 | +1.04 % | −1.03 % | **+297.7 %** | +303.7 ± 6.3 % | 0/100 |
| o3:8 | 1.09908e-07 | ×6.40 | +7.63 % | −7.09 % | **+495.0 %** | +544.7 ± 10.2 % | 0/100 |
| o3:7 | 1.73776e-07 | ×10.13 | +16.46 % | −14.13 % | **+769.5 %** | +918.6 ± 14.2 % | 0/100 |
| o3:6 | 2.71119e-07 | ×15.80 | +28.51 % | −22.19 % | **+1129 %** | +1488 ± 20.7 % | 0/100 |

The excess error (× − 1) falls by a factor of about 0.58 per extra exact age (14.8 → 9.1 → 5.4 → 3.0). At that
rate dropping reaches its break-even only past A ≈ 15, where nothing is dropped at all.

## (ii) harmonic tier: not built. The kill rule fires on (i)'s numbers

The kill rule is: o3:8 (ii) worse than +10 % unrefit closes the family. For (ii) to come in under +10 % at A = 8,
the radial tier would have to recover

    1 − 0.10 / 5.40 = 98.1 %   of the error the dropped content leaves.

The evidence on how much it can recover:
- **Measured at age 5** (v68, RESULTS.md "Step 1 (HANDOFF 09-27)"): drop ×17.8 against harmonic ×5.9 on DEV-100,
  so **71 %** recovered.
- **The handoff's own labels:** the non-gain residual of D21, which the radial part Sym(u⊗C) cannot carry, is 16–23 %
  of its energy with a flat spectrum. That caps recovery near **77–84 %**, unless the residual decays with age.
- **Needed:** 98.1 %, which is 4–7× less unrecovered error than either bound allows. Even an optimistic 90 % gives
  (ii) ≈ +54 % at A = 8; 84 % gives ≈ +86 %.

The handoff's own prediction for A = 8 (ii) is +15 to +40 %. Its low edge already fails the kill rule, and on (i)'s
magnitude the realistic range is +50 to +160 %. Porting `tools/build_harm.py` (v68) onto v84's KAS widths, fold and
P2-separable tower is about half a day, and the result cannot pass. **I closed the family without building it.**
This is a judgment call recorded here for the strategist to overrule. The one test that could change it is a direct
measurement that the radial recovery exceeds 98 % at age 8, and nothing on record points that way.

## Reading

1. **Content older than age 8 carries ×6.4 of v84's raw MSE.** The rank-256 tier holds it at 0.0385 B. O1 shows it
   cannot be cut by rank (r 224 is already +2.5 %), O2 shows it cannot be cut by leg, and O3 shows it cannot be
   dropped or replaced radially.
2. **On the "leaders carry old content at near-zero cost" reading of their bills:** in our formulation the old
   content is neither low-rank (O1: the error grows 5–10× faster than the lost energy) nor radial. A leader at
   0.105–0.116 B is either carrying it in a different representation or giving up raw MSE we do not see. Our numbers
   do not support a cheap old tier inside this tower.

Versus predictions: mine, (i) all fail with P(pass) ≈ 0.03 ✓. The magnitudes were far above my ranges (o3:9 +302 %
against +40 to +120 %; o3:8 +540 % against +60 to +180 %), so the old content matters more than I expected.

**No O3 hand-over to Lane P.**
