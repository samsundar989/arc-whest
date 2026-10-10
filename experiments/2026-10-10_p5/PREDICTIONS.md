# Lane P5 predictions (written before each run)

## P5.0b — v81d_starall2ws on the Oct 10 A/B slot (written 10:15 UTC, before FRESH-A)
Strategist: graded ≈ 3.39–3.40e-9. Worker: FRESH-A ≈ 1.023× better than v81b_star2ws (v81b→v81d
gain on A was 1.0239 × 1.0090 × 1.0121 ≈ 1.046 on FRESH-A; the ws twin tracks A within 0.1 %), so
FRESH-A raw ≈ 1.987e-8 × 1.001; graded ≈ 1.8833e-8 × 0.18030 = **3.396e-9** (+0.4 % over the FRESH-A projection as usual).

## P5.0c — v82a canary (see RESULTS § P5.0c)
Strategist −1.215 % F, raw unchanged. Worker 3.2795e-09, F/B 0.174135 exactly.

## P5.2 step 1 — two-sided D in the twin, VAL-100 unrefit (written ~11:00 UTC, before any arm ran)
Base = v81d's twin arm (v62_gb + mask R2.0/Hd2.5/Dz2.5 fixed K aggr_q50 + dress all + bank star,starall,
params refit_starall2; VAL 1.945887e-08). All arms keep the fixed K table (D counts 1024→800).

| arm | strategist | worker |
|---|---|---|
| control (Dz, rows-only) | = 1.945887e-08 | = 1.945887e-08 (±1e-4 rel.) |
| D2 (rows + cols, Dzero, d21e over kept cols, unscaled) | ≈ +10 % | +7 to +10 % |
| D2 + ρ (per-layer energy ratio, FIT-100) | +1 to +4 % | +2 to +5 % |
| D2 + ρ + Dleg (full two-sided: (i)–(iv)) | — | +2.5 to +5.5 % |
| D2c (diagnostic: corrector reads unmasked d21e) | — | +0.4 to +1.2 % |
| Dz + Dleg ((ii)+(iii) alone) | — | +0.1 to +0.6 % |
Worker reasoning: P1 measured D2.5 +10.55 % (per-net thresholds, no Dzero) and Dc2.5 +0.56 %; Dzero cut the
rows-only cost 4× (Dr +2.48 → Dz +0.62), and fixed K adds ~1 pt. A scalar ρ per layer fixes the mean
energy but not its per-row spread (rows whose energy sits on dead columns), so it recovers well under all
of the gap.

## v82b_bundle (items 1, 4, 5 + V8b) — written before its FRESH-A (≈ 13:15 UTC)
Measured before FRESH-A: meter F/B 0.16508 (−5.20 % vs v82a), 22,375 ops; DEV 0–9 raw −1.29 % vs v82a
(10/10); twin VAL unrefit (v82b arm vs control) −0.89 %.
Strategist (bundle, pre-pivot): −4 to −5 % F, Δraw within ±0.5 %. Worker: FRESH-A raw −0.9 to −1.3 % vs v82a,
projected ≈ 1.987e-8 × 0.989 × 0.16508 / 1.019 ≈ 3.18e-9 panel-projected; graded ≈ 1.8833e-8 × 0.989 × 0.16508
≈ **3.07e-9**.
