# O5 predictions (recorded 2026-10-12, before the run)

Rerun of M6.1's anatomy (`tools/anatomy2.py`, KAS off, 20 FIT nets, rolled exact substitution, share of
final-layer x-estimated MSE removed). The base moves from v82c_r1 to v84 (R8, fold + age 6). Age 6 cut raw MSE by
about 6 %, so a channel that kept its absolute error grows in share by 1/0.94.

| arm | M6.1 (v82c_r1, KAS off) | O5 prediction (v84) |
|---|---:|---:|
| chain final MSE (x-est) | 1.8022e-8 | 1.60–1.70e-8 |
| cov (var + C_off) | 42.8 % | 43–47 % |
| var, band 12–15 | 19.4 % | 18–22 % ("still 19 points": yes) |
| κ3 (D3 + D21) | 35.2 % | 28–34 % (age 6 took its gain mostly from late κ3) |
| D21 alone | 4.8 % | 2–5 % |
| κ4 | 9.3 % | 8–11 % |
| old tier exact (T0) | 7.4 % | 3–6 % (age 6 keeps one more generation exact; O1 shows the join is energy-exact) |
| all but μ | 75.0 % | 70–78 % |

Where the ~6 % came from: mostly the old-tier compression error of the 6th lineage. It enters through late D21/D3,
so T0 and κ3 fall together and the covariance share rises.
