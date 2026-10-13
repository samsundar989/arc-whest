# O3 (i) predictions (recorded 2026-10-12, before any O3 run)

Base: v84 (`fold+age6`, R8), F/B 0.17354, 22,113 ops. Atom `tools/atoms/o3.py` arm `o3:<A>`: no old tier, the A
youngest lineages exact, older ones dropped at the end of every layer. Unrefit VAL-100, twin `--raw`, paired vs v84.

Exact prices (`variants/prices.json`; they match `x3_tables.txt`'s no-old 0.149 / 0.161 / 0.172):

| arm | F/B | ΔF | break-even Δraw | ops |
|---|---:|---:|---:|---:|
| o3:6 | 0.13503 | −22.19 % | +28.5 % | 17,985 |
| o3:7 | 0.14901 | −14.13 % | +16.5 % | 18,070 |
| o3:8 | 0.16123 | −7.09 % | +7.6 % | 18,138 |
| o3:9 | 0.17175 | −1.03 % | +1.0 % | 18,189 |

History: the generation cap on the 09-05 chain (no old tier, exact ages only) was +9 % raw at 12 generations and
+26 % at 11. M6.1's anatomy: making the rank-256 old tier exact removes 7.4 % of the error, so the old tier is close
to exact and dropping it removes real content.

**Lane O (mine), (i) dropped:** o3:9 +40 to +120 %, o3:8 +60 to +180 %, o3:7 +80 to +250 %, o3:6 +100 to +300 %.
All fail. P(any (i) arm passes) ≈ 0.03. Arm (i) measures how much old content (age > A) carries. That sets the bar for
(ii): the harmonic tier has to recover all but ≈ 7.6 / Δraw(i) of the o3:8 loss to pass.

**Strategist (handoff), (ii) harmonic tier at A = 8:** +15 to +40 %; P(passes) ≈ 0.1. Kill: o3:8 (ii) worse than
+10 % unrefit closes the family.
