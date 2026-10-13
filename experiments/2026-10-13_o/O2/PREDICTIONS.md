# O2 predictions (recorded 2026-10-12, before any O2 run)

Base: v84 (`fold+age6`, R8), F/B 0.17354, 22,113 ops. Atom `tools/atoms/o2.py`: one one-pass basis per leg, each on
its own Gram, carried as `Q = [Q_A | Q_M]`. Unrefit VAL-100, twin `--raw` on the built files, paired against v84.

Exact prices (`variants/prices.json`):

| arm rA/rM | F/B | ΔF | break-even Δraw | ops (Δ) |
|---|---:|---:|---:|---:|
| 256/256 | 0.17746 | +2.26 % | −2.21 % | 25,193 (+3,080) |
| 256/192, 192/256 | 0.17276 | −0.45 % | +0.45 % | 25,031 (+2,918) |
| 256/128, 128/256 | 0.16848 | −2.91 % | +3.00 % | 24,113 (+2,000) |

**Pricing finding (before any accuracy run):** the split itself costs +2.26 % F and ~+3,000 ops (two range finders,
two QRs and two re-rotations per join, a wider transported Q), so a 192 leg saves only 0.45 % F. The handoff assumed
the saving of a smaller shared rank. The +3,000 ops are ≈ +1.1 s of median server wall at 0.36 ms/op, and +0.03 s of
projected residual (inside the 0.4 s cliff, but against A's wall margin).

**Strategist (handoff):** one leg tolerates 192 at ≤ +3 %; P ≈ 0.3.

**Lane O (mine):**
- 256/256: −0.5 to −1.5 % raw (two full bases hold more than one shared one); fails its −2.21 % break-even.
- 256/192 and 192/256: +0.2 to +2 %; P(either passes +0.45 %) ≈ 0.2.
- 256/128 and 128/256: +2 to +7 %; P(either passes +3.0 %) ≈ 0.2.
- The two asymmetric arms differ by more than their noise (se ≈ 0.2 %); which leg is the tolerant one I do not
  predict.
- P(any O2 arm net-negative unrefit) ≈ 0.3. Even a pass carries +3,000 ops of wall, which goes to Lane P's gate.
