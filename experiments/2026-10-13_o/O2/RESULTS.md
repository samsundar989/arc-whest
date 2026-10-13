# O2 results: separate old-tier ranks per leg: CLOSED (2026-10-12)

Setup as O1 (`../O1/RESULTS.md`): v84 R8 base, unrefit VAL-100, twin `--raw` on each built file, paired against the
v84 file (1.716117e-8). Arms are `tools/atoms/o2.py` `o2:<rA>/<rM>`, with exact prices in `variants/prices.json`.
Predictions are in `PREDICTIONS.md`.

| arm rA/rM | raw MSE | Δraw | break-even | ΔF | **net score** | per-net Δ ± se | wins | ops Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256/256 | 1.71293e-08 | −0.19 % | −2.21 % | +2.26 % | **+2.07 %** | −0.17 ± 0.08 % | 61/100 | +3,080 |
| 192/256 | 1.73452e-08 | +1.07 % | +0.45 % | −0.45 % | **+0.62 %** | +1.11 ± 0.16 % | 24/100 | +2,918 |
| 256/192 | 1.88133e-08 | +9.63 % | +0.45 % | −0.45 % | **+9.14 %** | +9.68 ± 0.37 % | 1/100 | +2,918 |
| 128/256 | 1.86768e-08 | +8.83 % | +3.00 % | −2.91 % | **+5.66 %** | +8.92 ± 0.36 % | 0/100 | +2,000 |
| 256/128 | 2.52440e-08 | +47.10 % | +3.00 % | −2.91 % | **+42.82 %** | +47.36 ± 1.07 % | 0/100 | +2,000 |

Every arm is net worse. The nearest, 192/256, misses 1.5× its break-even (≤ +0.67 %), so none is refit.

## Reading

1. **The split code is sound.** At 256/256 (two full bases) it is −0.19 % raw against the shared basis (61/100
   wins). That is the expected small gain from more total rank. Its cost is +2.26 % F and +3,080 ops, mostly the
   transport of the wider Q (512 columns against 256) and the doubled range finder per join.
2. **The legs are strongly asymmetric. The M leg carries the old tier's value:**
   - M at 192 costs +9.6 %; A at 192 costs +1.1 %.
   - M at 128 costs +47 %; A at 128 costs +8.8 %.
   - The A leg tolerates rank 192 almost freely. The M leg cannot go below 256 any more than the shared basis
     can (O1: shared 192 +8.0 %, shared 128 +41 %).
3. **Why no arm pays:** the split's fixed overhead (+2.26 % F) uses up what a smaller A leg saves. 192/256 saves
   only 0.45 % F against v84 for +1.07 % raw, and carries +2,918 ops (≈ +1 s median server wall).
4. **A usable lesson for the shared basis:** the join's Gram weighting decides which leg the 256 directions serve.
   Since M dominates, a shared basis weighted toward M is a free (zero-F) variant. It is not on this lane's list, and
   O1 shows the energy is already 99.9 % captured, so I do not expect it to move raw by more than ~0.3 %. It is not
   run here.

Versus predictions:
- strategist: one leg tolerates 192 at ≤ +3 % ✓ (A: +1.07 %), P ≈ 0.3. It tolerates it, but the split overhead
  means it does not pay.
- mine: 256/256 −0.5 to −1.5 % ✗ (−0.19 %, smaller); 192 legs +0.2 to +2 % (A ✓, M ✗ at +9.6 %); 128 legs +2 to
  +7 % (A ✗ +8.8 %, M ✗ +47 %); a strong asymmetry ✓; P(any net-negative) 0.3 → none.

**No O2 hand-over to Lane P.**
