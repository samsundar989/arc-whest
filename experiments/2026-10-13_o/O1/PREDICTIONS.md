# O1 predictions (recorded 2026-10-12, before any O1 run)

Base: v84 = `fold+age6`, R8 params (`c1b098c0`, F/B 0.17354, 22,113 ops). Unrefit, VAL-100 (`hold100`), twin `--raw`
on the built file, paired per net against the v84 file evaluated the same way.

Exact prices (`variants/prices.json`):

| arm | F/B | ΔF | break-even Δraw | ops |
|---|---:|---:|---:|---:|
| o1 (SVD, r 256) | 0.17507 | +0.88 % | −0.88 % | 21,356 |
| o1:224 | 0.17169 | −1.07 % | +1.08 % | 21,104 |
| o1:192 | 0.16682 | −3.87 % | +4.02 % | 21,104 |
| o1:128 | 0.15797 | −8.97 % | +9.86 % | 20,456 |

The handoff's ΔF estimates were −1.5 / −3.5 / −8 %; the exact ones are −1.07 / −3.87 / −8.97 %.

**Strategist (handoff):** r 224 +2 to +5 % unrefit; r 192 +10 to +18 %; P(any smaller rank passes) ≈ 0.25.

**Lane O (mine):**
- o1 at 256 on age 6: −0.1 to −0.3 % raw (M6 measured −0.14 % on R2), net worse (+0.6 to +0.8 %).
- o1:224: +1.5 to +4 % (fails its +1.08 % break-even). o1:192: +7 to +15 % (fails +4.02 %). o1:128: +25 to +50 %.
- Diagnostic: the one-pass basis captures ≥ 0.98 of the top-256 energy at most joins (M6's −1.5 % gain from the
  exact basis at age 5 was small), and the top-224 / top-256 eigenvalue mass ratio is ≥ 0.97: the spectrum is flat
  near rank 256, which is why rank cuts cost far more than the SVD gives back.
- P(any smaller rank passes break-even unrefit) ≈ 0.15.

Refit rule (handoff): refit only arms whose unrefit Δraw is within 1.5× their break-even
(r 224 ≤ +1.6 %, r 192 ≤ +6.0 %, r 128 ≤ +14.8 %).
