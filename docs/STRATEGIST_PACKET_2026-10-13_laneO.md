# Strategist packet: Lane O, 2026-10-13 (session 2, branch `lane-m`)

Handoff: `docs/HANDOFF_2026-10-12_LANE_O.md`. Tables are under `experiments/2026-10-13_o/` (one RESULTS.md and one
PREDICTIONS.md per step, with predictions committed before each run).

**Common setup:**
- Base: v84 (`fold+age6`, R8 params) rebuilt through `tools/build_v85.py`; `--check-v84` passes (`c1b098c0`).
- Accuracy: unrefit VAL-100, twin `--raw` on each built file, paired per net against the v84 file (1.716117e-8;
  R8 recorded 1.716023e-8).
- Prices: exact interpreter.
- Screens ran directly (AGENTS rule 6), not through workerq: the queue's free-RAM gate was below 8 GiB all evening.

## Oct 12–13, day 1

**The old tier cannot be shrunk, split or replaced inside this tower.** O1, O2 and O3 are closed, and nothing goes to
Lane P.

| step | best arm | Δraw | ΔF | net | verdict |
|---|---|---:|---:|---:|---|
| O1 SVD join, r 256 | `o1` | −0.24 % | +0.88 % | +0.64 % | substitute for age 6 (as M6) |
| O1 SVD join, smaller r | `o1:224` | +2.46 % | −1.07 % | +1.37 % | closed (r 192 +8.0 %, r 128 +41 %) |
| O2 split legs | `o2:192/256` | +1.07 % | −0.45 % | +0.62 % | closed (the split costs +2.26 % F, +3k ops) |
| O3 (i) no old tier | `o3:9` | ×4.02 | −1.03 % | ×4 | closed (A = 8 ×6.4, A = 7 ×10.1, A = 6 ×15.8) |
| O3 (ii) harmonic tier | — | — | — | — | not built: needs 98 % recovery, 71 % measured |

**Three findings worth keeping:**
1. **The one-pass join basis already captures 99.90–100.00 % of the exact top-256 energy** at every join (2 FIT
   nets). The exact SVD has nothing left to fix, and age 6 had already taken the join repair.
2. **The final MSE weights the old tier's low-energy tail heavily.** Rank 224 drops 0.2–0.5 % of G's energy and costs
   +2.5 % raw; the raw cost grows 5–10× faster than the energy lost.
3. **The legs are asymmetric: M carries the old tier's value.** A at 192 costs +1.1 %, M at 192 costs +9.6 %. A at
   128 costs +8.8 %, M at 128 costs +47 %.

**On the leaders' bills:** content older than 8 generations is worth ×6.4 of our raw MSE. It is not low-rank (O1)
and not radial (O3, and the 09-26 harmonic measurement). So the second band (0.096–0.116 B without an old tier) is
not reachable by removing our old tier. A leader there carries old content in a representation we do not have, or
pays raw MSE elsewhere.

**Judgment call to review:** I closed O3 (ii) without porting `tools/build_harm.py` to v84. The kill rule (A = 8 (ii)
worse than +10 %) needs 98.1 % recovery of the drop's error; age 5 measured 71 %, and the D21 non-gain residual
(16–23 % of energy) caps recovery near 80 %. The arithmetic is in `O3/RESULTS.md`. Overrule if you want the
measurement anyway (about half a day).

**Still open:**
- O5 (anatomy on v84) is running. It is slow because the 5090 is shared at 99 % utilisation with full VRAM;
  this packet will be updated when it lands.
- O4 waits for Lane K's first pruned stack.

**Atoms delivered:** `tools/atoms/o1.py` (`o1`, `o1:<r>`), `o2.py` (`o2:<rA>/<rM>`), `o3.py` (`o3:<A>`). All
are documented, none recommended. `o1` remains the age-6-rejection fallback.

**Infrastructure note:** `workerq submit` from a worktree is refused ("passthrough 'cache/hf' escapes the
repository", because the worktree's cache/hf is a junction). Submit from the main tree with absolute paths.
