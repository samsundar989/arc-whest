# Strategist Packet — 2026-09-23 (evening): the Sep 23 slice is executed

**Self-contained. Assumes no repository access and no shared chat history.**
Answers `STRATEGIST_RESPONSE_2026-09-23.md` / `HANDOFF_2026-09-23.md`. Every
number is measured unless marked **[I]** (inference). Paths are given so a
Skeptic can check them; nothing here depends on reading them.

**The short version.** Every step of the Sep 23 handoff has run except 4b
(joint refit) and 4c (κ4 on M). The structural bet (N1) died on its own kill
criterion. Every accuracy lever came in at ≤ 0.4%. No graded row beats 5.32e-09
by 3%, and none is in sight on the current list. The one large result is about
**wall, not score**: the server's wall is set by Strassen leaf size, which gave
us a wall-safe entry at 60.7 s (v55: 114 s) for +7.3% score. We need a new
hypothesis for the leaders' cost/accuracy, and a decision on how to spend the
remaining 24 days between score and the write-up.

---

## 1. The competition, in one paragraph

ARC White-Box Estimation Challenge 2026, Phase 2. Predict every neuron's
expected post-ReLU activation of a random He-init ReLU MLP (1024 wide, 16 deep,
no biases) under x ~ N(0, I). Score = mean over MLPs of
`final_layer_MSE × max(0.1, FLOPs/2^41)`, lower better; λ = 0, so cost is the
FLOP multiplier only. Hard per-MLP limits: wall 120 s, "residual" (un-metered
Python dispatch ≈ 0.36 ms per flopscope op call on the server… see §3) 0.4 s,
memory 8 GiB; a breach FAILS the MLP (scores ≈0.9 against ≈5e-9). The prize is
a private re-run of up to two designated entries on fresh MLPs. Close Oct 17,
team freeze Oct 2, write-up Oct 24 (co-equal $20k contribution prize).

Leaders (board Sep 21): J2W 2.10e-09 (raw 1.64e-08, 0.131 B), marius_binner
2.40e-09 (1.68e-08, 0.146 B); 504aldo's public code clones at 5.40e-09. Leaders'
per-MLP FLOPs vary ±6% across nets (ours are constant) and their walls sit at
100–117 s, with TIME_EXHAUSTED failures on sealed nets.

## 2. Graded rows (all ours, 100 nets each, 0 failed in every row)

| sub | candidate | score | raw MSE | F/B | ops | worst wall (median) | note |
|---|---|---:|---:|---:|---:|---|---|
| 331816 | C10 | 6.491e-09 | 2.969e-08 | 0.2187 | 23,421 | 87.6 (83.1) | |
| 331863/939/989 | v50 | 6.192e-09 | 2.876e-08 | 0.2153 | 25,728 | 88.7 / 87.7 / **104.1** (84–87) | third run on a noisy server |
| 331880 | v53 (age 5) | 5.560e-09 | 2.434e-08 | 0.2285 | 25,140 | 96.5 (92.8) | |
| 331967/980 | **v55** (P2 rank 4) | **5.320e-09** | 2.461e-08 | 0.2162 | 23,645 | **114.5 / 113.8** (92.8 / 92.5) | best score; noisy server |
| **332063** | **v56b** (v55 at Strassen depth 4) | 5.708e-09 | 2.461e-08 | 0.2319 | 19,804 | **60.7 (54.5)** | wall-safe twin |

Chain: deterministic cumulant propagation — mean and full covariance exact; κ3
carried as per-layer factored "lineages", those older than 5 layers compressed
into a shared rank-256 basis `Q` rebuilt at every layer from 5 to 14; κ4
regenerated (exact diagonal + λ·C_off, fitted per-layer λ); Strassen products;
float32; stdlib + flopscope only. Local DEV-100 → graded prediction is within
0.2–1% on every row.

## 3. New findings since the Sep 23 response

### 3.1 Server wall is set by Strassen leaf size, not by FLOPs (the big one)

The graded reports split each net's wall into `kernel` (metered arithmetic) and
`overhead` (per op call). Across ten runs:

| run | Strassen leaf | kernel ps/FLOP | overhead µs/op |
|---|---:|---:|---:|
| F21, v31 (depth 4) | 64 | 90.0, 91.8 | 391–408 |
| C10, v50, v53 (depth 5) | 32 | 155–166 | 350–361 |
| v55 (depth 5, age 5) | 32 | 176 | 366–371 |
| **v56b (depth 4, age 5)** | **64** | **92.8** | 352 |

- Overhead is flat (~0.36 ms/op). **Kernel time per FLOP is ~1.9× higher with
  32-wide leaves**: depth 5 saves ~7% FLOPs and costs ~35 s of wall.
- Worst-of-100 = median × 1.03–1.055 on a quiet server, × 1.20–1.235 on a noisy
  one (identical v50 bytes: 87.7 s at 03:00 UTC, 104.1 s at 10:00 UTC Sep 23).
  The tail is the day, not the candidate.
- New model `median = 0.371 ms × ops + FLOPs × rate(leaf, age)`, conservative on
  all ten runs; v56b projected 62.1 s quiet, graded 60.7 s.
- **Depth 6 (leaf 16) is dearer in FLOPs**: F/B 0.2201 vs v55's 0.2162 (glue
  outruns the saving). Leaf 32 is the FLOP minimum in our spelling.

**[I] What this says about the leaders.** At 0.13 B ≈ 286 GFLOP and a 100–117 s
wall, they spend ~300+ ps per FLOP — slower per FLOP than our slowest
configuration. Either their work is dominated by small or elementwise kernels
(e.g. many thin per-net decompositions — consistent with data-dependent cost),
or they issue far more ops than we do. They are not buying FLOPs with deeper
Strassen in our spelling (it costs FLOPs, above).

### 3.2 N1 — sensitivity-weighted old-tier basis: KILLED on its own criterion

v55 at rank 512, first join (layer 5), basis eigen-rotated by its carried Gram
(rotation parity 5e-07); each of the 512 directions zeroed and the real chain
continued to the output.

| | net 0 | net 1 |
|---|---:|---:|
| Spearman(energy, ‖Δμ_L‖²) | **0.994** | **0.958** |
| top-256 overlap (energy vs sensitivity order) | 249/256 | 239/256 |
| sensitivity captured, energy-256 / sensitivity-256 | 98.83 / 98.87% | 98.32 / 98.48% |

Energy spans ~10⁵ from rank 1 to 512, sensitivity ~10⁴, **in the same order**.
Ceiling for any 256-of-512 selection (DEV 100, paired vs rank 256):
**rank 384 −6.56% raw, rank 512 −6.93%**. Predicted: Spearman < 0.7 and
sensitivity-256 ≥15% better (you); Spearman 0.6–0.85 (me). Both wrong.
The downstream transport does not selectively contract old-tier directions, so
balanced truncation has nothing to add. Arms (c)/(d) not run — bounded below 5%.

**Also:** at age 5 the rank above 256 is worth only 6.6–6.9% raw (at age 4 it
was 19.9%). The extra exact layer already holds what higher rank carried.

### 3.3 The accuracy levers of your §4 — all measured, all small

DEV 100, cupy, paired vs v55 (cupy/numpy panel agreement 3e-5). Costs from the
flopscope f64 grader twin (which matches the server's FLOP count exactly).

| lever | Δraw | ±95% | wins | ΔFLOPs | Δscore | your prediction |
|---|---:|---:|---:|---:|---:|---|
| P3 D21-feedback rank 16 | −4.57% | 1.48 | 72 | +6.63% | **+1.76%** | −4–8% score |
| P3 rank 8 | −3.28% | 1.48 | 66 | +3.64% | **+0.24%** | |
| adaptive λ, β=1 (both the lagged form and 504aldo's same-layer form) | −0.39% / −0.35% | 0.18 | 62–65 | ~0 | −0.4% | 0–8% |
| exact bivariate Gaussian covariance tail (Price's theorem, 4-node GL) | −0.23% | 0.11 | 65 | ~0.1% | −0.2% | 0–5% |
| tiered old-tier rank 192 from layer 9 | +12.3% | 2.0 | 0/20 | −4.23% | **+7.6%** | −3–5% cost at ≤2% loss |
| tiered 192 @11, 128 @13 | +17.9% | 1.8 | 0/20 | −4.88% | **+12.1%** | |
| tiered 192 from layer 13 | +2.5% | 0.6 | 0/20 | −1.44% | **+1.0%** | |

Two facts behind those numbers:
- Adaptive λ is small because r = mean(dG)/mean(var) varies only 1.8–9.0%
  (CV) across 100 FIT nets; the [0.5, 2] clip never binds. **Per-network λ
  adaptation is not where per-net variation lives in this problem.**
- The covariance's Hermite truncation (at 2nd order in ρ) drops ~1% of the
  late-layer Gaussian covariance (|ρ| rms 0.12, max 0.6 by layer 14); fixing it
  exactly is worth 0.23%. The closure's Gaussian truncation is not where the
  error is.

Corrections to the record: (a) **P3 had never been on** in any shipped file
(C10 → v55); the note "D2's base turns it on at rank 16" was wrong. (b) The
Sep 22 "rank taper is inert because no compression event occurs in the tail"
was wrong about the reason: joins happen at every layer 5–14; the taper was
inert because the range finder's sketch is the old `Q`, whose width fixes the
new rank.

### 3.4 Not yet run from your list
- **4b joint refit** (λ table × θ × feed on v55; + adaptive-λ β): D6 already
  found 0.00% from 302 further evaluations on v53's structure. Expect ≤1% [I].
- **4c κ4 on M for the exact tier**: untested; your range 0–3%. The worker's
  next task unless you redirect.

## 4. Designation, as it stands (policy pending Samuel)

| role | file | score | worst wall (noisy-day status) |
|---|---|---:|---|
| best score, **not designable** | v55 | 5.320e-09 | 114.5 / 113.8 s noisy — 6 s from the cliff |
| **A and B (recommended)** | v56b | 5.708e-09 | 60.7 s (one moderate run, worst/median 1.114) |
| dominated | v50 | 6.192e-09 | 104.1 s noisy |

Partial twins in between (accuracy identical to v55, ±0.01%; walls projected
from the measured kernel saving, **not server readings**): `tr64` (tower
transports at leaf 64) +2.14% FLOPs → ≈5.43e-09, ≈101 s noisy worst;
`tr64_wide64` +5.27% → ≈5.60e-09, ≈82 s. A `tr64` canary fails §5's 3% gate
(it is not a new best) and awaits Samuel.

## 5. The strategic problem

1. **The chain is at a local optimum.** Since Sep 21: fitting saturated (2.7%,
   then 0.00%), richer term sets lose on score, old content cannot be dropped,
   diagonalised, regenerated, re-ranked or tiered, and the consumer-level levers
   are each ≤0.4% or score-negative. The only large wins all came from one axis,
   the exact tier's age (4 → 5: −10.3%) and the P2 residual rank (32 → 4: −4.5%).
2. **The leaders' mechanism is still unidentified.** They carry old content for
   ≤ ~20 cost units where we spend 48; their cost varies per net (±6%); they
   spend ≥300 ps per FLOP. Every hypothesis we have tested about *how* has
   failed.
3. **Time.** 24 days to close; freeze Oct 2; write-up from Oct 4 per your §6.

## 6. The questions

1. Given §3.2 and the per-net cost variation, **what else could the leaders be
   compressing adaptively?** Candidates we have not tested: the exact tier's
   per-lineage content (adaptive age per net or per lineage, instead of a fixed
   age 5); the covariance itself (low-rank-plus-diagonal); the κ4 channel. Rank
   them by a 1-day kill test, as in §3 of your last response.
2. Age is the only axis that has paid (4 → 5 −10.3%, 5 → 6 −0.7% on score
   because FLOPs rise 5%). Is there a way to make *age cheaper* — e.g. carry the
   age-5/6 lineages in a cheaper form than full exact legs, or an adaptive age
   per lineage by its energy — that you would bet a day on?
3. Does 3.1 change your read of the leaders? Their slow per-FLOP rate is
   evidence for many small kernels. Does that point at a specific mechanism
   (per-net decompositions, iterative solves, per-lineage adaptive truncation)?
4. **Stop rule.** If 4c and your next probe both fail, should the remaining
   worker time go to the write-up (starting before Oct 4), rather than to more
   single-digit levers?
5. Designation: do you agree v56b (5.71e-09, 60.7 s) should be designate-B now,
   and that A should be the best-scoring config that projects ≤105 s on a noisy
   day (currently `tr64`, ≈5.43e-09, pending one server reading)?

## 7. What we are not asking
Anything about sampling, QMC, control variates, mixtures, generic learned
correctors (all measured dead), K4 or unfactored AUGMENT (priced out 6× on
score), or the organizers' wall-time loophole (we compete clean).

## 8. Confidence and caveats
- Probe numbers are cupy (panel agreement with the numpy instrument 3e-5); all
  graded rows came within 1% of projection.
- N1 was measured on two nets, at the first join only; the kill criterion was
  defined on that measurement, and the rank-512 ceiling (100 nets) bounds it
  independently.
- The partial-twin walls are projections from one ratio (v55 → v56b); the leaf
  mechanism itself is measured on ten runs.
- Every prediction written this slice by the strategist was wrong in the
  optimistic direction (P3, N1, tiers); the worker's predictions were right on
  P3 and adaptive λ and wrong on N1 and the rank-384 value.
