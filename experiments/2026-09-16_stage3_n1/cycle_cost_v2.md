# Stage 3 / N1 v2 — cheaper EXACT join-(b) spellings (COST ONLY)

`local_sim/n1_cycle_meter.py --v2` (sha256 `a36e119d…`), raw numbers in
`cycle_cost_v2.json` (sha256 `42bf18e3…`). Real flopscope 0.12.1 meter, float32
random price operands, n = 1024, depth 16. r ∈ {128, 150}, s ∈ {32, 64}, a ∈ {2, 3, 4}.
The metered v31(gens=a) whole chains are reused from `cycle_cost.json` (same
`estimator_k3_v31.py` sha, checked by the script). Age-gate convention is unchanged
from `cycle_cost.md`: recent tier = at most `a` dense lineages, `15 - a` joins
(1 init + `14 - a` join-(b)), `14 - a` D21+D3 queries, 1 D3-only query.
Not an accuracy result.

## Decision gate: NOT MET

**No join-(b) configuration at r = 128, a ≤ 3 comes in at ≤ ~0.17 B.** The cheapest is
**0.1790 B at r=128, s=32, a=2**, which needs symmetric declaration. At a=3 the
cheapest is 0.2035 B. Without declared symmetry: 0.1948 B at a=2 and 0.2181 B at
a=3. Nothing at r = 150 or s = 64 gets under 0.22 B.

**Why no spelling of this join can pass at a=3 (r=128).** Everything except the
joins already costs 0.1349 B: recent tier 0.1179, transport 0.0013, declared queries
0.0132, init 0.0025. That leaves ≤ 0.035 B for 11 joins, i.e. ≤ 0.0032 B per join.
Building the lineage core alone costs 0.0037 B at s = 32: four einsums over the
n = 1024 source columns, 2 × 0.00123 + 2 × 0.00061. The full join costs 0.0062 B. The
floor is the O(n (r+s)^3) projection of the n source columns. It is not the
rotation or the truncation, and s → 0 is simply join (a) (fixed-basis).

Operations: 13,030–13,864 in every configuration (gate 22,000).

## What each cheaper spelling bought (r=128, s=32, per join; v1 = 0.00723 B)

| change | effect |
|---|---|
| (1) rotate with R11 (r×r) and fold into the (r,r,·) block | 0.00116 → 0.00073 on the rotation einsum; join 0.00723 → 0.00681 (mm Gram) |
| mode Gram as `einsum('ajk,bjk->ab', Gx, Gx)` instead of reshape+matmul | 0.00060 → 0.00030; join → 0.00651 |
| (2) block update (existing r directions / new s directions) | **0 FLOPs saved** (0.00651 either way): the aliased einsum on the full (m,n) legs already gets the (0,1) pair discount, and the blocks just re-split the same n·m^3. It lowers the join's peak temporary (143 → 95 MiB) and adds 7 operations |
| (3) declared-symmetric core (canonicalise + `as_symmetric`, billed) | truncation einsum 0.00116 → 0.00092, Gram 0.00030 → 0.00015, and the per-layer query halves (0.00208 → 0.00111); join → **0.00624**; +36 operations per join |

Best join per call: r=128 → 0.00624 (s=32), 0.00981 (s=64); r=150 → 0.00965 (s=32),
0.01427 (s=64). The best spelling is `rot-tri`, block or full core (equal FLOPs),
`gram-ein`, plus `sym-1` when declaration is allowed.

## Self-check (float64, n=12): PASS, worst 2.2e-14 over 144 new checks

For all 16 v2 variants, starting from a **non-orthonormal** (transported-like) old
basis, so R11 is a real rotation:
* at r+s = n it is exact: `G_new(V,V,V)` = old + new lineage (≤ 6.0e-15);
* truncated to r=8 it equals the total tensor projected onto span(V_new) in every mode
  (≤ 6e-15);
* V_new is orthonormal (≤ 1.6e-15);
* span(V_new) equals the **dense HOSVD subspace** (top-8 left singular vectors of the
  exact total's mode-0 unfolding): projector difference ≤ 2e-15. So the eigh-kept
  directions are the right ones, not just some subspace;
* declared variants: the canonicalised core is bitwise symmetric (deviation 0.0) and
  equals the plain one.
The v1 checks (definition vs v31 slices, queries, join (a), transport, rebase, join (b)) still pass.

## Hazard found: flopscope's symmetry declaration at float32

* `as_symmetric` **refuses** a float32 core produced by `sym3` or even a
  6-permutation average once magnitudes are realistic (deviation 0.086 at |G| ≈ 1e5,
  tolerance atol 1e-6 / rtol 1e-5 per entry). The declared variants therefore
  **canonicalise**: each entry is gathered from its sorted-index orbit
  representative (`take`, ≈ 20 m^3 FLOPs, 18 operations). Once per join is billed
  (plus once after init in declared chains).
* An einsum whose output flopscope **infers** to be symmetric (declared input, with
  the same factor object on every mode) validates its float32 output and **raises**
  `SymmetryError(max deviation = inf)`. That would be a failed MLP on the grader.
  So the declared rotation is not used: the rotation reads the plain core, and the
  truncation passes three distinct views of E so no output symmetry is inferred.
  **Any estimator using declared symmetry must avoid this pattern everywhere.**
  Whether the server's flopscope-client behaves the same is unverified.
* Local flopscope overhead (ADVISORY) of a declared join is 0.14–0.17 s at r=128,
  s=32, versus 0.05–0.07 s undeclared. Over 12 joins that is ≈ 2 s locally, about the
  size of v31's whole-chain local overhead of 2.6 s (graded residual 0.14 s). Residual
  risk must be measured with the subprocess runner.

## Largest temporaries

tracemalloc peak increment per join step: 95–251 MiB at r=128, s=32; ≤ 430 MiB at
r=128, s=64; ≤ 595 MiB at r=150, s=64 (declared variants hold the canonical copy and
index array). Queries 129–176 MiB. LAPACK workspace not included. Whole-chain v31
peaks (from v1): 0.86–1.25 GiB at gens 2–4. All far below 8 GiB.

## Still not priced

The output-aware/relevance basis (O(L n^3) dense or O(L n^2 q) probed); power
iterations in the sketch; a cheaper-than-O(n m^3) projection of the n source columns
(the actual floor: it would need compressing the lineage's columns before the core is
formed, which is no longer exact); accuracy; server residual; Linux memory enforcement.

## Chains

| symdeclared allowed | r | s | a | C/B | operations | join spelling chosen | join C/B per call | old-tier C/B (join / query / transport) | join step temp MiB |
|---|---|---|---|---|---|---|---|---|---|
| False | 128 | 32 | 2 | **0.1948** | 13103 | `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.00651 | 0.0806 / 0.0269 / 0.0014 | 95 |
| False | 128 | 32 | 3 | **0.2181** | 13030 | `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.00651 | 0.0741 / 0.0248 / 0.0013 | 95 |
| False | 128 | 32 | 4 | **0.2385** | 13486 | `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.00651 | 0.0676 / 0.0228 / 0.0012 | 95 |
| False | 128 | 64 | 2 | **0.2411** | 13103 | `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01036 | 0.1268 / 0.0269 / 0.0014 | 136 |
| False | 128 | 64 | 3 | **0.2605** | 13030 | `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01036 | 0.1165 / 0.0248 / 0.0013 | 136 |
| False | 128 | 64 | 4 | **0.2771** | 13486 | `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01036 | 0.1061 / 0.0228 / 0.0012 | 136 |
| False | 150 | 32 | 2 | **0.2557** | 13103 | `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.01014 | 0.1254 / 0.0427 / 0.0017 | 137 |
| False | 150 | 32 | 3 | **0.2741** | 13030 | `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.01014 | 0.1153 / 0.0394 / 0.0015 | 137 |
| False | 150 | 32 | 4 | **0.2896** | 13486 | `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.01014 | 0.1052 / 0.0361 / 0.0014 | 137 |
| False | 150 | 64 | 2 | **0.3160** | 13103 | `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01517 | 0.1858 / 0.0427 / 0.0017 | 194 |
| False | 150 | 64 | 3 | **0.3294** | 13030 | `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01517 | 0.1706 / 0.0394 / 0.0015 | 194 |
| False | 150 | 64 | 4 | **0.3399** | 13486 | `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01517 | 0.1555 / 0.0361 / 0.0014 | 194 |
| True | 128 | 32 | 2 | **0.1790** | 13553 | `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00624 | 0.0774 / 0.0143 / 0.0014 | 243 |
| True | 128 | 32 | 3 | **0.2035** | 13444 | `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00624 | 0.0711 / 0.0132 / 0.0013 | 243 |
| True | 128 | 32 | 4 | **0.2252** | 13864 | `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00624 | 0.0649 / 0.0121 / 0.0012 | 243 |
| True | 128 | 64 | 2 | **0.2219** | 13553 | `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.00981 | 0.1203 / 0.0143 / 0.0014 | 411 |
| True | 128 | 64 | 3 | **0.2429** | 13444 | `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.00981 | 0.1105 / 0.0132 / 0.0013 | 411 |
| True | 128 | 64 | 4 | **0.2609** | 13864 | `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.00981 | 0.1007 / 0.0121 / 0.0012 | 411 |
| True | 150 | 32 | 2 | **0.2296** | 13553 | `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00965 | 0.1197 / 0.0224 / 0.0017 | 358 |
| True | 150 | 32 | 3 | **0.2501** | 13444 | `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00965 | 0.1100 / 0.0206 / 0.0015 | 358 |
| True | 150 | 32 | 4 | **0.2677** | 13864 | `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00965 | 0.1004 / 0.0189 / 0.0014 | 358 |
| True | 150 | 64 | 2 | **0.2850** | 13553 | `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.01427 | 0.1751 / 0.0224 / 0.0017 | 571 |
| True | 150 | 64 | 3 | **0.3008** | 13444 | `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.01427 | 0.1608 / 0.0206 / 0.0015 | 571 |
| True | 150 | 64 | 4 | **0.3138** | 13864 | `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.01427 | 0.1465 / 0.0189 / 0.0014 | 571 |

### Every metered step (C/B per call / operations / tracemalloc peak MiB / local flopscope overhead s)

| step | r=128 | r=150 |
|---|---|---|
| `canon_sym_r` | 0.00004 / 18 / 96 / 0.030 | 0.00006 / 18 / 155 / 0.056 |
| `transport_last_mm` | 0.00011 / 24 / 12 / 0.006 | 0.00013 / 24 / 12 / 0.008 |
| `transport_mm` | 0.00011 / 25 / 12 / 0.008 | 0.00013 / 25 / 12 / 0.011 |
| `transport_last_plain` | 0.00012 / 1 / 1 / 0.008 | 0.00014 / 1 / 1 / 0.000 |
| `transport_plain` | 0.00012 / 2 / 1 / 0.002 | 0.00014 / 2 / 1 / 0.003 |
| `query_d3_einsum_symdeclared` | 0.00099 / 4 / 129 / 0.004 | 0.00159 / 4 / 176 / 0.006 |
| `query_d21_einsum_symdeclared` | 0.00111 / 7 / 129 / 0.017 | 0.00173 / 7 / 176 / 0.020 |
| `query_d3_einsum` | 0.00196 / 4 / 129 / 0.010 | 0.00315 / 4 / 176 / 0.005 |
| `query_d21_einsum` | 0.00208 / 7 / 129 / 0.015 | 0.00329 / 7 / 176 / 0.017 |
| `init_einsum` | 0.00250 / 21 / 83 / 0.054 | 0.00380 / 21 / 117 / 0.079 |
| `jb2_rot-tri_core-block_sym-1_gram-ein_s32` | 0.00624 / 77 / 243 / 0.155 | 0.00965 / 77 / 358 / 0.234 |
| `jb2_rot-tri_core-full_sym-1_gram-ein_s32` | 0.00624 / 70 / 240 / 0.155 | 0.00965 / 70 / 354 / 0.240 |
| `jb2_rot-tri_core-block_sym-0_gram-ein_s32` | 0.00651 / 41 / 95 / 0.049 | 0.01014 / 41 / 137 / 0.066 |
| `jb2_rot-tri_core-full_sym-0_gram-ein_s32` | 0.00651 / 34 / 143 / 0.060 | 0.01014 / 34 / 192 / 0.095 |
| `jb2_rot-full_core-full_sym-1_gram-ein_s32` | 0.00666 / 63 / 238 / 0.171 | 0.01032 / 63 / 349 / 0.302 |
| `jb2_rot-full_core-block_sym-1_gram-ein_s32` | 0.00666 / 74 / 251 / 0.142 | 0.01032 / 74 / 369 / 0.256 |
| `jb2_rot-tri_core-block_sym-1_gram-mm_s32` | 0.00668 / 78 / 243 / 0.153 | 0.01040 / 78 / 358 / 0.224 |
| `jb2_rot-tri_core-full_sym-1_gram-mm_s32` | 0.00668 / 71 / 240 / 0.149 | 0.01040 / 71 / 354 / 0.254 |
| `jb2_rot-tri_core-block_sym-0_gram-mm_s32` | 0.00681 / 42 / 95 / 0.054 | 0.01064 / 42 / 137 / 0.066 |
| `jb2_rot-tri_core-full_sym-0_gram-mm_s32` | 0.00681 / 35 / 143 / 0.071 | 0.01064 / 35 / 192 / 0.105 |
| `jb2_rot-full_core-full_sym-0_gram-ein_s32` | 0.00693 / 27 / 150 / 0.067 | 0.01080 / 27 / 211 / 0.102 |
| `jb2_rot-full_core-block_sym-0_gram-ein_s32` | 0.00694 / 38 / 103 / 0.053 | 0.01081 / 38 / 147 / 0.070 |
| `jb2_rot-full_core-full_sym-1_gram-mm_s32` | 0.00711 / 64 / 238 / 0.174 | 0.01107 / 64 / 349 / 0.325 |
| `jb2_rot-full_core-block_sym-1_gram-mm_s32` | 0.00711 / 75 / 251 / 0.150 | 0.01107 / 75 / 369 / 0.247 |
| `join_b_v1_s32` | 0.00723 / 28 / 150 / 0.141 | 0.01130 / 28 / 202 / 0.161 |
| `jb2_rot-full_core-full_sym-0_gram-mm_s32` | 0.00723 / 28 / 150 / 0.059 | 0.01130 / 28 / 202 / 0.113 |
| `jb2_rot-full_core-block_sym-0_gram-mm_s32` | 0.00723 / 39 / 103 / 0.076 | 0.01131 / 39 / 147 / 0.096 |
| `jb2_rot-tri_core-block_sym-1_gram-ein_s64` | 0.00981 / 77 / 411 / 0.246 | 0.01427 / 77 / 571 / 0.302 |
| `jb2_rot-tri_core-full_sym-1_gram-ein_s64` | 0.00981 / 70 / 402 / 0.280 | 0.01427 / 70 / 559 / 0.411 |
| `jb2_rot-tri_core-block_sym-0_gram-ein_s64` | 0.01036 / 41 / 136 / 0.065 | 0.01517 / 41 / 194 / 0.090 |
| `jb2_rot-tri_core-full_sym-0_gram-ein_s64` | 0.01036 / 34 / 210 / 0.095 | 0.01517 / 34 / 271 / 0.134 |
| `jb2_rot-tri_core-block_sym-1_gram-mm_s64` | 0.01074 / 78 / 411 / 0.238 | 0.01570 / 78 / 571 / 0.307 |
| `jb2_rot-tri_core-full_sym-1_gram-mm_s64` | 0.01074 / 71 / 402 / 0.297 | 0.01570 / 71 / 559 / 0.382 |
| `jb2_rot-full_core-full_sym-1_gram-ein_s64` | 0.01081 / 63 / 409 / 0.363 | 0.01580 / 63 / 566 / 0.380 |
| `jb2_rot-full_core-block_sym-1_gram-ein_s64` | 0.01081 / 74 / 430 / 0.249 | 0.01581 / 74 / 595 / 0.420 |
| `jb2_rot-tri_core-block_sym-0_gram-mm_s64` | 0.01098 / 42 / 136 / 0.068 | 0.01612 / 42 / 194 / 0.100 |
| `jb2_rot-tri_core-full_sym-0_gram-mm_s64` | 0.01098 / 35 / 210 / 0.099 | 0.01612 / 35 / 271 / 0.133 |
| `jb2_rot-full_core-full_sym-0_gram-ein_s64` | 0.01136 / 27 / 247 / 0.105 | 0.01670 / 27 / 341 / 0.232 |
| `jb2_rot-full_core-block_sym-0_gram-ein_s64` | 0.01136 / 38 / 155 / 0.065 | 0.01671 / 38 / 218 / 0.077 |
| `jb2_rot-full_core-full_sym-1_gram-mm_s64` | 0.01173 / 64 / 409 / 0.294 | 0.01723 / 64 / 566 / 0.405 |
| `jb2_rot-full_core-block_sym-1_gram-mm_s64` | 0.01174 / 75 / 430 / 0.386 | 0.01724 / 75 / 595 / 0.636 |
| `join_b_v1_s64` | 0.01198 / 28 / 229 / 0.149 | 0.01765 / 28 / 296 / 0.218 |
| `jb2_rot-full_core-full_sym-0_gram-mm_s64` | 0.01198 / 28 / 229 / 0.103 | 0.01765 / 28 / 296 / 0.155 |
| `jb2_rot-full_core-block_sym-0_gram-mm_s64` | 0.01198 / 39 / 155 / 0.096 | 0.01766 / 39 / 218 / 0.107 |
