# STRATEGIST PACKET — 2026-10-08 (Lane P3, session 1)

**P3.0 done.** Rev 20 docs committed unchanged (6e66fbc). **v80_hygws graded #334781 3.7056e-9**
(raw 2.0582e-8, F/B 0.18004): new wall-safe B, −12.8 % vs v78; +1.17 % vs its FRESH-A projection
(v79 +1.18 %, v80 +1.15 %: FRESH-A ÷ 1.019 reads ~1.2 % optimistic for these builds). Label-variance
read recorded in RESULTS as a disclosure.

**P3.0.2 memory growth — a risk for Samuel.** v80 in ONE Linux process under RLIMIT_AS 8 GiB, 60
sequential predicts exactly as `subprocess_worker` runs them: flopscope's session accumulator keeps
every predict's op log (25,631 records ≈ 540 B each), so **VmPeak grows +13.9 MiB per predict,
linearly** (6,161 → 6,942 MiB over 60). It crosses 8,192 MiB at ≈ 150 predicts; a 200-MLP suite in
one worker would end at ≈ 8.9 GB. 100-MLP suites end at ≈ 7.5 GB (why every graded run passed).
No `budget_reset()` used. In-rules levers: fewer op calls (v80ws 23k ops → 12.5 MiB/predict) and a
lower per-predict transient peak (start ≤ ≈ 5.6 GB clears 200). Whether the private re-run is one
200-MLP worker is unknown (#18238 unanswered).

## P3.2 — the κ4-core loop in the twin: KILLED (both kill conditions met)

Implementation, not a proxy: the off-diagonal r = 1 core births (17 traced (2,1,1) + 15 (3,1) AUGMENT
terms, the cumulant conversions, K211) in `tools/twin/k4loop.py` match ARC's own
`factored_nonlin_kprop_k3` to **1e-15** (n = 24, float64, 3 random states). Our-tower K211
contractions match the chain's own (2,1) readout to float32 round-off (incl. the shared old tier).
Core memory Core ← WᵀCoreW, dG = its diagonal, D31 = m·Off/2 replaces (m/2)λC in Q31; the 211→111
radial term exact (own transport), via hub i (the chain's memoryless λC version), or folded.
VAL-100 free-running, paired vs the v79/v80-equivalent twin (VAL 2.1147e-8):

| arm | unrefit | after refit (S2tk, 1 ep) vs refit control 2.0873e-8 | strategist |
|---|---:|---:|---|
| (i) core + K211 exact + radial exact | ×1.466 (0/100) | **+11.2 %** (2.3210e-8) | −3 to −8 % |
| (i) radial via hub i | ×4.35 | — | |
| (i′) K211 truncated | ×1.821 | not refit (worse than (i) unrefit by 24 pp) | within 3 pp of (i) |
| (ii) (i) + E21b | ×1.384 | **+10.9 %** (2.3149e-8) | −8 to −18 % |
| (iii) (ii) + exact dressing correction | not run (kill margin 19 pp; P2.1b was +0.3–1.9 %) | | further −2 to −6 % |
| (iv) (ii) + old tier dropped | **×19.99** | — | +15 to +60 % vs v80 |
| control: old tier dropped, no loop | ×19.16 | | |
| (v) (ii) + old tier rank 128 | ×2.38 | — | |
| control: rank 128, no loop | ×1.98 | | |

Reading: **the κ4 core is not memory for old κ3 lineages in our chain** — dropping the old tier costs
×20 with or without the loop (ratio of ratios 1.04); at rank 128 the loop narrows the damage only
from ×1.98 to ×1.72. The loop itself is +11 % after the protocol refit (it recovered from +47 %
unrefit; the curve was flat at the end: 2.3200 → 2.3149e-8 over the last 28 steps). E21b is worth
−5.6 % relative unrefit, −0.3 % after refit. Cost (not metered): exact K211 over all lineages is
≈ 5 wide products per lineage per layer (far over budget); truncated K211 + folded radial ≈ 7 raw u
per layer, as the strategist estimated. **Kill:** best of (i)–(iii) +10.9 % (needs ≤ −8 %), (iv)/(v)
×2.4–×20 (needs ≤ +10 % vs v80). **Recommendation: no GO on v90; close the loop port.** Predictions:
mine (+30–70 % unrefit, −3 to +8 % refit) closer than the strategist's on the refit; both wrong on
(iv) in size (the strategist's +15–60 % assumed the core absorbs part of the old tier; it absorbs ~4 %).

## P3.1 — bank

| item | result | verdict |
|---|---|---|
| a. Strassen leaf (block 16/12/8, depth 5–7) | F −0.57 to −0.94 % at ops +8 to +24 % (we already run leaf 32 + a big-product level) | not worth the residual/wall; not shipped |
| **b. star-D4** (O(n²k), learned per-layer γ, init 0) | twin VAL **−2.00 % vs refit control**; γ learned 0 → 0.11–0.51, rising with depth and still moving | **shipped as v81_star** |
| c. per-layer variance-level gain | +0.004 % vs control (v80 ships no σ corrector, so it was run) | closed |

**v81_star** (`tools/build_star.py`: v80 + star-D4 + the star refit's S2t tables/corrector baked;
rebaking v80's own params reproduces v80 byte-exact): DEV raw 2.0074e-8, **1.0333× vs v80, 100/100**,
F/B 0.17616 (+0.08 %), ops 25,741, residual 0.285 s, projected 3.407e-9 (DEV). FRESH-A + WSL gate:
see RESULTS (running at the time of writing). B twin `estimator_k3_v81_starws.py` built (block 40).
A longer star refit (2 epochs, faster γ steps) is queued as a possible v81b.

**Oct 10 inputs from Lane P3:** P3.2 table above (loop killed); v81 (−3.3 % on DEV) as the banked
improvement; the memory-growth risk. Without the loop, P3.3 says: ship v81, harden, designate.

## Addendum — Oct 8 evening / Oct 9 00:30 UTC (graded)

| build | what | graded | vs prev |
|---|---|---:|---:|
| v81_star #334848 | v80 + star-D4 (recent lineages) + refit | 3.4984e-9 | −3.43 % vs v80 |
| v81b_star2 #334855 | longer star refit (2 ep; gains settle near 1, 1.2–1.5 at depth) | 3.3939e-9 | −2.98 % |
| v81b_star2ws #334918 | its block-40 B twin (Oct 9 A/B slot) | **3.4711e-9 (new B)** | −6.33 % vs v80ws |
| **v81c_starall #334926** | star over every lineage incl. the old tier (Q legs; elementwise only) | **3.3642e-9 (new best)** | −0.88 % |

Total today: **−7.1 % vs v80** at F +0.15 %. Closed: path-D4 (−0.42 % at +3.6 % F), variance-level gain
(null), Strassen leaf (−0.6/−0.9 % F at +8–12 % ops). Projections from FRESH-A ÷ 1.019 graded within
+0.4–1.0 % on all four rows. A 2-epoch continuation of the starall refit is running (possible v81d).

**Direction needed (Oct 10):** the loop is killed (table above), so P3.3's default is "ship v81, harden,
designate". Open for Samuel / strategist: (1) confirm no-GO on v90; (2) the memory-growth risk
(VmPeak crosses 8 GiB at ≈ 150 predicts in one worker); (3) what Lane P should do with Oct 10–16 — more
shared-factor fourth-order pieces (star D22/D31 cost 1–2 u per lineage; Lane M3's M10 sizes them),
or cost work toward the 0.10 floor (we are at 0.176).
