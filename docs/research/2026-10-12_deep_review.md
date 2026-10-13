# Deep review — 2026-10-12: how v84 works, why the leaders are ahead, the limits, and plan rev 24

*Strategist session for Samuel, Oct 12 ~20:00–23:30 UTC. Inputs: repo `main` @ fdd8e7f and `lane-m` @ e147904 (three read-only agents: architecture, evidence ledger, external code); the live AIcrowd overview, rules v14, board, all 7,392 Phase-2 submissions, 12 submission pages, insights and all 97 forum topics through Samuel's Chrome (read-only, 20:00–20:45 UTC); two web/literature agents; and three new experiments run tonight on the device VM against the real flopscope 0.12.1, the exact cost interpreter and the exact N = 1e9 labels (§5). Tags: [M] measured or primary source, [I] inference or judgment.*

## 0. Bottom line

1. **Where we are [M].** v84_age6 #335562 = **2.8498e-9** (raw 1.6422e-8 × F/B 0.17354), **26th of 288** by exact score. Leader marius_binner 1.4172e-9 (raw 1.134e-8 × 0.1250); J2W 1.4347e-9 (1.353e-8 × 0.1060). Top-3 cut 1.50e-9, top-5 1.55e-9, top-10 1.91e-9, top-15 2.23e-9, top-20 2.47e-9.
2. **Our bottleneck is cost, not accuracy [M].** mk1981 (rank 6, 1.728e-9) has *exactly our raw error* (1.6425e-8) at F/B 0.1052. At our raw, F/B 0.116 is rank 10 and 0.105 is rank 6. We pay 1.65× the 0.1 floor; the top ten pay 1.05–1.32×.
3. **Three things were measured tonight that change the plan.**
   - **(a) The Sep 22 "pooled `out=` is infeasible" finding is wrong.** `fnp.add/subtract/matmul/copyto(..., out=view)` works on flopscope 0.12.1 and bills exactly numel(out). Rewriting our Strassen glue that way is **bit-identical and −4.7 % F**; a new in-place alternative-basis variant is −5.5 %; with the deeper recursion this unlocks, **−8.7 to −12.9 % F**, and wall time should *fall* (§5.1).
   - **(b) The gain mode is one scalar for all of κ3.** D3 and D21 give the same v to ≤ 1.5 % at every layer; it is 77–84 % of D21's energy and 95 % of D3 on decided neurons. Supplying that part analytically on pruned rows cuts the pruning error 2.4× and makes kink-only D21 rows at τ = 2.5 and dead-row D3 pruning safe — the two prunings that were closed as "+34 %" and "+66 %" (§5.2).
   - **(c) Exact prices of the whole pruning space.** Pruning alone reaches F/B 0.149 (safe set) to 0.132 (aggressive); the **rank-256 old tier (39 u) is the barrier** between us and 0.105–0.115; without it the same pruned chain is 0.096–0.116 (§5.3).
4. **What the leaders are doing [I, strong].** The same lineage chain as ours, with (i) per-network kink-aware pruning in all three product dimensions, (ii) Strassen with zero-copy glue and smaller leaves, (iii) op and wall budgets pushed to the caps, and (iv) better third/fourth-order physics or learned closures worth 15–30 % raw. The exact prices of (i)+(ii) on *our* file land at 0.126–0.138 — where marius, oliver_chen, Masato and Andreas are. J2W/mk1981's 0.105 needs either τ ≈ 2 pruning or an old tier we have not found a cheap form for (§3).
5. **Limits [M + I].** Label-noise floor 7.5e-11 raw. Complete K = 4 ≈ 1.7–2.3e-9 raw but ~2,000 budgets. Best raw anyone has shown at *any* compute is 1.13e-8 (marius; his 0.925 B probe was no better than his 0.15 B entry), so the K = 3 family's practical floor is ≈ 1.1e-8 and the score floor of known methods is ≈ 1.1–1.2e-9 (§4).
6. **Plan rev 24 (§6): all lanes on cost for 48 h, then spend.** Four Claude Code lanes + two pods:
   - **S** — Strassen v2 (bit-identical first; canary Oct 13);
   - **K** — pruning v2 with analytic gain replacement;
   - **O** — the old tier and the accuracy spend items;
   - **P** — integration, refits, canaries, hardening.
7. **Honest projection [I].** Banked items (S + K): F/B ≈ 0.130–0.138 at raw ≈ 1.65–1.70e-8 ⇒ **≈ 2.15–2.3e-9 (rank ~14–16)** by Oct 15–16. With refits and one spend item: **≈ 2.0–2.1e-9 (rank ~11–13)**. Top 10 needs one more thing to land (old tier −10 u, or −10 % raw): P ≈ 15 %. Top 5 ≈ 2 %; winning < 1 %. The rerun can return 1–3 ranks: J2W's latest entry lost 10 of 100 MLPs to the residual cap.

---

## 1. How v84 works — the whiteboard version

### 1.1 The question
A fixed network: 16 layers, 1,024 neurons each, `h_l = ReLU(W_lᵀ h_{l−1})`, random Gaussian weights (variance 2/n), no biases. Feed it random inputs x ~ N(0, I). For each of the 1,024 last-layer neurons, what is its **average output**? The grader compares to a 10⁹-sample Monte-Carlo answer and multiplies the squared error by max(0.1, FLOPs/2⁴¹). The whole budget buys ~65,000 forward passes, which would score ~1.1e-6. We score 400× better because we never sample: we push a *description of the distribution* through the network.

### 1.2 Five facts that make it possible
1. **Each neuron's input is almost a bell curve.** z_a = Σ_i W_ia h_i adds ~1,000 weakly dependent terms. If z_a were exactly Gaussian with mean μ_a and standard deviation σ_a, the average of ReLU(z_a) has a closed form: σ_a·φ(α_a) + μ_a·Φ(α_a), with α_a = μ_a/σ_a ("how many σ the neuron sits above its kink").
2. **To get σ at the next layer you need the whole covariance matrix.** Var(z_a) = w_aᵀ C w_a, where C is the 1024×1024 covariance of the previous layer's outputs. So we carry C: one "sandwich" WᵀCW per layer, then a per-pair formula for what ReLU does to a covariance. Stopping here ("K = 2") gives MSE 4e-6 — 250× worse than us.
3. **The bell curves are slightly lopsided, and that matters at our precision.** The lopsidedness is the third cumulant κ3 (skewness). The mean shifts by −κ3·αφ(α)/(6σ²). Our target is 1e-4 per neuron; skewness moves means by ~5e-4.
4. **A neuron's skewness depends on the *three-way* table of the layer below** — κ3(i,j,k) for every triple, 10⁹ numbers. Moving that table through a layer densely costs n⁴ ≈ half the budget per layer. Nobody can.
5. **But new skewness is born in a very structured way.** A neuron j that sits near its kink bends the joint distribution of everything correlated with it. The newborn three-way table is a sum over those "hub" neurons j of (column j of the covariance) ⊗ (column j of the covariance) ⊗ (neuron j), weighted by the kink density w2_j = φ(α_j)/σ_j. So one layer's births fit in **two 1024×1024 matrices** (the "legs"): that pair is a **lineage**.

### 1.3 One layer of the chain
State carried: mean vector μ; covariance C; the stack of lineages; a thin fourth-order model (one vector `dG` and a scalar λ per layer).

1. **Linear step (exact).** μ ← Wᵀμ, C ← WᵀCW, and every lineage leg ← Wᵀ·leg. *Two matrix products per lineage.*
2. **Per-neuron Gaussian coefficients.** σ, α, and w1 = Φ(α) (probability the neuron is on), w2 = φ(α)/σ (density at the kink), w3, w4.
3. **Read what the ReLU needs from the lineages:** D3_a = κ3(a,a,a) (a vector) and D21_ac = κ3(a,a,c) (a matrix). D21 is one big product per leg pair: *two more matrix products per lineage.*
4. **ReLU step.** New mean = Gaussian formula + D3 correction + D4 correction. New covariance = w1_a w1_c·C_ac + ½ w2_a w2_c·C_ac² + ½ w2_a w1_c·D21_ac (+ transpose) + fourth-order and second-order terms. Every term has a fitted per-layer gain.
5. **Push the lineages through the ReLU** (multiply leg rows by w1: off neurons erase, on neurons pass) and **append the newborn lineage** (legs = dressed covariance columns, hub weights w2).
6. **A small learned network nudges each neuron's mean** (33 features → 32 → 32 → 1, one per layer, inside the loop).

So each lineage costs **4 products per layer** and a new one is born every layer: cost grows like depth². Our compromise: the **6 youngest lineages stay exact** ("age 6"); older ones are projected into **one shared 256-dimensional basis** (the "old tier") and updated there.

### 1.4 The fourth order, and the gain mode
ReLU networks without biases are scale-covariant: h(c·x) = c·h(x). So any random "loudness" factor riding on the signal passes through every later layer untouched. That single mode explains most of the non-Gaussianity at depth: skewness ≈ 6vα, kurtosis ≈ 12v, with v ≈ 0.002–0.005 growing with depth. Our chain models it loosely (a diagonal core `dG`, a "λ × covariance" stand-in, star terms read off the κ3 legs). §5.2 measures it exactly for the first time.

### 1.5 Where the 177.8 units go (1 u = one dense 1024³ product) [M, exact]
| component | u | % |
|---|---:|---:|
| exact lineages: moving the legs | 60.5 | 34.1 |
| exact lineages: reading D21 | 57.0 | 32.1 |
| old tier (rebuild 13.4, read 10.3, join 10.2, move 1.0) | 34.9 | 19.6 |
| covariance sandwich | 15.7 | 8.8 |
| everything else (closure, corrector, hubs, births…) | 9.7 | 5.4 |

Inside the products: Strassen leaf multiplies 148.6 u, Strassen "glue" (block additions, stacking, reassembly) 18.7 u. By layer: 0.7 u at L1 rising to ~15 u at L7–L15; the last layer alone is 8.4 u.

### 1.6 Where the error comes from [M]
- It **accumulates**: per-layer MSE 6.9e-10 (L1, pure label noise) → 2.0e-9 (L5) → 6.1e-9 (L9) → 1.17e-8 (L13) → 1.64e-8 (L16). 79 % of the final error is carried in; each layer damps incoming mean error (×0.6–0.9) and adds more.
- **It is not the formulas.** With exact cumulants fed in at every layer (corrector off) 96 % of the error disappears. The error is in what we feed the ReLU step.
- Exact-substitution anatomy (v82c chain): covariance channel 43 % (19 points of it the variance at layers 12–15), κ3 35 % (15 points at layers 14–16), κ4 9 % (all via Q31), old tier made exact 7 %.
- The missing physics is the third-cumulant content with three distinct indices that our births do not create (κ3×edge, the κ4→κ3 feed) — n⁴ objects with no known cheap exact form — and no learned or local fix has reproduced it.

### 1.7 Thirty seconds, out loud
"We carry the input distribution through the layers as a mean, a covariance and a factored skewness tensor. Skewness is born at neurons near their kink, as a sum of 'hub' terms that fits in two matrices per layer; every later layer moves those matrices with one multiply each and reads two slices with one multiply each. The six newest layers' worth are kept exact, older ones are squeezed into a shared 256-dimensional basis. An Edgeworth expansion turns those cumulants into the next mean and covariance, a small learned network patches the mean, and Strassen multiplication plus dead-neuron pruning keep the bill at 17 % of budget."

---

## 2. The board and the gap [M, Oct 12 20:20–20:45 UTC]

| # | team | score | raw MSE | F/B | entries | wall mean / max | residual max |
|---|---|---:|---:|---:|---:|---|---:|
| 1 | marius_binner | 1.4172e-9 | 1.134e-8 | 0.1250 | 250 | 81 / 85 s | 0.352 s |
| 2 | J2W | 1.4347e-9 | 1.353e-8 | 0.1060 | 441 | 106 / 114 s | 0.382 s |
| 3 | oliver_chen | 1.5045e-9 | 1.186e-8 | 0.1268 | 78 | 93 / 102 s | 0.318 s |
| 4 | Luna | 1.5344e-9 | 1.319e-8 | 0.1163 | 53 | 98 / 106 s | 0.353 s |
| 5 | mliston | 1.5454e-9 | 1.343e-8 | 0.1151 | 85 | 80 / 86 s | 0.372 s |
| 6 | mk1981 | 1.7283e-9 | 1.6425e-8 | 0.1052 | 90 | 91 / 104 s | 0.384 s |
| 7 | Masato | 1.7663e-9 | 1.386e-8 | 0.1274 | 28 | 88 / 95 s | 0.370 s |
| 8 | AndreasHad04 | 1.8397e-9 | 1.395e-8 | 0.1319 | 185 | 91 / 101 s | 0.384 s |
| 10 | suliman_tadros | 1.9123e-9 | 1.571e-8 | 0.1218 | 109 | 82 / 92 s | 0.374 s |
| 15 | jlacombe | 2.2315e-9 | 1.517e-8 | 0.1471 | 70 | | |
| 20 | a_s6 | 2.4740e-9 | 2.002e-8 | 0.1236 | 108 | | |
| **26** | **neopok919** | **2.8498e-9** | **1.6422e-8** | **0.1735** | 64 | 82 / 86 s | 0.240 s |

- Gap to #1: 2.01× = raw 1.45× · cost 1.39×. Gap to J2W: 1.99× = raw 1.21× · cost 1.64×. Gap to mk1981: 1.65×, **all cost**.
- Every top-10 team spends a different FLOP count on each network (±3–8 %); ours is constant.
- Leaders gain 1–4 % per day. marius went 1.568 → 1.417e-9 over Oct 9–12. Expect cuts of ≈ 1.3e-9 (#1), 1.8e-9 (#10), 2.1e-9 (#15) by Oct 17 [I].
- **Fragility at the caps:** J2W's newest submission #335578 lost 10 of 100 MLPs to the 0.4 s residual cap (score 1.28e-2). mk1981, Andreas and lode_dockx also sit at 0.38–0.39 s. The private rerun runs each MLP once.
- **Process:** the selection UI is still not live; flopscope is still 0.12.1 and Rules §5.3 forbids scoring changes in the final 7 days (the window opened Oct 11), so the promised re-billing is now unlikely before the deadline. Phase 2 closes Oct 17 23:59 UTC; write-up due Oct 24.

## 3. What the leaders are doing

**Evidence [M].**
1. Their per-network FLOPs vary 1–3 % (CV) and correlate across teams at r = 0.80–0.99: everyone scales work with the same per-network property. Our adaptive-count model of kink-active pruning produces the same spread (CV 2.6–3.0 %, min–max ±7 %).
2. Per-MLP errors correlate with ours at 0.67–0.78 and they beat us on every public net by a near-constant offset: same family, extra mechanism.
3. A newcomer (mk1981) went from the public 504aldo clone (0.253 B, 2.13e-8) to 0.110 B at 1.94e-8 **in 36 hours**: the cost recipe is short.
4. marius described his Strassen as writing "directly into preallocated views" and routing always-on/dead neurons separately; 504aldo's public code uses `out=` buffers; AndreasHad04 makes ~50,000 flopscope calls per MLP and mentions a "cap 3776" on columns; several teams run at 0.37–0.39 s residual and 100–114 s wall.
5. More compute did not buy them accuracy (marius 1.33e-8 at 0.925 B on Sep 26; 1.14e-8 at 0.15 B four days later). Their raw gains came in sudden steps at fixed cost (a new channel or fitted correction each time).
6. No top-15 team has published a Phase-2 method. keenanpepper's public datasets are built for training closures on thousands of nets; mliston's and Puffi's Phase-1 entries used offline-trained drift correctors.

**Reading [I].**
- **Cost.** The exact prices of our own file (§5.3) reproduce the leaders' band from two ingredients only: kink-aware pruning of D21 rows, hub columns and leg output rows (−14 % safe, −24 % aggressive), and zero-copy Strassen at smaller leaves (−9 to −13 %). Together: 0.126–0.138 with our old tier. That is marius / oliver / Masato / Andreas. Luna and mliston (0.115) need slightly more; J2W and mk1981 (0.105) need the aggressive set or a cheaper old tier.
- **Accuracy.** Eight of the top fifteen sit at raw ≈ 1.4–1.5e-8 (ARC's complete third order); marius and oliver_chen are 20–25 % below it. We are at 1.64e-8 with incomplete physics compensated by 37k fitted numbers. What exactly they add is unknown; candidates are an off-diagonal κ4 core with memory, the κ3×edge family (−11.6 % in ARC's chain), more exact ages (cheap once pruned), and learned closures on cross-neuron features.
- **They are not "past the research frontier" in kind.** Nobody is near fourth-order accuracy (≈ 2e-9 raw). They are at the third-order frontier with better engineering and 3–7× our submission count.

## 4. Limits

| level | raw MSE at 1024×16 | cost |
|---|---:|---|
| label noise (N = 1e9) | 7.5e-11 | — |
| complete K = 4 (extrapolated from n ≤ 320) | 1.7–2.3e-9 | ≈ 2,000 B |
| best shown by anyone, any compute | 1.13e-8 | 0.125 B |
| ARC complete K = 3 (unfactored AUGMENT) | ≈ 1.50e-8 | > 11 B as written |
| us | 1.64e-8 | 0.1735 B |
| K = 2 (covariance only) | 4.1e-6 | 0.016 B |

- No lower bound is proven for a fixed FLOP budget (ARC lists it as open).
- Sampling hybrids are closed by arithmetic and by measurement: reaching 1e-4 with ≤ 6,500 passes needs a ~10⁴× variance reduction; the best realised is 1.5–4×.
- The practical score floor of known methods is raw ≈ 1.1e-8 at the 0.1 floor ≈ **1.1e-9**. Going lower needs fourth-order content at third-order cost, which no one has.
- **For us this week:** F/B ≈ 0.126–0.138 with today's representation (§5.3); raw ≈ 1.55–1.65e-8 ⇒ 2.0–2.3e-9.

---

## 5. New measurements (Oct 12)

### 5.1 Strassen glue with `out=` views [M, real flopscope 0.12.1; scripts in `docs/research/2026-10-12_strassen/`]
- `fnp.add/subtract/multiply(a, b, out=buf[view])`, `fnp.matmul(A, B, out=view)` (batched, N-D), `fnp.copyto(view, a)` and `fnp.concatenate(..., out=view)` all work, including non-contiguous quadrant views and in-place aliasing, and bill exactly numel(out) (matmul: (2k−1)·numel(out)), one op each. `fnp.empty` is 0 FLOPs, 1 op. Slice assignment and `+=` still raise.
- The Sep 22 test (`RESULTS.md` § D3 item 3) tried `out=buf` and slice assignment, saw that the add itself bills the same, and concluded there was nothing to gain. The saving is in the **copies not made**: the `concatenate` that stacks the seven operand combinations and the `block` that reassembles the output.
- Issue #275 concerns `out=` *larger* than the operands. We use same-shape `out=` only (its own control case); a fix that bills by output shape leaves our bill unchanged. 504aldo's graded V29 uses the same calls.

Glue per recursion level, in quarter-block areas q: **ours 33** · 504aldo 22 (18 on a fused leaf) · **variant A 21** (our Winograd, operands written into the stacked buffer, post-adds into output quadrants) · **variant B 15 on the last level** (A + fused leaf) · **variant C 12** (new: Winograd in a Karstadt–Schwartz alternative basis, everything in place).

| v84 with… | A | B | C |
|---|---|---|---|
| same depths: ΔF / F/B / ops | −3.82 % / 0.1669 / 23,964 | **−4.73 % / 0.1653 / 24,576** | −5.52 % / 0.1640 / 24,775 |
| optimal depth, leaf ≥ 16 | −4.93 % / 0.1650 | −5.96 % / 0.1632 / 26,875 | −7.00 % / 0.1614 / 26,975 |
| optimal depth, leaf ≥ 8 | −7.29 % / 0.1609 | −8.70 % / 0.1584 / 30,017 | −10.74 % / 0.1549 / 30,025 |
| zero-pad to 2^d, leaf ≥ 8 | −8.46 % / 0.1589 | −10.09 % / 0.1560 / 30,401 | **−12.87 % / 0.1512 / 30,489** |

- **A and B are bit-identical to `_sb`** (`np.array_equal` on five shapes); C rounds differently with the same error size. The cost models equal the real meter to the FLOP on ~60 configurations.
- Depth break-even for one more level on a block (m, k, n): k + 3 > g_A·k/m + g_B·k/n + g_C, with (g_A, g_B, g_C) = (11, 11, 11) ours, (7, 7, 7) A, (4, 4, 7) fused, (3, 3, 6) C. That is why leaf 16 was "dearer" for us and profitable for everyone else.
- The two big product families are stuck at depth 5 because their dims are odd multiples of 32; re-rounding the KAS tables to multiples of 64 is worth a further 1–2 points.
- **Wall [I, modelled from our server calibration]:** the wall is glue-bound (≈ 1.1 ns per glue element vs ≈ 12 ps per leaf FLOP). Same depths ≈ 63–67 s (from 82 s); leaf ≥ 8 ≈ 74–85 s. Peak memory falls (1.47 GB vs 1.71 GB at half scale). Float32 error doubles per extra level (od16 moved raw +0.075 %).

### 5.2 The gain mode on exact labels [M, 10 FIT nets × 16 layers; `docs/research/2026-10-12_gain/`]
Formulas (σ-normalised: γ3 = D3/σ³, Y_ac = D21_ac/(σ_a²σ_c), ρ = C/σσ): γ3 = 6vα; Y_ac = 2v(2α_aρ_ac + α_c).

| layer | v from D3 (×1e3) | v from D21 | ratio | spike in C | D4 | gain share of D21 energy | of D3 (decided / kink) |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 2.22 | 2.24 | 1.007 | 3.74 | 1.35 | 0.49 | 0.95 / 0.68 |
| 4 | 3.12 | 3.13 | 1.004 | 5.02 | 2.19 | 0.66 | 0.96 / 0.76 |
| 8 | 4.26 | 4.26 | 0.999 | 6.24 | 3.49 | 0.77 | 0.95 / 0.73 |
| 12 | 4.93 | 4.91 | 0.995 | 7.04 | 4.50 | 0.82 | 0.95 / 0.69 |
| 15 | 5.24 | 5.14 | 0.981 | 7.45 | 5.17 | 0.83 | 0.95 / 0.66 |

- **One scalar per layer describes the gain part of all of κ3** (D3 and D21 agree to ≤ 1.5 %; v fitted on kept rows only gives the same number).
- **Across orders it is not one scalar.** The rank-one spike in C needs 1.4–1.7 v; D4 needs 0.5 v (L1) rising to 1.0 v (L15); D22 0.5–0.7 v. A Monte-Carlo check identifies them: with p = h·m/|m|² (mean amplitude) and q = |h − p·m| (fluctuation amplitude), the spike is Var(p) (ratio 1.00 ± 0.02) and the κ3 v is Cov(p, q) (0.91–0.98). The gain is a **two-amplitude** mode, not a single random scale. The project note's "+12v·μ·D3" term in D4 over-counts 3× when the total D3 is used.
- **Pruning test.** D21's first-order effect on next-layer variances is 3e-3 (L2) to 1e-2 (L14) relative; tolerance ≈ 1e-4.

| rows dropped from D21 | zeroed (max over layers) | gain formula on dropped rows |
|---|---:|---:|
| always-on + dead, τ = 3.0 | 4.3e-5 | 1.7e-5 |
| always-on + dead, τ = 2.5 | 1.8e-4 | **7.6e-5** |
| always-on + dead, τ = 2.0 | 6.5e-4 | 2.7e-4 |
| dead columns, τ = 2.5 | 1.2e-5 | 5.3e-6 |
| always-on columns, τ = 2.5 | 9.2e-3 (essential) | — |

| D3 of dead neurons | zeroed (rms mean error) | 6vμσ² |
|---|---:|---:|
| τ = 2.5 | 3.3–4.4e-5 (≈ +30–60 % MSE: the old "+66 %") | **1.0–1.4e-5** (≈ +3–5 %) |
| τ = 3.0 | 0.7–1.2e-5 | ≤ 3.5e-6 |

- So: kink-only D21 rows are safe at τ = 3 even zeroed, and at τ = 2.5 with the gain formula (marginal for L ≥ 10); τ = 2 fails. Dead-row D3 is safe at τ ≥ 2.5 with the gain formula.
- **What does not work:** the gain formula everywhere (no lineage readout) leaves 2–4e-3 — the residual is a real, broad-spectrum signal (top-16 singular modes hold 26–60 % of it) and is as large on always-on rows as on kink rows; it is droppable there only because of the small weight φ(α_a).
- Single-layer errors with everything else exact; compounding and the refit are not measured.

### 5.3 Exact prices of the pruning space [M, shape-only interpreter on patched copies of v84; `docs/research/2026-10-12_costfloor/`]
Base reproduced to the FLOP (381,613,795,475; 22,113 ops). Class counts: FIT-100 medians rounded to 32.

| lever (alone) | ΔF |
|---|---:|
| S1: D21 rows = kink \|α\| ≤ 2.5 (≤ 3.0; −2.5 ≤ α ≤ 3.0) | −5.9 % (−2.5 %; −4.2 %) |
| S2: hub columns kink at birth, both legs, \|α\| ≤ 2.5 (≤ 2.0) | −5.9 % (−12.1 %) |
| S2b: A-leg/star columns kink ≤ 2.5, M-leg on all non-dead ("two-column set") | −3.1 % |
| S3: leg output rows α > −3.0 (−3.5; −2.5) | −4.6 % (−3.3 %; −6.2 %) |
| last layer: only the 2–4 youngest lineages transported | −3.0 to −3.8 % |

| combination (age 6) | old tier r256 | + glue A, leaf ≥ 8 | no old tier | no old tier + glue |
|---|---:|---:|---:|---:|
| S0 (today) | 0.1735 | 0.1609 | 0.1350 | 0.1261 |
| C1 = S1(2.5) + S2(2.5) + S3(−3.5) | 0.1490 | 0.1384 | 0.1159 | 0.1085 |
| C2 = S1(3.0) + S2b(2.5) + S3(−4.0) | 0.1605 | 0.1493 | 0.1252 | 0.1174 |
| C3 = S1(2.0) + S2(2.0) + S3(−3.0) | 0.1316 | 0.1227 | 0.1018 | 0.0962 |

- With variant C and zero-padding the "+ glue" columns improve by a further ≈ 6 % (C1: ≈ 0.130).
- **The old tier costs 39 u = 0.0385 B = three exact ages**, and pruning barely shrinks it (its lineages were born in early layers where everything is kink-active). Old-tier rank 192 / 128 / 64 would save 8 / 16 / 24 u.
- Adaptive per-network counts leave the mean F unchanged and give a CV of 2.6–3.0 % — the leaders' fingerprint.
- The Strassen divisibility trap is real: nearest-16 rounding costs +3–4 %, raw counts +44 %.

---

## 6. Plan rev 24 (Oct 13 → Oct 17 12:00 UTC freeze)

**Principle:** below 0.1 FLOPs are free and above it every 1 % of F is 1 % of score. We are 73 % above the floor, with ≈ 25 % of cost removable by measured, mostly accuracy-neutral changes. Bank those first; in parallel look for the old-tier and accuracy items that decide top 10.

| lane | owner of | work | first deliverable |
|---|---|---|---|
| **S** (new session, branch `lane-s`) | `_sb`, `_mm`, `_depth_for` | S1 variant B (bit-identical) → S2 depth by the break-even rule (leaf ≥ 16, then ≥ 8) → S3 variant C → S4 KAS tables on multiples of 64 | v85 patch atom Oct 13 morning |
| **K** (new session, branch `lane-k`) | twin arms for masks | K1 kink-only D21 rows with gain replacement · K2 leg output rows α > −τ with gain D3/D4 · K3 two-column-set H · K4 last-layer transport of the youngest lineages only · K5 adaptive counts (optional) | unrefit VAL-100 screens Oct 13; refits Oct 13–14 |
| **O** (lane-m continues) | old tier, spend items | O1 SVD join + rank 224/192 · O2 split ranks for the A and M legs · O3 more exact ages + no old tier + C-mode tail · O4 age 7 / J on the pruned stack · O5 anatomy rerun on v84 | screens Oct 13 |
| **P** (main) | estimator files, RESULTS, canaries, pods | integrate S/K/O atoms, joint refits, FRESH-A, canaries (≤ 5/day + A/B), B twin, FRESH-LOCK, designation recommendation | v85 canary Oct 13 |

**Sequence and predictions (strategist, on record):**

| build | content | F/B | raw | score | rank today |
|---|---|---:|---:|---:|---|
| v84 | today | 0.1735 | 1.642e-8 | 2.85e-9 | 26 |
| v85 (Oct 13) | Strassen B, same depths; bit-identical | 0.1653 | 1.642e-8 | 2.71e-9 | 24 |
| v86 (Oct 13–14) | + depth by break-even (leaf ≥ 8 where wall and residual allow) | 0.156–0.160 | +0.1–0.5 % | 2.57–2.64e-9 | 22–23 |
| v87 (Oct 14–15) | + K1–K3 (safe thresholds) + joint refit | 0.136–0.142 | 1.64–1.70e-8 | 2.25–2.40e-9 | 16–18 |
| v88 (Oct 15–16) | + K4, variant C, tables on 64, one spend item, refit polish | 0.126–0.135 | 1.58–1.66e-8 | 2.0–2.2e-9 | 11–15 |
| stretch | old tier −10 u or −10 % raw | 0.115–0.125 | 1.5e-8 | 1.75–1.9e-9 | 7–10 |

**Gates (changes in bold).** Wall A ≤ 105 s noisy-implied; B ≤ 95 s; memory WSL 8 GiB PASS; parity ≤ 1e-3; hash = tested source; FRESH-A paired for anything not bit-identical. **Ops: the 26,000 cap becomes "projected residual ≤ 0.33 s at the per-op rate measured on the v85 canary"** (pooled buffers should lower the per-op residual; if they do not, depth stops where 0.33 s binds). **`out=` is allowed only with an output of exactly the broadcast operand shape**; `where=` and symmetry tags stay banned.

**Kill rules.** S: any non-bit-identical output in variant B stops the line. K arms: accept on the marginal rule after a matched-control refit; kill an arm at > +2 % raw after refit. O3: kill at > +10 % unrefit at A = 8. Any canary graded > 2 % worse than projected stops the line.

**Designation.** A = best score; B = its wall- and memory-safe twin with margins ≥ 12 % on every cap. Do not chase the leaders' cap-hugging: it is their rerun risk and our opening.

## 7. Considered and not taken (with reasons)
- **Exact gain-mixture closure** (integrate the ReLU step over the gain amplitudes instead of expanding in them). Estimated tonight from the measured amplitudes: the series converges fast — truncation after the κ4-equivalent term is ≈ 7e-6 per neuron for the mean and ≈ 2.5e-5 relative for variances, below the 6e-5 / 1e-4 budgets; and the learned closure that fitted 85 % of the closure residual offline returned only −0.75 % in the chain. Not a 2× lever; useful as write-up material.
- **Gain law in place of the old tier.** Earlier measurements stand (Sym(u⊗C) tier ×5.9–6.6); tonight's show why — the non-gain residual of D21 is 16–23 % of its energy with a flat spectrum. O3 retests it only at ages ≥ 8, where the pruned chain makes exact ages cheap.
- **"Vector ⊗ symmetric-matrix" modes** (κ3 ≈ Σ Sym(q_p ⊗ G_p), sandwich-priced, O(n²) readouts). Exact if the old M legs had rank r; random-matrix theory and our rank cliffs say r is in the hundreds, so the mode count is too.
- **Randomised contraction of the n⁴ diagrams, sampling hybrids, cubature, low-rank D21 readouts, bigger correctors, more data, more fitting:** closed by measurement here and in three public corpora.

## 8. Decisions for Samuel
1. **Lift the "no `out=`" rule** for same-shape `out=` into views (measured: billed numel(out); used by graded public code). Optional and yours alone: ask arc-whestbench@aicrowd.com for written confirmation — Rules §5.5 lets a participant rely on one.
2. **Run four Claude Code lanes (S, K, O, P) and two pods** through Oct 17.
3. **Replace the 26,000-op gate** by "projected residual ≤ 0.33 s", measured on the v85 canary.
4. The write-up stays deferred (your Oct 10 decision). Tonight's results — the single-scalar κ3 gain law, the two-amplitude gain covariance, gain-replacement pruning, and the 12-quarter in-place Winograd glue — are new and belong in it.
