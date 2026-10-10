> **Superseded in part (Oct 10 ~12:00 UTC):** §7 (plan rev 21), §8 (odds) and §9 (decisions) are replaced by `docs/research/2026-10-10_pivot_rev22.md` (plan rev 22). §1–§6 stand, except §6.3: the cost package there missed the two pruning levers (two-sided D, H kink-active) that rev 22 measures at about −14 % F.

# Deep review — 2026-10-10: how the estimator works, where the limits are, what can still move us

*Strategist session for Samuel, Oct 10 ~05:15–10:00 UTC. Inputs: repo `main` @ 20511c2 and local `lane-m` @ 623cdf7 (three read-only agents); the live AIcrowd overview, rules, board, submission pages, insights, resources and forum through Samuel's Chrome (~05:30 UTC); public GitHub/HF/forum OSINT; two literature agents (ARC's paper + finite-width theory; numerical methods); a numerical toy run in this session (§6.1); and an independent verification pass over this document (its corrections are folded in). Tags: [M] measured or primary source, [I] inference. Score basis: unless stated, scores are the **public-50** values shown on submission pages; the live board rounds to 2 significant figures and links each team to the earliest submission reaching its rounded best.*

## 0. Bottom line

1. **Where we are.** v81d_starall2 #334948 = **3.32003e-9** (raw 1.88328e-8 × F/B 0.17629), **#33 of 278** (35th contributor). Best public score J2W #335075 = **1.44247e-9** (raw 1.3545e-8 × 0.1065). The gap is **2.30× = 1.39× accuracy × 1.66× cost**; our per-MLP MSE is 1.19–1.64× J2W's on all 50 public MLPs (median 1.39) — systematic, not luck [M].
2. **Our architecture is near its own ceiling.** Compressions trade accuracy faster than cost (rank 192: +21 % raw for −5.75 % F; fold age 3: +31.6 % for −5.4 %); decided-neuron masks reached their modeled floor (F/B ≈ 0.17); several grafts that work in ARC's learning-free chain failed or shrank in ours (κ₄-core loop +11 % after refit; exact dressing correction +0.3–1.9 %), most plausibly because our ~37k learned parameters are fitted to incomplete physics [I] — though dressed legs (−6.5 %) and star-D4 (−3.4 %) did transfer; refits now give ~1 % per 2 epochs. Certain gains left: **cost −3 to −5 % F, refits −1 to −3 % raw ⇒ ≈ 3.05–3.2e-9** [M + I].
3. **The leaders are ahead in engineering, not in kind.** Nobody is near complete-K=4 accuracy (best raw 1.14e-8 vs ~2e-9 for exact K=4); several ship complete-third-order-level accuracy (raw ≈ 1.35–1.5e-8, marius 1.14–1.17e-8) at 0.11–0.14 B — i.e. in a representation ~1.5× cheaper than ours — with per-network adaptive pruning, op and wall budgets up to ~43k ops and 95–116 s, strong learned closures, and heavy iteration on the fixed public set (J2W: 425 entries) [M telemetry + I].
4. **Limits.** Label-noise floor 7.47e-11 (score ≥ 7.5e-12). Complete K=3 ≈ 1.5e-8; complete K=4 ≈ 0.11–0.15 × that ≈ **1.7–2.3e-9 (7–9× better)** but ≈ 2,000 B. At the 0.1 B floor the practical floor of known methods is ≈ 1.2–1.5e-9 — where J2W already is. Going below needs fourth-order content at third-order cost.
5. **New this session (§6).** (a) **κ₄ readout anatomy:** for a κ₄ block moved through dense random layers, D4 and D31 have no cross terms (exact readouts are O(n³) per block-layer), while D22 splits into an additive part, an O(n³) coherent remainder and n⁴ cross terms that are larger and uncorrelated with it (toy: the O(n³) part holds ~1/3 of D22's traceless content). Combined with our own anatomy — κ₄'s whole measured value in our chain sits in **Q31 = κ₄(a,a,a,b)** — this points the one remaining fourth-order bet at the **D31 channel**, decided by an offline regression against the exact pre-activation κ₄ labels we already have. (b) **Homogeneity:** an exact Ward identity (a pure random gain cannot change E[h_L]), a refined quenched gain law, and the measurement that the previous layer's norm carries all mean excess kurtosis at depth ⇒ a calibrated diagnostic and, if it fires, an O(n²) gain-consistent closure. (c) **`fnp.block`** and fused W-products are clean, unexploited cost levers.
6. **Honest odds** for the score prizes: P(top 20) ≈ 10 %, P(top 10) ≈ 2–3 %, P(podium) < 1 %; expected finish ≈ 3.0–3.2e-9 (rank ~28–35 as others also move). Robustness can add a few ranks in the private rerun if leaders running at the caps fail MLPs. The **$20k algorithmic-contribution prize** is the realistic cash route and our science is competitive (§7.3). Plan rev 21 (§7) banks the certain gains, runs two cheap upside bets with ≤ 24 h kill gates, protects A/B against rerun failure modes, and starts the write-up now.

---

## 1. How v81d works — the whiteboard version

### 1.1 The task
A fixed 16-layer, 1024-wide ReLU network with Gaussian random weights (W_ij ~ N(0, 2/n), no biases). Input x ~ N(0, I). Wanted: E[h₁₆] for each of the 1,024 last-layer neurons. Score = (MSE vs Monte Carlo labels over 10⁹ inputs) × max(0.1, FLOPs/2⁴¹). A forward pass costs 3.36e7 FLOPs, so the whole budget buys ~65k samples: Monte Carlo would **score** ≈ 1.1e-6 (raw MSE ≈ 1e-5 at the 0.1-floor sample count). Our score 3.32e-9 is ~340× better than sampling because we never sample — we **compute** the average by carrying a compact description of the distribution through the layers.

### 1.2 Why cumulants work
Each pre-activation z_a = Σ_i W_ai h_i is a sum of ~1,000 weakly dependent terms, so it is nearly Gaussian; the deviations are the higher cumulants (κ₃ = skewness tensor, κ₄ = excess-kurtosis tensor, …). In a wide random net each order is suppressed by roughly L/n = 16/1024 = 1/64. Mean + covariance (order 2) gives MSE 4.1e-6; exact third order ~1.5e-8; exact fourth ~2e-9. The game: carry as much κ₃ (and κ₄) as possible, as cheaply as possible.

### 1.3 What is carried from layer to layer
| object | size | meaning |
|---|---|---|
| μ | 1024 | post-ReLU mean (after the learned correction) |
| C | 1024×1024 dense | post-ReLU covariance |
| κ₃ "lineages" | 5 exact + 1 compressed tier | the skewness tensor (10⁹ numbers) in factored form |
| g, λ, star weights | vectors/scalars | a thin model of κ₄ |

### 1.4 One layer, step by step
1. **Linear step (exact).** μ ← Wᵀμ; C ← WᵀCW (a symmetric "sandwich", 1.75 matrix products); every κ₃ lineage is moved by multiplying its leg matrices by Wᵀ.
2. **Gaussian coefficients.** Per neuron: σ, α = μ/σ, and the Wick coefficients w_k = E[ReLU^(k)(z)] for a Gaussian z: w₀ = σφ(α)+μΦ(α), w₁ = Φ(α) (average slope), w₂ = φ(α)/σ (density at the kink), w₃ = −αφ/σ², w₄ = (α²−1)φ/σ³ — all closed form.
3. **Read the cumulant slices the ReLU needs:** D3 = κ₃(a,a,a), D21 = κ₃(a,a,b) (1024×1024), D4 = κ₄(a,a,a,a), D22 = κ₄(a,a,b,b), Q31 = κ₄(a,a,a,b).
4. **ReLU step (Edgeworth expansion).** E[f(z)] = E_Gauss[exp(Σ κ_r ∂^r/r!) f], truncated: new mean pK1 = w₀ + D3·w₃/6 + D4·w₄/24 + second-order terms; new covariance from the bivariate version: C·w₁w₁ + ½C²w₂w₂ + D21-terms·w₁w₂ + ¼D22·w₂w₂ + Q31-terms·w₁w₃/6 + 27 second-order monomials. Every approximate term carries a learned per-layer gain.
5. **Births: new skewness.** The kink makes new κ₃ every layer: neuron j near its kink (weight w₂_j) creates a "star" Σ_j w₂_j X_j⊗X_j⊗e_j, X_j its (dressed) covariance column — a CP-like factorization held as two n×n leg matrices (A_L = X, M = identity) plus thin extras (exact diagonal D3 scale, a separable κ_jji scale, a rank-4 residual, two "κ₄→κ₃ hub" terms). That bundle is one **lineage**; reading D21 from it is one product: [elementwise functions of A_L, M] @ [A_L | M]ᵀ.
6. **Aging and compression.** Each lineage costs ~4 products per later layer (2 to move, 2 to read) and one is born per layer, so naive cost grows like L². The **5 youngest stay exact** ("recent tier", 59 % of FLOPs); older ones are projected into **one shared 256-dimensional basis** ("old tier", 27 %), which works because legs passed through many random layers become nearly low-rank. In ARC's own chain this recipe costs only ×1.09 in MSE (Lane M3 M8) [M].
7. **κ₄ (thin model).** No κ₄ tensor: a per-neuron diagonal core g moved by (Wᵀ∘Wᵀ) — i.e. κ₄ treated as diagonal-born, κ₄ ≈ m·Sym(G⊗I) — (dG); a "λ × covariance" stand-in for the off-diagonal part (the κ₄ of a random overall gain, the "radial" mode); and since v81 a **star-D4** term read off the κ₃ legs: D4[a] += 4·Σ_j w₃_j A_aj³ M_aj (free fourth-order content; −8.35 % graded together with its refits).
8. **Learned layer (~37k parameters).** A 33→32→32→1 MLP per layer corrects each neuron's mean inside the loop (−5.2 % when added); ~1,000 per-layer/per-term gains; small gate MLPs rescaling births. All fitted end-to-end through a differentiable torch copy of the shipped file (the "twin") on 900 organizer nets; selection on a 100-net holdout; acceptance on freshly minted nets with our own labels. Graded results have always come in slightly worse than projected (+0.35 to +1.18 % for v79–v81d).
9. **Cost tricks.** Dead-neuron pruning (KAS: drop rows/columns of dead neurons, α below about −2 to −2.5, by fixed per-layer counts); 5-level batched Strassen–Winograd (0.598 of a dense product, glue included); symmetric covariance sandwich; exact closed form for layer 1; variance-only covariance at the last layer.

### 1.5 Where the FLOPs go (v81d ≈ 180.6 billed units; 1 u = one 1024³ product ≈ 0.098 % of budget) [M v77 census, scaled]
| item | u | % |
|---|---:|---:|
| recent-tier D21 readout | 55.7 | 30.9 |
| recent-tier transport | 51.2 | 28.4 |
| old tier: rebuild legs Q·[FA\|FM] | 17.9 | 9.9 |
| covariance WᵀCW | 16.3 | 9.0 |
| old tier: join (range finder, QR, re-rotation) | 15.7 | 8.7 |
| old tier: D21 readout | 13.4 | 7.4 |
| everything else (closure, P2, hubs, D3, corrector 0.07, star 0.3) | ~10 | ~5.6 |

Inside the products: leaf multiplies 146 u, Strassen glue 20 u (pre/post adds 8, input stacks 8, output concatenation 4). Per layer: ~1 raw u at layer 1 rising to ~24 at layers 11–15.

### 1.6 Where the error goes [M]
- Per-layer MSE (#334948): 6.9e-10 at layer 1 (pure label noise — that layer is exact) → 2.0e-9 (L5) → 6.8e-9 (L9) → 1.32e-8 (L13) → **1.88e-8 (L16)**, growing faster than linearly.
- 79 % of the final error is carried in; each layer contracts incoming mean error (×0.60–0.88) but adds new error that grows with depth (v55).
- Exact-substitution anatomy (v59): covariance channel 42 %, κ₃ 51 % (mostly all-distinct births/closure), κ₄ 15 % (**all via Q31**: rolled exact D4 and D22 made the chain 1.3 % / 6.6 % worse because the gains compensate), everything but the mean 80 %; with refit the κ₄ ceiling is 27.9 %.
- Errors concentrate at neurons near their kink (α ≈ 0), with no consistent sign.

### 1.7 Say it out loud (30 seconds)
"We push the input distribution through the layers analytically instead of sampling. A wide random layer keeps things nearly Gaussian, so we carry the mean, the covariance and the skewness tensor — the skewness in factored 'lineages' that each layer's ReLU kinks create and every later layer moves by matrix multiplication. At each ReLU an Edgeworth expansion turns those cumulants into the next layer's mean and covariance. Old lineages are squeezed into a shared low-rank basis, a small learned network fixes residual mean errors, and Strassen multiplication cuts the billed FLOPs."

---

## 2. The board (Oct 10 ~05:30 UTC) and our gap [M]

Full-precision best public-50 entries (submission pages):

| team (sub) | score | raw | F/B | entries | server wall max | residual max | setup |
|---|---:|---:|---:|---:|---:|---:|---:|
| J2W (#335075) | 1.44247e-9 | 1.3545e-8 | 0.1065 | 425 (106 failed) | 116.0 s | 0.375 s | 0.45 s |
| marius_binner (#335025) | 1.56781e-9 | 1.1673e-8 | 0.1344 | 222 | 114.4 s | 0.347 s | 0.27 s |
| Luna (#335099) | 1.56921e-9 | 1.3537e-8 | 0.1159 | 43 | 103.0 s | 0.322 s | 0.07 s |
| oliver_chen (#335098) | 1.60970e-9 | 1.232e-8 | 0.1307 | 53 | 97.5 s | 0.354 s | **3.60 s** |
| mliston (#333607) | 1.66660e-9 | 1.424e-8 | 0.117 | 71 (idle since Oct 2) | 83.6 s | 0.356 s | **4.32 s** |
| mk1981 (#335113) | 1.73980e-9 | 1.649e-8 | 0.1056 | 81 | 98.2 s | 0.365 s | 0.11 s |
| **neopok919 (#334948)** | **3.32003e-9** | **1.8833e-8** | **0.1763** | 52 | **90.0 s** | **0.274 s** | **0.036 s** |

- The board-linked (earliest) entries differ slightly: e.g. marius #333468 1.641e-9 (raw **1.141e-8** — the best raw on the board, at 0.1438), Luna #333961 1.646e-9. Board cuts: top 10 ≤ 2.0e-9, top 20 ≤ 2.6e-9, top 30 ≤ 3.2e-9.
- Gap to J2W 2.30× (raw 1.39× · cost 1.66×); to marius 2.12× (raw 1.61× · cost 1.31×). At our raw, cost 0.10 would score 1.88e-9 (≈ #7); matching #3 at the floor needs raw ≤ 1.57e-8 (−17 %); beating J2W at the floor ≤ 1.44e-8 (−23 %).
- J2W improved −1.35 %/day over Oct 1–10; at 0.5–1.35 %/day it reaches ≈ 1.31–1.39e-9 by Oct 17 [I].
- Selection (designation) is **not open**; the organizer's Oct 4 sequence was "fix flopscope → re-evaluate affected submissions → open selection one week before the deadline". flopscope is still 0.12.1; issues #275/#285 have no replies; Rules §5.3 forbids scoring-affecting changes in the last 7 days.
- Oct 5 ruling: building on another participant's MIT code is allowed with credit and substantial own changes; the write-up prize judges only your own contribution.

## 3. What the leaders are doing (evidence → inference)
1. **Same family as us.** Per-MLP error correlations with ours 0.67–0.76; similar depth profiles (all-layer/final 0.39–0.43); ARC's paper is the common source; Warm-up winners were "factorized third-cumulant propagation + learned networks on cumulant features" [M].
2. **Per-network adaptive pruning.** Leaders' FLOPs vary 1–3 % (CV) across MLPs and co-vary across teams at r = 0.90–0.99 — all scale work with the same per-network property (live/decided-neuron count). Ours are constant (fixed KAS tables) [M].
3. **Bigger runtime budgets.** 25–43k flopscope ops, residual 0.30–0.375 s, wall 95–116 s vs our 25.7k / 0.274 s / 90 s; billed throughput 2.9–5.3 GFLOP/s vs 8.8 for the public V29 clone ⇒ deep fast-matmul recursion until the wall binds [M + I].
4. **Complete-K=3-level accuracy at 0.11–0.14 B.** Most of the top fifteen sit at raw ≈ 1.42–1.58e-8 (ARC's unfactored AUGMENT ≈ 1.50e-8); marius −22 to −24 %, oliver_chen −18 %, J2W/Luna −10 % below it. A direct lineage implementation of complete K=3 in our representation would cost ≈ 0.3 B (§5), so they carry AUGMENT-level content in a cheaper representation (public candidates: κ₃ ≈ Σ Sym(a⊗A) and κ₄ ≈ Sym(M⊗C) matrix cores closed under W — kaileh57; multi-mode Λ Sym(B⊗B) — Puffi; ARC's radial pieces), plus partial fourth-order pieces and/or strong learned closures for the sub-K3 part [I, medium-low].
5. **Iteration on the fixed public set.** All submissions are graded on the same 100 MLPs; J2W has 425 entries, mk1981 uses all 10 daily slots. The private rerun uses fresh nets ⇒ some overfit risk [M + I].
6. **Several run at the hard limits.** J2W 116 s / 0.375 s; marius already failed one hidden MLP on residual (#333214, 0.409 s); oliver_chen/mliston setups of 3.6/4.3 s against a 5 s cap that fails the whole submission. A zeroed MLP costs ~0.7/N_MLPs — fatal in a rerun [M + I].
7. **No top-10 team has published code or a write-up.** Public mechanisms: 504aldo's V29 (raw 2.13e-8 @ 0.253 B) and derivatives (Hisernberg: 3.96e-9 with an LT corrector worth ~−10 %), kaileh57's August notes, Paul Rosu's warm-up experiments [M].

## 4. Limits — how accurate can anything get?
1. **Metric floor.** N = 1e9 labels and mean layer-16 per-neuron variance 0.0747 ⇒ label-noise floor **7.47e-11** (score ≈ 7.5e-12 at the 0.1 B floor). Leaders are ~150–180× above it [M].
2. **Cumulant ladder at 1024×16 (L/n = 1/64).** K=2 4.1e-6; K=3 SIMPLE 3.93e-8; K=3 AUGMENT factored 1.79e-8, +E21b 1.583e-8, unfactored ≈ 1.50e-8; **K=4 SIMPLE ≈ 0.11–0.15 × complete K=3 ≈ 1.7–2.3e-9** (M9: ratios 0.56/0.50/0.43/0.36/0.28 at n = 96/128/192/256/320, extrapolated in log-ratio vs log-width). The series is asymptotic, but at this width it should keep improving past K=4 [M + I].
3. **Compute.** No proven lower bound exists for a fixed budget (ARC §7.3 lists it as open). Exact factored K=4 ≈ 2×10³ B (κ₄ factors 0.27 TB).
4. **Practical floor of known methods.** Complete-K=3 accuracy, minus 10–20 % from learning, at the 0.1 B floor ⇒ **≈ 1.2–1.5e-9**. J2W (1.44e-9) is there.
5. **Our architecture's own ceiling.** ≈ 0.165–0.17 B and ≈ 1.83–1.87e-8 raw ⇒ **≈ 3.05–3.2e-9** with the certain items of §7 banked; lower only if an upside bet lands.

## 5. Diagnosis — why we are stuck
- **Cost floor of the lineage representation.** ~4 dense products per lineage per layer; 5 exact lineages ≈ 60 % of F, old tier ≈ 27 %, covariance 9 %. Measured trades all lose (above); dropping the old tier ×19; always-on columns are needed for the κ₄ hubs; always-on D21 rows +34 % raw if dropped [M].
- **Grafts transfer unevenly.** In ARC's learning-free chain: exact dressing correction −41 %, κ₄-core loop indispensable, gain-mode E21b −11.6 %. In ours: +0.3–1.9 %; loop +11.2 % after refit; E21b on top of the loop −5.6 % unrefit and −0.3 % after refit (the diagonal-only E21b route keeps 10–14 % of its gain even in ARC's chain). Dressed legs and star-D4 did transfer [M].
- **Complete K=3 does not pay in our representation.** Exact asymmetric dressing (3 legs per lineage) ×1.5 on the recent tier plus the loop (+30–45 % F) ⇒ ≈ 0.3 B at raw ≈ 1.6e-8 ⇒ ≈ 5e-9 [I].
- **The learned layer is saturating.** Corrector −5.2 %, 400-net refit −1.35 %, twin stages −0.5 to −2 % each, now ~1 % per 2 epochs and shrinking; extra features bought ≤ 1 % publicly (Hisernberg) [M].

---

## 6. New mathematics — what this session found

### 6.1 κ₄ readout anatomy, and why the one remaining fourth-order bet is the D31 channel
Take a κ₄ block born at layer s and moved through dense random layers with propagator P (P = W for one step; P = the lineage's own M leg for older blocks). For the repeated-index (2,2) block K_rc = κ₄(h_r,h_r,h_c,h_c) (K symmetric, zero diagonal) the readouts at the next ReLU are exactly (verified against an explicit 4-tensor to 1.7e-13):
- **D4[a] = 3·rowsum[((P∘P)K) ∘ (P∘P)]** — all four indices equal a, so **no cross terms: O(n³)**.
- **D31[a,b] = 3·[((P∘P)K ∘ P) Pᵀ]** — **no cross terms: O(n³)**.
- **D22[a,b] = [(P∘P)K(P∘P)ᵀ] + 2·Σ_{r≠c} K_rc P_ar P_br P_ac P_bc** — a coherent part (O(n³)) plus **cross terms (n⁴)**.
With P_ar² = 2/n + δ_ar the coherent part is exactly const + **additive** f_a + f_b (f = (2/n)·δK1, a mat-vec — the form our dG model and ARC's r = 1 core carry) + a non-additive remainder [δKδᵀ].

Toy (proxy block K = C², one random transport step, `scratchpad/k4x/k4_anatomy.py`, 3 seeds per width):

| n | additive rms | coherent non-additive rms | cross rms | cross / coherent non-additive | corr | O(n³) share of traceless D22 variance |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5.2e-2 – 1.3e-1 | 1.1e-2 – 3.0e-2 | 1.8e-2 – 3.5e-2 | 1.19–1.59 | ≈ 0 | 0.29–0.40 |
| 128 | 4.0e-2 – 4.8e-2 | 6.5e-3 – 7.7e-3 | 9.2e-3 – 1.27e-2 | 1.39–1.66 | ≈ 0 | 0.26–0.34 |
| 256 | 2.6e-2 – 3.7e-2 | 3.0e-3 – 4.4e-3 | 4.2e-3 – 6.0e-3 | 1.37–1.38 | ≈ 0 | 0.34–0.35 |

Readings:
1. **D22's traceless content needs n⁴.** The cross terms are ~1.4× the O(n³) remainder at every width and uncorrelated with it, so an O(n³) D22 readout recovers only about a third of the traceless D22 content per block (toy) [M toy].
2. **The toy's additive dominance does not transfer to real nets.** Our own anatomy shows the additive form (m/6)(dG_a+dG_b), even fed the exact diagonal, explains R² ≈ 0–0.06 of the true off-diagonal κ₄(a,a,b,b) (no family tried exceeded 0.19): real D22 is dominated by content outside this toy (other blocks, many layers of transport) [M].
3. **The channel that carries κ₄'s value in our chain is Q31 = D31** (exact Q31 removed 15.5 % unrefit; exact D4/D22 made things worse) — and D31 readouts are O(n³) per block-layer for every block type. The current λ·C surrogate explains R² ≈ 0.47–0.50 of D31 (its radial part) [M]. So the one fourth-order bet worth a day is: **which O(n³) pieces explain the rest of the true D31** (star D31 per lineage: 2 u; path D31; the newborn's (2,2)-block D31: 2 u; the (3,1) block) — measured offline against the exact pre-activation κ₄(a,a,a,b) labels we already hold (keenanpepper full1000, N = 1e9; the source used by Anatomy 2), then built only where the explained variance is large (Lane M4.3).
4. Lane M3's verdict stands for the full K=4 gain: corrections alone close about half of the K3→K4 MSE gap at n = 320 (0.66 vs 0.28 full), low-rank truncation fails as width grows, single birth classes fail; M3 left open whether *any* O(n³) representation tracks the asymptotic gain. Our toy adds a plausible mechanism for the low-rank failure — truncating a block's B_r corrupts the coherent quadratic forms p_bᵀB_r p_b, which depend on its whole spectrum — and suggests two-hub structures have the same anatomy; both are hypotheses [I].
5. Estimating the D22 cross terms with random features (X̂ ≈ mean_s (P diag(y_s) Pᵀ)∘(P diag(y_s) Pᵀ) with y_s drawn from a diagonally shifted K, the shift's O(n³) contribution subtracted) is unbiased but needs ~10–20 sandwiches per block-layer — far over budget [I].

### 6.2 Homogeneity: an exact identity, a refined gain law, and a cheap closure
- **Ward identity (exact).** For a bias-free ReLU net h_L(cx) = c·h_L(x) (c > 0), so for any positive random gain G independent of layer ℓ with E G = 1, F: law(h_ℓ) → E[h_L] satisfies F[law(G·h)] = F[law(h)]. Per unit Var G the cumulant change is Δκ₁ = 0, Δκ₂ = C + μμᵀ, Δκ₃ = 3κ₃ + 6·Sym(μ⊗C), Δκ₄ = 6κ₄ + 12·Sym(C⊗C) + 12·Sym(μ⊗κ₃) + O(Var G²), with **Sym = the average over index permutations** (verified on an exact discrete distribution). In 1-D the cancellation is exact for the Gaussian base: (w₂/2)(σ²+μ²) + w₃μσ² + w₄σ⁴/2 = 0; the κ-proportional shifts (3vκ₃, 6vκ₄) do not cancel inside a truncated readout, so even an exact-within-truncation chain responds at O(v·κ₃, v·κ₄) [M].
- **Quenched gain law (refinement of R2 §1.6).** n·Var_x log‖h_ℓ‖² ≈ 2 + 5·Σ_{j<ℓ}[1 − c(ρ_j)], with ρ_j the NNGP correlation of two independent inputs and c(ρ) = Corr(ReLU(u)², ReLU(v)²) (R2 used 1 − ρ_j). Predicts 16.8 / 23.9 / 30.8 at ℓ = 4/8/16 (R2's form: 14.9 / 19.9 / 24.5) vs **15.9 / 23.3 / 28.7 measured** on our nets (16.9 / 24.6 / 30.7 on two synthetic nets this session); it saturates near 44/n because 1 − ρ_j ~ 1/j². The annealed (over-weights) law would give 22/42/82 [M + I].
- **Radial mode (two synthetic 1024-wide nets).** Dividing z^ℓ by ‖h^{ℓ−1}‖ removes all mean excess kurtosis at layer 16 (0.054–0.063 → 0.000–0.001) and ~60 % of rms skewness (~84 % of its mean square); 89–90 % of the per-neuron skewness variance is linear in α — consistent with an independent radial factor (γ₁ = 6vα, γ₂ = 12v, v_eff ≈ 0.65·Var log q /4 ≈ 0.005 at layer 16; Var log q ≈ 0.030) [M synthetic]. This is the mechanism behind the λ law.
- **Use.** (i) A diagnostic: inject a pure gain and measure |Δμ₁₆|, calibrated against ARC's exact chains (whose response is truncation-only). The simplest exact injection is at the input: x′ = G·x has κ₄ = 12v·Sym(I⊗I), which maps exactly onto g₀ = 12v/m with λ = 0 (D22 = 4v) and needs no κ₃ injection. (ii) If our chain's excess response is material, an O(n²) **gain-consistent ("radial-split") closure**: estimate v_ℓ from the chain's own fourth moments (CV²(‖h‖²) − CV₀² ≈ 4v), deflate the radial parts of D3/D21/D4/D22/Q31 inside the readout only, restore second moments by e^v. Synthetic closure tests: 1.2–3.2× lower closure error; expected raw gain a few % (closure truncation is a second-tier error). The operating-point error may be smaller than the injected response suggests (λ is a per-layer table, the corrector was fitted at the natural v, and per-net v self-averages to ~3 %) [I].

### 6.3 Cost: billing facts and the remaining levers [M from flopscope 0.12.1 source]
- Bill = int(cost × dtype_rate × weight): matmul 2mkn − mn; float64 ×2; transcendental ops (exp, log, tanh, power, `**`) ×16; where/take/fancy indexing/sort ×4; `reshape`/`ravel` bill numel even as views; slicing/transpose/split free; `concatenate` numel(out); **`fnp.block` numel(out) once for any nesting**; setup unbilled; score = mean over MLPs of MSE × max(0.1, F/B).
- Low-risk levers: **`fnp.block`** for each Strassen level's output (−1.13 % F, −1,350 ops, bit-identical on three shapes); **fused products sharing Wᵀ_r** (−0.3 to −2 % F, −2k to −6k ops; watch transient memory and wall); deeper symmetric sandwich split (−0.3 %); `nan_to_num` instead of `where(isfinite)` (−0.04 %). Package **≈ −3 to −5 % F**. With accuracy checks: old-leg rebuild on kept rows (up to −1.5 %). Not recommended: merging the (2,1) readout call sites (+28 % wall when tried, GOTCHAS) and per-shape depth beyond what P3.1a measured (−0.6 to −0.9 % F at +8–24 % ops; only with ops freed by the fusions).
- Gates: our server telemetry (#334948: residual max 0.274 s at 25,741 ops ≈ 10.6 µs/op; wall max 90.0 s) leaves room under the caps (0.4 s, 120 s), but leaders running at the caps are a risk to avoid, not a benchmark; a ~2k-op relaxation would buy ≤ 1 % F.
- **Memory.** flopscope's session accumulator keeps every op record (~540 B): +13.9 MiB per predict at 25.6k ops ⇒ one worker crosses 8 GiB at ~150 predicts — **for A as well as B**. Fewer ops is the only in-rules fix; the ≥2³⁰ extra Strassen level adds ~1.45 GB to the B twin's VmPeak.

### 6.4 Checked and rejected (do not reopen)
Monte-Carlo hybrids (need a control variate with ρ ≥ 0.99992; antithetics 0.91×, radial conditioning 0.992×, QMC ~4×); TensorSketch/Hutchinson/hub subsampling of traceless κ₃/κ₄ (sketch size ~n^p/ε²); skew-normal, Gaussian-mixture, quartic-saddlepoint and max-entropy closures (3–10× worse than Edgeworth here); partial second-order Edgeworth terms without κ₅/κ₆ (κ₃κ₄, κ₄² double the closure error); TAP/Onsager terms (no coupling reuse in a feed-forward chain); unscented/cubature/Gaussian-sum propagation; complex packing, float16, ≥3-operand einsum, AlphaTensor/AlphaEvolve 4×4 schemes (mod-2 or complex only; the rational 48-mult scheme saves ≤ 1.3 %); Romberg blending across K (different diagram families; a K=2 chain costs +15 % F); per-net MC calibration of scalars (they self-average to ~3 %).

---

## 7. Plan rev 21 (Oct 10 → Oct 17)
Principles: bank certain gains early; every bet has a ≤ 24 h kill gate; protect the private rerun; start the write-up now.

### 7.1 Lane P4 (main) — score and hardening
- **Oct 10:** gate + submit `v81d_starall2ws` on the A/B slot; read v79–v81d server telemetry; update DESIGNATION.md; start the refit portfolio on RunPod.
- **Oct 10–11, P4.1 cost bundle → v82:** `fnp.block`, `nan_to_num`, fused Wᵀ_r products, sandwich split (bit-identical ⇒ FRESH-A not needed; any non-identical change runs FRESH-A); target −3 to −5 % F.
- **Oct 10–13, P4.2 refit portfolio** (2–3 RunPod pods + the local 5090 when free): cosine restarts from v81d params, lr ∈ {5e-5, 1e-4, 2e-4}; accept only on FRESH-A; target −1 to −3 % raw.
- **Oct 11–12, P4.3 memory-safe B** (block 40, no ≥2³⁰ level, fused-op savings; VmPeak projected ≤ 7,800 MiB at 250 predicts).
- **Oct 12–14, P4.4:** integrate accepted Lane M4 items; refit; FRESH-A; canary.
- **Oct 14–16, P4.5:** FRESH-LOCK (≤ 3), DESIGNATION.md, claims ledger; freeze A by Oct 17 12:00 UTC.

### 7.2 Lane M4 (lane-m) — two cheap bets + science
- **M4.1 (Oct 10–11):** gain Ward-identity test (input-level first), calibrated against ARC's exact K=3 chain; gate ⇒ M4.2.
- **M4.2 (gated, Oct 11–13):** radial-split closure in the twin, refit; hand over if ≥ 2 % raw.
- **M4.3 (Oct 10–12): the D31 channel** — offline regression of the true D31 residual on O(n³) pieces; build the winners with learned gains (layers where they explain most); refit; price; hand over if they pass the marginal rule.
- **M4.4:** write-up science (κ₄ anatomy on real labels, ablation map, gain law, Ward identity).

### 7.3 The write-up (due Oct 24 23:59 UTC; one Phase-2 submission ID graded before Oct 17 23:59 UTC)
ARC rewards mechanistic novelty. Candidates: dressed legs and the exact asymmetric identity (AUGMENT's heavy legs reproduced to the last digit); star-D4 riding on κ₃ legs; the cumulant-order map of the board; the K=4 headroom at n = 1024 and the κ₄ readout anatomy (§6.1); AUGMENT's non-separability; the refined quenched gain law and the homogeneity Ward identity (§6.2). Keep the 504aldo provenance separate (Oct 5 ruling). Recommended: a third Claude session drafts from Oct 11; Samuel reviews Oct 18–22.

## 8. Odds and risks
- Score prizes: P(top 20) ≈ 10 %, P(top 10) ≈ 2–3 %, P(podium) < 1 %. Expected ≈ 3.0–3.2e-9, rank ~28–35 (others move too); a few ranks can come back in the rerun if teams at the caps fail MLPs [I].
- Risks: (1) a flopscope re-billing before selection (we use no flagged patterns; some leaders may); (2) a long-lived rerun worker — memory growth threatens **A** (crosses 8 GiB at ~150 predicts) and today's B; (3) refits that overfit VAL (accept only on FRESH-A); (4) fused products raising VmPeak or wall (measure every build); (5) `fnp.block`'s billing rule changing in a flopscope fix.

## 9. Decisions for Samuel
1. Accept the framing (bank + two cheap bets + write-up), or order a high-variance representation rebuild instead (not recommended: ~5 % chance to land in 7 days).
2. RunPod: 2–3 pods Oct 10–14 for the refit portfolio and Lane M4 (~$150–300). Without them the local 5090 is the bottleneck and M4's 24 h gates slip.
3. Keep the current op/wall gates (recommended).
4. Organizer email (yours alone): does the private rerun keep one worker process across > 100 MLPs, and may `flopscope.budget_reset()` be called at the start of `predict`? Which flopscope version bills the rerun, and is a re-evaluation still coming? (The answer decides whether A needs an op diet too.)
5. A third Claude Code session for the write-up from Oct 11 (yes/no).
