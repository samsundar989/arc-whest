# A. Our estimator architecture — a technical description for a non-ML engineer and a strategist

*Snapshot read 2026-09-24/25 from `/home/claude/arc-whest` (read-only). Every number below is either quoted from a repo document (cited) or measured by me in a scratch process on the repo's own code (marked **[measured here]**). Anything I could not verify is marked **[uncertain]** or **[inference]**.*

---

## 0. Which file is "the architecture", and one thing to know about the repo copy

| file | what it is | status |
|---|---|---|
| `estimator_k3_v55_p2r4.py` | the pre-corrector chain (v53 + P2 rank 4) | graded 5.3200e-9 (#331967) |
| `estimator_k3_v57_corr.py` | v55 + the in-loop learned mean corrector, **generated** by `tools/build_v57.py` | graded **5.0443e-9** (#332129), best graded |
| `estimator_k3_v59_corr400.py` | same code as v57, corrector weights refit on 400 nets | not graded; derived 4.9763e-9 |
| `estimator_k3_v60_ws400.py` | v59 with `_STRASSEN_MIN_BLOCK = 64` (one-line diff, **[measured here]** by `diff`) | graded 5.3387e-9 (#332223), the "wall-safe" designation candidate |

**Caveat on the repo copy of v57.** The file at the repo root named `estimator_k3_v57_corr.py` currently has `_CORR_ON = False`, `_CORR_B64 = {}` and `_CORR_NF = 10` (lines 1625/1637/1653) — it is the *weightless template* that `build_v57.py` regenerated on Sep 24 17:02 for the v61 "recursive feature" experiment (RESULTS.md row `v61_rec`). The graded #332129 bytes (sha8 `253c7b19`) were the NF = 9 build with the 100-net weights injected. The NF = 9 code path is the one in v59/v60 (`diff` shows only `_CORR_ON`, `_CORR_NF`, the blob, and the removal of the `_CORR_CPREV` recursive channel). Everything below describes the NF = 9 graded architecture; line numbers refer to the repo-root v57 file unless stated.

---

## 1. The mathematical method, step by step

### 1.0 The problem in one paragraph

A network is 16 matrices `W_1 … W_16`, each 1024×1024, entries drawn i.i.d. N(0, 2/1024) ("He init"), no biases. Input `x ~ N(0, I)`. Layer `l` computes `h_l = ReLU(W_lᵀ h_{l-1})`. We must output, for every neuron of every layer, `E[h_l]` — the average over the random input — and only the **last** layer's row is scored (MSE against a 1e9-sample Monte-Carlo label, whose own noise floor is 6.8e-10; RESULTS.md §1a). Score = MSE × max(0.1, FLOPs / 2.2e12), so both accuracy and metered arithmetic count. A 0.4 s per-net "residual" (Python time between metered calls) and a 120 s wall are hard cliffs that zero a net.

**The method is deterministic moment/cumulant propagation, never sampling.** Instead of pushing random inputs through the network, we push a *statistical description* of the pre-activation vector `y = Wᵀ h` through each layer — its mean, its covariance, and truncated pieces of its third and fourth cumulants — and close the ReLU exactly under a Gaussian-plus-corrections assumption. The whole thing is float32, Strassen-multiplied, single-file, imports only `flopscope`, `flopscope.numpy as fnp`, `whestbench` and stdlib.

### 1.1 The state carried from layer to layer

`_layer()` (line 1757) receives `state = (mu, cov, tw, c4, gk4)`:

| symbol | shape | meaning (plain words) |
|---|---|---|
| `mu` | (n,) | the mean of the previous layer's post-ReLU output `h` |
| `cov` | (n, n) | the full covariance matrix of `h` (dense, symmetric) |
| `tw` | a `_Tower` object | the **third cumulant** (κ3, a would-be n×n×n tensor) of `h`, carried in factored form as a list of "lineages" — see 1.3 |
| `gk4` | (n,) | the per-neuron **fourth-cumulant core** `g`: a diagonal summary of κ4 (504aldo's "P1" mechanism) |
| `c4` | scalar | a legacy isotropic κ4 scalar; **carried but never consumed** (grep: only assigned at line 2112 and returned) — dead state |

At layer 0 the state is exact: `mu = 0`, `cov = I`, no tower, `gk4 = 0` (the input is Gaussian, so κ3 = κ4 = 0).

**Definitions (cumulants).** The mean is the 1st cumulant, the covariance the 2nd. The 3rd cumulant κ3 is the n×n×n tensor of "skewness-like" third-order correlations `E[(y_i−μ_i)(y_j−μ_j)(y_k−μ_k)]`; the 4th, κ4, is n⁴ and measures "kurtosis-like" excess over Gaussian. For a Gaussian both are zero; a ReLU output is not Gaussian, so they are nonzero and they matter for `E[ReLU]` downstream through terms like κ3_iii/6 · E[ReLU'''].

### 1.2 One layer, in order (what `_layer` does)

**(a) The linear step — "transport".** `y = Wᵀ h` is linear, so all cumulants transform exactly:
- `mu_pre = Wᵀ mu` (line 1801).
- `cov_pre = Wᵀ cov W` via `_p4_sym_sandwich` (line 1260): the second product is done as a half-height block plus one quarter block, and block 21 is block 12 transposed (symmetry) — pays 3/4 of the second matmul. At the last layer only the diagonal (the variance) is needed, so it computes `rowsum((Wᵀ C) ∘ Wᵀ)` — one product instead of two (P4 final-var, line 1802-1807).
- The κ3 tower is transported by `_advance` (line 1114): each factor matrix of every lineage is left-multiplied by `Wᵀ` — three wide matmuls for the whole tower regardless of how many lineages it holds.
- κ4 core: `dG = (Wᵀ∘Wᵀ) @ (g_prev − λ·diag(cov)) + λ·var` (line 1853). This is the "κ4 regeneration" identity: the model for κ4 of `y` is `m · Sym(G ⊗ I)` with `G = diag(dG) + λ·C_off`, where `C_off` is the off-diagonal covariance and **λ is a per-layer scalar** (`_K4_LAM`, line 137, values ≈ 0.005–0.010, index = *previous* layer) times a scale 1.225, a linear depth tilt −0.1, and a fitted 4-knot spline (`_own_lambda`, line 1589). In plain words: instead of carrying an n⁴ object, κ4 is *regenerated* each layer from a per-neuron diagonal carry plus "λ times the covariance", with λ fitted end-to-end. Consumers: `D4 = metric·dG` (κ4_iiii, `metric = 2.0`), `D22 = (metric/6)(dG_i + dG_j)` off-diagonal (κ4_iijj), and `Q31 = (m/2)·λ·C_ij` (κ4_iiij) in the closure extras.

**(b) Read the two κ3 slices the closure needs.** The ReLU closure never needs the full n³ κ3; it needs only
- `D3_i = κ3_iii` — the diagonal, computed elementwise from the tower (`_dslice3`, line 1312, O(k·n²), no matmul), and
- `D21_ij = κ3_iij` — the "(2,1) slice", an n×n matrix, computed by **the one big contraction of the layer** (`_dslice21`, line 1337): three wide products regrouped into `[LA | LP] @ [A_L | M]ᵀ` plus thin terms, with the sum over lineages performed by the contraction itself. Skipped entirely at the last layer.

**(c) The nonlinear step — the ReLU closure.** With `sigma = sqrt(var)` and `alpha = mu_pre / sigma` (clipped to ±40, `ALPHA_CAP`), the *Wick coefficients* are computed in closed form (`_wick`, line 336):

> **w(k, p) = E[ (d/dz)^k ReLU(z)^p ]** for `z ~ N(mu_pre, sigma²)`, for k = 0..4, p = 1..4.

Decoding the embedded table **[measured here]**: `w(0,1) = σ(φ(α) + αΦ(α))` is `E[ReLU]`; `w(1,1) = Φ(α)` is P(neuron on); `w(2,1) = φ(α)/σ` is the density at the kink; `w(0,2) = σ²((1+α²)Φ + αφ)` is `E[ReLU²]`, etc. — polynomials in α times φ and Φ, evaluated as one batched Horner (line 266).

Then the **raw moments of the post-ReLU output** are assembled from a frozen 25-row term table (`terms`, decoded **[measured here]**) — a truncated Gram–Charlier/Edgeworth expansion, first order in the non-Gaussian cumulants:

- univariate `pK[(p,)] = w(0,p) + (D3/6)·w(3,p) + (D4/24)·w(4,p)` for p = 1..4 — i.e. `E[ReLU^p] ≈ E_Gauss[ReLU^p] + κ3/3!·E[ReLU^p ''' ] + κ4/4!·E[ReLU^p '''']` ("fast_1d" path, line 1872);
- bivariate `pK[(1,1)]_ij = C_ij·w(1,1)_i w(1,1)_j + ½·C_ij²·w(2,1)_i w(2,1)_j + ½·(D12_ij·w(1,1)_i w(2,1)_j + D21_ij·w(2,1)_i w(1,1)_j) + ¼·D22_ij·w(2,1)_i w(2,1)_j` (the table lists one κ3 term and the code symmetrises, line 1912) and analogous `(2,1)`, `(2,2)` outputs. The Gaussian part is the Hermite series in `C` **truncated at second order (C and C²)**; the exact tail was measured worth only −0.23% (RESULTS.md §4e).

Plus the **declared closure extras** (`_closure_apply`, line 536; `_CLOSURE_SCOPE = 'all'`): the *second-order* products of two cumulant blocks — 3 univariate (`D3²/72·w(6,p)`, `D3·D4/144·w(7,p)`, `D4²/1152·w(8,p)`) and **27 bivariate monomials** (`C·D12`, `C·D22`, `D21·D21`, `D22·D3i`, `Q31`, …, each with its symmetry coefficient), decoded **[measured here]**. Every extra is scaled by **θ = `_OWN_THETA_FAM` = 0.625** × a fitted spline `theta` (`_own_extra`, line 531) — a *damped* second-order correction, fitted rather than derived. Switching the extras off costs +18% raw MSE (RESULTS.md elasticity table).

Raw moments → cumulants (`pK_to_K`, line 1951): `K1 = pK1`, `K2 = pK2 − pK1²`, `K3 = pK3 − 3pK1pK2 + 2pK1³`, `K4 = …`, `K21 = pK21 − 2pK1·pK11`, etc.

- **`mu_out = K[(1,)]`** — the layer's answer, and the mean carried forward.
- **`cov_out = diag(K2) + K11`** (line 1996).
- `g_out` = the r = 1 harmonic (traceless) projection of `K4 + rowsum(K22)`: `g = (K4 + K22·1)·6/(n+4) − (Σ …)·3/((n+2)(n+4))` (line 2138-2143; identified against the 504aldo oracle, PROVENANCE row 1).

**(d) The in-loop learned corrector (v57+).** Between computing `mu_out` and returning it, `mu_out += _corr_delta(...)` (line 1986-1992). Features (`_corr_raw`, line 1691), all per neuron and all functions of `W_l` and the incoming state: `mu_pre, sigma, alpha, D3/σ³, D4/σ⁴, ‖W[:,i]‖, ‖D21 row‖²/σ⁴ (zero at the last layer), mu_out (uncorrected), mu_out − mean(mu_prev)` → 9 raw channels, plus their 9 squares, plus the 15 pairwise products of the first 6 → **33 columns**, standardised per layer (`fmu`, `fisd`), then a **33 → 32 → 32 → 1 tanh MLP with a separate weight page per layer** (16 pages; ~36k float32 weights embedded as base64, decoded with stdlib `array`). The target de-standardisation is folded into `W3/b3`. Applied to the *state*, so the correction propagates through every later layer via `mu_pre = Wᵀ mu`; the covariance and κ3/κ4 channels are not corrected (and `K2` uses the *uncorrected* `pK1`). Cost: +0.0326% FLOPs, +541 ops (RESULTS.md §2i, metered).

*How it was fitted* (`tools/fit_corr_seq.py`): **sequential teacher forcing**. For `l = 0..15`: run the chain on every training net with correctors `0..l−1` already active and frozen, hook the *uncorrected* `mu_out` and the 9 raw channels at layer `l` (through the estimator's own `_corr_raw`, so features cannot drift), set target `t_l = label_l − mu_out_l` (the organizers' all-layer means), fit page `l` with Adam (2,500 steps, lr 3e-3 halving every ~833 steps, wd 1e-5, 20% of nets held out for early stopping by R²; a page with val R² ≤ 0 is zeroed, which is exactly inert since `tanh(0)=0` and `b3=0`). The chain is aborted after layer `l` via a sentinel exception to halve the cost (136 layer-passes instead of 256). A single-pass fit on the *uncorrected* chain gave **+0.58% (worse)**; the sequential fit gave **−5.60%** on DEV — "exposure bias, measured" (RESULTS.md §2j). The 400-net refit (FIT 100 + pinned HOLDOUT-300) gave −7.17% DEV.

**(e) Birth of the new lineage.** The ReLU also *creates* new third-order structure. The closure gives `K3` (diagonal) and `K21` (n×n). Subtracting the part the incoming tower already carries (`p111_3 = w1³·D3`, `p111_21`, line 1947-1949) leaves the *newly born* content `S3`, `S21`. It is appended as a new lineage in factored form (line 2015-2091):
- `S21` is split into a **separable** part `S_sep = e_b ⊗ … ∘ C11 ∘ w1` (a column scale that rides for free on the correlation leg, `eb`) and a **residual** `R = S21 − S_sep`, which is compressed to **rank `_P2_RANK` = 4** by a deterministic range finder (`_p2_compress`, line 1162: sketch = R's 4 largest-norm columns, one power iteration, two QRs) → `Z` (n, 4), `L` (n, 4). "P2" = this separable (2,1) leg;
- `S3` rides on `M` as a column scale `sb`;
- the **two-leg block** `A_L = w1 ⊙ C11` (`fac1`) with `C_L = A_L·diag(w2)` (never stored) encodes the all-distinct κ3 born from pairwise correlations: `Σ_j w1_i C_ij · w2_j · w1_k C_kj`;
- the **hubs** (`_HUB_ROUTES = ('i','ii')`, line 2037-2068): the κ4 → κ3 feed. Input κ4 of the model form produces output κ3 of the form `Sym3(y ⊗ diag(c)·C)`; carried as `Y` (n, 2 per lineage) and column scales `CA`, `CB`, folded into the existing contractions with no new dense product. `_HUB_COMPENSATE` removes what the table already counts; `feed` = `_OWN_S_FEED`·spline scales it.

**(f) Aging and compression ("the old tier").** Before appending, the incoming tower is scaled by `w(1,1)` on every mode (`_wick_tower`, line 1138 — a ReLU passes κ3 through with factor `Φ(α)` per index) and, if it now holds **5 exact lineages** (`_SHARED_AGE = 5`), the oldest is folded into the **shared rank-256 basis** (`_shared_join_lean`, line 742). See 1.3 for what that means.

**(g) Output.** Row 0 is replaced by the exact closed form `‖W_1[:,j]‖/√(2π)` in float64 (line 2255). Non-finite entries → 0, negatives → 0.

### 1.3 Glossary of project terms

- **Lineage.** The κ3 content born at one layer, kept separate forever. Mathematically one lineage is `Sym(Σ_j A_D[:,j] ⊗ M[:,j] ⊗ M[:,j]) + Sym(Σ_j A_L[:,j] ⊗ 3M[:,j] ⊗ C_L[:,j])` — a sum over the birth layer's `n` neurons `j` of rank-1 three-way products (`_Tower` docstring, line 891). Born with `M = I`; every later layer applies the same `Wᵀ` to every factor ("mode"), so the equality of modes survives and only the factor matrices need transporting. The tower is `k` lineages, oldest first, stored as wide `(n, k·n)` blocks so one matmul serves all of them (the layout is ours; the 3-D `(k,n,n)` layout is 504aldo's).
- **Leg.** One factor of a lineage: `A_L` and `M` are the two dense legs (n×n each); `C_L = A_L·diag(w2b)` is regenerated, never stored. `A_D` is **never materialised** either: it is the identity `A_D = M·d(sb) + 3·A_L·d(eb) + Z·Lᵀ` (the "P2 separable leg" — `sb`, `eb` are column scales, `Z Lᵀ` the rank-4 residual). This 7→4 leg identity is 504aldo's regrouping plus our reconstruction identity (PROVENANCE, "P2 timeline").
- **Age / tier.** At layer `l` the lineage born at `l−1` is the youngest (`M = I`, free). A lineage is **exact** ("young tier") while fewer than 5 exact lineages are older than it; on the 5th layer of its life it is folded into the shared basis and becomes **old tier**. So each layer's `D21` contraction reads 5 exact lineages (ages 0–4 at that layer's consumption) plus everything older through the basis. Every net gets the same schedule (measured FLOP spread 0.00%; RESULTS.md §3e).
- **Old tier / shared basis / `Q`, `FA`, `FM`, `Sg`, `ko`.** Old lineages' legs are stored as `A_L ≈ Q·FA_j`, `M ≈ Q·FM_j` with one shared orthonormal `Q` (n×256) and per-lineage coordinates `FA_j`, `FM_j` (256×n). `ko` = number of old lineages. `Sg` (256×256) is the carried Gram (energy) of the old content, used to remember old lineages when the basis is rebuilt. The basis is found by a **range finder** (one weighted power iteration + QR; sketch = the previous `Q`, or on first use the 256 strongest columns by norm — deterministic and permutation-equivariant, which is *our* correction to 504aldo's `W[:, :r]` sketch). Columns are weighted by **consumption energy** (`_SHARED_WEIGHTS = "cons"`): how strongly this layer's contraction actually uses each column (the `_wts` stash at line 721). A **join** happens at every layer 5..14 and *re-truncates* the whole old tier to 256 dimensions each time (10 joins).
- **`T0`.** The unshared exact chain (`_SHARED_ON = False`), the accuracy reference: raw 2.3642e-8 vs v53's 2.5451e-8 (+7.7%) — i.e. the whole old-tier compression costs 7.7% raw MSE (RESULTS.md, single-leg T0 table).
- **D3, D21, D4, D22, D12, Q31, C11 / `C_off` / `off_mask`.** The slices of the pre-activation cumulants the closure consumes: `D3_i = κ3_iii`, `D21_ij = κ3_iij` (`D12 = D21ᵀ`), `D4_i = κ4_iiii`, `D22_ij = κ4_iijj`, `Q31_ij = κ4_iiij`. `C11 = cov_pre ∘ off_mask` is the off-diagonal covariance ("`C_off`"); `off_mask = 1 − I`.
- **Hub.** A rank-2-per-lineage extra piece of κ3 (`Y`, `CA`, `CB`) that carries the distinct-index κ4 → κ3 feed (routes "i" and "ii").
- **λ (lambda).** The per-layer κ4 regeneration coefficient (`_K4_LAM`). **θ (closure theta).** The multiplier on the second-order closure extras (0.625 × spline). **feed.** The multiplier on the hub column scales. These three 4-knot PCHIP spline profiles (`_COEFFS`, line 458) are the "fitted coefficient layer" (D2; CMA-ES on FIT; −2.7% raw). `d21`/`p2`/`old` gains exist as profiles but are `None` (identity).
- **`metric` = 2.0**, the κ4 model normalisation `METRIC_C` (matched to the oracle).
- **P1 / P2 / P3 / P4 / P5.** Mechanism labels from `docs/PROVENANCE_504aldo.md`: P1 = κ4 regeneration with per-neuron core (in), P2 = separable (2,1) leg (in), P3 = rank-r D21→birth feedback (**off**: `_P3_ON = False`; measured score-negative), P4 = Strassen/block-symmetric/final-layer trims (in), P5 = age-gated shared basis (in, our own spelling at age 5 / rank 256 instead of their 384/224 two-tier).
- **`pK` vs `K`.** Raw moments of the post-ReLU output vs. its cumulants.
- **Strassen (`_mm`, `_sb`, line 1023-1111).** Every product with all dimensions even and min-dimension ≥ 2·`_STRASSEN_MIN_BLOCK` is done by batched Strassen–Winograd: each level replaces 8 half-size products by 7 and issues them as ONE batched 3-D matmul (flopscope bills `b·m·n·(2k−1)` with no batching surcharge), so a level costs a flat 23 op calls. `_STRASSEN_DEPTH = 5`; `_STRASSEN_MIN_BLOCK = 32` → 1024-wide products go to depth 5 (32×32 leaves), 512-wide to depth 4, 256-wide (the rank-256 old-tier products) to depth 3, the rank-4 thin products to depth 0. v60/v56b set the min block to 64 (depth 4 at 1024, "leaf 64").
- **Budget guard.** `_projected_flops` (line 1510) prices the tower in pure Python before anything runs and would drop old generations if over 0.95·B; at 1024×16 it says 1.037 T ≤ 2.09 T, so the exact tower always runs **[measured here]** (the model ignores Strassen and the shared basis, so it over-predicts the real 475.55 G by 2.2×).
- **DEV / FIT / HOLDOUT / B / F/B / ops / residual / twin.** Panels of the organizer public split (100 mini / 100 fit / 900 holdout). `B = 2^41 ≈ 2.2e12` FLOPs. `F/B` = the score multiplier. `ops` = flopscope operation calls per net (drives residual and wall overhead). `twin` = the local f64 read-only flopscope 0.12.1 run that has matched the server's FLOP count exactly on every graded row.

---

## 2. Where the FLOPs go

Two totals are on record for the graded architecture: **475,550,316,409 FLOPs = 0.21626 B** (v57/v59, leaf 32; RESULTS.md §2i, f64 twin) and **510,179,282,809 = 0.23200 B** (v60, leaf 64). I reproduced the per-site split by running the v59 code on one He-init net at 1024×16 in numpy with every `@` rewritten to a billing hook at flopscope's `m·n·(2k−1)` convention **[measured here; scratch script, not in the repo]**. Matmuls total **411.3 G (86.5%)**; Strassen glue additions **21.8 G (4.6%)** (1 FLOP/element, as the meter bills f32 adds); the remaining **≈42 G (8.9%)** is everything elementwise (closure monomials and einsums — the cost model prices those at ≈1,280 n² per layer ≈ 20 G — hub/wick/D3 slices, wide-array Hadamard products, reductions, the 2²⁵ opening cast).

| component | what | GFLOP | % of 475.5 |
|---|---|---:|---:|
| κ3 exact-tier **(2,1) contraction** `[LA|LP]@[A_L|M]ᵀ` (`_shared_contract_lean:727`) | 14 layers × one (n, 2·k_exact·n)@(…, n) product at depth 5 | 131.9 | 27.7 |
| κ3 exact-tier **transport** `Wᵀ A_L`, `Wᵀ M` (+ thin `Wᵀ Z`, `Wᵀ Y`) | 15+14 wide products at depth 5 | 126.2 | 26.5 |
| **old tier**: dense readout of old legs `Q@[FA|FM]` (38.9) + old half of the contraction `X_old = [LA|LP]@FFᵀ`, `X_old Qᵀ` (35.5) + `Wᵀ Q` (3.6) + the **join / range finder** (39.9) | rank-256 products at depth 3 | 118.0 | 24.8 |
| covariance transport `Wᵀ C W` (block-symmetric) + last-layer diagonal | 15 layers | 31.4 | 6.6 |
| Strassen–Winograd glue additions | ∝ (7/4)^L per product | 21.8 | 4.6 |
| P2 residual leg (compress, reconstruct, thin contraction) | rank 4 | 3.7 | 0.8 |
| closure extras, hubs, wick, elementwise, casts (remainder) | | ≈42 | ≈8.9 |
| corrector (16 tiny matmuls + tanh) | metered +0.0326% | 0.16 | 0.03 |

Reading: **the κ3 tower is ~80% of the bill** (exact tier 54%, old tier 25%); the covariance is 7%; the mean and the corrector are free. Without Strassen the matmul part would be ≈800 G (depth-5 products are billed at (7/8)⁵ = 0.513 of textbook); RESULTS.md D3 measured C10 at 723.7 G / 8,724 ops with Strassen off vs 480.9 G / 23,421 ops at depth 5.

**Op calls (24,186 for v57).** Strassen glue = 23 ops per level per product. From my call census (458 leaf products: 59 at depth 5, 30 at depth 4, 78 at depth 3, the rest thin) the glue is **≈14.9k ops (62%)**, leaf matmuls 458, everything else ≈8.8k **[measured here]**. Check: moving to leaf 64 lowers every depth by one, i.e. −23 × (59+30+78) = **−3,841 ops**, which is exactly the measured 24,186 → 20,345 (RESULTS.md v58_ws/v60 rows).

**Knobs that trade FLOPs for accuracy (measured on this chain, RESULTS.md):**
- **age** (`_SHARED_AGE`): 4→5: +6.1% FLOPs, −15.5% raw (−10.3% score); 5→6: +5.2% FLOPs, −5.5% raw (−0.66% score, ~+700 MiB RSS). The single biggest lever found.
- **old-tier rank** (`_SHARED_RANK`): 256→224: −1.7% FLOPs, +6.1% raw (score-negative); →192: −6.8% FLOPs, +44% raw; →384: +19% FLOPs, −6.6% raw (score-negative). Rank is a worse axis than age.
- **P2 rank** (`_P2_RANK`): 32→4: −5.4% FLOPs, +0.94% raw (the only exact-tier component with score elasticity < 1); rank 2 is a cliff (+11%), rank 1 +1,175%.
- **Strassen depth / leaf**: depth 0→5 saves 33.6% FLOPs for +14.7k ops; leaf 32→64 (depth 4): +7.28% FLOPs, −3,841 ops, and **halves the wall** (see §4); depth 6 / leaf 16 is *dearer* (0.2201 B: glue outruns the saving).
- **closure scope** (`_CLOSURE_SCOPE`): off saves 1% FLOPs, costs +18% raw. **hubs** off: −0.6% FLOPs, +5.4% raw. **P3** on (rank 16): +6.6% FLOPs, −4.6% raw → +1.8% score (off).
- **float32 vs float64**: f64 is billed at 2× per FLOP; closed.

---

## 3. Where the accuracy comes from

Graded ladder (RESULTS.md rows; raw MSE at 1024×16, score = raw × F/B):

| step | file / mechanism | raw MSE | F/B | graded score |
|---|---|---:|---:|---:|
| exact factored κ3 tower + isotropic κ4, Strassen d4 | F21 #330950 | 3.78e-8 | 0.422 | 1.596e-8 |
| + per-neuron κ4 core with λ (P1) + separable (2,1) leg (P2) | v31 #331317 | 2.91e-8 | 0.301 | 8.752e-9 |
| + hubs, closure extras (all layers), shared basis age 4 / rank 256, P4 trims, Strassen d5 | C10 #331816 | 2.97e-8 | 0.2187 | 6.491e-9 |
| + fitted λ/θ/feed splines (D2, −2.7% raw) + Strassen routing (D3) | v50 #331863 | 2.88e-8 | 0.2153 | 6.192e-9 |
| + age 4→5 (D5) | v53 #331880 | 2.43e-8 | 0.2285 | 5.560e-9 |
| + P2 rank 32→4 | v55 #331967 | 2.461e-8 | 0.2162 | 5.320e-9 |
| + in-loop corrector, 100-net sequential fit | **v57 #332129** | 2.333e-8 (−5.22%) | 0.21626 | **5.0443e-9** |
| + 400-net fit, at leaf 64 (wall-safe) | v60 #332223 | 2.301e-8 (−1.35% vs v57) | 0.23200 | 5.3387e-9 |
| (same corrector at leaf 32, derived) | v59 | 2.301e-8 | 0.21626 | 4.9763e-9 (not graded) |

What each piece buys, from the ablations in RESULTS.md (all on DEV 100, paired):
- **Old κ3 content is the accuracy.** Dropping old lineages' (2,1) slice at age > 3: **15.5× worse**; replacing old lineages by their diagonal-only D-block at age > 1: 48×; dropping them outright: 69×; the fully memoryless (regenerating) κ3 flow graded raw ≈ 8.5e-7 = 36× the exact tower. The rank-256 shared basis costs only +7.7% raw against the unshared exact chain (T0) — "old content can be compressed cheaply but cannot be dropped".
- **Closure extras** (second-order terms, θ = 0.625): +18.1% raw if off. **Hubs** (κ4→κ3 feed): +5.4% raw if off. **λ/κ4 regeneration** (P1): 1.24× on design nets at unchanged cost (PROVENANCE).
- **Fitted coefficients**: D2's 12 spline knots −2.7% raw (v50 vs C10); D6's further 302 CMA-ES evaluations with 16 parameters: 0.00%. Per-net adaptive λ: −0.4%. Exact Gaussian bivariate tail: −0.23%. Per-lineage adaptive age: −0.72% score (fixed age in disguise). Sensitivity-weighted basis (N1): the energy basis already captures 98.8% of downstream sensitivity (Spearman 0.994). Tiered/tapered old-tier rank: +2.5% to +18% raw for −1.4% to −4.9% FLOPs (all score-negative).
- **The corrector**: −5.60% raw DEV / −4.76% HOLDOUT-600 / −5.22% graded (100-net fit); −7.17% DEV with 400 nets; per-layer held-out R² 0.166 (layer 2) → 0.0126 (layer 16). Ridge on the same 33 features: 2.13% (linear signal is < 0.10 correlation on every single feature; the signal is in interactions). Hidden 64 uniformly worse than 32; the recursive "previous correction" feature worth nothing (v61).
- **Error anatomy** (RESULTS.md §1b–§1e, on v55): layers 1–2 sit at the label noise floor (6.9e-10); MSE grows 1.13–1.23× per layer late; one-step (local) error is only 20.6% of the accumulated error at layer 16 → **79% of the final error is drift propagated from earlier layers**; a ReLU layer *contracts* incoming mean error (A_l = 0.60–0.88), so all growth is newly created error; the residual concentrates at α ≈ 0 (the kink) but with no consistent sign (no formula fix implicated).

---

## 4. Engineering constraints encountered

- **The wall law (RESULTS.md, Calibration, refit 2026-09-23):** `median wall ≈ 0.371 ms × ops + FLOPs × kernel rate`, worst-of-100 = median × 1.055 (quiet server) or × 1.24 (noisy). The kernel rate **depends on the Strassen leaf size**: ~92–95 ps/FLOP with 64-wide leaves, 164–177 ps/FLOP with 32-wide leaves (the server BLAS runs 32×32 leaves at ~55% efficiency; a mixed candidate read 141). So depth 5's 12% FLOP saving *costs* wall: v57 (leaf 32) graded median 90.4 s / worst 92.8 s on a quiet day with an implied noisy worst of 112 s against the 120 s cliff; v60 (leaf 64) 55.7 / 57.6 s, implied noisy 69 s. Age-5 chains were also ~5–8 s slower than the FLOP+ops model predicted (memory traffic). A net over 120 s is FAILED and scores ~0.9 — one failure in 100 is 1.8 million× worse than the score. This is why a "wall-safe twin" (same code, leaf 64) exists as the designation candidate and why v59 (best score) is not designated.
- **The residual cliff (0.4 s/net) is made of op calls, not FLOPs**: 10.1–11.1 µs per flopscope call on the server (local readings are an artefact and are never quoted). Gates: ops ≤ 26,000, projected residual ≤ 0.33 s. v57: 24,186 ops, 0.2507 s worst (1.6× headroom). ~62% of ops are Strassen glue, so routing more products through Strassen or deeper recursion is bounded by the residual, not by FLOP efficiency (RESULTS.md D3).
- **Memory (8 GiB, only enforceable under WSL `RLIMIT_AS`)**: VmPeak 6,434 MB for v57 (leaf 32) vs 4,941 MB for v60 (leaf 64). **[inference]** The batched Strassen spelling stacks the 7 sub-products of every level into one operand, so a `(n, k·n)` transport at depth L materialises `(7/4)^L` × its size — 16.4× at L = 5 ≈ 1 GB per operand for a 15-lineage-wide product — which is consistent with the measured peaks. the 15-addition Winograd form (the shipped `_sb` is the 18-addition classical form) was measured at −0.69% FLOPs / −8.2% ops but +29% RSS and was not shipped.
- **flopscope quirks that shaped the code**: billing is `m·n·(2k−1)` per product, f32 rate 1.0 and f64 rate 2.0 (so `flops.stats.norm.cdf/pdf`, which promote to f64, are cast back immediately; a stray numpy scalar would double the downstream bill); arrays are immutable (no `out=`, no `+=`, no slice assignment — pooled buffers are impossible and `fnp.empty` itself costs an op); slicing is free, `concatenate` is an op; `as_symmetric` discounts elementwise ops only, never matmuls; batched 3-D matmul is one op with no surcharge (the whole Strassen trick); a scalar materialisation (`__bool__`, `float()`) inside `predict` forces a client-server round trip — every graded row reports 0; `budget_summary()` inside `predict` costs a 157× residual blow-up; a scalar on the *left* of `*` or `/` makes numpy win the dispatch and the op is auto-routed with a warning (nearly billed by the wrong meter — the array goes on the left, `fnp.divide` explicitly); the opening f64→f32 weight cast is billed 2²⁵ FLOPs, and local `whest run` reads exactly that much low because its dataset hands out f32.
- **Determinism and shape-genericity**: no randomness anywhere (sketches select columns by norm — deterministic *and* permutation-equivariant, which 504aldo's `W[:, :r]` sketch is not); `_K4_LAM` and the corrector pages are indexed by layer and **switch off past the fitted depth** rather than extrapolate; `_proj_coef_d4_r2` returns 0 at width 2 (a genuine upstream singularity that killed two submissions in the smoke test); `setup` is best-effort (a raising `setup` kills the whole submission); no bare `except` returning a constant.
- **Numerics**: Strassen depth 5 amplifies per-matmul float32 error 5–30× yet DEV raw MSE is unchanged to ±0.01% between leaf 32 and 64 (rounding-order noise only); `ALPHA_CAP = 40` is a saturation, bit-inert at the live shape.
- **Cost is order-independent by design**: all constant arrays (Wick program, closure tables, corrector weights) are rebuilt inside every `predict` so no net is priced differently from the rest.

---

## 5. Limitations and unexplored degrees of freedom — a candid list

1. **The chain is not final-layer-aware in its objective.** Everything (state, closure, tower) is propagated to be accurate at every layer; only row 16 is scored. The intermediate rows are free by-products, but the *allocation* of effort is uniform. The anatomy says late-layer increments dominate the final error (layer 14–16 each contribute ~1.1–1.2 pp of the corrector's gain; layers 2–5 together 0.26 pp), and A_l < 1 damps early mean error — yet the age/rank schedule is identical at every layer. The one final-layer-aware trim (P4: skip cov/D21/births at the last layer) is in; the D5-H3 "taper the old tier in the tail" was measured dead (+2.5–18% raw). Untested: spending *more* late (e.g. age 6 only for the last few layers' consumption, or rank > 256 only in the tail — the opposite direction from H3), and cheapening the *early* layers' κ3 precision instead. **[inference]** The T1/T2 kill tests only dropped content globally by age, never by target layer.
2. **The corrector corrects the mean only.** `cov_out`, the born lineage (`e_b`, `S3`, `S21`) and `g_out` are all built from the *uncorrected* `pK1` (line 1996, 2002-2003, 2035, 2112) — the covariance is not made consistent with the corrected mean. There are no covariance labels, but a covariance (or σ) correction could be fitted end-to-end on the final-layer MSE; nobody has tried. The corrector also sees each neuron in isolation (9 local channels + one layer-mean scalar) — no cross-neuron or global-layer context, no knowledge of the old-tier truncation error — and it is trained greedily per layer on `label − chain` rather than by differentiating the final-layer MSE through the chain (the only end-to-end fit ever run is D2's 12-knot CMA-ES).
3. **Covariance is exact in transport, truncated in closure.** `Wᵀ C W` is exact; the ReLU covariance closure keeps the Hermite series in C to second order (tail measured worth 0.23%), κ3/κ4 cross terms to first order, and second-order products only through the θ-damped extras (θ = 0.625, fitted — the extras are *not* the exact next-order expansion, they are a scaled proxy). No κ5+.
4. **κ4 has no memory.** It is regenerated each layer as `diag(dG) + λ·C_off` with a fitted λ table; κ4 all-distinct content and the κ4→κ3 feed beyond the rank-2 hubs are absent. Exact/augmented κ4 was priced at 6× worse score (K4 closed on cost, RESULTS.md D4b); "κ4-on-M" (carrying κ4 on the tower's modes) was never run (recorded as NOT RUN).
5. **The old tier is re-truncated at every one of 10 joins** to a rolling 256-dim subspace chosen by *this layer's* consumption energy. Compounding truncation costs 7.7% raw vs. T0. N1 showed energy ≈ downstream sensitivity (Spearman 0.994), so a smarter *selection* buys nothing; but a cheaper *representation* of the same content is where 25% of the FLOPs sit (118 G), and 38.9 G of that is the dense readout `Q@[FA|FM]` done every layer purely to feed elementwise consumers (D3, LA/LP, hubs). **[inference]** Evaluating those consumers in the compressed coordinates costs r³n ≈ 17 G per lineage-pair, i.e. it is *not* cheaper, so this is a real floor of the current representation, not an oversight.
6. **The exact tier's contraction and transport are 54% of FLOPs** and are already at Strassen depth 5. The only remaining FLOP levers there are structural (fewer exact lineages: age 4 costs +18% raw) or a different representation of `D21` (e.g. the leaders' unexplained 0.13 B). P2 rank 4 is already at its cliff (rank 2: +11%).
7. **Fitting is saturated at 12–16 parameters** (D6: 0.00% after 302 evaluations). The λ table is an initialiser from 24 design nets scaled by a spline; only 3 of the 6 profiles carry non-identity knots (`d21`, `p2`, `old` are `None`). A *joint* refit of λ/θ/feed together with the corrector is listed as closed (CURRENT_STATE "joint refit"), on the D6 evidence.
8. **Per-net adaptivity is closed twice** (λ: driving ratio CV 1.8–9%; age: lineage energies too uniform). The leaders' ±6% per-net FLOP spread remains unexplained; ours is exactly 0.00%.
9. **Dead or legacy state**: `c4` (isotropic κ4 scalar) is computed and carried but never consumed; the P3 machinery (`F`, `R3`, `R4`, `w3b`, `_p3_parts`), the query-basis path (`_QUERY_BASIS`), `_PROTECT_FEED`, the non-lean shared functions and `_regen`/`_truncate` fallbacks are all inert at the live shape — a large fraction of the 2,300-line file is switched-off branches kept for parity with older controls.
10. **The budget guard over-prices by 2.2×** (`_projected_flops` = 1.037 T vs 475.5 G real) because it ignores Strassen and the shared basis; harmless at 1024×16 (bar 2.09 T) but **[uncertain]** it would needlessly drop old generations at a modestly larger shape (e.g. width ≈ 1400 or depth ≈ 30) — only relevant if the private rerun changed shape, which the rules do not suggest.
11. **Wall, not FLOPs, is the binding cost**: the leaf-32 chain (best score) sits at an implied noisy worst of 112 s against a 120 s cliff; the leaf-64 twin gives up 7.3% of score for 51 s of margin. A mixed-leaf schedule per site (only two variants — transports at 64, or the two wide contractions at 64 — were tried) and Winograd's 15-add form (rejected for +29% RSS) are partially explored middle grounds.
12. **Training data for the corrector**: 400 of the 1000 organizer nets used; HOLDOUT-600 untouched. The 100→400 step was worth 1.57 pp and closed the train/val gap to ~0.0015, so the remaining 600 are probably worth little **[inference]**; the shared-across-layers model with a layer embedding (`--shared`) was never run.
13. **Label noise floor** 6.8e-10 = 2.9% of our raw MSE; nothing below it is measurable and nothing on the board is near it.
14. **Provenance**: the κ4 regeneration law, the 7→4 leg regrouping, the range finder and the age-gated basis are 504aldo's published (MIT) mechanisms re-implemented in our code; ours are the wide 2-D Strassen layout, the reconstruction identity, the equivariant sketch, the closure-extras table, the fitted profiles, the per-neuron κ4 birth projection fix, and the in-loop corrector. Any write-up must say so (`docs/PROVENANCE_504aldo.md`, "Attribution").
