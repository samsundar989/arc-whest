# R3: ARC cumulant-propagation code, K=3 / K=4 technical inventory

*Code read: `arcrepo/mlp_cumulant_propagation` @ `93d091a` ("Skip kprop metric bookkeeping when no radial pieces are tracked"). Everything here comes from reading the code; torch is not available, so nothing in ARC's package was run. The tensor-algebra claims in §3 were checked with numpy re-implementations of ARC's normalisations, and the term sets with ARC's own `partitions.py` (scripts in `scratchpad/r3check/`, Appendix A). Line references are `src/mlp_kprop/<file>:<line>` unless a path is given.*

**Units.**

- 1 **u** = one n×n·n×n GEMM = n³ MAC = 2n³ FLOP. At n = 1024 that is 2.15e9 FLOP.
- **B** = 2⁴¹ FLOP ≈ 1024 u, the whole budget.
- A **block** is the n columns that one `add_factors_` / `from_dstensor` call adds to a factored tensor. It is ARC's analogue of a "lineage".
- C̃ is the covariance with its diagonal zeroed.
- w_k = E[∂ᵏφ(Z)], and w(k,p) = E[∂ᵏ(φ(Z)ᵖ)] (`wick.py:90-124`). So w0 = σφ(α)+μΦ(α), w1 = Φ(α), w2 = φ/σ, w3 = −αφ/σ², w4 = (α²−1)φ/σ³.
- For ReLU, w(1,2) = 2·w0 and w(2,2) = 2·w1 (`wick.py:111-117`).

---

## 0. Bottom line (most decision-relevant first)

**(a) Exact factored K=4 costs about 2×10³ budgets as ARC wrote it (SIMPLE), or about 2×10⁴ (AUGMENT), at n = 1024, L = 16.**

- **Why.** `FactoredTensor4` stores κ4 = Sym(Σ_r A_r ⊗ B_r) with an n×n matrix per column for both A and B (`factor_k4.py:73-123`).
  - Transporting one n-column block costs 8n⁴ FLOP (`factor_k4.py:215-227`), about 4 B per block per layer.
  - Re-forming the (2,1,1) slice from the transported blocks costs another 8n³R (`factor_k4.py:53-63`).
  - Blocks are never merged or compressed, so R grows by 2n per layer in SIMPLE and 7n per layer in AUGMENT. ARC's own FLOP fits are degree 2 in depth (`flop_utils.py:621-622`).
- **SIMPLE totals.** About (32ℓ+14)·n⁴ FLOP at layer ℓ, so 1.5e13 → 5.4e14 FLOP per layer and Σ ≈ 4.5e15 FLOP ≈ **2.0×10³ B**.
- **AUGMENT totals.** About **1.7×10⁴ B**, and it also needs dense n⁴ (2,1,1,1) κ5 slices (4.4 TB in fp32 at n = 1024).
- **Memory.** The κ4 factors alone are 0.27 TB for SIMPLE and ~0.9 TB for AUGMENT at n = 1024.
- **Verdict.** "~100–1000× over budget" is too optimistic for ARC's implementation. It is about right for an idealised exact K=4 that uses vector-leg lineages, because some n⁴-class pieces remain even then (§2.6, §4).
- **Reference point.** ARC's factored K=3 is also over budget at this size: about 1.4 B (SIMPLE) and about 11 B (AUGMENT).

**(b) The 3-leaf "star" births share the K=3 lineage's hub and legs exactly. Their slices can be read out with no extra transport.**

- **What a star is.** At K=4 SIMPLE, each layer's top-slice birth is one block whose B tensor is 12·path + 4·star + 12·(κ3×edge) (`factor_k4.py:470-507`).
- **Readouts.** In ARC's normalisation, with A = M·X, B = M, J = B·diag(w2) the K=3 lineage legs, and w3 at the birth layer:
  - D4[a] = 4·Σ_j w3_j A_aj³ B_aj. Cost is O(n²) elementwise. This is the term that feeds the mean (w4·D4/24).
  - D31 = 3·(A∘A∘B)·W3·Aᵀ + (A∘A∘A)·W3·Bᵀ. Cost 2 u.
  - D22 = 2·[(A∘B)·W3·(A∘A)ᵀ + transpose]. Cost 1 u.
  - The star's (2,1,1) slice is itself a rank-n factored 3-tensor.
- **The path also rides on the same legs.** It couples two hubs j, k of the same lineage through the birth-layer covariance: F = Σ_jk C̃_jk·A·J·J·A.
  - D4 costs 1 u: V = (A∘J)C̃.
  - D31 costs 2 more u.
  - D22 is only partly O(n³). Three of its four pieces are n⁴.
- **κ3×edge (3+2)** couples the newborn lineage with every older κ3 lineage. It costs O(n³) per (newborn, older) pair per layer.
- **Relation to the K=3-AUGMENT r=1 core.** The core is P_{≥1}κ4 (the trace part). The top-slice births are exactly traceless at birth, so K=3 AUGMENT discards them entirely.
  - K=3 AUGMENT and K=4 SIMPLE use identical Wick terms for the mean and covariance (verified by enumeration).
  - So the whole K=4 increment in mean and covariance is (exact D4/D31/D22) − (D4/D31/D22 of Sym(core⊗g)), i.e. the traceless part, plus κ3 differences.

**(c) Sweep.** Use `scripts/kprop_by_width.py` from inside `scripts/`, with `KPropByWidthCfg` and:

- `mlp_kwargs = dict(num_layers=17, nonlin='relu', init_kind='manual', w_var=2.0)`. That gives 16 ReLU layers act0..act15, all with 2/n weight variance.
- `sample_k_max = 1`, `output_d_max = 1`, and a new `sampK_cache_name`.

Run four configurations (snippet in §5.2):

- K=3 AUGMENT with `factor=False`. This is the clean baseline; factored AUGMENT drops κ3×edge.
- K=3 AUGMENT with `factor=True`.
- K=4 SIMPLE with `factor=True`, at n ≤ 256. Its peak is ~13–17 GB at n = 256.
- K=4 AUGMENT with `factor=True`, at n ≤ 128–160. Its dense n⁴ κ5 slices make n ≥ 192 exceed 32 GB.

Approximate n_max on 32 GB in fp32: factor_k4 SIMPLE ~300, factor_k4 AUGMENT ~160, dense harmonic K=4 SIMPLE ~150 (ARC's own guard treats n > 128 as OOM, `time_utils.py:208`), dense K=4 AUGMENT ~130.

**Interpretation caution.** K=4 SIMPLE ⊇ unfactored K=3 AUGMENT: same diagram budget 3, but κ3 is dense and κ4 is full. So "K=4 SIMPLE − factored K=3 AUGMENT" mixes the traceless-κ4 effect with the κ3×edge and triangle terms that factored K=3 drops. Compare against K=3 AUGMENT with `factor=False`.

---

## 1. Factored K=3, kind=AUGMENT (`factor_k3.py`, `kprop_harmonic.py`)

### 1.1 Representation and normalisation

`FactoredTensor` (`factor_k3.py:77-122`) stores T = Sym(Σ_r L1_r ⊗ L2_r ⊗ L3_r) with three n×R leg matrices.

- `contract_W` multiplies each leg by W: 3·2n²R FLOP (`:181-192`).
- `contract_wick_` row-scales all three legs by the same w1 (`:194-217`).
- `add_factors_` concatenates columns and incrementally updates a cache of the repeated-index slices (`:136-146`).
- `__add__` concatenates and drops the cache (`:153-168`).
- `get_dslice((2,1))` costs 3 rank-R GEMMs, 6n²R FLOP. `(3,)` costs O(nR). The all-distinct slice is never formed (`:31-54`).

Births are normalised so that **every labelled diagram has coefficient 1**. For example, the light birth uses fac2 = 3·I and Sym averages over the 3 hub placements (`:400`). Verified numerically. For one lineage with transported legs A (covariance leg), B (hub leg) and J = B·diag(w2_birth):

- κ3_pqr = Σ_j w2_j [B_pj A_qj A_rj + A_pj B_qj A_rj + A_pj A_qj B_rj].
- D3 = 3·Σ_j J_aj A_aj².
- D21 = 2(J∘A)Aᵀ + (A∘A)Jᵀ.

These match R1 §1. ARC transports three legs per block, but the third light leg equals L1·diag(w2) in SIMPLE.

### 1.2 Blocks that create or update κ3 (one row = one or more n-column blocks)

| id | block | lines | reads | legs (L1, L2 = hub, L3) at birth; hub = column r | coefficient | cost per layer |
|---|---|---|---|---|---|---|
| K3.0 | transport of old blocks (linear step) | `factor_k3.py:181-192` via `kprop_harmonic.py:310-311` | W | each leg ← W·leg; slice cache dropped | 1 | **6n²R FLOP = 3R/n u** (R = all accumulated columns) |
| K3.1 | old blocks through the nonlinearity (diagram k3[1,1,1]) | `:379-382` | w1 | each leg ← diag(w1)·leg | 1 | 3nR elementwise, plus cache rescale |
| K3.2 | **light–light birth** (SIMPLE and AUGMENT) | `:371-376, 389, 400-407` | C̃ = zero_repeated(C), w1, w2 | (w1∘C̃ = X, 3·I, (w2_h w1_k C̃_hk)ᵀ = X·diag(w2)) | 3 = number of 3-vertex 2-edge graphs; budget `2+augment` (`:373`) | build O(n²); `add_factors_` cache update 6n³ (new (2,1) slice) |
| K3.3 | AUGMENT only: heavy right legs folded into the light-left block's L3 | `:390-399, 405-406` | κ21[a,c] = κ3_aac/2 (`:392`, factored (2,1) slice), κ22 = κ4_aacc/4 (`:396`, r=1 `harmonic_diagslice`), C̃²/2 (`:399`) | L3 += w3_h w1_k κ3_hhk/2 + w2_h w2_k κ3_kkh/2 + w3_h w2_k κ4_hhkk/4 + w3_h w2_k C̃_hk²/2 | hub weight w(b_l+a_r): w3 for near-mult 2, w2 otherwise; s_r already includes 1/∏mult! | same block as K3.2 |
| K3.4 | AUGMENT only: 4 heavy-left births | `:391-399, 401-407` | as K3.3 | L1 ∈ {w2∘κ3_iih/2, w1∘κ3_hhi/2, w2∘κ4_iihh/4, w2∘C̃_ih²/2}; L2 = 3I; L3 = X·diag(w2) for the (2,1) leg, X·diag(w3) for the other three | idem (light right only, cost 2+1 ≤ 3) | 4 blocks: O(n²) each + 6n³ cache |
| K3.5 | AUGMENT only: "211 H(d=4,r=1) → 111", the κ4 r=1 (2,1,1)-slice feeding κ3 | `:409-453` | core (n×n), metric g (1-D under avg metric), w1, w2 | **B1** (`:431-436`): ((w2∘diag core)1ᵀ, w1∘g, w1∘I). With diagonal g it lives only on j = k, so it has no distinct-index content. **B2** (`:438-446`, only if g is 2-D): (w1∘core, w2∘I, w1∘g); hub = middle. **B3** (`:448-453`): (w1∘core, w1∘I, (w2∘diag g)1ᵀ) | 1/4, 1, 1/4 (= ½·3·{1/6, 4/6, 1/6}) | 2 blocks (avg metric) or 3; O(n²) + 6n³ cache each |
| K3.6 | **repeated-slice correction** (`from_dstensor`) | `:463-467, 526` (method `:251-275`) | dense pK→K slices (3,), (2,1), minus the factored part's own repeated values (`K_ds[3] -= pK_111.get_repeated()`) | (diag(K_(3))·I + 3·K_(2,1)ᵀ, I, I) | ×3 undoes the `DSTensor.to_tensor` scaling (`diagslice.py:414-423`) | O(n²) to build; then transported like any block |

**Births per layer.**

- SIMPLE: 1 (K3.2) + 1 (K3.6) = **2n columns**.
- AUGMENT: 5 (K3.2 with K3.3, plus 4×K3.4) + 2 or 3 (K3.5) + 1 (K3.6) = **8n columns with avg metric, 9n without**.
- At layer 0 only the light, doubled-light and correction blocks exist: 3n.

**The dense (non-factored) slices** come from the generic loop at `:327-361`. It applies `factored_keeps_term` (`kprop_harmonic.py:86-101`): the non-hypertree κ3×edge and the covariance triangle are dropped from the top slice (TODO.md:28-39). Each term is `eval_part` → `multiply_wicks` → `symmetrize` (`diagslice.py:1038-1059`, `kprop_harmonic.py:161-178`). Iso-class counts (Appendix A):

- SIMPLE: (1,) 3, (2,) 3, (1,1) 4, (2,1) 5, (3,) 3, (2,2) 4, (4,) 3.
- AUGMENT: (1,) 3, (2,) 3, (1,1) 10, (2,1) 15, (3,) 3, (2,1,1) 17 (n³ entries), (2,2) 10, (3,1) 15, (4,) 3.

### 1.3 Blocks that create or update κ4 (the r=1 "core")

At K=3 AUGMENT, κ4 is an `HTensor` with r = 1, i.e. κ4 ≈ Sym(core ⊗ g) with core n×n and g the metric (`get_r_x`, `kprop_harmonic.py:209-218`; `harmonic.py:98-128`). At K=3 SIMPLE it is r = 2: a scalar times Sym(g⊗g) (`factor_k3.py:533`).

| id | block | lines | reads | effect / formula | coefficient | cost per layer |
|---|---|---|---|---|---|---|
| K4.0 | transport | `harmonic.py:444-480` via `kprop_harmonic.py:309` | W | core ← W·core·Wᵀ; g ← init_scale[l]·I (`use_avg_metric`, `kprop_harmonic.py:578-579`) or W·g·Wᵀ | 1 | 2 u (+2 u for g without avg metric). Depth-independent. |
| K4.1 | readouts used by the dense terms | `harmonic.py:666-724` | core, g | D4 = core_aa·g_aa; D31_ac = (core_aa g_ac + core_ac g_aa)/2; D22_ac = (core_aa g_cc + core_cc g_aa + 4 core_ac g_ac)/6 (verified). (2,1,1) is materialised n³. | — | O(n²); (2,1,1) O(n³) |
| K4.2 | births: dense d=4 slices (4,),(3,1),(2,2),(2,1,1) → `DS_pK_to_K` → `DS_harmonic_proj(r_out=1)` | `factor_k3.py:327-361, 464-466, 529-530`; `harmonic.py:592-626`; `cumulants.py:193-202` | every WK slice | core = Σ_{r≥1} proj_coef(n,4,r)·R^{r−1}L^r(K_ds[4]) ≈ (1/(2n+8))·L(K_ds[4]) + O(n⁻²)·R L². These are traces over the doubled index. Example: the "star centred on the doubled index", k2[1,1,0]×k2[1,0,1] in the (2,1,1) slice, has hub weight w(2,2) = 2w1 and contributes c·X·diag(2w1)·Xᵀ. | proj_coef (`harmonic.py:305-336`) | O(n³) elementwise |
| K4.3 | K211_contrib (a): pK_111 × mean, from the [(1,1,1),(1,0,0)] pK→K term | `factor_k3.py:470-497` | all κ3 factors after births, with the repeated part subtracted via `from_dstensor(get_repeated())`; pK1 = E[φ] | core += −(24/(2n+8))·Sym((Σ_i pK1_i L1_ir)·L2_jr·L3_kr + 2 cyclic)/3 | −2·6·2/(2n+8) | 6n²(R+9n) FLOP |
| K4.4 | K211_contrib (b): the (2,1,1) term with vec_part (1,1,1) | `:499-521` | WK[3] factors minus repeated; w(1,2) = E[∂φ²]; w1 | core += (12/(2n+8))·Sym((Σ_i w(1,2)_i L1_ir)·w1_j L2_jr·w1_k L3_kr + cyclic)/3 | 6·2/(2n+8) | 6n²(R+n) FLOP |
| K4.5 | memory | — | — | core n², g n (avg) or n² | — | — |

### 1.4 What feeds the mean and covariance at K=3 AUGMENT

From the enumeration; these terms are identical at K=4 SIMPLE:

- **Mean:** E φ_a = w0 + w3·D3/6 + w4·D4/24.
- **Covariance:** w1w1·C + (w2w1·D21 + sym)/2 + (w3w1·D31 + sym)/6 + w2w2·C²/2 + w2w2·D22/4 + (w1w4·C·D3 + sym)/6 + (w2w3·C·D21 + sym)/2 + (w1w5·C·D4 + sym)/24 + w3w3·C³/6 + w3w3·C·D22/4.
- **Second-moment diagonal** (pK (2,) slice, turned into the variance by pK→K): w(0,2) + w(3,2)·D3/6 + w(4,2)·D4/24.

### 1.5 Cost of ARC's factored K=3 at n = 1024, L = 16

R accumulates without bound. Per layer:

- **SIMPLE:** transport plus the D21 slice = 12n²R_in, with R_in(ℓ) = 2nℓ. Σ ≈ 1.4×10³ u ≈ **1.4 B**.
- **AUGMENT:** about 24n²R_in + 110n³ (transport 6, D21 6, K211 12), with R_in(ℓ) = (8ℓ−5)n. Σ ≈ 1.15×10⁴ u ≈ **11 B**.

Memory is small (the factors are 3nR floats, ≈1.5 GB at n = 1024 for AUGMENT). The (2,1,1) dense n³ slices make AUGMENT peak at roughly 4–6·n³·4 B (17–26 GB at n = 1024).

---

## 2. Factored K=4 (`factor_k4.py`)

### 2.1 Representation

`FactoredTensor4`: T_ijkl = Sym(Σ_r A_{ij r} B_{kl r}), with A and B both n×n×R and symmetrised in their first two indices (`:73-123`, `:160-172`).

- Each birth adds R = n columns, a full n×n matrix per column for both A and B.
- Storage is **2n²R floats**.
- `contract_W` is `einsum('i1 j1 r, i2 i1, j2 j1 -> i2 j2 r')` on both factors: **8n³R FLOP** (`:215-227`).
- `contract_wick_` row- and column-scales by w1 (`:229-252`).
- Slices (`:28-71`):
  - (4,): 2nR.
  - (3,1): 4n²R.
  - (2,2): 6n²R.
  - **(2,1,1): 8n³R**. It is used by the dense terms and by the κ4 → κ3 feedback.
- `from_dstensor` (`:286-318`): A = δ₃ (A_{ijr} = δ_ir δ_jr) and B = δ(K_(4)) + 4K_(3,1) + 3K_(2,2) + 6K_(2,1,1) (full rank per column).
- **κ3 is dense** in `factor_k4`: an `HTensor` with r = 0 and an n³ core (`:411-413`, `:637-639`). Its transport is 6n⁴ FLOP per layer.

### 2.2 Per-layer births (top slice (1,1,1,1))

The four SIMPLE diagram classes (budget 3, all hypertrees; enumeration in Appendix A):

| diagram (iso count) | where | legs at birth (A-factor = leg i plus hub j; B = (k,l) per hub j) | relation to K=3 lineage |
|---|---|---|---|
| **k4[1,1,1,1]** (×1), old κ4 transport | `:431-437` | old blocks × w1 on every index | — |
| **star** k2·k2·k2 sharing one vertex (×4) | `:488-497`, alpha=4 | B_j(k,l) = w3_j X_kj X_lj | **same hub, same legs** as the light K=3 lineage |
| **path** k2·k2·k2 (×12) | `:478-487`, alpha=12 | B_j(k,l) = w2_j C_jk w2_k X_lk | same legs, two hubs (j,k) of one lineage, coupled by C |
| **3+2** k2·k3 (×12) | `:498-506`, alpha=12 | B_j(k,l) = w2_j w1_k w1_l κ3_jkl | newborn lineage × older κ3 |

In ARC all three births share **one A-leg block**, so B = 12·path + 4·star + 12·(3+2) is summed into a single n×n×n tensor:

- A[i,j,r] = sym(δ_jr X_ir) with X = w1∘C (`:475`). Note that `dsWK` does not zero the diagonal for d < 4 (`:411-413`); only repeated-index entries see it.
- Births per layer in SIMPLE: one block plus one `from_dstensor` correction block (`:640`), so **ΔR = 2n**. R after layer ℓ is 2n(ℓ+1), and R = 32n after 16 layers.

**AUGMENT** (budget 4, `:447`):

- `two_legs` holds 5 legs: (1,1) C; (1,2), (2,1) dense κ3 slices; (2,2) the κ4 slice from the factors; the doubled light C²/2 (`:449-460`).
- `three_legs` holds (1,1,1) and the κ4 (2,1,1)-slice permutations (`:462-468`).
- Each of the 5 A-legs gives one block, with all path, star and 3+2 combinations within budget.
- Add 1 block (avg metric) or 2 for **WK2111 → pK1111**: the κ5 r=1 core's (2,1,1,1) slice (`:509-549`). Its A_r = (w1⊗w1)∘core5[:,:,r] is full rank per column, and B_r = sym(w1_r e_r ⊗ (g∘w2))/5 plus a 12/10 term if g is 2-D.
- Add the correction block. So **ΔR = 7n per layer** (avg metric), 3n at layer 0.
- Dropped non-hypertree top-slice diagrams: k2×k4[1,1,1,1] (κ4×edge, ×6), k3×k3 sharing two vertices (×6), covariance 4-cycles and triangle-plus-pendant, and k2·k2·k3 cycles (TODO.md:28-39, `kprop_harmonic.py:86-101`).

### 2.3 Feedback of κ4 into lower orders

At SIMPLE these are 31 dense iso-terms read κ4 slices (Appendix A):

- **Mean:** w4·D4/24.
- **Variance:** w(4,2)·D4/24.
- **Covariance:** (w1w3·D31 + sym)/6, w2w2·D22/4, (w1w5·C·D4 + sym)/24, w3w3·C·D22/4.
- **κ3 top slice (dense):** k4[1,1,2] (×3), i.e. w1w1w2·κ4_abcc/2, the (2,1,1) slice. Also k2·k4[2,0,2] (×6): w2w1w3·C·D22/4.
- **κ3 and κ4 repeated slices:** further terms.

At AUGMENT there are 479 κ4-reading terms, including κ4×κ4 and κ3×κ4 products.

### 2.4 Augmented variant (the "trace of κ5")

Yes. `get_d_max(4, AUGMENT) = 6` (`kprop_harmonic.py:237-240`).

- **κ5 r=1** (Sym(core3 ⊗ g), core n³; `factor_k4.py:578`). It receives `K2111_contrib`:
  - pK1111 → K2111 with coefficient −2·10·2/(2n+12) (`:583-605`).
  - pK2111 → K2111 with coefficient 10·2/(2n+12) (`:607-628`).
  - Both are einsums over the κ4 factors costing about 2n³R each. They are added at `:632-633`.
- **κ6 r=3** (scalar·Sym(g⊗g⊗g); `:580`).
- κ5 feeds κ4 back through the WK2111 block above.
- AUGMENT also evaluates the **dense d=5 (2,1,1,1) slice: n⁴ entries, 80 iso-terms**, with 36 vector partitions in `DS_pK_to_K` (`:367-401, 562-581`). This dominates memory.

### 2.5 FLOPs at n = 1024, L = 16, as implemented

Per layer ℓ (0-based), with R_in = columns entering the layer:

| piece | SIMPLE | AUGMENT |
|---|---|---|
| κ4 factor transport | 8n³R_in | 8n³R_in |
| (2,1,1) slice of transported factors | 8n³R_in | 8n³R_in |
| K2111 einsums | — | ≈8n³R_in + 32n⁴ |
| birth cache updates ((2,1,1) of each new block) | 8n⁴ | 48n⁴ |
| dense κ3 (and κ5 core) transport | 6n⁴ | 12n⁴ |
| dense (2,1,1,1) d=5 slice: eval, symmetrize, pK→K (elementwise; rough) | — | ~9×10² n⁴ |
| R_in(ℓ) | 2nℓ | (7ℓ−4)n for ℓ ≥ 1 |
| **total** | **(32ℓ+14)n⁴** → 1.5e13 (ℓ=0) … 5.4e14 (ℓ=15) FLOP | ≈ 24n³R_in + ~10³n⁴ → 1.2e15 … 3.8e15 FLOP |
| **Σ over 16 layers** | 4064n⁴ = **4.5e15 FLOP ≈ 2.0×10³ B** | ≈ 3.5×10⁴n⁴ ≈ **3.8e16 FLOP ≈ 1.7×10⁴ B** |
| memory (fp32) | factors 2n²R·4 B = 275 GB at R = 32n (≈3× live) | ~0.9 TB factors + 4.4 TB per n⁴ slice |

ARC's FLOP counter discounts symmetric dense operations (`contract_factor` ≈ 0.39 for d=3, `slice_factor` ≈ 1/6 for (2,1,1,1), `symmetrize` counted as 0; `flop_utils.py:411-437`, `tensor_utils.py:16,84`). The factored einsums are not discounted, so the totals above are what a GPU would execute.

### 2.6 Verdict on "~100–1000× over 2⁴¹"

- **ARC's code:** refuted on the low side. SIMPLE is ~2×10³×, AUGMENT ~2×10⁴×. One average SIMPLE layer alone is ~130 B. A single n-block transport is 8n⁴ ≈ 4 B.
- **Idealised exact K=4** (vector legs, §3): star, path and old κ4 ride on K=3 legs for free. These pieces are still n⁴-class:
  - the D22 and (2,1,1) cross pieces of path and 3+2 (n⁴ per lineage-layer);
  - exact transport of the repeated-slice correction blocks (n×n matrix per column: ~2n u ≈ 2 B per block-layer, Σ ≈ 240 B);
  - the 3+2 pair couplings (~L³/6 pair-layers).
- So 10²–10³ B is the right order for an idealised exact K=4. Exact K=4 at n = 1024 is not viable. Only the shared-factor O(n³) subset (§3) is.

---

## 3. KEY: K=4 slices readable from the transported K=3 lineage factors

**Setup.** Lineage born at layer s:

- X = w1∘C̃ at s (factor_k3 convention; factor_k4 keeps the diagonal, which differs only on repeated indices).
- A = M·X, B = M, where M = W_t·diag(w1^{(t−1)})·…·W_{s+1} up to the readout pre-activation.
- J = B·diag(w2^{(s)}); J′ = B·diag(w3^{(s)}) = J·diag(−α/σ)_s.
- All formulas are in ARC's normalisation (coefficient 1 per labelled diagram) and were **verified to ~1e-15 relative** (Appendix A).

### 3.1 Star: same hub j, same legs (exact share, no transport)

T^star_pqrs = Σ_j w3_j Σ_{hub position} (B at the hub, A on the three leaves). This is ARC's `alpha=4` B factor after `Sym`.

| slice | formula | cost per lineage-layer |
|---|---|---|
| D4[a] = κ4_aaaa | **4·Σ_j w3_j A_aj³ B_aj** = 4·Σ_j J′_aj A_aj³ | O(n²) elementwise |
| D31[a,c] = κ4_aaac | **Σ_j w3_j (3A_aj² B_aj A_cj + A_aj³ B_cj)** = 3(A∘A∘J′)Aᵀ + (A∘A∘A)J′ᵀ | 2 u (one n×2n×n GEMM) |
| D22[a,c] = κ4_aacc | **2·Σ_j w3_j (A_aj B_aj A_cj² + A_aj² A_cj B_cj)** = 2[P + Pᵀ], P = (A∘J′)(A∘A)ᵀ | 1 u (reuses A∘A from D21) |
| (2,1,1) T_aabc | Σ_j w3_j [2(A∘B)_aj A_bj A_cj + (A∘A)_aj (B_bj A_cj + A_bj B_cj)] | a rank-n factored 3-tensor (n⁴ only if materialised) |

For comparison, the K=3 readout D21 = [2(J∘A), A∘A]·[A, J]ᵀ costs 2 u. The star adds about **3 u per lineage-layer**, of which D4 is free. That is ~360 u if all 120 lineage-layers of a 16-layer run are kept.

### 3.2 Path: same lineage legs, two hubs coupled by the birth covariance

F(x,y,z,w) = Σ_jk C̃^{(s)}_jk A_xj J_yj J_zk A_wk, and T^path sums F over the 12 labelled paths.

| slice | formula | cost |
|---|---|---|
| D4 | 12·rowsum((A∘J)∘V), V = (A∘J)·C̃^{(s)} | 1 u (V) |
| D31 | 6[(J∘V)Aᵀ + (A∘V)Jᵀ] | +2 u |
| D22 | 4·V(A∘J)ᵀ [cheap] + 4F(c,a,c,a) + 2F(c,a,a,c) + 2F(a,c,c,a) [each Σ_jk with both hubs carrying a and c] | 1 u + **3 n⁴ pieces** |

The C̃^{(s)} coupling costs n² of storage per lineage.

### 3.3 3+2 (κ3×edge): newborn lineage × each older κ3 lineage

Couplings, using the older lineage s′ legs (A^{s′}, B^{s′}) at the birth layer, evaluated on the new hub's neuron:

- G^B_jm = w2_j u_m B^{s′}_jm
- G^A_jm = w2_j u_m A^{s′}_jm

The legs afterwards are the new lineage's (A, B) and the older lineage's current (A^{s′}, B^{s′}), so there is no transport. Example (verified):

D4 = 12·rowsum((A∘B)∘[(A^{s′}∘A^{s′})G^{Bᵀ} + 2(A^{s′}∘B^{s′})G^{Aᵀ}])

This costs 2 u per (s, s′) pair per layer, and there are ~L³/6 ≈ 680 pair-layers at L = 16. D31 follows the same pattern; D22 has n⁴ cross pieces. The 3+2 term also couples to the non-lineage parts of κ3: K3.5, K3.6, and the heavy legs.

### 3.4 How this relates to the K=3 AUGMENT r=1 core (trace vs. traceless)

1. **The r=1 core is P_{≥1}κ4 = Sym(core⊗g).** It is fixed by the traces Σ_k κ4_abkk and comes only from repeated-index births (K4.2) and the factored-κ3 traces (K4.3, K4.4). Its readouts are D4 = g_aa·core_aa, D31 = (core_aa g_ac + core_ac g_aa)/2, D22 = (core_aa g_cc + core_cc g_aa + 4 core_ac g_ac)/6. With g = m·I these are m·core_aa, m·core_ac/2, m(core_aa+core_cc)/6, all O(n²).
2. **The top-slice births are supported on distinct indices, so their trace is exactly zero at birth** (checked numerically). P_{≥1} of them is therefore 0, and factor_k3 never forms them: the `len(int_part) ≤ 3` filter at `factor_k3.py:332`. They are pure harmonic (traceless) content.
3. After one W the transported star or path acquires both trace and traceless parts (trace ≠ 0 because WᵀW ≠ mI). K=3 AUGMENT has none of it.
4. **K=3 AUGMENT and K=4 SIMPLE use identical Wick terms for mean, variance and covariance** (§1.4, enumeration). So the K=4 increment there is exactly ΔD = D_exact − D_{Sym(core⊗g)}, i.e. the traceless part of the repeated-slice births plus the transported top-slice births (star, path, 3+2, old κ4).
5. **What K=3 AUGMENT already keeps:** the traces of the repeated-slice births; the κ4 (2,2) heavy leg (as the r=1 surrogate); and the κ4 (2,1,1) → κ3 feedback (K3.5), at trace level only.
6. **What is K=4-only:** the top-slice births and the off-trace structure of the repeated slices.
7. **No double counting** arises if star, path and 3+2 lineages are added on top of the core. Their trace is zero at birth, and the core never ingests them.
8. **One caveat.** Factored births evaluated on repeated indices are nonzero. For example, the star at birth has T_aaac = w3_c X_ac³ and T_aabc = w3_b X_ab² X_cb + w3_c X_ac² X_bc. ARC subtracts these (`factor_k4.py:570-572`) and re-adds the exact repeated slices through the correction block (`:640`).
   - Keeping them is an approximation of unknown size: they are legitimate separate-vertex diagrams, but with cumulant rather than power-cumulant bookkeeping.
   - Measure it at small n against ARC.

---

## 4. K=4 terms that do NOT share factors, and their share of cost

**Non-shared terms, i.e. own transport needed (or n⁴ readout):**

1. **Repeated-slice correction blocks** (`from_dstensor(K_ds[4])`, one per layer, `:640`). B carries 6·K_(2,1,1), an n×n matrix per column. The alternative is to project to r = 1, which is exactly what K=3 AUGMENT does.
2. **κ4 → κ3 feedback** through the (2,1,1) slice (k4[1,1,2]). For star lineages this spawns new κ3 terms with Hadamard-product legs (w2∘A∘B, w1∘A, …), which need their own transport. For path, 3+2 and correction blocks it has n⁴ cross pieces.
3. **3+2** cross-lineage couplings (O(L²) pairs).
4. **Path and 3+2 D22 cross pieces**: n⁴.
5. **AUGMENT only:** the κ5 r=1 core (n³, 6n⁴ per layer); WK2111 blocks (full-rank A per column); K2111 (n³R); heavy-leg births whose legs are κ3 or κ4 slices (κ3-heavy-edge × κ3, κ4(2,2) × κ3, …); the dense n⁴ (2,1,1,1) slice.

**Cost share in ARC's factor_k4 SIMPLE** (n = 1024, L = 16; total 4064n⁴):

- block transport 1920n⁴ (**47%**);
- (2,1,1) slice of transported blocks 1920n⁴ (**47%**);
- birth cache 128n⁴ (3%);
- dense κ3 transport 96n⁴ (2.4%).

Half of the transported blocks are correction blocks (non-shared). The other half are birth blocks whose B mixes star (shared), path (shared legs) and 3+2 (cross).

In a lineage reformulation the shared 47% transport vanishes for star, path and old κ4. What remains is the non-shared items above plus readouts. The star subset alone costs ≤ 3 u per lineage-layer.

**AUGMENT shares** (rough): transport 18%, (2,1,1) slice 18%, K2111 19%, dense n⁴ κ5 slice ~42%, other ~3%.

---

## 5. Practical: width sweep at depth 16

### 5.1 What `scripts/kprop_by_width.py` supports

- `KPropByWidthCfg` (`:150-165`) fields: `k_maxs`, `ns`, `sample_k_max`, `output_d_max`, `samples`, `batch_size`, `batches_per_bs`, `trials`, `base_seed`, `mlp_kwargs`, `kprop_kwargs`, `kind` (`'harmonic'` = `kprop_harmonic`, `'ds'` = deprecated).
- `kprop_kwargs` takes `kind=HKind.SIMPLE|AUGMENT|BASE`, `factor` (True = factored for k_max 3 and 4; ignored for k_max ≤ 2; False = dense "harmonic"), `use_avg_metric`, and `use_pK` (False only for BASE).
- Per (n, trial) it does three things:
  1. Seeds and builds the MLP (`:225-233`).
  2. Samples cumulants with bootstrap SEs, cached under `data/<sampK_cache_name>/seed*/n*/kmax*_dmax*/N*` (`:235-276`). The cache asserts that the weights match (`:244-249`).
  3. Runs `mlp_kprop(..., output_all=True)` for each k_max, largest first (`:299-305`). It records `err` = mean over entries of (est − sample)² per layer l ∈ {pre*, act*} and degree d (`:328-347`), plus `se2` (`:280-291`).
- It uses multiple GPUs if more than one is present (`:442-449`).
- ARC's own runs: widths (8…256), k_max 0..4, `num_layers=16` (= **15** ReLU layers), `factor=True`, both kinds × both metric settings (`:491-522`).

### 5.2 Configuration

Save as `mlp_cumulant_propagation/scripts/r3_sweep.py` and run **from `scripts/`**: `cd scripts && uv run python r3_sweep.py`.

```python
import logging
from kprop_by_width import KPropByWidthCfg, kprop_by_width, HKind  # import sets grad off, fp32, cuda
logging.basicConfig(level=logging.ERROR)
COMMON = dict(
    samples=2**32, batch_size=2**17, batches_per_bs=2**7,     # 256 bootstraps; samples % (bs*bpb) == 0
    trials=3, base_seed=0,
    sample_k_max=1, output_d_max=1,                           # means only; keeps factored towers unmaterialised
    sampK_cache_name="relu_w2/cache/sampK_L16",               # NEW name (weights differ from ARC's He init)
    mlp_kwargs=dict(num_layers=17, nonlin="relu", init_kind="manual", w_var=2.0),  # act0..act15, all W~N(0,2/n)
)
NS = (64, 96, 128, 192, 256)
RUNS = [  # (name, k_maxs, kprop_kwargs, widths)
    ("k3aug_dense",  (3,), dict(kind=HKind.AUGMENT, factor=False), NS),   # clean K=3 AUG (keeps kappa3 x edge)
    ("k3aug_fac",    (3,), dict(kind=HKind.AUGMENT, factor=True),  NS),   # ARC factored AUG (drops non-hypertree)
    ("k4simple_fac", (4,), dict(kind=HKind.SIMPLE,  factor=True),  NS),   # == dense SIMPLE (test_factor_k4.py:89-127)
    ("k4aug_fac",    (4,), dict(kind=HKind.AUGMENT, factor=True),  (64, 96, 128)),  # 160 may fit; 192+ > 32 GB
]
for name, k_maxs, kk, ns in RUNS:
    cfg = KPropByWidthCfg(k_maxs=k_maxs, ns=ns, name=f"relu_w2/{name}/kprop_by_width",
                          kprop_kwargs=dict(**kk, use_avg_metric=False), **COMMON)
    kprop_by_width(cfg)   # writes ../data/relu_w2/<name>/kprop_by_width.pt = (cfg, errs, sample_se2)
```

**Analysis.**

- From `errs`, keep `l.startswith('act')` and `d == '1'`. Use the de-biased MSE = mean_t(err) − mean_t(se2) per (n, k, l).
- Also compare estimates *directly* (K4 − K3), which has no sampling noise.
- To size each K=4 piece, run `k4simple_fac` at n = 64–128 with a patched copy of `factor_k4.py` that skips one of the path loop (`:478-487`), star loop (`:488-497`) or 3+2 loop (`:498-506`). The correction block adapts automatically.
- "Star only" is the cheap shared-factor variant from §3.1.

### 5.3 Memory (fp32; 16 ReLU layers; estimates, so measure `torch.cuda.max_memory_allocated()` at n = 64/96/128 and extrapolate)

| config | dominant objects | scaling | approx n_max at 32 GB |
|---|---|---|---|
| K=3 SIMPLE factored | 3 legs × nR, R = 2n(ℓ+1) | n²L | ≫ 1024 |
| K=3 AUG factored | dense (2,1,1) n³ slices (4–6 live) | n³ | ~1024 borderline; 768 safe |
| K=3 AUG dense | κ3 core n³ + slices (8–10 live) | n³ | ~768 |
| **K=4 SIMPLE factored** | 3–4 live copies of 2n²R, R = 32n at the end | n³L | **~290–320 (256 safe: 13–17 GB)** |
| K=4 SIMPLE dense | n⁴ core + 6–9 n⁴ temporaries (symmetrize, allclose, pK slice) | n⁴ | ~150 (ARC: n > 128 OOM, `time_utils.py:208`) |
| **K=4 AUG factored** | 3–4 n⁴ (2,1,1,1) slices + 2–3 × 2n²R, R ≈ 108n | n⁴ + n³L | **~160 (128 safe: 7–10 GB)** |
| K=4 AUG dense | n⁴ core + (2,1,1,1), (2,2,1,1) slices | n⁴ | ~130 |

Sampling: `mlp(X, output_acts=True)` stacks pre and act for all layers, about 2·batch·L·n·4 B. That is 17 GB at batch 2¹⁹ and n = 256, so use 2¹⁷.

### 5.4 Gotchas

1. **Layer-0 variance.** He init gives layer 0 a weight variance of 1/n (`mlp.py:237`, `w_var_by_layer=[1.0]`). The competition uses 2/n, so use `init_kind='manual', w_var=2.0`. That also sets `init_scale` and hence the avg metric (`kprop_harmonic.py:578-579`). Alternatively, copy real weights into `mlp.Ws[i].weight.data`. Use a new `sampK_cache_name`, or the weight assertion at `:244-249` fails.
2. **`num_layers` counts Linear layers.** act0..act15 needs `num_layers=17` (the last Linear is unused). ARC's "depth16" caches are 15 ReLU layers. Layer names are pre0…pre16 and act0…act15 (README:42-50; `kprop_harmonic.py:564-602`).
3. **Grad must be off.** `factor_k4` uses `torch.mul(..., out=tmp)` (`:483, 493`) and in-place ops on tensors built from `nn.Parameter` weights (`kprop_harmonic.py:565`). Without no-grad these fail or build huge graphs. The script sets `torch.set_grad_enabled(False)` at import (`:28`). In your own code, wrap calls in `torch.no_grad()`.
4. **Output paths.** The result goes to `../data/{name}.pt` relative to the **cwd** (`:468-469`), while the sample cache uses `__file__` (`:235`). Run from `scripts/`.
5. **Keep `output_d_max ≤ 2`.** `clone_tower` per layer, plus `.to_tensor()` in the error loop (`:338`), would materialise factored κ3/κ4 (n³/n⁴).
6. **fp32 throughout** (`:30-31`). Dense or symmetric checks (`check_symmetric_or_warn`) emit warnings and allocate extra full-size temporaries. Consider float64 at n ≤ 128 for K=4 − K=3 differences.
7. **Factored AUGMENT ≠ paper AUGMENT.** It drops κ3×edge and the triangle at K=3, and κ4×edge, cycles and κ3·κ3 at K=4 (TODO.md:28-39). The TODO also notes that k_max ≥ 2 caches and FLOP fits are stale.
8. **`use_avg_metric`.** True replaces W·g·Wᵀ with the scalar `init_scale[l]`. It is only consumed when a radial piece exists (`metric_needed`, `kprop_harmonic.py:247-259`). With a diagonal metric, the K3.5 B1 block is pure repeated-index content.
9. **Defaults are heavy.** `SAMPLES=2**34` with 5 trials (`:491`, `:159`). The sampling floor is se2 ≈ Var/N.

---

## Appendix A: verification artefacts (numpy; torch not used)

- `scratchpad/r3check/verify_readouts.py` re-implements ARC's `FactoredTensor4` births (A = sym(δ X), B_star = 4w3 X X, B_path = 12 w2 C w2 C w1) and the K=3 block (X, 3I, X·w2). It checks:
  - distinct-index entries equal the labelled-diagram sums (coefficient 1);
  - star D4/D31/D22/(2,1,1), path D4/D31/D22, K=3 D3/D21, and r=1 Sym(core⊗g) slices against transported full tensors (rel. err ≤ 4e-15);
  - trace of the top-slice star part = 0 at birth and ≠ 0 after one transport.
- `scratchpad/r3check/verify_3p2.py` checks the 3+2 cross-lineage D4 formula (rel. err 2e-15).
- `scratchpad/r3check/enum_k4_terms.py`, `enum_extra.py`, `enum_k3slice.py` and `count_top.py` (sharing `enum_k4_terms_lib.py`) restate `get_all_terms` / `factored_keeps_term` on ARC's `partitions.py`. Run with `python3 -I <script> arcrepo/patched/partitions.py`; that copy differs from the original only in py3.11 type-alias syntax. They yield:
  - SIMPLE K=4 top slice = {k4 transport ×1, 3+2 ×12, star ×4, path ×12}, all hypertrees.
  - AUGMENT K=4 top slice: 29 iso classes, 22 kept, 7 dropped.
  - K=3 AUGMENT top slice: 9 classes, 7 kept; the 2 dropped are κ3×edge and the triangle.
  - Mean/variance/covariance Wick-term sets identical for K=3 AUGMENT and K=4 SIMPLE.
  - factor_k4's dense κ3 (1,1,1) slice includes κ3×edge and the triangle (dropped in factored K=3 AUGMENT).
  - 80 iso-terms for the AUGMENT (2,1,1,1) n⁴ slice; 36 vector partitions of (2,1,1,1).
- `symb/` (quimb tensor-network kprop) implements only the deprecated `kprop_ds` algorithm (`symb/README.md`; `symb/kprop.py:29-30`). It is not relevant to K=4 factoring.
