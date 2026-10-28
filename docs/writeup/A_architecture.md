# A. Architecture of the shipped estimator — exact description and FLOP anatomy

*Reviewer: numerical-methods pass over the read-only snapshot `/home/claude/whest` (2026-10-19). Every claim is tagged **[code]** (read from source, cited `b636:LINE` = `src/estimator_b636.py`, `r1pa:LINE` = `src/estimator_r1pa.py`), **[docs]** (repo documents / ledgers, cited by path), **[measured]** (numbers I produced in this container with the repo's own shape-only flopscope interpreter, `docs/research/2026-10-15_exploration/audit/{fsmodel,census}.py`), or **[inference]**.*

## 0. What was audited, and how the two sources relate

| artefact | identity | role | bill |
|---|---|---|---|
| `src/estimator_b636.py` (5,754 lines) | sha256 `b636718510f68f7c…` **[measured]** | graded #336783, 2.0526e-9; parent of the current best | 305,090,053,346 F = 142.138 u, 22,217 ops **[measured]**, reproducing the repo's "142.14 u / 22,217 ops" to the FLOP |
| `src/estimator_r1pa.py` (3,795 lines) | sha256 `1951f647f77ad229…` **[measured]** | "R1 + P-A": the riding-band r288 / O704 / wall-aware-planner composition with the R1 refit, sibling of #336993 (never uploaded; "1951 stays fallback", `RESULTS.md:7380-7400`) | 291,322,458,915 F = 135.724 u, 22,454 ops **[measured]** |
| #336993 source `8be40031` | **not in the snapshot** | current best: "Literal Claude A4 rank288/O704 + faithful R1" (`RESULTS.md:7479,7509`) | 286,868,957,368 F / 22,235 ops, F/B 0.13045, raw 1.4536e-8, score 1.8962e-9 **[docs]** |

**How 8be differs from the two files I have** **[docs]** (`experiments/2026-10-18_p12/claude_handoff/README.md` §1, `RESULTS.md:7044-7060`): 8be = b636 + patch chain A1→A4: `STAR0` (`_STAR_GAIN = 0`), `GQ1` (`_K4T1_GQ = 1`), `SJG` (inert under riding), `LL704` (last-layer O-set 704 rows by `argsort(-|α|)`), riding band `r288`, planner (`_SW_PAD_KMAX=0`, `_SW_OP_FLOPS=1e6`, `_SW_WALL_FLOPS=4e8`), an "op diet" (−1,250 ops, −0.07 % F), then the R1 refit of the 28 fitted gain tables and the corrector transplanted onto it. `r1pa` (1951) has the riding band, O704, R1 gains and the planner at `_SW_OP_FLOPS=2.5e6`, but still carries `_STAR_GAIN ≠ 0` and `_K4T1_GQ = 0.5` (r1pa:2875, r1pa:2885) **[code]**. Running r1pa with `_SW_OP_FLOPS=1e6` reproduces 8be's real-meter transport bucket **to the FLOP** (104,248,748,525 F / 1,847 ops, vs `experiments/2026-10-19_research_lead/cost/analysis_3477/REPORT.txt`) and its covariance, old-tier and terminal-layer buckets within 0.3 % **[measured]**, so I use *r1pa @ OPF 1e6* as the per-site proxy for #336993 and quote the real-meter group totals where they exist.

Structural constants that differ between b636 and r1pa **[code]**: `_SHARED_RANK` 256→288 (b636:97 / r1pa:97), `_SW_OP_FLOPS` 4.2e6→2.5e6, `_SW_WALL_FLOPS` 0→4e8, `_SW_PAD_KMAX` 1024→0 (b636:1519-1572 / r1pa:1575-1628), `_K2_O[15]` 832→704 (b636:2308 / r1pa:2364), the last-layer O-set ranking `argsort(alpha)`→`argsort(-|alpha|)` (b636:4988 / r1pa:3029), and the join `_shared_join_lean` (b636:1118 re-truncating join vs r1pa:1224 riding band, `_RB_ENABLED = True` r1pa:1221). 28 fitted tables were refit (R1): `_LAM_GAIN, _THETA_GAIN, _FEED_GAIN, _TT_GAIN, _T1_GAIN, _CU_GAIN, _CB_GAIN, _YP_D4_GAIN, _YP_Q31_GAIN, _K0_VG, _K2_VG4, _HUB_A, _CORR_B64, _DRESS_DG, _STAR_GAIN, _K4T1_{B1,B2,C0,CD,CM}, _K4M_{QS,DS}, _K4T15_{C31,C211,CS}, _K4T2_{C31,C211,CS}` **[measured, by diff]**.

Units: 1 u = one 1024³ matmul as flopscope bills it = 2·1024³−1024² = 2,146,435,072 F; 1 GF = 1e9 F; B = 2⁴¹ = 2,199.0 GF; the 0.1 floor is 219.9 GF. n = 1024 throughout; "n³-class" means a product billed at ≥ 0.1 u.

---

## 1. THE CHAIN, layer by layer

### 1.1 The carried state

`_layer` (b636:4885) consumes and returns an 8-tuple `(mu, cov, tw, c4, gk4, lv, yp_meta, yc_meta)` (b636:4919-4920, 5564) **[code]**:

| field | shape / dtype | meaning |
|---|---|---|
| `mu` | (n,) f32 | post-ReLU mean of the previous layer (already corrected by the in-loop corrector) |
| `cov` | (n, n) f32 dense symmetric | post-ReLU covariance, **full rank, exact second cumulant in transport** |
| `tw` | `_Tower` (b636:1296-1384) | the carried third cumulant κ3 **plus** the κ4 hub pieces, in factored form (§1.2) |
| `c4` | Python list of (3, h_b) f32 per lineage | per-lineage "birth hub vectors" `[w(3,1); t31; t211]` (b636:5546-5557), consumed by the κ4 lineage read-outs (fold, star, Tier-2 D4) |
| `gk4` | (n,) f32 | per-neuron κ4 core `g` (504aldo's P1), transported by `(Wᵀ∘Wᵀ)` (b636:5133) |
| `lv` | int index (K_L,) or None | "covlive" rows of `cov` (α > −3.5 at the previous layer, `_CL_K`, b636:2299) used by the next sandwich |
| `yp_meta` | (mapping, ratio) | the young reader's H→R column map and `w2m/w1t` ratios (b636:915-931) |
| `yc_meta` | None | covariance-proxy channel, inert (`_YC_D4_GAIN` all 0, b636:822) |

At layer 0 the state is exact: `mu = 0, cov = I, tw = None, gk4 = 0` (b636:5710-5717).

### 1.2 The κ3 tower (what a "lineage" is)

One lineage is the κ3 content born at one layer. It represents (b636:1283-1294) **[code]**

    κ3_b = Sym( Σ_j A_D[:,j] ⊗ M[:,j] ⊗ M[:,j] ) + Sym( Σ_j A_L[:,j] ⊗ 3 M[:,j] ⊗ C_L[:,j] ) + hubs

with `C_L = A_L·diag(w2b)` never stored (b636:1331-1344) and `A_D = M·d(sb) + 3·A_L·d(eb) + Z·Lᵀ` never materialised (b636:1348-1353). The `_Tower` holds all lineages oldest-first as wide blocks:

| field | shape | transported? | notes |
|---|---|---|---|
| `A_L` | (n_R, Σ_b a_b) | yes | "two-leg" A leg of the **recent** lineages; a_b = `_K3_A[b]` kink columns (b636:2311) |
| `M` | (n_R, Σ_{b<youngest} h_b) | yes | mode leg of recent lineages except the youngest (whose M = I is implicit, materialised as Wᵀ[:,H] at transport, b636:1906-1907); h_b = `_KAS_K["H"][b]` live birth columns (b636:2314) |
| `w2b`, `sb`, `eb`, `CA`, `CB` | (Σ_b h_b,) | no (static column scales) | birth Wick w2, diag-κ3 scale (S3), separable (2,1) scale, hub column scales — **for all lineages incl. old** |
| `Z` | (n_R, 4k) | yes | rank-4 P2 residual left factors, all k lineages (`_P2_RANK = 4`, b636:133) |
| `L` | list of (g, h, 4) pages | no | static right factors of the P2 residual |
| `Y` | (n_R, 2k) | yes | hub left factors (routes i, ii), all lineages |
| `Q` | (n_R, r) | yes | old-tier basis, r = 256 (b636) / 288 (r1pa); orthonormal at the join in b636, a **non-orthogonal riding band** in r1pa |
| `FA`, `FM` | (r, Σ_old a_b), (r, Σ_old h_b) | no | old lineages' dense legs in band coordinates: `A_L^old ≈ Q·FA`, `M^old ≈ Q·FM` |
| `Sg` | (r, r) | — | carried Gram of the old content (b636 only; `None` under riding, r1pa:1269) |
| `rows`, `hcols`, `kd`, `aw` | indices | — | KAS compaction: R set of kept rows, H set of the youngest, (K1 rows, D cols) of this layer's D21, per-lineage A widths |

So the carried κ3 is **k = (#layers so far) lineages**: the 6 youngest ("exact tier", `_SHARED_AGE = 6`, b636:96) with dense legs, the older ones with dense legs replaced by `Q·FA`, `Q·FM` but with their thin legs (`Z`, `Y`) and column scales kept exactly. There are **no** per-lineage full-rank κ4 carriers; κ4 is `gk4` (diagonal core) + the `c4` birth vectors reused on the κ3 legs.

### 1.3 One layer, in execution order

Let `W` be the (n, n) weight, `WT = W.T` (b636:4922), `l` the layer index.

**(a) Linear step of μ and C.** `mu_pre = Wᵀ mu` (b636:4940). Covariance: `cov_pre = W[L]ᵀ C[L,L] W[L]` restricted to the live rows L of the previous layer (b636:4957-4964), done as `X = W[L]ᵀ C[L,L]` (one product, b636:4960) and the two-level symmetric second product `_p4_sym_rest2` (b636:2061-2073: rows [:512] full, rows [512:768] on cols 512:, rows [768:] on cols 768:, the rest transposed). Layer 0 uses `WᵀW` (b636:4954-4956). At the last layer only `var = rowsum((Wᵀ C)∘Wᵀ)` (b636:4941-4952). `var = max(diag, 1e-12)`, `sigma = √var`, `alpha = clip(mu_pre/sigma, ±40)` (b636:4968-4975).

**(b) Wick coefficients.** `w(k,p) = E[dᵏ/dzᵏ ReLU(z)ᵖ]`, z ~ N(mu_pre, σ²), for the 16 (k,p) pairs (b636:359-422; table decoded **[measured]**): `w(0,1) = σ(φ + αΦ)`, `w(1,1) = Φ`, `w(2,1) = φ/σ`, `w(3,1) = −αφ/σ²`, `w(4,1) = (α²−1)φ/σ³`, `w(0,2) = σ²(αφ + (1+α²)Φ)`, `w(1,2) = 2σ(φ + αΦ)`, `w(0,3) = σ³((2+α²)φ + (3α+α³)Φ)`, `w(0,4) = σ⁴((5α+α³)φ + (3+6α²+α⁴)Φ)`, the rest are `p!/(p−k)!`-reduced bases. One batched Horner (b636:289-299).

**(c) KAS index sets** (b636:4983-5053) **[code]**: from `argsort(alpha)` the layer builds O = top-`_K2_O[l]` by α (rows on which the κ3 slices are read; at the last layer the 704 smallest |α|, r1pa:3029), the next covlive set `_CL_K[l]`, R = top-`_KAS_K["R"][l]` (rows kept in the tower after this layer), H = top-`_KAS_K["H"][l]` (live birth columns of the newborn), A = the `_K3_A[l]` smallest α² inside H (kink columns that carry `A_L`, `w2b`, hubs), D = top-`_KAS_K["D"][l]` (D21 columns formed), K1 = the `_K1_ROWS[l]` smallest α² (D21 rows formed). Shipped values b636:2299-2314 (e.g. layer 14: O 832, R 736, H 800, A 576, D 800, K1 576, CL 864).

**(d) κ3 transport.** `adv = _advance(WT, tw, _orow)` (b636:5003, 1873-1920): **one fused product** `Wᵀ[O, R] @ [A_L | M | Z | Y | Q]` (padded to a multiple of 32 columns, b636:1887-1893), the youngest lineage's M entering as `Wᵀ[O, H]` for free (b636:1906-1907). Static scales (`w2b, sb, eb, L, CA, CB, FA, FM, Sg`) are untouched because every operation on a leg is a left multiplication (b636:1908-1912).

**(e) κ3 slices.** `D3 = κ3_iii` elementwise over all `k n` columns (b636:2146-2166): `d3 = Σ_j (A_D∘M)_ij M_ij + 3 Σ_j A_ij M_ij C_L,ij + 2 Σ_h Y_ih RY_ih`, with `A_D∘M = (M∘M)sb + 3(A∘M)eb + (Z Lᵀ)∘M` (`_p2_recon`, b636:1983-2021). Rows outside O get the "gain-mode" fill `D3 = 6 v μ var` with a scalar `v` fitted on the kink neurons (b636:5058-5068).
`D21 = κ3_iij` (b636:2171-2239): left factors `LA = 2(A∘M)w2b + (M∘M)eb + (2/3)(G∘M)`, `LP = (A∘A)w2b + (M∘M)sb/3 + (2/3)(A_D∘M) + (2/3)(G∘A)` (b636:2199-2207) plus the K9.2 **fold** (κ4 (3,1)/(2,1,1)/star births of the recent lineages added into LA/LP on rows |α| ≤ 2 with `rv = −α/σ`, b636:952-987, 5078-5084), then the contraction `D21 = ([LA|LP] @ [A_L|M]ᵀ + PPL @ Zᵀ/3 + (2/3) RY @ Yᵀ)∘off_mask` performed by `_yp_contract` (b636:933-1042) as two products (§2.2), masked to (K1 rows × D cols) and scattered back (b636:2233-2238), with the gain-mode fill `D21_ac = 2v(2μ_a C_ac + μ_c σ_a²)` on dropped rows (b636:5096-5101).

**(f) κ4 core and slices** (b636:5130-5198):
`dG = (Wᵀ∘Wᵀ) @ (gk4 − λ_{l−1}·diag C) + λ_{l−1}·var` (b636:5133), λ from `_K4_LAM`×1.225×(1−0.1·phase)×spline×`_LAM_GAIN` (b636:2545-2560).
`D22 = (m/6)(dG_i + dG_j)∘off_mask`, m = `metric` = 2.0 (b636:5196).
`D4`: shipped `m·dG` (b636:5139) (+ `star` = `4 Σ_j w3_j (A³∘M)_ij` ×`_STAR_GAIN`, b636:5143-5151 — zero in #336993), then the **Tier-1 law** `D4 ← K4M_DS·D4 + m·[C0·dG + CD·D3²/var + CM·μ·D3 + Σ_lineages(2C31 (M³∘A)·t31 + 3C211 (M²∘A²)·t211 + 2CS (A³∘M)·w3)]` (b636:5153-5172; the lineage sum runs over **all** lineages via `_full_legs_lean`, b636:5161), plus the **tp2 young reader** `D4 += g_l · 12 Σ_{j∈A} (Dx_ij w2_j/w1_j) M_ij A_ij` (b636:1022-1027, 5173-5178), and `_k2_fill` outside the O rows (b636:4868-4875, 5187-5188).
`Q31` (closure only, normalised): #336993 (GQ=1) uses kai's two-term law `Q31n = b1·Cn + b2·D3n_i·D21n_ij` (b636:646-647, `_K4T1_B1/B2` b636:4839-4840); b636 mixes 0.5 of `(m/2)λC/σ² + _Q31_COEF polynomial` with it (b636:648-660). There is **no nested-Dx (U2b) reader** in either file — that was a proposed P3 patch (`claude_handoff/README.md` §2) **[docs]**.

**(g) The ReLU closure** (b636:5200-5250, tables decoded **[measured]**): first-order Gram–Charlier in the carried cumulants, with raw post-ReLU moments

    pK_p      = w(0,p) + (D3/6)·w(3,p)·T1 + (D4/24)·w(4,p)·T1'           p = 1..4   (fast_1d, b636:5202-5214)
    pK_11     = C w1ᵢw1ⱼ + ½C² w2ᵢw2ⱼ + ½(D12 w1ᵢw2ⱼ + D21 w2ᵢw1ⱼ) + ¼D22 w2ᵢw2ⱼ
    pK_21     = ½C² w(2,2)ᵢw2ⱼ + C w(1,2)ᵢw1ⱼ + ½D12 w(1,2)ᵢw2ⱼ + ½D21 w(2,2)ᵢw1ⱼ + ¼D22 w(2,2)ᵢw2ⱼ
    pK_22     = ½C² w(2,2)ᵢw(2,2)ⱼ + C w(1,2)ᵢw(1,2)ⱼ + sym(D12 w(1,2)ᵢw(2,2)ⱼ) + ¼D22 w(2,2)ᵢw(2,2)ⱼ

(C = `C11 = cov_pre∘off_mask`, w1 = w(1,1), w2 = w(2,1); 25 table rows, per-row gains `_TT_GAIN`), **plus the θ-damped second-order "closure extras"** (`_closure_apply`, b636:603-684): univariate `D3n²/72·H(6,p) + D3n·D4n/144·H(7,p) + D4n²/1152·H(8,p)` and **27 bivariate monomials** (`C·D12, C·D21, C·D22, C·D3ᵢ, …, D22·D22, D22·D4ⱼ, Q13, Q31`, each `coef·Hᵢ(kᵢ,·)·Hⱼ(kⱼ,·)`, coefficients 1/2, 1/4, 1/6, 1/8, 1/24, 1/32, 1/48, 1/96 …), all multiplied by θ = 0.625·spline·`_THETA_GAIN[l]` (b636:596-600) and per-monomial gains `_CU_GAIN`, `_CB_GAIN`. `H(k,p) = σᵖ(P1_k(α)φ + P2_k(α)Φ)` are the higher Wick coefficients (b636:470-481).
Cumulants: `K1 = pK1`, `K2 = pK2 − pK1²`, `K11 = pK11`, `K21 = pK21 − 2pK1 pK11`, `K22_ij = pK22 − 2 pK1_j pK21_ij − 2 pK1_i pK12_ij + 4 pK1_i pK1_j pK11_ij − 2 pK11_ij²`, `K3 = pK3 − 3pK1pK2 + 2pK1³`, `K4 = pK4 − 4pK1pK3 − 3pK2² + 12pK1²pK2 − 6pK1⁴` (`pK_to_K` table, b636:5309-5329) **[measured]**.

**(h) Mean out and the learned corrector.** `mu_out = K1` (b636:5331) then `mu_out += δ_l` (b636:5347-5355) where `δ_l = MLP_l(x)`: 9 raw per-neuron channels `[μ_pre, σ, α, D3/σ³, D4/σ⁴, ‖W[:,i]‖, ρ_l·‖D21_i‖²/σ⁴, μ_out, μ_out − mean(μ_prev)]` (b636:4684-4723), expanded to 33 columns (squares + 15 pairwise products of the first 6, b636:2630-2631, 4747), standardised, **33→32→32→1 tanh MLP with one weight page per layer** (b636:4726-4756; 16×2,177 = 34,832 weights in `_CORR_B64`), plus a cross-neuron "hub" input to the first layer: `einsum("nk,nm,kmh->nh", H5, P, _HUB_A[l])` with `H5 = [q2@[1,Φ,φ], Σq2², 1]`, `q2 = C²/(σᵢ²σⱼ²)`, `P = [1, α, α²−1]·σφ` (b636:2594-2608). At the last layer the correction is symmetrised ("local even"): `δ ← ½(δ(μ,D3,W) + δ(−μ,−D3,−W, μ_out−μ_pre))` (b636:5356-5360, 4878-4883). The correction is applied to the **state**, so it propagates; `cov_out`, births and `g_out` use the uncorrected `pK1` (b636:5364-5372, 5407).

**(i) Covariance out and κ4 core out.** `cov_out = diag(K2) + K11` (b636:5364); `g_out = (K4 + rowsum K22)·6/(n+4) − (ΣK4 + ΣΣK22)·3/((n+2)(n+4))` (r = 1 harmonic projection, b636:5538-5543).

**(j) Birth of the new lineage** (b636:5272-5307, 5387-5491): the newly created κ3 is what the closure produced minus what the aged tower already carries: `S3 = K3 − w1t³·D3`, `S21 = K21 − (w1tᵢ² w1tⱼ D21 + (fac1²·w2)∘off)` (b636:5305-5307, 5370-5371). The newborn's legs: `fac1 = (w1·g_e)[:,None]·C11 + H_dress` (b636:5288-5293; `_dress_H` b636:4803-4828 adds `½(w2 D21·g₁ + w1 D21ᵀ·r·g₀) + ¼ w2 D22 r + ½ w2 C²·r·g₂`, `r = −α/σ`), restricted to A columns; `w2b = w2·g_m` (gated by the per-neuron gate MLP, b636:4779-4787, 5278-5286); `e_b = w(1,2) − 2 pK1 w1` minus the hub compensation (b636:5407, 5447); `S_sep = e_b ⊗ fac1ᵀ`; residual `S21 − S_sep − hub content` compressed to rank 4 by a deterministic range finder (`_p2_compress`, b636:1947-1980: sketch = 4 strongest columns, one power iteration, two QRs) → `Zb (n×4), Lb (n×4)`; `sb = S3`; hubs `Y = [w2, dG·w2]`, `cA = (m/8)(λ w1 + dG w3)·feed`, `cB = (m/8) w3·feed` (b636:5415-5436). Appended as one lineage (b636:5483-5484) with the H/A column compaction (b636:5469-5482).

**(k) Ageing.** Before the append the inherited tower is Wick-scaled on every mode (`_wick_tower`: `A_L, M, Z, Y, Q ← diag(w1t)·(·)`, b636:1923-1944, 5377), row-compacted to R (`_kas_rows`, b636:2370-2384, 5378-5379) and, if it holds ≥ 6 exact lineages and `l < depth−1−3 = 12`, the oldest exact lineage is folded into the band (`_shared_join_lean`, b636:5380-5383). Joins therefore happen at layers 6..11 only (`_JOIN_SKIP_TAIL = 3`, b636:95).

**(l) Final-layer read-out** (b636:4941-4952, 5193-5194, 5220-5221, 5331-5362) **[code]**: at l = 15 only `var`, `D3` (needs the full κ3 transport, §2) and `D4` (Tier-1 law + `_k2_fill`) are formed, no D21, no covariance, no birth. The scored row is

    E[h_i] = σφ(α) + μΦ(α)                                   (Gaussian part, w(0,1))
           + (D3/6)·T1₀·(−αφ/σ²) + (D4/24)·T1₁·((α²−1)φ/σ³)   (first-order κ3, κ4 Edgeworth terms)
           + θ₁₅·[CU·D3n²/72·H(6,1) + CU·D3n·D4n/144·H(7,1) + CU·D4n²/1152·H(8,1)]   (second-order extras)
           + ½(δ₁₅(x) + δ₁₅(x̄))                                (local-even corrector)

with `D3n = D3/σ³`, `D4n = D4/σ⁴`, `H(k,1) = σ(P1_k φ + P2_k Φ)` (the Hermite/derivative-of-Gaussian coefficients E[ReLU^{(k)}]), then `where(isfinite, ·, 0)` and `max(·, 0)` (b636:5696-5697); row 0 is the closed form `‖W₁[:,j]‖/√(2π)` in f64 (b636:5678-5680). All 16 rows are returned; only row 15 is scored.

---

## 2. TRANSPORT COST STRUCTURE

### 2.1 Lineage lifecycle (shipped schedule) **[code + measured]**

| layer l | lineages k | exact (dense legs) | old (band) | join after read-out? | fused transport product O_l × R_{l−1} × cols | D21 formed rows × cols | cov live K_L | layer GF (8be proxy) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | | — (WᵀW only) | — | 1024 | 1.45 |
| 1 | 1 | 1 | 0 | | 1024×1024×1056 | 1024×1024 | 1024 | 6.05 |
| 2 | 2 | 2 | 0 | | 1024×1024×3104 | 1024×1024 | 1024 | 10.80 |
| 3 | 3 | 3 | 0 | | 1024×992×5152 | 992×992 | 1024 | 14.97 |
| 4 | 4 | 4 | 0 | | 992×960×7168 | 928×992 | 1024 | 17.67 |
| 5 | 5 | 5 | 0 | | 992×928×9088 | 896×960 | 1024 | 20.78 |
| 6 | 6 | 6 | 0 | yes (first; builds Q) | 960×896×11008 | 832×928 | 992 | 23.69 |
| 7 | 7 | 6 | 1 | yes | 960×896×11040 | 800×928 | 992 | 24.08 |
| 8 | 8 | 6 | 2 | yes | 928×864×10720 | 768×896 | 992 | 24.22 |
| 9 | 9 | 6 | 3 | yes | 928×832×10368 | 736×864 | 960 | 23.27 |
| 10 | 10 | 6 | 4 | yes | 896×832×10016 | 704×864 | 960 | 22.63 |
| 11 | 11 | 6 | 5 | yes (last) | 896×800×9696 | 640×832 | 928 | 22.34 |
| 12 | 12 | 6 | 6 | | 864×768×9344 | 608×832 | 928 | 20.77 |
| 13 | 13 | 7 | 6 | | 864×768×10784 | 576×800 | 896 | 21.63 |
| 14 | 14 | 8 | 6 | | 832×768×12192 | 576×800 | 896 | 22.58 |
| 15 | 15 | 9 | 6 | | 704×736×13568 | — | 864 | 11.53 |

(cols = Σ_recent a_b + Σ_recent-but-youngest h_b + 4k (Z) + 2k (Y) + r (Q), padded to 32; exact-tier count read from `_SHARED_AGE`/`_JOIN_SKIP_TAIL`, widths from the KAS tables, and cross-checked against the interpreter's recorded shapes.) Per-layer GF: `census` BY LAYER for r1pa @ OPF 1e6; 8be's real meter gives 11.455 GF for layer 15 (`analysis_3477/REPORT.txt`) vs 11.53 here.

### 2.2 Is each lineage transported separately through every later layer?

**Yes for its exact life, as part of one fused product; merged afterwards** **[code]**:

- A lineage born at layer b is transported with its own dense legs `A_L^b` (n_R × a_b) and `M^b` (n_R × h_b) at every layer b+1 … b+6 — six layers — by the single fused matmul `Wᵀ[O,R] @ [A_L | M | Z | Y | Q]` (b636:1887-1893). The cost per layer is therefore ≈ `2·k_rec·n³` naive (6 lineages → ~10–13 k columns, table above), not O(L²): the **age cap turns O(L²) into O(L·6)**. After its 6th read-out (joins at layers 6..11) the dense legs are folded into the band and only `Q` (one n_R × r block for *all* old lineages) is transported; the thin legs `Z` (4 columns per lineage) and `Y` (2 per lineage) of **every** lineage, old or not, stay exact and ride in the same fused product (b636:1887; `_shared_join_lean` keeps `Z, L, sb, eb, Y, CA, CB`, r1pa:1265-1268) — a genuine O(L²) but at 6 columns per lineage (≈ 90 of ~10 k columns at layer 14).
- Lineages born at layers 6..14 are never joined (`_JOIN_SKIP_TAIL = 3` blocks joins at l ≥ 12), so the exact tier grows to 9 lineages by the last layer (13,568 columns at l = 15).
- The last layer transports the whole tower (704 × 736 × 13,568 → 7.28 GF) and rebuilds the old legs (2.71 GF) **only to read D3 and the Tier-2 D4 lineage terms** on 704 rows (b636:5003, 5055, 5161; the D21 branch is skipped, b636:5069) **[code]**.

**Old-source compression** **[code]**: rank r = 288 (#336993) / 256 (b636); age at entry = 6 exact read-outs; in b636 every join re-truncates the whole old tier: new basis `Qn = qr(J·(Jᵀ Om)∘dJ + Q Sg Qᵀ Om)` from one weighted power iteration seeded by the previous Q (weights `dJ` = this layer's consumption energy of the joiner's columns, `_wts`, b636:1053-1055, 1189-1195), `FJ = Qnᵀ J`, old coordinates re-rotated `FF = (Qnᵀ Q) @ [FA|FM]` and the Gram `Sg` updated (b636:1198-1210). In #336993/r1pa the band is **not** re-derived: P is built once at the first join (layer 6) by the same weighted range finder on the lineage-0 joiner (r1pa:1248-1258), then carried, transported (`_tr(tw.Q)`, b636:1920) and Wick-scaled (b636:1940) like a leg — hence non-orthogonal (κ(P) 6.7 at L7 → 107 at L11 **[docs]**, `claude_handoff/README.md`) — and each later entrant gets least-squares coordinates `FJ = R⁻¹ Qᵀ J` from a temporary `qr(P)` (r1pa:1240-1242) appended to `FA`/`FM`; no re-rotation, no `Sj`/`Sg` (r1pa:1243-1246, 1269). Old content therefore transports **exactly** (linear) in band coordinates; the only truncation is the projection of each entrant onto a 288-dim subspace chosen for lineage 0's consumption pattern at layer 6 **[inference from code]**.

**The young read-out is re-associated ("partial young reader", `_YP_RESTRUCTURE = True`, b636:811)** **[code]**: instead of `[LA|LP] @ [A_L|M]ᵀ` on the *transported* legs, `_yp_contract` (b636:933-1042) contracts the left factors of the non-newborn recent lineages, the old-tier `Xold`, `PPL/3` and `RY·2/3` against the **pre-transport** legs kept from the previous layer (`_YP_PRE`, b636:4997): `Dx = [LAr⁻ | LPr⁻ | Xold | PPL/3 | (2/3)RY] @ [A_L⁻ | M⁻ | Q⁻ | Z⁻ | Y⁻]ᵀ` (K1 × R_{l−1}) (b636:998-1013), then folds `Dx` into the newborn's product through the newborn's `M = Wᵀ[:,H]`: `D21 = [LAr_new + ff | LPr_new + DxH + ee | Z_new/3 | (2/3)Y_new] @ [A_new | M_new | Z_new | Y_new]ᵀ` (b636:1016-1042). Algebraically `LA @ (WᵀA⁻)ᵀ = (LA @ A⁻ᵀ) W`; this is FLOP-neutral but makes `Dx` available for the tp2 D4 term and the Q31 feedback (`ee`, `ff`, `_YP_Q31_GAIN`, b636:1030-1037). The old tier's `Xold = [LA_old|LP_old] @ [FA|FM]ᵀ` (K1 × 2·ko·n × r, b636:1001-1004) is the band-coordinate read of the old lineages' contribution; their dense legs are rebuilt every layer by `F = Q @ [FA|FM]` (`_full_legs_lean`, b636:792-807) because `LA_old`, `LP_old`, `D3_old` and the Tier-2 D4 terms are **elementwise** in the legs.

### 2.3 Count of n³-class matmuls per layer, by category (mid layer, l = 9, 8be proxy) **[measured]**

| category | `_mm` calls | shapes (m × k × n) | naive GF | billed GF | Strassen depth |
|---|---|---|---|---|---|
| exact K2 chain | 4 | 1024×960×960; 512×960×1024; 256×960×512; 256×960×256 | 3.28 | 1.91 | 5,5,2,0 |
| κ3 transport (fused) | 1 | 928×832×10368 | 16.00 | 8.07 | 6 (padded m→960) |
| κ3 young read-out: Dx pullback | 1 | 736×9600×832 | 11.76 | 6.06 | 6 (padded m→768) |
| κ3 young read-out: newborn | 1 | 736×1696×864 | 2.16 | 1.18 | 5 |
| old tier: dense rebuild `Q@[FA|FM]` | 1 | 928×288×6144 | 3.28 | 1.84 | 5 |
| old tier: `Xold` | 1 | 736×6144×288 | 2.60 | 1.47 | 5 |
| old tier: join (riding, layers 6-11) | 1 (+QR, solve) | 288×832×1984 | 0.95 | 0.54 | 5 |
| birth: P2 range finder | 4 thin | 1024×1024×4 (×3), 4×1024×1024 | 0.03 | 0.03 | 0 |
| corrector / gate / hub | 6 tiny | n×33×32, n×32×32, n×32×1, n×3×8, n×8×2, n×n×3 | ≈0.01 | ≈0.01 | 0 |

≈ 9 billed-heavy products per mid layer, ≈ 40 GF naive → ≈ 21 GF billed, i.e. the whole layer is ≈ 19 n³-equivalents naive and ≈ 10 n³-equivalents after Strassen. Over the 16 layers: 189 `_mm` calls, 503.8 GF naive → 265.0 GF billed (ratio 0.526); non-matmul work 23.5 GF **[measured]**.

---

## 3. FLOP LEDGER for #336993 (8be), with b636 for comparison

Real-meter group totals for 8be are from `experiments/2026-10-19_research_lead/cost/analysis_3477/REPORT.txt` **[docs]** (the independent per-span census of the actual package on flopscope 0.12.3: 286,868,957,368 F, 22,235 ops, matmul raw 241.996 GF = 84.36 % in 571 calls). Per-site splits inside a group are my interpreter run of r1pa @ `_SW_OP_FLOPS=1e6` **[measured]**, which reproduces the transport group exactly and the others within 0.3 %. The b636 column is the interpreter on the actual b636 file (exact to the FLOP vs the served bill).

| # | item | #336993 GF | % of 286.87 | ops (8be) | b636 GF (305.09) | notes |
|---|---|---:|---:|---:|---:|---|
| 1 | **κ3 transport**: fused `Wᵀ[O,R] @ [A_L|M|Z|Y|Q]`, 15 layers (`_advance`, b636:1893) | 104.25 (site 103.94) | 36.3 | 1,847 | 104.98 | 15 products, 11 of them zero-padded to 2⁶ multiples; depth 6 on all ≥ layer 2 |
| 2 | **κ3 young read-out, Dx pullback** (`_yp_mm_parts` 1st call, layers 3-14) | ≈ 68.9 (8be bucket "earlier-history Dx pullback" 71.20 incl. glue) | 24.0 | 1,506 | 72.3 | inner dim 4k–11.6k = all non-newborn recent columns + Q + Z + Y |
| 3 | **κ3 young read-out, newborn product** (2nd call, layers 2-14) + layer-1 `_shared_contract_lean` | 20.6 + 2.1 (8be bucket 18.2) | 7.9 | 1,313 | 22.7 | inner dim a+h+4+2 ≈ 1.4–2.1k |
| 4 | **exact K2 chain** `W[L]ᵀC[L,L]W[L]` + layer-0 `WᵀW` + last-layer diag | 29.19 | 10.2 | 3,705 | 31.55 | `X = WᵀC` 1.0–1.1 GF + 2-level symmetric second product 0.8–0.9 GF per layer |
| 5 | **old tier: dense rebuild** `Q @ [FA|FM]`, layers 7-15 | 22.10 | 7.7 | 958 | 19.77 | grows with ko: 0.6 → 3.3 GF/layer |
| 6 | **old tier: `Xold` read** `[LA|LP]_old @ [FA|FM]ᵀ`, layers 7-14 | 13.88 | 4.8 | 808 | 12.60 | |
| 7 | **old tier: join** (riding: first join 1.8 GF = 2 range-finder products + `PᵀJ`; 5 later joins ≈ 0.55 GF = `QᵀJ` + `qr(P)` + `solve`) | 7.35 | 2.6 | 1,140 | 20.59 (re-truncating join: `J(JᵀOm)dJ` 8.4, `QnᵀJ` 4.2, `T@[FA|FM]` 3.2, `Sj` 1.6, `Q Sg QᵀOm` 1.2, qr 1.2, …) | the riding band saved 13.2 GF of join vs b636 |
| 8 | **closure** (27-monomial `einsum("tij,ti,tj->ij")` 3.8 GF, monomial stacks, univariate extras, Q31 law) | 5.02 | 1.7 | 1,347 | 5.21 | |
| 9 | **κ3 read-out elementwise**: P2 recon (`Zres`, `PPL`) 2.55; hub parts 1.12; LA/LP assembly + fold 1.61; D3 0.61 | 5.89 | 2.1 | ≈ 1,630 | 5.95 | O(k n²) Hadamards and `(n,g,w)@(g,w,r)` thin batched products |
| 10 | **`_layer` inline remainder** (pK table terms, S/K assembly, C11, D22, dG, D4 laws, fills, gathers) | 2.77 | 1.0 | **4,482** | 3.0 | 20 % of all ops for 1 % of FLOPs |
| 11 | **birth**: P2 rank-4 range finder 0.54; gate/dress/append 0.31; Wick scaling + KAS row gathers 0.56 | 1.41 | 0.5 | 963 | 1.4 | |
| 12 | **corrector** (features + 3 matmuls + hub features) | 0.34 | 0.12 | 801 | 0.34 | |
| 13 | Wick 1-D coefficients, constants, init, harness | 0.06 | 0.02 | 636 | 0.06 | |
| | **last layer alone** (transport 7.28 + old rebuild 2.71 + var 0.84 + D3/D4 0.7) | 11.45 | 4.0 | 708 | 13.05 | spent to read D3/D4 on 704 rows |
| | **total** | **286.87** | 100 | **22,235** | **305.09** | |

Grouped: young κ3 window (1+2+3) **193.7 GF = 67.5 %**; old tier (5+6+7) **43.3 GF = 15.1 %**; exact K2 chain **29.2 GF = 10.2 %**; closure + births + corrector + elementwise ≈ 15.5 GF = 5.4 %; glue/gathers ≈ 5 GF.

**Strassen setting and the planner** **[code]**: every `_mm` (b636:1753-1781) asks `_sw_plan(m,k,n)` (b636:1575-1615) for `(depth, use_sks, padded dims)`: it minimises `F + _SW_OP_FLOPS·ops + _SW_WALL_FLOPS·T` over depth 0..`_SW_CAP = 6` with leaf floor `_SW_LEAF = 8` (min padded dim / 2^d ≥ 8), where `F, ops` come from the exact recurrences `_swo_cost` (classic 7-product Winograd, 20–22 ops/level, b636:1529-1549) and `_sw_ks_cost` (alternative-basis Karstadt–Schwartz Winograd `_sks`, 18 ops/level after `_SW_KS_LTRIM`, b636:1552-1564), padding m, k, n up to the next multiple of 2^d is allowed when `_SW_PAD` and the padded inner dim ≤ `_SW_PAD_KMAX` (1024 in b636, unlimited in #336993; each pad bills numel + 1 op, b636:1601-1606), and `T` is the server-time model `7.56e-11·leafF + 3.26e-6·leafGEMMs + 8.89e-10·glueF + 0.4ms·ops` (b636:1523-1526, 1609). Shipped λ: `_SW_OP_FLOPS` 4.2e6 (b636) / 2.5e6 (r1pa) / 1e6 (#336993), `_SW_WALL_FLOPS` 0 (b636) / 4e8 (r1pa, #336993). Outcome on the 8be proxy **[measured]**: 24 products at depth 6 (155.7 GF), 72 at depth 5 (103.4 GF), 1 at depth 4, 15 at depth 2 (`_p4_sym_rest2` 256-row piece), 77 at depth 0 (thin); 20 padded products carrying 134.5 GF; `_mm` bill/naive = 0.526 overall (depth-6 sites ≈ 0.49–0.52, depth-5 ≈ 0.51–0.57). Without Strassen the same arithmetic would bill ≈ 504 + 23 ≈ 527 GF (F/B ≈ 0.24). `_sks` itself is the 2-D alternative-basis form (b636:1794-1870): both operands are transformed once into a 7-slot Morton buffer, one batched `fnp.matmul` on 7^L leaves, inverse transform; the op count is 18L+11 per product.

**Irreducible exact-K2 chain at this setting**: 29.19 GF = **10.2 %** of the #336993 bill, = **13.3 %** of the 219.9 GF floor; at depth 6 it is already billed at 0.58× naive (3.28 GF naive → 1.91 GF per layer); the 15-layer naive cost is 1.0·n³ + 0.75·n³·(K_L/n) ≈ 1.7 n³/layer.

**Ops**: 22,235 served; residual max 0.3389 s against the 0.4 s cliff (`RESULTS.md:7509-7515`) **[docs]**. By op name on the proxy: add 6,416, multiply 5,990, subtract 5,118, reshape 1,141, take 598, matmul 594, concatenate 425, stack 397, einsum 346 **[measured]**; ≈ 11 k ops are Strassen glue inside `_mm`, ≈ 4.5 k are the `_layer` body, 1.9 k the closure.

---

## 4. KNOBS (shipped values and what each trades)

| constant | b636 | #336993 / r1pa | where | trades |
|---|---|---|---|---|
| `_SHARED_AGE` | 6 | 6 | b636:96 | exact read-outs per lineage; age 5 saves ≈ 13 GF (−4.5 % F) for +6.15 % raw, age 7 costs +11.5 GF (`docs/research/2026-10-16_round3/cost2/REPORT.md` §2) **[docs]** |
| `_SHARED_RANK` | 256 | **288** | b636:97 | band rank: rebuild + `Xold` scale ∝ r (36 GF at 288); r320 measured −1.53 % raw for +2.4 GF (`claude_handoff/README.md`) **[docs]** |
| `_JOIN_SKIP_TAIL` | 3 | 3 | b636:95 | no joins at layers 12-14 → 9 exact lineages at the end; the last-layer transport is 13.6 k columns |
| riding band (`_RB_ENABLED`) | re-truncating join + `Sg` | **riding, no re-rotation** | r1pa:1221-1272 | join 20.6 → 7.3 GF; carried P non-orthogonal (κ up to 107) |
| `_P2_RANK` | 4 | 4 | b636:133 | residual (2,1) leg rank; 4 is at its cliff (rank 2: +11 % raw) |
| `_HUB_ROUTES`, `_HUB_COMPENSATE` | ('i','ii'), True | same | b636:136-138 | κ4→κ3 birth feed (2 hub columns per lineage); off = +5.4 % raw (Sep-25 doc) |
| `_CLOSURE_SCOPE`, `_OWN_THETA_FAM` | 'all', 0.625 | same | b636:141,152 | second-order extras at every layer; off = +18 % raw |
| `_K4_LAM`, `_OWN_S_LAM` 1.225, `_OWN_T_LAM` −0.1, `_LAM_GAIN` | table | refit | b636:160,153-154,522 | κ4 regeneration coefficient λ_l |
| `_STAR_GAIN` | per-layer 0.35–1.5 | **0 (STAR0)** | b636:4834 | star-D4 `4Σ w3 A³M`; a physics approximation dropped in #336993 at −1.09 % F together with GQ1/LL704 (`claude_handoff/README.md` A1) |
| `_K4T1_GQ` / `_K4T1_GD` | 0.5 / 0.5 | **1 (GQ1)** / 0.5 | b636:4844-4845 | GQ: weight of kai's Q31 law vs the λC law (1 = pure two-term law); GD: gates star formation |
| `_K4T1_{B1,B2,C0,CD,CM}`, `_K4M_{QS,DS}`, `_K4T15_*`, `_K4T2_*` | tables | refit | b636:4839-4861 | Tier-1 D4 law, mixes, fold coefficients, Tier-2 lineage D4 coefficients |
| `_YP_RESTRUCTURE`, `_YP_D4_GAIN`, `_YP_Q31_GAIN` | True, 0.37–0.77, 0.83–1.0 | True, 0.60–0.80, 0.87–1.09 | b636:811-813 | the young Dx reader: tp2 D4 term and Q31 feedback |
| `_FOLD_GAIN`, `_FOLD_TAU` | table, 2.0 | same | b636:4831-4832 | κ4 births folded into D21 on rows \|α\| ≤ 2 |
| `_DRESS_*`, `_DRESS_DG` | C2 1, D21 1, D21T 0.9, D22 1 | refit DG | b636:4795-4800 | dressed newborn leg families |
| `_GATE_ON/_GATE_ARM/_GATE_S` | True/'birth'/0.5 | same | b636:4765-4768 | per-neuron tanh gates on the birth vectors |
| KAS sets `_K2_O, _KAS_K{R,D,H}, _K1_ROWS, _K3_A, _CL_K`, `_D21E_RHO`, `_K0_VG`, `_K2_VG4` | tables | `_K2_O[15]` 832→**704** | b636:2299-2314 | decided-neuron pruning: rows/columns kept per role; last-layer O set by \|α\| in #336993 |
| `_SW_OP_FLOPS` | 4.2e6 | 2.5e6 (r1pa) / **1e6** (8be) | b636:1519 | FLOPs one op call is "worth" in the planner; lower = deeper Strassen, more ops (23,575 at 1e6 before the op diet) |
| `_SW_WALL_FLOPS` | 0 | **4e8** | b636:1520 | FLOPs one modelled server second is worth |
| `_SW_PAD_KMAX` | 1024 | **0 (any)** | b636:1572 | allows padded depth-6 on the long-k read-out products (−1.0 % F, +~19 % local wall per `STRATEGIST_ADDENDUM_2026-10-18_a.md` §2.3) |
| `_SW_LEAF`, `_SW_CAP` | 8, 6 | same | b636:1517-1518 | leaf floor and max depth; depth 7 was priced at a further −3.5 % F but wall-risky (`audit/REPORT.md`) **[docs]** |
| `_CORR_ON`, `_CORR_NF` 9, hidden 32, `_HUB_A` | True | refit | b636:2585,2626,2590 | the in-loop mean corrector (34.8 k weights) + hub features |
| `_LOCAL_EVEN_ENABLED` | True | True | b636:4876 | symmetrised last-layer corrector |
| `ALPHA_CAP` 40, `metric` 2.0, `_ELEM_PER_LAYER` 400, `_BUDGET_FRACTION` 0.95 | | same | b636:255, 5720, 2432, 2438 | saturation; κ4 normalisation; the (over-pricing, inert) budget guard |

---

## 5. WHAT LOOKS STRUCTURALLY EXPENSIVE (independent view, quantified)

1. **Two n³-class passes per recent lineage per layer, for six lineages, for ~14 layers — 194 GF (67.5 %).** Every exact lineage costs ≈ 1.35 GF of transport + ≈ 1.2 GF of (2,1) read-out per layer at mid-depth (8.07 GF / 6 and 7.24 GF / 6 at layer 9) **[measured]**. The transport and the read-out contract over the *same* `~2n` columns of each lineage, once as `Wᵀ·legs` and once as `left-factors·legsᵀ`; no algebraic identity merges them because the read-out's left factors are Hadamard products of the transported legs (b636:2199-2207). The representation itself (n birth terms per lineage, two dense legs) fixes this at ≈ 4 n³ naive per lineage-layer; the only levers inside it are the ones already used (KAS column/row pruning, Strassen) or fewer exact read-outs (age), and the docs record age 5 as score-negative. **[inference]** The one structural asymmetry still open: the transport is issued on the O rows (704–1024), the next layer consumes the legs only on the R rows (R ⊂ O, 64–128 fewer: next transport inner dimension and next Dx right operand, b636:1882, 999) and this layer's D21 only on K1 rows × D columns; the O∖R rows (≈ 10 % of the transport, ≈ 10 GF over the chain) serve only this layer's elementwise consumers (D3, hub `RY`, the left factors on rows that are then dropped). Whether D3 on those rows is load-bearing for the score is not measured in the files I read.

2. **The old tier is read densely every layer: 36 GF (12.5 %) for content whose left side is rank 288.** `Q @ [FA|FM]` (22.1 GF) exists only to feed elementwise consumers (`LA_old`, `LP_old`, `D3_old`, Tier-2 D4), and `Xold` (13.9 GF) immediately projects their product back to the band. Both are `2·ko·n²·r` per layer and grow linearly with ko (0.6 → 3.3 GF per layer). A Khatri–Rao formulation in band coordinates is r² = 83 k wide, i.e. worse (`audit/REPORT.md` §4) **[docs]**; the floor of this design is therefore ≈ 36 GF at r = 288 unless the old tier's consumers are moved to lower-rank surrogates (D3_old and the Tier-2 D4 lineage sums are n-vectors; `LA_old`/`LP_old` are the only n×(ko·n) objects really needed). I did not find a measurement of how much of `Xold` is attributable to each of the two left factors.

3. **The last layer spends 11.5 GF (4.0 %) to read a diagonal.** The whole 13.6 k-column tower is transported (7.28 GF) and the old legs rebuilt (2.71 GF) at l = 15 to form `D3` on 704 rows plus the D4 lineage sums (b636:5003, 5055, 5161). D3_i = ⟨κ3, wᵢ⊗wᵢ⊗wᵢ⟩ is inherently an O(rows × k n × n) contraction in this representation, so the lever is only the row count (704 already) or dropping the oldest lineages from the final read (their D3 share is not quantified in the snapshot).

4. **Exact K2 chain: 29.2 GF (10.2 %)** — W[L]ᵀ C[L,L] W[L] with the two-level symmetric split at depth 5/6, 1.69 n³ naive per layer billed at 0.58×. Cholesky/Gram and symmetric einsum routes were priced and killed (`audit/REPORT.md` §4) **[docs]**; nothing cheaper than ≈ 0.93 u/layer appears available without changing the C representation.

5. **The riding band removed the join's re-rotation, but the join is still 7.3 GF**, of which the first join (two range-finder products + `PᵀJ`, 1.8 GF) and five `QᵀJ` + QR + solve (≈ 0.55 GF each) **[measured]**; small.

6. **Read-out products are dominated by inner dimension, not rows**: Dx at layer 9 is 736 × 9600 × 832; the newborn product only 736 × 1696 × 864. Any scheme that shortens the inner dimension of Dx (fewer non-newborn recent columns) saves ≈ 0.6 GF per 1,000 columns per layer at these shapes.

7. **Op budget is the other binding constraint**: 22,235 ops → residual 0.339 s of 0.4 s. 20 % of the ops (4,482) sit in the `_layer` body for 1 % of the FLOPs; every deeper Strassen level costs 18 ops per product (b636:1564). The planner's λ_op = 1e6 means the team values one op at 1e6 F ≈ 3.5e−6 of the bill.

8. **Everything physics-side is cheap**: closure 5.0 GF, corrector 0.34 GF, births 1.4 GF, hubs 1.1 GF — the accuracy machinery costs < 3 % of the bill; the bill is the κ3 representation.

---

## 6. OPEN QUESTIONS I could not resolve from the files

1. **The exact 8be source** is not in the snapshot; the per-site ledger for #336993 is the r1pa-at-`_SW_OP_FLOPS=1e6` proxy (transport bucket reproduced to the FLOP, the others within 0.3 %); the six "op diet" atoms (H1A, CLU, KAS, K4Q, PKK, K4B) and the SJG atom were not read in code.
2. **flopscope's billing of `linalg.solve`** (used only by the riding join, r1pa:1242) is not in the interpreter; I billed `2n³/3 + 2n²p` per call, which leaves the join group 0.34 GF above the real meter's 7.35 GF.
3. **What the R1 refit optimised and how**: the docs say 368 Adam updates with VAL-selected earliest best on TRAIN + d8b (`RESULTS.md:7420-7445`, `r1_canonical_launch_review_0736.txt`) and that 28 tables changed (diff), but the objective, learning-rate schedule and which tables were frozen are not in the two files.
4. **Derivations of the Tier-1/1.5/2 κ4 laws** (`_K4T1_*`, `_K4T15_*`, `_K4T2_*`, b636:4839-4861) and of the fold (b636:952-987): the code cites "k4 REPORT s4/s5" (`docs/research/2026-10-16_round3/k4/REPORT.md`), which I did not read in full; the coefficients are fitted per layer and their physical normalisation (why `2·C31`, `3·C211`, `t31 = 3(1−Φ)(E2 − 2E1²)`, `t211 = 2(Φ(1−Φ) − φ(αΦ+φ))`, b636:5553-5554) is asserted, not derived, in the source.
5. **Row/column role of the transport**: how much of the O-row transport beyond the K1 rows (≈ 25–30 GF) is load-bearing for the score (via D3, LA/LP and the next layer's left factors) is not measured anywhere I found.
6. **The accuracy share of the old tier at the final read** (D3_old, Tier-2 D4 old sums at l = 15, 2.7 GF rebuild) is not separated in the ledgers.
7. **Wall**: the served wall (81.4 s max for #336993) vs the planner's time model with `_SW_WALL_FLOPS = 4e8` — the calibration run for the `_TW_*` constants (b636:1523-1526) is referenced as Lane S S8.1 but not in the snapshot's docs I read.
8. **`_HUB_A` provenance**: the 5×3×32 per-layer tables feeding the corrector's first layer (b636:2590-2608) are fitted; the fitting script (`tools/build_hub.py --lean`) is referenced but its data/objective were not read.

---

## Appendix A — reproduction

```
cp docs/research/2026-10-15_exploration/audit/{census.py,fsmodel.py} <scratch>/ ; cd <scratch>
python3 -B census.py estimator_b636.py            # 305,090,053,346 F = 142.138 u, 22,217 ops
python3 -B census.py estimator_r1pa.py            # needs linalg.solve added to fsmodel (see §6.2): 291,322,458,915 F, 22,454 ops
python3 -B census_set.py estimator_r1pa.py --set _SW_OP_FLOPS=1000000.0   # 288,505,305,297 F, 23,575 ops (8be proxy before the op diet)
python3 -B mmcensus.py <file> --json mm.json      # every _mm call: layer, caller:line, m,k,n, depth, sks, pad, FLOPs, ops
```
(`census_set.py`, `mmcensus.py`, `agg.py` are 40-line wrappers I wrote in the scratchpad; the interpreter and its billing rules are the repo's, validated by the audit lane against flopscope 0.12.1 at n = 64 and against served bills at 1024×16.)

## Appendix B — largest single products, 8be proxy **[measured]**

| layer | site | shape | depth | GF |
|---|---|---|---|---|
| 6, 7 | `_advance` | 960×896×11008/11040 | 6 | 9.20, 9.27 |
| 8 | `_advance` | 928×864×10720 (pad 960×896) | 6 | 9.22 |
| 14 | `_advance` | 832×768×12192 | 6 | 7.69 |
| 15 | `_advance` | 704×736×13568 (pad 704×768) | 6 | 7.28 |
| 6, 7 | `_yp_mm_parts` Dx | 832×10080×896 (pad k 10112), 800×10176×896 | 6 | 7.41, 7.50 |
| 9 | `_yp_mm_parts` Dx | 736×9600×832 (pad m 768) | 6 | 6.06 |
| 13 | `_full_legs_lean` | 864×288×11904 | 5 | 3.31 |
| 13 | `_yp_contract` Xold | 576×11904×288 | 5 | 2.25 |
| 1 | `_shared_contract_lean` | 1024×2048×1024 | 6 | 2.11 |
| every | `_layer` X = W[L]ᵀC[L,L] | 1024×K_L×K_L | 5 | 0.84–1.10 |
| every | `_p4_sym_rest2` top half | 512×K_L×1024 | 5 | 0.52–0.58 |
