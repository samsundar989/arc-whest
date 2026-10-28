# L4d — D21 readout columns: already restricted (one-sided, by design); no kink-column variant built

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-21_L4_D21_COLUMN_CUT.md`. Session L4, 17:12–17:30 UTC Oct 21.
Source read: A4's export `experiments/2026-10-21_L7/src/a4_export_s368.py` (sha 67ed6090). Prediction (handoff): P(columns full) ≈ 0.4.
**Answer: the columns are not full.** Lane closed on the source reading; no bill, no VAL, no pod, $0.

## 1. Where D21 is formed and on which rows × columns
Per layer, `_layer` (≈ line 3420–3460) builds the KAS index sets once (`_perm = argsort(alpha)`, ascending), then
`adv.kd = (_kri, _krv, _kci, _kcv)`:
- **rows `kr = _kri`** = the `_K1_ROWS[l]` neurons with the **smallest kink score α²** (`_K1_RANK = ('abs',)`; two-sided: decided rows
  on *either* side are not formed and get Lane K k0's gain fill `D21 += k0G·(1−k1r)[:,None]·kz[None,:]`, line 3523);
- **columns `kc = _kci = _perm[n − _KAS_K["D"][l]:]`** = the `D[l]` **largest-α** neurons — the Dr2.5 cut: **dead** columns
  (α < −2.5) are dropped, decided-**on** columns (α > 2.5) are kept.

`D21 = _dslice21(adv, off_mask)` (line 3516). In A4 `_YP_ACTIVE` is true at layer indices 2..14 (`_YP_RESTRUCTURE`, `_P2_SEPARABLE`,
not `_P3_ON`), so the products are:

| product | where | rows | right legs / columns | output |
|---|---|---|---|---|
| young tier `[LA_r | LP_r] @ [pre.A_L | pre.M]ᵀ` (non-newborn recent lineages, via the transported legs) | `_yp_contract` → `Dx = _yp_mm_parts(ls, rs)` | `LA, LP` taken at `kr` before any product | contraction over recent-lineage widths; Dx columns = the pre-layer leg rows, mapped into H by `_YP_META` (`DxH`) | K1 × (pre legs) — an internal factor, not a D21 block |
| band / old tier `Xold = [LA_old | LP_old] @ [FA | FM]ᵀ`, then `Xold @ pre.Qᵀ` | same `ls`/`rs` (`ls.append(Xold); rs.append(pre.Q)`) | `kr` (LA, LP already taken) | contraction over ko·n old columns → band rank r = `_SHARED_RANK` 288, then Q | inside Dx |
| P2 thin leg `PPL @ Zᵀ`, hubs `RY @ Yᵀ` | same parts list | `kr` | Z / Y pages | inside Dx |
| newborn + Dx readout `[LA_new + ff | LP_new + DxH + ee | Z_new/3 | Y_new] @ [newA | newM | Z_new | Y_new]ᵀ` | final `_yp_mm_parts(ls, rs, kc)` | `kr` | **right stack taken at `kc`** (`rhs = take(rhs, rows=kc)`) | **K1 × D** |
| layers 0, 1, 15 (no yp): `_shared_contract_lean(tw, LA, LP, A_L, M, kr, kc)` (+ `TT@Fᵀ`, `RY@Yᵀ`, `PPL@Zᵀ`, all with `take(…, kc)`) | `_dslice21` | `kr` | `kc` | **K1 × D** |
| placement | `_dslice21` end | gather `kd[1]` on a one-zero-row pad | gather `kd[3]` on a one-zero-column pad | n × n, zeros outside K1 × D, then `· off_mask` |

So **every D21 block is formed as K1 × D**, and the closure receives the full n×n slice with zeros placed (it does not skip pairs).
There is no D21 product over all n columns except at L1–L3, where D = K1 = n = 1024 anyway (nothing is decided there).

KAS lists in A4 (layer = 1-based; the last layer forms no D21):

| list | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R (`_KAS_K["R"]`, transport rows, top α, R2.0) | 1024 | 1024 | 992 | 960 | 928 | 896 | 896 | 864 | 832 | 832 | 800 | 768 | 768 | 768 | 736 | 736 |
| H (`_KAS_K["H"]`, hub/birth columns, top α, Hd2.5) | 1024 | 1024 | 1024 | 992 | 992 | 960 | 928 | 928 | 896 | 864 | 864 | 832 | 832 | 800 | 800 | 800 |
| **D** (`_KAS_K["D"]`, **D21 columns**, top α, Dr2.5) | 1024 | 1024 | 1024 | 992 | 992 | 960 | 928 | 928 | 896 | 864 | 864 | 832 | 832 | 800 | 800 | 800 |
| **K1** (`_K1_ROWS`, **D21 rows**, smallest α²) | 1024 | 1024 | 1024 | 992 | 928 | 896 | 832 | 800 | 768 | 736 | 704 | 640 | 608 | 576 | 576 | 544 |
| K3_A (A-side birth columns inside H, smallest α²) | 1024 | 1024 | 1024 | 992 | 928 | 896 | 832 | 800 | 768 | 736 | 704 | 640 | 608 | 576 | 576 | 544 |
| K2_O (transported leg output rows, top α) | 1024 | 1024 | 1024 | 1024 | 992 | 992 | 960 | 960 | 928 | 928 | 896 | 896 | 864 | 864 | 832 | 704 |
| CL_K (live covariance-sandwich input rows) | 1024 | 1024 | 1024 | 1024 | 1024 | 992 | 992 | 992 | 960 | 960 | 928 | 928 | 896 | 896 | 864 | — |

## 2. Why a two-sided (kink-set) column cut is not ≈ 0 here
The closure consumes the slice as `p111_21 = (w1_i² w1_j) D21_ij + …` (line ≈ 3790) and through the Wick program's `(2,1)`/`(1,2)`
slots, where `w1 = w[(1,1)] = base(1) = Φ(α)` (line 390: `k == 1 → v = Phi`). The column neuron j therefore enters at **first order
with Φ(α_j)**: ≈ 0 for a dead column (α_j < −2.5 — exactly what the Dr2.5 D cut drops) but **≈ 1 for a decided-on column**
(α_j > 2.5, a linear unit). Physically: for a linear j, E[relu(x_i)·x_j] still carries κ3(i,i,j) through i's kink; only i needs a kink.
The handoff's premise ("first order in the kink weight of *both* neurons") holds for the rows (K1 is two-sided by α², correctly) but
not for the columns. A column cut to the kink set |α_j| ≤ 2.5 would drop full-weight entries — a first-order raw error, not the
KAS floor — while the saving is only the on-saturated share of D (the dead share is already gone).

## 3. Decision (17:30 UTC)
**"Already done."** The D21 read is K1 × D at every layer; the columns are the one-sided Dr2.5 set, which is the cut the closure's
Φ(α_j) weighting permits. No variant built, no bill/VAL (the handoff's step 2 is conditional on a full-n column product, which does not
exist). Nothing for L2. Spend $0.
