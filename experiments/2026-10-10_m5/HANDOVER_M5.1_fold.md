# Hand-over to Lane P — M5.1 masked star-Q31 fold (recent lineages), 2026-10-10

**Status: passes its kill gate. Ready to integrate.**

## What it is
The star κ₄ that v81d already carries for D4 (`_STAR_GAIN`, every lineage) also has a D31 = κ₄(a,a,a,b)
readout, Q★_ab = Σ_j c4_j (A_aj³ M_bj + 3 M_aj A_aj² A_bj). Its covariance term is ½ w2_a w1_b (r_a/3) Q★_ab
with r = w3/w2 = −α/σ. It folds into D21's two existing left operands — no new product:
LA += (M∘A∘A)·diag(c4)·rv, LP += (A∘A∘A)·diag(c4)·rv/3, rv_a = g★_l·(−α_a/σ_a)·1{α_a² ≤ 4}.
Only the RECENT lineages' columns are folded ("rec" mode), after the joiner's weights / query views are
read. Folding the old tier too measured worse (−0.95 % vs −1.23 % unrefit) and is not free in v81d.

## Numbers (twin v81d = v62_gb + mask+dress+bank star,starall; VAL-100 = full 900–999)
| | VAL raw | Δ |
|---|---:|---:|
| unrefit, g★ = `_STAR_GAIN` | 1.921925e-08 vs 1.945887e-08 | −1.23 % (62/100 wins) |
| DEV-100 unrefit (twin arm) | 1.897246e-08 vs 1.914226e-08 | −0.89 % |
| refit S2tsf, 1 ep, lr 1e-4, from `refit_starall2` | **1.891407e-08** | **−2.58 % vs matched control 1.941552e-08** (S2ts, same recipe) |

- **Twin parity** (built file `--raw` vs twin arm, DEV-100, unrefit g★): −4.8e-5 mean, max net 3.9e-3
  (`parity/fold0_*.json`). v81d's own arm-vs-file was 1.2e-4.
- **Interpreter** (flopmodel, alone on v81d): **+983,123,392 FLOPs (+0.254 % F, +0.46 u), +243 ops
  (25,741 → 25,984)**, matmul total unchanged.
- Marginal rule: −Δraw 2.58 % ≫ ΔF/(1+ΔF) 0.25 %. Expected score effect ≈ −2.3 %.

## Build
1. Bake the refit's S2t leaves exactly as v81d baked `refit_starall2`, from
   `C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/refit_fold_rec/params_best.npz`
   (stage S2tsf = corrector + S2t gains + sg + fg; sg moved < 0.5 %).
2. `python experiments/2026-10-10_m5/build_fold.py <baked.py> <that params_best.npz> <out.py>`
   (bakes `_FOLD_GAIN` = gain/fg; τ = 2). Anchors: `_STAR_GAIN = [`, the `D21 = (_dslice21(...` line in
   `_layer`, and `rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),` in `_shared_contract_lean`;
   each must occur once. It honours v81d's KAS row restriction `kr` and compacted lineage widths
   (`c4[adv.ko:]` are the recent lineages' `_kh`-compacted w3).
3. Ops: 25,984 alone — under the 26k gate with ~16 to spare; the v82a `block` bundle frees ~1.4k.

## Caveats
- The refit is 1 epoch; v81d's star refits kept improving with epochs — a joint refit with P's other
  items should re-learn fg together.
- The fold changes D21 for all consumers (H, d21e, births, closure); |ΔH|/|H| ≤ 1.3 %, |Δd21e| ≤ 0.4 %.
  The refit absorbs that; no variant with τ 1.5 was needed.

## Update — joint refit (fold rec + H2.5 literal, S2tsf 1 ep from starall2)
VAL 1.916160e-08: −1.31 % vs the control (1.941552e-08), −2.62 % vs H2.5 literal alone (1.967743e-08).
The fold keeps its whole gain when the always-on columns are dropped. Stacked estimate ≈ −7 % score
(raw −1.3 %, F −6.03 % + 0.25 %). Params: `refit_foldhlit/params_best.npz`.
