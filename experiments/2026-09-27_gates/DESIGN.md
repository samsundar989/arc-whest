# State-dependent factor gates on the third-order tower — design (2026-09-27)

Prototype for `docs/research/2026-09-27_architecture_audit_codex.md` § "A distinct, cheap
candidate family" and `docs/research/2026-09-27_literature_audit_codex.md` §4B. Nothing here is
trained or submitted. Files: `tools/build_gate.py` (generator), additive stages in
`tools/twin/twin.py`, generated estimators and checks in this directory.

## 1. What is gated, in plain English

The chain carries the third cumulant of each layer's post-ReLU activations as a factored tensor.
Its *repeated-index* slices — `K3` (i,i,i) and `K21` (i,i,j) — are computed by the closure
table every layer. Its *all-distinct* part (i,j,k all different) is never computed directly; it
is made of two things whose shape is fixed by Gaussian (Wick) formulas:

* **Birth** — the new "two-leg path" block born at this layer: for a path a–m–c,
  `w1_a C11[a,m] · w2_m · w1_c C11[c,m]` (ends carry the Wick coefficient `w1`, middle carries
  `w2`).
* **Transport** — every older block, pushed through the layer's weights and then multiplied on
  each of its three modes by `w1` (`_wick_tower`).

A gate replaces the fixed vector `w1` or `w2` in one of these by `w1 * g` or `w2 * g`, where
`g_i` is a small learned function of neuron *i*'s own normalised state. It changes the
all-distinct part; by recomputing the bookkeeping (§3) it leaves `K3`/`K21` unchanged.

## 2. Source locations (base `estimator_k3_v68_wino.py`; `v62_s1` line in brackets)

| object | v68 line | (v62_s1) |
|---|---|---|
| `w1, w2 = w[(1,1)], w[(2,1)]`; `fac1 = w1[:,None] * C11` (birth end factor `A_L`) | 3982–3983 | 3930–3931 |
| `w2b = w2[None,:]`, `w2b_vec = w2` (birth middle vector; `C_L = A_L diag(w2b)`) | 3988–3989 | 3936–3937 |
| `p111_3 = w1^3 D3`, `p111_21 = w1_i^2 w1_j D21 + fac1^2 w2 · off` | 3990–3993 | 3938–3941 |
| `S3 = K3 - p111_3`, `S21 = K21 - p111_21` | 4043–4044 | 3991–3992 |
| `adv = _wick_tower(w[(1,1)][:,None], adv)` (transport Wick scale) | 4050 | 3998 |
| `e_b`, `S_sep = (e_b[:,None] * C11) * w1[None,:]` (separable leg, rides on `A_L`) | 4076–4077 | 4024–4025 |
| hubs `yA,yB,cA,cB` + `_sf` feed scale | 4082–4103 | 4030–4051 |
| hub compensation `resid -= ((y*w1)[:,None]*C11)*(c*2/3)`, `e_b -= (yA cA + yB cB) 2/3` | 4105–4111 | 4053–4059 |
| `_p2_compress(resid)` → `(Zb, Lb)`; `_append(adv, None, fac1, w2b_vec, n, Zb, Lb, S3, e_b, Y, cA, cB)` | 4126–4129 | 4072–4075 |
| `_p2_compress` definition (`_P2_RANK = 4`, line 110) | 1204–1237 | same |
| `_wick_tower`, `_dslice3`, `_dslice21`, `_p2_recon`, `_hub_parts` | 1180, 1354, 1379, 1240, 1274 | same |
| `_constants` (`"corr": _corr_arrays(dtype)}`) | 4238 | 4184 |

## 3. The math and the compensation

A born lineage (M = I at birth) represents

    T_out = diag(w1)^{⊗3} T_adv                                  inherited, Wick-scaled
          + Sym(A_L ⊗ 3I ⊗ A_L diag(w2))                         two-leg birth, A_L = diag(w1) C11
          + Sym(y ⊗ A_L diag(c) ⊗ I)  (hubs A, B)                  κ4 feed hubs
          + Sym(A_D ⊗ I ⊗ I),  A_D = diag(sb) + 3 A_L diag(eb) + Z Lᵀ    the dense-slice leg

The last block has only repeated-index entries; it exists to make the repeated slices of
`T_out` equal the table's `K3`, `K21`. With `P = p111` = the repeated slices of the first two
blocks, the code sets `sb = S3 = K3 - p111_3` and stores `S21 = K21 - p111_21` as
`S_sep + hub repeated content + resid`, with `S_sep[a,c] = e_b[a] C11[a,c] w1[c]` (which is
exactly `(A_L diag(e_b))ᵀ`), and `Z Lᵀ ≈ 3 residᵀ` (rank `_P2_RANK`).

**Birth arm** (`_GATE_ARM = "birth"`, 2 outputs): `w1e = w1 * g_e`, `w2m = w2 * g_m`.

    fac1   = w1e[:,None] * C11            (A_L: both path ends)
    w2b    = w2m                          (path middle)
    p111_21 two-leg part = (fac1 * fac1) * w2m · off           recomputed from the gated factors
    S_sep  = (e_b[:,None] * C11) * w1e[None,:]                 A_L changed, so the leg that rides on it must too
    hub compensation uses (yA * w1e), (yB * w1e)               the hubs' second leg IS A_L

All-distinct effect: path a–m–c is scaled by `g_e[a] g_m[m] g_e[c]`. The (3,) slice of the
two-leg block is 0 (C11 has a zero diagonal), so `p111_3` is unchanged. **Coupling:** because
the hubs share `A_L`, `g_e` also rescales the κ4 hubs' all-distinct content on that leg
(`g_m` does not). A clean sub-arm is `g_e ≡ 1` (middle only); it needs no S_sep / hub change.

**Transport arm** (`_GATE_ARM = "trans"`, 1 output): `w1t = w1 * g_t`.

    adv     = _wick_tower(w1t[:,None], adv)          all three modes of every inherited lineage
    p111_3  = w1t^3 * D3
    p111_21 = (w1t_i^2 w1t_j) * D21 + (two-leg part, unchanged)

`D3`, `D21` are the exact slices of the advanced tower, so after the gated scale its slices are
exactly `w1t^3 D3` and `w1t_i^2 w1t_j D21`; subtracting those keeps `K3/K21`. All-distinct
effect: `T_adv[a,b,c]` scaled by `g_t[a] g_t[b] g_t[c] w1_a w1_b w1_c`. Inherited content that
later joins the shared rank-256 tier is handled unchanged (`_wick_tower` scales `Q` too).

**Feed hubs (arm 3, mapped only).** `cA = w1·(m/8)λ + dG·w3·(m/8)`, `cB = w3·(m/8)`,
`yA = w2`, `yB = dG·w2` (4085–4096), times the feed scale `_sf` (4098–4103). A gate would
multiply `cA, cB` (static column scales) or `yA, yB` (transported leg) *after* the `_sf` block.
Compensation: the hub-compensation lines (4105–4111) already read the same `cA, cB, yA, yB`, so
applying the gate before them keeps the slices; hubs have no (3,) slice. Avoid touching the
`_sf` lines themselves — they are a twin anchor (TWIN_PATCHES feed hook).

**Gate function.** `x = clip(((α, D3/σ³, D4/σ⁴) − fmu[l]) · fisd[l], ±6)`,
`h = tanh(x W1[l] + b1[l])` (H = 8), `g = 1 + S · tanh(h W2[l] + b2[l])`, S = 0.5 (birth),
0.25 (trans). Per-layer tables; past the table (deeper nets) the gate is off. Init: W1 ~
N(0, 1/3), b1 ~ U(±0.5) from a fixed seed, **W2 = b2 = 0 ⇒ g ≡ 1.0 exactly**. Standardisers are
the closure-fit statistics of (α, γ3, γ4) (`clos_init.npz` `ufmu/ufisd`). All inputs already
exist at the site (α, σ from line 3851–3856, D3 3859, D4 3897). Deterministic, shape-generic,
imports unchanged (flopscope + stdlib), no materialisation (smoke: 0).

## 4. The rank-4 residual and the secondary error

`_p2_compress(R)` stores `3Rᵀ ≈ Z Lᵀ` with `L = Q` an orthonormal basis of a 4-dim range
(sketch = R's four largest-norm columns, one power iteration): `R ≈ Q Qᵀ R`. The dropped part
`(I − QQᵀ) R` is simply lost from the (2,1) slice, and `QQᵀR` has a non-zero diagonal, which adds
a spurious (3,) slice through `A_D`'s diagonal. So even in the base the outgoing slices equal
`K3/K21` only up to this truncation.

A gate moves content between the exact factor blocks and `R`: `R_gated = R + Δ`, where
`Δ_trans = (1 − g_i² g_j) w1_i² w1_j D21` and `Δ_birth = (1 − g_e,a² g_m,c) fac1² w2 + e_b C11
w1 (1 − g_e) + hub terms`. `Δ` is full rank (it is a Hadamard-weighted copy of `D21` or of
`C11²`), so rank 4 keeps little of it: the part of `Δ` outside the 4-dim range is dropped, i.e.
**part of the gate's change leaks into the repeated slices** instead of being compensated. The
algebra is exact; the leak is purely the compression's. Section 6 measures it.

## 5. Cost (1024 × 16)

Gated objects are O(n) vectors; `fac1`, `p111`, `S_sep` and the hub terms are the products the
base already forms, with a different vector. No matmul, factor or lineage is added.

| | per non-final layer | per MLP |
|---|---|---|
| op calls | ~20 (7 feature, 9 MLP, 2–3 apply) | +291 / +306 (≤ 26,000 easily) |
| FLOPs | ~2.7e5 | ~4e6 = **~2e-6 of 2^41** |

Measured with the real meter (`tools/_cost_worker.py`, f64 read-only, FIT net 0; worker-q
#2249, `check_cpu.json`), identity gates (they execute in full):

| file | FLOPs | multiplier | Δ multiplier | op calls | Δ ops | materialisations |
|---|---|---|---|---|---|---|
| v68_wino | 472,199,315,449 | 0.2147314 | — | 22,701 | — | 0 |
| v70g_birth | 472,203,417,465 | 0.2147333 | **+1.9e-6** | 23,007 | **+306** | 0 |
| v70g_trans | 472,202,879,721 | 0.2147330 | **+1.6e-6** | 22,992 | **+291** | 0 |

The base row reproduces RESULTS.md exactly. Server wall projection from the op delta:
+306 × 0.36 ms ≈ +0.11 s median; residual +306 × ~11 µs ≈ +3.4 ms. Peak RSS unchanged
(4.82 GiB in this meter run).

## 6. Verification

Artifacts: `check_small.json` (CPU, float64), `check_gpu.json` (worker-q #2238, FIT full rows
0–1, 63 s, peak 6.7 GiB VRAM), `check_cpu.json` (worker-q real meter), `smoke_real.py` output.
Scripts: `gate_check.py small|gpu`, `run_cpu.py`, `smoke_real.py`.

**(a) Identity parity.**
* Twin, raw (shipped v68 arithmetic), nets full 0, 1: `v70g_birth` and `v70g_trans` vs
  `v68_wino` — predictions **bit-identical** (max |Δ| = 0; final MSE 1.936003e-08, 2.010216e-08).
* Twin, hooked training path + S3 params: `v62_gb`, `v62_gt` vs `v62_s1` — **bit-identical**.
  (For reference v62_s1+S3 vs v68 raw: +0.025% / +0.13% final MSE on these two nets — the
  Winograd/Strassen rounding difference, not the gates.)
* Real flopscope meter, f64 read-only weights, 128×6 random net: both identity files
  bit-identical to v68; 0 materialisations; a perturbed gate runs finite with 0 materialisations.
* The numpy-shim `_parity.py` read at 1024×16 needs ~7 GiB (not ≤ 4 GiB) and ran > 45 min on a
  CPU-saturated queue; it was cancelled. The three reads above cover the same property.

**(b) Gradients** (twin, net full 0, loss = final + 0.1·all-layer MSE, ×1e8 as in training):

| arm / state | ‖∇W2‖ | ‖∇b2‖ | ‖∇W1‖ | ‖∇b1‖ |
|---|---|---|---|---|
| birth, identity | 2.96 | 2.44 | 0 | 0 |
| birth, W2 ~ N(0, 0.05²) | 5.11 | 4.08 | 0.50 | 0.43 |
| trans, identity | 3.85 | 2.52 | 0 | 0 |
| trans, W2 ~ N(0, 0.05²) | 19.7 | 13.5 | 1.66 | 1.17 |

All finite. W1/b1 are exactly 0 at identity by construction (they sit behind W2 = 0) and
become live as soon as W2 moves. Note the random W2 = 0.05 perturbation raised this net's
final MSE (1.94e-8 → 2.38e-8 birth, 3.05e-8 trans): the chain is sensitive to these factors,
which is both the opportunity and a divergence risk.

**(c) Repeated-slice preservation, full-rank P2 control** (`gate_check.py small`: n = 64,
depth 6, float64, `_P2_RANK = 0`, gates up to |g−1| = 0.50 birth / 0.25 trans, chain run
gated): outgoing-tower slices vs the table's K3 / K21, all layers —
max rel error **2.3e-16 / 1.1e-15 (birth)**, **6.2e-16 / 2.0e-15 (trans)**, identical to the
identity rows. The gate moves the next layer's consumed D3 by 5–26%, so the all-distinct part
really changed while the slices did not. At n = 1024 float32 with the full-rank control
(incoming rank-4 legs zero-padded, exact), layers 0–4: K21 error 3e-5–8e-5 gated and
identity alike (float32). From layer 5 on both are ~6e-3–1e-2 (K3 ≤ 3.6e-3): that is the
rank-256 shared-tier join of aged lineages, present in the base and unrelated to the gates.

**(d) The secondary error at the shipped rank 4** (net full 0, gates perturbed at random,
identity chain state, each layer run four ways from the same incoming state). Base rank-4
K21 slice error is 2–4% at every layer. "Effect" = relative change of the next layer's
consumed D3 caused by the gate (full rank = exact); "leak" = rank-4 vs full-rank error of
that same D3, gated minus identity.

| layer | birth g−1 rms | effect | leak | trans g−1 rms | effect | leak |
|---|---|---|---|---|---|---|
| 1 | 0.20 | 11.3% | +12.0% | 0.13 | 12.7% | +0.6% |
| 3 | 0.08 | 2.3% | +1.4% | 0.07 | 7.4% | +0.8% |
| 5 | 0.04 | 0.7% | +0.1% | 0.09 | 7.8% | +1.4% |
| 6 | 0.11 | 3.1% | +5.5% | 0.03 | 1.1% | 0.0% |
| 8 | 0.04 | 0.6% | +0.1% | 0.10 | 10.1% | +2.9% |
| 9 | 0.04 | 1.4% | +0.2% | 0.15 | 11.9% | +3.0% |
| 12 | 0.13 | 3.2% | +0.4% | 0.08 | 7.5% | +0.5% |
| 14 | 0.12 | 2.9% | +0.4% | 0.13 | 14.2% | +1.9% |

(all 15 layers in `check_gpu.json` → `leak`). Reading: **transport** — the leak is ~5–30% of
the intended effect (typ. ~15–25% at |g−1| ≈ 0.1); the K21 slice error rises from ~4% to
4–17%, and K3 (small in absolute terms) is hit harder (up to 42% rel at layer 14), because the
dropped content's diagonal lands on A_D's diagonal. **Birth** — per unit of gate the effect on
consumed D3 is smaller (it acts on this layer's births only) and the leak is uneven: small at
most layers, but at layers 1 and 6 as large as the effect itself (there the end gate `g_e`
pushes S_sep/hub content into the rank-4 residual). A middle-only birth gate avoids those two
terms and should leak less (not separately measured). Practical consequence: the twin trains
the real rank-4 function, so this is not a correctness bug, but the "slice-preserving" claim
holds only up to these numbers; if the learned gates are large, re-measure with the trained
parameters and consider `_P2_RANK` 8 for the gated arm (thin cost).

## 7. Training (twin) and shipping

Why the twin uses `estimator_k3_v62_s1.py` as `--cand`: the twin's gain hooks (`TWIN_PATCHES`)
anchor on *un-baked* source lines (`for row in table["terms"]:`, the bare `_sf` block, …);
`tools/build_s2.py` rewrites exactly those lines when it bakes the S2–S3 gains, so a baked file
(v67_s3, v68_wino) can only be loaded `--raw`, and `cmd_train` never uses raw. v62_s1 is v59 + S1
corrector pages; every learned value (pages, λ/θ/feed, term gains, q31) enters from
`--init S3/params_best.npz`. v62_s1 + S3 in the twin = v67_s3 = v68_wino up to Winograd
reassociation (§6 parity row). **So the gate patch must be applied to v62_s1 for training**
(`estimator_k3_v62_gb.py` / `_gt.py` here) **and to the baked file for shipping.**

Stages added to `tools/twin/twin.py` (additive): `G1b`, `G1t` = S3 leaves + `gate/{W1,b1,W2,b2}`;
`G1bg`, `G1tg` = gate leaves only. The group is `const["gate"]`, picked up like `corr/sig/clos`.

Train (one arm per run, never together):

    workerq submit --project arc-whest --ram 16 --vram 9 --cpus 4 --preemptible \
      --describe "twin G1bg: birth gates only, init S3" --eta 3h -- \
      .venv-gpu/Scripts/python.exe tools/twin/twin.py train --stage G1bg \
      --cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py \
      --init C:/Users/samsu/Documents/arc-whest/experiments/2026-09-25_twin/S3/params_best.npz \
      --out C:/Users/samsu/Documents/arc-whest/experiments/2026-09-27_gates/G1bg \
      --epochs 2 --accum 8 --lr 3e-4 --zero-scale 1.0 --eval-every 28 --patience 3

(transport: `--stage G1tg --cand .../estimator_k3_v62_gt.py --out .../G1tg`). Gate-only runs
need no training control (nothing else moves; the init HOLDOUT-100 read is the control). The
joint stages `G1b`/`G1t` at `--lr 1.5e-4` compare against S3b (S3 continued at 1.5e-4 on TRAIN
900, HOLDOUT-100 −0.17%). `--zero-scale` sets the step of the zero-initialised W2/b2
(lr × zero-scale per step); 1.0 lets |g−1| reach ~0.1 within an epoch, 0.3 (S5 precedent) is
~3× slower.

Ship: gate-only → `python tools/build_gate.py --arm birth --base estimator_k3_v68_wino.py
--params <G1bg>/params_best.npz --out <file>` (the S3 leaves are v68's). Joint →
`build_s2.py --base estimator_k3_v62_s1.py --params P --out a.py`; `build_gate.py --base a.py
--arm birth --params P --out b.py`; `build_winograd.py --base b.py --out <file>`.

## 8. Risks

1. **Rank-4 leak** (§4, §6): the gate partly changes K21 instead of only the all-distinct part.
   The twin trains through the real rank-4 arithmetic, so it optimises the actual function,
   but the "slices preserved" story is only exact at full rank.
2. **Birth `g_e` couples to the κ4 hubs** through the shared `A_L`; the middle gate `g_m` does
   not. If a result must be attributed to the two-leg block alone, pin `W2[:, :, 0] = 0`.
3. **Signal may not be in the span.** A per-neuron diagonal rescaling of fixed Wick factors can
   only reweight existing paths; if the missing all-distinct content is a different tensor, the
   gate learns ~identity (like S5u ≈ control).
4. **Standardisers are global**, from label statistics, not per layer from the chain; tanh and
   the ±6 clip keep this safe, but conditioning may be uneven across layers.
5. **Feature-path gradients**: g depends on D3, D4, which depend on earlier gates, so the
   gate is itself recurrent; bounded S limits blow-up; watch for divergence as in S5z.
6. **Shape-generic**: past 16 layers the gate is off (identity), as the other tables.
