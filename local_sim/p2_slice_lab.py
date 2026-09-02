"""WP13 / E0-TOL, E1, E2: the two read-slices -- their tolerance, selector and rank.

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP13", HEAD
`8b46574`. Seed block: `P01[40,821,001,168 .. 40,821,001,175]`, 8 DESIGN
networks, claimed on the host before submission (`seed_ledger` refuses to write
under a `gpuq` snapshot -- the WP9 defect).

Research only. **Never imported by `estimator.py`**: upstream needs torch and the
grader sandbox has none.

THE ONE STRUCTURAL FACT EVERYTHING HERE RESTS ON
------------------------------------------------
`kappa_3` reaches the scored mean through exactly two diagonal slices. Upstream
calls `get_dslice` with no partition other than `(3,)` and `(2, 1)`, and
`FactoredTensor.from_dstensor` rebuilds the whole degree-3 tensor from those two
alone, as factors `(diag(S3) + 3 S21^T, I, I)`. A *regenerating* flow -- one that
rebuilds `kappa_3` from its own slices every layer, so the rank never exceeds
`2n` instead of growing to `32n` -- is therefore exactly the already-measured
`dsproj` cell, and its cost is E0-COST's `0.1527 B` projection.

What `dsproj` throws away is the all-distinct block, worth `8.832e-07` of drift
(`22.1 x MSE_full`). The three experiments here ask, in order, how accurate the
slices must be, whether a better selector reaches that accuracy at affordable
cost, and whether what regeneration discards is low rank in the slice metric.

WHAT IS AN ORACLE HERE AND WHAT IS NOT
--------------------------------------
`E2`'s rank-`r` correction is computed FROM THE TRUE RESIDUAL. It is an upper
bound on achievable accuracy and never an operating point: no selector a deployed
estimator could run is being proposed. WP10 measured that injecting a moment's
truth in flight can be *destructive*; if the oracle correction makes drift worse,
that is the result and it is reported, not tuned away.

Run (GPU venv; heavy -- submit through gpuq, never launch):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_slice_lab.py --selftest
    .venv-gpu\\Scripts\\python.exe local_sim/p2_slice_lab.py --all --seeds-from <json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import p2_kprop_bridge as bridge  # noqa: E402
import p2_pb_rank as rank  # noqa: E402

WIDTH, DEPTH = 1024, 16

#: The seeds this slice runs on. Claimed through `seed_ledger.claim("P01", 8, ...)`
#: on 2026-09-02, recorded in `experiments/seed_ledger.jsonl`. Written here as a
#: literal ONLY because the claim already exists in the ledger and a `gpuq`
#: snapshot may not re-claim it; `--verify-ledger` asserts the claim is present.
#: seed-ledger: block P01
P01_SLICE_SEEDS = [40_821_001_168 + i for i in range(8)]

#: P-A's measured full-K3 raw MSE on 24 DESIGN mints, with its GT-noise floor.
#: Every drift is quoted against this.
MSE_FULL_ANCHOR = 4.0047e-08
GT_NOISE_FLOOR = 1.155e-09

#: The committed `dsproj` drift. CORRECTED 2026-09-02 by `SKEPTIC WP13`: the
#: source is `experiments/2026-08-27_pb_rank/drift_dsproj.json` (rows
#: 9.355e-07 / 7.896e-07 / 9.244e-07, mean 8.8317e-07), NOT
#: `2026-08-28_pb_oracle/drift_oracle.json`, which contains no `dsproj` row at
#: all -- its cells are `oracle:n` and `oracle_offdiag:n`. f32, seeds
#: 1000201-203. Used as a REGRESSION check, not as a gate, and note it is an
#: n = 3 sample of DIFFERENT networks from the ones this module runs.
DSPROJ_DRIFT_REF = 8.8317e-07


# ==========================================================================
# the re-driven flow, with a per-layer hook
# ==========================================================================

def run_flow(mlp, K_in, *, hook=None, k_max=3, factor=True, use_avg_metric=True,
             up_to_layer=None, output_d_max=1):
    """`mlp_kprop(output_all=True)` with a hook called after each nonlinear step.

    Mirrors `kprop_harmonic.mlp_kprop` rev `93d091a4` exactly, in the same shape
    as `p2_pb_rank.run_kprop_capped` -- whose own `--selftest` proves that
    re-drive bit-identical to upstream. `--selftest` here proves THIS re-drive
    bit-identical to `p2_pb_rank.predict(cap="none")`, which chains the two.

    `hook(layer, K, W_next) -> K` may replace the tower. It is the only thing
    this function adds; with `hook=None` nothing differs.
    """
    from mlp_kprop.kprop_harmonic import (
        SIMPLE, clone_tower, coerce_input, linear_kprop, metric_needed,
        nonlin_kprop,
    )
    from mlp_kprop.wick import WICK_COEF_D

    kind = SIMPLE
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    init_scale = mlp.init_scale
    skip_metric = not metric_needed(k_max, kind)

    K = coerce_input(K_in, k_max=k_max, kind=kind)
    out = {}
    for l, W_module in enumerate(mlp.Ws):
        W = W_module.weight
        if up_to_layer == f"pre{l}" or (l == len(mlp.nonlins) and up_to_layer is None):
            K = linear_kprop(K, W, k_max=k_max, d_max=output_d_max,
                             set_metric=1.0 if skip_metric else None,
                             bias=W_module.bias)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            break
        if l < len(mlp.nonlins):
            metric = init_scale[l] if use_avg_metric else (1.0 if skip_metric else None)
            K = linear_kprop(K, W, k_max=k_max, set_metric=metric, bias=W_module.bias)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                             kind=kind, use_pK=True, factor=factor)
            if hook is not None:
                W_next = (mlp.Ws[l + 1].weight if l + 1 < len(mlp.Ws) else None)
                K = hook(l, K, W_next)
            out[f"act{l}"] = clone_tower(K, d_max=output_d_max)
            if up_to_layer == f"act{l}":
                break
    return out


def means_of(weights, *, hook=None, dtype=None, device="cuda"):
    """`(depth, width)` predicted post-ReLU means from the hooked flow."""
    import torch

    dtype = dtype or torch.float32
    depth, width = len(weights), weights[0].shape[0]
    mlp = bridge.build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    # `no_grad` is a pure memory fix -- `build_mlp` wraps weights in `nn.Linear`,
    # so the whole 16-layer flow silently retains an autograd graph otherwise.
    # `p2_pb_rank` records this reaching 75 GiB on a 32 GiB card.
    with torch.no_grad():
        out = run_flow(mlp, K_in, hook=hook, up_to_layer=f"act{depth - 1}",
                       output_d_max=1)
    return np.stack([bridge._degree1(out[f"act{l}"][1]) for l in range(depth)], axis=0)


# ==========================================================================
# slice algebra -- exact, and by subsets of columns
# ==========================================================================

def slices_of(factors, *, weights=None):
    """Exact `(3,)` and `(2, 1)` slices of `Sym(sum_r A1_r (x) A2_r (x) A3_r)`.

    This is `factor_k3._factored_get_dslice` written out for `d = 3`, so that a
    column SUBSET (and a per-column reweighting, for the importance-sampling
    arms) can be priced and evaluated without materialising anything `n^3`.

    `part = (3,)`  : one block `{1,2,3}`, coef `3!/3! = 1`, so `sum_r A1 A2 A3`.
    `part = (2, 1)`: three block perms, coef `2!1!/3! = 1/3`, diagonal zeroed.
    """
    import torch

    A1, A2, A3 = factors
    if weights is not None:
        w = weights[None, :]
        A1 = A1 * w
    s3 = (A1 * A2 * A3).sum(dim=1)
    s21 = ((A1 * A2) @ A3.T + (A1 * A3) @ A2.T + (A2 * A3) @ A1.T) / 3.0
    s21 = s21 - torch.diag(torch.diagonal(s21))
    return s3, s21


def column_weights(factors, kind: str):
    """`O(n R)` per-column weights. No adjoint, no eigendecomposition.

    `frob`  -- `||A1_r|| ||A2_r|| ||A3_r||`, the weight the whole rank-capping
               lane was selected on.
    `slice` -- the `(2, 1)` functional's OWN weight: a column enters that slice
               through Hadamard products, so its contribution is bounded by
               `||A1_r . A2_r|| ||A3_r||` summed over the three block
               permutations. This differs from `frob` exactly by how aligned the
               legs are, which is the strategist's thesis (Dharangutte-Musco:
               a diagonal-slice functional's error is controlled by off-diagonal
               mass, not Frobenius norm).
    """
    A1, A2, A3 = factors
    n1, n2, n3 = A1.norm(dim=0), A2.norm(dim=0), A3.norm(dim=0)
    if kind == "frob":
        return n1 * n2 * n3
    if kind == "slice":
        return ((A1 * A2).norm(dim=0) * n3
                + (A1 * A3).norm(dim=0) * n2
                + (A2 * A3).norm(dim=0) * n1) / 3.0
    raise ValueError(kind)


def estimate_slices(factors, k: int, *, selector: str, seed: int):
    """A `k`-column estimate of both slices. Cost is `O(n^2 k)` by construction."""
    import torch

    A1, A2, A3 = factors
    R = A1.shape[1]
    if k >= R:
        return slices_of(factors)

    g = torch.Generator(device=A1.device).manual_seed(int(seed))

    if selector in ("top_frob", "top_slice"):
        w = column_weights(factors, "frob" if selector == "top_frob" else "slice")
        keep = torch.topk(w, k, sorted=False).indices
        return slices_of(tuple(F.index_select(1, keep) for F in factors))

    if selector == "uniform":
        p = torch.full((R,), 1.0 / R, device=A1.device, dtype=A1.dtype)
    elif selector in ("dkm_frob", "dkm_slice"):
        w = column_weights(factors, "frob" if selector == "dkm_frob" else "slice")
        w = w.clamp_min(0)
        tot = w.sum()
        if float(tot) <= 0:
            p = torch.full((R,), 1.0 / R, device=A1.device, dtype=A1.dtype)
        else:
            p = w / tot
    else:
        raise ValueError(selector)

    # DKM: sample WITH replacement at `p`, rescale each draw by `1/(k p_r)`.
    # Unbiased for a sum of rank-1 terms; the whole question is its variance.
    idx = torch.multinomial(p, k, replacement=True, generator=g)
    scale = 1.0 / (k * p[idx])
    sub = tuple(F.index_select(1, idx) for F in factors)
    return slices_of(sub, weights=scale)


def _rel(a, b):
    """Relative Frobenius error of `a` against `b`, in f64."""
    a = a.double()
    b = b.double()
    den = float(b.norm())
    return float((a - b).norm()) / den if den > 0 else float("nan")


def eff_rank(sv, frac: float) -> int:
    """Smallest `r` whose leading singular values carry `frac` of the ENERGY."""
    e = (sv.double() ** 2)
    c = np.cumsum(e.cpu().numpy())
    if c[-1] <= 0:
        return 0
    return int(np.searchsorted(c / c[-1], frac) + 1)


# ==========================================================================
# E0-TOL -- the slice-error amplification law
# ==========================================================================

def _regen_hook(eps: float, mode: str, base_seed: int):
    """Regenerate `kappa_3` from its own slices each layer, optionally perturbed.

    `eps = 0` is the `dsproj` arm exactly, which is the point: this hook's null
    is a cell whose drift is already committed to the repository
    (`8.8317e-07`), so a regression check exists that no fresh measurement can
    quietly move.

    **`eps = 0` IS NOT A NULL CONTROL, and `SKEPTIC WP13` was right to say so.**
    `cmd_tol` computes `base` and the `eps = 0` row with the same deterministic
    call and the same arguments -- the `if eps != 0.0` branch skips the
    perturbation entirely -- so `excess = 0.0` cannot fail except under GPU
    nondeterminism. It is a determinism check, and it should be described as
    one. The load-bearing null is **S2** in `cmd_selftest`, which compares this
    hook against `p2_pb_rank.predict(cap="dsproj")` through an independent code
    path; S2 has only ever run at `256 x 4`, not at the `1024 x 16` shape every
    reported number was taken at. Do not quote the standing null-control debt as
    discharged on the strength of this function.

    `mode="incoherent"` multiplies both slices elementwise by `1 + eps * z`,
    `z ~ N(0, 1)` fresh per layer. `mode="coherent"` applies the SAME relative
    shift `1 + eps` to every entry at every layer -- a systematic bias, which
    WP10 measured to be the worse of the two regimes by about `sqrt(L)`.
    """
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor

    def hook(l, K, W_next):
        if 3 not in K or not hasattr(K[3], "_factors"):
            return K
        ds = K[3].get_repeated()
        if eps != 0.0:
            for part in ((3,), (2, 1)):
                if part not in ds.slices:
                    continue
                s = ds.slices[part]
                if mode == "coherent":
                    ds.slices[part] = s * (1.0 + eps)
                elif mode == "incoherent":
                    g = torch.Generator(device=s.device).manual_seed(
                        int(base_seed) * 1_000_003 + l * 97 + len(part))
                    z = torch.randn(s.shape, generator=g, device=s.device,
                                    dtype=s.dtype)
                    ds.slices[part] = s * (1.0 + eps * z)
                else:
                    raise ValueError(mode)
        K[3] = FactoredTensor.from_dstensor(ds)
        return K

    return hook


def _mint(a, seed):
    """Mint at the requested shape. `--width/--depth` exist ONLY for the
    small-shape smoke; every reported number is minted at the live 1024 x 16."""
    from mint_provenance import canonical_mint
    w, _e, _s = canonical_mint(a.width, a.depth, seed)
    return w


def cmd_tol(a) -> int:
    """E0-TOL. Excess final-layer MSE against the UNPERTURBED regenerating arm."""
    import torch

    rows = []
    for seed in a.seeds:
        w = _mint(a, seed)
        t0 = time.perf_counter()
        base = means_of(w, hook=_regen_hook(0.0, "coherent", seed),
                        dtype=torch.float32, device=a.device)
        unc = means_of(w, hook=None, dtype=torch.float32, device=a.device)
        drift_regen = float(((base[-1] - unc[-1]) ** 2).mean())
        for mode in ("coherent", "incoherent"):
            for eps in a.eps:
                m = means_of(w, hook=_regen_hook(eps, mode, seed),
                             dtype=torch.float32, device=a.device)
                excess = float(((m[-1] - base[-1]) ** 2).mean())
                rows.append({"seed": seed, "mode": mode, "eps": eps,
                             "excess_final_mse": excess,
                             "drift_regen_vs_uncapped": drift_regen})
                print(f"    seed {seed}  {mode:<11} eps {eps:9.3e}  "
                      f"excess {excess:.6e}")
        print(f"    seed {seed}  regenerating arm vs uncapped: "
              f"drift {drift_regen:.4e}  ({drift_regen / MSE_FULL_ANCHOR:.2f} x MSE_full)"
              f"   [{time.perf_counter() - t0:.1f}s]")

    # the NULL, and it must be measured null: eps = 0 twice is bit-identical.
    null_ok = all(r["excess_final_mse"] == 0.0 for r in rows if r["eps"] == 0.0)

    fits = {}
    for mode in ("coherent", "incoherent"):
        xs, ys = [], []
        for r in rows:
            if r["mode"] == mode and r["eps"] > 0 and r["excess_final_mse"] > 0:
                xs.append(np.log10(r["eps"]))
                ys.append(np.log10(r["excess_final_mse"]))
        if len(xs) >= 3:
            sl, ic = np.polyfit(xs, ys, 1)
            pred = np.polyval([sl, ic], xs)
            ss_res = float(((np.array(ys) - pred) ** 2).sum())
            ss_tot = float(((np.array(ys) - np.mean(ys)) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            # excess = A eps^slope; the tolerance for a 1e-8 excess:
            A = 10.0 ** ic
            eps_star = (1e-8 / A) ** (1.0 / sl) if sl != 0 else float("nan")
            fits[mode] = {"slope": float(sl), "log10_A": float(ic), "A": float(A),
                          "r2": float(r2), "n_points": len(xs),
                          "eps_for_1e-8_excess": float(eps_star)}
            print(f"\n  {mode:<11} excess = {A:.4e} * eps^{sl:.4f}   R^2 {r2:.5f}"
                  f"   -> eps* for 1e-8 excess = {eps_star:.4e}")

    doc = {"check": "WP13 E0-TOL: slice-error amplification law",
           "contract": "LOG 2026-09-02 CONTRACT WP13, HEAD 8b46574",
           "seed_block": "P01", "seeds": a.seeds, "dtype": "f32",
           "width": a.width, "depth": a.depth,
           "mse_full_anchor": MSE_FULL_ANCHOR, "gt_noise_floor": GT_NOISE_FLOOR,
           "dsproj_drift_reference": DSPROJ_DRIFT_REF,
           "null_eps0_bit_identical": null_ok,
           "rows": rows, "fits": fits}
    _dump(a.out_tol, doc)
    print(f"\n  NULL CONTROL  eps = 0 reproduces the unperturbed arm exactly: {null_ok}")
    return 0


# ==========================================================================
# E1 / E2 -- per-layer structure, one flow per network
# ==========================================================================

def cmd_struct(a) -> int:
    """E1 (selector at equal cost) and E2 (residual spectrum), same flow."""
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor

    e1_rows, e2_rows = [], []

    for seed in a.seeds:
        w = _mint(a, seed)
        mlp = bridge.build_mlp(w, device=a.device, dtype=torch.float32)
        n = a.width
        K_in = {1: torch.zeros(n, device=a.device, dtype=torch.float32),
                2: torch.eye(n, device=a.device, dtype=torch.float32)}
        t0 = time.perf_counter()

        def hook(l, K, W_next):
            if l not in a.layers or W_next is None or 3 not in K:
                return K
            FT = K[3]
            if not hasattr(FT, "_factors"):
                return K
            fac = FT._factors
            R = int(fac[0].shape[1])

            # ---- E2: what does regeneration discard, and is it low rank? ----
            ds = FT.get_repeated()
            FT_ds = FactoredTensor.from_dstensor(ds)
            # positive control: regeneration is EXACT on both slices HERE.
            s3_t, s21_t = slices_of(fac)
            s3_d, s21_d = slices_of(FT_ds._factors)
            ctl3, ctl21 = _rel(s3_d, s3_t), _rel(s21_d, s21_t)

            # the loss appears one layer on, after contract_W by the NEXT weight.
            nxt_t = tuple(W_next @ F for F in fac)
            nxt_d = tuple(W_next @ F for F in FT_ds._factors)
            n3_t, n21_t = slices_of(nxt_t)
            n3_d, n21_d = slices_of(nxt_d)
            E21 = (n21_t - n21_d).double()
            E3 = (n3_t - n3_d).double()
            sv = torch.linalg.svdvals(E21)
            rel21 = float(E21.norm() / n21_t.double().norm())
            rel3 = float(E3.norm() / n3_t.double().norm())
            caps = {}
            for r in a.ranks:
                if r >= min(E21.shape):
                    caps[str(r)] = 0.0
                    continue
                e = (sv.double() ** 2)
                left = float(e[r:].sum() / e.sum())
                # residual left AFTER a rank-r oracle correction, relative
                caps[str(r)] = float(np.sqrt(left) * rel21)

            # covariance's own effective rank at the same layer (P8)
            cov = K[2].core if hasattr(K[2], "core") else K[2]
            sv_cov = torch.linalg.svdvals(cov.double())

            e2_rows.append({
                "seed": seed, "layer": l, "rank_R": R,
                "ctl_rel_slice3": ctl3, "ctl_rel_slice21": ctl21,
                "rel_E3": rel3, "rel_E21": rel21,
                "r50_E21": eff_rank(sv, 0.50), "r90_E21": eff_rank(sv, 0.90),
                "r99_E21": eff_rank(sv, 0.99),
                "sv_top16_E21": [float(x) for x in sv[:16].cpu()],
                "rel_E21_after_rank": caps,
                "r50_cov": eff_rank(sv_cov, 0.50), "r90_cov": eff_rank(sv_cov, 0.90),
                "r99_cov": eff_rank(sv_cov, 0.99),
            })

            # ---- E1: selector at equal cost, on the SAME captured factors ----
            for k in a.ks:
                if k >= R:
                    continue
                for sel in ("top_frob", "top_slice", "uniform",
                            "dkm_frob", "dkm_slice"):
                    est3, est21 = estimate_slices(nxt_t, k, selector=sel,
                                                  seed=seed + 31 * l + k)
                    e1_rows.append({
                        "seed": seed, "layer": l, "rank_R": R, "k": k,
                        "selector": sel,
                        "rel_slice3": _rel(est3, n3_t),
                        "rel_slice21": _rel(est21, n21_t),
                    })
            return K

        with torch.no_grad():
            run_flow(mlp, K_in, hook=hook, up_to_layer=f"act{a.depth - 1}",
                     output_d_max=1)
        print(f"    seed {seed}  layers {a.layers}  [{time.perf_counter() - t0:.1f}s]")

    _dump(a.out_e2, {"check": "WP13 E2: residual spectrum after regeneration",
                     "contract": "LOG 2026-09-02 CONTRACT WP13, HEAD 8b46574",
                     "seed_block": "P01", "seeds": a.seeds, "dtype": "f32",
                     "width": a.width, "depth": a.depth, "svd_dtype": "f64 after upcast",
                     "layers": a.layers, "ranks": a.ranks, "rows": e2_rows})
    _dump(a.out_e1, {"check": "WP13 E1: slice-targeted vs Frobenius selector",
                     "contract": "LOG 2026-09-02 CONTRACT WP13, HEAD 8b46574",
                     "seed_block": "P01", "seeds": a.seeds, "dtype": "f32",
                     "width": a.width, "depth": a.depth,
                     "layers": a.layers, "ks": a.ks, "rows": e1_rows})
    return 0


# ==========================================================================
# self-tests -- the re-drive IS upstream, and the null IS null
# ==========================================================================

def cmd_selftest(a) -> int:
    """S1 the hookless re-drive is `p2_pb_rank.predict(cap='none')`, bitwise.
       S2 the regenerating hook at `eps = 0` is `p2_pb_rank.predict(cap='dsproj')`.
       S3 `slices_of` reproduces upstream `_factored_get_dslice` to f64 rounding.
       S4 `estimate_slices` at `k >= R` is EXACT for every selector.
    """
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor, _factored_get_dslice

    ok = {}
    n = a.selftest_width
    seeds = a.seeds[:1]
    from mint_provenance import canonical_mint
    w, _e, _s = canonical_mint(n, 4, seeds[0])

    mine = means_of(w, hook=None, dtype=torch.float32, device=a.device)
    theirs, _ = rank.predict(w, cap="none", dtype=torch.float32, device=a.device)
    ok["S1_redrive_bit_identical"] = bool(np.array_equal(mine, theirs))
    print(f"  S1  hookless re-drive == p2_pb_rank(cap=none)   "
          f"{'BIT-IDENTICAL' if ok['S1_redrive_bit_identical'] else 'DIFFERS'}")

    mine_ds = means_of(w, hook=_regen_hook(0.0, "coherent", seeds[0]),
                       dtype=torch.float32, device=a.device)
    theirs_ds, _ = rank.predict(w, cap="dsproj", r_star=n, dtype=torch.float32,
                                device=a.device)
    ok["S2_regen_is_dsproj"] = bool(np.array_equal(mine_ds, theirs_ds))
    print(f"  S2  regen hook (eps=0) == p2_pb_rank(cap=dsproj)  "
          f"{'BIT-IDENTICAL' if ok['S2_regen_is_dsproj'] else 'DIFFERS'}")

    g = torch.Generator(device=a.device).manual_seed(7)
    fac = tuple(torch.randn((n, 3 * n), generator=g, device=a.device,
                            dtype=torch.float64) for _ in range(3))
    s3, s21 = slices_of(fac)
    u3 = _factored_get_dslice(fac, (3,))
    u21 = _factored_get_dslice(fac, (2, 1))
    ok["S3_rel_slice3"] = _rel(s3, u3)
    ok["S3_rel_slice21"] = _rel(s21, u21)
    ok["S3_pass"] = bool(max(ok["S3_rel_slice3"], ok["S3_rel_slice21"]) < 1e-12)
    print(f"  S3  slices_of vs upstream _factored_get_dslice   "
          f"(3,) {ok['S3_rel_slice3']:.3e}   (2,1) {ok['S3_rel_slice21']:.3e}   "
          f"{'PASS' if ok['S3_pass'] else 'FAIL'}")

    worst = 0.0
    for sel in ("top_frob", "top_slice", "uniform", "dkm_frob", "dkm_slice"):
        e3, e21 = estimate_slices(fac, 3 * n, selector=sel, seed=1)
        worst = max(worst, _rel(e3, s3), _rel(e21, s21))
    ok["S4_full_k_exact_worst"] = worst
    ok["S4_pass"] = bool(worst < 1e-12)
    print(f"  S4  estimate_slices at k >= R is exact, all selectors   "
          f"worst {worst:.3e}   {'PASS' if ok['S4_pass'] else 'FAIL'}")

    allp = (ok["S1_redrive_bit_identical"] and ok["S2_regen_is_dsproj"]
            and ok["S3_pass"] and ok["S4_pass"])
    ok["pass"] = bool(allp)
    _dump(a.out_selftest, {"check": "WP13 self-tests", "width": n,
                           "seed": seeds[0], "results": ok})
    print(f"\n  SELF-TEST {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


def cmd_verify_ledger(a) -> int:
    """The seeds this module runs on are CLAIMED, on the live ledger."""
    import seed_ledger as sl

    cs = [c for c in sl.claims() if c.block == "P01"
          and c.start <= P01_SLICE_SEEDS[0] <= c.end]
    ok = bool(cs) and (cs[-1].end >= P01_SLICE_SEEDS[-1])
    print(f"  P01[{P01_SLICE_SEEDS[0]:,}..{P01_SLICE_SEEDS[-1]:,}] claimed: {ok}")
    if cs:
        print(f"    by {cs[-1].consumer!r} on {cs[-1].ts}")
    return 0 if ok else 1


def _dump(path, doc):
    if not path:
        return
    p = os.path.abspath(os.path.join(_ROOT, path))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  wrote {os.path.relpath(p, _ROOT)}")


def main() -> int:
    d = "experiments/2026-09-02_wp13_slice"
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-ledger", action="store_true")
    ap.add_argument("--tol", action="store_true")
    ap.add_argument("--struct", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest-width", type=int, default=256)
    ap.add_argument("--width", type=int, default=WIDTH,
                    help="SMOKE ONLY below 1024; every reported number is 1024.")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--tol-seeds", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*",
                    default=[1, 4, 7, 10, 13, 14])
    ap.add_argument("--ks", type=int, nargs="*",
                    default=[128, 256, 330, 512, 1024, 2048])
    ap.add_argument("--ranks", type=int, nargs="*",
                    default=[8, 16, 32, 64, 128, 256, 330, 512])
    ap.add_argument("--eps", type=float, nargs="*",
                    default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--out-tol", default=f"{d}/e0_tol.json")
    ap.add_argument("--out-e1", default=f"{d}/e1_selector.json")
    ap.add_argument("--out-e2", default=f"{d}/e2_residual.json")
    ap.add_argument("--out-selftest", default=f"{d}/selftest.json")
    a = ap.parse_args()

    src = bridge._src()
    v = bridge.verify_upstream(src)
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream checkout is not rev 93d091a4 plus "
                         "exactly the recorded patch.")

    print("=" * 78)
    print("WP13 SLICE LAB  --  E0-TOL / E1 / E2 on the two read-slices")
    print("=" * 78)
    print(f"  upstream {v['rev'][:12]}  patch IDENTICAL")
    print(f"  seed block P01, {a.n_seeds} of 8 claimed DESIGN networks")
    print()

    a.seeds = P01_SLICE_SEEDS[:a.n_seeds]
    rc = 0
    if a.verify_ledger or a.all:
        rc |= cmd_verify_ledger(a)
    if a.selftest or a.all:
        print("\n  SELF-TESTS")
        rc |= cmd_selftest(a)
    if a.tol or a.all:
        print("\n  E0-TOL  slice-error amplification")
        full = a.seeds
        a.seeds = full[:a.tol_seeds]
        rc |= cmd_tol(a)
        a.seeds = full
    if a.struct or a.all:
        print("\n  E1 / E2  selector and residual spectrum")
        rc |= cmd_struct(a)
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
