"""WP14 / C1, C2: is what regeneration discards inside the covariance's own range?

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP14", HEAD
`d915f83`. Seed block: `P01[40,821,001,168 .. 40,821,001,175]` -- the EIGHT
networks WP13 already claimed. **No new seeds are drawn here**, and because these
are the same networks WP13 measured `E2` on, this module's containment result and
WP13's spectra are CORRELATED evidence and must never be quoted as independent.

Research only. **Never imported by `estimator.py`**: upstream needs torch and the
grader sandbox has none.

THE QUESTION, AND WHY IT IS NOT A REOPENING
-------------------------------------------
WP13 closed the off-diagonal lane at the SELECTOR: five column-weight selectors
are `1.45x`-`2.10x` *worse than not correcting at all*. What it did not close --
and explicitly recorded as open -- is the information itself. `E21`, the `n x n`
matrix a regenerating flow discards, is low rank (`r90` `264 -> 78`) and cheap to
carry. The one subspace never tried is the one the flow **already holds**:
`Sigma`'s dominant range, whose own effective rank collapses harder
(`r90` `237 -> 26`).

So: build `Q_l = orth(Sigma_l Omega)` by a randomized range-finder
(Halko-Martinsson-Tropp), and ask what FRACTION of `E21`'s leading energy it
captures. A fraction, not a yes/no, because `SKEPTIC WP13` showed
`r90(E21)/r90(Sigma)` grows to `2.94` and total containment is therefore
impossible at depth.

WHERE THE HOOK GOES, AND WHY IT IS NOT WHERE WP13's HOOK GOES
--------------------------------------------------------------
`E21` is defined AFTER `contract_W`: regeneration is EXACT on both read-slices at
the layer it happens (WP13's positive control, `ctl_rel_slice21 ~ 6e-08`), and
the loss only appears once the next weight mixes indices, turning the discarded
all-distinct block into repeated-index mass. So a correction cannot be applied
where WP13's regeneration hook sits (after the nonlinearity); it has to be applied
**after the linear step of the next layer**, which is also exactly where the flow
holds `Sigma_pre` -- the object `Q` is built from. `run_flow` therefore takes TWO
hooks:

    hook_nl(l, K, W_next)  -- WP13's regeneration point, unchanged
    hook_lin(l, K)         -- NEW: the correction point, in pre-activation space

**S-IDEM RAN FIRST AND CHANGED THE MECHANISM**, at `128 x 4`, before a single
GPU-hour was spent. Inserting `K[3] = from_dstensor(K[3].get_repeated())` at
`hook_lin` in the UNCAPPED flow is **not** a no-op: it moves the final means by
`3.056e-03` relative. The mechanism is `factor_k3.py:381`, `pK_111 = WK[3].clone()`
-- the nonlinear step carries the INCOMING `kappa_3`'s entire factored tensor
forward, all-distinct block included, and only ever *reads* the two slices. So
rebuilding at the linear point silently applies a SECOND truncation, killing a
block that would otherwise have reached the NEXT layer's slices.

**Be exact about what that refutes.** It refutes *"a rebuild is a no-op"*. It does
NOT refute *"the mean reads `kappa_3` only through its two slices"* -- that premise
stands, and the `exact` cell demonstrates it end to end: making both slices equal
the reference's at every layer reproduces the uncapped flow's final means to
`~1e-15` absolute drift at `128 x 4`. `G-POS` is therefore unchanged from the
contract, and it is the check that this remains true at `1024 x 16`.

The correction is therefore **added, never rebuilt**. For a `(2,1)`-slice
correction `Delta` (diagonal-free) and a `(3,)`-slice correction `delta`,

    K[3] <- K[3].add_factors( (diag(delta) + 3 Delta^T, I, I) )

adds exactly `delta` to the `(3,)` slice and exactly `Delta` to the `(2,1)` slice
and nothing else to either -- the same `(M, I, I)` form `from_dstensor` itself
produces, verified term by term against `_factored_get_dslice` and asserted at run
time by **S-SLICE**. It does create an all-distinct block of its own, which then
propagates; that is a property of the deployable operation and not an error, and
measuring what it costs is part of what `C2` is for.

INDEXING -- READ THIS BEFORE COMPARING ANY NUMBER TO WP13
----------------------------------------------------------
`hook_lin` fires at **pre-activation layer `p`**. The residual it sees was
discarded by the regeneration at post-ReLU layer `p - 1`. Every row therefore
carries BOTH `pre_layer = p` and `layer = p - 1`, and `layer` is WP13's index.
WP13's layers `[1, 4, 7, 10, 13, 14]` are pre-layers `[2, 5, 8, 11, 14, 15]`.

WHAT IS AN ORACLE HERE AND WHAT IS NOT
---------------------------------------
`C2` injects the TRUE residual, taken from a reference uncapped flow run on the
same network. It is an upper bound on what any deployable coefficient map could
achieve with this basis, and it is never an operating point. `C3` -- deriving `V`
from the flow's own state -- is a separate cell with its own gate. Also note that
`Q` itself is **not free** unless `C4`'s low-rank backbone is adopted: building it
costs `(1 + 2q)` `n^2 r` matmuls per layer, which this module counts and reports
rather than waving away.

Run (GPU venv; heavy -- submit through gpuq, never launch):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_range_lab.py --selftest
    .venv-gpu\\Scripts\\python.exe local_sim/p2_range_lab.py --all
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

#: The eight networks WP13 claimed on 2026-09-02 and this contract re-uses.
#: Written as a literal ONLY because the claim already exists in the ledger and a
#: `gpuq` snapshot may not re-claim it; `--verify-ledger` asserts it is present.
#: seed-ledger: block P01
P01_SLICE_SEEDS = [40_821_001_168 + i for i in range(8)]

#: P-A's full-K3 raw MSE on 24 DESIGN mints, and its GT-noise floor.
MSE_FULL_ANCHOR = 4.0047e-08
GT_NOISE_FLOOR = 1.155e-09

#: WP13's committed drift of the regenerating arm against the uncapped flow.
#: A REGRESSION reference, never a gate: `G-E2E` is scored on the paired
#: uncorrected drift measured in the SAME run, not on this number.
REGEN_DRIFT_REF = 8.4822e-07

#: WP13's layer indices, so every table here lines up with `e2_residual.json`.
WP13_LAYERS = (1, 4, 7, 10, 13, 14)


# ==========================================================================
# the re-driven flow, with a hook at BOTH the linear and the nonlinear step
# ==========================================================================

def run_flow(mlp, K_in, *, hook_lin=None, hook_nl=None, k_max=3, factor=True,
             use_avg_metric=True, up_to_layer=None, output_d_max=1):
    """`mlp_kprop(output_all=True)` with a hook after EACH of the two steps.

    Mirrors `kprop_harmonic.mlp_kprop` rev `93d091a4` exactly, in the same shape
    as `p2_slice_lab.run_flow` -- from which this differs by exactly one thing:
    the added `hook_lin`, called after `linear_kprop` and before the tower is
    cloned into `out[f"pre{l}"]`. With both hooks `None` nothing differs, and
    `--selftest` S1 asserts that against `p2_pb_rank.predict(cap="none")`.
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
            if hook_lin is not None:
                K = hook_lin(l, K)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            break
        if l < len(mlp.nonlins):
            metric = init_scale[l] if use_avg_metric else (1.0 if skip_metric else None)
            K = linear_kprop(K, W, k_max=k_max, set_metric=metric, bias=W_module.bias)
            if hook_lin is not None:
                K = hook_lin(l, K)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                             kind=kind, use_pK=True, factor=factor)
            if hook_nl is not None:
                W_next = (mlp.Ws[l + 1].weight if l + 1 < len(mlp.Ws) else None)
                K = hook_nl(l, K, W_next)
            out[f"act{l}"] = clone_tower(K, d_max=output_d_max)
            if up_to_layer == f"act{l}":
                break
    return out


def means_of(weights, *, hook_lin=None, hook_nl=None, dtype=None, device="cuda"):
    """`(depth, width)` predicted post-ReLU means from the hooked flow."""
    import torch

    dtype = dtype or torch.float32
    depth, width = len(weights), weights[0].shape[0]
    mlp = bridge.build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    # `no_grad` is a pure memory fix -- `build_mlp` wraps weights in `nn.Linear`,
    # so the whole 16-layer flow silently retains an autograd graph otherwise.
    with torch.no_grad():
        out = run_flow(mlp, K_in, hook_lin=hook_lin, hook_nl=hook_nl,
                       up_to_layer=f"act{depth - 1}", output_d_max=1)
    return np.stack([bridge._degree1(out[f"act{l}"][1]) for l in range(depth)], axis=0)


# ==========================================================================
# the randomized range-finder -- Halko / Martinsson / Tropp, stabilised
# ==========================================================================

def range_finder(S, r: int, *, seed: int, power_iters: int = 1):
    """Orthonormal basis for the dominant `r`-dimensional range of `S`.

    `Q = orth(S Omega)`, `Omega ~ N(0, 1)` `n x r`, then `power_iters` stabilised
    power iterations. **No `eigh`**: the contract forbids it, and only the
    projector `Q Q^T` is ever used downstream, so the basis's internal rotation
    -- which a randomized method leaves arbitrary -- cannot affect any number
    reported here.

    Stabilised means re-orthonormalising between applications of `S`. The
    unstabilised form computes `S^(2q+1) Omega` in one go, and at f32 with a
    spectrum spanning the four decades this covariance actually spans, that
    collapses onto the top singular vector and silently returns a basis of rank
    far below `r`. `--selftest` S-RF measures the orthonormality that would
    otherwise be assumed.

    Cost, reported rather than hidden: `1 + 2 * power_iters` applications of `S`
    at `n^2 r` each, plus `1 + 2 * power_iters` thin QRs at `O(n r^2)`.
    """
    import torch

    n = S.shape[0]
    g = torch.Generator(device=S.device).manual_seed(int(seed) % (2 ** 63 - 1))
    Om = torch.randn((n, r), generator=g, device=S.device, dtype=S.dtype)
    Y = S @ Om
    for _ in range(power_iters):
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Y = S.T @ Q
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Y = S @ Q
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    return Q


def random_basis(n: int, r: int, *, seed: int, device, dtype):
    """N-RAND: an orthonormal basis that never touches `Sigma`.

    This is the null control, and it is code-path-distinct in the sense
    `SKEPTIC WP13` requires: it is not `range_finder` with an argument changed,
    it is a different matrix being factorised. Its expected containment of any
    fixed subspace is `r / n`, which P4 predicts and this module measures.
    """
    import torch

    g = torch.Generator(device=device).manual_seed(int(seed) % (2 ** 63 - 1))
    G = torch.randn((n, r), generator=g, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    return Q


def eff_rank(sv, frac: float) -> int:
    """Smallest `r` whose leading singular values carry `frac` of the ENERGY."""
    e = (sv.double() ** 2)
    c = np.cumsum(e.cpu().numpy())
    if c[-1] <= 0:
        return 0
    return int(np.searchsorted(c / c[-1], frac) + 1)


def containment(U, sv, Q, *, k: int) -> float:
    """Fraction of `sum_{j<k} sv_j^2` that lies in `span(Q)`, via the projector.

    With `E = U diag(sv) V^T` and `V`'s columns orthonormal,

        || Q Q^T E_k ||_F^2 = sum_{j<k} sv_j^2 || Q^T u_j ||^2

    so the whole statistic is `Q^T U`, an `r x k` product, and no `n x n` matrix
    is ever formed. `k = r90` gives the gated quantity; `k = len(sv)` gives the
    total-energy containment that `C2`'s drift prediction maps through.
    """
    Uk = U[:, :k]
    w = (sv[:k].double() ** 2)
    proj = (Q.T.double() @ Uk.double())          # (r, k)
    num = float(((proj ** 2).sum(dim=0) * w).sum())
    den = float(w.sum())
    return num / den if den > 0 else float("nan")


# ==========================================================================
# slices, and the two hooks built on them
# ==========================================================================

def _repeated(FT):
    """Upstream's own `(3,)` and `(2,1)` slices. Uses `get_repeated`, not a
    re-derivation, so no transcription of `_factored_get_dslice` can drift."""
    return FT.get_repeated()


def _capture_hook(store, layers):
    """Cache the reference flow's two read-slices at each requested pre-layer."""
    def hook(p, K):
        if p in layers and 3 in K and hasattr(K[3], "_factors"):
            ds = _repeated(K[3])
            store[p] = (ds.slices[(3,)].clone(), ds.slices[(2, 1)].clone())
        return K
    return hook


def _regen_hook_nl():
    """WP13's regeneration point, unchanged: rebuild `kappa_3` from its own
    repeated slices after the nonlinearity. `--selftest` S2 asserts this
    reproduces `p2_pb_rank.predict(cap="dsproj")` bitwise AT 1024 x 16, which is
    the shape `SKEPTIC WP13` recorded the load-bearing null had never run at."""
    from mlp_kprop.factor_k3 import FactoredTensor

    def hook(l, K, W_next):
        if 3 not in K or not hasattr(K[3], "_factors"):
            return K
        K[3] = FactoredTensor.from_dstensor(_repeated(K[3]))
        return K

    return hook


def _idem_hook_lin():
    """S-IDEM: rebuild from the repeated slices at the LINEAR point, changing
    nothing. If the nonlinear step reads only those two slices this is a no-op
    and the final means are bit-identical; if it reads more, it is not, and
    `C2`'s injection point is invalid."""
    from mlp_kprop.factor_k3 import FactoredTensor

    def hook(p, K):
        if 3 not in K or not hasattr(K[3], "_factors"):
            return K
        K[3] = FactoredTensor.from_dstensor(_repeated(K[3]))
        return K

    return hook


def _correct_hook_lin(ref, *, mode: str, r: int, seed: int, power_iters: int,
                      record=None, probe=None):
    """Inject the true residual at the linear point, in one of five ways.

    `mode`:
      `none`   nothing is injected. With `hook_nl` regenerating, this IS the
               uncorrected arm and its drift is `G-E2E`'s paired baseline.
      `e3`     the `(3,)`-slice residual EXACTLY. An `n`-vector: `O(n)` a layer,
               and therefore free at any budget.
      `proj`   `Q Q^T E21` with `Q` from the flow's OWN `Sigma_pre` -- the
               deployable-shaped correction, at oracle coefficients.
      `e3proj` both of the above.
      `svd`    the rank-`r` SVD truncation of `E21`. The best any rank-`r`
               subspace can do; an upper bound on `proj`, never an operating
               point.
      `exact`  `E21` and `E3` in full. **This is no longer a positive control
               that must reproduce the uncapped flow** -- S-IDEM refuted that
               premise. It now measures the one thing that decides this lane:
               how much of the regenerating flow's drift is carried by the two
               read-slices, and how much by the all-distinct block that no slice
               correction can reach.

    The correction is applied by `add_factors`, NOT by rebuilding from
    `from_dstensor`, because the nonlinear step propagates the incoming
    `kappa_3`'s whole factored tensor (`factor_k3.py:381`) and a rebuild would
    truncate it a second time.

    `Delta`'s diagonal is zeroed before use. `Q Q^T E21` is not diagonal-free
    even though `E21` is, and `(M, I, I)`'s `(3,)` slice is `diag(M)`, so an
    unzeroed diagonal would leak `3 * Delta_ii` into a slice this cell is not
    correcting.

    `r = 0` under `proj` leaves both slices untouched, so nothing is added at
    all and the result is bit-identical to `none`. That is a DETERMINISM CHECK
    and is labelled one -- `SKEPTIC WP13` finding (4) is that the same
    deterministic call compared with itself is not a null control.
    """
    import torch

    def hook(p, K):
        if 3 not in K or not hasattr(K[3], "_factors") or p not in ref:
            return K
        ds = _repeated(K[3])
        s3, s21 = ds.slices[(3,)], ds.slices[(2, 1)]
        r3, r21 = ref[p]
        E3 = r3 - s3
        E21 = r21 - s21

        d3 = torch.zeros_like(s3)
        d21 = torch.zeros_like(s21)
        if mode == "none":
            pass
        elif mode == "e3":
            d3 = E3
        elif mode == "exact":
            d3, d21 = E3, E21
        elif mode in ("proj", "e3proj", "svd"):
            if mode == "e3proj":
                d3 = E3
            if r > 0:
                if mode == "svd":
                    U, sv, Vh = torch.linalg.svd(E21.double(), full_matrices=False)
                    k = min(r, int(sv.shape[0]))
                    d21 = ((U[:, :k] * sv[:k]) @ Vh[:k]).to(s21.dtype)
                else:
                    Sig = K[2].core if hasattr(K[2], "core") else K[2]
                    Q = range_finder(Sig, r, seed=seed * 7919 + p * 31 + r,
                                     power_iters=power_iters)
                    d21 = Q @ (Q.T @ E21)
                    if record is not None:
                        record.append({"pre_layer": p, "layer": p - 1, "r": r,
                                       "sigma_applications": 1 + 2 * power_iters})
        else:
            raise ValueError(mode)

        if not bool(d3.any()) and not bool(d21.any()):
            return K            # r = 0 / mode none: touch nothing at all

        # The (2,1) slice is defined with repeated indices removed, and (M, I, I)
        # routes diag(M) into the (3,) slice. Zero the diagonal explicitly.
        d21 = d21 - torch.diag(torch.diagonal(d21))
        n = int(s3.shape[0])
        eye = torch.eye(n, device=s3.device, dtype=s3.dtype)
        M = torch.diag(d3) + 3.0 * d21.T
        K[3] = K[3].add_factors((M, eye, eye))

        if probe is not None:
            # S-SLICE, run in flight on the first corrected layer: the two slices
            # of the corrected object must be the ones we intended, not the ones
            # we assumed.
            ds2 = _repeated(K[3])
            probe.append({
                "pre_layer": p,
                "rel_err_s3": float((ds2.slices[(3,)] - (s3 + d3)).double().norm()
                                    / max(float((s3 + d3).double().norm()), 1e-300)),
                "rel_err_s21": float((ds2.slices[(2, 1)] - (s21 + d21)).double().norm()
                                     / max(float((s21 + d21).double().norm()), 1e-300)),
            })
        return K

    return hook


def _mint(a, seed):
    from mint_provenance import canonical_mint
    w, _e, _s = canonical_mint(a.width, a.depth, seed)
    return w


def _dump(path, doc):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


# ==========================================================================
# C1 -- the containment fraction, with two nulls and an f64 control
# ==========================================================================

def cmd_contain(a) -> int:
    """C1. Per layer, what fraction of `E21`'s r90-energy is in `span(Q_l)`?"""
    import torch

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    pre_layers = sorted({l + 1 for l in a.layers})
    cache = {}                      # (seed, pre_layer) -> (Sigma_cpu, U_cpu, sv_cpu, r90)

    for seed in a.seeds:
        w = _mint(a, seed)
        mlp = bridge.build_mlp(w, device=a.device, dtype=dtype)
        n = a.width
        K_in = {1: torch.zeros(n, device=a.device, dtype=dtype),
                2: torch.eye(n, device=a.device, dtype=dtype)}
        t0 = time.perf_counter()

        # The reference flow's slices at each pre-layer of interest.
        ref = {}
        with torch.no_grad():
            run_flow(mlp, K_in, hook_lin=_capture_hook(ref, set(pre_layers)),
                     up_to_layer=f"act{a.depth - 1}", output_d_max=1)

        # The regenerating flow's slices at the same points, plus its Sigma.
        got = {}

        def grab(p, K):
            if p in pre_layers and 3 in K and hasattr(K[3], "_factors"):
                ds = _repeated(K[3])
                Sig = K[2].core if hasattr(K[2], "core") else K[2]
                got[p] = (ds.slices[(3,)].clone(), ds.slices[(2, 1)].clone(),
                          Sig.clone())
            return K

        with torch.no_grad():
            run_flow(mlp, K_in, hook_lin=grab, hook_nl=_regen_hook_nl(),
                     up_to_layer=f"act{a.depth - 1}", output_d_max=1)

        for p in pre_layers:
            if p not in ref or p not in got:
                continue
            r3, r21 = ref[p]
            s3, s21, Sig = got[p]
            E21 = (r21 - s21).double()
            U, sv, _Vh = torch.linalg.svd(E21, full_matrices=False)
            r90 = eff_rank(sv, 0.90)
            # The FULL left basis is kept, not a truncation: `contain_full`
            # weighs every singular direction, and a `keep = max(r90, max(ranks))`
            # cut silently made that statistic index past the end of its own
            # array. At 1024 this is 4 MB per (network, layer) in f32.
            cache[(seed, p)] = {
                "Sigma": Sig.clone(),
                "U": U.clone(),
                "sv": sv.clone(),
                "r90": r90,
                "rel_E21": float(E21.norm() / r21.double().norm()),
                "rel_E3": float((r3 - s3).double().norm() / r3.double().norm()),
                "r50": eff_rank(sv, 0.50), "r99": eff_rank(sv, 0.99),
            }
        del ref, got
        torch.cuda.empty_cache() if a.device == "cuda" else None
        print(f"    seed {seed}  pre-layers {pre_layers}  "
              f"[{time.perf_counter() - t0:.1f}s]")

    # ---- containment, against Q_own, N-RAND and N-XLAYER ------------------
    rows = []
    n = a.width
    for i, seed in enumerate(a.seeds):
        other = a.seeds[(i + 1) % len(a.seeds)]
        for p in pre_layers:
            c = cache.get((seed, p))
            if c is None:
                continue
            co = cache.get((other, p))
            for r in a.ranks:
                Q = range_finder(c["Sigma"], r, seed=seed * 7919 + p * 31 + r,
                                 power_iters=a.power_iters)
                qr_err = float((Q.T.double() @ Q.double()
                                - torch.eye(r, dtype=torch.float64,
                                            device=Q.device)).abs().max())
                Qr = random_basis(n, r, seed=seed * 104_729 + p * 97 + r,
                                  device=c["Sigma"].device, dtype=c["Sigma"].dtype)
                row = {
                    "seed": seed, "pre_layer": p, "layer": p - 1, "r": r,
                    "dtype": a.dtype, "power_iters": a.power_iters,
                    "r50_E21": c["r50"], "r90_E21": c["r90"], "r99_E21": c["r99"],
                    "rel_E21": c["rel_E21"], "rel_E3": c["rel_E3"],
                    "q_orthonormality_max_err": qr_err,
                    "contain_r90": containment(c["U"], c["sv"], Q, k=c["r90"]),
                    "contain_full": containment(c["U"], c["sv"], Q,
                                                k=int(c["sv"].shape[0])),
                    "null_rand_r90": containment(c["U"], c["sv"], Qr, k=c["r90"]),
                    "null_rand_full": containment(c["U"], c["sv"], Qr,
                                                  k=int(c["sv"].shape[0])),
                    "null_expected_r_over_n": r / n,
                }
                if co is not None:
                    Qx = range_finder(co["Sigma"], r,
                                      seed=other * 7919 + p * 31 + r,
                                      power_iters=a.power_iters)
                    row["null_xnet_seed"] = other
                    row["null_xnet_r90"] = containment(c["U"], c["sv"], Qx,
                                                       k=c["r90"])
                rows.append(row)
                print(f"      seed {seed} L{p - 1:>2} r={r:>4}  "
                      f"contain {row['contain_r90']:.4f}  "
                      f"rand {row['null_rand_r90']:.4f}  "
                      f"xnet {row.get('null_xnet_r90', float('nan')):.4f}")

    _dump(a.out_c1, {
        "check": "WP14 C1: containment of E21's r90-energy in Sigma's dominant range",
        "contract": "LOG 2026-09-02 CONTRACT WP14, HEAD d915f83",
        "seed_block": "P01", "seeds": a.seeds, "dtype": a.dtype,
        "width": a.width, "depth": a.depth, "ranks": list(a.ranks),
        "layers_wp13": list(a.layers), "pre_layers": pre_layers,
        "power_iters": a.power_iters,
        "svd_dtype": "f64 after upcast",
        "indexing": "layer == WP13's index == pre_layer - 1",
        "correlated_with": "experiments/2026-09-02_wp13_slice/e2_residual.json "
                           "-- SAME EIGHT NETWORKS, not independent evidence",
        "rows": rows,
    })
    return 0


# ==========================================================================
# C2 -- inject the projected residual in flight, end to end
# ==========================================================================

def _parse_cells(specs):
    out = []
    for spec in specs:
        parts = spec.split(":")
        mode = parts[0]
        r = int(parts[1]) if len(parts) > 1 else 0
        out.append((spec, mode, r))
    return out


def cmd_e2e(a) -> int:
    """C2. End-to-end drift of the corrected regenerating flow vs the UNCAPPED
    flow. This is the first thing this project has ever run end to end with a
    rank-`r` correction applied, and it is therefore also the first direct test
    of WP10's finding that injecting truth in flight can be destructive."""
    import torch

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    cells = _parse_cells(a.cells)
    rows, cost_rows = [], []
    pre_layers = set(range(1, a.depth))

    for seed in a.seeds:
        w = _mint(a, seed)
        mlp = bridge.build_mlp(w, device=a.device, dtype=dtype)
        n = a.width
        K_in = {1: torch.zeros(n, device=a.device, dtype=dtype),
                2: torch.eye(n, device=a.device, dtype=dtype)}
        t0 = time.perf_counter()

        ref_slices = {}
        with torch.no_grad():
            out = run_flow(mlp, K_in,
                           hook_lin=_capture_hook(ref_slices, pre_layers),
                           up_to_layer=f"act{a.depth - 1}", output_d_max=1)
        base = np.stack([bridge._degree1(out[f"act{l}"][1])
                         for l in range(a.depth)], axis=0)
        del out

        per_cell = {}
        for spec, mode, r in cells:
            rec, prb = [], []
            m = means_of(w, hook_nl=_regen_hook_nl(),
                         hook_lin=_correct_hook_lin(
                             ref_slices, mode=mode, r=r, seed=seed,
                             power_iters=a.power_iters, record=rec, probe=prb),
                         dtype=dtype, device=a.device)
            drift = float(((m[-1] - base[-1]) ** 2).mean())
            per_cell[spec] = drift
            rows.append({"seed": seed, "cell": spec, "mode": mode, "r": r,
                         "dtype": a.dtype,
                         "drift_final_layer": drift,
                         "ratio_to_mse_full": drift / MSE_FULL_ANCHOR,
                         "slice_probe": prb[:2]})
            if rec:
                cost_rows.append({"seed": seed, "cell": spec,
                                  "layers_corrected": len(rec),
                                  "sigma_applications_per_layer":
                                      rec[0]["sigma_applications"]})
            print(f"      seed {seed}  {spec:<14} drift {drift:.6e}"
                  f"   ({drift / MSE_FULL_ANCHOR:7.2f} x MSE_full)")
        unc = per_cell.get("none")
        if unc:
            for spec in per_cell:
                if spec != "none":
                    print(f"        {spec:<14} {unc / per_cell[spec]:8.3f}x "
                          f"better than uncorrected")
        del ref_slices
        if a.device == "cuda":
            torch.cuda.empty_cache()
        print(f"    seed {seed}  [{time.perf_counter() - t0:.1f}s]")

    _dump(a.out_c2, {
        "check": "WP14 C2: end-to-end drift with the projected residual injected in flight",
        "contract": "LOG 2026-09-02 CONTRACT WP14, HEAD d915f83",
        "seed_block": "P01", "seeds": a.seeds, "dtype": a.dtype,
        "width": a.width, "depth": a.depth, "cells": list(a.cells),
        "power_iters": a.power_iters,
        "mse_full_anchor": MSE_FULL_ANCHOR, "gt_noise_floor": GT_NOISE_FLOOR,
        "regen_drift_reference": REGEN_DRIFT_REF,
        "baseline_cell": "none",
        "gate": "G-E2E: studentised-t lower bound of mean(drift_none - 5*drift_proj512) > 0",
        "positive_control": "G-POS, unchanged from the contract: cell 'exact' must "
                            "reproduce the UNCAPPED flow to relative final-layer MSE "
                            "<= 1e-12. A failure VOIDS C2 and C3. Note what selftest "
                            "S-IDEM did and did NOT refute: the MEAN reads kappa_3 "
                            "only through its two repeated slices, and that premise "
                            "stands; what fails is that REBUILDING from those slices "
                            "is a no-op, because factor_k3.py:381 propagates the "
                            "incoming tensor whole and a rebuild kills an "
                            "all-distinct block that would have reached the NEXT "
                            "layer's slices. C2 therefore ADDS factors rather than "
                            "rebuilding.",
        "slice_algebra_check": "S-SLICE, in flight: add_factors((diag(d3)+3 d21^T, "
                               "I, I)) moves (3,) by exactly d3 and (2,1) by exactly "
                               "d21 -- see 'slice_probe' on every row",
        "determinism_check": "cell 'proj:0' must equal cell 'none' exactly; this is "
                             "NOT a null control (SKEPTIC WP13 finding 4)",
        "rows": rows, "selector_cost": cost_rows,
    })
    return 0


# ==========================================================================
# self-tests -- the re-drive IS upstream, and the new hook point is sound
# ==========================================================================

def cmd_selftest(a) -> int:
    """S1  the two-hook re-drive with both hooks None is `predict(cap='none')`.
       S2  the regeneration hook is `predict(cap='dsproj')`, at the FULL shape.
       S-IDEM  a DIAGNOSTIC, not a gate. Rebuilding from the repeated slices at
               the linear point is NOT a no-op (measured 3.056e-03 relative at
               128 x 4), because `factor_k3.py:381` carries the incoming
               `kappa_3`'s whole factored tensor forward. Recorded with its
               number so the refuted premise stays visible; C2 ADDS factors
               instead of rebuilding, which is what S-SLICE checks.
       S-SLICE  `add_factors((diag(d3) + 3 d21^T, I, I))` moves the `(3,)` slice
               by exactly `d3` and the `(2,1)` slice by exactly `d21`, and
               NEITHER by anything else. This is the algebra C2 rests on, and it
               is executed rather than derived in a comment.
       S-ADDZERO  adding an exactly-zero factor block is a numerical no-op. NOT
               a null control -- it is the same call with a zeroed argument, and
               it is labelled a determinism check for that reason.
       S-RF  the range finder returns an orthonormal basis, and a stabilised
             power iteration is not optional at f32.
       S-CONT  containment of a matrix built INSIDE span(Q) is 1, and of one
               built in its orthogonal complement is 0.
    """
    import torch

    ok = {}
    n, d = a.selftest_width, a.selftest_depth
    seed = a.seeds[0]
    from mint_provenance import canonical_mint
    w, _e, _s = canonical_mint(n, d, seed)

    mine = means_of(w, dtype=torch.float32, device=a.device)
    theirs, _ = rank.predict(w, cap="none", dtype=torch.float32, device=a.device)
    ok["S1_redrive_bit_identical"] = bool(np.array_equal(mine, theirs))
    print(f"  S1     hookless re-drive == p2_pb_rank(cap=none)       "
          f"{'BIT-IDENTICAL' if ok['S1_redrive_bit_identical'] else 'DIFFERS'}")

    mine_ds = means_of(w, hook_nl=_regen_hook_nl(), dtype=torch.float32,
                       device=a.device)
    theirs_ds, _ = rank.predict(w, cap="dsproj", r_star=n, dtype=torch.float32,
                                device=a.device)
    ok["S2_regen_is_dsproj"] = bool(np.array_equal(mine_ds, theirs_ds))
    ok["S2_width"] = n
    ok["S2_depth"] = d
    print(f"  S2     regen hook == p2_pb_rank(cap=dsproj) at {n}x{d}   "
          f"{'BIT-IDENTICAL' if ok['S2_regen_is_dsproj'] else 'DIFFERS'}")

    mine_idem = means_of(w, hook_lin=_idem_hook_lin(), dtype=torch.float32,
                         device=a.device)
    idem_rel = float(np.sqrt(((mine_idem[-1] - mine[-1]) ** 2).mean()
                             / max((mine[-1] ** 2).mean(), 1e-300)))
    ok["S_IDEM_linear_rebuild_is_noop"] = bool(np.array_equal(mine_idem, mine))
    ok["S_IDEM_rel_divergence"] = idem_rel
    print("  S-IDEM from_dstensor at the LINEAR point -- DIAGNOSTIC, not a gate")
    print(f"         relative divergence {idem_rel:.3e}   non-zero is EXPECTED:")
    print("         factor_k3.py:381 carries the incoming kappa_3 forward whole,")
    print("         so a rebuild there truncates a second time. C2 ADDS instead.")

    # ---- S-SLICE: the algebra C2 rests on, executed in flight -------------
    g0 = torch.Generator(device=a.device).manual_seed(17)
    d3 = torch.randn(n, generator=g0, device=a.device, dtype=torch.float32)
    d21 = torch.randn((n, n), generator=g0, device=a.device, dtype=torch.float32)
    d21 = d21 - torch.diag(torch.diagonal(d21))
    slice_rows = []

    def _slice_probe(p_, K):
        if p_ != 1 or 3 not in K or not hasattr(K[3], "_factors"):
            return K
        ds = _repeated(K[3])
        b3, b21 = ds.slices[(3,)].clone(), ds.slices[(2, 1)].clone()
        eye = torch.eye(n, device=a.device, dtype=torch.float32)
        K[3] = K[3].add_factors((torch.diag(d3) + 3.0 * d21.T, eye, eye))
        ds2 = _repeated(K[3])
        slice_rows.append({
            "rel_err_s3": float((ds2.slices[(3,)] - (b3 + d3)).double().norm()
                                / max(float((b3 + d3).double().norm()), 1e-300)),
            "rel_err_s21": float((ds2.slices[(2, 1)] - (b21 + d21)).double().norm()
                                 / max(float((b21 + d21).double().norm()), 1e-300)),
        })
        return K

    means_of(w, hook_lin=_slice_probe, dtype=torch.float32, device=a.device)
    e3v = slice_rows[0]["rel_err_s3"] if slice_rows else float("nan")
    e21v = slice_rows[0]["rel_err_s21"] if slice_rows else float("nan")
    ok["S_SLICE_rel_err_s3"] = e3v
    ok["S_SLICE_rel_err_s21"] = e21v
    print(f"  S-SLICE add_factors moves (3,)  by exactly d3            {e3v:.3e}")
    print(f"  S-SLICE add_factors moves (2,1) by exactly d21           {e21v:.3e}")

    # ---- S-ADDZERO: a zero correction is a numerical no-op ----------------
    def _zero_add(p_, K):
        if 3 not in K or not hasattr(K[3], "_factors"):
            return K
        z = torch.zeros((n, n), device=a.device, dtype=torch.float32)
        eye = torch.eye(n, device=a.device, dtype=torch.float32)
        K[3] = K[3].add_factors((z, eye, eye))
        return K

    mine_zero = means_of(w, hook_lin=_zero_add, dtype=torch.float32,
                         device=a.device)
    zrel = float(np.sqrt(((mine_zero[-1] - mine[-1]) ** 2).mean()
                         / max((mine[-1] ** 2).mean(), 1e-300)))
    ok["S_ADDZERO_rel"] = zrel
    print(f"  S-ADDZERO a zero factor block is a numerical no-op       {zrel:.3e}")
    print("            (determinism check, NOT a null control)")

    # S-RF: orthonormality, and that stabilisation matters.
    g = torch.Generator(device=a.device).manual_seed(3)
    A = torch.randn((n, n), generator=g, device=a.device, dtype=torch.float32)
    S = A @ A.T                                 # symmetric PSD, wide spectrum
    for r_ in (32, min(128, n // 2)):
        Q = range_finder(S, r_, seed=11, power_iters=1)
        err = float((Q.T.double() @ Q.double()
                     - torch.eye(r_, dtype=torch.float64, device=Q.device)
                     ).abs().max())
        ok[f"S_RF_orthonormal_r{r_}"] = err
        print(f"  S-RF   Q^T Q - I max abs err at r={r_:<4}                  {err:.3e}")

    # S-CONT: the statistic is 1 inside the span and 0 in its complement.
    r_ = 64
    Q = random_basis(n, r_, seed=5, device=a.device, dtype=torch.float32)
    Bin = Q @ torch.randn((r_, n), generator=g, device=a.device,
                          dtype=torch.float32)
    U, sv, _ = torch.linalg.svd(Bin.double(), full_matrices=False)
    c_in = containment(U, sv, Q, k=r_)
    P = torch.eye(n, device=a.device, dtype=torch.float32) - Q @ Q.T
    Bout = P @ torch.randn((n, n), generator=g, device=a.device,
                           dtype=torch.float32)
    U2, sv2, _ = torch.linalg.svd(Bout.double(), full_matrices=False)
    c_out = containment(U2, sv2, Q, k=32)
    ok["S_CONT_inside_span"] = c_in
    ok["S_CONT_orthogonal_complement"] = c_out
    print(f"  S-CONT containment inside span(Q)                        {c_in:.9f}")
    print(f"  S-CONT containment in the orthogonal complement          {c_out:.3e}")

    hard = (ok["S1_redrive_bit_identical"] and ok["S2_regen_is_dsproj"]
            and ok["S_SLICE_rel_err_s3"] < 1e-5
            and ok["S_SLICE_rel_err_s21"] < 1e-5
            and ok["S_ADDZERO_rel"] < 1e-5
            # S-CONT is built from f32 matrices and its SVD is upcast, so the
            # tolerance is f32 rounding on the construction, not on the
            # statistic: 1.4e-08 is what an exact answer looks like here.
            and abs(c_in - 1.0) < 1e-6 and c_out < 1e-9)
    ok["ALL_PASS"] = bool(hard)
    print(f"\n  {'PASS' if hard else 'FAIL'} -- "
          f"{'the instrument is upstream and the hook point is sound' if hard else 'DO NOT RUN C1/C2'}")
    _dump(a.out_selftest, {"check": "WP14 selftests", "width": n, "depth": d,
                           "seed": seed, "results": ok})
    return 0 if hard else 1


def cmd_verify_ledger(a) -> int:
    import seed_ledger as sl

    claimed = set()
    for c in sl.claims():
        if c.block == "P01":
            claimed.update(range(c.start, c.start + c.n))
    missing = [s for s in a.seeds if s not in claimed]
    print(f"  P01[{a.seeds[0]:,}..{a.seeds[-1]:,}] claimed: {not missing}")
    if missing:
        print(f"  MISSING: {missing}")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--contain", action="store_true", help="C1")
    ap.add_argument("--e2e", action="store_true", help="C2")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify-ledger", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--selftest-width", type=int, default=256)
    ap.add_argument("--selftest-depth", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--power-iters", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=P01_SLICE_SEEDS)
    ap.add_argument("--layers", type=int, nargs="*", default=list(WP13_LAYERS),
                    help="WP13's post-ReLU indices; the hook fires at layer+1")
    ap.add_argument("--ranks", type=int, nargs="*",
                    default=[128, 256, 512, 768, 1024])
    ap.add_argument("--cells", nargs="*", default=[
        "none", "proj:0", "e3", "proj:128", "proj:256", "proj:512", "proj:768",
        "proj:1024", "e3proj:512", "svd:512", "exact"])
    ap.add_argument("--out-c1", default="")
    ap.add_argument("--out-c2", default="")
    ap.add_argument("--out-selftest", default="")
    a = ap.parse_args()

    # The upstream checkout is put on the path and VERIFIED to be rev 93d091a4
    # plus exactly the recorded patch, before any number is produced. A quietly
    # edited third-party reference would corrupt every row here while looking
    # completely normal.
    src = bridge._src()
    v = bridge.verify_upstream(src)
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream checkout is not rev 93d091a4 plus "
                         "exactly the recorded patch.")

    print("=" * 78)
    print("WP14 / C1 + C2  --  Sigma's dominant range as the selector")
    print("=" * 78)
    print(f"  shape {a.width} x {a.depth}   dtype {a.dtype}   device {a.device}")
    print(f"  upstream        {src}  rev 93d091a4 VERIFIED")
    print(f"  seed block P01, {len(a.seeds)} networks -- ALREADY CLAIMED by WP13, "
          "no new draw")
    print()

    rc = 0
    if a.verify_ledger or a.all:
        rc |= cmd_verify_ledger(a)
    if a.selftest or a.all:
        print("\n-- self-tests -------------------------------------------------")
        rc |= cmd_selftest(a)
        if rc:
            print("  REFUSING to run C1/C2 on a failed self-test.")
            return rc
    if a.contain or a.all:
        print("\n-- C1: containment fraction ------------------------------------")
        rc |= cmd_contain(a)
    if a.e2e or a.all:
        print("\n-- C2: oracle-in-range, end to end -----------------------------")
        rc |= cmd_e2e(a)
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
