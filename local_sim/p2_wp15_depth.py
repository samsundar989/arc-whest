"""WP15 / Arm M, gate `G-M-DEPTH`: **how deep is `E21`'s memory?**

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP15 — THE MAP",
HEAD `1b6ac04`. This module scores **no gate on its own** — the contract puts
`G-M-DEPTH`'s 16 networks on `P01`'s free range as *structure only* — but the
number it produces decides whether Arm M is built or killed.

THE QUESTION
------------
The regenerating flow rebuilds `kappa_3` from its own two repeated slices every
layer and throws the all-distinct block away. `E21` is what it throws away, read
at the `(2,1)` slice after `contract_W`:

    E21_p = kappa3_uncapped(2,1) at pre-layer p  -  kappa3_regen(2,1) at pre-layer p

The all-distinct part at layer `l` is a sum of blocks **generated** at layers
`l-1, l-2, ..., 0` and propagated forward. `WP4`'s `kappa_3`-locality result says
that sum should be dominated by its youngest terms. **If it is, `E21` can be
rebuilt from a short window and Arm M is affordable; if the memory is deep or
diffuse, Arm M is dead** — that is the contract's falsifier, and this module is
what fires it.

WHAT A "WINDOW-k" FLOW IS
-------------------------
A flow that regenerates exactly like the shipped one, except that the
all-distinct blocks generated in the last `k` layers are **kept** rather than
discarded. `k = 0` is the regenerating flow; `k = depth` is the uncapped flow.
The state is carried as

    K[3] = from_dstensor(rep_total - rep_of_kept_blocks)  (+)  the kept blocks

so the repeated slices are always the true ones and the *only* thing the window
changes is how much of the all-distinct block survives. Then

    explained(k) = 1 - ||r21 - s21_k||_F^2 / ||r21 - s21_0||_F^2

is the fraction of `E21`'s Frobenius energy a `k`-deep memory recovers. It is an
`R^2`, not a norm ratio: a window that recovered a component orthogonal to `E21`
would not be credited for it.

THE FRESH BLOCK, AND HOW THIS MODULE PROVES IT HAS IT RIGHT
------------------------------------------------------------
Upstream generates the fresh all-distinct block inside `factored_nonlin_kprop_k3`
and never exposes it, so it is rebuilt here from the pre-nonlinearity state:

    fac1 = w(1)[:, None] * WK_11 ,  fac2 = 3 I ,  fac3 = (w(2)[:, None] * w(1)[None, :] * WK_11).T

with `WK_11 = zero_repeated(cov)` and `w(k) = relu_wick_coef(mean, var, k, 1)`
called from upstream, not transcribed. **A re-derivation is exactly the kind of
thing that was wrong for a month in `p2_k3_fnp` (WP16), so it is checked rather
than trusted**: at layer 0 there is no incoming `kappa_3`, so the uncapped flow's
all-distinct part IS the fresh block alone, and a window-1 flow must therefore
reproduce the **uncapped** flow's `(2,1)` slice at pre-layer 1 exactly.
`--selftest` asserts that, and refuses if it drifts.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/p2_wp15_depth.py --selftest --width 128 --depth 4
    .venv-gpu\\Scripts\\python.exe local_sim/p2_wp15_depth.py --depth-scan --nets 16 \\
        --out experiments/<slice>/gm_depth.json
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

WIDTH, DEPTH = 1024, 16

#: `CONTRACT WP15`'s composition rule, and it is not negotiable here: layers 1-3
#: are excluded because the flow has barely accumulated a `kappa_3` there and
#: would flatter any short-memory hypothesis; layer 15 is excluded because
#: nothing downstream reads its `(2,1)` slice.
GATE_LAYERS = tuple(range(4, 15))

#: `G-M-DEPTH`: ages `{l-1, l-2}` must explain `>= 70%`, layer-averaged over
#: `GATE_LAYERS`. Below 70% at ALL THREE window sizes kills Arm M.
GM_DEPTH_THRESHOLD = 0.70


def _upstream():
    import p2_kprop_bridge as bridge

    src = bridge._src()
    ver = bridge.verify_upstream(src)
    if not ver["pass"]:
        raise SystemExit(
            f"REFUSED: upstream is not rev {bridge.UPSTREAM_REV[:8]} plus exactly "
            f"the recorded patch: {ver}")
    return bridge


def fresh_block(K, n, device, dtype):
    """The all-distinct block layer `l` GENERATES, from the pre-nonlinearity state.

    `factor_k3.factored_nonlin_kprop_k3` builds it as one two-leg path diagram,
    `legs = [(1, 1, WK_11, 1)]` with `rights` the same single entry, giving
    factors `(w(1) . WK_11, 3 I, (w(2) (x) w(1) . WK_11)^T)`. In SIMPLE mode that
    single block is the whole of the fresh all-distinct part.

    The Wick coefficients come from upstream's `relu_wick_coef`, not from a
    transcription, and `zero_repeated` is upstream's too.
    """
    import torch
    from mlp_kprop.diagslice import zero_repeated
    from mlp_kprop.wick import relu_wick_coef

    mean = K[1].core if hasattr(K[1], "core") else K[1]
    cov = K[2].core if hasattr(K[2], "core") else K[2]
    var = cov.diagonal()
    w1 = relu_wick_coef(mean=mean, var=var, k=1, p=1)
    w2 = relu_wick_coef(mean=mean, var=var, k=2, p=1)
    WK_11 = zero_repeated(cov)
    fac1 = w1[:, None] * WK_11
    fac2 = torch.eye(n, device=device, dtype=dtype) * 3.0
    fac3 = (w2[:, None] * w1[None, :] * WK_11).T
    return (fac1, fac2, fac3)


def _rep_of(factors, n, device, dtype):
    """The `(3,)` and `(2,1)` repeated slices of a factored block, via upstream."""
    from mlp_kprop.factor_k3 import _factored_get_dslice

    return {(3,): _factored_get_dslice(factors, (3,)),
            (2, 1): _factored_get_dslice(factors, (2, 1))}


def _run_layers(mlp, K, lo, hi, *, regenerate, read_pre=None):
    """Run layers `lo .. hi-1` on a tower, optionally regenerating after each.

    Mirrors `p2_range_lab.run_flow`'s body for a SLICE of the network, which is
    what a delayed-regeneration flow needs: restart from a snapshot and continue.
    If `read_pre` is given, the linear step of that layer is taken and the tower
    returned at that point without running its nonlinearity.
    """
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor
    from mlp_kprop.kprop_harmonic import (
        SIMPLE, clone_tower, linear_kprop, metric_needed, nonlin_kprop,
    )
    from mlp_kprop.wick import WICK_COEF_D

    k_max = 3
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    skip_metric = not metric_needed(k_max, SIMPLE)
    for l in range(lo, hi):
        W_module = mlp.Ws[l]
        metric = mlp.init_scale[l] if l < len(mlp.nonlins) else (
            1.0 if skip_metric else None)
        K = linear_kprop(K, W_module.weight, k_max=k_max, set_metric=metric,
                         bias=W_module.bias)
        if read_pre is not None and l == read_pre:
            return K
        K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                         kind=SIMPLE, use_pK=True, factor=True)
        if regenerate and 3 in K and hasattr(K[3], "_factors"):
            K[3] = FactoredTensor.from_dstensor(K[3].get_repeated())
    return K


def regen_snapshots(mlp, K_in, depth):
    """The regenerating flow, keeping a full clone of the tower after every layer.

    `snap[l]` is the tower AFTER layer `l`'s nonlinearity and regeneration, i.e.
    exactly what layer `l+1`'s linear step consumes. `snap[-1]` is the input.
    """
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor
    from mlp_kprop.kprop_harmonic import (
        SIMPLE, clone_tower, coerce_input, linear_kprop, metric_needed,
        nonlin_kprop,
    )
    from mlp_kprop.wick import WICK_COEF_D

    k_max = 3
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    skip_metric = not metric_needed(k_max, SIMPLE)
    K = coerce_input(K_in, k_max=k_max, kind=SIMPLE)
    snap = {-1: clone_tower(K)}
    with torch.no_grad():
        for l in range(depth):
            K = linear_kprop(K, mlp.Ws[l].weight, k_max=k_max,
                             set_metric=mlp.init_scale[l],
                             bias=mlp.Ws[l].bias)
            K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                             kind=SIMPLE, use_pK=True, factor=True)
            if 3 in K and hasattr(K[3], "_factors"):
                K[3] = FactoredTensor.from_dstensor(K[3].get_repeated())
            snap[l] = clone_tower(K)
    return snap


def slices_window(mlp, snap, p, k):
    """The `(3,)`/`(2,1)` slices at pre-layer `p` of a window-`k` flow.

    A window-`k` flow is **delayed regeneration**: regenerate everywhere up to
    layer `p - k - 1`, then propagate UNCAPPED for the last `k` layers before the
    readout. `k = 0` is the regenerating flow itself; `k >= p` is the uncapped
    flow, because there was never an older generation to discard.

    This replaces an earlier "carry the fresh two-leg block" spelling that was
    WRONG and whose own ladder check said so: the regenerated base is
    `(A, I, I)`, whose all-distinct part is zero, but `contract_W` turns it into
    `(W A, W, W)`, whose all-distinct part is not. Age is therefore a property of
    WHEN THE FLOW LAST REGENERATED, not of which two-leg block a term came from.
    """
    import torch

    from mlp_kprop.kprop_harmonic import clone_tower

    start = max(p - k - 1, -1)
    K = clone_tower(snap[start])
    with torch.no_grad():
        K = _run_layers(mlp, K, start + 1, p + 1, regenerate=False, read_pre=p)
    ds = K[3].get_repeated()
    return ds.slices[(3,)].clone(), ds.slices[(2, 1)].clone()


def uncapped_slices(mlp, K_in, depth, pre_layers):
    import torch

    import p2_range_lab as rl

    out = {}
    with torch.no_grad():
        rl.run_flow(mlp, K_in, hook_lin=rl._capture_hook(out, set(pre_layers)),
                    up_to_layer=f"act{depth - 1}", output_d_max=1)
    return out


def _build(seed, width, depth, device, dtype):
    import torch

    from mint_provenance import canonical_mint

    bridge = _upstream()
    w, _e, _s = canonical_mint(width, depth, seed)
    mlp = bridge.build_mlp(w, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    return mlp, K_in


def cmd_selftest(a) -> int:
    """The window flow must BE what it claims to be, at both ends of `k`.

    S1  `k >= p` is the UNCAPPED flow at pre-layer `p`. With `k >= p` the flow
        never regenerates before the readout, so there is nothing to discard.
    S2  `k = 0` is the regenerating flow, bit-identical to `p2_range_lab`'s own
        hook -- the spelling every earlier slice was measured with.
    S3  `k = 1` genuinely differs from `k = 0`, or the window does nothing.

    S1 is the check that killed the previous spelling of this module: it carried
    the fresh two-leg block by hand and failed S1 at `k = 2` by `35%`, because
    the regenerated base picks up an all-distinct part the moment `contract_W`
    touches it.
    """
    import torch

    import p2_range_lab as rl

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    mlp, K_in = _build(a.seed, a.width, a.depth, a.device, dtype)
    pre = list(range(1, min(a.depth, 5)))

    unc = uncapped_slices(mlp, K_in, a.depth, pre)
    snap = regen_snapshots(mlp, K_in, a.depth)

    ref0 = {}

    def grab(l, K):
        if l in pre and 3 in K and hasattr(K[3], "_factors"):
            ds = K[3].get_repeated()
            ref0[l] = (ds.slices[(3,)].clone(), ds.slices[(2, 1)].clone())
        return K

    with torch.no_grad():
        rl.run_flow(mlp, K_in, hook_lin=grab, hook_nl=rl._regen_hook_nl(),
                    up_to_layer=f"act{a.depth - 1}", output_d_max=1)

    def rel(x, y):
        return float((x - y).double().norm()) / max(float(y.double().norm()), 1e-300)

    tol = 1e-10 if a.dtype == "f64" else 1e-5
    ok = {}
    print(f"  {a.width} x {a.depth}, {a.dtype}, seed {a.seed}")
    for p in pre:
        v = rel(slices_window(mlp, snap, p, p)[1], unc[p][1])
        ok[f"S1_k=p_is_uncapped_at_pre{p}"] = v
        print(f"  S1   k >= p == UNCAPPED at pre-layer {p}, (2,1)   {v:.3e}   "
              f"{'PASS' if v <= tol else 'FAIL'}")
    v0 = max(rel(slices_window(mlp, snap, p, 0)[1], ref0[p][1]) for p in pre)
    ok["S2_k0_is_the_regen_hook"] = v0
    print(f"  S2   k = 0 == p2_range_lab's regen hook          {v0:.3e}   "
          f"{'PASS' if v0 <= tol else 'FAIL'}")
    v1 = rel(slices_window(mlp, snap, pre[-1], 1)[1],
             slices_window(mlp, snap, pre[-1], 0)[1])
    ok["S3_k1_differs_from_k0"] = v1
    print(f"  S3   k = 1 DIFFERS from k = 0 at pre-layer {pre[-1]}      {v1:.3e}   "
          f"{'PASS' if v1 > tol else 'FAIL -- the window does nothing'}")
    hard = (all(v <= tol for kk, v in ok.items() if kk.startswith("S1_"))
            and v0 <= tol and v1 > tol)
    print(f"  VERDICT  {'PASS' if hard else 'FAIL'}")
    if a.out:
        _dump(a.out, {"check": "WP15 Arm M delayed-regeneration window selftest",
                      "width": a.width, "depth": a.depth, "dtype": a.dtype,
                      "seed": a.seed, "values": ok, "tol": tol,
                      "pass": bool(hard)})
    return 0 if hard else 1


def cmd_depth_scan(a) -> int:
    """`G-M-DEPTH`: the fraction of `E21` that `k` un-regenerated layers recover."""
    import torch

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    ks = [int(x) for x in a.ks.split(",")]
    pre = sorted({l + 1 for l in GATE_LAYERS if l + 1 <= a.depth - 1})
    rows = []
    for seed in a.seeds:
        t0 = time.perf_counter()
        mlp, K_in = _build(seed, a.width, a.depth, a.device, dtype)
        unc = uncapped_slices(mlp, K_in, a.depth, pre)
        snap = regen_snapshots(mlp, K_in, a.depth)
        for p in pre:
            r21 = unc[p][1].double()
            base = slices_window(mlp, snap, p, 0)[1].double()
            e_full = float((r21 - base).norm() ** 2)
            for k in ks:
                got = base if k == 0 else slices_window(mlp, snap, p, k)[1].double()
                rows.append({
                    "seed": int(seed), "pre_layer": p, "k": k,
                    "explained": 1.0 - float((r21 - got).norm() ** 2)
                    / max(e_full, 1e-300),
                    "E21_energy": e_full,
                    "rel_E21": float((r21 - base).norm()
                                     / max(float(r21.norm()), 1e-300)),
                })
        del mlp, unc, snap
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"    seed {seed}  [{time.perf_counter() - t0:.1f}s]")

    print()
    print("  G-M-DEPTH -- fraction of E21's Frobenius energy recovered by")
    print(f"  k un-regenerated layers, layer-averaged over layers "
          f"{GATE_LAYERS[0]}..{GATE_LAYERS[-1]}")
    print(f"    {'k':>3}{'mean':>10}{'min net':>10}{'max net':>10}{'  verdict':>12}")
    summary = {}
    for k in ks:
        per_net = []
        for seed in a.seeds:
            vals = [r["explained"] for r in rows
                    if r["k"] == k and r["seed"] == seed]
            if vals:
                per_net.append(float(np.mean(vals)))
        m = float(np.mean(per_net)) if per_net else float("nan")
        summary[k] = {"mean": m,
                      "min": float(np.min(per_net)) if per_net else None,
                      "max": float(np.max(per_net)) if per_net else None,
                      "per_net": per_net}
        verdict = "" if k == 0 else (" >= 70%" if m >= GM_DEPTH_THRESHOLD else " < 70%")
        print(f"    {k:>3}{m:>10.4f}{summary[k]['min']:>10.4f}"
              f"{summary[k]['max']:>10.4f}{verdict:>12}")
    two = summary.get(2, {}).get("mean", float("nan"))
    passed = bool(two >= GM_DEPTH_THRESHOLD)
    print()
    print(f"  G-M-DEPTH is written on ages {{l-1, l-2}}, i.e. k = 2: "
          f"{two:.4f} vs {GM_DEPTH_THRESHOLD:.2f}   "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed and any(summary[k]["mean"] >= GM_DEPTH_THRESHOLD
                          for k in ks if k > 2):
        print("  (a LARGER k clears it, which is not the gate the contract wrote")
        print("   and does not revive Arm M: the affordability horizon is r <= 192")
        print("   and every extra kept generation costs matmuls.)")
    print("  DESIGN-tier structure, not a promotion: the contract scores no")
    print("  outcome on these networks, and the end-to-end gate is G-M-E2E.")
    if a.out:
        _dump(a.out, {
            "check": "WP15 Arm M / G-M-DEPTH: E21 memory depth by DELAYED "
                     "REGENERATION -- regenerate everywhere except the last k "
                     "layers before the readout",
            "contract": "experiments/LOG.md 2026-09-02 CONTRACT WP15, HEAD 1b6ac04",
            "width": a.width, "depth": a.depth, "dtype": a.dtype,
            "seeds": [int(s) for s in a.seeds], "ks": ks,
            "gate_layers": list(GATE_LAYERS), "threshold": GM_DEPTH_THRESHOLD,
            "rows": rows, "summary": {str(k): v for k, v in summary.items()},
            "gate_k": 2, "pass": passed,
            "caveat": "DESIGN tier, structure only. No gate is scored on these "
                      "networks; G-M-E2E on P06 is the outcome gate.",
        })
    return 0


def _dump(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--depth-scan", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--dtype", choices=("f64", "f32"), default="f32")
    ap.add_argument("--device", default="cuda")
    # seed-ledger: block P01. The 16 networks G-M-DEPTH runs on, claimed on
    # the host by `local_sim/p2_wp15_depth.py` on 2026-09-03 and passed back
    # with `--already-claimed` when the run is brokered, because a gpuq
    # snapshot cannot write the ledger.
    ap.add_argument("--seed", type=int, default=40_821_001_176)
    ap.add_argument("--nets", type=int, default=16)
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--block", default="P01")
    ap.add_argument("--already-claimed", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print("=" * 78)
    print("WP15 / Arm M  --  G-M-DEPTH: how deep is E21's memory?")
    print("=" * 78)
    _upstream()
    if a.selftest:
        print("\n-- the window flow must BE what it claims to be -------------")
        return cmd_selftest(a)
    if a.depth_scan:
        import seed_ledger as sl

        if a.already_claimed:
            lo = a.seed
            a.seeds = [lo + i for i in range(a.nets)]
            print(f"  seeds {a.seeds[0]:,}..{a.seeds[-1]:,} (already claimed)")
        else:
            a.seeds = sl.claim(
                a.block, a.nets, "local_sim/p2_wp15_depth.py",
                ts="2026-09-03", artifact=a.out,
                note="WP15 Arm M / G-M-DEPTH: E21 memory depth by age-selected "
                     "exact injection. DESIGN tier, STRUCTURE ONLY -- the "
                     "contract scores no outcome on these networks.")
            print(f"  claimed {a.seeds[0]:,}..{a.seeds[-1]:,} from {a.block}")
        print("\n-- age-selected exact injection ------------------------------")
        return cmd_depth_scan(a)
    ap.error("one of --selftest / --depth-scan")


if __name__ == "__main__":
    raise SystemExit(main())
