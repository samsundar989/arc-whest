"""WP4 — is the K3 -> final-mean response LINEAR and ADDITIVE?

Contract: `experiments/LOG.md`, *CONTRACT WP4* (2026-08-31), committed at
`98b0973` **before any linearity measurement**. `GAME_PLAN.md` § WP4.

THE QUESTION, AND WHY IT COMES BEFORE ANY PORT
-----------------------------------------------
`GAME_PLAN.md` § 3C proposes

    delta m_L  ~=  sum_l  J_(L <- l)  S_l^(3)

— contract layer-local third-cumulant sources straight into backward-propagated
output queries, instead of transporting the recurrent `kappa_3` tower, which
W-EFF already closed on price at `1.0732 B` against a `<= 0.35 B` screen.

**That formulation is worth nothing unless the response really is linear in the
source and additive across layers.** If it is not, no amount of engineering
repairs it, so this slice measures the property and kills the lane if it fails —
before a multi-day port, not after one.

It is also the property the late-window retraction left open. That lane's
failure was attributed to `kappa_3` being *"accumulated, not local"*, and the
attribution was **WITHDRAWN** (`eb016ac`): `k3pre:m` showed the prefix explained
`89.2%` of the drift. Locality was never actually measured. This measures it.

THE INTERVENTION
----------------
One thing, and only one: **rescale `K[3]` by `eps` after the nonlinear step**.

| arm | meaning |
|---|---|
| `full` | no intervention — must be BIT-IDENTICAL to `mlp_kprop` |
| `scale:0` | every layer's `kappa_3` memory zeroed: no K3 transport |
| `scale:eps` | the response curve in the source amplitude |
| `scale_at:l:eps` | only layer `l` keeps its `kappa_3` |

`eps` is a **swept control, not a fitted parameter**. Nothing here estimates a
coefficient from the drift it is compared against; `GAME_PLAN.md` § WP4 forbids
a post-hoc rank choice and this slice fits nothing at all.

READ-ONLY
---------
The upstream checkout is a separate project and is never modified.
`p2_window.py` and `p2_pb_rank.py` are closed-lane artifacts and are not
modified either; this module has its own loop, mirroring `mlp_kprop` at rev
`93d091a4`, and `--selftest` proves the untouched arm reproduces it bit for bit.

Small shapes on CPU. Nothing here needs a GPU, so nothing goes through `gpuq`.

Run:  .venv-gpu\\Scripts\\python.exe local_sim/p2_adjoint_k3.py --selftest
      .venv-gpu\\Scripts\\python.exe local_sim/p2_adjoint_k3.py --sweep --width 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import p2_kprop_bridge as BRIDGE  # noqa: E402

#: The eps ladder for the response curve. Includes 0 and 1 as endpoints and is
#: geometric in between so a power law would be visible as a straight line on a
#: log axis. Frozen here rather than chosen after seeing a curve.
EPS_LADDER = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)


def _ensure_upstream():
    src = BRIDGE.DEFAULT_SRC
    if src not in sys.path:
        sys.path.insert(0, src)
    v = BRIDGE.verify_upstream(src)
    if not v["pass"]:
        raise RuntimeError("upstream is NOT the pinned object: %r" % (v,))
    return v


def _scale_k3(obj, eps):
    """Rescale an upstream `HTensor` in place-free fashion, preserving its type.

    Upstream's `HTensor` is `Sym(core (x) metric^(x)r)`. Scaling the core scales
    the whole tensor linearly, which is exactly the source amplitude this slice
    sweeps. The object is rebuilt rather than mutated because upstream's towers
    are shared by reference between layers.
    """
    core = getattr(obj, "core", None)
    if core is None:
        return obj * eps
    return obj.__class__(core * eps, r=getattr(obj, "r", 0),
                         metric=getattr(obj, "metric", None))


def run_arm(mlp, K_in, *, arm="full", k_max=3, output_d_max=1, trace=None):
    """`mlp_kprop(output_all=True)` with at most ONE intervention.

    Mirrors `kprop_harmonic.mlp_kprop` rev `93d091a4`. `--selftest` proves the
    `full` arm is bit-for-bit identical to it; that identity is what licenses
    reading every other arm as "upstream with one intervention" rather than as
    this module's re-derivation of upstream.
    """
    from mlp_kprop.kprop_harmonic import (
        SIMPLE, clone_tower, coerce_input, linear_kprop, nonlin_kprop,
    )
    from mlp_kprop.wick import WICK_COEF_D

    kind = SIMPLE
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    L = len(mlp.nonlins)

    eps, at_layer = None, None
    if arm.startswith("scale_at:"):
        _, l_s, e_s = arm.split(":")
        at_layer, eps = int(l_s), float(e_s)
    elif arm.startswith("scale:"):
        eps = float(arm.split(":", 1)[1])
    elif arm != "full":
        raise ValueError("unknown arm %r" % (arm,))

    K = coerce_input(K_in, k_max=k_max, kind=kind)
    out = {}
    for l, W_module in enumerate(mlp.Ws):
        if l >= L:
            break
        K = linear_kprop(K, W_module.weight, k_max=k_max,
                         set_metric=mlp.init_scale[l], bias=W_module.bias)
        out["pre%d" % l] = clone_tower(K, d_max=output_d_max)
        K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                         kind=kind, use_pK=True, factor=False)

        # ---- THE ONLY INTERVENTION -------------------------------------- #
        if eps is not None and 3 in K:
            apply_here = (at_layer is None) or (l == at_layer)
            # `scale_at:l:eps` keeps layer `l`'s source at `eps` and ZEROES
            # every other layer's, so the arm isolates ONE source rather than
            # merely emphasising it.
            factor = eps if apply_here else 0.0
            K = dict(K)
            K[3] = _scale_k3(K[3], factor)
            if trace is not None:
                core = getattr(K[3], "core", K[3])
                trace.append({"layer": l, "eps_applied": factor,
                              "k3_norm": float(core.detach().norm())})
        out["act%d" % l] = clone_tower(K, d_max=output_d_max)
    return out


def means(mlp, K_in, arm="full", k_max=3):
    L = len(mlp.nonlins)
    out = run_arm(mlp, K_in, arm=arm, k_max=k_max)
    return np.stack([BRIDGE._degree1(out["act%d" % l][1]) for l in range(L)],
                    axis=0)


def build(weights, dtype=None, device="cpu"):
    import torch
    dtype = dtype or torch.float64
    n = weights[0].shape[0]
    mlp = BRIDGE.build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(n, device=device, dtype=dtype),
            2: torch.eye(n, device=device, dtype=dtype)}
    return mlp, K_in


# --------------------------------------------------------------------------- #
# the measurements the contract names
# --------------------------------------------------------------------------- #

def response_curve(weights, ladder=EPS_LADDER, dtype=None):
    """`m(eps)` on the SCORED row, plus the linearity diagnostic.

    Linearity is judged against the straight line through `eps = 0` with the
    SMALL-EPS slope — a tangent, not a least-squares fit through the data it is
    being tested against. Fitting the line to the curve would make any curve
    look linear.
    """
    mlp, K_in = build(weights, dtype=dtype)
    m = {e: means(mlp, K_in, arm="scale:%g" % e)[-1] for e in ladder}
    m0 = m[ladder[0]]
    eps_small = ladder[1]
    slope = (m[eps_small] - m0) / eps_small
    rows = []
    for e in ladder:
        pred = m0 + slope * e
        actual = m[e]
        denom = float(np.linalg.norm(actual - m0)) or 1.0
        rows.append({
            "eps": e,
            "drift_norm": float(np.linalg.norm(actual - m0)),
            "linear_pred_norm": float(np.linalg.norm(pred - m0)),
            "abs_dev": float(np.linalg.norm(actual - pred)),
            "rel_dev": float(np.linalg.norm(actual - pred)) / denom,
        })
    worst = max(r["rel_dev"] for r in rows if r["eps"] > 0)
    return {"ladder": list(ladder), "rows": rows,
            "worst_rel_dev_from_tangent": worst,
            "tangent_from_eps": eps_small}


def per_layer_response(weights, dtype=None):
    """`m(scale_at:l:1) - m(0)` for every `l`, and how it sums."""
    mlp, K_in = build(weights, dtype=dtype)
    L = len(mlp.nonlins)
    m0 = means(mlp, K_in, arm="scale:0")[-1]
    m1 = means(mlp, K_in, arm="scale:1")[-1]
    total = m1 - m0
    deltas = []
    for l in range(L):
        ml = means(mlp, K_in, arm="scale_at:%d:1" % l)[-1]
        deltas.append(ml - m0)
    summed = np.sum(deltas, axis=0)
    tn = float(np.dot(total, total)) or 1e-300
    sq = [float(np.dot(d, d)) for d in deltas]
    tot_sq = sum(sq) or 1e-300
    cos = float(np.dot(summed, total)
                / ((np.linalg.norm(summed) or 1e-300)
                   * (np.linalg.norm(total) or 1e-300)))
    return {
        "n_layers": L,
        "total_drift_norm": float(np.linalg.norm(total)),
        "summed_response_norm": float(np.linalg.norm(summed)),
        # the share of the TRUE squared drift the additive model reproduces,
        # 1 - ||total - summed||^2 / ||total||^2
        "explained_squared_drift": 1.0 - float(np.dot(total - summed,
                                                      total - summed)) / tn,
        "cosine": cos,
        "per_layer_sq_share": [s / tot_sq for s in sq],
        "max_single_layer_share": max(sq) / tot_sq,
    }


def cost_model(n, L, query_rank=None):
    """Explicit state / FLOP formulas, so a measurement can DISAGREE.

    The full tower is `Theta(n^3)` of state per layer and `Theta(n^3 L)` at
    least to transport. An adjoint formulation carries, per layer, a query
    matrix of `q x n` and contracts each local source into it. The source itself
    is what decides everything: if it must be MATERIALISED as `n^3` to be
    contracted, the formulation has reconstructed the tower and
    `GAME_PLAN.md` § 7 kills it.
    """
    q = query_rank or n
    return {
        "width": n, "depth": L, "query_rank": q,
        "full_tower_state_per_layer": n ** 3,
        "full_tower_transport_flops": 2 * (n ** 3) * n * L,
        "adjoint_query_state": q * n * L,
        "adjoint_contraction_flops_IF_source_is_factored": 2 * q * n * n * L,
        "adjoint_contraction_flops_IF_source_is_dense": 2 * q * (n ** 3) * L,
        "verdict_rule": (
            "if the local source must be materialised dense to be contracted, "
            "the n^3 term returns and GAME_PLAN section 7 kills the lane"),
    }


# --------------------------------------------------------------------------- #

def cmd_selftest(a):
    """Q1: the untouched arm must be BIT-IDENTICAL to upstream."""
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop
    from mint_provenance import canonical_mint
    ok = True
    for n, L in ((16, 3), (32, 4)):
        w, _, _ = canonical_mint(n, L, a.seed)
        mlp, K_in = build(w)
        mine = means(mlp, K_in, arm="full")
        up = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=SIMPLE,
                       use_avg_metric=True, factor=False,
                       up_to_layer="act%d" % (L - 1), output_d_max=1)
        theirs = np.stack([BRIDGE._degree1(up["act%d" % l][1])
                           for l in range(L)], axis=0)
        same = np.array_equal(mine, theirs)
        ok &= same
        print("  n=%-4d L=%d  full arm vs mlp_kprop: %s  (max|diff| %.3e)"
              % (n, L, "BIT-IDENTICAL" if same else "DIFFERS",
                 float(np.abs(mine - theirs).max())))
    print("  %s" % ("SELFTEST PASS -- every other arm is upstream with ONE "
                    "intervention" if ok else "SELFTEST FAIL -- no arm below "
                    "may be read as upstream"))
    return 0 if ok else 1


def cmd_sweep(a):
    from mint_provenance import canonical_mint
    res = {"artifact": "p2_adjoint_k3_sweep", "artifact_version": 1,
           "width": a.width, "depth": a.depth, "networks": []}
    for i, seed in enumerate(range(a.seed, a.seed + a.networks)):
        w, _, _ = canonical_mint(a.width, a.depth, seed)
        rc = response_curve(w)
        pl = per_layer_response(w)
        res["networks"].append({"seed": seed, "response_curve": rc,
                                "per_layer": pl})
        print("  net %d (seed %d)" % (i, seed))
        print("    Q3 linearity: worst deviation from the tangent %.4f"
              % rc["worst_rel_dev_from_tangent"])
        print("    Q4 additivity: explained squared drift %.4f"
              % pl["explained_squared_drift"])
        print("    Q5 direction : cosine %.4f" % pl["cosine"])
        print("    Q6 locality  : max single-layer share %.4f"
              % pl["max_single_layer_share"])
    res["cost_model"] = cost_model(a.width, a.depth)
    agg = {
        "worst_linearity_dev": max(x["response_curve"]["worst_rel_dev_from_tangent"]
                                   for x in res["networks"]),
        "min_explained_squared_drift": min(x["per_layer"]["explained_squared_drift"]
                                           for x in res["networks"]),
        "min_cosine": min(x["per_layer"]["cosine"] for x in res["networks"]),
        "max_single_layer_share": max(x["per_layer"]["max_single_layer_share"]
                                      for x in res["networks"]),
    }
    res["aggregate"] = agg
    res["verdict"] = verdict(agg)
    print()
    print("  AGGREGATE over %d networks" % len(res["networks"]))
    for k, v in agg.items():
        print("    %-32s %.4f" % (k, v))
    print("  VERDICT: %s -- %s" % (res["verdict"]["verdict"],
                                   res["verdict"]["why"]))
    if a.emit:
        d = os.path.dirname(os.path.abspath(a.emit))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.emit, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(res, indent=2, default=str) + "\n")
        print("  emitted -> %s" % a.emit)
    return 0


def verdict(agg):
    """The contract's kill rules, applied. Chosen before any measurement."""
    lin = agg["worst_linearity_dev"]
    exp = agg["min_explained_squared_drift"]
    cos = agg["min_cosine"]
    if lin > 0.25:
        return {"verdict": "KILL", "why": "linearity deviation %.3f > 0.25; the "
                "linearisation is invalid and no engineering repairs it" % lin}
    if exp < 0.50:
        return {"verdict": "KILL", "why": "explained squared drift %.3f < 0.50; "
                "the response is not additive, so sum_l J S_l is the wrong "
                "object" % exp}
    if cos < 0.50:
        return {"verdict": "KILL", "why": "cosine %.3f < 0.50; the predicted "
                "response points the wrong way" % cos}
    if lin < 0.10 and exp > 0.70 and cos > 0.90:
        return {"verdict": "KEEP", "why": "linear, additive and correctly "
                "directed within the contract's bars"}
    return {"verdict": "CONDITIONAL", "why": "between the falsifier and the "
            "kill bar: recorded, not built on, and needing a mechanism before "
            "any port"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--networks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=90000000801)
    ap.add_argument("--emit", default="")
    a = ap.parse_args()
    v = _ensure_upstream()
    print("WP4 - the K3 -> final-mean response")
    print("  upstream rev %s  patch md5 %s  VERIFIED"
          % (v["rev"][:12], v["changed_md5"][:8]))
    if a.selftest:
        return cmd_selftest(a)
    if a.sweep:
        return cmd_sweep(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
