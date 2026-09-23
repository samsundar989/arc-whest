"""P-N1 diagnostic: energy vs final-mean SENSITIVITY of the old-tier basis.

    python tools/n1_probe.py <candidate.py> --net 0 --rank 512 --out <json>
                             [--dirs 512] [--block 1]

HANDOFF 2026-09-23 Step 3.2. On ONE DEV net, run the chain with the old-tier
rank raised to `--rank` (so there is something to rank), stop right after the
layer at which `Q` is first built, and eigen-rotate that basis so each direction
has its own Gram energy:

    Sg = V diag(e) V^T      Q <- Q V,  FA <- V^T FA,  FM <- V^T FM,  Sg <- diag(e)

The rotation is exact (the old legs `Q FA`, `Q FM` are unchanged) and the next
join's range finder starts from `span(Q)`, which it does not change, so the
rotated continuation must reproduce the unrotated one to float32 rounding. That
is CHECKED, not assumed (`rotation_parity` in the output).

Then, for each direction i (or each block of `--block` directions, by energy
order), zero that direction's coefficients -- row i of `FA` and `FM`, row and
column i of `Sg` -- continue the ACTUAL remaining chain to the final layer, and
record `||delta mu_L||^2` against the unmodified continuation, plus the signed
change in final-layer MSE against the organizer labels.

Accuracy-only, cupy. Nothing here is a RESULTS.md cost figure.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools"))
import probe  # noqa: E402  (sets up sys.path: shims first, as probe.py does)

import numpy as np  # noqa: E402

CACHES = ("_legs", "_p2", "_p3", "_hub")


def _fresh(tw):
    t = copy.copy(tw)
    for c in CACHES:
        setattr(t, c, None)
    return t


def run_layers(mod, w, state, lo, hi, depth, table, dtype, const, gens):
    fnp = mod.fnp
    for l in range(lo, hi):
        state = mod._layer(w[l], state, table, dtype, const, 2.0,
                           last_layer=(l == depth - 1), gens=gens,
                           layer_index=l, depth=depth)
        mu = fnp.where(fnp.isfinite(state[0]), state[0], dtype(0.0))
        state = (mu, state[1], state[2], state[3], state[4])
    return state


def with_tower(state, tw):
    return (state[0], state[1], tw, state[3], state[4])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--net", type=int, default=0)
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--dirs", type=int, default=None, help="how many directions (default all)")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import cupy as cp
    import panels
    import real_data as rd
    cp.get_default_memory_pool().set_limit(size=20 * 2 ** 30)

    cand = Path(a.candidate)
    cand = cand if cand.is_absolute() else ROOT / cand
    mod = probe.load_module(cand)
    mod._SHARED_RANK = int(a.rank)
    fnp = mod.fnp

    split, idx, seed, name = panels.nets(a.panel)[a.net]
    w_np, tr, _ = rd.get(idx, split, dtype=np.float32)
    depth, width = w_np.shape[0], w_np.shape[1]
    truth = cp.asarray(tr[-1].astype(np.float32))
    est = mod.Estimator()
    est.setup(probe._Ctx(width, depth, probe.BUDGET, 0))
    dtype = fnp.float32
    w = [cp.asarray(np.asarray(x, dtype=np.float64)).astype(dtype) for x in w_np]
    table = mod._tables()
    const = est._constants(width, dtype)
    gens = mod._fit_generations(width, depth, probe.BUDGET)

    # ---- the reference trajectory, stopping after the first join ------------
    state = (fnp.zeros(width, dtype=dtype), fnp.eye(width, dtype=dtype), None, 0.0,
             fnp.zeros(width, dtype=dtype))
    l0 = None
    for l in range(depth):
        state = run_layers(mod, w, state, l, l + 1, depth, table, dtype, const, gens)
        tw = state[2]
        if tw is not None and getattr(tw, "ko", 0) >= 1:
            l0 = l
            break
    if l0 is None:
        raise SystemExit("no join happened")
    s0 = state
    tw0 = s0[2]
    r = tw0.Q.shape[1]
    print(f"[n1] net {a.net} ({name}): first join at layer {l0}, rank {r}, "
          f"FA {tw0.FA.shape}, FM {tw0.FM.shape}", flush=True)

    def tail(tw):
        st = run_layers(mod, w, with_tower(s0, tw), l0 + 1, depth, depth, table, dtype,
                        const, gens)
        return st[0]

    t = time.time()
    mu_ref_unrot = tail(_fresh(tw0))
    t_tail = time.time() - t

    # ---- eigen-rotate the basis so each direction has its own energy ---------
    Sg = cp.asarray(tw0.Sg, dtype=cp.float64)
    Sg = 0.5 * (Sg + Sg.T)
    e, V = cp.linalg.eigh(Sg)
    order = cp.argsort(e)[::-1]
    e, V = e[order], V[:, order]
    Vf = V.astype(dtype)
    rot = _fresh(tw0)
    rot.Q = mod._mm(tw0.Q, Vf)
    rot.FA = Vf.T @ tw0.FA
    rot.FM = Vf.T @ tw0.FM
    rot.Sg = cp.diag(e).astype(dtype)
    mu_ref = tail(_fresh(rot))
    rot_parity = float(cp.linalg.norm(mu_ref - mu_ref_unrot) / cp.linalg.norm(mu_ref_unrot))
    mse_ref = float(((mu_ref - truth) ** 2).mean())
    coef_energy = cp.asnumpy(((rot.FA.astype(cp.float64) ** 2).sum(axis=1)
                              + (rot.FM.astype(cp.float64) ** 2).sum(axis=1)))
    print(f"[n1] tail {t_tail:.2f} s; rotation parity {rot_parity:.2e}; "
          f"ref MSE {mse_ref:.4e}", flush=True)

    nd = r if a.dirs is None else min(a.dirs, r)
    B = max(1, int(a.block))
    res = {"candidate": cand.name, "net": a.net, "name": name, "first_join_layer": l0,
           "rank": r, "block": B, "tail_s": t_tail, "rotation_parity": rot_parity,
           "mse_ref": mse_ref, "gram_energy": cp.asnumpy(e).tolist(),
           "coef_energy": coef_energy.tolist(), "sens": [], "dmse": []}
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in range(0, nd, B):
        mask = cp.ones(r, dtype=dtype)
        mask[s:s + B] = 0
        t2 = _fresh(rot)
        t2.FA = rot.FA * mask[:, None]
        t2.FM = rot.FM * mask[:, None]
        t2.Sg = rot.Sg * mask[:, None] * mask[None, :]
        mu = tail(t2)
        d = mu - mu_ref
        res["sens"].append(float((d.astype(cp.float64) ** 2).sum()))
        res["dmse"].append(float(((mu - truth) ** 2).mean()) - mse_ref)
        if (s // B) % 16 == 0:
            el = time.time() - t0
            print(f"[n1] {s + B}/{nd} ({el:.0f}s) sens {res['sens'][-1]:.3e} "
                  f"dmse {res['dmse'][-1]:+.3e}", flush=True)
            out.write_text(json.dumps(res) + "\n", encoding="utf-8")
        cp.get_default_memory_pool().free_all_blocks()
    out.write_text(json.dumps(res) + "\n", encoding="utf-8")
    print(f"[n1] done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
