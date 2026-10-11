"""Lane M7, step 1 -- v77's own state at layers 1-6 (post-ReLU mean and covariance). `uv run`.

Runs `estimator_k3_v77_kash._layer` exactly as `Estimator._tower` does (same constants,
same generation count for a 1024 x 16 net at budget 2**41), but stops after L_MAX layers
and keeps the state. Real flopscope (CPU); FLOPs are irrelevant here, only values.

Saves per net and layer: v77 mean, per-neuron variance, and the 5 probe directions
U = [m̂, u1..u4] (m̂ = v77 mean / ‖mean‖, u = top-4 eigenvectors of v77's covariance) with
v77's own uᵀCu, so the MC (m7_mc.py) can label exactly those projections.

    uv run python experiments/2026-10-07_laneM2/m7_state.py --nets 0:8 --out <abs.npz>
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = Path("C:/Users/samsu/Documents/arc-whest")
sys.path.append(str(ROOT / "tools" / "fresh"))
import gen  # noqa: E402

L_MAX = 6
BUDGET = 2 ** 41


def load_v77():
    spec = importlib.util.spec_from_file_location("_v77", MAIN / "estimator_k3_v77_kash.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v77"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="0:8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mod = load_v77()
    import flopscope.numpy as fnp
    mini = np.load(gen.cache_dir() / "mini16_truth.npz")
    lo, hi = (int(t) for t in a.nets.split(":"))
    width, depth = gen.WIDTH, gen.DEPTH
    est = mod.Estimator()
    est.setup(None)
    out = {"seeds": [], "mu": [], "var": [], "U": [], "uCu": [], "trC": []}
    for i in range(lo, hi):
        seed = int(mini["seeds"][i])
        w = [fnp.asarray(x) for x in gen.mint_numpy(seed, width, depth)]
        dtype = fnp.float32
        const = est._constants(width, dtype)
        gens = mod._fit_generations(width, depth, BUDGET)
        state = (fnp.zeros(width, dtype=dtype), fnp.eye(width, dtype=dtype), None, 0.0,
                 fnp.zeros(width, dtype=dtype))
        mus, vs, Us, uCus, trs = [], [], [], [], []
        for l in range(L_MAX):
            state = mod._layer(w[l], state, mod._tables(), dtype, const, 2.0, last_layer=False,
                               gens=gens, layer_index=l, depth=depth)
            mu = np.asarray(state[0], dtype=np.float64)
            C = np.asarray(state[1], dtype=np.float64)
            ev, evec = np.linalg.eigh(C)
            U = np.concatenate([(mu / np.linalg.norm(mu))[:, None], evec[:, ::-1][:, :4]], 1)
            mus.append(mu)
            vs.append(np.diag(C).copy())
            Us.append(U)
            uCus.append(np.einsum("ia,ij,ja->a", U, C, U))
            trs.append(np.trace(C))
            print(f"net {i} layer {l + 1}: mean|mu| {mu.mean():.4f} trC/n {np.trace(C) / width:.4f} "
                  f"top eig share {ev[-1] / ev.sum():.3f} cos(m̂,u1) {abs(U[:, 0] @ U[:, 1]):.3f}", flush=True)
        out["seeds"].append(seed)
        out["mu"].append(mus)
        out["var"].append(vs)
        out["U"].append(Us)
        out["uCu"].append(uCus)
        out["trC"].append(trs)
    np.savez(a.out, **{k: np.array(v) for k, v in out.items()}, nets=np.arange(lo, hi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
