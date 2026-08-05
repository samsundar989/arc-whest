"""
Why is our Sobol RQMC stuck at alpha = 1.00, and can a rotation fix it?

A Sobol net is equidistributed best in its LOW-INDEXED dimensions; quality decays
with dimension index. Our estimator feeds the 256 input coordinates in arbitrary
order, so the network's most influential input directions land on arbitrary
(often poor) Sobol dimensions. That is the classic cause of QMC degenerating to
Monte Carlo rates in high dimension.

Because x ~ N(0, I) is rotationally invariant, we may substitute x = Q u for ANY
orthogonal Q at zero statistical cost. Choosing Q so that the most influential
directions come first should raise the effective QMC rate.

Two candidate rotations, both computable online:
  eigW1  eigenvectors of W1 W1^T, ordered by eigenvalue  (one symmetric eigh,
         ~1e7 FLOPs; no covariance propagation needed)
  jac    eigenvectors of J J^T where J = W1 diag(Phi1) ... W_L diag(Phi_L) is the
         linearised input->output Jacobian (needs the firing probabilities, so it
         piggybacks on covariance propagation)

Cost of applying the rotation is one (N x 256) @ (256 x 256) matmul = the same as
one extra network layer, i.e. ~3% of the sampling budget.

Run: uv run python local_sim/rotation_probe.py [--mlps 40] [--ns 4096,16384]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402
from covprop_features import covprop  # noqa: E402

F32 = np.float32
COST_PER_SAMPLE = 32 * 2 * 256 * 256 + 32 * 256
BUDGET = 2.72e11


def sobol_normals(n, d, seed):
    """Digitally-shifted Sobol normals — the same construction estimator_v1_1 uses."""
    import importlib.util

    from scipy.special import ndtri

    spec = importlib.util.spec_from_file_location("v11_rot", os.path.join(_ROOT, "estimator_v1_1.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v11_rot"] = mod
    spec.loader.exec_module(mod)
    bits = mod._SOBOL_BITS
    dirn = mod._direction_numbers()
    vt = np.array([dirn[b][:d] for b in range(bits)], dtype=np.uint32)
    idx = np.arange(n, dtype=np.uint32)
    idx = (idx ^ (idx >> np.uint32(1)))[:, None]
    pts = np.zeros((n, d), dtype=np.uint32)
    for b in range(bits):
        pts = np.bitwise_xor(pts, ((idx >> np.uint32(b)) & np.uint32(1)) * vt[b])
    shift = np.random.default_rng(seed).integers(0, 1 << bits, size=d, dtype=np.uint32)
    u = (np.bitwise_xor(pts, shift).astype(np.float64) + 0.5) / float(1 << bits)
    np.clip(u, 1e-12, 1 - 1e-12, out=u)
    return ndtri(u).astype(F32)


def rot_eigW1(w):
    a = np.asarray(w[0], dtype=np.float64)
    ev, Q = np.linalg.eigh(a @ a.T)
    return Q[:, ::-1].astype(F32)  # descending eigenvalue


def rot_jac(w, diag):
    """J = product of W_k diag(Phi_k); take input-space eigenvectors of J J^T."""
    J = np.eye(w[0].shape[0])
    for k in range(len(w)):
        J = J @ np.asarray(w[k], dtype=np.float64)
        J = J * diag[k][None, :]  # diag(Phi_k) on the right
    ev, Q = np.linalg.eigh(J @ J.T)
    return Q[:, ::-1].astype(F32)


def phis(w):
    """Firing probabilities per layer, from the same closure covprop uses."""
    from scipy.special import ndtr

    n = w[0].shape[1]
    mu = np.zeros(n)
    cov = np.eye(n)
    off = np.ones((n, n)) - np.eye(n)
    out = []
    for wl in w:
        wl = np.asarray(wl, dtype=np.float64)
        mu_pre = wl.T @ mu
        cov_pre = wl.T @ cov @ wl
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        Phi = ndtr(a)
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * a * a)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(Phi, Phi) * cov_pre * off + np.diag(var)
        cov = 0.5 * (cov + cov.T)
        out.append(Phi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=40)
    ap.add_argument("--offset", type=int, default=800)
    ap.add_argument("--split", default="full")
    ap.add_argument("--ns", default="4096,16384")
    args = ap.parse_args()
    ns = [int(v) for v in args.ns.split(",")]

    methods = ("plain", "eigW1", "jac")
    acc = {(mth, n): [] for mth in methods for n in ns}

    for j in range(args.mlps):
        i = args.offset + j
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        wl32 = [np.asarray(x, dtype=F32) for x in w]
        t = truth[-1]
        Q1 = rot_eigW1(w)
        Qj = rot_jac(w, phis(w))
        for n in ns:
            z = sobol_normals(n, 256, int(mlp.seed) & 0x7FFFFFFF)
            for mth in methods:
                x = z if mth == "plain" else (z @ (Q1.T if mth == "eigW1" else Qj.T))
                h = x
                for ww in wl32:
                    h = np.maximum(h @ ww, F32(0))
                est = h.mean(axis=0, dtype=np.float64)
                acc[(mth, n)].append(float(((est - t) ** 2).mean()))
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{args.mlps}", flush=True)

    print(f"\n{args.mlps} held-out MLPs\n")
    print(f"{'method':<10}" + "".join(f"{'N=' + str(n):>16}" for n in ns) + f"{'alpha':>8}")
    print("-" * (10 + 16 * len(ns) + 8))
    for mth in methods:
        vals = [float(np.mean(acc[(mth, n)])) for n in ns]
        alpha = -np.polyfit(np.log(ns), np.log(vals), 1)[0] if len(ns) > 1 else float("nan")
        print(f"{mth:<10}" + "".join(f"{v:>16.4e}" for v in vals) + f"{alpha:>8.2f}")

    base = float(np.mean(acc[("plain", ns[-1])]))
    print(f"\nat N={ns[-1]} (C/B={ns[-1] * COST_PER_SAMPLE / BUDGET:.3f}):")
    for mth in methods:
        v = float(np.mean(acc[(mth, ns[-1])]))
        print(f"  {mth:<8} {v:.4e}  ({100 * (1 - v / base):+.1f}% vs plain)")


if __name__ == "__main__":
    main()
