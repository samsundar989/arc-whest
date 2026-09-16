"""N1 prerequisite (i): does a cheap relevance weighting predict what compression COSTS?

Label-free, tiny shapes, float64. Execution contract: `experiments/LOG.md`
2026-09-16 amendment, "N1 old-history compression, economics added".

WHAT IS MEASURED (the definition the audit asked for)
----------------------------------------------------
State perturbation: at layer l0, the third-cumulant content of every lineage
with age >= a (the "old tier", summed into one tensor T_old in the pre-activation
coordinates of layer l0) is replaced ONCE by its multilinear projection

    T_hat = T_old(Pi, Pi, Pi),   Pi a rank-r projector (orthogonal or oblique),

and the recurrence then continues EXACTLY (the file's own `_layer`, dense
tensors, every covariance / Wick / birth / kappa4 route active).
Output error: E(Pi) = mean_i (mu_L,i(compressed) - mu_L,i(exact))^2 over the
final layer -- the quantity the score adds, in the score's units.

BASES COMPARED at equal rank
----------------------------
  energy : top-r eigenvectors of the mode Gram T_(1) T_(1)^T  (forward energy, 504aldo F72)
  meanJ  : the mean-path proxy made a VALID PSD objective. With the linear map
           J = D_L W_L^T ... D_{l0+1} W_{l0+1}^T D_l0  (D = diag Phi(alpha)) and
           K = J^T J + ridge = R R^T (Cholesky), minimise ||(T - T_hat) x_1 R^T x_2 R^T x_3 R^T||_F:
           whiten Tw = T(R^T, R^T, R^T), take the top-r left singular vectors Uw of
           Tw_(1), and use the K-orthogonal projector Pi = R^{-T} Uw Uw^T R^T.
           (A naive product of two Gram matrices is not symmetric; this is.)
  random : a random orthogonal basis -- the floor any informed choice must beat.

Also: signed-perturbation ranking. For random signed rank-one symmetric
perturbations dT = s u(x)u(x)u (u random unit, s = +-1), the true response
||mu_L(T+dT) - mu_L(T)|| vs the proxy ||dT x J||_F (Spearman rank correlation),
and the same for the energy proxy (which is constant for unit u: a null control).

VERDICTS (fixed before running)
-------------------------------
  proxy_informative : Spearman(true, meanJ proxy) >= 0.5 on >= 2 of 3 networks
  proxy_pays        : at r = n/4, E(meanJ) <= 0.5 E(energy) on >= 2 of 3 networks
                      (the kill-test's "halves the rank" translated to error at
                      equal rank) -- reported with r = n/8, 3n/8, n/2 beside it.
Neither verdict decides N1; they decide whether this PROXY is worth metering.

Run: uv run python local_sim/n1_relevance_oracle.py [--n 40 --depth 12 --l0 7 --age 3]
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
from scipy.special import ndtr  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import k4feed_oracle as K  # noqa: E402


class LTower(object):
    """Lineage-separated dense tower: list of (birth_layer, T)."""

    def __init__(self, items):
        self.items = items
        self.k = len(items)


def load_lineage(path):
    import flopscope.numpy as fnp
    mod = K._load(path)
    orig_append = mod._append
    state = {"layer": 0}

    def _np(x):
        return np.asarray(x, dtype=np.float64)

    def advance(WT, tw):
        W = _np(WT)
        return LTower([(b, np.einsum("ai,bj,ck,ijk->abc", W, W, W, T, optimize=True))
                       for b, T in tw.items])

    def wick_tower(col, tw):
        c = _np(col).reshape(-1)
        cc = c[:, None, None] * c[None, :, None] * c[None, None, :]
        return LTower([(b, T * cc) for b, T in tw.items])

    def total(tw):
        return sum(T for _, T in tw.items)

    def dslice3(tw):
        T = total(tw)
        i = np.arange(T.shape[0])
        return fnp.asarray(T[i, i, i].copy(), dtype=fnp.float64)

    def dslice21(tw, off_mask):
        return fnp.asarray(np.einsum("aab->ab", total(tw)).copy(), dtype=fnp.float64) * off_mask

    def append(tw, *args, **kwargs):
        n = args[3]
        Tn = K.tower_tensor(orig_append(None, *args, **kwargs), n)
        items = [] if tw is None else list(tw.items)
        items.append((state["layer"], Tn))
        return LTower(items)

    mod._advance, mod._wick_tower = advance, wick_tower
    mod._dslice3, mod._dslice21 = dslice3, dslice21
    mod._append, mod._truncate = append, (lambda tw, g, n: tw)
    return mod, state


def run(mod, state, Ws, *, l0=None, age=None, project=None, capture=None):
    import flopscope as flops
    import flopscope.numpy as fnp
    n = Ws[0].shape[0]
    dt = fnp.float64
    table = mod._tables()
    const = mod.Estimator()._constants(n, dt)
    st = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    depth = len(Ws)
    for l in range(depth):
        mu, cov, tw, c4, gk4 = st
        state["layer"] = l
        if l == l0 and tw is not None and project is not None:
            # compress the old tier IN THE COORDINATES THE LAYER WILL READ: the
            # file transports the tower inside `_layer`, so project the tier
            # after transport by pre-transporting a copy and pulling it back is
            # not possible; instead project in post-previous-layer coordinates
            # with the basis supplied for those coordinates.
            old = [(b, T) for b, T in tw.items if l - b >= age]
            new = [(b, T) for b, T in tw.items if l - b < age]
            if old:
                To = sum(T for _, T in old)
                P = project
                That = np.einsum("ai,bj,ck,ijk->abc", P, P, P, To, optimize=True)
                tw = LTower(new + [(-1, That)])
        if capture is not None:
            capture(l, st)
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            out = mod._layer(fnp.asarray(Ws[l], dtype=dt),
                             (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, c4,
                              fnp.asarray(gk4, dtype=dt)),
                             table, dt, const, 2.0, last_layer=(l == depth - 1), gens=None,
                             layer_index=l)
        if l == depth - 1:
            return np.asarray(out[0], dtype=np.float64)
        st = (np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64),
              out[2], out[3], np.asarray(out[4], dtype=np.float64))


def mean_path_J(caps, Ws, l0):
    """J maps a perturbation of the tier's coordinates at layer l0 (post-activation
    of layer l0-1, i.e. BEFORE W_l0 transport) to final means along the mean path:
    J = D_L W_L^T ... D_l0 W_l0^T, D = diag Phi(alpha) at each layer."""
    n = Ws[0].shape[0]
    J = np.eye(n)
    for l in range(l0, len(Ws)):
        mu, cov = caps[l][0], caps[l][1]
        W = Ws[l]
        mp = W.T @ mu
        var = np.maximum(np.diag(W.T @ cov @ W), 1e-12)
        D = ndtr(mp / np.sqrt(var))
        J = (D[:, None] * W.T) @ J
    return J


def basis_energy(T, r):
    n = T.shape[0]
    G = T.reshape(n, -1)
    U, _, _ = np.linalg.svd(G, full_matrices=False)
    Q = U[:, :r]
    return Q @ Q.T


def basis_weighted(T, K, r, ridge=1e-10):
    n = T.shape[0]
    Kr = K + ridge * np.trace(K) / n * np.eye(n)
    R = np.linalg.cholesky(Kr)          # K = R R^T
    Tw = np.einsum("ai,bj,ck,ijk->abc", R.T, R.T, R.T, T, optimize=True)
    U, _, _ = np.linalg.svd(Tw.reshape(n, -1), full_matrices=False)
    Uw = U[:, :r]
    return np.linalg.solve(R.T, Uw @ (Uw.T @ R.T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=os.path.join(_ROOT, "estimator_k3_v32_dense.py"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--l0", type=int, default=7)
    ap.add_argument("--age", type=int, default=3)
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--probes", type=int, default=40)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import flopscope as flops
    from mint_provenance import canonical_mint
    flops.configure(symmetry_warnings=False)
    n = a.n
    ranks = sorted({max(1, n // 8), n // 4, (3 * n) // 8, n // 2})
    report = {"candidate": os.path.basename(a.candidate), "n": n, "depth": a.depth,
              "l0": a.l0, "age": a.age, "nets": []}
    informative = pays = 0
    for net in range(a.nets):
        w, _es, _ = canonical_mint(n, a.depth, 90_000_017_000 + net)   # outside every block
        Ws = [np.asarray(x, dtype=np.float64) for x in w]
        mod, state = load_lineage(a.candidate)
        caps = {}
        snap = {}

        def cap(l, st):
            caps[l] = st
            if l == a.l0 and st[2] is not None:
                snap["old"] = sum(T for b, T in st[2].items if l - b >= a.age)
        mu_exact = run(mod, state, Ws, capture=cap)
        To = snap["old"]
        J = mean_path_J(caps, Ws, a.l0)
        Kmat = J.T @ J
        rng = np.random.default_rng(1000 + net)
        row = {"net": net, "old_tier_fro": float(np.linalg.norm(To)), "ranks": {}}
        for r in ranks:
            errs = {}
            for name, P in (("energy", basis_energy(To, r)),
                            ("meanJ", basis_weighted(To, Kmat, r)),
                            ("random", (lambda Q: Q @ Q.T)(np.linalg.qr(rng.normal(size=(n, r)))[0]))):
                mu_c = run(mod, state, Ws, l0=a.l0, age=a.age, project=P)
                errs[name] = float(np.mean((mu_c - mu_exact) ** 2))
            row["ranks"][r] = errs
        full = run(mod, state, Ws, l0=a.l0, age=a.age, project=np.eye(n))
        row["identity_projection_err"] = float(np.mean((full - mu_exact) ** 2))
        # signed rank-one probes, scaled to 1e-3 of the tier norm (linear regime)
        true_r, prox_r = [], []
        eps = 1e-3 * np.linalg.norm(To)
        for _ in range(a.probes):
            u = rng.normal(size=n)
            u /= np.linalg.norm(u)
            s = rng.choice([-1.0, 1.0])
            dT = s * eps * np.einsum("a,b,c->abc", u, u, u)
            mu_p = run(mod, state, Ws, l0=a.l0, age=a.age,
                       project=None) if False else None
            mod2, state2 = load_lineage(a.candidate)

            def inj(l, st, dT=dT):
                return None
            # inject by projecting with identity and adding dT to the old tier
            mu_pp = _run_inject(mod2, state2, Ws, a.l0, a.age, dT)
            true_r.append(float(np.linalg.norm(mu_pp - mu_exact)))
            prox_r.append(float(np.linalg.norm(np.einsum("ai,bj,ck,ijk->abc", J, J, J, dT,
                                                         optimize=True))))
        rho = float(spearmanr(true_r, prox_r).correlation)
        row["spearman_true_vs_meanJ"] = rho
        report["nets"].append(row)
        r4 = n // 4
        e = row["ranks"][r4]
        informative += rho >= 0.5
        pays += e["meanJ"] <= 0.5 * e["energy"]
        print(f"net {net}: |T_old| {row['old_tier_fro']:.3e}  identity-proj err {row['identity_projection_err']:.1e}"
              f"  spearman(true, meanJ) {rho:+.3f}")
        for r in ranks:
            e = row["ranks"][r]
            print(f"   r={r:<3} E energy {e['energy']:.3e}  meanJ {e['meanJ']:.3e}  random {e['random']:.3e}"
                  f"   meanJ/energy {e['meanJ'] / max(e['energy'], 1e-300):.3f}")
    report["verdicts"] = {"proxy_informative": informative >= 2, "proxy_pays": pays >= 2,
                          "informative_nets": informative, "pays_nets": pays}
    print("\nVERDICT proxy_informative:", report["verdicts"]["proxy_informative"],
          f"({informative}/{a.nets})   proxy_pays:", report["verdicts"]["proxy_pays"],
          f"({pays}/{a.nets})")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


def _run_inject(mod, state, Ws, l0, age, dT):
    """Exact run with dT added to the old tier at layer l0 (same coordinates as `project`)."""
    import flopscope as flops
    import flopscope.numpy as fnp
    n = Ws[0].shape[0]
    dt = fnp.float64
    table = mod._tables()
    const = mod.Estimator()._constants(n, dt)
    st = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    depth = len(Ws)
    for l in range(depth):
        mu, cov, tw, c4, gk4 = st
        state["layer"] = l
        if l == l0 and tw is not None:
            tw = LTower(list(tw.items) + [(-1, dT)])
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            out = mod._layer(fnp.asarray(Ws[l], dtype=dt),
                             (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, c4,
                              fnp.asarray(gk4, dtype=dt)),
                             table, dt, const, 2.0, last_layer=(l == depth - 1), gens=None,
                             layer_index=l)
        if l == depth - 1:
            return np.asarray(out[0], dtype=np.float64)
        st = (np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64),
              out[2], out[3], np.asarray(out[4], dtype=np.float64))


if __name__ == "__main__":
    raise SystemExit(main())
