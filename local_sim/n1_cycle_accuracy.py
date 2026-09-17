"""Skeptic follow-up (B): accuracy of a CONTINUOUS merged old tier, at tiny width.

Dense float64 lineage harness (`n1_relevance_oracle.load_lineage`) on v32_dense (full-
rank P2 residual, so only the old-tier compression is approximate). Before each
layer, every lineage whose age reaches `a` is merged into ONE old-tier tensor:

  join_b : T_old <- (T_old + T_new)(Pi, Pi, Pi), Pi = energy HOSVD projector of that sum, rank r
  join_a : fixed basis -- Pi from the FIRST merge; its span V is then transported with
           the lineages (V <- diag(w1) W^T V per layer, re-orthonormalised) and every later
           join is projected onto it without enlargement
  delete : the tier is dropped (the no-memory control)

Final-layer raw MSE against two independent Monte Carlo replicas (cross estimator
mean((p-r1)(p-r2)), unbiased for the true-function MSE). Pre-registered reading
(LOG 2026-09-17): the pre-registered N1 target is <= 5% raw loss vs the no-merge
chain at the rank the metered cycle can afford (r = n/8 maps to r = 128 at 1024).
"""
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(_HERE)
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE] + [R, _HERE]
import numpy as np
from scipy.special import ndtr
import flopscope as flops
flops.configure(symmetry_warnings=False)
import n1_relevance_oracle as O
from mint_provenance import canonical_mint


def energy_projector(T, r):
    U, _, _ = np.linalg.svd(T.reshape(T.shape[0], -1), full_matrices=False)
    Q = U[:, :r]
    return Q @ Q.T, Q


def run_variant(Ws, mode, a, r):
    import flopscope.numpy as fnp
    mod, state = O.load_lineage(os.path.join(R, "estimator_k3_v32_dense.py"))
    n = Ws[0].shape[0]
    dt = fnp.float64
    table = mod._tables()
    const = mod.Estimator()._constants(n, dt)
    st = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    V = None
    depth = len(Ws)
    for l in range(depth):
        mu, cov, tw, c4, gk4 = st
        state["layer"] = l
        if tw is not None and mode != "exact":
            old_items = [(b, T) for b, T in tw.items if b >= 0 and l - b >= a]
            keep = [(b, T) for b, T in tw.items if b >= 0 and l - b < a]
            tier = [T for b, T in tw.items if b < 0]
            if mode == "delete":
                tw = O.LTower(keep)
            elif old_items or tier:
                total = sum(T for _, T in old_items) + (tier[0] if tier else 0.0)
                if mode == "join_b":
                    P, _ = energy_projector(total, r)
                elif mode == "join_a":
                    if V is None:
                        _, V = energy_projector(total, r)
                    Q, _ = np.linalg.qr(V)
                    P = Q @ Q.T
                T_hat = np.einsum("ai,bj,ck,ijk->abc", P, P, P, total, optimize=True)
                tw = O.LTower(keep + [(-1, T_hat)])
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            out = mod._layer(fnp.asarray(Ws[l], dtype=dt),
                             (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, c4,
                              fnp.asarray(gk4, dtype=dt)),
                             table, dt, const, 2.0, last_layer=(l == depth - 1), gens=None,
                             layer_index=l)
        if l == depth - 1:
            return np.asarray(out[0], dtype=np.float64)
        if V is not None:
            # transport the fixed basis exactly as the lineages are: W^T, then diag(w1) at the next birth
            W = Ws[l]
            mp = W.T @ mu
            var = np.maximum(np.diag(W.T @ cov @ W), 1e-12)
            w1 = ndtr(np.clip(mp / np.sqrt(var), -40, 40))
            V = w1[:, None] * (W.T @ V)
        st = (np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64),
              out[2], out[3], np.asarray(out[4], dtype=np.float64))


def main():
    z = np.load(os.path.join(R, "experiments", "2026-09-17_n1_cycle", "tiny_truth.npz"))
    depth = 12
    report = {}
    for n in (64, 96):
        rows = []
        for net in range(4):
            key = f"n{n}_net{net}"
            seed = int(z[key + "_seed"])
            w, _, _ = canonical_mint(n, depth, seed)
            Ws = [np.asarray(x, dtype=np.float64) for x in w]
            r1, r2 = z[key + "_r1"][-1], z[key + "_r2"][-1]
            res = {}
            for mode, a, r in [("exact", 0, 0)] + [(m, a, r) for a in (2, 3) for m in ("delete",) for r in (0,)] + \
                    [(m, a, r) for a in (2, 3) for m in ("join_b", "join_a") for r in (n // 8, n // 4)]:
                p = run_variant(Ws, mode, a, r)
                mse = float(np.mean((p - r1) * (p - r2)))
                res[f"{mode}_a{a}_r{r}"] = mse
            ex = res["exact_a0_r0"]
            rows.append({"net": net, "seed": seed, "mse": res,
                         "raw_loss_pct": {k: 100.0 * (v / ex - 1.0) for k, v in res.items()}})
            print(n, net, " ".join(f"{k}:{100*(v/ex-1):+.2f}%" for k, v in res.items() if k != "exact_a0_r0"),
                  f"(exact mse {ex:.3e})", flush=True)
        report[n] = rows
        for k in rows[0]["mse"]:
            if k == "exact_a0_r0":
                continue
            c = sum(r["mse"][k] for r in rows) / sum(r["mse"]["exact_a0_r0"] for r in rows)
            print(f"  n={n} pooled {k}: raw loss {100*(c-1):+.2f}%")
    with open(os.path.join(R, "experiments", "2026-09-17_n1_cycle", "cycle_accuracy.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
