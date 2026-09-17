"""D2 DIAGNOSTIC: per-layer kappa4 state and error, public v17 vs our v32, on P11 networks.

Contract: `experiments/LOG.md` 2026-09-16 diagnostic contract (gap location) and the
stage-1b result entry. Diagnostic only: instrumented copies live in a temp dir, are
never packaged, and promote nothing.

Per network and layer it records, for each arm: the prediction MSE against the
averaged P11 truth for THAT layer (P11 stores all 16 layers), and the kappa4 state
statistics the two files share by formula -- dG (transported core), g_prev
(regenerated post-ReLU core), var (diag C_pre), lambda -- as mean, rms and the
ratio mean(dG)/mean(var). v17 exposes these through its own V17_DEBUG capture;
v32 through a one-line capture inserted after its dG line.

Run under workerq (1024x16 predicts). Truth is reserved through `truth_reads`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v32_instrumented(tmp):
    src = open(os.path.join(_ROOT, "estimator_k3_v32.py"), encoding="utf-8").read()
    anchor = "    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + "    _D2.append(dict(layer=layer_index, dG=dG, g_prev=gk4, var=var, "
                      "lam=float(_lam_prev), var_prev=fnp.diagonal(cov)))\n", 1)
    src = src.replace("_TABLES = None", "_D2 = []\n_TABLES = None", 1)
    p = os.path.join(tmp, "v32_d2.py")
    open(p, "w", encoding="utf-8").write(src)
    return p


def _stats(x):
    a = np.asarray(x, dtype=np.float64)
    return {"mean": float(a.mean()), "rms": float(np.sqrt((a * a).mean())),
            "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["V17_DEBUG"] = "1"
    import flopscope as flops
    import harness
    import truth_reads as tr
    from mint_provenance import canonical_mint
    flops.configure(symmetry_warnings=False)
    truth = os.path.join(_ROOT, "experiments", "2026-09-16_p11", "truth_P11.npz")
    with tr.reserve("P11", truth, consumer="local_sim/diag_d2_trace.py",
                    purpose="D2 diagnostic: per-layer MSE and kappa4 state, v17 vs v32",
                    ts=a.ts, instrument=os.path.abspath(__file__), result_artifact=a.out,
                    note="diagnostic; promotes nothing") as tp:
        z = dict(np.load(tp))
    tmp = tempfile.mkdtemp(prefix="d2_")
    arms = {"v32": _load(_v32_instrumented(tmp), "v32_d2"),
            "v17": _load(os.path.join(_ROOT, "external", "504aldo", "estimators", "estimator_v17.py"), "v17_d2")}
    width, depth = int(z["width"]), int(z["depth"])
    out = {"nets": []}
    for i in range(a.nets):
        seed = int(z["seeds"][i])
        y = (z["means_r1"][i].astype(np.float64) + z["means_r2"][i].astype(np.float64)) / 2.0
        w, es, _ = canonical_mint(width, depth, seed)
        rec = {"seed": seed, "arms": {}}
        for name, mod in arms.items():
            cap = mod._D2 if name == "v32" else mod.DEBUG
            del cap[:]
            weights = []
            for x in w:
                arr = np.array(x, dtype=np.float64, copy=True)
                arr.setflags(write=False)
                weights.append(arr)
            mlp = harness.MLP(weights=weights, width=width, depth=depth, seed=es)
            with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
                p = np.asarray(mod.Estimator().predict(mlp, 2 ** 41), dtype=np.float64)
                layers = []
                for c in cap:
                    d = {"layer": int(c["layer"]), "lam": float(c["lam"])}
                    for k in ("dG", "g_prev", "var", "var_prev"):
                        if c.get(k) is not None:
                            d[k] = _stats(c[k])
                    d["dG_over_var"] = d["dG"]["mean"] / max(d["var"]["mean"], 1e-300)
                    layers.append(d)
            rec["arms"][name] = {"layer_mse": [float(np.mean((p[l] - y[l]) ** 2)) for l in range(depth)],
                                 "trace": layers}
            print(name, seed, "final mse %.4e" % rec["arms"][name]["layer_mse"][-1], flush=True)
        out["nets"].append(rec)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    # summary
    for rec in out["nets"]:
        print("seed", rec["seed"])
        t17 = {d["layer"]: d for d in rec["arms"]["v17"]["trace"]}
        t32 = {d["layer"]: d for d in rec["arms"]["v32"]["trace"]}
        for l in range(depth):
            m17, m32 = rec["arms"]["v17"]["layer_mse"][l], rec["arms"]["v32"]["layer_mse"][l]
            s = f"  L{l:<2} mse v17 {m17:.3e} v32 {m32:.3e} x{m32 / max(m17, 1e-300):.2f}"
            if l in t17 and l in t32:
                s += (f" | dG/var v17 {t17[l]['dG_over_var']:.3e} v32 {t32[l]['dG_over_var']:.3e}"
                      f" | g_prev mean v17 {t17[l]['g_prev']['mean']:.3e} v32 {t32[l]['g_prev']['mean']:.3e}"
                      f" | lam v17 {t17[l]['lam']:.4f} v32 {t32[l]['lam']:.4f}")
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
