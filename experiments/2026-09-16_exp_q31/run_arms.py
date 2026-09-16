"""Run 504aldo estimators (v16b / v17_open) under env-knob arms against MC truth.
Usage: python3 run_arms.py n L nseeds [arms_json]"""
import sys, os, json, time, importlib, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimators"))
from whestbench.domain import MLP
from whestbench import SetupContext

V17_KEYS = ["V17_LAM_DIAG_SCALE", "V17_LAM_OFF_SCALE", "V17_NO_FEED", "V17_NO_WK431", "V17_NO_REGEN", "V17_NO_CORR", "V17_LAM_SCALE", "V17_FORCE_RIDERS", "V17_DEBUG"]

def load(module, env):
    for k in V17_KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    if module in sys.modules:
        return importlib.reload(sys.modules[module])
    return importlib.import_module(module)

def run_arm(name, module, env, n, L, seeds):
    mod = load(module, env)
    per_layer = []; finals = []; t0 = time.time()
    for s in seeds:
        d = np.load(f"mlp_n{n}_L{L}_s{s}.npz")
        ws = [w.astype(np.float32) for w in d["weights"]]
        truth = d["truth"]
        est = mod.Estimator(); est.setup(SetupContext(seed=0))
        pred = np.asarray(est.predict(MLP(width=n, depth=L, weights=ws, seed=s), 0), dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            print(f"  [{name}] seed {s}: NON-FINITE prediction", flush=True)
        pl = np.mean((pred - truth) ** 2, axis=1)
        per_layer.append(pl); finals.append(pl[-1])
    per_layer = np.array(per_layer)
    res = dict(name=name, module=module, env=env, final_mse=float(np.mean(finals)),
               final_mse_per_seed=[float(x) for x in finals],
               per_layer_mse=[float(x) for x in per_layer.mean(0)], wall_s=time.time() - t0)
    print(f"{name:34s} final={res['final_mse']:.4e}  per-seed sd={np.std(finals):.1e}  wall={res['wall_s']:.1f}s", flush=True)
    return res

if __name__ == "__main__":
    n, L, ns = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    arms = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    seeds = list(range(ns))
    if arms is None:
        FR = {"V17_FORCE_RIDERS": 1}
        arms = [
            ("A0 v16b shipped (scalar-K4 base path)", "estimator_v16b", {}),
            ("A1 v17 NO_REGEN (v16b + vec+ww, no corr)", "estimator_v17_open", {**FR, "V17_NO_REGEN": 1}),
            ("B  v17 regen as shipped (lam x1)", "estimator_v17_open", {**FR}),
            ("C  v17 no wk431", "estimator_v17_open", {**FR, "V17_NO_WK431": 1}),
            ("D  v17 no feed", "estimator_v17_open", {**FR, "V17_NO_FEED": 1}),
            ("E  v17 lam=0 (diag core only)", "estimator_v17_open", {**FR, "V17_LAM_SCALE": 0}),
            ("F  v17 no wk431 + no feed", "estimator_v17_open", {**FR, "V17_NO_WK431": 1, "V17_NO_FEED": 1}),
        ]
        for sc in [0.5, 2, 4]:
            arms.append((f"B  v17 regen lam x{sc}", "estimator_v17_open", {**FR, "V17_LAM_SCALE": sc}))
    results = []
    noise = []
    for s in seeds:
        d = np.load(f"mlp_n{n}_L{L}_s{s}.npz"); noise.append(d["noise_var"].mean(1))
    noise = np.array(noise).mean(0)
    print(f"MC noise floor (mean var/N) per layer: {np.array2string(noise, precision=2)}", flush=True)
    for name, module, env in arms:
        results.append(run_arm(name, module, env, n, L, seeds))
    out = dict(n=n, L=L, seeds=seeds, noise_floor_per_layer=[float(x) for x in noise], arms=results)
    json.dump(out, open(f"results_n{n}_L{L}.json", "w"), indent=1)
