"""LABEL-FREE proxy for shared-coordinate damage: deviation from exact v32 at 1024x16.

No truth, no seed block. Canonical networks (seeds 90_000_016_001 + i, outside every
block -- `# seed-ledger: no block named`), float64 read-only weights, the grader's
MLP type. For each (candidate, knobs) the final-layer prediction is compared with
exact v32 on the same network:

    dev_mse = mean((p_arm - p_v32)^2)     (added squared error if the arm's error is
                                           uncorrelated with v32's -- an approximation)

When an arm's damage is much larger than v32's own error (P11: 3.3e-8) the arm's raw MSE is
roughly v32's plus dev_mse. The four P11-screened arms are included as CALIBRATION:
their measured raw MSE increases are +2.54e-8 (a3r256), +6.34e-8 (a3r192), +9.07e-8 (a2r256),
+1.83e-7 (a2r192). The proxy is used only if it reproduces that ordering within 2x.

    uv run python local_sim/s7_proxy.py --out experiments/2026-09-17_step3/proxy.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

SEED = 90_000_016_001      # seed-ledger: no block named (as s2_cost_meter)
W, D, B = 1024, 16, 2 ** 41
N_NETS = int(os.environ.get("PROXY_NETS", "2"))

# (label, file, knobs)
CONFIGS = [
    ("cal a3r256", "estimator_k3_v36_v32_a3r256.py", {}),
    ("cal a3r192", "estimator_k3_v36_v32_a3r192.py", {}),
    ("cal a2r256", "estimator_k3_v36_v32_a2r256.py", {}),
    ("cal a2r192", "estimator_k3_v36_v32_a2r192.py", {}),
]
LEAN = "estimator_k3_v36_v32_a3r256_lean.py"
for age in (3, 4):
    for rank in (256, 384, 512):
        for passes in (1, 3):
            for wts in ("cons", "sqrt", "unif"):
                if passes == 3 and wts != "cons" and rank != 384:
                    continue
                CONFIGS.append((f"a{age}r{rank}p{passes}{wts}", LEAN,
                                {"_SHARED_AGE": age, "_SHARED_RANK": rank, "_SHARED_PASSES": passes,
                                 "_SHARED_WEIGHTS": wts}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import flopscope
    import whestbench
    import importlib.util
    from mint_provenance import canonical_mint
    flopscope.configure(symmetry_warnings=False)

    def load(path, knobs):
        spec = importlib.util.spec_from_file_location("cand_%d" % abs(hash((path, tuple(knobs.items())))),
                                                      os.path.join(_ROOT, path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, v in knobs.items():
            if not hasattr(mod, k):
                raise SystemExit(f"{path} has no knob {k}")
            setattr(mod, k, v)
        return mod

    class _Ctx:
        seed = SEED
        scratch_dir = submission_dir = _ROOT
        width, depth, flop_budget, n_mlps = W, D, B, N_NETS

    def run(mod, i):
        w, es, _ = canonical_mint(W, D, SEED + i)
        weights = []
        for x in w:
            arr = np.array(x, dtype=np.float64, copy=True)
            arr.setflags(write=False)
            weights.append(flopscope.numpy.asarray(arr))
        est = mod.Estimator()
        est.setup(_Ctx())
        with flopscope.BudgetContext(B, quiet=True) as ctx:
            p = est.predict(whestbench.MLP(weights=weights, width=W, depth=D, seed=es), B)
            f = int(ctx.flops_used)
            ops = len(ctx.op_log)
        return np.asarray(p, dtype=np.float64)[-1], f, ops

    t0 = time.time()
    ref = [run(load("estimator_k3_v32.py", {}), i) for i in range(N_NETS)]
    out = {"artifact": "s7_proxy", "label_free": True, "nets": N_NETS, "seed0": SEED,
           "v32_sha256": hashlib.sha256(open(os.path.join(_ROOT, "estimator_k3_v32.py"), "rb").read()).hexdigest(),
           "rows": []}
    for label, path, knobs in CONFIGS:
        mod = load(path, knobs)
        devs, fl, opsl = [], [], []
        for i in range(N_NETS):
            p, f, ops = run(mod, i)
            devs.append(float(((p - ref[i][0]) ** 2).mean()))
            fl.append(f)
            opsl.append(ops)
        row = {"label": label, "file": path, "knobs": knobs,
               "sha256": hashlib.sha256(open(os.path.join(_ROOT, path), "rb").read()).hexdigest(),
               "dev_mse": float(np.mean(devs)), "dev_mse_per_net": devs,
               "c_over_b": float(np.mean(fl)) / B, "ops": int(np.mean(opsl))}
        out["rows"].append(row)
        print("%-22s dev_mse %.3e  C/B %.4f  ops %d  (%.0fs)" % (label, row["dev_mse"], row["c_over_b"],
                                                               row["ops"], time.time() - t0), flush=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
