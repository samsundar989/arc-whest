"""Real-flopscope smoke of the gated files at a SMALL shape (tiny; runs directly).

    .venv/Scripts/python.exe experiments/2026-09-27_gates/smoke_real.py [width depth]

(1) identity build == base, bit for bit, under the real meter (f64 read-only weights);
(2) FLOPs / op-call delta of the identity gates at this shape;
(3) a non-identity gate runs clean (no materialisation, finite, different answer).
`local_sim` is APPENDED (real packages win), as in tools/_cost_worker.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "local_sim")):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
import lifecycle_eval as le  # noqa: E402
from p2_weff import MaterializationCounter  # noqa: E402

assert "local_sim" not in fnp.__file__, fnp.__file__
BUDGET = 2 ** 41
D = ROOT / "experiments" / "2026-09-27_gates"


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


def run(path, w_np, width, depth, perturb=None):
    mod, _ = le.load_candidate(str(path))
    if perturb is not None:
        rng = np.random.default_rng(perturb)
        t = mod._GATE_TAB
        t["W2"] = (rng.standard_normal(np.shape(t["W2"])) * 0.5).tolist()
        t["b2"] = (rng.standard_normal(np.shape(t["b2"])) * 0.5).tolist()
    est = mod.Estimator()
    est.setup(le.SetupContext(le.LifecycleSpec(width=width, depth=depth, flop_budget=BUDGET,
                                               seed=0)))
    mlp = _MLP(le.build_weights(list(w_np), "f64_readonly", fnp), width, depth, 0)
    budget = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
    with MaterializationCounter() as mc:
        with budget:
            out = est.predict(mlp, BUDGET)
            out = le.materialise_predictions(out, depth=depth, width=width)
    ns = budget.summary_dict(by_namespace=True).get("by_namespace") or {}
    ops = sum(float(v.get("calls", 0.0)) for v in ns.values())
    return np.asarray(out), float(budget.flops_used), ops, sum(mc.counts.values())


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rng = np.random.default_rng(7)
    w_np = [(rng.standard_normal((width, width)) * np.sqrt(2.0 / width)).astype(np.float32)
            for _ in range(depth)]
    base = run(ROOT / "estimator_k3_v68_wino.py", w_np, width, depth)
    rep = {"shape": [width, depth], "base": {"flops": base[1], "ops": base[2], "mat": base[3]}}
    for arm in ("birth", "trans"):
        f = D / f"estimator_k3_v70g_{arm}.py"
        idn = run(f, w_np, width, depth)
        per = run(f, w_np, width, depth, perturb=1)
        rep[arm] = {"identity_bit_equal": bool(np.array_equal(idn[0], base[0])),
                    "identity_max_abs": float(np.abs(idn[0] - base[0]).max()),
                    "dflops": idn[1] - base[1], "dops": idn[2] - base[2], "mat": idn[3],
                    "perturbed_finite": bool(np.isfinite(per[0]).all()),
                    "perturbed_final_rel_change": float(
                        np.abs(per[0][-1] - base[0][-1]).max() / np.abs(base[0][-1]).max()),
                    "perturbed_mat": per[3]}
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
