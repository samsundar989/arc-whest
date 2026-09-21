"""One candidate, ONE net, the REAL meter. Run as a subprocess.

    python tools/_cost_worker.py <candidate.py> <panel> <index>

`local_sim` is APPENDED, never inserted, so `flopscope` and `whestbench` resolve
to the genuine installed packages. A process that inserted the shims and then
printed a FLOP count would be printing fiction, and this file's entire output is
a FLOP count.

FLOPs are net-independent: every graded run we hold reports spread 0 across all
100 nets, and so does every local 100-net mini. One net is therefore the whole
cost answer, which is what makes the fast loop affordable.

Weights are built in the `f64_readonly` grader-twin mode — float64 and
immutable, which is what the grading server actually hands a candidate. That
mode reproduced submission 330212's `flops_used` to delta 0 out of sample
(`docs/GOTCHAS.md`, 2026-09-07), so its reading needs no offset added.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"

# APPEND. The real packages must win.
for p in (str(ROOT), str(ROOT / "tools"), str(LOCAL_SIM)):
    if p not in sys.path:
        sys.path.append(p)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

import panels  # noqa: E402
import real_data as rd  # noqa: E402
import lifecycle_eval as le  # noqa: E402
from _peakrss import peak_rss_mib  # noqa: E402

BUDGET = 2 ** 41


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


def main(argv) -> int:
    cand, panel, index = argv[1], argv[2], int(argv[3])
    split, idx, seed, name = panels.nets(panel)[index]

    mod, full = le.load_candidate(cand)
    w_np, truth, _meta = rd.get(idx, split, dtype=np.float32)
    depth, width = w_np.shape[0], w_np.shape[1]

    est = mod.Estimator()
    ctx = le.SetupContext(le.LifecycleSpec(width=width, depth=depth,
                                           flop_budget=BUDGET, seed=0))
    t = time.perf_counter()
    setup_error = ""
    try:
        est.setup(ctx)
    except Exception as exc:
        setup_error = type(exc).__name__
    setup_s = time.perf_counter() - t

    # Built OUTSIDE the BudgetContext: `MLP.from_row` runs at dataset load on the
    # grader, so construction is free there too.
    mlp = _MLP(le.build_weights(list(w_np), "f64_readonly", fnp), width, depth, int(seed))

    from p2_weff import MaterializationCounter  # noqa: PLC0415

    budget = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
    err = ""
    with MaterializationCounter() as mc:
        try:
            with budget:
                raw = est.predict(mlp, BUDGET)
                raw = le.materialise_predictions(raw, depth=depth, width=width)
            le.validate_predictions(raw, depth=depth, width=width)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"

    try:
        ns = (budget.summary_dict(by_namespace=True).get("by_namespace") or {})
        ops = sum(float(v.get("calls", 0.0)) for v in ns.values())
    except Exception:
        ops = float("nan")

    print(json.dumps({
        "net": name, "panel": panel, "split": split, "index": idx,
        "flops": float(budget.flops_used),
        "multiplier": float(budget.flops_used) / BUDGET,
        "ops": ops,
        "materializations": sum(mc.counts.values()),
        "materialization_detail": mc.counts,
        "wall_s": float(budget.wall_time_s or 0.0),
        "residual_local_s": float(budget.residual_wall_time_s or 0.0),
        "setup_s": round(setup_s, 3),
        "setup_error": setup_error,
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "weights_mode": "f64_readonly",
        "error": err,
    }))
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
