"""Per-op-family bill of one candidate under whichever flopscope the interpreter has (S10.1).

    python tools/bill_meter.py <candidate.py> --panel dev --nets 0,1,0 --out <json>

Runs several predicts in ONE process, each in its own BudgetContext, so the first
predict's bill (first-call premium, intel2 M10) can be compared with later ones.
Weights are the f64 read-only grader twin (`_cost_worker.py`), built outside the
budget. The per-op rollup is read with `summary_dict()` AFTER each context closes,
never inside predict (AGENTS grader contract 4 is about the shipped estimator; this
is the harness). Run it once under the repo venv (0.12.1) and once under a scratch
venv with 0.12.2, then diff with `tools/bill_diff.py`.

`local_sim` is APPENDED so the real flopscope wins over the shim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tools"), str(ROOT / "local_sim")):
    if p not in sys.path:
        sys.path.append(p)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

import panels  # noqa: E402
import real_data as rd  # noqa: E402
import lifecycle_eval as le  # noqa: E402

BUDGET = 2 ** 41


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


def _ops_table(summary):
    out = {}
    for name, rec in (summary.get("operations") or {}).items():
        if isinstance(rec, dict):
            out[name] = {"flops": float(rec.get("flop_cost", rec.get("flops", 0.0)) or 0.0),
                         "calls": float(rec.get("calls", 0.0) or 0.0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--panel", default="dev")
    ap.add_argument("--nets", default="0,1,0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dist = importlib.metadata.distribution("flopscope")
    fs_file = str(Path(flops.__file__).resolve())
    if "local_sim" in fs_file:
        raise SystemExit(f"flopscope resolved to the shim: {fs_file}")
    src = Path(a.candidate).read_bytes()
    mod, full = le.load_candidate(a.candidate)
    rows = []
    est = None
    for k, index in enumerate(int(x) for x in a.nets.split(",")):
        split, idx, seed, name = panels.nets(a.panel)[index]
        w_np, _truth, _meta = rd.get(idx, split, dtype=np.float32)
        depth, width = w_np.shape[0], w_np.shape[1]
        if est is None:
            est = mod.Estimator()
            est.setup(le.SetupContext(le.LifecycleSpec(width=width, depth=depth,
                                                       flop_budget=BUDGET, seed=0)))
        mlp = _MLP(le.build_weights(list(w_np), "f64_readonly", fnp), width, depth, int(seed))
        budget = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
        t = time.perf_counter()
        with budget:
            raw = est.predict(mlp, BUDGET)
            raw = le.materialise_predictions(raw, depth=depth, width=width)
        wall = time.perf_counter() - t
        le.validate_predictions(raw, depth=depth, width=width)
        summ = budget.summary_dict()
        ops = _ops_table(summ)
        rows.append({"call": k, "net": name, "index": index, "flops": int(budget.flops_used),
                     "F_over_B": budget.flops_used / BUDGET,
                     "ops": int(sum(v["calls"] for v in ops.values())),
                     "local_wall_s_instrument_only": round(wall, 2),
                     "operations": ops,
                     "pred_sha": hashlib.sha256(np.asarray(raw, dtype=np.float64).tobytes()).hexdigest()[:16]})
        print(json.dumps({kk: v for kk, v in rows[-1].items() if kk != "operations"}), flush=True)
        del raw, mlp
    res = {"candidate": full, "sha256": hashlib.sha256(src).hexdigest(),
           "flopscope_version": dist.version, "flopscope_file": fs_file,
           "panel": a.panel, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
