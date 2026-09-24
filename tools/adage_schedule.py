"""How many DISTINCT fold schedules does each tau produce? Integers only.

    python tools/adage_schedule.py --arms <arms.json> --n 20

The adaptive-age candidate's FLOP count is a function of ONE thing: at which
layers it folded the oldest exact lineage. Two nets with the same fold schedule
cost exactly the same. So before paying for `tools/cost_sweep.py` (a full f64
twin `predict` per net per variant, ~90 s each), this records the schedule -- a
tuple of integers -- for every net and every tau, and reports how many distinct
ones there are.

If a tau yields ONE schedule across every net, its cost is net-independent after
all and a single twin reading is its exact cost. If it yields many, the spread is
real and `cost_sweep.py` has to meter enough nets to bound it.

This records integers, never a FLOP. It runs on the accuracy shim, which is
exactly what AGENTS.md forbids quoting a counted FLOP from -- and nothing here
quotes one.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _rel(p) -> str:
    """`p` relative to the repo root for LOGGING, never raising.

    `Path.relative_to` raises when the path is not under `ROOT`, and under
    worker-q `ROOT` is the job's snapshot directory while an output path is
    deliberately absolute and outside it (`.gpuq.toml`: "Always give heavy jobs
    an ABSOLUTE output path"). That combination failed jobs 2090 and 2091 on a
    print statement AFTER all their work was written -- a cosmetic line that
    cost two exit-code-1 job reports.
    """
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

sys.path.insert(0, str(ROOT / "local_sim"))
BACKEND = os.environ.get("WHEST_BACKEND", "cupy").lower()
if BACKEND == "cupy":
    sys.path.insert(0, str(ROOT / "tools" / "gpu_shim"))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tools"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

BUDGET = 2 ** 41


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", default="estimator_k3_v60_adage.py")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        xp.get_default_memory_pool().set_limit(size=12 * 2 ** 30)

    import panels
    import real_data as rd
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ads", ROOT / a.candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ads"] = mod
    spec.loader.exec_module(mod)

    arms = json.loads(Path(a.arms).read_text(encoding="utf-8"))
    orig = mod._adapt_gate
    trace: list = []

    def spy(tw, LA, LP, n, c):
        d = orig(tw, LA, LP, n, c)
        trace.append((tw.k - tw.ko, bool(d)))
        return d

    mod._adapt_gate = spy
    est = mod.Estimator()
    rows = panels.nets("dev")[: a.n]
    sched: dict = {k: [] for k in arms}

    for i, (split, idx, seed, _nm) in enumerate(rows):
        w, _tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        est.setup(_Ctx(width, depth, BUDGET, 0))
        for name, over in arms.items():
            for k, v in over.items():
                setattr(mod, k, v)
            trace.clear()
            est.predict(_MLP(wf, width, depth, int(seed)), BUDGET)
            sched[name].append(tuple(1 if d else 0 for _ex, d in trace))
        mod._ADAPT_TAU = None
        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()
        print(f"[sched] net {i + 1}/{len(rows)}", flush=True)

    print(f"\n{'arm':10s} {'distinct':>8} {'folds min':>10} {'mean':>7} {'max':>5} "
          f"{'mean exact-age':>15}")
    summary = {}
    for name, lst in sched.items():
        c = collections.Counter(lst)
        folds = [sum(s) for s in lst]
        # mean number of exact lineages carried = (#gate evaluations) / (#folds)
        # is not it; report mean run length between folds instead.
        mean_age = float(np.mean([len(s) / max(1, sum(s)) for s in lst]))
        print(f"{name:10s} {len(c):>8} {min(folds):>10} {np.mean(folds):>7.2f} "
              f"{max(folds):>5} {mean_age:>15.3f}")
        summary[name] = {"distinct_schedules": len(c), "folds_min": min(folds),
                         "folds_mean": float(np.mean(folds)), "folds_max": max(folds),
                         "layers_per_fold": mean_age,
                         "most_common": list(c.most_common(3)[0][0]),
                         "most_common_count": c.most_common(1)[0][1], "n": len(lst)}
    if a.out:
        p = Path(a.out)
        if not p.is_absolute():
            p = ROOT / p
        p.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"\n[sched] -> {_rel(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
