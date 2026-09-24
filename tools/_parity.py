"""Bit-parity of two candidates' full (depth, width) predictions. Run as a subprocess.

    python tools/_parity.py <a.py> <b.py> [n_nets]

Used by `tools/build_v57.py --verify` to assert that the generated corrector file
is bit-identical to its base while `_CORR_ON` is False. "Bit-identical" here is
`np.array_equal` on the whole returned array, not a tolerance: a generated file
that changes the base chain by one ulp has a bug in its insertion, and a
tolerance would hide it.

cupy backend by default, like `tools/probe.py` -- this reports no FLOP, only
whether two prediction arrays agree.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def load(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
    nn = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        xp.get_default_memory_pool().set_limit(size=20 * 2 ** 30)

    import panels
    import real_data as rd

    ma, mb = load(pa, "_par_a"), load(pb, "_par_b")
    ea, eb = ma.Estimator(), mb.Estimator()

    bad = 0
    for k, (split, idx, seed, _nm) in enumerate(panels.nets("dev")[:nn]):
        w, _tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        ea.setup(_Ctx(width, depth, BUDGET, 0))
        eb.setup(_Ctx(width, depth, BUDGET, 0))
        A = ea.predict(_MLP(wf, width, depth, int(seed)), BUDGET)
        B = eb.predict(_MLP(wf, width, depth, int(seed)), BUDGET)
        A = np.asarray(xp.asnumpy(A) if xp is not None else A)
        B = np.asarray(xp.asnumpy(B) if xp is not None else B)
        same = A.shape == B.shape and np.array_equal(A, B)
        mx = float(np.abs(A - B).max()) if A.shape == B.shape else float("nan")
        print(f"[parity] net {k}: bit-identical={same}  max|diff|={mx:.3e}  "
              f"shape {A.shape}")
        if not same:
            bad += 1
        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()
    print("PARITY PASS" if not bad else f"PARITY FAIL ({bad}/{nn})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
