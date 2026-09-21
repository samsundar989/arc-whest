"""One candidate, one range of panel nets, predictions only. Run as a subprocess.

**This process deliberately runs the NON-COUNTING numpy shims.** `local_sim` goes
to the FRONT of `sys.path`, so `import flopscope` and `import whestbench` resolve
to `local_sim/flopscope` and `local_sim/whestbench` rather than the real
packages. That is the whole point: the shim is ~10x faster than the metered
package and the accuracy path does not need a FLOP count.

It is also why this is a SEPARATE PROCESS. The repository rule is "append
`local_sim` to sys.path, never insert", because a process that inserts it and
then reports a FLOP number is reporting fiction. Here nothing can: this file
computes predictions and nothing else, and `tools/ev.py` takes every cost figure
from a different process running the real meter. The isolation is the safety
property, not the sys.path order.

Writes one .npz to the cache directory and prints one JSON line.

    python tools/_predict_worker.py <candidate.py> <panel> <lo> <hi> <out.npz>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_SIM = ROOT / "local_sim"

# DELIBERATE INSERT -- see the module docstring. Only ever in this file.
sys.path.insert(0, str(LOCAL_SIM))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _real_dataset():
    """Load panel weights through the REAL whestbench.dataset.

    `real_data` already knows how to reach past the shim for this one import,
    so the shim stays in front for the candidate while the data comes from the
    genuine package.
    """
    import importlib
    return importlib.import_module("real_data")


class _MLP:
    """What the grader hands `predict`: weights, width, depth, seed."""

    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights = weights
        self.width = width
        self.depth = depth
        self.seed = seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width = width
        self.depth = depth
        self.flop_budget = budget
        self.api_version = "1.0"
        self.scratch_dir = None
        self.submission_dir = None
        self.seed = seed


BUDGET = 2 ** 41


def main(argv) -> int:
    cand, panel, lo, hi, out = argv[1], argv[2], int(argv[3]), int(argv[4]), argv[5]

    import numpy as np
    import importlib.util

    sys.path.append(str(ROOT / "tools"))
    import panels  # noqa: PLC0415
    from _peakrss import peak_rss_mib  # noqa: PLC0415

    rd = _real_dataset()

    spec = importlib.util.spec_from_file_location("_ev_candidate", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ev_candidate"] = mod
    spec.loader.exec_module(mod)

    rows = panels.nets(panel)[lo:hi]
    if not rows:
        Path(out).write_bytes(b"")
        print(json.dumps({"n": 0}))
        return 0

    est = mod.Estimator()
    # The grader calls setup() once per submission, not per MLP.
    first_split, first_idx = rows[0][0], rows[0][1]
    w0, truth0, _ = rd.get(first_idx, first_split, dtype=np.float32)
    depth, width = w0.shape[0], w0.shape[1]
    try:
        est.setup(_Ctx(width, depth, BUDGET, 0))
    except Exception as exc:  # the grader tolerates a failed setup; so do we
        print(json.dumps({"setup_error": type(exc).__name__}), file=sys.stderr)

    preds = np.empty((len(rows), width), dtype=np.float32)
    truth = np.empty((len(rows), width), dtype=np.float64)
    errors = []
    t0 = time.perf_counter()
    for k, (split, idx, seed, _name) in enumerate(rows):
        if k == 0:
            w, tr = w0, truth0
        else:
            w, tr, _ = rd.get(idx, split, dtype=np.float32)
        # The grader's weights are float64 and immutable. Reproduce both, so the
        # opening cast in the candidate takes the branch it takes on the server.
        wf = []
        for x in w:
            a = np.asarray(x, dtype=np.float64)
            a.setflags(write=False)
            wf.append(a)
        mlp = _MLP(wf, width, depth, int(seed))
        try:
            raw = np.asarray(est.predict(mlp, BUDGET), dtype=np.float32)
            if raw.shape != (depth, width) or not np.all(np.isfinite(raw)):
                raise ValueError(f"bad prediction shape/finiteness {raw.shape}")
            preds[k] = raw[-1]
        except Exception as exc:
            # A failed MLP is scored as zeros by the grader. Record it as such
            # AND report it, so a candidate cannot fail quietly into a good mean.
            preds[k] = 0.0
            errors.append({"i": lo + k, "error": type(exc).__name__, "detail": str(exc)[:200]})
        truth[k] = tr[-1]
    dt = time.perf_counter() - t0

    np.savez_compressed(out, preds=preds, truth=truth,
                        indices=np.array([r[1] for r in rows], dtype=np.int64))
    print(json.dumps({"n": len(rows), "seconds": round(dt, 2),
                      "per_net": round(dt / len(rows), 3),
                      "peak_rss_mib": round(peak_rss_mib(), 1),
                      "errors": errors}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
