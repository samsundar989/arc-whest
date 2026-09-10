"""Reproduce the nine `raised during predict` MLPs from the depth-2 mini-100.

The 2026-09-10 mini-100 run of `estimator_k3_strassen_L2.py` reported
`100 of 100 MLPs failed (0 combined, 0 FLOP, 91 residual, 0 time, 9 error)`.

The 91 are the documented local-residual artefact and are not in question here.
**The 9 are.** Their indices are exactly the complement of the residual
warnings -- `6, 33, 34, 35, 36, 37, 38, 39, 40` -- and eight of them are
contiguous, which is the signature of a machine-state problem rather than of a
data-dependent defect. This module decides which, by running those MLPs one at a
time, in process, with the traceback printed rather than counted.

WHY IN PROCESS AND NOT THROUGH THE RUNNER
-----------------------------------------
The runner catches the exception and reports a count. A count cannot distinguish
`MemoryError` from an arithmetic bug, and those two have opposite consequences:
one is this host, the other would stop the canary. So the estimator is called
directly, with the REAL flopscope, on the organisers' own mini-split weights.

CONFIRM TIER, READ-ONLY. It reads the public mini split's WEIGHTS. It does not
read `all_layer_means`, computes no error, fits nothing and chooses nothing --
the question is whether `predict` raises, which the truth cannot answer.
`# seed-ledger: no block named`.

Run:
    uv run python local_sim/p2_mini_error_probe.py \
        --candidate estimator_k3_strassen_L2.py --indices 6 33 34 35 36 37 38 39 40
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# `local_sim` ships numpy shims NAMED flopscope/whestbench. APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import flopscope as flops  # noqa: E402  (the REAL one)

BUDGET = 2 ** 41


def _load(path):
    spec = importlib.util.spec_from_file_location("cand_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cand_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rss_gib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="estimator_k3_strassen_L2.py")
    ap.add_argument("--indices", type=int, nargs="+",
                    default=[6, 33, 34, 35, 36, 37, 38, 39, 40])
    ap.add_argument("--split", default="mini")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import real_data  # noqa: E402

    cand = a.candidate if os.path.isabs(a.candidate) \
        else os.path.join(_ROOT, a.candidate)
    sha = hashlib.sha256(open(cand, "rb").read()).hexdigest()
    mod = _load(cand)

    print("=" * 78)
    print("MINI-100 ERROR PROBE -- the nine that RAISED, one at a time")
    print("=" * 78)
    print(f"  candidate  {os.path.relpath(cand, _ROOT)}  sha256 {sha[:16]}...")
    print(f"  split      {a.split}   revision {real_data.REVISION}")
    print(f"  indices    {a.indices}")
    print()

    est = mod.Estimator()
    if hasattr(est, "setup"):
        class _Ctx:
            seed = 12345
        try:
            est.setup(_Ctx())
        except Exception:
            pass

    rows = []
    for idx in a.indices:
        w, _truth, meta = real_data.get(idx, split=a.split, dtype=np.float32)
        # The grader hands float64 IMMUTABLE weights. Reproduce both, because the
        # 2026-09-05 canary cost five slots on a branch only the grader ran.
        weights = []
        for x in np.asarray(w, dtype=np.float64):
            y = np.array(x, dtype=np.float64)
            y.setflags(write=False)
            weights.append(y)

        class _M:
            pass
        m = _M()
        m.width = weights[0].shape[0]
        m.depth = len(weights)
        m.weights = weights
        m.seed = int(meta.get("mlp_seed") or 0)

        rec = {"index": idx, "mlp_id": meta.get("mlp_id"),
               "width": m.width, "depth": m.depth,
               "rss_before_gib": _rss_gib()}
        t0 = time.perf_counter()
        try:
            with flops.budget(BUDGET, quiet=True) as b:
                out = est.predict(m, BUDGET)
                rec["flops_used"] = int(b.flops_used)
                rec["ops"] = len(b.op_log or [])
            arr = np.asarray(out, dtype=np.float64)
            rec.update({"raised": False, "shape": list(arr.shape),
                        "finite": bool(np.isfinite(arr).all()),
                        "min": float(arr.min()), "max": float(arr.max())})
        except BaseException as exc:                       # noqa: BLE001
            rec.update({"raised": True,
                        "exc_type": type(exc).__name__,
                        "exc": str(exc)[:400],
                        "traceback": traceback.format_exc()[-2500:]})
        rec["seconds"] = time.perf_counter() - t0
        rec["rss_after_gib"] = _rss_gib()
        rows.append(rec)

        if rec.get("raised"):
            print(f"  [{idx:3d}] *** RAISED {rec['exc_type']}: {rec['exc'][:90]}")
            print("        " + rec["traceback"].replace("\n", "\n        ")[-1200:])
        else:
            rss = rec.get("rss_after_gib")
            print(f"  [{idx:3d}] ok   {rec['seconds']:6.1f}s  "
                  f"flops {rec['flops_used']:,}  ops {rec['ops']}  "
                  f"finite {rec['finite']}  "
                  + (f"rss {rss:.2f} GiB" if rss is not None
                     else "rss unavailable (no psutil)"))
        del weights, m
        gc.collect()

    n_raised = sum(1 for r in rows if r.get("raised"))
    print()
    print(f"  RAISED {n_raised} of {len(rows)}")
    if n_raised == 0:
        print("  Every one of the nine completes in process. The failures were a")
        print("  property of the RUN, not of the candidate -- which is what the")
        print("  contiguous block of eight already suggested.")
    else:
        print("  At least one is reproducible in isolation. That is a candidate")
        print("  defect and it STOPS the canary until it is understood.")

    rec_all = {
        "artifact_kind": "mini100_error_probe",
        "candidate": os.path.relpath(cand, _ROOT),
        "candidate_sha256": sha,
        "split": a.split, "revision": real_data.REVISION,
        "n_raised": n_raised, "rows": rows,
        "tier": "CONFIRM (weights only; no truth read, nothing chosen)",
        "seed_ledger": "no block named",
    }
    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec_all, fh, indent=2)
        print(f"  artifact  {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
