"""Meter a candidate's COMPLETE live-shape cost under variant knobs. Label-free.

Stage 2/4 cost work (`experiments/LOG.md`, 2026-09-16 execution contract, P4).
One canonical 1024x16 network (fixed seed outside every registered block --
`# seed-ledger: no block named`), float64 read-only weights as the grader hands
them, one `Estimator` with `setup` outside the meter, `predict` on MLP index 1
(steady state). Each variant runs in a FRESH subprocess so peak memory is its
own.

Reported per variant: counted FLOPs, C/B, metered operations (len(op_log)),
matmul-class operations, local wall (ADVISORY), process peak commit and working
set (Windows), and the max relative difference of the prediction against the
first variant listed (a float32 reassociation diagnostic, never an accuracy
claim -- accuracy is decided on DESIGN truth).

    uv run python local_sim/s2_cost_meter.py --candidate estimator_k3_v31.py \\
        --variant depth=4 --variant depth=5 --out experiments/<dir>/meter.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SEED = 90_000_016_001      # outside every registered block
WIDTH, DEPTH, BUDGET = int(os.environ.get("S2_W", 1024)), int(os.environ.get("S2_D", 16)), 2 ** 41


def _peak_mem():
    if os.name != "nt":
        import resource
        return {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return {"peak_working_set_bytes": int(pmc.PeakWorkingSetSize),
            "peak_commit_bytes": int(pmc.PeakPagefileUsage)}


def _ver(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


def _child(candidate, knobs, pred_out):
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]
    import importlib.util

    import numpy as np
    import flopscope
    import whestbench
    from mint_provenance import canonical_mint

    here = os.path.abspath(getattr(flopscope, "__file__", ""))
    if here.startswith(_HERE + os.sep):
        raise SystemExit("REFUSED: flopscope resolved to the local shim")
    flopscope.configure(symmetry_warnings=False)
    path = os.path.join(_ROOT, candidate)
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    spec = importlib.util.spec_from_file_location("cand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in knobs.items():
        if not hasattr(mod, k):
            raise SystemExit(f"candidate has no knob {k}")
        setattr(mod, k, v)

    class _Ctx:
        seed = SEED
        scratch_dir = _ROOT
        submission_dir = _ROOT
        width, depth, flop_budget, n_mlps = WIDTH, DEPTH, BUDGET, 2
    est = mod.Estimator()
    est.setup(_Ctx())
    out = None
    for i in range(2):
        w, es, _ = canonical_mint(WIDTH, DEPTH, SEED + i)
        weights = []
        for x in w:
            a = np.array(x, dtype=np.float64, copy=True)
            a.setflags(write=False)
            weights.append(flopscope.numpy.asarray(a))
        mlp = whestbench.MLP(weights=weights, width=WIDTH, depth=DEPTH, seed=es)
        with flopscope.BudgetContext(BUDGET, quiet=True) as ctx:
            t0 = time.perf_counter()
            p = est.predict(mlp, BUDGET)
            wall = time.perf_counter() - t0
            flops = int(ctx.flops_used)
            log = list(ctx.op_log)
        if i == 1:
            out = np.asarray(p, dtype=np.float64)
            names = {}
            for rec in log:
                names[rec.op_name] = names.get(rec.op_name, 0) + 1
            res = {"candidate": candidate, "candidate_sha256": sha, "knobs": knobs,
                   "flops": flops, "c_over_b": flops / BUDGET, "operations": len(log),
                   "matmul_ops": sum(v for k, v in names.items() if "matmul" in k),
                   "op_names": names, "local_wall_s_advisory": wall,
                   "flopscope": getattr(flopscope, "__version__", "?"),
                   "whestbench": _ver("whestbench")}
    np.save(pred_out, out)
    res.update(_peak_mem())
    print("RESULT " + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True,
                    help="repeatable; paired with --variant lists in order, or one candidate for all")
    ap.add_argument("--variant", action="append", default=[],
                    help="comma list of knob=value, e.g. _STRASSEN_DEPTH=5,_STRASSEN_MIN_BLOCK=32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        cand, knobs, pred = json.loads(a.child)
        _child(cand, knobs, pred)
        return 0
    import numpy as np
    variants = a.variant or [""]
    cands = a.candidate
    runs = []
    if len(cands) == 1:
        runs = [(cands[0], v) for v in variants]
    else:
        runs = [(c, v) for c in cands for v in variants]
    results = []
    base_pred = None
    outdir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(outdir, exist_ok=True)
    for i, (cand, var) in enumerate(runs):
        knobs = {}
        for kv in [x for x in var.split(",") if x]:
            k, v = kv.split("=")
            knobs[k] = int(v) if v.lstrip("-").isdigit() else v
        pred = os.path.join(outdir, f"_meter_pred_{i}.npy")
        cp = subprocess.run([sys.executable, "-u", __file__, "--candidate", cand, "--out", a.out,
                             "--child", json.dumps([cand, knobs, pred])],
                            capture_output=True, text=True, cwd=_ROOT)
        line = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
        if cp.returncode != 0 or not line:
            res = {"candidate": cand, "knobs": knobs, "failed": True,
                   "returncode": cp.returncode, "stderr_tail": cp.stderr[-3000:]}
        else:
            res = json.loads(line[0][7:])
            p = np.load(pred)
            if base_pred is None:
                base_pred = p
            s = max(float(np.abs(base_pred[1:]).max()), 1e-300)
            res["rel_diff_vs_first"] = float(np.abs(p[1:] - base_pred[1:]).max()) / s
            os.remove(pred)
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if k not in ("op_names", "stderr_tail")}),
              flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
