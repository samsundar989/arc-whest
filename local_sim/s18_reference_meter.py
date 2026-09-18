"""One-network, offline-only counted meter for C10 same-closure references.

Submit --all through workerq for width 1024. It launches one child process per
arm so process peak memory is arm-specific, writes each complete arm atomically,
and never opens truth. A high instrumented budget prevents the estimator's
normal generation cap; such a reference cannot be treated as a grader candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != HERE]

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from whestbench import MLP  # noqa: E402

sys.path.append(str(HERE))
import lifecycle_eval as le  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402
from p2_weff import MaterializationCounter  # noqa: E402
from s18_same_closure_reference import CONTROL, SWITCHES, build, load, offline_predict  # noqa: E402

OFFICIAL_BUDGET = 2**41
DIAGNOSTIC_BUDGET = 2**55
PROBE_SEED = 19_990_918  # label-free probe, not a DESIGN/DECIDE seed block


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _peak_rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        value = Counters()
        value.cb = ctypes.sizeof(value)
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        get_memory.restype = wintypes.BOOL
        handle = get_current()
        if not get_memory(handle, ctypes.byref(value), value.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(value.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1 if sys.platform == "darwin" else 1024))


def _assert_ledger_conserved(total, operations):
    """Every billed FLOP must occur in the saved operation ledger, exactly."""
    if not isinstance(total, int) or total <= 0 or not operations:
        raise ValueError("empty or invalid counted FLOP ledger")
    if any(not isinstance(rec["flops"], int) or rec["flops"] < 0 or
           not isinstance(rec["calls"], int) or rec["calls"] < 0
           for rec in operations.values()):
        raise ValueError("negative or noninteger operation totals")
    billed = sum(rec["flops"] for rec in operations.values())
    calls = sum(rec["calls"] for rec in operations.values())
    if billed != total or calls <= 0:
        raise ValueError(f"operation ledger incomplete: {billed} vs {total} FLOPs, {calls} calls")
    return calls


def _one(name, *, width=1024, depth=16):
    if name not in SWITCHES:
        raise ValueError(f"unknown reference {name}")
    if width < 2 or depth < 2:
        raise ValueError("invalid probe shape")
    flops.configure(symmetry_warnings=False)
    source = build(name)
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    mod = load(name)
    weights_np, estimator_seed, _ = canonical_mint(width, depth, PROBE_SEED)
    weights = le.build_weights(weights_np, "f64_readonly", fnp)
    mlp = MLP(weights=weights, width=width, depth=depth, seed=int(estimator_seed))
    est = mod.Estimator()
    context = type("Context", (), {"width": width, "depth": depth,
                                   "flop_budget": DIAGNOSTIC_BUDGET, "seed": 0})()
    est.setup(context)  # grader setup is outside the per-MLP counted window
    with MaterializationCounter() as materializations:
        with flops.BudgetContext(flop_budget=DIAGNOSTIC_BUDGET, quiet=True) as ctx:
            pred = offline_predict(mod, mlp, diagnostic_budget=DIAGNOSTIC_BUDGET,
                                   estimator=est)
            scored = le.materialise_predictions(pred, depth=depth, width=width)
    if est.last_run["mode"] != "full" or est.last_run["generations"] is not None:
        raise RuntimeError(f"capped reference: {est.last_run}")
    scored = np.asarray(scored, dtype=np.float32)
    if scored.shape != (depth, width) or not np.isfinite(scored).all():
        raise RuntimeError("nonfinite or incomplete reference output")
    by_namespace = ctx.summary_dict(by_namespace=True).get("by_namespace") or {}
    operations = {}
    for bucket in by_namespace.values():
        for op, rec in (bucket.get("operations") or {}).items():
            entry = operations.setdefault(op, {"flops": 0, "calls": 0})
            entry["flops"] += int(rec.get("flop_cost", 0))
            entry["calls"] += int(rec.get("calls", 0))
    total_calls = _assert_ledger_conserved(int(ctx.flops_used), operations)
    return {
        "artifact": "s18_same_closure_reference_meter", "status": "complete",
        "arm": name, "width": width, "depth": depth, "input_seed": PROBE_SEED,
        "input_weight_dtype": "f64_readonly", "diagnostic_budget": DIAGNOSTIC_BUDGET,
        "official_budget": OFFICIAL_BUDGET, "deployable_cost_claim": False,
        "source_sha256": source_sha, "control_sha256": _sha(CONTROL),
        "generator_sha256": _sha(HERE / "s18_same_closure_reference.py"),
        "meter_sha256": _sha(__file__),
        "dependency_sha256": {"local_sim/lifecycle_eval.py": _sha(le.__file__),
                              "local_sim/mint_provenance.py": _sha(HERE / "mint_provenance.py")},
        "last_run": est.last_run, "flops_used": int(ctx.flops_used),
        "operation_calls": total_calls, "operations": operations,
        "scalar_materializations": dict(materializations.counts),
        "scalar_materializations_total": materializations.total,
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "final_prediction_sha256": hashlib.sha256(scored[-1].tobytes()).hexdigest(),
    }


def _write_exclusive(path: Path, data):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm", choices=tuple(SWITCHES))
    group.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.arm:
        _write_exclusive(args.out, _one(args.arm))
        print(args.out)
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    for name in SWITCHES:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", name,
               "--out", str(args.out / f"{name}.json")]
        subprocess.run(cmd, check=True)
    report = {"artifact": "s18_same_closure_reference_meter_group", "status": "complete",
              "arms": {name: _sha(args.out / f"{name}.json") for name in SWITCHES}}
    _write_exclusive(args.out / "manifest.json", report)
    print(args.out / "manifest.json")


if __name__ == "__main__":
    main()
