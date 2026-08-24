"""The deployment envelope: what fails a Phase-2 MLP, and what this host can prove.

P2-P0 (Phase 0 of `docs/GAME_PLAN.md`). Contract gate G4: `experiments/LOG.md`,
2026-08-24.

WHY THIS FILE EXISTS
--------------------
`local_sim/lifecycle_eval.py` models the six FAILURE paths and provokes every one
of them it can reach in-process. Three parts of the live envelope it cannot reach
at all, and until today the repository had no record of which was which:

    the 8,192 MB memory limit      -- RLIMIT_AS, set by the subprocess worker
    worker death and restart       -- a process that dies mid-suite
    the predict-response timeout   -- a runner-side wait, not an in-process clock

An evaluator that models three of six paths and says nothing about the other three
does not have a gap; it has a *hidden* gap, and a hidden gap reads as coverage.

THE FINDING THIS FILE EXISTS TO RECORD
--------------------------------------
**This host cannot enforce the memory limit by any route.** `--runner local`
enforces nothing (`runner.py:163-167` logs "advisory in --runner local"), and
`--runner subprocess` enforces it with `resource.setrlimit(RLIMIT_AS, ...)`
(`subprocess_worker.py:172-183`) -- and the `resource` module **does not exist on
Windows**. The worker catches the failure and prints a warning; the run then
proceeds with NO memory cap at all.

So a Phase-2 candidate can be memory-safe on this laptop by construction, because
nothing here is measuring, and still be killed on the grader. The Phase-1 limit
was 65,536 MB and it already killed v2.5 (`docs/GOTCHAS.md`). The Phase-2 limit is
**8,192 MB -- eight times tighter** -- and it applies to a shape with sixteen times
the weight bytes. Memory is therefore a DESIGN constraint to be accounted for, not
a test to be passed, and this file provides the accounting instead of the test.

WHAT IT ACTUALLY MEASURES
-------------------------
1. **The enforceability matrix.** For each envelope constraint: is it enforced
   here, and by what. Derived from the platform and the installed package, never
   asserted from memory. This is the part that must never quietly become "all
   green".
2. **Worker death and restart, provoked for real** through the official
   `whest run --runner subprocess`, using a sacrificial estimator that hard-exits
   on one MLP. The grader restarts the worker and scopes the failure to that MLP
   (`runner.py:359-380`); a suite that scored every remaining MLP as failed would
   turn one bad network into a whole-submission loss.
3. **A peak-memory accounting number** at the live shape, from the Windows
   working-set counter, with its limitation stated: RLIMIT_AS caps *virtual
   address space*, working set measures *resident* memory, and the first is
   normally the larger. The number here is a LOWER BOUND on what the grader's
   limit would see, and is quoted as such or not at all.
4. **The timeout arithmetic**, which is not what it looks like: the runner waits
   `max(predict_timeout_s, wall_time_limit_s + grace)`, so under Phase 2's 120 s
   wall the 30 s `predict_timeout_s` is NOT the binding constraint.

Run:  uv run python local_sim/test_deployment_envelope.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- local_sim/ holds numpy shims named flopscope/whestbench.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import official_config  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0")
OUT = os.path.join(OUT_DIR, "deployment_envelope.json")

_PASS = 0
_FAIL = 0
_UNPROVABLE: list = []


def check(name, ok, detail=""):
    global _PASS, _FAIL
    ok = bool(ok)
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}")
    return ok


def unprovable(name, why):
    """Record a constraint this host CANNOT test. Never counted as a pass."""
    _UNPROVABLE.append({"constraint": name, "why": why})
    print(f"  ----  {name:<62} NOT PROVABLE HERE: {why}")


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


# --------------------------------------------------------------------------- #
# 1. the enforceability matrix
# --------------------------------------------------------------------------- #

def _rlimit_available() -> bool:
    try:
        import resource  # noqa: F401
    except Exception:
        return False
    import resource

    return hasattr(resource, "RLIMIT_AS")


def enforceability() -> dict:
    """Which envelope constraints this host can actually enforce, and how."""
    r = official_config.LIVE
    rl = _rlimit_available()
    return {
        "platform": sys.platform,
        "rlimit_as_available": rl,
        "constraints": {
            "flop_budget": {
                "value": r.flop_budget,
                "enforced_locally": True,
                "by": "flopscope.BudgetContext raises BudgetExhaustedError",
                "provoked_by": "local_sim/lifecycle_eval.py --self-test (L4)",
            },
            "wall_time_limit_s": {
                "value": r.wall_time_limit_s,
                "enforced_locally": True,
                "by": "flopscope.BudgetContext wall clock -> TimeExhaustedError",
                "provoked_by": "local_sim/lifecycle_eval.py --self-test (L4)",
            },
            "residual_wall_time_limit_s": {
                "value": r.residual_wall_time_limit_s,
                "enforced_locally": True,
                "by": "post-predict comparison against BudgetContext residual",
                "provoked_by": "local_sim/lifecycle_eval.py --self-test (L8)",
                "note": ("the LOCAL residual is a different instrument from the "
                         "graded one; the cliff is modelled faithfully, the "
                         "seconds are not transferable"),
            },
            "setup_timeout_s": {
                "value": r.setup_timeout_s,
                "enforced_locally": True,
                "by": "timed module load + setup(), plus HOST_COLD_START_S",
                "provoked_by": "local_sim/lifecycle_eval.py --self-test (4b)",
            },
            "memory_limit_mb": {
                "value": r.memory_limit_mb,
                "enforced_locally": bool(rl),
                "by": ("resource.setrlimit(RLIMIT_AS) in the subprocess worker"
                       if rl else "NOTHING -- the `resource` module does not exist "
                                  "on this platform, so subprocess_worker.py's "
                                  "setrlimit call is caught and skipped, and "
                                  "--runner local is advisory by design"),
                "provoked_by": None,
                "accounting_instead": "peak working set, measured below",
            },
            "worker_death_restart": {
                "value": None,
                "enforced_locally": True,
                "by": "whest run --runner subprocess restarts a dead worker",
                "provoked_by": "this file, section 3",
            },
            "predict_response_timeout_s": {
                "value": r.predict_timeout_s,
                "enforced_locally": False,
                "by": ("runner-side wait of max(predict_timeout_s, "
                       "wall_time_limit_s + grace); under Phase 2's 120 s wall "
                       "the 30 s predict_timeout is NOT the binding constraint"),
                "provoked_by": None,
            },
        },
    }


# --------------------------------------------------------------------------- #
# 2. peak memory, measured -- and labelled with what it is not
# --------------------------------------------------------------------------- #

def peak_memory_bytes() -> dict:
    """Peak memory of THIS process: `{"working_set": ..., "commit": ...}`.

    Two numbers, because they are not the same thing and only one of them is a
    fair analogue of what the grader caps:

      * **working set** -- RESIDENT memory. What `ru_maxrss` reports on POSIX.
      * **commit** (`PeakPagefileUsage`) -- private committed bytes, the closest
        Windows counter to the process's own address-space charge, and therefore
        the closer analogue of what `RLIMIT_AS` bounds.

    `RLIMIT_AS` caps VIRTUAL ADDRESS SPACE, which includes mapped files and every
    reservation, so it is normally larger than either of these. Both numbers here
    are LOWER BOUNDS on what the grader's limit would see and must be quoted as
    such. Missing keys mean the platform would not report them.
    """
    if sys.platform != "win32":
        try:
            import resource

            # ru_maxrss is KiB on Linux, bytes on macOS.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            v = int(rss * 1024) if sys.platform.startswith("linux") else int(rss)
            return {"working_set": v}
        except Exception:
            return {}
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
    if fn is None:  # pre-Win8 it lives in psapi.dll
        try:
            fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        except Exception:
            return {}
    # argtypes/restype are NOT optional here. GetCurrentProcess() returns the
    # pseudo-handle -1; without an explicit HANDLE argtype ctypes passes it as a
    # 32-bit int on a 64-bit process and the call fails silently, returning None
    # for a measurement that looks merely unavailable rather than wrong.
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                   wintypes.DWORD]
    fn.restype = wintypes.BOOL
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not fn(handle, ctypes.byref(counters), counters.cb):
        return {}
    return {
        "working_set": int(counters.PeakWorkingSetSize),
        "commit": int(counters.PeakPagefileUsage),
    }


def measure_live_shape_memory() -> dict:
    """Run one `predict` at the live shape in a FRESH process and report its peak.

    A fresh process is the point: measuring in an interpreter that has already run
    a test suite reports the suite's peak, not the candidate's.
    """
    src = textwrap.dedent(
        """
        import json, os, sys
        _ROOT = sys.argv[1]
        sys.path = [p for p in sys.path
                    if os.path.abspath(p or ".") != os.path.join(_ROOT, "local_sim")]
        for p in (os.path.join(_ROOT, "local_sim"), _ROOT):
            if p not in sys.path:
                sys.path.append(p)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from test_deployment_envelope import peak_memory_bytes
        from mint_provenance import canonical_mint
        import official_config, harness, importlib.util as u

        r = official_config.LIVE
        base = peak_memory_bytes()
        w, seed, _ = canonical_mint(r.width, r.depth, 5_150_900)
        mlp = harness.MLP(weights=[fnp.array(x) for x in w],
                          width=r.width, depth=r.depth, seed=seed)
        spec = u.spec_from_file_location("cand", os.path.join(_ROOT, "estimator.py"))
        mod = u.module_from_spec(spec); spec.loader.exec_module(mod)
        est = mod.Estimator()
        class C:
            width, depth, flop_budget, seed = r.width, r.depth, r.flop_budget, 0
        try:
            est.setup(C())
        except Exception:
            pass
        with flops.BudgetContext(flop_budget=r.flop_budget, quiet=True) as ctx:
            est.predict(mlp, r.flop_budget)
        print(json.dumps({"baseline": base,
                          "peak": peak_memory_bytes(),
                          "weight_bytes": int(sum(x.nbytes for x in w))}))
        """
    )
    tmp = os.path.join(_HERE, "_envelope_mem_tmp.py")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(src)
        p = subprocess.run([sys.executable, tmp, _ROOT], capture_output=True,
                           text=True, cwd=_ROOT)
        if p.returncode != 0:
            return {"error": p.stderr[-2000:]}
        return json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# 3. worker death and restart, through the OFFICIAL subprocess runner
# --------------------------------------------------------------------------- #

_SACRIFICIAL = '''"""A deliberately fatal estimator: hard-exits the worker EXACTLY ONCE.

Used by local_sim/test_deployment_envelope.py to provoke the grader's worker
restart path. `os._exit` skips every finally/atexit hook, which is the closest
reachable analogue of an OOM kill or a segfault.

The "exactly once" is the whole difficulty, and getting it wrong makes the runner
look broken. A module-global call counter does NOT work: the replacement worker
imports this file fresh, its counter starts at 0 again, and it kills itself on ITS
first MLP too -- so every MLP in the suite dies and the runner appears not to
recover at all. The state has to outlive the process, so it lives in a SENTINEL
FILE whose path arrives by environment variable.
"""
import os

import flopscope.numpy as fnp
from whestbench.sdk import BaseEstimator

_SENTINEL = os.environ.get("ARC_WHEST_SACRIFICE_SENTINEL", "")


class Estimator(BaseEstimator):
    """Must SUBCLASS BaseEstimator -- the loader discovers candidates by subclass,
    not by name, and refuses a file with none.
    """

    def setup(self, context):
        pass

    def predict(self, mlp, budget):
        if _SENTINEL and not os.path.exists(_SENTINEL):
            with open(_SENTINEL, "w", encoding="utf-8") as fh:
                fh.write(str(int(mlp.seed)))
            os._exit(9)
        return fnp.zeros((len(mlp.weights), mlp.width), dtype=fnp.float32)
'''


def _whest_exe() -> str:
    """The `whest` console script beside the running interpreter."""
    d = os.path.dirname(sys.executable)
    for name in ("whest.exe", "whest"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return "whest"


def provoke_worker_death(n_mlps: int = 3) -> dict:
    """Run the official subprocess runner over a candidate that kills its worker."""
    cand = os.path.join(_HERE, "_envelope_sacrificial_tmp.py")
    sentinel = os.path.join(_HERE, "_envelope_sacrifice_sentinel_tmp")
    try:
        if os.path.exists(sentinel):
            os.remove(sentinel)
        with open(cand, "w", encoding="utf-8") as fh:
            fh.write(_SACRIFICIAL)
        env = dict(os.environ, PYTHONIOENCODING="utf-8",
                   ARC_WHEST_SACRIFICE_SENTINEL=sentinel)
        # The `whest` console script, not `python -m whestbench.cli`: the module
        # has no __main__ guard, so `-m` exits 0 having done nothing, which reads
        # as an empty report rather than as an error.
        p = subprocess.run(
            [_whest_exe(), "run",
             "--estimator", cand,
             "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
             "--split", "mini", "--runner", "subprocess",
             "--n-mlps", str(n_mlps), "--json"],
            capture_output=True, text=True, cwd=_ROOT, env=env,
        )
        out = p.stdout
        i = out.find("{")
        if i < 0:
            return {"error": (p.stderr or out)[-3000:]}
        return {"report": json.loads(out[i:])}
    finally:
        for p in (cand, sentinel):
            if os.path.exists(p):
                os.remove(p)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-worker-death", action="store_true",
                    help="skip section 3 (it runs the official runner, ~30 s)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    r = official_config.LIVE
    print("=" * 96)
    print("DEPLOYMENT ENVELOPE -- what fails a Phase-2 MLP, and what this host can prove")
    print("=" * 96)

    head("1. the enforceability matrix")
    enf = enforceability()
    print(f"  platform {enf['platform']}   RLIMIT_AS available: "
          f"{enf['rlimit_as_available']}")
    for name, c in enf["constraints"].items():
        if c["enforced_locally"]:
            check(f"{name} is enforced here", True, str(c["value"]))
        else:
            unprovable(name, c["by"])
    check("the matrix is honest about memory on this platform",
          enf["constraints"]["memory_limit_mb"]["enforced_locally"]
          == _rlimit_available(),
          f"enforced_locally={enf['constraints']['memory_limit_mb']['enforced_locally']}")
    check("at least one constraint is recorded as NOT provable here",
          len(_UNPROVABLE) > 0,
          f"{len(_UNPROVABLE)} unprovable: "
          f"{[u['constraint'] for u in _UNPROVABLE]}")

    head("2. the timeout arithmetic is not what it looks like")
    from whestbench.runner import ResourceLimits, _predict_response_timeout_s
    lim = ResourceLimits(
        flop_budget=r.flop_budget,
        predict_timeout_s=r.predict_timeout_s,
        memory_limit_mb=r.memory_limit_mb,
        setup_timeout_s=r.setup_timeout_s,
        wall_time_limit_s=r.wall_time_limit_s,
        residual_wall_time_limit_s=r.residual_wall_time_limit_s,
    )
    eff = _predict_response_timeout_s(lim)
    check("the runner waits max(predict_timeout, wall + grace), not predict_timeout",
          eff > r.predict_timeout_s,
          f"{eff:.1f} s effective vs predict_timeout_s {r.predict_timeout_s:.1f} s")
    check("so under Phase 2 the 120 s WALL binds, not the 30 s predict timeout",
          eff >= r.wall_time_limit_s,
          f"wall {r.wall_time_limit_s:.0f} s")

    head("3. peak memory at the live shape -- accounting, not enforcement")
    mem = measure_live_shape_memory()
    mem_report: dict = {"measurement": mem, "limit_mb": r.memory_limit_mb}
    if "error" in mem or not mem.get("peak"):
        check("a peak-memory measurement was obtained", False,
              str(mem.get("error", mem))[:200])
    else:
        MB = 1024 * 1024
        peak = mem["peak"]
        ws_mb = peak["working_set"] / MB
        # `commit` is the closer analogue of what RLIMIT_AS bounds; fall back to
        # working set where the platform does not report it.
        binding_mb = peak.get("commit", peak["working_set"]) / MB
        weights_mb = mem["weight_bytes"] / MB
        headroom = r.memory_limit_mb / binding_mb
        mem_report.update({"peak_working_set_mb": ws_mb,
                           "peak_commit_mb": binding_mb,
                           "weights_mb": weights_mb,
                           "headroom_lower_bound": headroom})
        print(f"  weights alone            {weights_mb:>10.1f} MB "
              f"({r.depth} x {r.width}x{r.width} float32)")
        print(f"  peak WORKING SET         {ws_mb:>10.1f} MB   (resident)")
        print(f"  peak COMMIT              {binding_mb:>10.1f} MB   "
              f"(closest analogue of the RLIMIT_AS charge)")
        print(f"  Phase-2 limit            {r.memory_limit_mb:>10.1f} MB")
        check("the shipped file's peak memory fits the Phase-2 limit",
              binding_mb < r.memory_limit_mb,
              f"{headroom:.1f}x headroom -- a LOWER BOUND. RLIMIT_AS caps VIRTUAL "
              f"address space, which is larger than either counter above")
        check("...and the peak is at least as large as the weights alone",
              binding_mb > weights_mb,
              f"{binding_mb:.1f} MB > {weights_mb:.1f} MB (a peak below the "
              f"weights would mean the probe never materialised them)")

    death: dict = {"skipped": True}
    if args.skip_worker_death:
        head("4. worker death and restart -- SKIPPED by request")
        unprovable("worker_death_restart", "--skip-worker-death was passed")
    else:
        head("4. worker death and restart, through the OFFICIAL subprocess runner")
        d = provoke_worker_death(3)
        if "error" in d:
            check("the official runner produced a report", False, d["error"][:300])
            death = {"error": d["error"][:2000]}
        else:
            res = d["report"]["results"]
            rows = res["per_mlp"]
            failed = [i for i, x in enumerate(rows)
                      if x["budget_exhausted"] or x["time_exhausted"]
                      or x["residual_wall_time_exhausted"]
                      or x["combined_budget_exhausted"] or x["traceback"]]
            death = {
                "skipped": False,
                "n_mlps": len(rows),
                "n_failed_mlps": res["n_failed_mlps"],
                "failed_indices": failed,
                "failure_breakdown": res.get("failure_breakdown"),
            }
            check("the suite SURVIVED a worker that hard-exited mid-run",
                  len(rows) == 3, f"{len(rows)} rows returned")
            check("the failure is SCOPED to the MLP that killed the worker",
                  res["n_failed_mlps"] <= 1,
                  f"n_failed_mlps = {res['n_failed_mlps']}, indices {failed}")
            # "flops_used" is a weak witness here -- the sacrificial estimator
            # returns fnp.zeros, which costs nothing, so 0 is the CORRECT count and
            # proves nothing. What proves the row ran is that it carries a real
            # MSE and no failure flag of any kind.
            last = rows[-1]
            check("...and the MLPs after it were actually SCORED, not zeroed",
                  len(rows) >= 2 and not last["traceback"]
                  and not any(last[k] for k in ("budget_exhausted", "time_exhausted",
                                                "residual_wall_time_exhausted",
                                                "combined_budget_exhausted"))
                  and last["final_layer_mse"] > 0.0,
                  f"last row mse {last['final_layer_mse']:.6e}, no failure flag")

    print("\n" + "=" * 96)
    print(f"{_PASS}/{_PASS + _FAIL} checks passed; "
          f"{len(_UNPROVABLE)} constraint(s) NOT PROVABLE on this host")
    if _UNPROVABLE:
        print("\nNOT PROVABLE HERE -- these are gaps in the local evaluator, and a")
        print("candidate that passes everything above has NOT been shown to satisfy")
        print("them. They are enforced only on the grader:")
        for u in _UNPROVABLE:
            print(f"  - {u['constraint']}: {u['why']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "artifact": "deployment_envelope",
            "artifact_version": 1,
            "round": r.tag,
            "enforceability": enf,
            "unprovable_here": _UNPROVABLE,
            "predict_response_timeout_s": eff,
            "memory": mem_report,
            "worker_death": death,
            "checks_passed": _PASS,
            "checks_total": _PASS + _FAIL,
        }, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
