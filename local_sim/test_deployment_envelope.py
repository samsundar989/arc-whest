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
**Windows cannot enforce the memory limit by any route.** `--runner local`
enforces nothing (`runner.py:163-167` logs "advisory in --runner local"), and
`--runner subprocess` enforces it with `resource.setrlimit(RLIMIT_AS, ...)`
(`subprocess_worker.py:172-183`) -- and the `resource` module **does not exist on
Windows**. The worker catches the failure and prints a warning; the run then
proceeds with NO memory cap at all.

**So this file reports differently on the two platforms, and that is the point.**
Where `RLIMIT_AS` exists it PROVOKES the limit and proves the path fires; where it
does not, it says NOT PROVABLE and points at the Linux evidence. A gate clause that
asks for a provocation is not satisfied by documenting why the provocation is
impossible -- the documentation was honest and it was still not the check.

Provoked on Ubuntu 24.04 / WSL2 under the same official stack, 2026-08-24
(`experiments/2026-08-24_p2_p0/linux/`): the worker reported
`RLIMIT_AS soft = 8,589,934,592` (exactly 8 GiB, so the cap really was applied), a
256 MiB control mapping SUCCEEDED, a 9 GiB anonymous reservation was REFUSED with
`OSError: [Errno 12] Cannot allocate memory` without a byte being touched, that MLP
was scored through the error path, and **the next two MLPs ran and were scored**.

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
3. **A peak-memory accounting number** at the live shape, with its limitation
   stated and its direction stated correctly. RLIMIT_AS caps *virtual address
   space*; the counters here measure *resident* and *committed* memory, both of
   which are smaller. So the measured peak is a **lower bound on the charged
   quantity**, and the headroom derived from it is therefore an **UPPER bound** --
   the most optimistic reading, never a safety margin. (An earlier revision of
   this file called that headroom a lower bound. It is not, and inverting the
   sign of the one constraint this host cannot test is the worst place to do it.)
4. **The timeout arithmetic**, which is not what it looks like: the runner waits
   `max(predict_timeout_s, wall_time_limit_s + grace)`, so under Phase 2's 120 s
   wall the 30 s `predict_timeout_s` is NOT the binding constraint.

Run:  uv run python local_sim/test_deployment_envelope.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


def _resolve_witness(witness):
    """Does a `provoked_by` string name something that exists?

    Returns True when the named file is present, False when it names a path that
    is not, and None when the claim is prose this function cannot adjudicate (for
    example "this file, section 6"). None is treated as acceptable -- the point is
    to catch a claim pointing at nothing, not to parse English.
    """
    if not witness:
        return False
    text = str(witness)
    hits = re.findall(r"[\w./\\-]+\.(?:py|json|txt|md)|experiments/[\w./\\-]+", text)
    if not hits:
        return None
    for h in hits:
        if not os.path.exists(os.path.join(_ROOT, h.replace("\\", "/").rstrip("/"))):
            return False
    return True


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
                # Until 2026-08-24 this named L4 while L4 provoked only the
                # RESIDUAL limit. The wall provocation now exists beside it --
                # `dataclasses.replace(spec, wall_time_limit_s=1e-9)` -- and
                # asserts time_exhausted set with residual_exhausted clear, so
                # the two paths are shown to be distinct rather than assumed to be.
                "provoked_by": "local_sim/lifecycle_eval.py --self-test (L4, wall)",
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
                "provoked_by": ("this file, section 5" if rl else
                                "experiments/2026-08-24_p2_p0/linux/ -- provoked on "
                                "Ubuntu 24.04 under the same official stack, since "
                                "it cannot be provoked on this platform"),
                "accounting_instead": ("peak working set and peak commit, measured "
                                       "below; both are LOWER bounds on the "
                                       "RLIMIT_AS charge, so the headroom they "
                                       "imply is an UPPER bound"),
            },
            "worker_death_restart": {
                "value": None,
                "enforced_locally": True,
                "by": "whest run --runner subprocess restarts a dead worker",
                "provoked_by": "this file, section 6",
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


#: The sacrificial estimator that provokes RLIMIT_AS. Kept as a repository FILE
#: rather than an embedded string so the Linux transcript and this suite run the
#: byte-identical thing -- two copies of a probe drift, and then the transcript
#: stops being evidence about the code that ships.
MEMORY_SACRIFICE = os.path.join(
    _ROOT, "experiments", "2026-08-24_p2_p0", "linux", "sacrificial_memory.py")

#: Recorded on Ubuntu 24.04 / WSL2, whestbench 0.16.0 / flopscope 0.12.0, 2026-08-24.
#: Cited on platforms that cannot run the provocation themselves.
LINUX_EVIDENCE = os.path.join(
    _ROOT, "experiments", "2026-08-24_p2_p0", "linux",
    "rlimit_as_worker_report.json")


def provoke_memory_limit(n_mlps: int = 3, request_gib: float = 9.0) -> dict:
    """Reserve `request_gib` of ADDRESS SPACE under the 8 GiB cap, and never touch it.

    `RLIMIT_AS` bounds virtual address space, so the kernel refuses the mapping
    outright -- no page is faulted in and no memory is filled. Requesting address
    space is precisely what the limit governs, which makes this the cheapest
    honest way to exercise it.

    Returns the runner's report plus the worker's own view of its limit, written
    before either mapping is attempted so the cap is proven applied rather than
    assumed.
    """
    sentinel = os.path.join(_HERE, "_envelope_mem_sentinel_tmp")
    report = os.path.join(_HERE, "_envelope_mem_report_tmp.json")
    for f in (sentinel, report):
        if os.path.exists(f):
            os.remove(f)
    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        ARC_WHEST_MEM_SENTINEL=sentinel,
        ARC_WHEST_MEM_REPORT=report,
        ARC_WHEST_MEM_REQUEST=str(int(request_gib * 1024 ** 3)),
        ARC_WHEST_MEM_CONTROL=str(256 * 1024 ** 2),
    )
    try:
        p = subprocess.run(
            [_whest_exe(), "run", "--estimator", MEMORY_SACRIFICE,
             "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
             "--split", "mini", "--runner", "subprocess",
             "--n-mlps", str(n_mlps), "--json"],
            capture_output=True, text=True, cwd=_ROOT, env=env)
        out: dict = {}
        i = p.stdout.find("{")
        if i < 0:
            return {"error": (p.stderr or p.stdout)[-2000:]}
        doc = json.loads(p.stdout[i:])
        if not doc.get("results"):
            return {"refused": doc.get("error") or doc}
        out["report"] = doc
        if os.path.exists(report):
            with open(report, encoding="utf-8") as fh:
                out["worker"] = json.load(fh)
        return out
    finally:
        for f in (sentinel, report):
            if os.path.exists(f):
                os.remove(f)


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
        doc = json.loads(out[i:])
        if not doc.get("results"):
            # The runner REFUSED the whole run -- typically SETUP_TIMEOUT, which
            # is an environment condition and not an estimator defect. Return it
            # as a refusal rather than a report, because the caller reading
            # `doc["results"]` would otherwise die with a KeyError and report an
            # environment hiccup as a crashed test suite. That happened once.
            return {"refused": doc.get("error") or doc}
        return {"report": doc}
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
    ap.add_argument("--no-write", action="store_true",
                    help=("run the checks and write NOTHING. An auditor re-running "
                          "this file should not have to `git checkout` the "
                          "committed artifact afterwards -- which is exactly what "
                          "an independent Skeptic had to do, and it is the kind of "
                          "side effect that makes a read-only protocol untrue."))
    args = ap.parse_args()

    r = official_config.LIVE
    print("=" * 96)
    print("DEPLOYMENT ENVELOPE -- what fails a Phase-2 MLP, and what this host can prove")
    print("=" * 96)

    head("1. the enforceability matrix")
    enf = enforceability()
    print(f"  platform {enf['platform']}   RLIMIT_AS available: "
          f"{enf['rlimit_as_available']}")
    # A CONDITION, not the literal `True`. This loop used to read
    #     check(f"{name} is enforced here", True, str(c["value"]))
    # which cannot fail, and it counted FIVE constraints as passing -- including
    # `wall_time_limit_s`, which at the time was provoked NOWHERE in the
    # repository, while the artifact recorded `provoked_by:
    # "local_sim/lifecycle_eval.py --self-test (L4)"`. L4 set a tiny RESIDUAL
    # limit and never a tiny WALL limit: two different constants, one untested,
    # and a green line asserting otherwise. Found by a final Skeptic pass after
    # two earlier reviews of the same slice missed it, and it is this
    # repository's signature failure class (docs/GOTCHAS.md).
    #
    # What is checked now is the thing that was being assumed: a constraint
    # claiming local enforcement must NAME a provocation, and that name must
    # resolve to a file that exists. A claim with no witness fails here.
    for name, c in enf["constraints"].items():
        if not c["enforced_locally"]:
            unprovable(name, c["by"])
            continue
        witness = c.get("provoked_by")
        resolved = _resolve_witness(witness)
        check(f"{name} is enforced here, and NAMES its provocation",
              bool(witness) and resolved is not False,
              f"{c['value']}  <- {witness or 'NO PROVOCATION NAMED'}"
              + ("" if resolved is not False else "   *** witness not found ***"))
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
        # SIGN, and an independent Skeptic caught it pointing the wrong way.
        # `peak_commit` is a LOWER bound on the quantity RLIMIT_AS charges (that
        # bounds VIRTUAL ADDRESS SPACE, which is larger). Dividing a fixed limit by
        # a LOWER bound on the denominator therefore yields an UPPER bound on the
        # headroom -- the most OPTIMISTIC reading available, not the floor.
        # Labelling it a lower bound inverted the safety direction of the one
        # constraint this host cannot test at all.
        headroom_ub = r.memory_limit_mb / binding_mb
        mem_report.update({"peak_working_set_mb": ws_mb,
                           "peak_commit_mb": binding_mb,
                           "weights_mb": weights_mb,
                           "headroom_upper_bound": headroom_ub})
        print(f"  weights alone            {weights_mb:>10.1f} MB "
              f"({r.depth} x {r.width}x{r.width} float32)")
        print(f"  peak WORKING SET         {ws_mb:>10.1f} MB   (resident)")
        print(f"  peak COMMIT              {binding_mb:>10.1f} MB   "
              f"(a LOWER bound on the RLIMIT_AS charge)")
        print(f"  Phase-2 limit            {r.memory_limit_mb:>10.1f} MB")
        check("the shipped file's peak memory fits the Phase-2 limit",
              binding_mb < r.memory_limit_mb,
              f"at most {headroom_ub:.1f}x headroom -- an UPPER BOUND, i.e. the "
              f"most optimistic reading. RLIMIT_AS caps VIRTUAL address space, "
              f"which exceeds both counters above by an unmeasured amount")
        check("...and the peak is at least as large as the weights alone",
              binding_mb > weights_mb,
              f"{binding_mb:.1f} MB > {weights_mb:.1f} MB (a peak below the "
              f"weights would mean the probe never materialised them)")

    # ---------------------------------------------------------------- #
    # 5. the memory limit -- PROVOKED where the limit exists
    # ---------------------------------------------------------------- #
    mem_gate: dict = {}
    head("5. the 8 GiB memory limit -- provoked, or explicitly not provable here")
    if not _rlimit_available():
        unprovable(
            "memory_limit_mb (provocation)",
            "no RLIMIT_AS on this platform; provoked instead on Ubuntu 24.04 "
            "under the same official stack -- see "
            "experiments/2026-08-24_p2_p0/linux/")
        if os.path.exists(LINUX_EVIDENCE):
            with open(LINUX_EVIDENCE, encoding="utf-8") as fh:
                lin = json.load(fh)
            mem_gate = {"provoked_here": False, "linux_evidence": lin}
            print(f"      Linux worker saw RLIMIT_AS soft = "
                  f"{lin.get('rlimit_as_soft'):,} "
                  f"({lin.get('rlimit_as_soft_gib')} GiB)")
            print(f"      control {lin.get('control_mmap')}   "
                  f"9 GiB request -> {lin.get('provocation_mmap')}")
            check("the Linux evidence shows the cap was actually APPLIED",
                  lin.get("rlimit_as_soft") == r.memory_limit_mb * 1024 * 1024,
                  f"{lin.get('rlimit_as_soft'):,} == "
                  f"{r.memory_limit_mb * 1024 * 1024:,}")
            check("...the control mapping SUCCEEDED, so the mechanism works",
                  lin.get("control_mmap") == "OK", str(lin.get("control_mmap")))
            check("...and the over-limit reservation was REFUSED",
                  bool(lin.get("provocation_refused")),
                  str(lin.get("provocation_mmap")))
        else:
            mem_gate = {"provoked_here": False, "linux_evidence": None}
            check("Linux evidence for the memory path is present", False,
                  f"missing {os.path.relpath(LINUX_EVIDENCE, _ROOT)}")
    else:
        d = provoke_memory_limit()
        if "error" in d:
            check("the official runner produced a report", False,
                  d["error"][:300])
            mem_gate = {"provoked_here": True, "error": d["error"][:2000]}
        elif "refused" in d:
            mem_gate = {"provoked_here": True, "refused": d["refused"]}
            check("the official runner accepted the run", False,
                  f"REFUSED: {(d['refused'] or {}).get('code', '?')} -- an "
                  f"environment condition; re-run on a quiet host")
        else:
            res = d["report"]["results"]
            rows = res["per_mlp"]
            w = d.get("worker") or {}
            mem_gate = {"provoked_here": True, "worker": w,
                        "n_failed_mlps": res["n_failed_mlps"],
                        "failure_breakdown": res.get("failure_breakdown")}
            print(f"      worker saw RLIMIT_AS soft = {w.get('rlimit_as_soft'):,} "
                  f"({w.get('rlimit_as_soft_gib')} GiB)")
            check("the worker actually APPLIED the 8 GiB cap",
                  w.get("rlimit_as_soft") == r.memory_limit_mb * 1024 * 1024,
                  f"{w.get('rlimit_as_soft'):,} == "
                  f"{r.memory_limit_mb * 1024 * 1024:,}")
            check("the 256 MiB control mapping SUCCEEDED",
                  w.get("control_mmap") == "OK", str(w.get("control_mmap")))
            check("the 9 GiB reservation was REFUSED by the limit",
                  bool(w.get("provocation_refused")),
                  str(w.get("provocation_mmap")))
            check("...and it was refused, never filled -- no page was touched",
                  "Cannot allocate memory" in str(w.get("provocation_mmap"))
                  or "MemoryError" in str(w.get("provocation_mmap")),
                  "address space refused at mmap time")
            check("exactly ONE MLP failed, through the error path",
                  res["n_failed_mlps"] == 1
                  and (res.get("failure_breakdown") or {}).get("error") == 1,
                  f"n_failed_mlps={res['n_failed_mlps']} "
                  f"breakdown={res.get('failure_breakdown')}")
            later = rows[1:]
            check("...and every LATER MLP ran and was scored",
                  len(later) >= 1 and all(not m["traceback"] for m in later)
                  and all(m["final_layer_mse"] > 0 for m in later),
                  f"{len(later)} later MLPs, mse "
                  f"{[round(m['final_layer_mse'], 4) for m in later]}")

    death: dict = {"skipped": True}
    if args.skip_worker_death:
        head("6. worker death and restart -- SKIPPED by request")
        unprovable("worker_death_restart", "--skip-worker-death was passed")
    else:
        head("6. worker death and restart, through the OFFICIAL subprocess runner")
        d = provoke_worker_death(3)
        if "error" in d:
            check("the official runner produced a report", False, d["error"][:300])
            death = {"error": d["error"][:2000]}
        elif "refused" in d:
            code = (d["refused"] or {}).get("code", "?")
            death = {"refused": d["refused"]}
            check("the official runner accepted the run", False,
                  f"REFUSED at {(d['refused'] or {}).get('stage', '?')}: {code}")
            if code == "SETUP_TIMEOUT":
                print("      SETUP_TIMEOUT is an ENVIRONMENT condition, not an")
                print("      estimator defect: this host's `import whestbench`")
                print("      alone is 1.3 s quiet and 4.7-9.1 s loaded against a")
                print("      5 s ceiling, while our own setup is 0.6 ms. Re-run on")
                print("      a quiet machine. It is reported as a FAILURE and not")
                print("      a skip because a gate that cannot verify deployment")
                print("      must not certify it.")
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

    if args.no_write:
        print("\n  --no-write: the artifact on disk was NOT touched.")
        return 0 if _FAIL == 0 else 1

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
            "memory_limit_provocation": mem_gate,
            "worker_death": death,
            "checks_passed": _PASS,
            "checks_total": _PASS + _FAIL,
        }, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
