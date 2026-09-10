"""The 8,192 MB limit, ACTUALLY ENFORCED — on Linux, in a child, with a control.

Contract: `experiments/LOG.md`, *CONTRACT WP0C+WP0D* (2026-08-31), committed at
`71bf62b` before this file existed. `GAME_PLAN.md` § WP0.4.

WHY THIS EXISTS
---------------
`docs/CURRENT_STATE.md` records the same constraint twice, and the pair is the
whole problem:

  * *"the 8,192 MB limit — the one constraint this host cannot test at all — is
    also the one with the weakest provenance"*;
  * *"Windows has no `resource` module, so the subprocess worker's `RLIMIT_AS`
    call is skipped and `--runner local` is advisory by design."*

Every memory figure this repository holds is therefore **accounting**: a peak
working set, a peak commit, an admitted upper bound. `RLIMIT_AS` caps virtual
address space, which exceeds both counters by an amount no Windows counter
reports — so the true headroom is SMALLER than the recorded `8.9x` by an
unmeasured factor, and "8.9x headroom" is the optimistic reading of a number
nobody has ever tested.

WHAT THE GRADER ACTUALLY DOES, AND WHY "NO EXCEPTION" IS NOT EVIDENCE
--------------------------------------------------------------------
`whestbench/subprocess_worker.py` calls `resource.setrlimit(RLIMIT_AS, …)`,
catches `(ImportError, ValueError, OSError, AttributeError)`, **warns to stderr
and runs on uncapped**. So a child that raises nothing may simply never have
been capped. This module therefore does two things no previous probe here did:

1. it **reads the applied soft limit back** with `getrlimit` after the call, and
   compares it to the requested cap;
2. it runs a **NEGATIVE CONTROL** — a child under the same cap that deliberately
   requests more than 8 GiB. If that child survives, the cap is not in force and
   **every pass this module printed is withdrawn**. A memory gate without a
   control is a gate that reports a state.

A **POSITIVE CONTROL** runs too: the same over-allocation with no cap must
succeed, which is what distinguishes "the cap killed it" from "this machine
cannot allocate 9 GiB".

HOW IT REACHES LINUX
--------------------
Native Linux if `resource.RLIMIT_AS` exists here. Otherwise WSL: this host has
`Ubuntu` (kernel 6.6.87.2-microsoft-standard-WSL2, Python 3.12.3) with the
repository visible at `/mnt/c/…`, and `scripts/linux_stack_setup.sh` builds a
Linux venv carrying the OFFICIAL Phase-2 stack — whestbench 0.16.0 and
flopscope 0.12.0, the same pins as `pyproject.toml`.

**A WSL kernel is not the grading server.** What transfers is the enforcement
mechanism (`RLIMIT_AS` on Linux) and the candidate's demand under it. What does
not transfer is any absolute second, and the grader's own configuration remains
unobserved — `memory_limit_mb` appears in no `run_config` this repository holds
and in no graded report.

Run:
    uv run python local_sim/linux_memory_gate.py --candidate estimator.py
    uv run python local_sim/linux_memory_gate.py --candidate estimator.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import datetime as _dt
import hashlib
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

#: The installed-package default. Read, not typed: `official_config` asserts it
#: against `whestbench.scoring.ContestSpec`.
DEFAULT_LIMIT_MB = 8192
DEFAULT_WIDTH = 1024
DEFAULT_DEPTH = 16
DEFAULT_BUDGET = 2 ** 41
DEFAULT_SUITE_N_MLPS = 100
#: An arbitrary seed OUTSIDE every registered block (`local_sim/seed_blocks.py`).
#: This gate measures memory, not accuracy; it claims no block and reads no truth.
DEFAULT_SEED = 90000000101

#: The official Phase-2 counted-FLOP anchor for the SHIPPED file at 1024x16,
#: reconciled op by op against a real `whest run --profile --json` report by
#: `local_sim/p2_flop_anchor.py --gate` (delta zero, 19 operations). Used here
#: only as a cross-host agreement check on `estimator.py`; a candidate has no
#: anchor by definition.
ANCHOR_FLOPS_1024x16 = 137_600_326_656
LIFECYCLE_GROWTH_THRESHOLD_BYTES = 64 * (1 << 20)
LIFECYCLE_CONTROL_MIB = 96

LINUX_VENV = "$HOME/.arcwhest-linux"
WSL_DISTRO = "Ubuntu"


# --------------------------------------------------------------------------- #
# the child programs
# --------------------------------------------------------------------------- #

CHILD_PREDICT = r'''
import gc, json, os, resource, sys, time, warnings
from importlib.metadata import version as distribution_version

LIMIT_MB     = int(sys.argv[1])
REPO         = sys.argv[2]
CANDIDATE    = sys.argv[3]
WIDTH        = int(sys.argv[4])
DEPTH        = int(sys.argv[5])
BUDGET       = int(sys.argv[6])
SEED         = int(sys.argv[7])
# artifact_version 3, 2026-09-10. Before this the child hard-coded
# `fnp.array(x)` on the mint's float32 arrays -- a WRITABLE float32 arm, which
# is not the deployment contract (`lifecycle_eval.DEFAULT_WEIGHTS_MODE` is
# "f64_readonly"). A memory peak measured on float32 weights is a peak for a
# different program: the f64 inputs are twice the bytes and a candidate that
# casts them carries both. REPEATS runs predict() more than once in the SAME
# worker, because one prediction cannot show a cache that grows across a suite.
WEIGHTS_MODE = sys.argv[8] if len(sys.argv) > 8 else "f64_readonly"
REPEATS      = int(sys.argv[9]) if len(sys.argv) > 9 else 2

out = {"stage": "start", "limit_mb": LIMIT_MB}

# --- 1. APPLY THE CAP EXACTLY AS whestbench/subprocess_worker.py DOES ------
L = LIMIT_MB * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (L, L))
    out["setrlimit_raised"] = None
except Exception as e:
    out["setrlimit_raised"] = type(e).__name__ + ": " + str(e)[:120]
# ...and READ IT BACK. `setrlimit` not raising is not evidence it applied.
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
out["soft_limit_bytes"] = soft
out["hard_limit_bytes"] = hard
out["cap_requested_bytes"] = L
out["cap_applied"] = (soft == L)

if not out["cap_applied"]:
    out["stage"] = "cap_not_applied"
    print(json.dumps(out)); sys.exit(3)

# --- 2. the official stack, under the cap ---------------------------------
# APPEND, never insert: local_sim/ ships numpy shims NAMED flopscope and
# whestbench, and inserting would silently swap the real packages for them.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in
            (os.path.join(REPO, "local_sim"),)]
for p in (REPO, os.path.join(REPO, "local_sim")):
    if p not in sys.path:
        sys.path.append(p)

import flopscope as flops
import flopscope.numpy as fnp
import whestbench
import numpy as np

# The shipped candidate intentionally performs operations whose symmetry tag is
# dropped by flopscope. The warning is known, typed, and unrelated to memory.
# Suppress only that class; every other stderr byte remains a failed v5 gate.
warnings.filterwarnings("ignore", category=flops.SymmetryLossWarning)

out["whestbench_file"] = whestbench.__file__
out["flopscope_file"] = flops.__file__
if "local_sim" in whestbench.__file__ or "local_sim" in flops.__file__:
    out["stage"] = "shim_leak"
    print(json.dumps(out)); sys.exit(4)
out["numpy_version"] = np.__version__
out["whestbench_version"] = distribution_version("whestbench")
out["flopscope_version"] = distribution_version("flopscope")
out["stack_version_source"] = "importlib.metadata.version"
out["python_version"] = sys.version.split()[0]
out["weights_mode"] = WEIGHTS_MODE
out["repeats"] = REPEATS
out["suite_n_mlps"] = 100

from mint_provenance import canonical_mint
from whestbench import MLP

# Do not call flops.configure here. Under flopscope-client it is deliberately a
# no-op and emits ConfigureNoOpWarning to stderr. The memory gate treats every
# unexpected child stderr byte as an instrument failure, so invoking a known
# no-op merely to suppress in-process symmetry warnings would invalidate the
# evidence it is trying to collect.

# --- 3. build the network OUTSIDE the budget window, as the grader does ----
# ONE owner of the weight representation. `lifecycle_eval.build_weights` is it;
# this file used to construct its own and drifted from the deployment contract
# without anything noticing.
from lifecycle_eval import build_weights, WEIGHTS_MODES
if WEIGHTS_MODE not in WEIGHTS_MODES:
    out["stage"] = "bad_weights_mode"
    print(json.dumps(out)); sys.exit(5)
w_np, est_seed, _ = canonical_mint(WIDTH, DEPTH, SEED)
_w = build_weights(w_np, WEIGHTS_MODE, fnp)
del w_np
out["weight_dtype"] = str(getattr(_w[0], "dtype", ""))
_probe = None
try:
    _probe = np.asarray(_w[0])
    out["weight_writeable"] = bool(_probe.flags.writeable)
except Exception:
    out["weight_writeable"] = None
del _probe
mlp = MLP(weights=_w, width=WIDTH, depth=DEPTH, seed=est_seed)
out["stage"] = "mlp_built"

# --- 4. setup() once, OFF budget, timed -----------------------------------
import importlib.util as _iu
spec = _iu.spec_from_file_location("candidate_under_test", CANDIDATE)
mod = _iu.module_from_spec(spec)
t0 = time.perf_counter()
spec.loader.exec_module(mod)
est = mod.Estimator()

class _Ctx:
    def __init__(self, w, d, b, s):
        self.width, self.depth, self.flop_budget, self.seed = w, d, b, s
        self.n_mlps = 100

try:
    est.setup(_Ctx(WIDTH, DEPTH, BUDGET, SEED))
    out["setup_raised"] = None
except Exception as e:
    out["setup_raised"] = type(e).__name__ + ": " + str(e)[:160]
out["setup_s"] = time.perf_counter() - t0
out["stage"] = "setup_done"

# --- 5. predict, metered, under the cap -----------------------------------
from lifecycle_eval import materialise_predictions

ctx = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
err = ""
pred = None
t1 = time.perf_counter()
try:
    with ctx:
        pred = est.predict(mlp, BUDGET)
        # IN-WINDOW, because whestbench 0.16 does it in-window (scoring.py,
        # inside `with budget_ctx:`). Omitting it undercounts by exactly
        # 32,768 FLOPs on a float64 return -- 16 x 1024 x 2 -- which is the
        # last term of the official anchor reconciliation
        # (local_sim/p2_flop_anchor.py). The first run of this gate reported
        # 137,600,293,888 for exactly that reason.
        pred = materialise_predictions(pred, depth=DEPTH, width=WIDTH)
except MemoryError as e:
    err = "MemoryError: " + str(e)[:120]
except Exception as e:
    err = type(e).__name__ + ": " + str(e)[:160]
out["predict_s"] = time.perf_counter() - t1
out["predict_error"] = err
first_flops = float(getattr(ctx, "flops_used", float("nan")))
first_residual = float(getattr(ctx, "residual_wall_time_s", 0.0) or 0.0)
first_wall = float(getattr(ctx, "wall_time_s", 0.0) or 0.0)

# --- 5b. REPEAT in the same worker. Measure SETTLED current address space
# after dropping every instrument-owned network. VmPeak remains the hard-cap
# quantity, but a monotone peak cannot distinguish retained state from reuse.
def _proc_value(prefix):
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(prefix + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0

a = None
if pred is not None:
    a = np.asarray(pred)
    out["shape"] = list(a.shape)
    out["finite"] = bool(np.all(np.isfinite(a)))
    out["dtype"] = str(a.dtype)
    out["largest_array_bytes"] = int(a.nbytes)
del a, pred, mlp, _w, ctx
gc.collect()
reps = [{"i": 0, "vm_peak_bytes": _proc_value("VmPeak"),
         "vm_size_settled_bytes": _proc_value("VmSize"),
         "flops_used": first_flops,
         "error": err}]
if not err:
    for _i in range(1, max(1, REPEATS)):
        _s2 = (SEED + 1000 * _i)
        _raw2, _es2, _ = canonical_mint(WIDTH, DEPTH, _s2)
        _weights2 = build_weights(_raw2, WEIGHTS_MODE, fnp)
        del _raw2
        _mlp2 = MLP(weights=_weights2,
                    width=WIDTH, depth=DEPTH, seed=_es2)
        _ctx2 = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
        _e2 = ""
        _p2 = None
        try:
            with _ctx2:
                _p2 = est.predict(_mlp2, BUDGET)
                _p2 = materialise_predictions(_p2, depth=DEPTH, width=WIDTH)
        except MemoryError as e:
            _e2 = "MemoryError: " + str(e)[:120]
        except Exception as e:
            _e2 = type(e).__name__ + ": " + str(e)[:160]
        _flops2 = float(getattr(_ctx2, "flops_used", float("nan")))
        del _p2, _mlp2, _weights2, _ctx2
        gc.collect()
        reps.append({"i": _i, "vm_peak_bytes": _proc_value("VmPeak"),
                     "vm_size_settled_bytes": _proc_value("VmSize"),
                     "flops_used": _flops2,
                     "error": _e2})
        if _e2:
            break
out["repeat_runs"] = reps
out["repeat_error"] = next((r["error"] for r in reps if r["error"]), "")
_first = reps[0]["vm_size_settled_bytes"]
_last = reps[-1]["vm_size_settled_bytes"]
out["repeat_vm_size_growth_bytes"] = int(_last - _first)
out["repeat_flops_identical"] = bool(
    len({r["flops_used"] for r in reps if not r["error"]}) <= 1)
out["repeat_flops_observed"] = [r["flops_used"] for r in reps]
out["flops_used"] = first_flops
out["residual_wall_time_s"] = first_residual
out["wall_time_s"] = first_wall
out["multiplier"] = max(0.1, out["flops_used"] / BUDGET)

# --- 6. what it actually cost --------------------------------------------
ru = resource.getrusage(resource.RUSAGE_SELF)
out["peak_rss_bytes"] = int(ru.ru_maxrss) * 1024          # Linux reports KiB
try:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmPeak:"):
                out["vm_peak_bytes"] = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                out["vm_hwm_bytes"] = int(line.split()[1]) * 1024
except OSError:
    pass
out["stage"] = "done"
out["ok"] = (not err) and out.get("finite", False) and out.get("setup_raised") is None
print(json.dumps(out))
sys.exit(0 if out["ok"] else 1)
'''

CHILD_CONTROL = r'''
import json, resource, sys

LIMIT_MB = int(sys.argv[1])        # 0 means: do NOT apply a cap
REQUEST_GIB = float(sys.argv[2])

out = {"limit_mb": LIMIT_MB, "request_gib": REQUEST_GIB}
if LIMIT_MB > 0:
    L = LIMIT_MB * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (L, L))
    except Exception as e:
        out["setrlimit_raised"] = type(e).__name__ + ": " + str(e)[:120]
    soft, _h = resource.getrlimit(resource.RLIMIT_AS)
    out["cap_applied"] = (soft == L)
else:
    out["cap_applied"] = False

n = int(REQUEST_GIB * (1 << 30))
try:
    # bytearray TOUCHES the pages, so this is not a lazy reservation that a
    # virtual-address-space cap might or might not notice.
    buf = bytearray(n)
    buf[0] = 1
    buf[-1] = 1
    out["allocated"] = True
    out["survived"] = True
    del buf
except MemoryError as e:
    out["allocated"] = False
    out["survived"] = False
    out["error"] = "MemoryError: " + str(e)[:80]
print(json.dumps(out))
sys.exit(0)
'''

CHILD_LIFECYCLE_CONTROL = r'''
import gc, json, resource, sys

LIMIT_MB, ALLOC_MIB, REPEATS, RETAIN = map(int, sys.argv[1:5])
limit = LIMIT_MB * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def vm_size():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    return 0

held, rows, error = [], [], ""
try:
    for i in range(REPEATS):
        buf = bytearray(ALLOC_MIB * (1 << 20))
        for j in range(0, len(buf), 4096):
            buf[j] = 1
        if RETAIN:
            held.append(buf)
        del buf
        gc.collect()
        rows.append({"i": i, "vm_size_settled_bytes": vm_size()})
except Exception as exc:
    error = type(exc).__name__ + ": " + str(exc)[:160]

growth = (rows[-1]["vm_size_settled_bytes"] - rows[0]["vm_size_settled_bytes"]
          if len(rows) >= 2 else 0)
print(json.dumps({"completed": not error and len(rows) == REPEATS,
                  "retain": bool(RETAIN), "allocation_mib": ALLOC_MIB,
                  "repeats": REPEATS, "rows": rows, "growth_bytes": growth,
                  "error": error, "cap_applied":
                  resource.getrlimit(resource.RLIMIT_AS)[0] == limit}))
'''


# --------------------------------------------------------------------------- #
# where Linux is
# --------------------------------------------------------------------------- #

class LinuxRunner:
    """How to run a Linux Python here, or an honest statement that we cannot."""

    def __init__(self, kind: str, argv_prefix, repo_for_child: str, note: str):
        self.kind = kind
        self.argv_prefix = argv_prefix
        self.repo_for_child = repo_for_child
        self.note = note

    def run(self, src: str, args, timeout: int = 900):
        argv = list(self.argv_prefix) + ["-c", src] + [str(a) for a in args]
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, errors="replace")


def _native_linux() -> "LinuxRunner | None":
    try:
        import resource
    except Exception:
        return None
    if not hasattr(resource, "RLIMIT_AS"):
        return None
    return LinuxRunner("native", [sys.executable], _ROOT,
                       "native Linux; RLIMIT_AS present in this interpreter")


def _wsl() -> "LinuxRunner | None":
    exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if exe is None:
        return None
    # The repository as the Linux side sees it.
    drive, rest = os.path.splitdrive(os.path.abspath(_ROOT))
    if not drive:
        return None
    repo = "/mnt/" + drive[0].lower() + rest.replace("\\", "/")
    # Resolve $HOME in one fixed shell command, then use WSL's --exec form for
    # every child. Passing "$HOME/..." as the executable after bare `--`
    # caused WSL to concatenate the entire command through a shell; backticks
    # inside the Python source were then command-substituted before Python saw
    # them. An instrument hash cannot bind source that a launcher rewrites.
    probe = subprocess.run(
        [exe, "-d", WSL_DISTRO, "--exec", "sh", "-c",
         'p="$HOME/.arcwhest-linux/bin/python"; '
         'test -x "$p" && printf "%s\\n" "$HOME"'],
        capture_output=True, text=True, errors="replace", timeout=180)
    home = (probe.stdout or "").replace("\x00", "").strip()
    py = home + "/.arcwhest-linux/bin/python"
    ready = probe.returncode == 0 and home.startswith("/") and "\n" not in home
    if not ready:
        return LinuxRunner(
            "wsl-missing-venv", [], repo,
            "WSL is present but the Linux venv is absent. Build it with:\n"
            "    wsl.exe -d %s -- bash %s/scripts/linux_stack_setup.sh"
            % (WSL_DISTRO, repo))
    return LinuxRunner("wsl", [exe, "-d", WSL_DISTRO, "--exec", py], repo,
                       "WSL2 %s, official stack in %s" % (WSL_DISTRO, LINUX_VENV))


def find_linux() -> "LinuxRunner | None":
    return _native_linux() or _wsl()


def _json_tail(text: str):
    for line in reversed((text or "").replace("\x00", "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #

_FAILED = []
_TOTAL = [0]


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62}{detail}")
    if not ok:
        _FAILED.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True,
                    help="the estimator file to run under the cap. REQUIRED: a "
                         "default would let a run against the shipped file be "
                         "mistaken for a run against a candidate.")
    ap.add_argument("--limit-mb", type=int, default=DEFAULT_LIMIT_MB)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--weights-mode", default="f64_readonly",
                    help="the weight representation the candidate is measured "
                         "under. DEPLOYMENT TRUTH is f64_readonly; f32 is a "
                         "control and its peak is NOT a memory proof.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="predictions in the SAME worker. One cannot show a "
                         "cache that grows across a 100-MLP suite.")
    ap.add_argument("--lifecycle-control-mib", type=int,
                    default=LIFECYCLE_CONTROL_MIB,
                    help="allocation retained/released by lifecycle controls")
    ap.add_argument("--control-gib", type=float, default=9.0,
                    help="how much the controls request; must exceed the cap")
    ap.add_argument("--json", default="", help="write the report here")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    if a.repeats < 3:
        print("--repeats must be >= 3 for lifecycle evidence")
        return 2

    cand = os.path.abspath(a.candidate)
    if not os.path.exists(cand):
        print(f"no such candidate: {cand}")
        return 2

    print("Linux 8-GiB enforcement gate")
    print(f"  candidate   {os.path.relpath(cand, _ROOT)}")
    print(f"  cap         {a.limit_mb} MB   shape {a.width}x{a.depth}   "
          f"B = {a.budget:,}")

    runner = find_linux()
    # sha256 of the exact bytes tested. A path is not an identity: a report keyed
    # only on `candidate` stays "for this candidate" after the file is edited,
    # so certify section 9 could read a memory result for code that no longer
    # exists. Added 2026-09-05; `artifact_version` goes to 2 to say so.
    _h = hashlib.sha256()
    with open(cand, "rb") as _fh:
        for _chunk in iter(lambda: _fh.read(1 << 20), b""):
            _h.update(_chunk)
    # The INSTRUMENT's own bytes. A report that cannot say which version of this
    # gate produced it cannot be re-audited after the gate is edited -- which is
    # what happened between artifact_version 2 and 3.
    _ih = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as _fh:
        for _chunk in iter(lambda: _fh.read(1 << 20), b""):
            _ih.update(_chunk)
    report = {
        "artifact": "linux_memory_gate",
        # 4 (2026-09-10): binds current-source instrument identity,
        # distribution metadata and discriminating lifecycle controls.
        # official package versions and the instrument's own sha256, because
        # `certify_candidate` section 9 accepted a version-2 report without
        # checking that its shape, cap, budget or stack matched the candidate it
        # was certifying. See docs/research/2026-09-10_validation_audit.md #1.
        "artifact_version": 5,
        "candidate": os.path.relpath(cand, _ROOT).replace("\\", "/"),
        "candidate_sha256": _h.hexdigest(),
        "instrument": "local_sim/linux_memory_gate.py",
        "instrument_sha256": _ih.hexdigest(),
        "instrument_policy": "current-source-sha256",
        "limit_mb": a.limit_mb,
        "width": a.width, "depth": a.depth, "budget": a.budget, "seed": a.seed,
        "suite_n_mlps": DEFAULT_SUITE_N_MLPS,
        "weights_mode": a.weights_mode,
        "repeats": a.repeats,
        "host_platform": sys.platform,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }

    if runner is None or runner.kind == "wsl-missing-venv":
        why = ("no Linux execution environment is reachable from this host"
               if runner is None else runner.note)
        print()
        print("  DEPLOYMENT BLOCKER -- the 8,192 MB limit is NOT tested.")
        print("  " + why.replace("\n", "\n  "))
        print()
        print("  Exact command once a Linux environment exists:")
        print("    wsl.exe -d %s -- bash <repo>/scripts/linux_stack_setup.sh"
              % WSL_DISTRO)
        print("    uv run python local_sim/linux_memory_gate.py "
              "--candidate %s --json experiments/<slice>/linux_memory_gate.json"
              % os.path.relpath(cand, _ROOT))
        print()
        print("  This is NOT a pass. Windows accounting is accounting; it is "
              "not enforcement,")
        print("  and no run here may be quoted as clearing the memory limit.")
        report["status"] = "BLOCKED"
        report["blocker"] = why
        report["expected_artifact"] = "experiments/<slice>/linux_memory_gate.json"
        _emit(report, a.json)
        return 2

    print(f"  linux       {runner.kind} -- {runner.note}")
    report["linux_kind"] = runner.kind
    report["linux_note"] = runner.note
    print()

    # ---- N. the NEGATIVE control: the cap must KILL an over-allocation ----
    print("N  the cap is in force -- controls first, so a pass below means something")
    neg = runner.run(CHILD_CONTROL, [a.limit_mb, a.control_gib], a.timeout)
    negd = _json_tail(neg.stdout) or {}
    report["negative_control"] = negd
    child_stderr = {"negative": (neg.stderr or "").replace("\x00", "")}
    check("N: the negative-control child reported", bool(negd),
          (neg.stderr or "")[-70:] if not negd else "json returned")
    check("N: the negative-control child wrote no stderr",
          not child_stderr["negative"], child_stderr["negative"][-70:])
    check("N: the cap was APPLIED in that child (soft == requested)",
          negd.get("cap_applied") is True, str(negd.get("cap_applied")))
    check(f"N: requesting {a.control_gib} GiB under a {a.limit_mb} MB cap DIES",
          negd.get("survived") is False,
          negd.get("error", "IT SURVIVED -- the cap is not in force"))

    pos = runner.run(CHILD_CONTROL, [0, a.control_gib], a.timeout)
    posd = _json_tail(pos.stdout) or {}
    report["positive_control"] = posd
    child_stderr["positive"] = (pos.stderr or "").replace("\x00", "")
    check("P: the positive-control child wrote no stderr",
          not child_stderr["positive"], child_stderr["positive"][-70:])
    check("P: the SAME allocation with NO cap succeeds",
          posd.get("survived") is True,
          "so the death above is the cap, not the machine"
          if posd.get("survived") else str(posd.get("error"))[:60])

    controls_ok = (negd.get("cap_applied") is True
                   and negd.get("survived") is False
                   and posd.get("survived") is True
                   and not child_stderr["negative"]
                   and not child_stderr["positive"])
    if not controls_ok:
        print()
        print("  CONTROLS FAILED. Every memory claim below is WITHDRAWN: a gate "
              "whose cap")
        print("  is not demonstrably in force reports a state, not a property.")
        report["status"] = "CONTROLS_FAILED"
        _emit(report, a.json)
        return 1

    # ---- L. distinguish instrument retention from candidate retention ----
    print("\nL  repeated-worker lifecycle controls")
    stateless_run = runner.run(
        CHILD_LIFECYCLE_CONTROL,
        [a.limit_mb, a.lifecycle_control_mib, max(3, a.repeats), 0], a.timeout)
    retaining_run = runner.run(
        CHILD_LIFECYCLE_CONTROL,
        [a.limit_mb, a.lifecycle_control_mib, max(3, a.repeats), 1], a.timeout)
    stateless = _json_tail(stateless_run.stdout) or {}
    retaining = _json_tail(retaining_run.stdout) or {}
    child_stderr["lifecycle_stateless"] = (
        stateless_run.stderr or "").replace("\x00", "")
    child_stderr["lifecycle_retaining"] = (
        retaining_run.stderr or "").replace("\x00", "")
    report["lifecycle_controls"] = {
        "growth_threshold_bytes": LIFECYCLE_GROWTH_THRESHOLD_BYTES,
        "stateless": stateless, "retaining": retaining,
    }
    stateless_ok = (stateless.get("completed") is True
                    and stateless.get("cap_applied") is True
                    and abs(int(stateless.get("growth_bytes", 1 << 60)))
                    < LIFECYCLE_GROWTH_THRESHOLD_BYTES)
    retaining_ok = (retaining.get("completed") is True
                    and retaining.get("cap_applied") is True
                    and int(retaining.get("growth_bytes", -1))
                    >= LIFECYCLE_GROWTH_THRESHOLD_BYTES)
    check("L: stateless allocations settle below the growth threshold",
          stateless_ok,
          f"{int(stateless.get('growth_bytes', 0)) / 2**20:+.1f} MB")
    check("L: deliberately retained allocations exceed the threshold",
          retaining_ok,
          f"{int(retaining.get('growth_bytes', 0)) / 2**20:+.1f} MB")
    check("L: lifecycle-control children wrote no stderr",
          not child_stderr["lifecycle_stateless"]
          and not child_stderr["lifecycle_retaining"],
          (child_stderr["lifecycle_stateless"]
           + child_stderr["lifecycle_retaining"])[-70:])
    if not (stateless_ok and retaining_ok
            and not child_stderr["lifecycle_stateless"]
            and not child_stderr["lifecycle_retaining"]):
        print("\n  LIFECYCLE CONTROLS FAILED. Candidate memory is not run: the ")
        print("  instrument did not distinguish released from retained state.")
        report["status"] = "CONTROLS_FAILED"
        report["checks_total"] = _TOTAL[0]
        report["checks_failed"] = list(_FAILED)
        _emit(report, a.json)
        return 1

    # ---- C. the candidate, under the real cap ----------------------------
    print("\nC  the candidate's own predict(), under the real cap")
    res = runner.run(CHILD_PREDICT,
                     [a.limit_mb, runner.repo_for_child, _child_path(runner, cand),
                      a.width, a.depth, a.budget, a.seed,
                      a.weights_mode, a.repeats], a.timeout)
    d = _json_tail(res.stdout) or {}
    report["candidate_run"] = d
    child_stderr["candidate"] = (res.stderr or "").replace("\x00", "")
    report["child_stderr"] = child_stderr
    report["child_stderr_tail"] = child_stderr["candidate"][-600:]
    check("C: the child reported", bool(d),
          (res.stderr or "").replace("\x00", "")[-70:] if not d else
          f"stage={d.get('stage')}")
    check("C: the candidate child wrote no stderr",
          not child_stderr["candidate"], child_stderr["candidate"][-70:])
    check("C: the cap was applied to the candidate's process too",
          d.get("cap_applied") is True,
          f"soft={d.get('soft_limit_bytes')} requested={d.get('cap_requested_bytes')}")
    check("C: the REAL packages loaded, not local_sim's shims",
          d.get("stage") != "shim_leak" and "local_sim" not in
          str(d.get("whestbench_file", "")),
          str(d.get("whestbench_file", ""))[-46:])
    check("C: installed distribution metadata names both package versions",
          d.get("stack_version_source") == "importlib.metadata.version"
          and bool(d.get("whestbench_version"))
          and bool(d.get("flopscope_version")),
          f"whestbench={d.get('whestbench_version')!r}, "
          f"flopscope={d.get('flopscope_version')!r}")
    check("C: setup() raised nothing", d.get("setup_raised") is None,
          f"{d.get('setup_s', float('nan')):.4f} s"
          if d.get("setup_s") is not None else str(d.get("setup_raised")))
    check("C: predict() completed under the cap", d.get("predict_error") == "",
          d.get("predict_error") or f"{d.get('predict_s', float('nan')):.3f} s")
    check("C: the output has the right shape",
          d.get("shape") == [a.depth, a.width], str(d.get("shape")))
    check("C: every value is finite", d.get("finite") is True, str(d.get("dtype")))
    peak = d.get("vm_peak_bytes") or d.get("peak_rss_bytes") or 0
    cap_b = a.limit_mb * 1024 * 1024
    check("C: peak address space is under the cap", 0 < peak <= cap_b,
          f"VmPeak {peak / 2**20:.1f} MB of {a.limit_mb} MB "
          f"-> {cap_b / peak:.2f}x headroom" if peak else "unmeasured")
    check("C: the weights were the DEPLOYMENT representation",
          d.get("weights_mode") == a.weights_mode
          and (a.weights_mode != "f64_readonly"
               or (d.get("weight_dtype") == "float64"
                   and d.get("weight_writeable") is not True)),
          f"{d.get('weights_mode')} / dtype {d.get('weight_dtype')} / "
          f"writeable {d.get('weight_writeable')}")
    check("C: repeated predictions in the SAME worker all completed",
          d.get("repeat_error") == "" and
          len(d.get("repeat_runs") or []) >= a.repeats,
          d.get("repeat_error") or f"{len(d.get('repeat_runs') or [])} runs")
    check("C: ...and SETTLED address space did not GROW across them",
          # Instrument-owned raw mints, weights, MLPs and predictions are
          # deleted before each sample. Current VmSize can therefore expose
          # candidate-held state; monotone VmPeak cannot.
          abs(int(d.get("repeat_vm_size_growth_bytes", 1 << 60)))
          < LIFECYCLE_GROWTH_THRESHOLD_BYTES,
          f"{int(d.get('repeat_vm_size_growth_bytes', 0)) / 2**20:+.1f} MB "
          f"over {len(d.get('repeat_runs') or [])} predictions")
    print("      repeated FLOPs  "
          + ", ".join(f"{float(x):,.0f}" for x in d.get("repeat_flops_observed", []))
          + (" (identical)" if d.get("repeat_flops_identical") else
             " (value-dependent; recorded, not rejected)"))
    check("C: the single largest returned array is under 4 GiB",
          0 < int(d.get("largest_array_bytes", 0)) < 4 * (1 << 30),
          f"{int(d.get('largest_array_bytes', 0)) / 2**20:.3f} MB")
    # A cross-host FLOP agreement is a strong incidental check: it says the
    # Linux run executed the same program, under a different numpy, and was
    # billed identically. It is only asserted for the shape and file the anchor
    # describes -- a candidate has no anchor yet, by definition.
    if (a.width, a.depth) == (1024, 16) and os.path.samefile(
            cand, os.path.join(_ROOT, "estimator.py")):
        check("C: counted FLOPs equal the official Windows anchor exactly",
              float(d.get("flops_used", -1)) == float(ANCHOR_FLOPS_1024x16),
              f"{d.get('flops_used', float('nan')):,.0f} vs "
              f"{ANCHOR_FLOPS_1024x16:,}")

    print()
    print(f"  counted FLOPs   {d.get('flops_used', float('nan')):,.0f}")
    print(f"  multiplier      {d.get('multiplier', float('nan')):.6f}")
    print(f"  peak RSS        {(d.get('peak_rss_bytes') or 0) / 2**20:.1f} MB")
    print(f"  VmPeak          {(d.get('vm_peak_bytes') or 0) / 2**20:.1f} MB "
          "(address space -- THE quantity RLIMIT_AS caps)")
    print(f"  VmHWM           {(d.get('vm_hwm_bytes') or 0) / 2**20:.1f} MB")
    print()
    print("  SCOPE: this is a Linux kernel applying the installed package's own")
    print("  default. It is not the grading server, whose `memory_limit_mb` "
          "appears in")
    print("  no run_config and no graded report this repository holds. No "
          "absolute")
    print("  second measured here transfers either.")

    report["stack"] = {
        "whestbench": d.get("whestbench_version", ""),
        "flopscope": d.get("flopscope_version", ""),
        "numpy": d.get("numpy_version", ""),
        "python": d.get("python_version", ""),
        "source": d.get("stack_version_source", ""),
    }
    report["status"] = "PASS" if not _FAILED else "FAIL"
    report["checks_total"] = _TOTAL[0]
    report["checks_failed"] = list(_FAILED)
    _emit(report, a.json)

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED))
        return 1
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")
    return 0


def _child_path(runner: LinuxRunner, path: str) -> str:
    """The candidate's path as the Linux side sees it."""
    if runner.kind != "wsl":
        return path
    rel = os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")
    return runner.repo_for_child + "/" + rel


def _emit(report: dict, path: str) -> None:
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(report, indent=2) + "\n")
    print(f"  report -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
