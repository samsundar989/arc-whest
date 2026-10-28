"""Nominee verifier: is this exact committed file safe to designate? One PASS/FAIL table.

    uv run python tools/verify_nominee.py <source.py> [options]      (heavy stages: via workerq)

Stages, in order (``--stages`` picks a subset; default all but ``memory``):

  checkout  the file as committed at HEAD is copied alone into a temp dir; its sha256 must
            equal the working file, ``--expect-sha`` (if given) and the sha8 in
            docs/DESIGNATION.md (WARN when not yet listed there).
  imports   AST scan: every import (any scope) is flopscope / whestbench / stdlib. Then an
            isolated import (``-I``, temp cwd) in a CLIENT-ONLY venv (flopscope-client 0.12.3,
            no numpy, as the grader's participant process): must succeed, numpy not loaded.
  static    grader-contract AST checks: no bare ``except``; no broad handler that returns a
            constant; no ``budget_summary``; no ``random.seed`` / time-based seeding; no raw-
            array escape hatches (``.base``, ``__array__``, ``._array`` ...).
  rig       REAL client <-> local flopscope-server 0.12.3 over tcp (the split the grader runs):
            predict at 1024x16, 512x16 and 256x4 on seeded He nets must return a finite
            (depth, width) array. This is also the immutability and shape-generic check:
            a RemoteArray has no in-place ops, and a hard-coded shape fails at 512 / 256.
  meter     real flopscope 0.12.3 in-process, float64 READ-ONLY weights (grader twin), two
            separate processes: run A on DEV 0..7, run B on DEV 0..2. Determinism = B's
            prediction sha256 equals A's on all three; bill = one FLOP count on all eight
            nets, equal to ``--expect-flops`` exactly and within [0, 1e6] FLOPs above the
            graded per-MLP ``flops_used`` (the constructor/accounting offset); ops equal
            ``--expect-ops``.
  project   served-equivalent residual = reference served residual max x ops / ref ops
            (<= 0.37 s); served-equivalent wall = reference served wall max x wall-law ratio
            (0.36 ms/op + 93 ps/FLOP; <= 100 s). Reference defaults to #336993's graded report.
  memory    (opt-in, ~1 h) local_sim/linux_memory_gate.py in WSL under RLIMIT_AS 8 GiB on the
            real 0.12.3 stack, 100 calls: VmPeak <= 7,200 MiB, every call finite with the same
            FLOPs, settled growth from call 10 to 100 <= 64 MiB (docs/GOTCHAS.md item 9).

Writes ``<out>/verify.json`` and ``<out>/verify.md`` (the table for docs/DESIGNATION.md).
Exit 0 iff no stage FAILs. Never submits, designates or changes anything outside ``<out>``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ROOT is re-pointed in main() at the git toplevel of the SOURCE, so a run from a workerq
# snapshot still checks the source against the real repository's HEAD and finds cache/.
ROOT = Path(__file__).resolve().parent.parent
FS0123 = Path.home() / ".venvs" / "fs0123" / "Scripts" / "python.exe"
RIG = RIG_SRV = RIG_CLI = REF_REPORT = None


def _set_root(root: Path):
    global ROOT, RIG, RIG_SRV, RIG_CLI, REF_REPORT
    ROOT = root
    RIG = ROOT / "cache" / "verify_rig"
    RIG_SRV = RIG / "srv" / "Scripts" / "python.exe"
    RIG_CLI = RIG / "cli" / "Scripts" / "python.exe"
    REF_REPORT = ROOT / "experiments" / "run_reports" / "literal_a4_r1_336993_graded_report.json"


_set_root(ROOT)
REF_OPS, REF_FLOPS = 22235, 286868957368          # #336993 own canonical 0.12.3 bill
WSL_FAMILY = Path("C:/Users/samsu/.codex/worktrees/final-push-research/arc-whest/"
                  "experiments/2026-10-16_finalpush")
WSL_NP = WSL_FAMILY / "hub/young_selected/.wsl-np226"
WSL_FS = WSL_FAMILY / "cost/.wsl-fs0123"
WSL_PY = "/home/samsu/.arcwhest-linux/bin/python"
ALLOWED_ROOTS = {"flopscope", "whestbench"}
STAGES = ("checkout", "imports", "static", "rig", "meter", "project", "memory")

ROWS: list = []


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def row(stage, check, status, detail=""):
    ROWS.append({"stage": stage, "check": check, "status": status, "detail": str(detail)})
    print(f"  {status:<4}  {stage:<8} {check:<44} {detail}", flush=True)


def linux(p: Path) -> str:
    p = Path(p).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


# --------------------------------------------------------------------------- children

# Runs in the CLIENT-ONLY venv: import alone, then (rig) predict through the real client.
CLIENT_CHILD = r'''
import importlib.util, json, math, os, random, sys, time
stub = sys.argv[1]; cand = sys.argv[2]; shapes = sys.argv[3:]
sys.path.insert(0, stub)                       # participant-side MLP/BaseEstimator only
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("candidate", cand)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
loaded = sorted({m.split(".")[0] for m in set(sys.modules) - before})
res = {"import_ok": True, "numpy_loaded": "numpy" in sys.modules, "new_roots": loaded, "shapes": {}}
if shapes:
    import flopscope as flops, flopscope.numpy as fnp
    from whestbench import MLP
    for s in shapes:
        w, d = (int(v) for v in s.split("x"))
        rnd = random.Random(w * 1000 + d)
        t = time.time()
        try:
            with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
                W = [fnp.array([[rnd.gauss(0.0, math.sqrt(2.0 / w)) for _ in range(w)] for _ in range(w)],
                               dtype=fnp.float64) for _ in range(d)]
                ctx = type("C", (), dict(width=w, depth=d, flop_budget=2 ** 41, seed=0, api_version="1.0",
                                         scratch_dir=None, submission_dir=None))()
                est = mod.Estimator(); est.setup(ctx)
                out = est.predict(MLP(width=w, depth=d, weights=W, seed=7), 2 ** 41)
                finite = bool(fnp.all(fnp.isfinite(out)))
                shape = tuple(int(x) for x in out.shape)
            res["shapes"][s] = {"ok": finite and shape == (d, w), "finite": finite, "shape": shape,
                                "seconds": round(time.time() - t, 1)}
        except Exception as e:
            res["shapes"][s] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                "seconds": round(time.time() - t, 1)}
print("@@" + json.dumps(res))
'''

# Runs in the fs0123 venv (real flopscope 0.12.3 + numpy): DEV nets, real meter.
METER_CHILD = r'''
import hashlib, importlib.util, json, sys, time
root = sys.argv[1]; cand = sys.argv[2]; idx = [int(x) for x in sys.argv[3:]]
spec = importlib.util.spec_from_file_location("candidate", cand)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for p in (root, root + "/tools", root + "/local_sim"):   # APPEND: the real packages win
    sys.path.append(p)
import numpy as np, flopscope as flops, flopscope.numpy as fnp
import panels, real_data as rd, lifecycle_eval as le
out = []
for i in idx:
    split, j, seed, name = panels.nets("dev")[i]
    w_np, _t, _m = rd.get(j, split, dtype=np.float32)
    depth, width = w_np.shape[0], w_np.shape[1]
    est = mod.Estimator()
    est.setup(le.SetupContext(le.LifecycleSpec(width=width, depth=depth, flop_budget=2 ** 41, seed=0)))
    mlp = type("M", (), {})()
    mlp.weights, mlp.width, mlp.depth, mlp.seed = le.build_weights(list(w_np), "f64_readonly", fnp), width, depth, int(seed)
    b = flops.BudgetContext(flop_budget=2 ** 41, quiet=True)
    t = time.perf_counter()
    with b:
        p = est.predict(mlp, 2 ** 41)
        p = le.materialise_predictions(p, depth=depth, width=width)
    wall = time.perf_counter() - t
    le.validate_predictions(p, depth=depth, width=width)
    ns = b.summary_dict(by_namespace=True).get("by_namespace") or {}
    a = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    out.append({"dev": i, "name": name, "flops": float(b.flops_used),
                "ops": sum(float(v.get("calls", 0)) for v in ns.values()),
                "sha": hashlib.sha256(a.tobytes()).hexdigest(), "finite": bool(np.isfinite(a).all()),
                "local_wall_s": round(wall, 1)})
    print(json.dumps(out[-1]), flush=True)
print("@@" + json.dumps(out))
'''


def _tail(text: str):
    for line in reversed(text.splitlines()):
        if line.startswith("@@"):
            return json.loads(line[2:])
    return None


def run_child(python, code, args, cwd, timeout, env=None):
    f = Path(cwd) / f"_child_{abs(hash(code)) % 10**8}.py"
    f.write_text(code, encoding="utf-8")
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    r = subprocess.run([str(python), "-I", str(f), *map(str, args)], cwd=cwd, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout, env=e)
    return r, _tail(r.stdout)


# --------------------------------------------------------------------------- stages

def stage_checkout(src: Path, tmp: Path, a) -> Path:
    rel = src.resolve().relative_to(ROOT).as_posix()
    work = src.read_bytes()
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        row("checkout", "committed at HEAD", "FAIL", f"{rel} not in HEAD")
        dst = tmp / "estimator.py"
        dst.write_bytes(work)
        return dst
    blob = r.stdout
    dst = tmp / "estimator.py"
    dst.write_bytes(blob)
    h = sha(blob)
    row("checkout", "committed at HEAD", "PASS", f"{rel} sha {h[:16]}")
    row("checkout", "working file == HEAD", "PASS" if sha(work) == h else "FAIL",
        "" if sha(work) == h else f"working {sha(work)[:16]} differs")
    if a.expect_sha:
        row("checkout", "sha == --expect-sha", "PASS" if h.startswith(a.expect_sha) else "FAIL", a.expect_sha)
    des = (ROOT / "docs" / "DESIGNATION.md").read_text(encoding="utf-8", errors="replace")
    row("checkout", "sha8 listed in docs/DESIGNATION.md", "PASS" if h[:8] in des else "WARN",
        h[:8] if h[:8] in des else f"{h[:8]} not listed yet")
    return dst


def _import_roots(tree):
    roots = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots |= {x.name.split(".")[0] for x in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            roots.add(n.module.split(".")[0])
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in (
                "__import__", "import_module"):
            roots.add("<dynamic import>")
    return roots


def stage_imports(cand: Path, tmp: Path, a):
    tree = ast.parse(cand.read_text(encoding="utf-8"))
    roots = _import_roots(tree)
    bad = sorted(r for r in roots if r not in ALLOWED_ROOTS and r not in sys.stdlib_module_names)
    row("imports", "AST: only flopscope/whestbench/stdlib", "FAIL" if bad else "PASS",
        bad or sorted(roots))
    if not RIG_CLI.exists():
        row("imports", "client-only venv import", "FAIL", f"missing {RIG_CLI} (see --build-rig)")
        return
    stub = tmp / "stub"
    (stub / "whestbench").mkdir(parents=True, exist_ok=True)
    (stub / "whestbench" / "__init__.py").write_text(
        "class BaseEstimator:\n    pass\n\n\nclass MLP:\n"
        "    def __init__(self, width=0, depth=0, weights=None, seed=0):\n"
        "        self.width, self.depth, self.weights, self.seed = width, depth, weights, seed\n")
    r, res = run_child(RIG_CLI, CLIENT_CHILD, [stub, cand], tmp, 300)
    if res is None:
        row("imports", "client-only venv import (no numpy)", "FAIL", r.stderr[-400:])
        return
    ok = res["import_ok"] and not res["numpy_loaded"]
    extra = [m for m in res["new_roots"] if m not in sys.stdlib_module_names and not m.startswith("_")]
    row("imports", "client-only venv import (no numpy)", "PASS" if ok else "FAIL",
        f"numpy_loaded={res['numpy_loaded']}; non-stdlib loaded: {extra}")


def _const_return(body):
    """'array' if a handler returns a constant array, 'scalar' for a literal, else ''."""
    kind = ""
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Return) and n.value is not None:
            v = n.value
            if isinstance(v, ast.Call) and getattr(v.func, "attr", "") in ("zeros", "ones", "full", "zeros_like",
                                                                           "full_like", "ones_like"):
                return "array"
            if isinstance(v, ast.Constant):
                kind = "scalar"
    return kind


def _handlers(tree):
    """(enclosing function name, handler) for every except clause."""
    out = []

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(ch, ch.name)
            else:
                if isinstance(ch, ast.ExceptHandler):
                    out.append((fn, ch))
                visit(ch, fn)
    visit(tree, "<module>")
    return out


def stage_static(cand: Path, tmp: Path, a):
    tree = ast.parse(cand.read_text(encoding="utf-8"))
    bare, broad, const, helper = [], [], [], []
    for fn, n in _handlers(tree):
        if n.type is None:
            bare.append(n.lineno)
        elif getattr(n.type, "id", "") in ("Exception", "BaseException"):
            broad.append(f"{fn}:{n.lineno}")
            k = _const_return(n.body)
            # a constant ANSWER is the forbidden pattern: a constant array anywhere, or any
            # literal returned from predict itself; a helper's False/None is a code path choice
            if k == "array" or (k and fn == "predict"):
                const.append(f"{fn}:{n.lineno}")
            elif k:
                helper.append(f"{fn}:{n.lineno}")
    row("static", "no bare except", "FAIL" if bare else "PASS", bare or "")
    row("static", "no broad except returning a constant answer", "FAIL" if const else "PASS",
        const or (f"broad handlers {broad}" if broad else ""))
    if helper:
        row("static", "helper handlers returning a literal (review)", "WARN", helper)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    bs = sorted((names | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
                & {"budget_summary", "budget_summary_dict"})
    row("static", "no budget_summary in source", "FAIL" if bs else "PASS", bs or "")
    seeds = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "seed"]
    clock = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") in ("time", "time_ns", "perf_counter", "monotonic")]
    row("static", "no global RNG seeding", "FAIL" if seeds else "PASS", seeds or "")
    row("static", "no clock reads (time-based randomness)", "WARN" if clock else "PASS", clock or "")
    raw = sorted(names & {"base", "__array__", "_array", "_data", "_value", "ctypes", "__array_interface__"})
    row("static", "no raw-array escape hatch attributes", "WARN" if raw else "PASS", raw or "")


def stage_rig(cand: Path, tmp: Path, a):
    if not (RIG_SRV.exists() and RIG_CLI.exists()):
        row("rig", "rig venvs present", "FAIL", f"build with --build-rig ({RIG})")
        return
    shapes = a.rig_shapes.split(",")
    stub = tmp / "stub"
    url = f"tcp://127.0.0.1:{a.rig_port}"
    env = dict(os.environ, FLOPSCOPE_MAX_ARRAY_BYTES=str(4 << 30))
    log = open(tmp / "rig_server.log", "w")
    srv = subprocess.Popen([str(RIG_SRV), "-m", "flopscope_server", "--url", url, "--timeout", "7200"],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        time.sleep(6)
        r, res = run_child(RIG_CLI, CLIENT_CHILD, [stub, cand, *shapes], tmp, 7200,
                           env={"FLOPSCOPE_SERVER_URL": url, "FLOPSCOPE_MAX_ARRAY_BYTES": str(4 << 30)})
    finally:
        srv.kill()
        try:
            srv.wait(timeout=30)
        finally:
            log.close()
    if res is None:
        row("rig", "client<->server run", "FAIL", (r.stderr or r.stdout)[-400:])
        return
    for s in shapes:
        x = res["shapes"].get(s, {})
        row("rig", f"real client predict {s}", "PASS" if x.get("ok") else "FAIL",
            x.get("error") or f"finite={x.get('finite')} shape={x.get('shape')} {x.get('seconds')} s")


def stage_meter(cand: Path, tmp: Path, a, out: dict):
    nA = list(range(a.dev_n))
    rA, A = run_child(FS0123, METER_CHILD, [ROOT.as_posix(), cand, *nA], tmp, 7200)
    rB, B = run_child(FS0123, METER_CHILD, [ROOT.as_posix(), cand, *nA[:a.det_n]], tmp, 7200)
    if A is None or B is None:
        row("meter", "real-meter runs", "FAIL", ((rA.stderr if A is None else rB.stderr) or "")[-400:])
        return
    out["meter"] = {"A": A, "B": B}
    fin = all(x["finite"] for x in A + B)
    row("meter", f"finite on DEV 0..{a.dev_n - 1}", "PASS" if fin else "FAIL")
    same = all(x["sha"] == y["sha"] for x, y in zip(A, B))
    row("meter", f"determinism: 2 processes x {a.det_n} nets bit-identical", "PASS" if same else "FAIL",
        [x["sha"][:12] for x in B])
    fl = {x["flops"] for x in A + B}
    ops = {x["ops"] for x in A + B}
    flops, nops = (fl.pop() if len(fl) == 1 else None), (ops.pop() if len(ops) == 1 else None)
    out["bill"] = {"flops": flops, "ops": nops}
    row("meter", "one FLOP count on every net", "PASS" if flops is not None else "FAIL",
        f"{flops:,.0f}" if flops is not None else sorted(fl))
    if a.expect_flops is not None and flops is not None:
        row("meter", "FLOPs == --expect-flops", "PASS" if flops == a.expect_flops else "FAIL",
            f"{flops - a.expect_flops:+,.0f}")
    if a.expect_ops is not None and nops is not None:
        row("meter", "ops == --expect-ops", "PASS" if nops == a.expect_ops else "FAIL", f"{nops:,.0f}")
    if a.graded_report and flops is not None:
        rep = json.loads(Path(a.graded_report).read_text())
        g = {m["telemetry"]["flops_used"] for m in rep["evaluation"]["results"]["per_mlp"]}
        if len(g) == 1:
            d = flops - g.pop()
            row("meter", "bill vs graded flops_used (offset 0..1e6)", "PASS" if 0 <= d <= 1e6 else "FAIL",
                f"local - graded = {d:+,.0f}")
        else:
            row("meter", "bill vs graded flops_used", "FAIL", f"graded not constant: {len(g)} values")


def stage_project(cand: Path, tmp: Path, a, out: dict):
    b = out.get("bill") or {}
    if b.get("flops") is None or b.get("ops") is None:
        row("project", "needs the meter stage", "FAIL", "no bill")
        return
    rep = json.loads(Path(a.ref_report).read_text())
    tel = [m["telemetry"] for m in rep["evaluation"]["results"]["per_mlp"]]
    res_max = max(t["residual_wall_time_s"] for t in tel)
    wall_max = max(t["participant_predict_wall_s"] for t in tel)
    law = lambda f, o: 0.36e-3 * o + 93e-12 * f
    r_eq = res_max * b["ops"] / a.ref_ops
    w_eq = wall_max * law(b["flops"], b["ops"]) / law(a.ref_flops, a.ref_ops)
    out["projection"] = {"ref_report": str(a.ref_report), "ref_residual_max": res_max, "ref_wall_max": wall_max,
                         "residual_eq_s": r_eq, "wall_eq_s": w_eq}
    row("project", "served-equivalent residual <= 0.37 s", "PASS" if r_eq <= 0.37 else "FAIL",
        f"{r_eq:.4f} s (ref max {res_max:.4f} s x {b['ops']:.0f}/{a.ref_ops})")
    row("project", "served-equivalent wall <= 100 s", "PASS" if w_eq <= 100 else "FAIL",
        f"{w_eq:.1f} s (ref max {wall_max:.1f} s x wall-law ratio)")


def stage_memory(cand: Path, tmp: Path, a, out: dict):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_lmg", ROOT / "local_sim" / "linux_memory_gate.py")
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    env = ["/usr/bin/env", "PYTHONPATH=" + linux(WSL_NP) + ":" + linux(WSL_FS),
           "OPENBLAS_NUM_THREADS=2", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"]
    gate.find_linux = lambda: gate.LinuxRunner(
        "wsl", ["C:/Windows/System32/wsl.exe", "-d", "Ubuntu", "--exec", *env, WSL_PY], linux(ROOT),
        "WSL Ubuntu, real 0.12.3 overlay (verify_nominee)")
    rep_path = Path(a.out) / "memory_report.json"
    old = sys.argv
    sys.argv = ["linux_memory_gate.py", "--candidate", str(cand), "--limit-mb", "8192", "--width", "1024",
                "--depth", "16", "--budget", str(2 ** 41), "--seed", "90000000101", "--weights-mode",
                "f64_readonly", "--repeats", str(a.mem_calls), "--lifecycle-control-mib", "16",
                "--control-gib", "9", "--reset-accumulator", "--blas-threads", "2", "--timeout", str(a.mem_timeout),
                "--json", str(rep_path)]
    try:
        gate.main()
    finally:
        sys.argv = old
    rep = json.loads(rep_path.read_text())
    c = rep.get("candidate_run") or {}
    runs = c.get("repeat_runs") or []
    ok_controls = rep.get("status") in ("PASS", "FAIL") and c.get("cap_applied")
    row("memory", "RLIMIT_AS 8 GiB applied, controls ran", "PASS" if ok_controls else "FAIL",
        f"status {rep.get('status')}; failed {rep.get('checks_failed')}")
    if len(runs) < a.mem_calls:
        row("memory", f"{a.mem_calls} calls completed", "FAIL", f"{len(runs)} runs; {c.get('repeat_error')}")
        return
    peak = max(r["vm_peak_bytes"] for r in runs) / 2 ** 20
    s = [r["vm_size_settled_bytes"] / 2 ** 20 for r in runs]
    grow = s[-1] - s[min(10, len(s) - 1)]
    errs = [r for r in runs if r.get("error")]
    out["memory"] = {"vm_peak_mib": peak, "settled_growth_after10_mib": grow, "calls": len(runs),
                     "flops_identical": c.get("repeat_flops_identical"), "finite": c.get("finite")}
    row("memory", f"{a.mem_calls} calls, none raised, FLOPs identical", "PASS" if not errs and c.get(
        "repeat_flops_identical") and c.get("finite") else "FAIL", f"{len(errs)} errors")
    row("memory", "VmPeak <= 7,200 MiB", "PASS" if peak <= 7200 else "FAIL", f"{peak:,.0f} MiB")
    row("memory", "settled growth call 10 -> last <= 64 MiB", "PASS" if grow <= 64 else "FAIL", f"{grow:+.1f} MiB")


# --------------------------------------------------------------------------- main

def build_rig():
    uv = shutil.which("uv") or "uv"
    for name, pkgs in (("srv", ["flopscope-server==0.12.3", "flopscope==0.12.3", "numpy==2.2.6"]),
                       ("cli", ["flopscope-client==0.12.3"])):
        v = RIG / name
        if not (v / "Scripts" / "python.exe").exists():
            subprocess.run([uv, "venv", "-q", "-p", "3.10", str(v)], check=True)
            subprocess.run([uv, "pip", "install", "-q", "-p", str(v / "Scripts" / "python.exe"), *pkgs], check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--stages", default="checkout,imports,static,rig,meter,project")
    ap.add_argument("--out", default="")
    ap.add_argument("--expect-sha", default="")
    ap.add_argument("--expect-flops", type=float, default=None)
    ap.add_argument("--expect-ops", type=float, default=None)
    ap.add_argument("--graded-report", default="", help="this candidate's own graded report, if graded")
    ap.add_argument("--ref-report", default="", help="default: #336993's graded report")
    ap.add_argument("--ref-ops", type=float, default=REF_OPS)
    ap.add_argument("--ref-flops", type=float, default=REF_FLOPS)
    ap.add_argument("--dev-n", type=int, default=8)
    ap.add_argument("--det-n", type=int, default=3)
    ap.add_argument("--rig-shapes", default="1024x16,512x16,256x4")
    ap.add_argument("--rig-port", type=int, default=5631)
    ap.add_argument("--mem-calls", type=int, default=100)
    ap.add_argument("--mem-timeout", type=int, default=18000,
                    help="seconds for the 100-call child; 7200 timed out on a loaded box (L2 #3609)")
    ap.add_argument("--build-rig", action="store_true")
    a = ap.parse_args(argv)
    src = Path(a.source).resolve()
    top = subprocess.run(["git", "-C", str(src.parent), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if top.returncode == 0:
        _set_root(Path(top.stdout.strip()).resolve())
    a.ref_report = a.ref_report or str(REF_REPORT)
    if a.build_rig:
        build_rig()
    stages = [s for s in STAGES if s in a.stages.split(",")]
    a.out = a.out or str(ROOT / "experiments" / "verify" / f"{src.stem}_{time.strftime('%Y%m%dT%H%M%S')}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = {"source": str(src), "stages": stages, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"verify_nominee: {src}\n  stages {stages}\n  out {a.out}", flush=True)
    with tempfile.TemporaryDirectory(prefix="verify_nominee_", ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        cand = stage_checkout(src, tmp, a)       # always: every later stage runs the committed copy
        out["sha256"] = sha(cand.read_bytes())
        for s in stages:
            if s == "checkout":
                continue
            fn = globals()[f"stage_{s}"]
            try:
                fn(cand, tmp, a, out) if s in ("meter", "project", "memory") else fn(cand, tmp, a)
            except Exception as e:  # a crashed stage is a FAIL row, never a silent pass
                row(s, "stage completed", "FAIL", f"{type(e).__name__}: {str(e)[:300]}")
    out["rows"] = ROWS
    out["verdict"] = "FAIL" if any(r["status"] == "FAIL" for r in ROWS) else "PASS"
    Path(a.out, "verify.json").write_text(json.dumps(out, indent=1))
    md = [f"**verify_nominee** `{src.relative_to(ROOT).as_posix()}` sha `{out['sha256'][:16]}` — "
          f"**{out['verdict']}** ({out['started_utc']})", "", "| stage | check | status | detail |",
          "|---|---|---|---|"]
    md += [f"| {r['stage']} | {r['check']} | {r['status']} | {r['detail'].replace('|', '/')} |" for r in ROWS]
    Path(a.out, "verify.md").write_text("\n".join(md) + "\n")
    print(f"\nVERDICT {out['verdict']}  ({a.out})", flush=True)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
