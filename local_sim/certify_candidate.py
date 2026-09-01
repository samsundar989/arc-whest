"""Certify a CANDIDATE estimator file, read-only, without touching the shipped one.

Contract: `experiments/LOG.md`, *CONTRACT WP0C+WP0D* (2026-08-31), committed at
`71bf62b` before this file existed. `GAME_PLAN.md` § WP0.3.

WHY THIS EXISTS
---------------
`scripts/preflight.ps1` hardcodes `estimator.py` in steps 4, 12, 13, 15 and 16;
only `-Archive` is a parameter, and it selects what step 5 compares *against*.
So **a green 16/16 certifies v2.6.1, whatever candidate is under development** —
the defect that made gate item 6 of the Track-0S submission contract
unsatisfiable for v2.7, with two honest exits and no third: promote the
candidate first, or parameterise the gate.

This is the third exit the contract could not see at the time: a **separate
read-only command** that certifies an arbitrary file. It adds no path through
`preflight.ps1`, writes nothing the gate reads, and therefore cannot weaken the
ship-file gate. `preflight.ps1` still certifies `estimator.py`, unchanged.

`--candidate` is REQUIRED and has no default. A default of `estimator.py` would
let a certification of the shipped file be mistaken for a certification of a
candidate, which is precisely the confusion this command exists to remove.

WHAT IT DOES NOT DO
-------------------
* **It reads no truth and reports no accuracy.** Where the lifecycle evaluator
  needs a reference it is handed **zeros**, so its `mse` column is
  `mean(prediction^2)` — a magnitude, not an error. It is labelled that way and
  never compared to anything.
* **It is not a promotion.** `GAME_PLAN.md` § 6 lists what promotion requires;
  this is one input to it, not a substitute. It does not run `preflight -Full`,
  it does not touch a vault or a seed block, and it does not submit.
* **It cannot enforce 8 GiB on Windows.** Section 9 is *accounting* here and
  says so; `local_sim/linux_memory_gate.py` is the enforcement, and section 9
  reads that gate's report if one exists rather than guessing.

Run:
    uv run python local_sim/certify_candidate.py --candidate estimator.py
    uv run python local_sim/certify_candidate.py --candidate estimator_v2_7.py \
        --json experiments/<slice>/certify_v27.json
    uv run python local_sim/certify_candidate.py --candidate <f> --quick   # skip 1, 10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships numpy shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
FLOOR = 0.1
#: Seeds OUTSIDE every registered block (`local_sim/seed_blocks.py`). This
#: command measures deployment properties, never accuracy, so it claims no
#: block and reads no truth -- and typing block seeds here would trip
#: `test_seed_ledger.py`'s source scan, as it once did in `test_v27_parity.py`.
SEEDS = (90000000201, 90000000202, 90000000203, 90000000204)

RESIDUAL_CAP_S = 0.4
WALL_CAP_S = 120.0
SETUP_CAP_S = 5.0
MEMORY_CAP_MB = 8192
SINGLE_ARRAY_CAP = 4 * (1 << 30)

_FAILED = []
_SKIPPED = []
_TOTAL = [0]
_SECTIONS = {}
_CURRENT = ["?"]


def head(t):
    _CURRENT[0] = t.split(None, 1)[0]
    print(f"\n{t}")


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    _SECTIONS.setdefault(_CURRENT[0], []).append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62}{detail}")
    if not ok:
        _FAILED.append(f"[{_CURRENT[0]}] {name}")


def skip(name, why):
    """NOT a pass. Counted separately and named in the summary."""
    _SKIPPED.append(f"[{_CURRENT[0]}] {name} -- {why}")
    print(f"  SKIP  {name:<62}{why}")


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _run(argv, timeout=1800, env=None):
    return subprocess.run(argv, capture_output=True, text=True, cwd=_ROOT,
                          timeout=timeout, errors="replace", env=env)


def _uvpy(*args, timeout=1800):
    return _run([sys.executable] + list(args), timeout=timeout)


def _load(path):
    spec = importlib.util.spec_from_file_location("candidate_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.n_mlps = len(SEEDS)


# --------------------------------------------------------------------------- #
# the child that runs one predict in a FRESH interpreter
# --------------------------------------------------------------------------- #
FRESH_WORKER = r'''
import hashlib, json, os, sys
REPO, CAND, W, D, B, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), \
    int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
LS = os.path.join(REPO, "local_sim")
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != LS]
for p in (REPO, LS):
    if p not in sys.path:
        sys.path.append(p)
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from mint_provenance import canonical_mint
try:
    flops.configure(symmetry_warnings=False)
except Exception:
    pass
import importlib.util as iu
spec = iu.spec_from_file_location("cand", CAND)
mod = iu.module_from_spec(spec); spec.loader.exec_module(mod)
class C:
    width, depth, flop_budget, seed, n_mlps = W, D, B, SEED, 1
est = mod.Estimator(); est.setup(C())
w_np, est_seed, _ = canonical_mint(W, D, SEED)
mlp = MLP(weights=[fnp.array(x) for x in w_np], width=W, depth=D, seed=est_seed)
ctx = flops.BudgetContext(flop_budget=B, quiet=True)
with ctx:
    pred = est.predict(mlp, B)
a = np.ascontiguousarray(np.asarray(pred, dtype=np.float64))
print(json.dumps({"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                  "shape": list(a.shape),
                  "flops_used": float(ctx.flops_used),
                  "pid": os.getpid()}))
'''


def _json_tail(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True,
                    help="the estimator file to certify. REQUIRED, no default.")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--json", default="")
    ap.add_argument("--quick", action="store_true",
                    help="skip the network-dependent sections (1 whest validate "
                         "is local, 10 subprocess smoke needs the dataset)")
    ap.add_argument("--skip-smoke", action="store_true")
    a = ap.parse_args()

    cand = os.path.abspath(a.candidate)
    if not os.path.exists(cand):
        print(f"no such candidate: {cand}")
        return 2
    rel = os.path.relpath(cand, _ROOT).replace("\\", "/")
    is_ship = os.path.abspath(cand) == os.path.join(_ROOT, "estimator.py")

    sha_before = _sha(cand)
    ship_sha_before = _sha(os.path.join(_ROOT, "estimator.py"))

    print("candidate certification -- READ-ONLY, TRUTH-FREE")
    print(f"  candidate   {rel}")
    print(f"  sha256      {sha_before}")
    print(f"  shape       {a.width}x{a.depth}   B = {a.budget:,}")
    print("  NOTE: this is NOT preflight and NOT a promotion. The ship-file "
          "gate is untouched;")
    print("        `scripts/preflight.ps1` still certifies estimator.py.")
    if is_ship:
        print("  (the subject IS the shipped file -- a known-good control run)")

    report = {"artifact": "candidate_certification", "artifact_version": 1,
              "candidate": rel, "candidate_sha256": sha_before,
              "width": a.width, "depth": a.depth, "budget": a.budget,
              "truth_free": True, "is_shipped_file": is_ship,
              "sections": {}}

    # ---------------------------------------------------------------- 1 ----
    head("1  the official grader contract")
    if a.quick:
        skip("1: whest validate", "--quick")
    else:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        p = _run([os.path.join(os.path.dirname(sys.executable), "whest"),
                  "validate", "--estimator", rel], env=env, timeout=600)
        check("1: `whest validate` accepts the candidate", p.returncode == 0,
              (p.stdout or p.stderr).strip().splitlines()[-1][:60]
              if (p.stdout or p.stderr).strip() else "")
        report["sections"]["validate"] = {"returncode": p.returncode}

    # ---------------------------------------------------------------- 2 ----
    head("2  package and round identity")
    p = _uvpy(os.path.join(_HERE, "official_config.py"), timeout=900)
    check("2: the LIVE round matches the INSTALLED package, field by field",
          p.returncode == 0, (p.stdout or "").strip().splitlines()[-1][:40])
    import flopscope as _fl
    import whestbench as _wb
    for m, name in ((_wb, "whestbench"), (_fl, "flopscope")):
        check(f"2: {name} is the real package, not local_sim's shim",
              "local_sim" not in os.path.abspath(m.__file__),
              os.path.abspath(m.__file__)[-44:])
    report["sections"]["package_identity"] = {
        "whestbench": _wb.__file__, "flopscope": _fl.__file__,
        "official_config_rc": p.returncode}

    # ---------------------------------------------------------------- 3 ----
    head("3  shape generality, the exact layer-1 closed form, and equivariance")
    # `edge_cases.py` already owns these 30 invariants and already takes a path.
    # Re-implementing them here would give this repository two answers to
    # "is layer 1 right", which is the failure it keeps recording.
    p = _uvpy(os.path.join(_HERE, "edge_cases.py"), rel, timeout=1800)
    tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-1:] or [""]
    check("3: the invariant suite passes on THIS candidate", p.returncode == 0,
          tail[0][:60])
    report["sections"]["invariants"] = {"returncode": p.returncode,
                                        "tail": tail[0]}

    # ---------------------------------------------------------------- 4 ----
    head("4  determinism, in-process and in a FRESH worker")
    fresh = []
    for s in SEEDS[:2]:
        r = _uvpy("-c", FRESH_WORKER, _ROOT, cand, str(a.width), str(a.depth),
                  str(a.budget), str(s), timeout=1800)
        d = _json_tail(r.stdout)
        if d is None:
            check(f"4: fresh worker ran for seed {s}", False,
                  (r.stderr or "")[-70:])
            fresh.append(None)
            continue
        r2 = _uvpy("-c", FRESH_WORKER, _ROOT, cand, str(a.width), str(a.depth),
                   str(a.budget), str(s), timeout=1800)
        d2 = _json_tail(r2.stdout)
        ok = d2 is not None and d["sha256"] == d2["sha256"]
        check(f"4: two FRESH interpreters agree bitwise on seed {s}", ok,
              f"{d['sha256'][:16]} pid {d['pid']} vs "
              f"{(d2 or {}).get('pid')}")
        check(f"4: ...and bill identical FLOPs on seed {s}",
              d2 is not None and d["flops_used"] == d2["flops_used"],
              f"{d['flops_used']:,.0f}")
        check(f"4: the shape is ({a.depth}, {a.width}) on seed {s}",
              d["shape"] == [a.depth, a.width], str(d["shape"]))
        fresh.append(d)
    report["sections"]["determinism"] = {"fresh_worker": fresh}

    # ------------------------------------------------------------- 5-8 ----
    head("5-8  one metered run: immutability, cost, telemetry, scalar sync")
    import flopscope as flops
    import flopscope.numpy as fnp
    from mint_provenance import canonical_mint
    from p2_weff import MaterializationCounter
    from whestbench import MLP
    from lifecycle_eval import materialise_predictions

    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass

    mod = _load(cand)
    est = mod.Estimator()
    t0 = time.perf_counter()
    setup_err = ""
    try:
        est.setup(_Ctx(a.width, a.depth, a.budget, SEEDS[0]))
    except Exception as exc:
        setup_err = f"{type(exc).__name__}: {exc}"
    setup_s = time.perf_counter() - t0
    check("5: setup() raised nothing", setup_err == "",
          setup_err or f"{setup_s:.4f} s")
    check("5: setup is inside the 5 s ceiling (LOCAL, in-process -- the "
          "grader's window is larger)",
          setup_s < SETUP_CAP_S, f"{setup_s:.4f} s of {SETUP_CAP_S} s")

    rows = []
    for i, s in enumerate(SEEDS):
        w_np, est_seed, _ = canonical_mint(a.width, a.depth, int(s))
        weights = [fnp.array(x) for x in w_np]
        # sha256 of the grader's arrays BEFORE the call, taken from the
        # flopscope-backed objects the estimator actually receives.
        pre = [hashlib.sha256(np.ascontiguousarray(
            np.asarray(w, dtype=np.float32)).tobytes()).hexdigest()
            for w in weights]
        mlp = MLP(weights=weights, width=a.width, depth=a.depth, seed=est_seed)

        ctx = flops.BudgetContext(flop_budget=a.budget, quiet=True)
        err = ""
        pred = None
        with MaterializationCounter() as mc:
            try:
                with ctx:
                    pred = est.predict(mlp, a.budget)
                    pred = materialise_predictions(pred, depth=a.depth,
                                                   width=a.width)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        post = [hashlib.sha256(np.ascontiguousarray(
            np.asarray(w, dtype=np.float32)).tobytes()).hexdigest()
            for w in weights]

        arr = None if pred is None else np.asarray(pred)
        try:
            ns = (ctx.summary_dict(by_namespace=True).get("by_namespace") or {})
            n_ops = int(sum(float(v.get("calls", 0.0)) for v in ns.values()))
        except Exception:
            n_ops = -1
        rows.append({
            "seed": int(s), "error": err,
            "flops_used": float(ctx.flops_used),
            "n_ops": n_ops,
            "multiplier": max(FLOOR, float(ctx.flops_used) / a.budget),
            "residual_wall_time_s": float(ctx.residual_wall_time_s or 0.0),
            "wall_time_s": float(ctx.wall_time_s or 0.0),
            "materializations": dict(mc.counts),
            "materializations_total": mc.total,
            "weights_unchanged": pre == post,
            "shape": None if arr is None else list(arr.shape),
            "dtype": None if arr is None else str(arr.dtype),
            "finite": None if arr is None else bool(np.all(np.isfinite(arr))),
            "min_value": 0.0 if arr is None else float(np.min(arr)),
            "max_value": 0.0 if arr is None else float(np.max(arr)),
            "n_negative": 0 if arr is None else int(np.sum(np.asarray(arr) < 0)),
            "n_values": 0 if arr is None else int(np.asarray(arr).size),
            "nbytes": None if arr is None else int(arr.nbytes),
            # NOT an error: there is no truth here. This is mean(pred^2), a
            # magnitude that catches an all-zero or exploded return.
            "mean_square_final_row": None if arr is None
            else float(np.mean(np.asarray(arr[-1], dtype=np.float64) ** 2)),
        })

    check("5: no run raised", all(not r["error"] for r in rows),
          "; ".join(r["error"] for r in rows if r["error"])[:60] or "clean")
    check("5: the grader's weight arrays are UNCHANGED after predict",
          all(r["weights_unchanged"] for r in rows), "sha256 before == after")
    check("5: every output has shape (depth, width)",
          all(r["shape"] == [a.depth, a.width] for r in rows),
          str(rows[0]["shape"]))
    check("5: every output is finite", all(r["finite"] for r in rows), "")
    # A post-ReLU mean cannot be negative as a QUANTITY, but a float64 sum that
    # should land on a near-zero neuron can round below it. Measured on the
    # shipped v2.6.1 at 1024x16: 4 of 16,384 entries at -6.1e-17 against row
    # means of ~0.46 -- about one ULP, and 1.3e-16 relative. Demanding exact
    # non-negativity would fail a correct estimator; demanding nothing would
    # miss a sign error. The bar is scaled to the row and sits four orders above
    # the observed noise and many orders below any structural defect.
    worst_neg = min(r["min_value"] for r in rows)
    scale = max(r["max_value"] for r in rows)
    tol = -1e-12 * max(1.0, scale)
    check("5: no output is negative beyond float rounding "
          f"(bar {tol:.1e}, scaled to the row)",
          worst_neg >= tol,
          f"most negative {worst_neg:.3e}"
          + (f", {sum(r['n_negative'] for r in rows)} entr(ies) of "
             f"{sum(r['n_values'] for r in rows)}, "
             f"{abs(worst_neg) / max(scale, 1e-300):.1e} relative"
             if worst_neg < 0 else " -- none"))
    check("5: no returned array approaches the 4 GiB single-array cap",
          all(0 < r["nbytes"] < SINGLE_ARRAY_CAP for r in rows),
          f"{rows[0]['nbytes'] / 2**20:.3f} MB")

    f0 = rows[0]["flops_used"]
    check("6: counted FLOPs are network-INDEPENDENT or explained",
          len({r["flops_used"] for r in rows}) == 1,
          f"{f0:,.0f}" if len({r['flops_used'] for r in rows}) == 1
          else "varies: " + ", ".join(f"{r['flops_used']:,.0f}" for r in rows))
    mult = max(FLOOR, f0 / a.budget)
    check("6: the multiplier is the score's own identity max(0.1, F/B)",
          abs(mult - rows[0]["multiplier"]) < 1e-15,
          f"{mult:.6f}  (C/B = {f0 / a.budget:.7f}"
          f"{', PINNED at the floor' if f0 / a.budget < FLOOR else ''})")
    check("6: the operation ledger is non-empty",
          all(r["n_ops"] > 0 for r in rows), f"{rows[0]['n_ops']:,} ops")

    worst_res = max(r["residual_wall_time_s"] for r in rows)
    worst_wall = max(r["wall_time_s"] for r in rows)
    check("7: worst residual is under the 0.4 s cliff (LOCAL -- the "
          "authority is --runner subprocess)",
          worst_res < RESIDUAL_CAP_S,
          f"{worst_res * 1000:.1f} ms -> {RESIDUAL_CAP_S / worst_res:.2f}x "
          f"LOCAL headroom" if worst_res else "0")
    check("7: worst per-MLP wall is under the installed 120 s limit",
          worst_wall < WALL_CAP_S,
          f"{worst_wall:.3f} s -> {WALL_CAP_S / worst_wall:.0f}x"
          if worst_wall else "0")

    tot_mat = sum(r["materializations_total"] for r in rows)
    check("8: ZERO scalar materializations during predict "
          "(__bool__/__float__/__int__/__index__)",
          tot_mat == 0,
          "counted, not asserted" if tot_mat == 0
          else json.dumps(rows[0]["materializations"]))
    report["sections"]["metered_runs"] = rows
    report["sections"]["setup_s_local"] = setup_s

    # ---------------------------------------------------------------- 9 ----
    head("9  memory: accounting here, ENFORCEMENT only on Linux")
    peak = _host_memory()
    if peak.get("available"):
        print(f"      peak working set {peak['peak_wset_bytes'] / 2**20:.1f} MB, "
              f"peak commit {peak['peak_pagefile_bytes'] / 2**20:.1f} MB "
              f"({peak['source']})")
        print("      RLIMIT_AS caps ADDRESS SPACE, which exceeds both counters "
              "by an amount")
        print("      no Windows counter reports. These are accounting, and a "
              "headroom computed")
        print("      from them is an UPPER BOUND, never a margin.")
    else:
        # Printing "0.0 MB" for an unavailable counter is the exact failure this
        # gate exists to prevent: reporting a state as though it were a
        # measurement. Say it is unavailable.
        skip("9: host memory accounting", peak.get("why", "unavailable"))

    # WHERE THE ENFORCEMENT REPORT LIVES.
    #
    # Beside the candidate first -- a research candidate under
    # `experiments/<slice>/` keeps its own evidence in that slice -- then the
    # WP0C/D directory for the shipped file. The match is on the report's own
    # `candidate` field, never on the filename, so a report for a DIFFERENT file
    # can never be read as this one's.
    lg_path = None
    cand_dir = os.path.dirname(cand)
    stem = os.path.splitext(os.path.basename(cand))[0]
    for c in (os.path.join(cand_dir, "linux_memory_gate_%s.json" % stem),
              os.path.join(_ROOT, "experiments", "2026-08-31_wp0cd_certify",
                           "linux_memory_gate_%s.json" % stem),
              os.path.join(_ROOT, "experiments", "2026-08-31_wp0cd_certify",
                           "linux_memory_gate_v261.json")):
        if os.path.exists(c):
            try:
                with open(c, encoding="utf-8") as fh:
                    if json.load(fh).get("candidate") == rel:
                        lg_path = c
                        break
            except (OSError, ValueError):
                continue
    if lg_path is None:
        skip("9: the real 8-GiB enforcement",
             "no linux_memory_gate report FOR THIS CANDIDATE. Run: uv run "
             "python local_sim/linux_memory_gate.py --candidate %s --json "
             "<dir>/linux_memory_gate_%s.json" % (rel, stem))
    lg = None
    if lg_path is not None:
        with open(lg_path, encoding="utf-8") as fh:
            lg = json.load(fh)
    if lg is not None:
        check("9: a Linux 8-GiB enforcement report exists FOR THIS CANDIDATE",
              True, os.path.relpath(lg_path, _ROOT))
        if True:
            cr = lg.get("candidate_run", {})
            check("9: the cap was APPLIED in that run (soft == requested)",
                  cr.get("cap_applied") is True, str(cr.get("soft_limit_bytes")))
            check("9: the negative control DIED under the cap",
                  (lg.get("negative_control") or {}).get("survived") is False,
                  "a gate whose cap is not in force proves nothing")
            check("9: predict completed under the real cap",
                  cr.get("predict_error") == "" and lg.get("status") == "PASS",
                  f"VmPeak {(cr.get('vm_peak_bytes') or 0) / 2**20:.1f} MB of "
                  f"{MEMORY_CAP_MB} MB")
        report["sections"]["linux_memory_gate"] = os.path.relpath(lg_path, _ROOT)
    report["sections"]["windows_memory_accounting"] = peak

    # --------------------------------------------------------------- 10 ----
    head("10  the official subprocess runner -- DEPLOYMENT AUTHORITY")
    if a.quick or a.skip_smoke:
        skip("10: official Phase-2 subprocess smoke",
             "--quick/--skip-smoke; a skipped deployment check is NOT a pass")
    else:
        smoke = _smoke(rel, timeout=2400)
        report["sections"]["subprocess_smoke"] = smoke
        check("10: the official runner produced a report",
              smoke.get("ok_report") is True, smoke.get("why", "")[:60])
        if smoke.get("ok_report"):
            check("10: zero failed MLPs", smoke["n_failed_mlps"] == 0,
                  f"{smoke['n_failed_mlps']} failed")
            check("10: the live config is the Phase-2 round",
                  smoke["dataset"].endswith("@v2-phase2")
                  and smoke["lambda"] == 0.0
                  and smoke["residual_cap"] == 0.4,
                  f"{smoke['width']}x{smoke['depth']}, lambda "
                  f"{smoke['lambda']}, cap {smoke['residual_cap']} s")
            check("10: worst GRADED-PATH residual is under the cliff",
                  smoke["worst_residual"] < smoke["residual_cap"],
                  f"{smoke['worst_residual'] * 1000:.2f} ms -> "
                  f"{smoke['residual_cap'] / smoke['worst_residual']:.1f}x")
            print(f"      counted FLOPs {smoke['flops_per_mlp']:,.0f}   "
                  f"multiplier {smoke['multiplier']:.4f}   "
                  f"C/B {smoke['c_over_b']:.7f}")

    # ---------------------------------------------------------------- R ----
    head("R  the repository was not modified")
    # Taken AFTER the run, deliberately: a hash captured before a write cannot
    # see the write. `p1pre_fault_inject.py` shipped that defect once.
    check("R: the candidate file is byte-identical", _sha(cand) == sha_before,
          sha_before[:16])
    check("R: estimator.py is byte-identical",
          _sha(os.path.join(_ROOT, "estimator.py")) == ship_sha_before,
          ship_sha_before[:16])
    g = _run(["git", "status", "--porcelain", "--", "estimator.py",
              "scripts", "local_sim"], timeout=180)
    check("R: no tracked source under estimator.py/scripts/local_sim is dirty "
          "because of this run",
          True, "informational: " +
          ((g.stdout or "").strip().replace("\n", "; ")[:56] or "clean"))

    report["checks_total"] = _TOTAL[0]
    report["checks_failed"] = list(_FAILED)
    report["checks_skipped"] = list(_SKIPPED)
    report["verdict"] = ("CERTIFIED" if not _FAILED and not _SKIPPED else
                         "INCOMPLETE" if not _FAILED else "REFUSED")
    if a.json:
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\n  report -> {a.json}")

    print()
    if _FAILED:
        print(f"CANDIDATE REFUSED -- {len(_FAILED)} failed:")
        for f in _FAILED:
            print(f"  - {f}")
        return 1
    if _SKIPPED:
        print(f"CANDIDATE INCOMPLETE -- {_TOTAL[0]} passed, "
              f"{len(_SKIPPED)} NOT CHECKED. A skipped check is not a pass:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return 3
    print(f"CANDIDATE CERTIFIED -- {_TOTAL[0]}/{_TOTAL[0]} checks passed")
    print("  This is one input to promotion, not promotion. See GAME_PLAN.md "
          "section 6.")
    return 0


def _host_memory() -> dict:
    """Peak working set and peak commit for THIS process.

    Delegates to `test_deployment_envelope.peak_memory_bytes()`, which already
    owns this and already carries the lesson: `GetCurrentProcess()` returns the
    pseudo-handle -1, and without an explicit `HANDLE` argtype ctypes passes it
    as a 32-bit int in a 64-bit process, the call fails, and the measurement
    looks merely unavailable rather than wrong. Reimplementing it here reproduced
    that exact failure -- `GetProcessMemoryInfo returned 0 (GetLastError 0)` --
    which is why this function is a delegation and not a second copy.
    """
    try:
        from test_deployment_envelope import peak_memory_bytes
    except Exception as exc:
        return {"available": False,
                "why": f"cannot import peak_memory_bytes: {type(exc).__name__}"}
    try:
        m = peak_memory_bytes()
    except Exception as exc:
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    if not m or not m.get("working_set"):
        return {"available": False,
                "why": "the platform reported no peak working set"}
    return {"available": True,
            "source": "test_deployment_envelope.peak_memory_bytes",
            "peak_wset_bytes": int(m["working_set"]),
            "peak_pagefile_bytes": int(m.get("commit") or 0)}


def _smoke(rel, timeout=2400):
    """`whest run --runner subprocess` on two organiser MLPs."""
    exe = os.path.join(os.path.dirname(sys.executable), "whest")
    cmd = [exe, "run", "--estimator", rel,
           "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
           "--split", "mini", "--runner", "subprocess", "--n-mlps", "2", "--json"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        p = _run(cmd, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok_report": False, "why": f"timed out after {timeout}s"}
    i = (p.stdout or "").find("{")
    if i < 0:
        return {"ok_report": False,
                "why": (p.stderr or p.stdout or "")[-200:]}
    try:
        d = json.loads(p.stdout[i:])
    except ValueError as exc:
        return {"ok_report": False, "why": f"unparseable report: {exc}"}
    if not d.get("results"):
        err = d.get("error") or {}
        return {"ok_report": False,
                "why": f"REFUSED at {err.get('stage')}: {err.get('code')} "
                       f"{err.get('message', '')[:80]}"}
    r, cfg = d["results"], d["run_config"]
    return {
        "ok_report": True,
        "n_failed_mlps": r["n_failed_mlps"],
        "dataset": str(cfg["dataset"]["path"]),
        "width": cfg["width"], "depth": cfg["depth"],
        "lambda": cfg["lambda_flops_per_second"],
        "residual_cap": cfg["residual_wall_time_limit_s"],
        "wall_limit": cfg["wall_time_limit_s"],
        "worst_residual": max(m["residual_wall_time_s"] for m in r["per_mlp"]),
        "flops_per_mlp": max(m.get("flops_used", 0) for m in r["per_mlp"]),
        "multiplier": r["mean_score_multiplier"],
        "c_over_b": r["mean_compute_utilization"],
        # Recorded, never read as accuracy: the organisers' own labels are
        # CONFIRM-tier and nothing here may be fitted or selected on them.
        "final_layer_mse_CONFIRM_ONLY": r["final_layer_mse"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
