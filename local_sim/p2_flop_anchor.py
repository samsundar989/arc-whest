"""The exact op-by-op FLOP anchor: does the local meter bill what the grader bills?

P2-P0 (Phase 0 of `docs/GAME_PLAN.md`). Contract and predictions B1-B4:
`experiments/LOG.md`, 2026-08-24.

WHY THIS FILE EXISTS
--------------------
The official one-row Phase-2 subprocess anchor for the shipped `estimator.py` is

    137,600,326,656 counted FLOPs        (whest run --runner subprocess, v2-phase2)

and every local measurement of the same file on the same shape returns

    137,562,547,200 counted FLOPs        (local_sim, experiments/.../p2_price_delta.json)

a shortfall of **37,779,456 FLOPs (0.02746%)**. A total that is 0.0275% low is not
the problem. The problem is that nobody could name the mechanism, and a meter whose
disagreement with the grader has no name cannot be trusted to price a method it has
never seen -- which is the entire job of the next six slices.

THE MECHANISM THIS FILE TESTS
-----------------------------
`whestbench.domain.MLP.from_row` builds weights with `fnp.array(w)` -- flopscope
arrays. `local_sim/lifecycle_eval.py` (and every probe in this directory) builds
them with `np.asarray(w)` -- raw NumPy. Operations invoked as *methods on a weight*
therefore go through flopscope's meter on the grader and through bare NumPy here.

`estimator.py:868`, the live Phase-2 path, opens with

    w64 = [w.astype(fnp.float64) for w in mlp.weights]

which is exactly such a call. On the grader that is 16 x 1024 x 1024 x 2 FLOPs; in
every local probe written before today it is free.

THE SECOND MECHANISM, found by running this probe
-------------------------------------------------
Fixing the weight representation closes 37,746,688 of the 37,779,456 and leaves
**32,768** — one `asarray` call. It is not ours. `whestbench` 0.16 calls

    materialise_predictions(raw_predictions, depth=..., width=...)

*inside* the metered window (`scoring.py`, in the `with budget_ctx:` block), and
that function ends in `fnp.asarray(predictions, dtype=fnp.float32)`. Its own
docstring says "the shape check and ``fnp.asarray`` are both free ops, so a
correct estimator is billed nothing for it" -- and that is true only if the
estimator already returns float32. The shipped file returns float64, so the
harness's own materialisation is a dtype conversion, and a dtype conversion is
2 FLOPs/element: 16 x 1024 x 2 = **32,768 FLOPs billed to the participant**.

That is a deployment fact about the live runtime, it applies to every candidate
this repository will build, and no local probe written before today modelled it.

WHAT IT MEASURES
----------------
Three arms x two flopscope versions, because three things differ at once and a
repository that conflates them learns nothing:

    arm       weights        in-window materialisation   what it isolates
    --------  -------------  --------------------------  ------------------------
    raw       np.asarray     no                          the historical local
                                                         number (137,562,547,200)
    fnp       fnp.array      no                          the WEIGHT-REPR effect
    grader    fnp.array      yes                         the HARNESS-MATERIALISE
                                                         effect; this arm must
                                                         equal the official anchor

The `fnp.array` weights are built OUTSIDE the `BudgetContext`, exactly as the
grader builds them (`MLP.from_row` runs during dataset load, not during
`predict`), so array construction itself is free on every arm.

WHAT IT DOES NOT DO
-------------------
It does not mint from a registered block, does not read the vault, does not score,
and does not touch `estimator.py`. Counted FLOPs on this dense path are
value-independent, so the probe seed is a bare literal and selects nothing.
`# seed-ledger: no block named` -- see `local_sim/seed_ledger.py::scan_sources`.

It also does not claim anything about wall time. Making weights flopscope-backed
changes local wall time as well as counted FLOPs (contract C5); this file reports
counted FLOPs only.

Run (from the repo root; the two-sided run needs BOTH venvs):
    uv run python local_sim/p2_flop_anchor.py

One side at a time, which is what the driver does internally:
    .venv\\Scripts\\python.exe       local_sim/p2_flop_anchor.py --emit
    .venv-fs012\\Scripts\\python.exe local_sim/p2_flop_anchor.py --emit

Self-check (no venv orchestration, current interpreter only):
    uv run python local_sim/p2_flop_anchor.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert. `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`; prepending this directory makes them shadow the real packages and
# every "counted FLOP" below becomes fiction. `_require_real_flopscope()` asserts
# we got a real one before any number is produced.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0")
OUT = os.path.join(OUT_DIR, "p2_flop_anchor.json")

#: Interpreters to run, keyed by the flopscope version each is EXPECTED to carry.
#: The expectation is asserted against the version the interpreter actually
#: reports, so a re-pinned venv fails loudly instead of relabelling itself. `.venv`
#: is the main environment and follows `pyproject.toml`; `.venv-fs010` exists only
#: to keep the 0.10-vs-0.12 comparison reproducible after the main environment
#: moved to the official Phase-2 stack.
VENVS = {
    "0.10": os.path.join(_ROOT, ".venv-fs010", "Scripts", "python.exe"),
    "0.12": os.path.join(_ROOT, ".venv", "Scripts", "python.exe"),
}

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41

#: A bare probe seed. Counted FLOPs on this dense path are value-independent, so
#: this draws no seed block and decides nothing.
#: seed-ledger: no block named
PROBE_SEED = 1_000_003

#: The official Phase-2 one-row subprocess anchor for the shipped estimator.
#: `whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2
#:  --split mini --runner subprocess`, recorded in docs/CURRENT_STATE.md.
OFFICIAL_ANCHOR_FLOPS = 137_600_326_656

#: The historical local figure, from experiments/2026-08-21_phase2_manifest/.
HISTORICAL_LOCAL_FLOPS = 137_562_547_200

#: The raw official report the anchor is diffed against, op by op. Produced by
#:   whest run --estimator estimator.py \
#:     --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
#:     --runner subprocess --n-mlps 1 --profile --detail full --json
#: A total that matches while an operation does not is a coincidence, not parity,
#: so the op table is the check and the total is the summary.
OFFICIAL_REPORT = os.path.join(OUT_DIR, "official_anchor_1row.json")


def load_official_ops(path: str = OFFICIAL_REPORT) -> dict | None:
    """Per-operation counted FLOPs from an official `whest run --json` report.

    Returns `{op: {"flops": float, "calls": float}}` for MLP 0, or None if the
    report is absent. Tolerates the UTF-8 BOM PowerShell writes.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as fh:
        raw = fh.read()
    doc = json.loads(raw[raw.find("{"):])
    row = doc["results"]["per_mlp"][0]
    ops: dict = {}
    for _ns, bucket in (row["breakdowns"]["estimator"].get("by_namespace") or {}).items():
        for op, rec in (bucket.get("operations") or {}).items():
            e = ops.setdefault(op, {"flops": 0.0, "calls": 0.0})
            e["flops"] += float(rec.get("flop_cost", 0.0))
            e["calls"] += float(rec.get("calls", 0.0))
    return {
        "ops": ops,
        "total": float(row["flops_used"]),
        "whestbench": doc.get("whestbench_version"),
        "run_config": doc.get("run_config"),
        "report": os.path.relpath(path, _ROOT).replace("\\", "/"),
    }


def _require_real_flopscope(flops) -> None:
    """Refuse to produce a number under the numpy shim."""
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"flopscope resolved to the local_sim SHIM at {path}. Every counted "
            "FLOP from this interpreter would be fiction. Fix sys.path (append, "
            "never insert) and re-run."
        )
    if not hasattr(flops, "BudgetContext"):
        raise RuntimeError(f"flopscope at {path} has no BudgetContext.")


def _stub_whestbench() -> None:
    """A minimal `whestbench` so the isolated flopscope-0.12 venv can load the
    candidate. The shipped estimator imports `whestbench` for its base class only;
    nothing it needs from the package participates in metering."""
    if "whestbench" in sys.modules:
        return
    wb = types.ModuleType("whestbench")

    class MLP:
        def __init__(self, weights, width, depth, seed):
            self.weights, self.width = weights, width
            self.depth, self.seed = depth, seed

    class BaseEstimator:
        def setup(self, context):
            pass

        def predict(self, mlp, budget):
            raise NotImplementedError

    wb.MLP = MLP
    wb.BaseEstimator = BaseEstimator
    wb.__path__ = []
    sys.modules["whestbench"] = wb


def _mint_weights():
    """Grader-identical float32 weights for PROBE_SEED, as raw NumPy.

    Routes through `mint_provenance.canonical_mint`, which is proven bit-identical
    to `whestbench.sample_mlp` (docs/VALIDATION.md section 1). Its numpy path does
    not import whestbench, so it runs in the whestbench-less 0.12 venv.
    """
    from mint_provenance import canonical_mint

    weights, estimator_seed, _sample_ss = canonical_mint(WIDTH, DEPTH, PROBE_SEED)
    return [w for w in weights], int(estimator_seed)


def _load_candidate(candidate: str):
    import importlib.util as u

    spec = u.spec_from_file_location("cand_anchor", os.path.join(_ROOT, candidate))
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The three arms: (weight representation, materialise inside the window).
ARMS = {
    "raw": ("np", False),
    "fnp": ("fnp", False),
    "grader": ("fnp", True),
}


def materialise_like_grader(predictions, depth: int, width: int):
    """`whestbench.scoring.materialise_predictions`, in-window, spelled locally.

    Prefers the installed function so this cannot drift from the package. Falls
    back to the two lines the package runs, because the flopscope-0.10 replay venv
    deliberately has no whestbench and must still be able to price this arm.
    """
    import flopscope.numpy as fnp

    try:
        from whestbench.scoring import materialise_predictions  # type: ignore
    except Exception:
        shape = tuple(predictions.shape) if hasattr(predictions, "shape") else ()
        if shape != (depth, width):
            raise ValueError(f"predictions have shape {shape}, expected {(depth, width)}")
        return fnp.asarray(predictions, dtype=fnp.float32)
    return materialise_predictions(predictions, depth=depth, width=width)


def _run_arm(candidate: str, arm: str) -> dict:
    """One arm: per-op counted FLOPs for one `predict` at 1024 x 16.

    See `ARMS`. "raw" is the historical local spelling, "fnp" fixes the weight
    representation, "grader" additionally performs the harness's own in-window
    materialisation.
    """
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    # The shipped file's covariance line legitimately drops flopscope's symmetry
    # tag, and the warning is emitted per call. It says nothing about counted
    # FLOPs, and on Windows PowerShell a native command's stderr is wrapped in an
    # ErrorRecord, so leaving it on makes this probe look like a failing step.
    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass
    _stub_whestbench()

    try:
        weight_repr, materialise = ARMS[arm]
    except KeyError:
        raise ValueError(f"unknown arm {arm!r}") from None

    w_np, est_seed = _mint_weights()
    if weight_repr == "np":
        weights = [np.asarray(w) for w in w_np]
    else:
        # Built OUTSIDE the BudgetContext, exactly as the grader builds them
        # during dataset load. Construction is free on every arm.
        weights = [fnp.array(w) for w in w_np]

    mlp = sys.modules["whestbench"].MLP(
        weights=weights, width=WIDTH, depth=DEPTH, seed=est_seed
    )

    mod = _load_candidate(candidate)
    est = mod.Estimator()

    class _Ctx:
        width, depth, flop_budget = WIDTH, DEPTH, BUDGET
        seed = 0

    # setup() is OFF-budget on the grader (runner.py:150). Mirror that here.
    try:
        est.setup(_Ctx())
    except Exception:
        pass

    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
        pred = est.predict(mlp, BUDGET)
        if materialise:
            pred = materialise_like_grader(pred, DEPTH, WIDTH)
        total = float(ctx.flops_used)
        try:
            ns = ctx.summary_dict(by_namespace=True).get("by_namespace") or {}
        except Exception:
            ns = {}

    ops = {}
    for _nsname, blob in ns.items():
        for op, rec in (blob.get("operations") or {}).items():
            e = ops.setdefault(op, {"flops": 0.0, "calls": 0.0})
            e["flops"] += float(rec.get("flop_cost", 0.0))
            e["calls"] += float(rec.get("calls", 0.0))
    return {"total": total, "budget": BUDGET, "cb": total / BUDGET, "ops": ops}


def _emit(candidate: str) -> dict:
    import flopscope
    import numpy as np

    _require_real_flopscope(flopscope)
    return {
        "flopscope": getattr(flopscope, "__version__", "?"),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "candidate": candidate,
        "arms": {a: _run_arm(candidate, a) for a in ARMS},
    }


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #

def _op_delta(a: dict, b: dict) -> dict:
    """Per-op (flops, calls) that differ between two arms. Empty means identical."""
    moved = {}
    for op in sorted(set(a["ops"]) | set(b["ops"])):
        ra = a["ops"].get(op, {"flops": 0.0, "calls": 0.0})
        rb = b["ops"].get(op, {"flops": 0.0, "calls": 0.0})
        if ra["flops"] != rb["flops"] or ra["calls"] != rb["calls"]:
            moved[op] = {
                "flops": [ra["flops"], rb["flops"]],
                "calls": [ra["calls"], rb["calls"]],
                "d_flops": rb["flops"] - ra["flops"],
                "d_calls": rb["calls"] - ra["calls"],
            }
    return moved


def reconcile(sides: dict) -> dict:
    """Turn the arm table into the numbers the contract asks for (B1-B4)."""
    out: dict = {}

    have_both = "0.10" in sides and "0.12" in sides
    live = sides.get("0.12") or sides.get("0.10")

    # B1 -- the VERSION effect, holding the arm fixed.
    if have_both:
        out["version_effect"] = {
            a: {
                "d_total": sides["0.12"]["arms"][a]["total"]
                - sides["0.10"]["arms"][a]["total"],
                "ops_moved": _op_delta(
                    sides["0.10"]["arms"][a], sides["0.12"]["arms"][a]
                ),
            }
            for a in ARMS
            if a in sides["0.10"]["arms"] and a in sides["0.12"]["arms"]
        }

    # B3 -- the WEIGHT-REPRESENTATION effect, holding version fixed.
    out["weight_repr_effect"] = {
        v: {
            "d_total": s["arms"]["fnp"]["total"] - s["arms"]["raw"]["total"],
            "ops_moved": _op_delta(s["arms"]["raw"], s["arms"]["fnp"]),
        }
        for v, s in sides.items()
    }

    # B4 -- the HARNESS-MATERIALISE effect, holding version and weights fixed.
    out["materialise_effect"] = {
        v: {
            "d_total": s["arms"]["grader"]["total"] - s["arms"]["fnp"]["total"],
            "ops_moved": _op_delta(s["arms"]["fnp"], s["arms"]["grader"]),
        }
        for v, s in sides.items()
        if "grader" in s["arms"]
    }

    # The anchor itself: the full grader spelling under the official version.
    if live is not None:
        raw_total = live["arms"]["raw"]["total"]
        fnp_total = live["arms"]["fnp"]["total"]
        grader_total = live["arms"]["grader"]["total"]
        out["anchor"] = {
            "official": OFFICIAL_ANCHOR_FLOPS,
            "historical_local": HISTORICAL_LOCAL_FLOPS,
            "local_raw": raw_total,
            "local_fnp_weights": fnp_total,
            "local_grader_spelling": grader_total,
            "shortfall_before": OFFICIAL_ANCHOR_FLOPS - raw_total,
            "closed_by_weight_repr": fnp_total - raw_total,
            "closed_by_materialise": grader_total - fnp_total,
            "unexplained_after": OFFICIAL_ANCHOR_FLOPS - grader_total,
            "reconciled": grader_total == float(OFFICIAL_ANCHOR_FLOPS),
            "flopscope": live["flopscope"],
        }

        # G1 -- op by op, not merely in total.
        official = load_official_ops()
        if official is not None:
            moved = _op_delta(
                {"ops": official["ops"]}, {"ops": live["arms"]["grader"]["ops"]}
            )
            out["op_parity"] = {
                "report": official["report"],
                "whestbench": official["whestbench"],
                "official_total": official["total"],
                "local_total": grader_total,
                "n_ops_official": len(official["ops"]),
                "ops_disagreeing": moved,
                "exact": not moved and grader_total == official["total"],
            }
            out["anchor"]["op_parity_exact"] = out["op_parity"]["exact"]
    return out


def _print_report(sides: dict, rec: dict) -> None:
    def _f(x):
        return f"{x:,.0f}"

    print("=" * 74)
    print("P2-P0  EXACT FLOP ANCHOR -- local meter vs the official Phase-2 runner")
    print("=" * 74)
    labels = {
        "raw":    "np weights, no materialise   (historical local)",
        "fnp":    "fnp weights, no materialise  (weight repr fixed)",
        "grader": "fnp weights + materialise    (full grader spelling)",
    }
    for v, s in sorted(sides.items()):
        print(f"\n  flopscope {s['flopscope']}  (numpy {s['numpy']}, py {s['python']})")
        for a in ARMS:
            if a not in s["arms"]:
                continue
            r = s["arms"][a]
            print(f"    {labels[a]:<48} {_f(r['total']):>18}   C/B={r['cb']:.10f}")

    if "version_effect" in rec:
        print("\n  B1  VERSION effect, flopscope 0.10 -> 0.12, arm held fixed")
        for a, d in rec["version_effect"].items():
            n = len(d["ops_moved"])
            print(f"    {a:<7} delta = {_f(d['d_total']):>14} FLOPs   ops moved: {n}")
            for op, m in d["ops_moved"].items():
                print(f"          {op}: {m['flops'][0]:,.0f} -> {m['flops'][1]:,.0f}")

    print("\n  B3  WEIGHT-REPRESENTATION effect, np.asarray -> fnp.array")
    for v, d in sorted(rec["weight_repr_effect"].items()):
        print(f"    flopscope {v}: delta = {_f(d['d_total'])} FLOPs")
        for op, m in d["ops_moved"].items():
            print(
                f"          {op:<18} {m['flops'][0]:>16,.0f} -> {m['flops'][1]:>16,.0f}"
                f"   ({m['d_flops']:+,.0f} over {m['d_calls']:+,.0f} calls)"
            )

    print("\n  B4  HARNESS-MATERIALISE effect, whestbench 0.16 materialise_predictions")
    for v, d in sorted(rec.get("materialise_effect", {}).items()):
        print(f"    flopscope {v}: delta = {_f(d['d_total'])} FLOPs")
        for op, m in d["ops_moved"].items():
            print(
                f"          {op:<18} {m['flops'][0]:>16,.0f} -> {m['flops'][1]:>16,.0f}"
                f"   ({m['d_flops']:+,.0f} over {m['d_calls']:+,.0f} calls)"
            )

    a = rec.get("anchor")
    if a:
        print("\n  B2  RECONCILIATION against the official one-row anchor")
        print(f"    official (whest run, v2-phase2)          {_f(a['official']):>18}")
        print(f"    local, historical spelling               {_f(a['local_raw']):>18}"
              f"   shortfall {_f(a['shortfall_before'])}")
        print(f"      + weight representation                {_f(a['closed_by_weight_repr']):>18}")
        print(f"      + harness materialisation              {_f(a['closed_by_materialise']):>18}")
        print(f"    local, full grader spelling              {_f(a['local_grader_spelling']):>18}"
              f"   unexplained {_f(a['unexplained_after'])}")
        verdict = "RECONCILED, exactly" if a["reconciled"] else "*** NOT RECONCILED ***"
        print(f"\n    {verdict}")

    op = rec.get("op_parity")
    if op is None:
        print("\n  G1  OP-BY-OP PARITY: NOT CHECKED -- no official report at")
        print(f"      {os.path.relpath(OFFICIAL_REPORT, _ROOT)}")
        print("      A matching total is not parity. Produce the report and re-run.")
    else:
        print(f"\n  G1  OP-BY-OP PARITY vs {op['report']} (whestbench {op['whestbench']})")
        print(f"      {op['n_ops_official']} operations compared")
        if op["exact"]:
            print("      every operation name, call count and FLOP figure agrees")
        else:
            for o, m in op["ops_disagreeing"].items():
                print(f"      {o:<18} official {m['flops'][0]:>16,.0f} / "
                      f"{m['calls'][0]:>4,.0f} calls   local {m['flops'][1]:>16,.0f} / "
                      f"{m['calls'][1]:>4,.0f} calls")
    print()


# --------------------------------------------------------------------------- #
# self-test -- properties, not a transcript of today's numbers
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    import numpy as np

    import flopscope as flops
    import flopscope.numpy as fnp

    ok = [0, 0]

    def check(name, cond, detail=""):
        ok[1] += 1
        if cond:
            ok[0] += 1
            print(f"  PASS  {name}   {detail}")
        else:
            print(f"  FAIL  {name}   {detail}")

    print("p2_flop_anchor self-test")

    _require_real_flopscope(flops)
    check("flopscope is the real package, not the local_sim shim",
          not os.path.abspath(flops.__file__).startswith(os.path.abspath(_HERE)),
          flops.__file__)

    # The mechanism, at a size that is instant: a weight-METHOD call is metered
    # on an fnp array and invisible on a raw NumPy array.
    n = 64
    w = np.ones((n, n), dtype=np.float32)
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as c:
        _ = np.asarray(w).astype(np.float64)
        raw_cost = float(c.flops_used)
    wf = fnp.array(w)
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as c:
        _ = wf.astype(fnp.float64)
        fnp_cost = float(c.flops_used)
    check("a raw-NumPy .astype is invisible to the meter", raw_cost == 0.0,
          f"{raw_cost:,.0f} FLOPs")
    check("an fnp .astype IS metered", fnp_cost > 0.0, f"{fnp_cost:,.0f} FLOPs")
    check("and it is priced at 2 FLOPs/element (f32->f64)",
          fnp_cost == 2.0 * n * n, f"{fnp_cost:,.0f} vs {2 * n * n:,}")

    # The probe's own hygiene: the mint is grader-canonical and deterministic.
    w1, s1 = _mint_weights()
    w2, s2 = _mint_weights()
    check("the probe mint is deterministic",
          s1 == s2 and all(np.array_equal(a, b) for a, b in zip(w1, w2)))
    check("the probe mint is float32 at the contest shape",
          len(w1) == DEPTH and w1[0].shape == (WIDTH, WIDTH)
          and w1[0].dtype == np.float32,
          f"{len(w1)} x {w1[0].shape} {w1[0].dtype}")

    # The SECOND mechanism, at a size that is instant: the harness's own in-window
    # materialisation is free on a float32 return and billed on a float64 one.
    d, wd = 4, 16
    p64 = fnp.array(np.ones((d, wd), dtype=np.float64))
    p32 = fnp.array(np.ones((d, wd), dtype=np.float32))
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as c:
        materialise_like_grader(p64, d, wd)
        cost64 = float(c.flops_used)
    with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as c:
        materialise_like_grader(p32, d, wd)
        cost32 = float(c.flops_used)
    check("materialise_predictions is FREE on a float32 return", cost32 == 0.0,
          f"{cost32:,.0f} FLOPs")
    check("materialise_predictions BILLS a float64 return at 2 FLOPs/element",
          cost64 == 2.0 * d * wd, f"{cost64:,.0f} vs {2 * d * wd:,}")

    # The whestbench-less replay venv prices this arm with the local fallback
    # spelling. Where the real function IS importable, the two must cost the same,
    # or the 0.10 side of the version comparison is measuring a different thing.
    try:
        from whestbench.scoring import materialise_predictions as _real
    except Exception:
        print("  SKIP  fallback == package materialise_predictions "
              "(no whestbench in this interpreter)")
    else:
        with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as c:
            _real(p64, depth=d, width=wd)
            real64 = float(c.flops_used)
        check("the local fallback costs exactly what the package costs",
              real64 == cost64, f"{real64:,.0f} vs {cost64:,.0f}")

    # Reconciliation arithmetic is a property of the function, not of today's run.
    def _arm(total):
        return {"total": float(total), "budget": BUDGET, "cb": 0.0, "ops": {}}

    fake = {"0.12": {"flopscope": "0.12.0", "numpy": "x", "python": "x",
                     "arms": {"raw": _arm(HISTORICAL_LOCAL_FLOPS),
                              "fnp": _arm(OFFICIAL_ANCHOR_FLOPS - 32_768),
                              "grader": _arm(OFFICIAL_ANCHOR_FLOPS)}}}
    r = reconcile(fake)
    check("reconcile() reports an exact match as reconciled",
          r["anchor"]["reconciled"] and r["anchor"]["unexplained_after"] == 0)
    check("reconcile() splits the shortfall across both mechanisms",
          r["anchor"]["closed_by_weight_repr"] + r["anchor"]["closed_by_materialise"]
          == r["anchor"]["shortfall_before"],
          f"{r['anchor']['closed_by_weight_repr']:,.0f} + "
          f"{r['anchor']['closed_by_materialise']:,.0f} == "
          f"{r['anchor']['shortfall_before']:,.0f}")
    fake["0.12"]["arms"]["grader"]["total"] = float(OFFICIAL_ANCHOR_FLOPS) - 1.0
    r = reconcile(fake)
    check("reconcile() refuses a one-FLOP miss",
          (not r["anchor"]["reconciled"]) and r["anchor"]["unexplained_after"] == 1)

    print(f"\n  {ok[0]}/{ok[1]}")
    return 0 if ok[0] == ok[1] else 1


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true",
                    help="run this interpreter's side and print JSON")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help=("run ONLY the grader arm on this interpreter and assert "
                          "op-by-op parity with the official report. This is the "
                          "form preflight runs: it needs one venv, not two, so a "
                          "missing replay venv cannot turn the gate green by "
                          "skipping the comparison."))
    ap.add_argument("--candidate", default="estimator.py")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.emit:
        print(json.dumps(_emit(args.candidate)))
        return 0

    if args.gate:
        official = load_official_ops()
        if official is None:
            print("REFUSED: no official report at "
                  f"{os.path.relpath(OFFICIAL_REPORT, _ROOT)}.")
            print("A matching total is not parity, and an ABSENT comparison is not")
            print("a pass. Produce the report and re-run:")
            print("  whest run --estimator estimator.py --profile --detail full "
                  "--json \\")
            print("    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \\")
            print("    --split mini --runner subprocess --n-mlps 1")
            return 3
        arm = _run_arm(args.candidate, "grader")
        moved = _op_delta({"ops": official["ops"]}, {"ops": arm["ops"]})
        print(f"  candidate         {args.candidate}")
        print(f"  official report   {official['report']} "
              f"(whestbench {official['whestbench']})")
        print(f"  official total    {official['total']:>20,.0f}")
        print(f"  local total       {arm['total']:>20,.0f}   "
              f"delta {arm['total'] - official['total']:+,.0f}")
        print(f"  operations        {len(official['ops'])} compared, "
              f"{len(moved)} disagreeing")
        for op, m in moved.items():
            print(f"      {op:<18} official {m['flops'][0]:>16,.0f} / "
                  f"{m['calls'][0]:>4,.0f} calls   local {m['flops'][1]:>16,.0f} / "
                  f"{m['calls'][1]:>4,.0f} calls")
        if moved or arm["total"] != official["total"]:
            print("\n  *** THE LOCAL METER DOES NOT BILL WHAT THE GRADER BILLS ***")
            return 2
        print("\n  op-by-op parity with the official runner: EXACT")
        return 0

    sides = {}
    for tag, py in VENVS.items():
        if not os.path.exists(py):
            print(f"  (skipping flopscope {tag}: no interpreter at {py})")
            continue
        p = subprocess.run(
            [py, os.path.abspath(__file__), "--emit", "--candidate", args.candidate],
            capture_output=True, text=True, cwd=_ROOT,
        )
        if p.returncode != 0:
            print(f"  side {tag} FAILED:\n{p.stderr[-3000:]}")
            return 1
        side = json.loads(p.stdout.strip().splitlines()[-1])
        # A venv that has been re-pinned must fail, not quietly relabel itself.
        got = str(side["flopscope"]).split("+")[0]
        if not got.startswith(tag):
            print(f"  side {tag}: {py} reports flopscope {side['flopscope']}, "
                  f"expected {tag}.x. Re-pin the venv or fix VENVS.")
            return 1
        sides[tag] = side

    if not sides:
        print("no interpreter available")
        return 1

    rec = reconcile(sides)
    _print_report(sides, rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "artifact": "p2_flop_anchor",
            "artifact_version": 1,
            "candidate": args.candidate,
            "shape": {"width": WIDTH, "depth": DEPTH},
            "budget": BUDGET,
            "probe_seed": PROBE_SEED,
            "sides": sides,
            "reconciliation": rec,
            "note": (
                "Counted FLOPs on this dense path are value-independent, so this "
                "probe uses an unregistered probe seed, draws NO seed block, and "
                "selects nothing. Wall time is deliberately not reported: the "
                "weight representation changes it too, and the two must not be "
                "conflated (P2-P0 contract, confounder C5)."
            ),
        }, fh, indent=2)
    print(f"  wrote {args.out}")

    a = rec.get("anchor") or {}
    op = rec.get("op_parity")
    if not a.get("reconciled"):
        return 2
    # An unchecked op table is not a passing op table.
    if op is None or not op["exact"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
