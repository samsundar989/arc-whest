"""flopscope 0.10 vs 0.12: where the price table moved, and where it did not.

P2-CP0 follow-up, opened by an independent Skeptic. The checkpoint originally
claimed "0.10 and 0.12 price our primitives identically" on the evidence of three
primitives. Three primitives agreeing is not a price table agreeing, and the
Skeptic found two families that genuinely differ. This module replaces the claim
with a measurement.

It runs the SAME probe under two interpreters and diffs:

    .venv         flopscope 0.10.0   (what every Phase-1 number was priced under)
    .venv-fs012   flopscope 0.12.0   (what Phase 2 was baked under, released
                                      2026-08-21, the morning round 1429 opens)

Two layers:

  A. **whole-estimator**: load a candidate file, run `predict` on one canonically
     minted 1024 x 16 network under a real `BudgetContext`, and compare the
     per-operation table op by op. This is the number that actually decides
     whether a `C/B` measured here transfers.
  B. **primitive sweep**: the operations a K2/Edgeworth/sampling arm is likely to
     spell, at both float32 and float64, with and without a Python scalar
     operand. Only the cells that MOVE are reported -- a table of 60 identical
     rows says nothing.

Why the scalar cases are in the sweep: flopscope 0.12's release notes say a
Python scalar mixed with a narrow array now follows NEP-50 weak promotion instead
of being widened to 64-bit first. That is exactly the spelling a mixed-precision
covariance path produces, and it is the repository's own headline cost lever
("a float32 covariance would halve C/B"). If 0.10 over-charges it, a float32
saving measured on 0.10 is understated -- conservative, but wrong.

Run (from the repo root, needs BOTH venvs):
    uv run python local_sim/p2_price_delta.py

Or one side at a time, which is what the driver does internally:
    .venv\\Scripts\\python.exe      local_sim/p2_price_delta.py --emit
    .venv-fs012\\Scripts\\python.exe local_sim/p2_price_delta.py --emit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-21_phase2_manifest")
OUT = os.path.join(OUT_DIR, "p2_price_delta.json")

VENVS = {
    "0.10": os.path.join(_ROOT, ".venv", "Scripts", "python.exe"),
    "0.12": os.path.join(_ROOT, ".venv-fs012", "Scripts", "python.exe"),
}

WIDTH, DEPTH = 1024, 16
SEED = 1_000_003          # a bare probe seed: counted FLOPs are value-independent
                          # for a dense path, so this draws no block and decides
                          # nothing. seed-ledger: no block named.


# --------------------------------------------------------------------------- #
# side A: the whole estimator
# --------------------------------------------------------------------------- #

def _estimator_ops(candidate: str) -> dict:
    """Per-op counted FLOPs for one `predict` at 1024 x 16.

    Runs with a STUB whestbench on the path so the isolated flopscope-0.12 venv
    (which deliberately has no whestbench) can execute the same candidate file.
    """
    import types

    import numpy as np

    if "whestbench" not in sys.modules:                     # minimal shim
        wb = types.ModuleType("whestbench")

        class MLP:                                          # noqa: D401
            def __init__(self, weights, width, depth, seed):
                self.weights, self.width = weights, width
                self.depth, self.seed = depth, seed

        class BaseEstimator:                                # noqa: D401
            def setup(self, context):
                pass

            def predict(self, mlp, budget):
                raise NotImplementedError

        wb.MLP = MLP
        wb.BaseEstimator = BaseEstimator
        wb.__path__ = []                                    # make it a package
        sys.modules["whestbench"] = wb

    import flopscope as flops
    import importlib.util as u

    scale = float(np.sqrt(2.0 / WIDTH))
    ss = np.random.SeedSequence(SEED).spawn(3)
    rng = np.random.default_rng(ss[0])
    weights = [(rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
               for _ in range(DEPTH)]
    mlp = sys.modules["whestbench"].MLP(
        weights=weights, width=WIDTH, depth=DEPTH,
        seed=int(ss[2].generate_state(1)[0]))

    spec = u.spec_from_file_location("cand", os.path.join(_ROOT, candidate))
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    est = mod.Estimator()

    class _Ctx:
        width, depth, flop_budget = WIDTH, DEPTH, 2 ** 41
        seed = 0
    try:
        est.setup(_Ctx())
    except Exception:
        pass

    budget = 2 ** 41
    with flops.BudgetContext(flop_budget=budget, quiet=True) as ctx:
        est.predict(mlp, budget)
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
    return {"total": total, "budget": budget, "cb": total / budget, "ops": ops}


# --------------------------------------------------------------------------- #
# side B: the primitive sweep
# --------------------------------------------------------------------------- #

def _primitives() -> dict:
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np

    n = 256
    f32 = np.ones((n, n), dtype=np.float32)
    f64 = np.ones((n, n), dtype=np.float64)
    mask = np.ones((n, n), dtype=bool)
    v32 = np.ones(n, dtype=np.float32)

    cases = {
        # dense linear algebra -- the K2 backbone
        "matmul f32": lambda: fnp.asarray(f32) @ fnp.asarray(f32),
        "matmul f64": lambda: fnp.asarray(f64) @ fnp.asarray(f64),
        "eigh f64": lambda: fnp.linalg.eigh(fnp.asarray(f64 + np.eye(n))),
        "qr f64": lambda: fnp.linalg.qr(fnp.asarray(f64), mode="complete"),
        "outer f64": lambda: fnp.outer(fnp.asarray(v32.astype(np.float64)),
                                       fnp.asarray(v32.astype(np.float64))),
        "diag f64": lambda: fnp.diag(fnp.asarray(v32.astype(np.float64))),
        "diagonal f64": lambda: fnp.diagonal(fnp.asarray(f64)),
        # the Gaussian closure's transcendentals
        "norm.cdf f64": lambda: flops.stats.norm.cdf(fnp.asarray(f64)),
        "norm.pdf f64": lambda: flops.stats.norm.pdf(fnp.asarray(f64)),
        "norm.ppf f64": lambda: flops.stats.norm.ppf(fnp.asarray(f64 * 0.5)),
        "sqrt f32": lambda: fnp.sqrt(fnp.asarray(f32)),
        "sqrt f64": lambda: fnp.sqrt(fnp.asarray(f64)),
        "exp f64": lambda: fnp.exp(fnp.asarray(f64 * 0.0)),
        # symmetry tagging -- the 0.12 notes call this out explicitly
        "as_symmetric f32": lambda: flops.as_symmetric(
            fnp.asarray(np.eye(n, dtype=np.float32)), symmetry=(0, 1)),
        "as_symmetric f64": lambda: flops.as_symmetric(
            fnp.asarray(np.eye(n)), symmetry=(0, 1)),
        # reductions, incl. the nan-family the 0.12 notes reprice
        "sum f32": lambda: fnp.sum(fnp.asarray(f32)),
        "sum f64": lambda: fnp.sum(fnp.asarray(f64)),
        "mean f32": lambda: fnp.mean(fnp.asarray(f32)),
        "nansum f32": lambda: fnp.nansum(fnp.asarray(f32)),
        "nansum f64": lambda: fnp.nansum(fnp.asarray(f64)),
        "nanmean f32": lambda: fnp.nanmean(fnp.asarray(f32)),
        "nanmax f32": lambda: fnp.nanmax(fnp.asarray(f32)),
        # PYTHON-SCALAR MIXING -- the NEP-50 change, and the reason this file
        # exists. This is the spelling a mixed-precision path produces.
        "where(mask, f32, 0.0 py)": lambda: fnp.where(fnp.asarray(mask),
                                                      fnp.asarray(f32), 0.0),
        "where(mask, f64, 0.0 py)": lambda: fnp.where(fnp.asarray(mask),
                                                      fnp.asarray(f64), 0.0),
        "maximum(f32, 0.0 py)": lambda: fnp.maximum(fnp.asarray(f32), 0.0),
        "maximum(f64, 0.0 py)": lambda: fnp.maximum(fnp.asarray(f64), 0.0),
        "f32 * 2.0 py": lambda: fnp.asarray(f32) * 2.0,
        "f32 * np.float32(2)": lambda: fnp.asarray(f32) * np.float32(2),
        "clip(f32, 0.0 py, None)": lambda: fnp.clip(fnp.asarray(f32), 0.0, None),
        "astype f32->f64": lambda: fnp.asarray(f32).astype(fnp.float64),
    }

    out = {}
    for name, fn in cases.items():
        try:
            with flops.BudgetContext(flop_budget=int(1e15), quiet=True) as b:
                fn()
                out[name] = float(b.flops_used)
        except Exception as exc:
            out[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return out


def _emit(candidate: str) -> dict:
    import importlib.metadata as md

    return {
        "flopscope": md.version("flopscope"),
        "python": sys.version.split()[0],
        "estimator": _estimator_ops(candidate),
        "primitives": _primitives(),
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true",
                    help="internal: print one side's JSON to stdout")
    ap.add_argument("--candidate", default="estimator.py")
    a = ap.parse_args()

    if a.emit:
        print("@@JSON@@" + json.dumps(_emit(a.candidate)))
        return 0

    sides = {}
    for label, exe in VENVS.items():
        if not os.path.exists(exe):
            raise SystemExit(f"missing interpreter for flopscope {label}: {exe}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run(
            [exe, os.path.abspath(__file__), "--emit", "--candidate", a.candidate],
            capture_output=True, text=True, env=env, cwd=_ROOT,
        )
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.startswith("@@JSON@@")), None)
        if line is None:
            raise SystemExit(f"probe failed under {label}:\n{r.stdout}\n{r.stderr}")
        sides[label] = json.loads(line[len("@@JSON@@"):])

    lo, hi = sides["0.10"], sides["0.12"]
    print("=" * 78)
    print("FLOPSCOPE PRICE DELTA  --  0.10.0  vs  0.12.0")
    print("=" * 78)
    print(f"  candidate    {a.candidate}   at {WIDTH} x {DEPTH}")
    print(f"  installed    {lo['flopscope']}  and  {hi['flopscope']}")
    print()

    # ---- A. the whole estimator -------------------------------------------
    el, eh = lo["estimator"], hi["estimator"]
    same_total = el["total"] == eh["total"]
    print("A. WHOLE ESTIMATOR, one predict()")
    print(f"     0.10  {el['total']:>20,.0f}   C/B = {el['cb']:.9f}")
    print(f"     0.12  {eh['total']:>20,.0f}   C/B = {eh['cb']:.9f}")
    op_diffs = {
        k: (el["ops"].get(k, {}).get("flops"), eh["ops"].get(k, {}).get("flops"))
        for k in sorted(set(el["ops"]) | set(eh["ops"]))
        if el["ops"].get(k, {}).get("flops") != eh["ops"].get(k, {}).get("flops")
    }
    print(f"     ops compared: {len(set(el['ops']) | set(eh['ops']))}"
          f"   ops that MOVED: {len(op_diffs)}")
    for k, (x, y) in op_diffs.items():
        print(f"       {k:<28} {x} -> {y}")
    if same_total and not op_diffs:
        print("     -> IDENTICAL to the unit, op by op. A C/B measured on 0.10")
        print("        transfers to 0.12 FOR THIS OP MIX. That is a statement")
        print("        about this candidate, not about the price table.")
    print()

    # ---- B. the primitive sweep -------------------------------------------
    print("B. PRIMITIVE SWEEP -- only the cells that MOVED")
    moved = {k: (lo["primitives"][k], hi["primitives"][k])
             for k in sorted(lo["primitives"])
             if lo["primitives"][k] != hi["primitives"].get(k)}
    if not moved:
        print("     none of the probed primitives changed price")
    for k, (x, y) in moved.items():
        rel = (f"{y / x:.2f}x" if isinstance(x, float) and isinstance(y, float)
               and x else "")
        print(f"     {k:<28} {x!s:>14} -> {y!s:>14}   {rel}")
    print()
    print(f"     {len(lo['primitives'])} primitives probed, {len(moved)} moved.")
    print("     A primitive that moved is a primitive whose 0.10 cost may NOT")
    print("     be quoted for Phase 2. Any arm spelling one must be priced on")
    print("     0.12 and reported with both readings.")
    print("=" * 78)

    art = {
        "artifact": "p2_price_delta",
        "artifact_version": 1,
        "candidate": a.candidate,
        "shape": {"width": WIDTH, "depth": DEPTH},
        "sides": sides,
        "estimator_total_identical": bool(same_total),
        "estimator_ops_moved": op_diffs,
        "primitives_moved": moved,
        "n_primitives_probed": len(lo["primitives"]),
        "note": "Counted FLOPs on a dense path are value-independent, so this "
                "probe uses an unregistered probe seed and draws NO seed block. "
                "It selects nothing.",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            old = json.load(fh)
        if json.dumps(old, sort_keys=True) != json.dumps(art, sort_keys=True):
            raise SystemExit(
                f"REFUSED: {os.path.relpath(OUT, _ROOT)} exists with different "
                "content. Delete it deliberately if the stack really moved."
            )
        print(f"  artifact unchanged -> {os.path.relpath(OUT, _ROOT)}")
        return 0
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(art, indent=2) + "\n")
    print(f"  artifact -> {os.path.relpath(OUT, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
