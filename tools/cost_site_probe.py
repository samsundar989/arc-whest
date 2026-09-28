"""Exclusive charged-FLOP profile of unchanged v72 with the real meter.

    python tools/cost_site_probe.py check                 # tiny CPU tests only
    python tools/cost_site_probe.py run --out <directory> # queue: 16 GiB/0 GPU/2 CPU

Harness wrappers observe BudgetContext.flops_used, verified to be an O(1)
integer getter. They do not call a budget summary inside prediction, mutate
arithmetic, materialize candidate scalars, or wrap recursive Strassen leaves.
Inclusive parent counts overlap; ONLY exclusive rows/categories may be summed.
Instrumented local wall/residual are deliberately not server performance evidence.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

# Set before numpy/BLAS import. This standalone CPU probe must not use all cores.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "2"
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tools", ROOT / "local_sim"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))  # Real flopscope must precede the local shim.

EXPECTED_FLOPS = 472202879721
EXPECTED_SHA = "3b6cc39790295d3be60a88f9dd9053915f695bdf0a0da2cf2e99587cce692bac"
BUDGET = 2**41
FUNCTIONS = (
    "_layer", "_advance", "_dslice3", "_dslice21", "_shared_contract_lean",
    "_shared_join_lean", "_full_legs_lean", "_wick_tower", "_p4_sym_sandwich",
    "_p2_compress", "_p2_recon", "_hub_parts", "_p3_compress", "_p3_parts",
    "_wick", "_closure_apply", "_closure_what", "_protect_factors",
    "_corr_delta", "_corr_raw", "_gate_eval", "_append", "_truncate", "_mm",
)


def meter_stack(flops):
    """Use installed distribution metadata; flopscope exports no __version__."""
    dist = importlib.metadata.distribution("flopscope")
    if dist.version != "0.12.1":
        raise RuntimeError(f"this anchor requires flopscope 0.12.1, found {dist.version}")
    module_path = Path(flops.__file__).resolve()
    installed_root = Path(dist.locate_file("flopscope")).resolve()
    if module_path.parent != installed_root:
        raise RuntimeError(f"flopscope module {module_path} is not from {installed_root}")
    return {"flopscope_version": dist.version, "flopscope_file": str(module_path),
            "flopscope_distribution_root": str(installed_root)}


def counter_evidence(budget_class):
    """Refuse unknown implementations instead of assuming the getter is cheap."""
    prop = inspect.getattr_static(budget_class, "flops_used")
    if not isinstance(prop, property):
        raise RuntimeError("BudgetContext.flops_used is not a verified property")
    source = textwrap.dedent(inspect.getsource(prop.fget))
    fn = ast.parse(source).body[0]
    body = fn.body
    expected = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                             attr="_flops_used", ctx=ast.Load())
    if (len(body) != 1 or not isinstance(body[0], ast.Return) or
            ast.dump(body[0].value) != ast.dump(expected)):
        raise RuntimeError("FLOP counter is no longer a direct O(1) integer getter")
    return {"file": inspect.getsourcefile(prop.fget),
            "line": inspect.getsourcelines(prop.fget)[1], "source": source.strip(),
            "verified_constant_time": True}


class Sites:
    """Static source metadata; no runtime tracing or candidate source rewriting."""
    def __init__(self, source):
        self.source = source
        self.lines = source.splitlines()
        self.assignments = []
        self.functions = {}
        tree = ast.parse(source)
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef):
                self.functions[fn.name] = (fn.lineno, fn.end_lineno)
                for node in ast.walk(fn):
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        names = [x.id if isinstance(x, ast.Name) else x.attr
                                 for t in targets for x in ast.walk(t)
                                 if isinstance(x, (ast.Name, ast.Attribute))]
                        self.assignments.append((fn.name, node.lineno, node.end_lineno, names))

    def assignment(self, caller, line):
        matches = [(hi - lo, names) for fn, lo, hi, names in self.assignments
                   if fn == caller and lo <= line <= hi]
        return min(matches, key=lambda x: x[0])[1] if matches else []

    def category(self, function, caller, line):
        target = self.assignment(caller, line)
        if function == "_mm":
            if caller == "_advance":
                if "Q" in target or "_carry_shared" in self.lines[line - 1]:
                    return "old.transport"
                if "A_L" in target or "M" in target:
                    return "recent.transport"
                if "Z" in target:
                    return "history.P2_transport"
                if "Y" in target:
                    return "history.hub_transport"
                if "F" in target:
                    return "history.P3_transport"
                return "history.other_transport"
            if caller == "_shared_contract_lean":
                if "rec" in target:
                    return "recent.D21_readout"
                if "Xold" in target or "old" in target:
                    return "old.D21_readout"
                return "history.unclassified_shared_readout"
            if caller == "_full_legs_lean":
                return "old.reconstruct_legs"
            if caller == "_shared_join_lean":
                return "old.compression"
            if caller == "_p4_sym_sandwich" or (caller == "_layer" and
                                                             ("_X4" in target or "cov_pre" in target)):
                return "covariance.transport"
            if caller in ("_corr_raw", "_corr_delta"):
                return "mean_corrector"
            if caller == "_p2_compress":
                return "birth.P2_compression"
            if caller == "_p3_compress":
                return "birth.P3_compression"
            if caller == "_dslice21":
                return "history.thin_D21_readout"
            return "matrix.other"
        return {
            "_advance": "history.transport_glue",
            "_dslice3": "history.D3_readout_mixed",
            "_dslice21": "history.D21_readout_glue_mixed",
            "_shared_contract_lean": "history.shared_readout_glue",
            "_shared_join_lean": "old.compression",
            "_full_legs_lean": "old.reconstruct_legs",
            "_wick_tower": "history.Wick_scale_mixed",
            "_p4_sym_sandwich": "covariance.transport",
            "_p2_compress": "birth.P2_compression",
            "_p2_recon": "history.P2_readout_mixed",
            "_hub_parts": "history.hub_readout_mixed",
            "_p3_compress": "birth.P3_compression",
            "_p3_parts": "history.P3_readout_mixed",
            "_wick": "closure.Gaussian_moments",
            "_closure_apply": "closure.correction",
            "_closure_what": "closure.correction",
            "_protect_factors": "birth.protected_factors",
            "_corr_delta": "mean_corrector", "_corr_raw": "mean_corrector",
            "_gate_eval": "transport_gate", "_append": "history.bookkeeping",
            "_truncate": "history.bookkeeping",
            "_layer": "layer.other_table_birth_covariance_glue",
            "predict_and_materialize": "prediction.casts_constants_output",
        }.get(function, "other")


class Profiler:
    def __init__(self, counter, sites):
        self.counter, self.sites = counter, sites
        self.stack = []
        self.rows = {}

    @contextmanager
    def span(self, function, caller="harness", line=0, layer=-1, shapes=()):
        category = self.sites.category(function, caller, line)
        key = (function, caller, line, layer, category, shapes)
        frame = {"start": self.counter(), "children": 0, "layer": layer}
        self.stack.append(frame)
        try:
            yield
        finally:
            inclusive = self.counter() - frame["start"]
            exclusive = inclusive - frame["children"]
            assert self.stack.pop() is frame
            if inclusive < 0 or exclusive < 0:
                raise RuntimeError("non-monotone FLOP accounting")
            if self.stack:
                self.stack[-1]["children"] += inclusive
            row = self.rows.setdefault(key, {"function": function, "caller": caller,
                "line": line, "layer": layer, "category": category,
                "matrix_shapes": shapes, "calls": 0, "inclusive_flops": 0, "exclusive_flops": 0})
            row["calls"] += 1
            row["inclusive_flops"] += inclusive
            row["exclusive_flops"] += exclusive

    def wrapper(self, function, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            caller = sys._getframe(1)
            name, line = caller.f_code.co_name, caller.f_lineno
            del caller  # Never retain estimator frames or their arrays.
            layer = self.stack[-1]["layer"] if self.stack else -1
            if function == "_layer":
                layer = int(kwargs.get("layer_index", args[8] if len(args) > 8 else 0))
            shapes = (tuple(args[0].shape), tuple(args[1].shape)) if function == "_mm" else ()
            with self.span(function, name, line, layer, shapes):
                return original(*args, **kwargs)
        return wrapped

    @contextmanager
    def install(self, module):
        originals = {name: getattr(module, name) for name in FUNCTIONS}
        try:
            for name, fn in originals.items():
                setattr(module, name, self.wrapper(name, fn))
            yield
        finally:
            for name, fn in originals.items():
                setattr(module, name, fn)

    def result(self, total):
        if self.stack:
            raise RuntimeError("unclosed profiling span")
        categories = defaultdict(int)
        functions = defaultdict(lambda: {"calls": 0, "inclusive_flops": 0, "exclusive_flops": 0})
        layer_totals = defaultdict(int)
        rows = sorted(self.rows.values(), key=lambda r: (-r["exclusive_flops"], r["function"]))
        for r in rows:
            categories[r["category"]] += r["exclusive_flops"]
            layer_totals[str(r["layer"])] += r["exclusive_flops"]
            for k in ("calls", "inclusive_flops", "exclusive_flops"):
                functions[r["function"]][k] += r[k]
        accounted = sum(categories.values())
        return {"total_flops": total, "exclusive_sum": accounted,
                "exclusive_reconciles": accounted == total,
                "categories": dict(sorted(categories.items(), key=lambda x: -x[1])),
                "functions": dict(functions), "layer_exclusive_totals": dict(layer_totals),
                "call_sites": rows}


def check(candidate):
    """Metadata plus tiny real-meter arithmetic; no network, dataset, or CUDA."""
    import flopscope as flops
    import flopscope.numpy as fnp
    import lifecycle_eval as le

    le._require_real_flopscope()
    stack = meter_stack(flops)
    evidence = counter_evidence(flops.BudgetContext)
    source = candidate.read_text(encoding="utf-8")
    sites = Sites(source)
    assert not set(FUNCTIONS) - set(sites.functions), "missing wrapped function"
    assert "_sb" not in FUNCTIONS, "recursive leaves must not be wrapped"
    # Call-site classification is tied to named assignments, not stale line numbers.
    expected = {("_advance", "A_L"): "recent.transport", ("_advance", "M"): "recent.transport",
                ("_advance", "Z"): "history.P2_transport",
                ("_shared_contract_lean", "rec"): "recent.D21_readout",
                ("_shared_contract_lean", "Xold"): "old.D21_readout",
                ("_shared_contract_lean", "old"): "old.D21_readout",
                ("_layer", "_X4"): "covariance.transport"}
    for (caller, assignment), category in expected.items():
        line = next(lo for fn, lo, hi, names in sites.assignments if fn == caller and assignment in names)
        assert sites.category("_mm", caller, line) == category, (caller, assignment)
    # Instrumented and uninstrumented spelling must have the same real FLOPs.
    x = fnp.ones((3, 3), dtype=fnp.float32)
    plain = flops.BudgetContext(flop_budget=100000, quiet=True)
    with plain:
        y = x + x
        z = y @ x
        z = z * 2.0
    budget = flops.BudgetContext(flop_budget=100000, quiet=True)
    p = Profiler(lambda: budget.flops_used, sites)
    with budget:
        with p.span("predict_and_materialize"):
            y = x + x
            with p.span("_advance", layer=0):
                with p.span("_mm", "_advance", next(lo for fn, lo, hi, names in sites.assignments
                                                      if fn == "_advance" and "A_L" in names), layer=0):
                    z = y @ x
                z = z * 2.0
    result = p.result(budget.flops_used)
    assert budget.flops_used == plain.flops_used
    assert result["exclusive_reconciles"]
    assert result["functions"]["_advance"]["inclusive_flops"] > result["functions"]["_advance"]["exclusive_flops"]
    # Exception unwinding must still account children exactly and empty the stack.
    value = [0]
    p2 = Profiler(lambda: value[0], sites)
    try:
        with p2.span("predict_and_materialize"):
            value[0] += 2
            with p2.span("_advance"):
                value[0] += 5
                raise ValueError("test")
    except ValueError:
        pass
    assert p2.result(7)["exclusive_reconciles"]
    # Exercise installed wrappers and restoration, not just manual spans.
    from types import SimpleNamespace
    toy_source = "def _mm(a, b):\n    return a @ b\ndef _advance(a, b):\n    A_L = _mm(a, b)\n    return A_L * 2.0\n"
    namespace = {}
    exec(compile(toy_source, "tiny_cost_probe_fixture.py", "exec"), namespace)
    toy = SimpleNamespace(**{name: namespace.get(name, lambda *a, **kw: None) for name in FUNCTIONS})
    original_mm, original_advance = toy._mm, toy._advance
    wrapped_budget = flops.BudgetContext(flop_budget=100000, quiet=True)
    p3 = Profiler(lambda: wrapped_budget.flops_used, Sites(toy_source))
    with p3.install(toy):
        # The toy functions use their definition's globals, as a real module does.
        namespace["_mm"] = toy._mm
        with wrapped_budget:
            with p3.span("predict_and_materialize"):
                toy._advance(x, x)
    assert toy._mm is original_mm and toy._advance is original_advance
    wrapped_result = p3.result(wrapped_budget.flops_used)
    assert wrapped_result["exclusive_reconciles"]
    assert wrapped_result["categories"]["recent.transport"] > 0
    assert wrapped_result["functions"]["_mm"]["calls"] == 1
    assert wrapped_result["functions"]["_advance"]["calls"] == 1
    return {"pass": True, "stack": stack, "counter": evidence, "wrapped_functions": FUNCTIONS,
            "tiny_real_meter_flops": int(budget.flops_used),
            "same_as_uninstrumented": True, "exclusive_reconciles": True,
            "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}


def markdown(result):
    total = result["total_flops"]
    lines = ["# Current v72 charged-FLOP profile", "",
        f"Total: **{total:,} FLOPs**. Expected anchor: {EXPECTED_FLOPS:,}. "
        f"Exclusive sum reconciles: {result['exclusive_reconciles']}.", "",
        "Categories sum child-subtracted exclusive costs. Inclusive function totals overlap and "
        "must not be added. Recent/old labels identify measured matrix calls; mixed elementwise "
        "readout and Wick work remain explicitly mixed. This does not infer costs from tensor sizes "
        "or reuse an old version's percentage. No instrumented wall or residual value is a server estimate.", "",
        "| category | exclusive FLOPs | share |", "|---|---:|---:|"]
    for k, v in result["categories"].items():
        lines.append(f"| {k} | {v:,} | {100*v/total:.3f}% |")
    lines += ["", "| function | calls | inclusive FLOPs | exclusive FLOPs |", "|---|---:|---:|---:|"]
    for name, row in sorted(result["functions"].items(), key=lambda x: -x[1]["inclusive_flops"]):
        lines.append(f"| {name} | {row['calls']} | {row['inclusive_flops']:,} | {row['exclusive_flops']:,} |")
    lines += ["", "Full JSON retains layer, original source call-site line, matrix shapes, and both "
              "counts. Prediction initialization/casts/output and materialization are covered by the "
              "root span, so no unassigned FLOPs are silently omitted."]
    return "\n".join(lines) + "\n"


def run(a):
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    import lifecycle_eval as le
    import panels
    import real_data as rd
    from _cost_worker import _MLP
    from _peakrss import peak_rss_mib
    from p2_weff import MaterializationCounter

    le._require_real_flopscope()
    stack = meter_stack(flops)
    candidate = (ROOT / a.candidate).resolve()
    source_bytes = candidate.read_bytes()
    sha = hashlib.sha256(source_bytes).hexdigest()
    if sha != EXPECTED_SHA:
        raise RuntimeError("candidate does not match the verified current v72 source hash")
    counter = counter_evidence(flops.BudgetContext)
    mod, _ = le.load_candidate(str(candidate))
    split, idx, seed, name = panels.nets("dev")[a.index]
    w_np, truth, _ = rd.get(idx, split, dtype=np.float32)
    depth, width = w_np.shape[:2]
    if (width, depth) != (1024, 16):
        raise RuntimeError("the current FLOP anchor is for 1024x16")
    est = mod.Estimator()
    ctx = le.SetupContext(le.LifecycleSpec(width=width, depth=depth, flop_budget=BUDGET, seed=0))
    setup_start = time.perf_counter()
    est.setup(ctx)
    setup_s = time.perf_counter() - setup_start
    mlp = _MLP(le.build_weights(list(w_np), "f64_readonly", fnp), width, depth, int(seed))
    del w_np
    budget = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
    profiler = Profiler(lambda: budget.flops_used, Sites(source_bytes.decode("utf-8")))
    started = time.perf_counter()
    error = ""
    with profiler.install(mod), MaterializationCounter() as mc:
        try:
            with budget:
                with profiler.span("predict_and_materialize"):
                    raw = est.predict(mlp, BUDGET)
                    raw = le.materialise_predictions(raw, depth=depth, width=width)
            le.validate_predictions(raw, depth=depth, width=width)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    # Summary is called ONCE, after predict and after every profiling wrapper.
    summary = budget.summary_dict(by_namespace=True)
    ops = sum(int(v.get("calls", 0)) for v in (summary.get("by_namespace") or {}).values())
    result = profiler.result(int(budget.flops_used))
    unchanged = hashlib.sha256(candidate.read_bytes()).hexdigest() == sha
    result.update(candidate=a.candidate, candidate_sha256=sha, counter=counter,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  **stack,
                  net=name, split=split, index=idx, seed=int(seed),
                  weights_mode="f64_readonly", ops=ops, setup_s=setup_s,
                  local_instrumented_elapsed_s=time.perf_counter()-started,
                  peak_rss_mib=peak_rss_mib(), materializations=sum(mc.counts.values()),
                  materialization_detail=mc.counts, error=error,
                  candidate_source_unchanged=unchanged,
                  expected_flops=EXPECTED_FLOPS,
                  exact_anchor_match=int(budget.flops_used) == EXPECTED_FLOPS)
    result["pass"] = (not error and unchanged and result["exclusive_reconciles"] and
                      result["exact_anchor_match"] and result["materializations"] == 0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("pass", "total_flops", "exclusive_sum",
                    "exact_anchor_match", "ops", "materializations", "peak_rss_mib", "error")}))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--candidate", default="estimator_k3_v72_gt.py")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "experiments/2026-09-28_cost_sites"))
    args = parser.parse_args()
    if args.command == "check":
        result = check((ROOT / args.candidate).resolve())
        print(json.dumps(result, indent=2))
        if "--out" in sys.argv:
            dest = Path(args.out)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "cpu_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        sys.exit(run(args))
