"""Focused, truth-free tests for allocator-bounded edge-suite receipts."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import weakref

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import certify_candidate as cert  # noqa: E402
import edge_cases as edge  # noqa: E402

FAILED = []
TOTAL = 0


def check(name, ok):
    global TOTAL
    TOTAL += 1
    print("  %s  %s" % ("PASS" if ok else "FAIL", name))
    if not ok:
        FAILED.append(name)


def allocator_fixture():
    if sys.platform == "win32":
        return {"platform": "win32", "gc_collected": 0,
                "mechanism": "ucrtbase._heapmin", "available": True,
                "returncode": 0, "succeeded": True}
    if sys.platform.startswith("linux"):
        return {"platform": sys.platform, "gc_collected": 0,
                "mechanism": "libc.malloc_trim(0)", "available": True,
                "returncode": 1, "succeeded": True}
    return {"platform": sys.platform, "gc_collected": 0,
            "mechanism": "gc_only", "available": False,
            "returncode": None, "succeeded": True}


def meter_fixture():
    return {"flop_budget": cert._implicit_default_budget(),
            "flops_used": 1, "n_ops": 1, "status": "returned",
            "output_sha256": "d" * 64,
            "allocator_release": allocator_fixture()}


def fixture(group="normal", count=1):
    candidate_sha = "a" * 64
    edge_sha = "b" * 64
    payload = {
        "artifact": "edge_cases_group",
        "artifact_version": cert.EDGE_GROUP_RECEIPT_VERSION,
        "candidate": "candidate.py",
        "candidate_sha256_before": candidate_sha,
        "candidate_sha256_after": candidate_sha,
        "edge_cases_sha256": edge_sha,
        "meter_reset_policy": cert.EDGE_METER_RESET_POLICY,
        "metered_predictions": [
            meter_fixture() for _ in range(
                dict(cert.EDGE_CASE_COUNTS)[group]
                * cert.EDGE_PREDICTIONS_PER_CASE[group])],
        "group_registry": [
            {"group": name, "expected_result_count": expected}
            for name, expected in cert.EDGE_GROUP_REGISTRY
        ],
        "world": {
            "phase2": True, "real_flopscope": True,
            "weights_mode": "f64_readonly",
            "width": cert.WIDTH, "depth": cert.DEPTH,
            "budget": cert.BUDGET,
            "flopscope_file": cert._expected_edge_instrument()["path"],
            "flopscope_version": cert._expected_edge_instrument()["version"],
        },
        "group": group,
        "expected_result_count": count,
        "results": [True] * count,
        "passed": count,
        "all_passed": True,
    }
    kwargs = {
        "group": group, "expected_count": count,
        "candidate": "candidate.py", "candidate_sha256": candidate_sha,
        "edge_cases_sha256": edge_sha, "weights_mode": "f64_readonly",
    }
    return payload, kwargs


def case_fixture(case="normal:0", group="normal", case_index=0):
    payload, group_kwargs = fixture(group=group, count=1)
    payload.pop("expected_result_count")
    payload.pop("group")
    payload.update({
        "artifact": "edge_cases_case",
        "artifact_version": cert.EDGE_CASE_RECEIPT_VERSION,
        "case_registry": [
            {"case": name, "group": logical, "case_index": index}
            for name, logical, index in cert.EDGE_CASE_REGISTRY
        ],
        "case": case, "logical_group": group, "case_index": case_index,
        "details": {},
        "metered_predictions": [meter_fixture() for _ in range(
            cert.EDGE_PREDICTIONS_PER_CASE[group])],
    })
    kwargs = {key: value for key, value in group_kwargs.items()
              if key not in ("expected_count",)}
    kwargs.update({"case": case, "group": group, "case_index": case_index})
    return payload, kwargs


def budget_rows(log2ns, *, invalid_index=None, error_index=None):
    rows = []
    for index, (budget, log2n) in enumerate(zip(cert.EDGE_BUDGET_LADDER,
                                                log2ns)):
        finite = index != invalid_index
        error = "RuntimeError: injected" if index == error_index else ""
        rows.append({
            "case": "budget_monotone:%d" % index,
            "case_index": index, "result": bool(finite and not error),
            "details": {
                "budget": budget, "has_ladder": True, "log2n": log2n,
                "finite": finite, "error": error,
            },
        })
    return rows


def main():
    check("producer and consumer registries are exact and equal",
          edge.GROUP_REGISTRY == cert.EDGE_GROUP_REGISTRY
          and edge.CASE_COUNTS == cert.EDGE_CASE_COUNTS
          and edge.CASE_REGISTRY == cert.EDGE_CASE_REGISTRY
          and sum(count for _, count in edge.GROUP_REGISTRY) == 30
          and len({group for group, _ in edge.GROUP_REGISTRY})
          == len(edge.GROUP_REGISTRY))

    import flopscope
    import flopscope.numpy as fnp
    reset_events = []
    context_refs = []
    original_reset = flopscope.budget_reset
    original_current = flopscope.current_budget
    original_context = flopscope.BudgetContext
    original_release = edge._release_allocator

    class ProbeEstimator:
        def predict(self, mlp, budget):
            reset_events.append(("predict", id(self), id(mlp), budget))
            return "prediction"

    probe_estimator = ProbeEstimator()
    probe_mlp = object()
    original_backend = edge.WEIGHTS_BACKEND

    class Snapshot:
        flop_budget = int(1e15)

    class ProbeBudgetContext:
        def __init__(self, flop_budget, quiet):
            self.flop_budget, self.flops_used = flop_budget, 7
            self.op_log = [1, 2]
            reset_events.append(("context", flop_budget, quiet))
            context_refs.append(weakref.ref(self))

        def __enter__(self):
            reset_events.append(("enter",))
            return self

        def __exit__(self, *_args):
            reset_events.append(("exit",))

    try:
        flopscope.current_budget = lambda: Snapshot()
        flopscope.BudgetContext = ProbeBudgetContext
        flopscope.budget_reset = lambda: reset_events.append(
            ("reset", context_refs[-1]() is None))
        edge._release_allocator = lambda _collected=None: (
            reset_events.append(("release",)) or allocator_fixture())
        edge.WEIGHTS_BACKEND = "flopscope"
        edge.METER_PREDICTIONS.clear()
        first = edge._predict(probe_estimator, probe_mlp, 11)
        second = edge._predict(probe_estimator, probe_mlp, 12)
        expected_one = [
            ("context", int(1e15), True), ("enter",),
            ("predict", id(probe_estimator), id(probe_mlp), 11),
            ("exit",), ("reset", True), ("release",),
        ]
        expected_two = [
            ("context", int(1e15), True), ("enter",),
            ("predict", id(probe_estimator), id(probe_mlp), 12),
            ("exit",), ("reset", True), ("release",),
        ]
        check("fresh contexts bound prediction then teardown and preserve objects",
              first == second == "prediction"
              and reset_events == expected_one + expected_two
              and len(edge.METER_PREDICTIONS) == 2
              and all(row["flop_budget"] == int(1e15)
                      and row["flops_used"] == 7 and row["n_ops"] == 2
                      for row in edge.METER_PREDICTIONS))
        reset_events.clear()
        edge.WEIGHTS_BACKEND = "numpy"
        edge._predict(probe_estimator, probe_mlp, 13)
        check("historical numpy prediction performs no instrument lifecycle",
              reset_events == [
                  ("predict", id(probe_estimator), id(probe_mlp), 13)])
    finally:
        edge.WEIGHTS_BACKEND = original_backend
        flopscope.budget_reset = original_reset
        flopscope.current_budget = original_current
        flopscope.BudgetContext = original_context
        edge._release_allocator = original_release

    from flopscope._budget import _reset_global_default, get_active_budget

    class BudgetProbeEstimator:
        def predict(self, _mlp, budget):
            x = fnp.array([float(budget) + 2.0])
            y = x + x
            active = flopscope.current_budget()
            return edge.np.asarray(
                [float(y[0]), float(active.flop_budget),
                 float(active.flops_used)])

    def isolated_control(estimator, budget, cap):
        context = flopscope.BudgetContext(flop_budget=cap, quiet=True)
        result = None
        caught = None
        try:
            with context:
                result = estimator.predict(probe_mlp, budget)
        except Exception as exc:
            caught = exc
        output_sha = None if result is None else hashlib.sha256(
            edge.np.ascontiguousarray(edge.np.asarray(result)).tobytes()).hexdigest()
        row = {"flops_used": int(context.flops_used),
               "n_ops": len(context.op_log), "output_sha256": output_sha}
        context = None
        import gc
        gc.collect()
        original_reset()
        cert._release_allocator(0)
        return result, caught, row

    parity = True
    parity_estimator = BudgetProbeEstimator()
    edge.WEIGHTS_BACKEND = "flopscope"
    edge.METER_PREDICTIONS.clear()
    try:
        for tiny_budget in (0, 1):
            _reset_global_default()
            original_reset()
            history = fnp.array([1.0])
            _ = history + history
            implicit = get_active_budget()
            implicit_before = len(implicit.op_log)
            implicit_flops_before = int(implicit.flops_used)
            control, control_error, control_row = isolated_control(
                parity_estimator, tiny_budget, int(implicit.flop_budget))
            _reset_global_default()
            original_reset()
            history = fnp.array([1.0])
            _ = history + history
            implicit = get_active_budget()
            implicit_before_helper = len(implicit.op_log)
            implicit_flops_before_helper = int(implicit.flops_used)
            new = edge._predict(parity_estimator, probe_mlp, tiny_budget)
            helper_row = edge.METER_PREDICTIONS[-1]
            parity &= (control_error is None
                       and edge.np.array_equal(control, new)
                       and helper_row["output_sha256"]
                       == control_row["output_sha256"]
                       and helper_row["flops_used"] == control_row["flops_used"]
                       and helper_row["n_ops"] == control_row["n_ops"]
                       and len(implicit.op_log) == implicit_before_helper
                       and int(implicit.flops_used)
                       == implicit_flops_before_helper
                       and implicit_before > 0 and implicit_flops_before > 0
                       and implicit_before_helper > 0
                       and implicit_flops_before_helper > 0)
        check("helper matches no-reset explicit control and isolates global log",
              parity and len(edge.METER_PREDICTIONS) == 2
              and all(cert._allocator_release_valid(
                  row["allocator_release"])
                      for row in edge.METER_PREDICTIONS))

        class ExhaustEstimator:
            def predict(self, _mlp, _budget):
                x = fnp.array([1.0, 2.0])
                return x + x

        old_env = os.environ.get("FLOPSCOPE_DEFAULT_BUDGET")
        os.environ["FLOPSCOPE_DEFAULT_BUDGET"] = "5"
        try:
            _reset_global_default()
            original_reset()
            _, control_error, control_row = isolated_control(
                ExhaustEstimator(), 0, 5)
            _reset_global_default()
            original_reset()
            _ = fnp.array([0.0])
            helper_error = None
            try:
                edge._predict(ExhaustEstimator(), probe_mlp, 0)
            except Exception as exc:
                helper_error = exc
            helper_row = edge.METER_PREDICTIONS[-1]
            check("small-cap helper and explicit control fail on same operation",
                  control_error is not None and helper_error is not None
                  and type(control_error) is type(helper_error)
                  and str(control_error) == str(helper_error)
                  and helper_row["status"] == "raised"
                  and helper_row["output_sha256"] is None
                  and helper_row["flops_used"] == control_row["flops_used"]
                  and helper_row["n_ops"] == control_row["n_ops"])
        finally:
            if old_env is None:
                os.environ.pop("FLOPSCOPE_DEFAULT_BUDGET", None)
            else:
                os.environ["FLOPSCOPE_DEFAULT_BUDGET"] = old_env
    finally:
        edge.WEIGHTS_BACKEND = original_backend
        _reset_global_default()
        original_reset()

    edge_source = inspect.getsource(edge)
    check("every edge-suite candidate prediction routes through one helper",
          edge_source.count(".predict(") == 2
          and "return est.predict(mlp, budget)" in edge_source)

    context_events = []

    class ProbeFlops:
        @staticmethod
        def budget_reset():
            context_events.append("reset")

    class ProbeContext:
        def __enter__(self):
            context_events.append("enter")
            return self

        def __exit__(self, *_args):
            context_events.append("exit")

    with cert._reset_then_context(ProbeFlops, ProbeContext()):
        context_events.append("predict")
    check("section-5 reset is outside and immediately before context entry",
          context_events == ["reset", "enter", "predict", "exit"])
    cert_source = inspect.getsource(cert.main)
    seed_drop = cert_source.find("del ctx, pred, arr, weights, mlp")
    seed_collect = cert_source.find("collected = int(gc.collect())", seed_drop)
    seed_reset = cert_source.find("flops.budget_reset()", seed_collect)
    seed_trim = cert_source.find(
        "row[\"allocator_release\"] = _release_allocator(collected)",
        seed_reset)
    arm_drop = cert_source.find("del c, p_, wts, m, e2")
    arm_collect = cert_source.find("collected = int(gc.collect())", arm_drop)
    arm_reset = cert_source.find("flops.budget_reset()", arm_collect)
    arm_trim = cert_source.find(
        "arms[mode][\"allocator_release\"] = _release_allocator(collected)",
        arm_reset)
    check("section 5 and every 5b arm use the reset/context boundary",
          "with _reset_then_context(flops, ctx):" in cert_source
          and "with _reset_then_context(flops, c):" in cert_source
          and -1 < seed_drop < seed_collect < seed_reset < seed_trim
          and -1 < arm_drop < arm_collect < arm_reset < arm_trim)

    original_reset()
    closed = flopscope.BudgetContext(flop_budget=1_000_000, quiet=True)
    with closed:
        value = fnp.array([1.0, 2.0])
        _ = value + value
    closed_flops = closed.flops_used
    closed_summary = closed.summary_dict(by_namespace=True)
    original_reset()
    next_context = flopscope.BudgetContext(flop_budget=1_000_000, quiet=True)
    with next_context:
        _ = fnp.array([1.0]) + fnp.array([2.0])
    check("reset preserves the prior closed-context row ledger",
          closed_flops > 0 and next_context.flops_used > 0
          and closed.flops_used == closed_flops
          and closed.summary_dict(by_namespace=True) == closed_summary)
    check("certifier native allocator release is available and succeeds",
          cert._allocator_release_valid(cert._release_allocator()))

    payload, kwargs = fixture()
    check("complete receipt accepts",
          cert.validate_edge_group_receipt(payload, **kwargs) == [])

    mutations = (
        ("non-object", lambda _p: [], "not a JSON object"),
        ("wrong artifact", lambda p: p.__setitem__("artifact", "other"),
         "artifact identity"),
        ("extra field", lambda p: p.__setitem__("unregistered", True),
         "receipt schema"),
        ("boolean version", lambda p: p.__setitem__("artifact_version", True),
         "artifact version"),
        ("wrong group", lambda p: p.__setitem__("group", "other"),
         "group identity"),
        ("wrong candidate", lambda p: p.__setitem__("candidate", "other.py"),
         "candidate path"),
        ("changed candidate", lambda p: p.__setitem__(
            "candidate_sha256_after", "c" * 64), "after differs"),
        ("wrong instrument", lambda p: p.__setitem__(
            "edge_cases_sha256", "c" * 64), "edge_cases.py hash"),
        ("wrong meter reset policy", lambda p: p.__setitem__(
            "meter_reset_policy", "omitted"), "meter reset policy"),
        ("missing meter evidence", lambda p: p.pop("metered_predictions"),
         "metered predictions are not"),
        ("failed allocator release", lambda p: p["metered_predictions"][0][
            "allocator_release"].__setitem__("succeeded", False),
         "allocator release failed"),
        ("malformed output hash", lambda p: p["metered_predictions"][0].__setitem__(
            "output_sha256", "0" * 63), "schema differs"),
        ("boolean FLOP count", lambda p: p["metered_predictions"][0].__setitem__(
            "flops_used", True), "schema differs"),
        ("boolean operation count", lambda p: p["metered_predictions"][0].__setitem__(
            "n_ops", False), "schema differs"),
        ("truncated registry", lambda p: p["group_registry"].pop(),
         "group registry"),
        ("boolean registry count", lambda p: p["group_registry"][0].__setitem__(
            "expected_result_count", True), "group registry"),
        ("missing world field", lambda p: p["world"].pop("budget"),
         "world schema"),
        ("boolean width", lambda p: p["world"].__setitem__("width", True),
         "world field width"),
        ("wrong dtype", lambda p: p["world"].__setitem__(
            "weights_mode", "f32"), "world field weights_mode"),
        ("forged shim path", lambda p: p["world"].__setitem__(
            "flopscope_file", os.path.join(_HERE, "flopscope", "__init__.py")),
         "world field flopscope_file"),
        ("forged package version", lambda p: p["world"].__setitem__(
            "flopscope_version", "0.0-forged"),
         "world field flopscope_version"),
        ("boolean count", lambda p: p.__setitem__(
            "expected_result_count", True), "expected result count"),
        ("short vector", lambda p: p.__setitem__("results", []),
         "exact boolean result vector"),
        ("integer result", lambda p: p.__setitem__("results", [1]),
         "exact boolean result vector"),
        ("wrong passed", lambda p: p.__setitem__("passed", 0),
         "passed count differs"),
        ("boolean passed", lambda p: p.__setitem__("passed", True),
         "passed count is not"),
        ("integer all_passed", lambda p: p.__setitem__("all_passed", 1),
         "all_passed is not"),
    )
    for name, mutate, needle in mutations:
        bad = copy.deepcopy(payload)
        result = mutate(bad)
        if name == "non-object":
            bad = result
        errors = cert.validate_edge_group_receipt(bad, **kwargs)
        check(name + " refuses", any(needle in error for error in errors))

    case_payload, case_kwargs = case_fixture()
    check("complete atomic-case receipt accepts",
          cert.validate_edge_case_receipt(case_payload, **case_kwargs) == [])
    case_mutations = (
        ("boolean case index", lambda p: p.__setitem__("case_index", False),
         "case index"),
        ("truncated case registry", lambda p: p["case_registry"].pop(),
         "case registry"),
        ("boolean registry index", lambda p: p["case_registry"][0].__setitem__(
            "case_index", False), "case registry"),
        ("non-object details", lambda p: p.__setitem__("details", []),
         "details are not"),
        ("wrong result length", lambda p: p.__setitem__("results", []),
         "one exact boolean"),
    )
    for name, mutate, needle in case_mutations:
        bad = copy.deepcopy(case_payload)
        mutate(bad)
        errors = cert.validate_edge_case_receipt(bad, **case_kwargs)
        check(name + " refuses", any(needle in error for error in errors))

    permutation_payload, permutation_kwargs = case_fixture(
        case="permutation:3", group="permutation", case_index=3)
    permutation_payload["details"] = cert._expected_permutation_details(3)
    check("effective permutation bar and digest receipt accepts",
          cert.validate_edge_case_receipt(
              permutation_payload, **permutation_kwargs) == [])
    bad_bar = copy.deepcopy(permutation_payload)
    bad_bar["details"]["bar"] *= 2.0
    check("unbound permutation bar refuses",
          "permutation evidence differs" in cert.validate_edge_case_receipt(
              bad_bar, **permutation_kwargs))
    bad_digest = copy.deepcopy(permutation_payload)
    bad_digest["details"]["permutation_sha256"] = "0" * 64
    check("wrong permutation stream digest refuses",
          "permutation evidence differs" in cert.validate_edge_case_receipt(
              bad_digest, **permutation_kwargs))

    permutation_cases = cert.EDGE_PERMUTATION_CASES
    monolith = list(edge._permutation_case_stream(
        edge.np.random.default_rng(30_806_006_100), permutation_cases))
    streams_match = len(monolith) == len(permutation_cases)
    for index in range(len(permutation_cases)):
        atomic = list(edge._permutation_case_stream(
            edge.np.random.default_rng(30_806_006_100), permutation_cases,
            case_index=index))
        streams_match &= (len(atomic) == 1 and atomic[0][0] == index
                          and edge.np.array_equal(atomic[0][2],
                                                  monolith[index][2]))
    check("atomic permutations preserve every monolith RNG prefix",
          streams_match)

    increasing = list(range(len(cert.EDGE_BUDGET_LADDER)))
    budget_ok, budget_errors = cert.reduce_budget_monotone_cases(
        budget_rows(increasing))
    check("budget reducer accepts an exact valid monotone ladder",
          budget_ok and budget_errors == [])
    invalid_rows = budget_rows(
        [0, 1, 2, 3, 10, 9, 10, 11, 12, 13], invalid_index=4)
    invalid_ok, invalid_errors = cert.reduce_budget_monotone_cases(invalid_rows)
    check("non-throwing invalid rung still advances comparison state",
          not invalid_ok and any("budget_monotone:5: sample ladder decreased"
                                 in error for error in invalid_errors))
    thrown_rows = budget_rows(
        [0, 1, 2, 3, 10, 9, 10, 11, 12, 13], error_index=4)
    thrown_ok, thrown_errors = cert.reduce_budget_monotone_cases(thrown_rows)
    check("throwing rung does not advance comparison state",
          not thrown_ok and not any(
              "budget_monotone:5: sample ladder decreased" in error
              for error in thrown_errors))

    # The default path and isolated path share the one frozen dispatcher.  This
    # source-level identity is the cheap parity proof; numerical parity is then
    # exercised by the real positive control under workerq.
    expected_registry_sha = hashlib.sha256(repr(edge.GROUP_REGISTRY).encode()).hexdigest()
    check("default and isolated paths share one frozen dispatcher",
          callable(edge._execute_group)
          and expected_registry_sha
          == hashlib.sha256(repr(cert.EDGE_GROUP_REGISTRY).encode()).hexdigest())

    with tempfile.TemporaryDirectory(prefix="edge-group-cli-test-") as td:
        candidate = os.path.join(td, "candidate.py")
        receipt = os.path.join(td, "receipt.json")
        with open(candidate, "w", encoding="utf-8") as fh:
            fh.write("class Estimator:\n"
                     "    def setup(self, context):\n"
                     "        pass\n")
        run = subprocess.run(
            [sys.executable, edge.__file__, candidate,
             "--phase2", "--real-flopscope", "--weights-mode", "f64_readonly",
             "--case", "seed_hygiene:0", "--receipt", receipt],
            cwd=_ROOT, capture_output=True, text=True, timeout=60)
        produced = json.load(open(receipt, encoding="utf-8")) if os.path.exists(
            receipt) else {}
        check("lightweight CLI case writes an exact exclusive receipt",
              run.returncode == 0
              and produced.get("case") == "seed_hygiene:0"
              and produced.get("results") == [True]
              and produced.get("candidate_sha256_before")
              == produced.get("candidate_sha256_after")
              and produced.get("edge_cases_sha256")
              == cert._sha(edge.__file__)
              and cert.validate_edge_case_receipt(
                  produced, case="seed_hygiene:0", group="seed_hygiene",
                  case_index=0,
                  candidate=os.path.relpath(candidate, _ROOT).replace("\\", "/"),
                  candidate_sha256=cert._sha(candidate),
                  edge_cases_sha256=cert._sha(edge.__file__),
                  weights_mode="f64_readonly") == [])
        again = subprocess.run(
            [sys.executable, edge.__file__, candidate,
             "--phase2", "--real-flopscope", "--weights-mode", "f64_readonly",
             "--case", "seed_hygiene:0", "--receipt", receipt],
            cwd=_ROOT, capture_output=True, text=True, timeout=60)
        check("CLI receipt refuses overwrite", again.returncode != 0
              and "never overwritten" in again.stderr)

        # A stateless zero-output fixture makes every array construction and
        # dispatch path real while keeping candidate work tiny.  The layer-1
        # oracle deliberately fails; parity compares the complete boolean
        # vector, not a hand-picked green path.
        with open(candidate, "w", encoding="utf-8") as fh:
            fh.write("import numpy as np\n"
                     "class Estimator:\n"
                     "    def setup(self, context):\n"
                     "        pass\n"
                     "    def predict(self, mlp, budget):\n"
                     "        return np.zeros((len(mlp.weights), mlp.width))\n")
        default = subprocess.run(
            [sys.executable, edge.__file__, candidate], cwd=_ROOT,
            capture_output=True, text=True, timeout=60)
        by_group = {group: [] for group, _ in edge.GROUP_REGISTRY}
        isolated_ok = True
        for case, group, case_index in edge.CASE_REGISTRY:
            path = os.path.join(td, "parity-" + case.replace(":", "_") + ".json")
            run = subprocess.run(
                [sys.executable, edge.__file__, candidate,
                 "--case", case, "--receipt", path],
                cwd=_ROOT, capture_output=True, text=True, timeout=60)
            if not os.path.exists(path):
                isolated_ok = False
                continue
            row = json.load(open(path, encoding="utf-8"))
            by_group[group].append((case_index, row.get("results", [False])[0]))
            isolated_ok &= run.returncode == (0 if row.get("all_passed") else 1)
        isolated_results = []
        for group, logical_count in edge.GROUP_REGISTRY:
            values = [value for _, value in sorted(by_group[group])]
            isolated_results.extend(values if logical_count > 1 else [all(values)])
        expected_tail = "%d/%d checks passed" % (
            sum(isolated_results), len(isolated_results))
        check("cheap default execution equals aggregated atomic cases",
              isolated_ok and len(isolated_results) == 30
              and expected_tail in default.stdout
              and default.returncode == (0 if all(isolated_results) else 1))

    if FAILED:
        print("FAILED %d: %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("%d/%d checks passed" % (TOTAL, TOTAL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
