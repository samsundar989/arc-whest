"""
Contract and numerical-edge-case suite for the SHIPPED estimator file.

`robustness.py` measures the score distribution using a GPU mirror of the
pipeline. This runs the actual `estimator.py` through the flopscope/whestbench
shims, because the things that turn into a zero-scored MLP on the grader —
an exception, a NaN, a shape violation, a non-deterministic rerun — live in the
real file, not in the mirror.

Cases, each chosen to hit a specific line rather than to be generically nasty:

  degenerate pruning   every neuron in a layer below tau  -> the argmax fallback
  tiny sigma           near-zero pre-activation variance  -> alpha = mu/sigma
  extreme alpha        saturated layers                   -> norm.cdf/pdf tails
  wide dynamic range   one layer scaled 100x / 0.01x      -> f32 overflow paths
  seed extremes        seed 0, 1, 2**63-1                 -> rng contract
  shape generality     width 64/128, depth 2/8            -> off-calibration path
  tiny budget          budget forcing log2n below the floor -> analytic fallback

INVARIANTS (added 2026-08-06). The cases above ask "does it survive?"; these ask
"is it computing the right thing?" — which no amount of finite output proves:

  layer-1 closed form  E[ReLU(w.x)] = ||w||/sqrt(2pi) exactly for x ~ N(0,I),
                       so row 0 has a known answer and must hit it to fp
                       tolerance. Any drift here is arithmetic, not sampling.
  permutation          relabelling the neurons of a hidden layer leaves the
  equivariance         network's function unchanged, so predictions must permute
                       with it and every other layer must be untouched. This is
                       the one test that catches a transposed index, an axis
                       swapped in a reduction, or a basis applied on the wrong
                       side — mistakes that produce perfectly finite, perfectly
                       plausible, wrong numbers. AMENDED 2026-09-03: judged in
                       MSE units against the candidate's own raw MSE at a 1%
                       bar, on EVERY row. The old absolute clause was an f64-era
                       float32-precision bar; see `inv_permutation`.
  budget monotonicity  more budget must never buy fewer samples, and every rung
                       of the ladder down to zero must degrade rather than throw.
  determinism x3       three runs bit-identical, plus a re-instantiated
                       Estimator, so setup state cannot leak between MLPs.
  seed hygiene         live seed blocks are disjoint (seed_blocks.py).

Run: uv run python local_sim/edge_cases.py
"""

import ctypes
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

#: THE WORLD THIS SUITE RUNS IN. Until 2026-09-05 all three were literals
#: scattered through `main()`: Phase 1's 256x32 shape, Phase 1's budget, and raw
#: NumPy float32 weights. The suite therefore graded every candidate -- including
#: Phase-2 ones -- in a world the competition stopped grading on 2026-08-21, and
#: in a dtype the grading server has never handed anyone. `--phase2` and
#: `--real-flopscope` move it into the live world; the defaults are unchanged, so
#: every historical invocation still means exactly what it meant.
BUDGET = 272_000_000_000

#: The PRIMARY shape. The deliberately odd shapes elsewhere in the suite (width
#: 64, depth 2, non-square sweeps) are shape-genericity tests and stay literal.
PD, PW = 32, 256

#: Weight representation. "numpy" is the historical world: raw np.float32.
#: "flopscope" is the grader's: metered arrays in `WEIGHTS_MODE`. Operations
#: invoked as METHODS ON A WEIGHT are billed on the grader and free on bare
#: NumPy, and the dtype decides which branch of a `x.dtype == ...` guard runs --
#: which is how a defect reached five submission slots on 2026-09-05.
WEIGHTS_BACKEND = "numpy"
WEIGHTS_MODE = "f64_readonly"
PHASE2_MODE = False
LIVE_DEPTH, LIVE_WIDTH, LIVE_BUDGET = 16, 1024, 2 ** 41
INSTRUMENT_IDENTITY = {"kind": "historical_numpy", "path": None,
                       "version": None}

# Frozen execution partition used by ``certify_candidate.py`` to bound the
# allocator lifetime of the live Phase-2 suite.  The ordinary no-argument suite
# still executes these groups in this order with one Estimator, exactly as it
# did before the partition existed.
GROUP_REGISTRY = (
    ("normal", 1),
    ("degenerate", 3),
    ("dynamic_range", 4),
    ("seed_extremes", 5),
    ("shape_generality", 5),
    ("budget_edges", 5),
    ("immutability", 1),
    ("layer1", 1),
    ("permutation", 1),
    ("budget_monotone", 1),
    ("determinism", 1),
    ("seed_hygiene", 1),
    ("gt_noise", 1),
)
CASE_COUNTS = (
    ("normal", 1), ("degenerate", 3), ("dynamic_range", 4),
    ("seed_extremes", 5), ("shape_generality", 5), ("budget_edges", 5),
    ("immutability", 1), ("layer1", 4), ("permutation", 4),
    ("budget_monotone", 10), ("determinism", 1), ("seed_hygiene", 1),
    ("gt_noise", 1),
)
CASE_REGISTRY = tuple(
    ("%s:%d" % (group, index), group, index)
    for group, count in CASE_COUNTS for index in range(count)
)
GROUP_RECEIPT_VERSION = 5
CASE_RECEIPT_VERSION = 5
METER_RESET_POLICY = (
    "real-flopscope: fresh explicit context at captured implicit-global cap, "
    "capture ledger/output after every candidate predict, then drop context, "
    "collect, reset accumulator, and trim; historical "
    "numpy mode: direct predict; Phase2 off-shape correctness budgets scale "
    "with dense work; v4"
)
CORRECTNESS_BUDGET_POLICY = (
    "Phase2 implicit off-shape: min(B_live, "
    "floor(B_live*d*w^2/(D_live*W_live^2))); live-shape, explicit-budget, "
    "and historical calls unchanged; v1"
)
METER_PREDICTIONS = []


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write_json_exclusive(path, payload):
    out = os.path.abspath(path)
    parent = os.path.dirname(out)
    if not os.path.isdir(parent):
        raise ValueError("receipt parent directory does not exist: " + parent)
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _weights(arrays):
    """Build the estimator's weight list under the configured world."""
    if WEIGHTS_BACKEND == "numpy":
        return [np.asarray(w, dtype=np.float32) for w in arrays]
    import flopscope.numpy as fnp
    from lifecycle_eval import build_weights
    return build_weights([np.asarray(w, dtype=np.float32) for w in arrays],
                         WEIGHTS_MODE, fnp)


def _implicit_default_budget():
    """Capture flopscope 0.12's effective budget outside an explicit context."""
    raw = os.environ.get("FLOPSCOPE_DEFAULT_BUDGET")
    return int(float(raw)) if raw is not None else int(1e15)


def _correctness_budget(depth, width):
    """Phase2-only dense-work budget for implicit off-shape correctness calls."""
    if not PHASE2_MODE or (depth, width) == (LIVE_DEPTH, LIVE_WIDTH):
        return BUDGET
    scaled = (LIVE_BUDGET * depth * width * width
              // (LIVE_DEPTH * LIVE_WIDTH * LIVE_WIDTH))
    return min(LIVE_BUDGET, scaled)


def _release_allocator(gc_collected=None):
    """Best-effort return native pages after caller drops and collects refs."""
    collected = int(gc.collect()) if gc_collected is None else gc_collected
    result = {
        "platform": sys.platform, "gc_collected": collected,
        "mechanism": "gc_only", "available": False,
        "returncode": None, "succeeded": True,
    }
    try:
        if sys.platform == "win32":
            fn = ctypes.CDLL("ucrtbase.dll")._heapmin
            fn.argtypes, fn.restype = [], ctypes.c_int
            rc = int(fn())
            result.update({"mechanism": "ucrtbase._heapmin",
                           "available": True, "returncode": rc,
                           "succeeded": rc == 0})
        elif sys.platform.startswith("linux"):
            fn = ctypes.CDLL(None).malloc_trim
            fn.argtypes, fn.restype = [ctypes.c_size_t], ctypes.c_int
            rc = int(fn(0))
            result.update({"mechanism": "libc.malloc_trim(0)",
                           "available": True, "returncode": rc,
                           "succeeded": True})
    except Exception as exc:
        result.update({"succeeded": False,
                       "error": "%s: %s" % (type(exc).__name__, exc)})
    return result


def _predict(est, mlp, budget):
    """Run one prediction without retaining its instrument history."""
    if WEIGHTS_BACKEND != "flopscope":
        return est.predict(mlp, budget)
    import flopscope
    # Every real-world MLP has already been constructed from fnp weights, so
    # the implicit default exists. Capture its actual cap rather than treating
    # the candidate's 0/1 fallback argument as an enforcement budget.
    effective_cap = int(flopscope.current_budget().flop_budget)
    context = flopscope.BudgetContext(
        flop_budget=effective_cap, quiet=True)
    status = "raised"
    output_sha256 = None
    try:
        with context:
            result = est.predict(mlp, budget)
        status = "returned"
        try:
            output = np.ascontiguousarray(np.asarray(result))
            output_sha256 = hashlib.sha256(output.tobytes()).hexdigest()
        except Exception:
            # Preserve the returned object so the existing shape/dtype gates
            # remain authoritative; a successful receipt still fails closed.
            output_sha256 = None
        return result
    finally:
        row = {
            "flop_budget": int(context.flop_budget),
            "flops_used": int(context.flops_used),
            "n_ops": len(context.op_log), "status": status,
            "output_sha256": output_sha256,
            "candidate_budget": int(budget),
            "mlp_width": int(mlp.width), "mlp_depth": int(mlp.depth),
        }
        context = None
        collected = int(gc.collect())
        flopscope.budget_reset()
        row["allocator_release"] = _release_allocator(collected)
        METER_PREDICTIONS.append(row)


def load(path="estimator.py"):
    full = os.path.join(_ROOT, path)
    spec = importlib.util.spec_from_file_location("shipped_est", full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shipped_est"] = mod
    spec.loader.exec_module(mod)
    return mod


class MLP:
    def __init__(self, weights, seed=0):
        self.weights = _weights(weights)
        self.width = int(np.asarray(self.weights[0]).shape[1])
        self.depth = len(self.weights)
        self.seed = seed


def he(depth, width, seed, scale=None):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((depth, width, width)).astype(np.float32) * np.float32(np.sqrt(2.0 / width))
    if scale is not None:
        w = w.copy()
        for li, s in scale.items():
            w[li] = w[li] * np.float32(s)
    return w


def check(name, est, mlp, budget=None, expect_depth=None, rerun=True):
    # Resolve the default at call time.  ``--phase2`` updates the module's
    # active BUDGET after functions are defined; a default of ``budget=BUDGET``
    # silently retained the historical Phase-1 value in otherwise-live calls.
    if budget is None:
        budget = BUDGET
    try:
        out = _predict(est, mlp, budget)
    except Exception as e:  # a raise on the grader is a zero-scored MLP
        print(f"  {name:<34} RAISED {type(e).__name__}: {e}")
        return False
    a = np.asarray(out, dtype=np.float64)
    depth = expect_depth if expect_depth is not None else len(mlp.weights)
    ok = True
    if a.shape != (depth, mlp.width):
        print(f"  {name:<34} BAD SHAPE {a.shape}, expected {(depth, mlp.width)}")
        ok = False
    if not np.isfinite(a).all():
        print(f"  {name:<34} NON-FINITE ({np.sum(~np.isfinite(a))} entries)")
        ok = False
    # A post-ReLU mean cannot be negative as a QUANTITY, but a float64 sum that
    # should land on a near-zero neuron can round below it, and the wider the
    # layer the more terms there are to round. At Phase 1's width 256 the
    # shipped file never crossed zero, so an absolute test looked exact; at the
    # live width 1024 it does, by ~1e-16 relative to row means of ~0.46, and the
    # absolute test then fails 17 of 30 invariants on a correct candidate.
    # This is the SAME bar `certify_candidate.py` section 5 already reasoned to
    # from the same measurement -- scaled to the row, four orders above the
    # observed rounding and many orders below any structural sign error. The
    # Phase-1 default is unaffected: it never produced a negative value at all.
    neg_tol = -1e-12 * max(1.0, float(np.max(np.abs(a))))
    if (a < neg_tol).any():
        print(f"  {name:<34} NEGATIVE MEANS (min {a.min():.3e}, bar "
              f"{neg_tol:.1e}) — post-ReLU means cannot be < 0")
        ok = False
    ident = True
    if rerun and ok:
        b = np.asarray(_predict(est, mlp, budget), dtype=np.float64)
        ident = bool(np.array_equal(a, b))
        if not ident:
            print(f"  {name:<34} NON-DETERMINISTIC (max diff {np.abs(a - b).max():.3e})")
            ok = False
    if ok:
        print(f"  {name:<34} ok   shape {a.shape}  range [{a.min():.3e}, {a.max():.3e}]  "
              f"final mean {a[-1].mean():.4f}  deterministic={ident}")
    return ok


def _one(items, case_index):
    return items if case_index is None else (items[case_index],)


def _execute_group(group, est, est_mod, case_index=None, evidence=None):
    """Execute one frozen group and return its existing result booleans."""
    results = []
    if group == "normal":
        print("1. NORMAL CASE")
        if case_index in (None, 0):
            results.append(check(f"he {PW}x{PD}", est,
                                 MLP(he(PD, PW, 1), seed=7)))
    elif group == "degenerate":
        print("\n2. DEGENERATE PRUNING (whole layers pushed below tau)")
        cases = ((0.02, "layer 5 scaled 0.02", 2, 11),
                 (0.001, "layer 5 scaled 0.001", 2, 11))
        selected_cases = cases if case_index is None else (
            (cases[case_index],) if case_index < len(cases) else ())
        for s, label, weight_seed, mlp_seed in selected_cases:
            results.append(check(label, est,
                                 MLP(he(PD, PW, weight_seed, {5: s}),
                                     seed=mlp_seed)))
        if case_index in (None, 2):
            w = he(PD, PW, 3)
            w[7] = (w[7] * np.float32(0.0)).copy()
            results.append(check("layer 7 identically zero", est,
                                 MLP(w, seed=13)))
    elif group == "dynamic_range":
        print("\n3. EXTREME DYNAMIC RANGE")
        cases = ((100.0, "layer 12 scaled 100x"),
                 (1e-3, "layer 12 scaled 1e-3"),
                 (1e4, "layer 12 scaled 1e4"))
        selected_cases = cases if case_index is None else (
            (cases[case_index],) if case_index < len(cases) else ())
        for s, label in selected_cases:
            results.append(check(label, est,
                                 MLP(he(PD, PW, 4, {12: s}), seed=17)))
        if case_index in (None, 3):
            w = he(PD, PW, 5).copy()
            w[3] = w[3] + np.float32(2.0) * np.float32(np.sqrt(2.0 / PW))
            results.append(check("layer 3 mean-shifted (all on)", est,
                                 MLP(w, seed=19)))
    elif group == "seed_extremes":
        print("\n4. SEED EXTREMES (the rng contract)")
        base = he(PD, PW, 6)
        for sd in _one((0, 1, 2**31 - 1, 2**63 - 1,
                        12345678901234567890 % (2**64)), case_index):
            results.append(check(f"seed {sd}", est, MLP(base, seed=sd)))
    elif group == "shape_generality":
        print("\n5. SHAPE GENERALITY (calibration must disable itself off-width)")
        for d, wd in _one(((2, 256), (8, 128), (32, 64), (4, 256),
                           (64, 256)), case_index):
            results.append(check(f"depth {d} x width {wd}", est,
                                 MLP(he(d, wd, 8), seed=23),
                                 budget=_correctness_budget(d, wd)))
    elif group == "budget_edges":
        print("\n6. BUDGET EDGES (the analytic fallback path)")
        m = MLP(he(PD, PW, 9), seed=29)
        cases = ((0, "budget 0"), (1, "budget 1"),
                 (10**7, "budget 1e7 (below min N)"),
                 (10**9, "budget 1e9"), (10 * BUDGET, "budget 10x"))
        for b, label in _one(cases, case_index):
            results.append(check(label, est, m, budget=b))
    elif group == "immutability":
        print("\n7. GRADER-ARRAY IMMUTABILITY (weights must be untouched)")
        if case_index in (None, 0):
            m = MLP(he(PD, PW, 10), seed=31)
            before = [w.copy() for w in m.weights]
            _predict(est, m, BUDGET)
            same = all(np.array_equal(a, b) for a, b in zip(before, m.weights))
            print(f"  {'input weights unmodified':<34} "
                  f"{'ok' if same else 'MUTATED INPUT'}")
            results.append(same)
    elif group == "layer1":
        results += inv_layer1(est, case_index=case_index)
    elif group == "permutation":
        results += inv_permutation(est, case_index=case_index,
                                   evidence=evidence)
    elif group == "budget_monotone":
        results += inv_budget_monotone(est, est_mod)
    elif group == "determinism":
        results += inv_determinism(est, est_mod, evidence=evidence)
    elif group == "seed_hygiene":
        results += inv_seed_hygiene()
    elif group == "gt_noise":
        results += inv_gt_noise_reporting()
    else:  # argparse prevents this; retain a fail-closed programmatic boundary.
        raise ValueError("unknown edge-case group: " + str(group))
    return results


def _budget_monotone_case(est, est_mod, case_index):
    """Execute one rung; the certifier reconstructs the ordered monotonicity."""
    ladder = (0, 1, 10**6, 10**7, 10**8, 10**9, 10**10,
              27_200_000_000, 272_000_000_000, 2_720_000_000_000)
    b = ladder[case_index]
    m = canon(PD, PW, 30_806_006_021)
    has_ladder = callable(getattr(est, "_log2_samples", None)) and \
        hasattr(est_mod, "_MIN_LOG2_N")
    log2n = None
    error = ""
    finite = False
    try:
        if has_ladder:
            log2n = est._log2_samples(PW, PD, b)
            if isinstance(log2n, np.generic):
                log2n = log2n.item()
        out = np.asarray(_predict(est, m, b), dtype=np.float64)
        tol = -1e-12 * max(1.0, float(np.max(np.abs(out))))
        finite = (bool(np.isfinite(out).all()) and out.shape == (PD, PW)
                  and bool((out >= tol).all()))
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    print("\n10. BUDGET MONOTONICITY  (isolated rung %d)" % case_index)
    print("  budget %16s  valid=%s  log2n=%s%s" % (
        format(b, ","), finite, log2n,
        "  RAISED " + error if error else ""))
    return [bool(finite and not error)], {
        "budget": b, "has_ladder": bool(has_ladder), "log2n": log2n,
        "finite": bool(finite), "error": error,
    }


def _execute_case(case_id, est, est_mod):
    matches = [row for row in CASE_REGISTRY if row[0] == case_id]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate edge-case id: " + str(case_id))
    _, group, case_index = matches[0]
    if group == "budget_monotone":
        results, details = _budget_monotone_case(est, est_mod, case_index)
    elif group == "permutation":
        evidence = []
        results = _execute_group(group, est, est_mod, case_index=case_index,
                                 evidence=evidence)
        details = evidence[0] if len(evidence) == 1 else {
            "evidence_error": "expected exactly one permutation record",
        }
    elif group == "determinism":
        evidence = []
        results = _execute_group(group, est, est_mod, case_index=case_index,
                                 evidence=evidence)
        details = {"cases": evidence}
    else:
        results = _execute_group(group, est, est_mod, case_index=case_index)
        details = {}
    return group, case_index, results, details


def main():
    global BUDGET, PD, PW, WEIGHTS_BACKEND, WEIGHTS_MODE, INSTRUMENT_IDENTITY
    global PHASE2_MODE, LIVE_DEPTH, LIVE_WIDTH, LIVE_BUDGET
    import argparse
    ap = argparse.ArgumentParser(
        description="The invariant suite. Defaults reproduce the historical "
                    "Phase-1 world exactly; --phase2 and --real-flopscope move "
                    "it into the world the competition actually grades.")
    ap.add_argument("candidate", nargs="?", default="estimator.py")
    ap.add_argument("--phase2", action="store_true",
                    help="run at the LIVE shape (1024x16) and budget (2**41) "
                         "instead of Phase 1's 256x32 / 2.72e11")
    ap.add_argument("--real-flopscope", action="store_true",
                    help="build weights as flopscope-metered arrays in the "
                         "grader's dtype, not raw NumPy float32")
    ap.add_argument("--weights-mode", default="f64_readonly",
                    choices=("f32", "f64", "f64_readonly", "f64_listconstructed"),
                    help="which grader-twin arm --real-flopscope uses")
    ap.add_argument("--group", choices=tuple(tag for tag, _ in GROUP_REGISTRY),
                    help="run one frozen allocator-bounded group")
    ap.add_argument("--case", choices=tuple(case for case, _, _ in CASE_REGISTRY),
                    help="run one atomic allocator-bounded case")
    ap.add_argument("--receipt", default="", metavar="PATH",
                    help="exclusive-create a machine-readable group receipt; "
                         "requires --group or --case")
    a = ap.parse_args()
    if a.group and a.case:
        ap.error("--group and --case are mutually exclusive")
    if a.receipt and not (a.group or a.case):
        ap.error("--receipt requires --group or --case")
    if a.receipt and os.path.exists(os.path.abspath(a.receipt)):
        ap.error("receipt already exists; evidence is never overwritten")

    if a.phase2:
        import official_config as _oc
        PD, PW = _oc.LIVE.depth, _oc.LIVE.width
        BUDGET = _oc.LIVE.flop_budget
        PHASE2_MODE = True
        LIVE_DEPTH, LIVE_WIDTH, LIVE_BUDGET = PD, PW, BUDGET
    if a.real_flopscope:
        # This script lives beside a historical package named ``flopscope``.
        # Move that directory behind site-packages before the candidate import;
        # otherwise --real-flopscope silently loads the unmetered local shim.
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or ".") != _HERE] + [_HERE]
        import flopscope as real_flopscope
        instrument_path = os.path.abspath(real_flopscope.__file__)
        if instrument_path.startswith(os.path.join(_HERE, "flopscope")):
            raise RuntimeError("--real-flopscope resolved the local shim")
        WEIGHTS_BACKEND = "flopscope"
        WEIGHTS_MODE = a.weights_mode
        INSTRUMENT_IDENTITY = {
            "kind": "installed_flopscope", "path": instrument_path,
            "version": importlib.metadata.version("flopscope"),
        }

    print(f"world: {PW}x{PD}  B = {BUDGET:,}  weights = {WEIGHTS_BACKEND}"
          + (f"/{WEIGHTS_MODE}" if WEIGHTS_BACKEND != "numpy" else " float32")
          + ("   [LIVE PHASE-2 WORLD]" if a.phase2 and a.real_flopscope
             else "   [HISTORICAL DEFAULT -- not the graded world]"))

    candidate_path = os.path.abspath(os.path.join(_ROOT, a.candidate))
    candidate_sha_before = _sha(candidate_path)
    est_mod = load(a.candidate)
    est = est_mod.Estimator()

    class Ctx:
        width, depth, flop_budget, seed = PW, PD, BUDGET, 0

    est.setup(Ctx())
    # A BANNER, not a check. It used to read `est_mod._PRUNE_TAU` directly, and
    # those three constants are v2.x sampling-architecture internals: any
    # candidate that does not happen to define them died here with an
    # `AttributeError` before a single invariant ran. WP0C's own positive
    # control -- a pure wrapper around the shipped file -- was the thing that
    # exposed it, and a research candidate from `p2_candidate_architecture.py`
    # would have hit it too. Reading them defensively changes no invariant and
    # no verdict; it only stops the banner from gating the suite.
    _b = {k: getattr(est_mod, k, "n/a")
          for k in ("_PRUNE_TAU", "_TARGET_CB", "_CHUNK")}
    print(f"candidate: tau={_b['_PRUNE_TAU']}  target C/B={_b['_TARGET_CB']}  "
          f"chunk={_b['_CHUNK']}\n")

    results = []
    group_rows = []
    expected_by_group = dict(GROUP_REGISTRY)
    case_details = {}
    if a.case:
        group, case_index, results, case_details = _execute_case(
            a.case, est, est_mod)
    else:
        selected = [a.group] if a.group else [tag for tag, _ in GROUP_REGISTRY]
        for group in selected:
            group_results = _execute_group(group, est, est_mod)
            results += group_results
            group_rows.append({
                "group": group,
                "expected_result_count": expected_by_group[group],
                "results": group_results,
            })

    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} checks passed")
    if a.receipt:
        candidate_sha_after = _sha(candidate_path)
        common = {
            "candidate": os.path.relpath(candidate_path, _ROOT).replace("\\", "/"),
            "candidate_sha256_before": candidate_sha_before,
            "candidate_sha256_after": candidate_sha_after,
            "edge_cases_sha256": _sha(__file__),
            "meter_reset_policy": METER_RESET_POLICY,
            "correctness_budget_policy": CORRECTNESS_BUDGET_POLICY,
            "metered_predictions": METER_PREDICTIONS,
            "group_registry": [
                {"group": group, "expected_result_count": count}
                for group, count in GROUP_REGISTRY
            ],
            "world": {
                "phase2": bool(a.phase2),
                "real_flopscope": bool(a.real_flopscope),
                "weights_mode": WEIGHTS_MODE if a.real_flopscope else "numpy_float32",
                "width": int(PW), "depth": int(PD), "budget": int(BUDGET),
                "flopscope_file": INSTRUMENT_IDENTITY["path"],
                "flopscope_version": INSTRUMENT_IDENTITY["version"],
            },
            "results": results,
            "passed": n_ok,
            "all_passed": n_ok == len(results),
        }
        if a.case:
            common.update({
                "artifact": "edge_cases_case",
                "artifact_version": CASE_RECEIPT_VERSION,
                "case_registry": [
                    {"case": case, "group": logical, "case_index": index}
                    for case, logical, index in CASE_REGISTRY
                ],
                "case": a.case, "logical_group": group,
                "case_index": case_index, "details": case_details,
            })
        else:
            common.update({
                "artifact": "edge_cases_group",
                "artifact_version": GROUP_RECEIPT_VERSION,
                "group": a.group,
                "expected_result_count": expected_by_group[a.group],
            })
        _write_json_exclusive(a.receipt, common)
    return 0 if n_ok == len(results) else 1


# =====================================================================
# INVARIANTS — properties the true answer has, that a wrong answer will not
# =====================================================================

def canon(depth, width, seed):
    """A grader-identical MLP (mint_provenance) as an edge-case MLP object."""
    from mint_provenance import canonical_mint

    w, est_seed, _ = canonical_mint(width, depth, seed)
    m = MLP(w, seed=est_seed)
    return m


def inv_layer1(est, case_index=None):
    """Row 0 must equal ||W1[:,j]||_2 / sqrt(2*pi) to floating-point tolerance.

    For x ~ N(0, I) the layer-1 pre-activation z_j = <W1[:,j], x> is exactly
    N(0, ||W1[:,j]||^2), and E[ReLU(N(0, s^2))] = s/sqrt(2*pi). No sampling, no
    closure, no approximation — this row has a closed form and it is checkable.
    """
    print("\n8. LAYER-1 CLOSED FORM  (exact: ||W1[:,j]|| / sqrt(2*pi))")
    ok = True
    cases = ((PD, PW, 30_806_006_001), (8, 128, 30_806_006_002),
             (4, 64, 30_806_006_003), (2, 256, 30_806_006_004))
    for depth, width, seed in _one(cases, case_index):
        m = canon(depth, width, seed)
        candidate_budget = _correctness_budget(depth, width)
        pred = np.asarray(_predict(est, m, candidate_budget), dtype=np.float64)
        w0 = np.asarray(m.weights[0], dtype=np.float64)
        exact = np.sqrt((w0 * w0).sum(axis=0)) / np.sqrt(2.0 * np.pi)
        rel = float(np.abs(pred[0] - exact).max() / np.abs(exact).max())
        good = rel < 1e-12
        ok &= good
        print(f"  {f'{width}x{depth}':<34} max rel err {rel:.3e}   "
              f"{'ok' if good else 'FAILED (>1e-12)'}")
    return [ok]


#: The MSE this project makes decisions on. Frozen; `VALIDATION.md` quotes it,
#: and the GT-noise invariant (13) reports the vault floor as a share of it.
DECISION_MSE = 1.4e-6
#: Invariant 9's bar: a relabelling may not move any row by more than this
#: share of the candidate's own raw MSE.
DRIFT_SHARE = 0.01


def raw_mse_denominator():
    """The `raw MSE` invariant 9 judges drift against.

    Defaults to the frozen decision signal, so amending the invariant did NOT
    move the bar on the shipped file. `ARC_WHEST_RAW_MSE` lets a candidate be
    held to ITS OWN measured raw MSE -- the canary's is 8.5075e-07, a 1.65x
    STRICTER bar than the default -- and is clamped so an override can only ever
    tighten. An override that could loosen a gate is a hole, and this repository
    has been bitten by that shape of thing before.
    """
    raw = os.environ.get("ARC_WHEST_RAW_MSE")
    if not raw:
        return DECISION_MSE, "frozen decision signal"
    try:
        v = float(raw)
    except ValueError:
        return DECISION_MSE, "frozen decision signal (ARC_WHEST_RAW_MSE unparseable)"
    if not (v > 0.0):
        return DECISION_MSE, "frozen decision signal (ARC_WHEST_RAW_MSE non-positive)"
    if v >= DECISION_MSE:
        return DECISION_MSE, ("frozen decision signal (override %.4e would loosen; "
                              "clamped)" % v)
    return v, "candidate raw MSE %.4e" % v


def perm_gate(base, perm, P, k, depth, denom=None):
    """THE gate invariant 9 applies, factored out so nothing can disagree with it.

    `base` is the prediction on the network, `perm` the prediction on the same
    network with hidden layer `k` relabelled by `P`. Equivariance says
    `perm[k] == base[k][P]` and `perm[l] == base[l]` for every `l != k`; we
    require each deviation, in MSE units, to be at most `DRIFT_SHARE` of the
    candidate's own raw MSE. `local_sim/test_edge_cases_inv9.py` calls this
    directly with deliberately broken predictions.
    """
    if denom is None:
        denom, _ = raw_mse_denominator()
    bar = DRIFT_SHARE * denom
    others = [l for l in range(depth) if l != k]
    d = {
        "d_k": float(np.abs(perm[k] - base[k][P]).max()),
        "d_other": float(np.abs(perm[others] - base[others]).max()),
        "d_final": (float(np.abs(perm[-1] - base[-1]).max())
                    if k != depth - 1 else 0.0),
        "mse_k": float(((perm[k] - base[k][P]) ** 2).mean()),
        "mse_other": float(((perm[others] - base[others]) ** 2).mean()),
        "mse_final": (float(((perm[-1] - base[-1]) ** 2).mean())
                      if k != depth - 1 else 0.0),
    }
    # SKEPTIC (2026-09-03) found a REGRESSION here, and it is this repository's
    # own signature failure class. `max()` compares with `>`, `nan > x` is False,
    # so a NaN anywhere but the first position was DROPPED before the finiteness
    # check ever saw it: injecting nan into the scored final row gave
    # `worst = 0.0, ok = True`. The OLD code rejected that exact input, because
    # `nan < 1.4e-8` is False. Each component is therefore tested for finiteness
    # on its own, BEFORE any ordering. GOTCHAS.md, 2026-09-02: "A `>` comparison
    # is not a check. It silently admits the one value it cannot order."
    parts = (d["mse_k"], d["mse_other"], d["mse_final"])
    d["finite"] = bool(np.all(np.isfinite(np.asarray(parts, dtype=np.float64))))
    d["worst"] = float(np.max(np.asarray(parts, dtype=np.float64))) if d["finite"]         else float("nan")
    d["bar"] = bar
    d["denom"] = denom
    d["ok"] = d["finite"] and d["worst"] <= bar
    return d


def _permutation_case_stream(rng, cases, case_index=None):
    """Yield selected cases after consuming the monolith's exact RNG prefix."""
    for index, case in enumerate(cases):
        permutation = rng.permutation(case[1])
        if case_index is None or index == case_index:
            yield index, case, permutation
        if case_index is not None and index >= case_index:
            break


def inv_permutation(est, case_index=None, evidence=None):
    """Relabelling hidden layer k must permute row k and change nothing else.

    Swapping the output columns of W_k and the matching input rows of W_{k+1} by
    the same permutation P leaves the network's input-output map identical and
    simply renames the layer-k neurons. So the true answer satisfies
        pred'[k] = pred[k][P],   pred'[l] = pred[l]  for every l != k.

    The residual we tolerate is not slop: reordering columns reorders the
    summation inside every downstream matmul, so JJ^T shifts at the 1e-16 level
    and the alignment basis inherits that. We therefore report the deviation as
    a share of the MSE we make decisions on, rather than asserting an exact
    equality the hardware cannot deliver.

    AMENDED 2026-09-03 (strategist Ruling 1; LOG "IMPORTANT (instrument)").
    This used to gate additionally on `d_k / scale < 1e-6` -- a max-absolute
    deviation against a fixed relative bar about 10x float32 epsilon. That is a
    bar only a float64 backbone can hold, and it was written when the K2 flow
    had one. The live design deliberately runs the covariance backbone in f32,
    which is what puts C/B = 0.0767 on the 0.1 multiplier floor; curing it was
    measured at exactly 2.0x billing, i.e. 1.53x worse score. So the clause did
    not ask a candidate to be correct, it asked it to be expensive. That it is a
    DTYPE bar and not an index bug was established independently by
    SKEPTIC WP18-C, which ran the same flow on the same networks at f32 and f64
    and moved d_k/scale from 2.116e-06 to 2.365e-15 -- a ~1e9 ratio, which is
    the eps ratio.

    What replaces it is the criterion the surviving clause already used,
    extended to the two deviations the old gate merely PRINTED: every row's
    drift, in MSE units, must be <= 1% of the candidate's own raw MSE. That is a
    net widening on row k and a net TIGHTENING on the other rows and the final
    row, which had no gate at all -- and `d_other` on the canary is 1.91e-06,
    LARGER than the d_k the old gate rejected on, with nothing looking at it.
    It still catches the mistakes the invariant exists for by five or more
    orders of magnitude: a transposed index or a swapped reduction axis moves a
    row by O(1), not by O(eps). `local_sim/test_edge_cases_inv9.py` is the
    standing proof and it is a gate, not a comment.
    """
    denom, denom_src = raw_mse_denominator()
    bar = DRIFT_SHARE * denom
    print("\n9. PERMUTATION EQUIVARIANCE  (hidden-layer relabelling)")
    print("  bar: every row MSE <= %.0f%% of raw MSE = %.3e   [%s]"
          % (DRIFT_SHARE * 100, bar, denom_src))
    rng = np.random.default_rng(30_806_006_100)
    ok = True
    # The layer indices 7/20/30 were chosen for Phase 1's depth 32 and are kept
    # EXACTLY there, so the historical default is bit-for-bit what it was. At any
    # other depth they are scaled proportionally and clamped to `depth - 2` (the
    # permutation renames layer k's outputs and layer k+1's inputs, so k+1 must
    # exist): at Phase 2's depth 16 that is 4/10/14, still early/middle/last.
    # Before this, `--phase2` died with an IndexError at k = 20.
    def _k_for(depth_, k_):
        if depth_ == 32:
            return k_
        return max(1, min(depth_ - 2, int(round(k_ * depth_ / 32.0))))

    cases = ((PD, PW, 30_806_006_011, 7),
             (PD, PW, 30_806_006_012, 20),
             (8, 128, 30_806_006_013, 3),
             (PD, PW, 30_806_006_014, 30))
    for index, (depth, width, seed, k), P in _permutation_case_stream(
            rng, cases, case_index):
        k = _k_for(depth, k)
        candidate_budget = _correctness_budget(depth, width)
        m = canon(depth, width, seed)
        base = np.asarray(_predict(est, m, candidate_budget), dtype=np.float64)

        w = [np.array(x) for x in m.weights]
        w[k] = w[k][:, P]                      # rename layer-k outputs
        w[k + 1] = w[k + 1][P, :]              # and the matching inputs
        mp = MLP(w, seed=m.seed)
        perm = np.asarray(_predict(est, mp, candidate_budget), dtype=np.float64)

        g = perm_gate(base, perm, P, k, depth, denom=denom)
        if evidence is not None:
            evidence.append({
                "case_index": index, "depth": depth, "width": width,
                "seed": seed, "layer": k,
                "permutation_sha256": hashlib.sha256(
                    np.asarray(P, dtype="<i8").tobytes()).hexdigest(),
                "raw_mse_env": os.environ.get("ARC_WHEST_RAW_MSE"),
                "denom": denom, "bar": bar, "denom_source": denom_src,
            })
        ok &= g["ok"]
        print("  %-34s row-k %.2e  other-rows %.2e  final %.2e"
              % ("%dx%d permute layer %d" % (width, depth, k),
                 g["d_k"], g["d_other"], g["d_final"]))
        print("  %-34s MSE row-k %.2e  other %.2e  final %.2e"
              % ("", g["mse_k"], g["mse_other"], g["mse_final"]))
        print("  %-34s worst %.2e = %.4f%% of raw MSE   %s%s"
              % ("", g["worst"], g["worst"] / denom * 100,
                 "ok" if g["ok"] else "FAILED",
                 "" if g["finite"] else "  (NON-FINITE component)"))
    return [ok]


def inv_budget_monotone(est, est_mod):
    """More budget must never buy fewer samples, and no rung may throw."""
    print("\n10. BUDGET MONOTONICITY  (the failsafe ladder)")
    m = canon(PD, PW, 30_806_006_021)
    ladder = [0, 1, 10**6, 10**7, 10**8, 10**9, 10**10, 27_200_000_000,
              272_000_000_000, 2_720_000_000_000]
    # TWO invariants live here, and only one of them is universal:
    #
    #   (a) every rung DEGRADES rather than throws, and returns a valid array;
    #   (b) more budget never buys FEWER samples.
    #
    # (b) needs `est._log2_samples`, a v2.x sampling-architecture internal. This
    # loop used to call it unconditionally, so any candidate that does not
    # define it -- including every research candidate from
    # `p2_candidate_architecture.py`, and WP0C's own positive control -- failed
    # (a) as well, with ten `AttributeError`s that said nothing about the
    # candidate. (a) is now measured for everyone; (b) is measured where it is
    # measurable and reported NOT PROVABLE where it is not. A skip is not a pass
    # and is printed as such.
    has_ladder = callable(getattr(est, "_log2_samples", None)) and \
        hasattr(est_mod, "_MIN_LOG2_N")
    prev_n, ok, rows = -2, True, []
    for b in ladder:
        log2n = None
        try:
            if has_ladder:
                log2n = est._log2_samples(PW, PD, b)
            out = np.asarray(_predict(est, m, b), dtype=np.float64)
            # Same scaled non-negativity bar as `check()` above and as
            # `certify_candidate.py` section 5: an absolute `>= 0` is exact at
            # width 256 and wrong at width 1024, where a float64 sum on a
            # near-zero neuron rounds to ~-1e-16 against row means of ~0.46.
            _tol = -1e-12 * max(1.0, float(np.max(np.abs(out))))
            fin = (bool(np.isfinite(out).all()) and out.shape == (PD, PW)
                   and bool((out >= _tol).all()))
        except Exception as e:
            print(f"  budget {b:>16,}  RAISED {type(e).__name__}: {e}")
            ok = False
            continue
        if has_ladder:
            mono = log2n >= prev_n
            mode = "analytic" if log2n < est_mod._MIN_LOG2_N else f"N=2^{log2n}"
            prev_n = log2n
        else:
            mono, mode = True, "n/a"
        ok &= fin and mono
        rows.append((b, mode, fin, mono))
    for b, mode, fin, mono in rows:
        print(f"  budget {b:>16,}  {mode:<10} valid={fin}  non-decreasing={mono}")
    if has_ladder:
        print(f"  {'ladder monotone and total':<34} {'ok' if ok else 'FAILED'}")
    else:
        print(f"  {'every rung degrades, none throws':<34} "
              f"{'ok' if ok else 'FAILED'}")
        print(f"  {'sample-count monotonicity':<34} NOT PROVABLE -- this "
              "candidate exposes no _log2_samples")
    return [ok]


def inv_determinism(est, est_mod, evidence=None):
    """Three bit-identical runs, plus a fresh Estimator instance."""
    print("\n11. DETERMINISM  (3x bit-identity + fresh instance)")
    ok = True
    for case_index, (depth, width, seed) in enumerate(
            ((PD, PW, 30_806_006_031), (8, 128, 30_806_006_032))):
        candidate_budget = _correctness_budget(depth, width)
        m = canon(depth, width, seed)
        runs = [np.asarray(_predict(est, m, candidate_budget), dtype=np.float64)
                for _ in range(3)]
        same3 = all(np.array_equal(runs[0], r) for r in runs[1:])

        fresh = est_mod.Estimator()

        class Ctx:
            flop_budget, seed = candidate_budget, 0

        Ctx.width, Ctx.depth = width, depth
        fresh.setup(Ctx())
        r4 = np.asarray(_predict(fresh, m, candidate_budget), dtype=np.float64)
        if evidence is not None:
            evidence.append({
                "case_index": case_index, "width": width, "depth": depth,
                "candidate_budget": candidate_budget,
                "setup_width": Ctx.width, "setup_depth": Ctx.depth,
                "setup_budget": Ctx.flop_budget,
                "reused_estimator_identity": id(est),
                "fresh_setup_estimator_identity": id(fresh),
                "fresh_predict_estimator_identity": id(fresh),
                "fresh_distinct_from_reused": id(fresh) != id(est),
            })
        same_fresh = np.array_equal(runs[0], r4)
        ok &= same3 and same_fresh
        print(f"  {f'{width}x{depth}':<34} 3x identical={same3}  "
              f"fresh-instance identical={same_fresh}  "
              f"{'ok' if same3 and same_fresh else 'FAILED'}")
    return [ok]


def inv_seed_hygiene():
    print("\n12. SEED-BLOCK HYGIENE")
    from seed_blocks import DECISION_BLOCKS, VAULT_BLOCK, check_disjoint

    try:
        ok = check_disjoint()
    except AssertionError as e:
        print(f"  {'live blocks disjoint':<34} FAILED: {e}")
        return [False]
    print(f"  {'live blocks disjoint':<34} ok   "
          f"({len(DECISION_BLOCKS)} decision blocks + vault block "
          f"{VAULT_BLOCK[1]:,}..{VAULT_BLOCK[1] + VAULT_BLOCK[2] - 1:,})")
    return [ok]


def inv_gt_noise_reporting():
    """The vault's GT noise, so every MSE quoted against it is read correctly."""
    print("\n13. GT-NOISE FLOOR  (what counts as signal)")
    import json

    man_path = os.path.join(_ROOT, "experiments", "vault", "manifest.json")
    if not os.path.exists(man_path):
        print(f"  {'vault not minted yet':<34} skipped")
        return []
    man = json.load(open(man_path))
    noise = man["gt_noise_mse_mean"]
    print(f"  {'vault GT samples/MLP':<34} {man['gt_samples_per_mlp']:,}")
    print(f"  {'vault GT-noise MSE (mean)':<34} {noise:.3e}  "
          f"({noise / 1.4e-6 * 100:.3f}% of the 1.4e-6 decision signal)")
    print(f"  {'vault touches':<34} {man['touches']}")
    ok = noise < 1.4e-7  # under 10% of the signal, or the vault cannot arbitrate
    print(f"  {'noise < 10% of signal':<34} {'ok' if ok else 'FAILED'}")
    return [ok]


def invariants(est, est_mod):
    out = []
    out += inv_layer1(est)
    out += inv_permutation(est)
    out += inv_budget_monotone(est, est_mod)
    out += inv_determinism(est, est_mod)
    out += inv_seed_hygiene()
    out += inv_gt_noise_reporting()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
