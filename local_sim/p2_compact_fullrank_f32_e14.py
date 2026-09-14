"""E14: dtype-matched actual-f32 compact conformance, preserving the original failed B4 record.

The compact arm interprets the frozen coefficient *data* independently.  It
never calls a frozen nonlinear helper.  A line trace observes the unchanged
frozen ``_layer`` reference at its existing local checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
import flopscope as flops
import flopscope.numpy as fnp
if "site-packages" not in str(Path(flops.__file__).resolve()):
    raise RuntimeError("Real flopscope required; refusing the local NumPy shim")
sys.path.append(str(_HERE))
import numpy as np

import p2_compact_fullrank_bridge_b1 as b1
import p2_compact_f32_backend_e14 as compact
from p2_compact_fullrank_precision_e14 import json_safe


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "2026-09-14_e14" / "b4_real_backend" / "b4_f32.json"
TOL = 1e-5
N = RANK = RESIDUAL_RANK = 8
ORIGINAL = ROOT / "experiments" / "2026-09-12_b4_fullrank_deep.json"
ORIGINAL_SHA256 = "a5c6295c45695cc104f6db4d4f779cce059b18773184dfb418092953e420e52c"
CHECKS, ACTIVITY, CONTEXTS = [], [], []


class FirstMismatch(RuntimeError):
    """Abort the recurrence at the first checked discrepancy."""


def snapshot(value):
    """Copy diagnostic operands without relying on mutable trace dictionaries."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    if isinstance(value, (compact.Shared, compact.Source, compact.Factors)):
        return snapshot(vars(value))
    if hasattr(value, "A_D"):
        return snapshot({key: getattr(value, key)
                         for key in ("A_D", "A_L", "M", "w2b", "k")})
    return {"dtype": str(value.dtype), "values": arr(value).tolist()}


def append_check(checks, row):
    checks.append(row)
    # The historical 1e-5 status is retained, not used to silently retune or
    # truncate the separately preregistered f32 conformance trajectory.


def fixture_weights():
    """Explicit index formula preregistered in LOG; no RNG or data read."""
    return np.array([[[
        (0.78 + 0.05 * ((i + 2 * layer) % 4)) if i == j else
        (0.18 * (((3 * i + 5 * j + 7 * layer) % 13) - 6) / 6
         + 0.03 * (((i - j + 2 * layer) % 5) - 2) / 2)
        for j in range(N)] for i in range(N)] for layer in range(6)],
        dtype=np.float64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(value):
    return np.asarray(b1.arr(value), dtype=np.float64)


def record(checks, layer, name, got, want):
    g, w = arr(got), arr(want)
    same_shape = g.shape == w.shape
    finite = bool(np.isfinite(g).all() and np.isfinite(w).all())
    error = (float(np.max(np.abs(g - w) / np.maximum(1.0, np.abs(w))))
             if same_shape and finite and g.size else
             0.0 if same_shape and finite else 1e300)
    append_check(checks, {"layer": layer, "field": name, "shape": list(w.shape),
                   "finite": finite, "normalized_error": error,
                   "got": g.tolist(), "want": w.tolist(),
                   "passed": same_shape and finite and error <= TOL})


def inventory(checks, layer, name, got, want):
    actual = sorted(map(str, got))
    expected = sorted(map(str, want))
    append_check(checks, {"layer": layer, "field": name, "actual": actual,
                   "expected": expected, "passed": actual == expected})


def positive(checks, layer, name, value, threshold=1e-10):
    magnitude = float(np.max(np.abs(arr(value))))
    finite = math.isfinite(magnitude)
    if not finite:
        magnitude = 1e300
    append_check(checks, {"layer": layer, "field": name, "magnitude": magnitude,
                   "threshold": threshold,
                   "passed": finite and magnitude > threshold})


def poly(coefficients, x):
    out = fnp.zeros_like(x, dtype=fnp.float32)
    for coefficient in reversed(coefficients):
        out = out * x + float(coefficient)
    return out


def independent_wick(mean, sigma, alpha, table):
    # Scalar stdlib normal functions avoid frozen _wick and scipy's vector API.
    phi = fnp.array([math.exp(-float(x) ** 2 / 2) / math.sqrt(2 * math.pi)
                    for x in alpha], dtype=fnp.float32)
    Phi = fnp.array([0.5 * math.erfc(-float(x) / math.sqrt(2))
                    for x in alpha], dtype=fnp.float32)
    out = {}
    for text_key, spec in table["wick"].items():
        k, p = int(spec["k"]), int(spec["p"])
        if spec["branch"] == "poly":
            value = (sigma ** (p - k)
                     * (poly(spec["P1"], alpha) * phi
                        + poly(spec["P2"], alpha) * Phi))
        else:
            degree = k if spec["branch"] == "base" else int(spec["k_reduced"])
            if degree == 0:
                value = sigma * phi + mean * Phi
            elif degree == 1:
                value = Phi
            elif degree == 2:
                value = phi / sigma
            elif degree == 3:
                value = -alpha * phi / sigma ** 2
            elif degree == 4:
                value = (alpha ** 2 - 1) * phi / sigma ** 3
            else:
                raise ValueError((text_key, degree))
            if spec["branch"] != "base":
                value = value * float(spec["factorial_p"])
        out[(k, p)] = value
    return out


def independent_pk(S, w, table, final):
    pK, used = {}, []
    for index, row in enumerate(table["terms"]):
        ip = tuple(int(x) for x in row["int_part"])
        # Frozen fast_1d materializes every 1-D pK row even on the final
        # mean-only layer; only the later pK-to-K transform prunes to K[(1,)].
        if final and len(ip) != 1:
            continue
        term = float(row["scalar"])
        for block in row["blocks"]:
            term = term * S[tuple(int(x) for x in block["vec"])]
        for axis, key in enumerate(row["wick"]):
            wk = w[(int(key[0]), int(key[1]))]
            term = term * (wk if len(ip) == 1 else
                           wk[:, None] if axis == 0 else wk[None, :])
        pK[ip] = term if ip not in pK else pK[ip] + term
        used.append(index)
    for ip in pK:
        if len(ip) == 2 and ip[0] == ip[1]:
            pK[ip] = (pK[ip] + pK[ip].T) / 2
    return pK, used


def independent_center(pK, table, final):
    K, used = {}, []
    for text_key, rows in table["pK_to_K"].items():
        ip = tuple(int(x) for x in text_key.strip("()").split(",") if x.strip())
        if final and ip != (1,):
            continue
        total = None
        for index, row in enumerate(rows):
            product = None
            for raw_block in row["blocks"]:
                block = tuple(int(x) for x in raw_block)
                nz = tuple(i for i, value in enumerate(block) if value > 0)
                key = tuple(sorted((block[i] for i in nz), reverse=True))
                factor = pK[key]
                if len(key) == 2 and block[0] < block[1]:
                    factor = factor.T
                if len(nz) != len(ip):
                    factor = factor[:, None] if nz == (0,) else factor[None, :]
                product = factor if product is None else product * factor
            term = float(row["coef"]) * product
            total = term if total is None else total + term
            used.append((text_key, index))
        K[ip] = total
    return K, used


def projection_coefficient(n):
    # Direct n-dependent degree-4/rank-2 coefficient, independent of L2 helper.
    c = 1.0 + n / 2.0
    denominator = (1.0 - c) * (2.0 - c)
    return (c - 2.0) / (32.0 * c * denominator)


def scale_old(shared, wick):
    if shared.Q is None:
        return shared
    col = wick[:, None]
    factors = [compact.Factors(
        f.FA, f.FP, col * f.Z, f.L, f.s, f.e, f.w2,
        f.dA, f.dP, f.c1, f.c2,
        None if f.Y is None else wick * f.Y, f.age)
        for f in shared.factors]
    return compact.Shared(col * shared.Q, factors, shared.gram)


def observe_layer(frozen, W, state, table, dtype, const, final):
    """Call the unmodified reference and snapshot locals at source checkpoints."""
    source, first = inspect.getsourcelines(frozen._layer)
    markers = {"before_pk": "    pK = {}",
               "before_K": "    K = {}",
               "after_K": "    mu_out = K[(1,)]",
               "after_subtraction": "    if adv is not None:",
               "after_birth": "    lap = table[\"lap2_d4\"]"}
    lines = {}
    for label, marker in markers.items():
        hits = [first + i for i, line in enumerate(source) if line.startswith(marker)]
        if label == "after_subtraction":
            hits = [line for line in hits if line > lines["after_K"]]
        if len(hits) != 1:
            raise RuntimeError((label, hits))
        lines[label] = hits[0]
    by_line = {line: label for label, line in lines.items()}
    snapshots = {}
    previous = sys.gettrace()

    def tracer(frame, event, arg):
        if frame.f_code is not frozen._layer.__code__:
            return tracer
        if event == "line" and frame.f_lineno in by_line:
            snapshots[by_line[frame.f_lineno]] = dict(frame.f_locals)
        return tracer

    try:
        sys.settrace(tracer)
        output = frozen._layer(W, state, table, dtype, const, 2.0,
                               last_layer=final, gens=None)
    finally:
        sys.settrace(previous)
    required = ("before_pk", "before_K", "after_K") if final else tuple(markers)
    if not all(key in snapshots for key in required):
        raise RuntimeError(("missing trace checkpoints", sorted(snapshots)))
    return output, snapshots


def compare_sources(checks, layer, shared, frozen_tower,
                    prefix="source", age_offset=0):
    got = sorted(compact.reconstruct(shared), key=lambda s: -s.age)
    want = b1.tower_sources(frozen_tower, age_offset=age_offset)
    record(checks, layer, prefix + ".age_count", len(got), len(want))
    inventory(checks, layer, prefix + ".ages", [s.age for s in got],
              [s.age for s in want])
    for g, w in zip(got, want):
        for field in ("A", "P", "Z", "L", "s", "e", "w2", "dA", "dP"):
            record(checks, layer, f"{prefix}.age{w.age}.{field}",
                   getattr(g, field), getattr(w, field))
        record(checks, layer, f"{prefix}.age{w.age}.materialized_A_D",
               compact.materialize(g), compact.materialize(w))
        record(checks, layer, f"{prefix}.age{w.age}.feed_absent",
               int(g.c1 is g.c2 is g.Y is None), 1)


def run():
    if sha(ORIGINAL) != ORIGINAL_SHA256:
        raise RuntimeError("Original failed B4 artifact changed")
    CHECKS.clear()
    ACTIVITY.clear()
    CONTEXTS.clear()
    frozen = b1.load_frozen()
    table = frozen._tables()  # coefficient data only; no frozen nonlinear helper
    weights = fixture_weights().astype(np.float32).astype(np.float64)
    assert weights.shape == (6, N, N)
    checks, activity = CHECKS, ACTIVITY
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        dtype = fnp.float32
        const = {"off_mask": fnp.ones((N, N), dtype=dtype) - fnp.eye(N, dtype=dtype),
                 "ones_n": fnp.ones(N, dtype=dtype),
                 "wick": frozen._wick_program(table, dtype)}
        reference = (fnp.zeros(N, dtype=dtype), fnp.eye(N, dtype=dtype), None, 0.0)
        compact_state = (fnp.zeros(N, dtype=dtype), fnp.eye(N, dtype=dtype), compact.Shared(), 0.0)
        for layer, fixed in enumerate(weights):
            final = layer == len(weights) - 1
            W = fnp.array(fixed, dtype=dtype)
            Wc = fnp.array(fixed.T, dtype=dtype)
            mu, cov, shared, c4 = compact_state
            context = {"layer": layer, "final": final,
                       "weights": snapshot(W),
                       "reference_incoming": snapshot(reference),
                       "compact_incoming": snapshot(compact_state)}
            CONTEXTS.append(context)
            record(checks, layer, "incoming.mu", mu, reference[0])
            record(checks, layer, "incoming.cov", cov, reference[1])
            record(checks, layer, "incoming.c4", c4, reference[3])
            if layer:
                compare_sources(checks, layer, shared, reference[2])
            advanced = compact.transport_shared(shared, Wc, fnp.ones(N, dtype=dtype))
            d3, d21_raw = compact.compact_slices(advanced, need_d21=not final)
            D3 = fnp.zeros(N, dtype=dtype) if d3 is None else d3
            off_mask = fnp.ones((N, N), dtype=dtype) - fnp.eye(N, dtype=dtype)
            D21 = (None if final else
                   fnp.zeros((N, N), dtype=dtype) if d21_raw is None else d21_raw * off_mask)
            pre_mu = Wc @ mu
            pre_cov = Wc @ cov @ Wc.T
            var = fnp.maximum(fnp.diagonal(pre_cov), dtype(1e-12))
            sigma = fnp.sqrt(var)
            alpha = fnp.clip(pre_mu / sigma, dtype(-40.0), dtype(40.0))
            w = independent_wick(pre_mu, sigma, alpha, table)
            D4 = c4 * 4.0 * fnp.ones(N, dtype=dtype)
            S = {(1,): pre_mu, (3,): D3, (4,): D4}
            if not final:
                C11 = pre_cov * off_mask
                D22 = (c4 * 4.0 / 3.0) * off_mask
                S.update({(2,): var, (1, 1): C11, (2, 1): D21,
                          (1, 2): D21.T, (2, 2): D22})
            pK, used_terms = independent_pk(S, w, table, final)
            K, used_center = independent_center(pK, table, final)
            ref_next, trace = observe_layer(frozen, W, reference, table, dtype,
                                            const, final)
            context["reference_checkpoints"] = {
                label: snapshot({key: value for key, value in frame.items()
                                 if key in {"mu_pre", "cov_pre", "var", "sigma", "alpha",
                                            "w", "S", "pK", "K", "p111_3", "p111_21",
                                            "S3", "S21", "adv", "out_tw", "c4"}})
                for label, frame in trace.items()}
            context["compact_pre"] = snapshot({
                "mu_pre": pre_mu, "cov_pre": pre_cov, "var": var,
                "sigma": sigma, "alpha": alpha, "w": w, "S": S,
                "pK": pK, "K": K, "advanced": advanced})
            observed = trace["before_K"]
            record(checks, layer, "pre.mu", pre_mu, trace["before_pk"]["mu_pre"])
            record(checks, layer, "pre.cov", pre_cov, trace["before_pk"]["cov_pre"])
            inventory(checks, layer, "S.keys", S, observed["S"])
            inventory(checks, layer, "wick.keys", w, observed["w"])
            inventory(checks, layer, "wick.table_keys", w,
                      [(int(spec["k"]), int(spec["p"]))
                       for spec in table["wick"].values()])
            for key in sorted(S):
                record(checks, layer, f"S.{key}", S[key], observed["S"][key])
            record(checks, layer, "D3.compact", D3, observed["S"][(3,)])
            if not final:
                record(checks, layer, "D21.compact_masked", D21,
                       observed["S"][(2, 1)])
                record(checks, layer, "D21.transpose", D21.T,
                       observed["S"][(1, 2)])
                record(checks, layer, "D21.diagonal_zero",
                       fnp.diagonal(D21), np.zeros(N))
            for key in sorted(w):
                record(checks, layer, f"wick.{key}", w[key], observed["w"][key])
            inventory(checks, layer, "pK.keys", pK, observed["pK"])
            for key in sorted(pK):
                record(checks, layer, f"pK.{key}", pK[key], observed["pK"][key])
            inventory(checks, layer, "K.keys", K, trace["after_K"]["K"])
            for key in sorted(K):
                record(checks, layer, f"K.{key}", K[key], trace["after_K"]["K"][key])
            inventory(checks, layer, "terms.row_inventory", used_terms,
                      [i for i, row in enumerate(table["terms"])
                       if not final or len(row["int_part"]) == 1])
            inventory(checks, layer, "center.row_inventory", used_center,
                      [(key, i) for key, rows in table["pK_to_K"].items()
                       if not final or key.replace(" ", "") in ("(1,)", "(1)")
                       for i in range(len(rows))])
            mu_out = K[(1,)]
            record(checks, layer, "post.mu", mu_out, ref_next[0])
            if final:
                record(checks, layer, "final.tower_absent", int(ref_next[2] is None), 1)
                record(checks, layer, "final.c4_zero", ref_next[3], 0)
                compact_state = (mu_out, None, compact.Shared(), 0.0)
                activity.append({"layer": layer, "ages": 0,
                                 "D3_max_abs": float(np.max(np.abs(arr(D3)))),
                                 "D21_max_abs": None, "qr_rank": None})
                reference = ref_next
                continue
            w1, w2 = w[(1, 1)], w[(2, 1)]
            fac1 = w1[:, None] * C11
            p111_3 = (w1 ** 3) * D3
            p111_21 = ((w1 ** 2)[:, None] * w1[None, :]) * D21
            p111_21 = p111_21 + fac1 ** 2 * w2[None, :] * off_mask
            S3 = K[(3,)] - p111_3
            S21 = K[(2, 1)] - p111_21
            context["compact_subtraction"] = snapshot({
                "p111_3": p111_3, "p111_21": p111_21, "S3": S3, "S21": S21})
            for name, value in (("p111_3", p111_3), ("p111_21", p111_21)):
                record(checks, layer, name, value, observed[name])
            record(checks, layer, "birth.S3", S3,
                   trace["after_subtraction"]["S3"])
            record(checks, layer, "birth.S21", S21,
                   trace["after_subtraction"]["S21"])
            old_scaled = scale_old(advanced, w1)
            compare_sources(checks, layer, old_scaled,
                            trace["after_birth"]["adv"],
                            prefix="old_after_wick", age_offset=1)
            newborn = compact.Source(
                A=fac1, P=fnp.eye(N, dtype=dtype), Z=fnp.diag(S3) + 3.0 * S21.T,
                L=fnp.eye(N, dtype=dtype), s=fnp.zeros(N, dtype=dtype), e=fnp.zeros(N, dtype=dtype), w2=w2,
                dA=fnp.ones(N, dtype=dtype), dP=fnp.ones(N, dtype=dtype), c1=None, c2=None, Y=None, age=0)
            sketch = newborn.A @ (newborn.dA[:, None]
                                   * (newborn.A.T @ fnp.eye(N, dtype=dtype)))
            sketch = sketch + newborn.P @ (newborn.dP[:, None]
                                    * (newborn.P.T @ fnp.eye(N, dtype=dtype)))
            if old_scaled.Q is not None:
                sketch = sketch + old_scaled.Q @ (old_scaled.gram
                                          @ (old_scaled.Q.T @ fnp.eye(N, dtype=dtype)))
            sketch_rank = int(np.linalg.matrix_rank(arr(sketch)))
            record(checks, layer, "qr.pre_sketch_rank", sketch_rank, N)
            next_shared = compact.weighted_gram_join(old_scaled, newborn,
                                                      RANK, fnp.eye(N, dtype=dtype))
            context["compact_birth"] = snapshot({
                "old_scaled": old_scaled, "newborn": newborn,
                "sketch": sketch, "next_shared": next_shared})
            next_cov = fnp.diag(K[(2,)]) + K[(1, 1)]
            lap = table["lap2_d4"]
            c4_out = projection_coefficient(N) * (
                float(lap["(4,)"]["terms"][0]["coef"])
                * float(lap["(4,)"]["int_partition_coef"])
                * float(fnp.sum(K[(4,)]))
                + float(lap["(2, 2)"]["terms"][0]["coef"])
                * float(lap["(2, 2)"]["int_partition_coef"])
                * float(fnp.sum(K[(2, 2)])))
            record(checks, layer, "post.cov", next_cov, ref_next[1])
            record(checks, layer, "post.c4", c4_out, ref_next[3])
            record(checks, layer, "birth.A_L", fac1,
                   b1.tower_sources(ref_next[2])[-1].A)
            record(checks, layer, "birth.A_D", compact.materialize(newborn),
                   b1.tower_sources(ref_next[2])[-1].Z)
            compare_sources(checks, layer, next_shared, ref_next[2])
            qr_rank = int(np.linalg.matrix_rank(arr(next_shared.Q)))
            record(checks, layer, "qr.rank", qr_rank, N)
            activity.append({"layer": layer, "ages": len(next_shared.factors),
                             "incoming_ages": len(advanced.factors),
                             "D3_max_abs": float(np.max(np.abs(arr(D3)))),
                             "D21_max_abs": float(np.max(np.abs(arr(D21)))),
                             "c4_max_abs": float(abs(c4_out)),
                             "p111_3_max_abs": float(np.max(np.abs(arr(p111_3)))),
                             "p111_21_max_abs": float(np.max(np.abs(arr(p111_21)))),
                             "pre_sketch_rank": sketch_rank,
                             "qr_rank": qr_rank})
            compact_state = (mu_out, next_cov, next_shared, c4_out)
            reference = ref_next
            context["compact_outgoing"] = snapshot(compact_state)
            context["reference_outgoing"] = snapshot(reference)
        positive(checks, 4, "activity.off_diagonal_D21", activity[4]["D21_max_abs"])
        positive(checks, 4, "activity.scalar_c4", activity[4]["c4_max_abs"])
        positive(checks, 4, "activity.p111_3", activity[4]["p111_3_max_abs"])
        positive(checks, 4, "activity.p111_21", activity[4]["p111_21_max_abs"])
        record(checks, 4, "activity.four_ages_nonfinal",
               int(activity[4]["incoming_ages"] >= 4), 1)
        record(checks, 5, "final.output", compact_state[0], reference[0])
    return {"fixture": "B4 fixed asymmetric n8/depth6 formula; full shared/residual rank",
            "scope": "tiny full-rank nonlinear parity only; no compression/cost/score/truth",
            "tolerance": TOL,
            "normalization": "abs(got-want)/max(1,abs(want)) elementwise",
            "source_sha256": {"adapter": sha(__file__), "original_B4_probe": sha(ROOT / "local_sim" / "p2_compact_fullrank_deep_b4.py"), "frozen_L2": sha(b1.FROZEN),
                              "B1_probe": sha(b1.__file__),
                              "B3_probe": sha(ROOT / "local_sim" / "p2_compact_fullrank_nonlinear_b3.py"),
                              "compact_reference": sha(b1.COMPACT)},
            "weights": weights.tolist(), "activity": activity, "checks": checks,
            "check_count": len(checks),
            "failed_count": sum(not row["passed"] for row in checks),
            "max_normalized_error": max(row.get("normalized_error", 0)
                                        for row in checks)}


OUTPUT_ALLOWANCE = 2e-10
ENVELOPE_FACTOR = 4.0
EPS_FACTOR = 64.0
F21 = ROOT / 'estimator_k3_strassen_L4_f21.py'
F21_SHA = 'd77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94'


def output_check(got, want):
    g, w = arr(got), arr(want)
    drift = float(np.mean((g - w) ** 2))
    return {'prediction_squared_drift': drift, 'allowance': OUTPUT_ALLOWANCE,
            'passed': bool(np.isfinite(drift) and drift <= OUTPUT_ALLOWANCE)}


def field_envelope(got, ref32, ref64):
    g, r, h = arr(got), arr(ref32), arr(ref64)
    shape = g.shape == r.shape == h.shape
    finite = bool(np.isfinite(g).all() and np.isfinite(r).all() and np.isfinite(h).all())
    error = float(np.max(np.abs(g - r))) if shape and finite else 1e300
    drift = float(np.max(np.abs(r - h))) if shape and finite else 1e300
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(r)))) if shape and finite else 0
    allowed = ENVELOPE_FACTOR * drift + EPS_FACTOR * float(np.finfo(np.float32).eps) * scale
    return {'max_abs_difference': error, 'reference32_vs_reference64': drift,
            'reference_scale': scale, 'allowed': allowed,
            'passed': bool(shape and finite and error <= allowed)}


def load_reference(path, forced):
    import importlib.util
    spec = importlib.util.spec_from_file_location('e14_reference_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if forced:
        # Tiny shapes exercise the exact frozen Strassen code to their maximum
        # supported depth (3 for n8, 2 for n4); L4 still requires width >=16.
        module._STRASSEN_MIN_BLOCK = 1
    return module


def execute_case(base, weights, model, forced, outdir, label):
    """Adapt only fixed shape/activity bounds and reference import per case.

    The exact generated run source is saved before execution. The full nonlinear
    function/helper source lives in this driver (f32) or the E14 f64 adapter.
    """
    from types import SimpleNamespace
    n, depth = weights.shape[1], weights.shape[0]
    reference = load_reference(model, forced)
    bridge = SimpleNamespace(**vars(b1))
    bridge.FROZEN = model
    bridge.load_frozen = lambda: reference
    namespace = dict(vars(base))
    namespace.update(N=n, RANK=n, RESIDUAL_RANK=n, CHECKS=[], ACTIVITY=[],
                     CONTEXTS=[], fixture_weights=lambda: weights.copy(), b1=bridge)
    source = inspect.getsource(base.run)
    source = source.replace('assert weights.shape == (6, N, N)',
                            'assert weights.shape == (len(weights), N, N)')
    source = source.replace('activity[4]', 'activity[len(weights) - 2]')
    source = source.replace('positive(checks, 4,', 'positive(checks, len(weights) - 2,')
    source = source.replace('record(checks, 4,', 'record(checks, len(weights) - 2,')
    source = source.replace('>= 4', '>= len(weights) - 2')
    source = source.replace('activity.four_ages_nonfinal', 'activity.all_ages_nonfinal')
    source = source.replace('record(checks, 5, "final.output"',
                            'record(checks, len(weights) - 1, "final.output"')
    adapter_path = outdir / (label + '_run_adapter.py')
    adapter_path.write_text(source, encoding='utf-8')
    exec(compile(source, str(adapter_path), 'exec'), namespace)
    result = namespace['run']()
    result['checkpoints'] = namespace['CONTEXTS']
    result['runtime_adapter'] = {'path': str(adapter_path.relative_to(ROOT)),
                                 'sha256': sha(adapter_path), 'shape': [depth, n],
                                 'reference_source': str(model.relative_to(ROOT)),
                                 'reference_sha256': sha(model),
                                 'forced_min_block': 1 if forced else None,
                                 'depth_on_square_n': reference._depth_for(n, n, n)}
    result['fixture'] = label
    return result


def dtype_inventory(value, prefix=''):
    rows = []
    if isinstance(value, dict):
        if 'dtype' in value and 'values' in value:
            rows.append({'path': prefix, 'dtype': value['dtype'],
                         'passed': value['dtype'] == 'float32'})
        else:
            for key, item in value.items():
                rows.extend(dtype_inventory(item, prefix + '/' + key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            rows.extend(dtype_inventory(item, prefix + '/' + str(i)))
    return rows


def assess(single32, single64):
    fields = []
    if len(single32['checks']) != len(single64['checks']):
        raise RuntimeError('f32/f64 checkpoint inventory differs')
    for low, high in zip(single32['checks'], single64['checks']):
        if (low['layer'], low['field']) != (high['layer'], high['field']):
            raise RuntimeError('f32/f64 checkpoint order differs')
        row = {'layer': low['layer'], 'field': low['field'],
               'original_tolerance_passed': low['passed']}
        if 'got' in low:
            row.update(field_envelope(low['got'], low['want'], high['want']))
        else:
            row['passed'] = low['passed'] and high['passed']
        fields.append(row)
    final = next(row for row in single32['checks'] if row['field'] == 'final.output')
    output = output_check(final['got'], final['want'])
    output_cast = output_check(np.asarray(final['got'], dtype=np.float32),
                               np.asarray(final['want'], dtype=np.float32))
    dtype_rows = []
    for checkpoint in single32['checkpoints']:
        for key, value in checkpoint.items():
            if key.startswith('compact_'):
                dtype_rows.extend(dtype_inventory(value, str(checkpoint['layer']) + '/' + key))
    if not dtype_rows:
        raise RuntimeError('No actual compact dtype evidence captured')
    passed = (all(row['passed'] for row in fields)
              and all(row['passed'] for row in dtype_rows)
              and output['passed'] and output_cast['passed'])
    return {'passed': passed, 'fields': fields,
            'failed_fields': [row for row in fields if not row['passed']],
            'original_tolerance_failed_count': single32['failed_count'],
            'dtype_inventory': dtype_rows, 'output': output,
            'official_output_cast': output_cast}


def main():
    from threadpoolctl import threadpool_info, threadpool_limits
    import p2_compact_fullrank_precision_e14 as f64
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT.parent)
    parser.add_argument('--reuse-completed-fixed-cases', action='store_true',
                        help='Read existing synthetic fixture records; never accesses labels or seeds.')
    args = parser.parse_args()
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if sha(F21) != F21_SHA:
        raise RuntimeError('F21 source identity mismatch')
    cases = [('b4', fixture_weights()), ('b3', b1.WEIGHTS)]
    results = []
    with threadpool_limits(limits=1):
        backend = threadpool_info()
        for fixture, weights in cases:
            for name, model, forced in [('l2', b1.FROZEN, False), ('f21_forced', F21, True)]:
                label = fixture + '_' + name
                path = outdir / (label + '_f32_conformance.json')
                if args.reuse_completed_fixed_cases and path.exists():
                    case = json.loads(path.read_text(encoding='utf-8'))
                    if case['fixture'] != fixture or case['control'] != name:
                        raise RuntimeError('Existing fixed-case identity mismatch')
                    lo = case['f32']
                else:
                    hi = execute_case(f64, weights, model, forced, outdir, label + '_f64')
                    lo = execute_case(sys.modules[__name__], weights, model, forced, outdir, label + '_f32')
                    case = {'fixture': fixture, 'control': name, 'f64': hi, 'f32': lo,
                            'conformance': assess(lo, hi)}
                    path.write_text(json.dumps(json_safe(case), indent=2, allow_nan=False) + '\n', encoding='utf-8')
                results.append({'fixture': fixture, 'control': name,
                                'artifact': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                'passed': case['conformance']['passed'],
                                'original_tolerance_failed_count': lo['failed_count'],
                                'failed_field_count': len(case['conformance']['failed_fields']),
                                'output': case['conformance']['output']})
                print(json.dumps(results[-1]), flush=True)
    identity = output_check(np.zeros(8), np.zeros(8))
    injected = output_check(np.ones(8) * math.sqrt(2 * OUTPUT_ALLOWANCE), np.zeros(8))
    result = {'schema': 'e14.b4_actual_f32_conformance.v1',
              'regime': {'commit': '70819fc', 'original_tolerance': TOL,
                         'envelope_factor': ENVELOPE_FACTOR, 'epsilon_factor': EPS_FACTOR,
                         'output_allowance': OUTPUT_ALLOWANCE},
              'source_sha256': {'driver': sha(__file__), 'backend': sha(compact.__file__),
                                'f64_adapter': sha(f64.__file__), 'compact_reference': sha(b1.COMPACT),
                                'L2': sha(b1.FROZEN), 'F21': sha(F21)},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'flopscope_import_path': flops.__file__,
                          'fnp_import_path': fnp.__file__,
                          'flopscope': importlib.metadata.version('flopscope'),
                          'threadpools': backend},
              'cases': results, 'controls': {'identity': identity, 'injected_drift': injected},
              'passed': all(row['passed'] for row in results) and identity['passed'] and not injected['passed'],
              'original_B4_unchanged': sha(ORIGINAL) == ORIGINAL_SHA256,
              'limitations': ['Fixed n8/d6 and n4/d4 only; no 1024x16 production conformance.',
                              'Forced F21 tests at most3 or2 Strassen levels; no production L4 evidence.',
                              'No compression, ground truth, noise floor, prediction accuracy, counted cost or promotion claim.',
                              'The frozen normal helper and independent scalar normal evaluator both round to f32; exact operation-order equality is not required.']}
    path = outdir / 'b4_f32_summary.json'
    path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
