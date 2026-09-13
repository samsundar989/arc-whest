"""B3: fixed n4/depth4 full-rank compact nonlinear recurrence differential.

The compact arm interprets the frozen coefficient *data* independently.  It
never calls a frozen nonlinear helper.  A line trace observes the unchanged
frozen ``_layer`` reference at its existing local checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

import p2_compact_fullrank_bridge_b1 as b1
import p2_compact_reference as compact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "2026-09-13_b3_fullrank_nonlinear.json"
TOL = 1e-5
N = RANK = RESIDUAL_RANK = 4


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
    checks.append({"layer": layer, "field": name, "shape": list(w.shape),
                   "finite": finite, "normalized_error": error,
                   "passed": same_shape and finite and error <= TOL})


def inventory(checks, layer, name, got, want):
    actual = sorted(map(str, got))
    expected = sorted(map(str, want))
    checks.append({"layer": layer, "field": name, "actual": actual,
                   "expected": expected, "passed": actual == expected})


def positive(checks, layer, name, value, threshold=1e-10):
    magnitude = float(np.max(np.abs(arr(value))))
    finite = math.isfinite(magnitude)
    if not finite:
        magnitude = 1e300
    checks.append({"layer": layer, "field": name, "magnitude": magnitude,
                   "threshold": threshold,
                   "passed": finite and magnitude > threshold})


def poly(coefficients, x):
    out = np.zeros_like(x, dtype=np.float64)
    for coefficient in reversed(coefficients):
        out = out * x + float(coefficient)
    return out


def independent_wick(mean, sigma, alpha, table):
    # Scalar stdlib normal functions avoid frozen _wick and scipy's vector API.
    phi = np.array([math.exp(-float(x) ** 2 / 2) / math.sqrt(2 * math.pi)
                    for x in alpha])
    Phi = np.array([0.5 * math.erfc(-float(x) / math.sqrt(2))
                    for x in alpha])
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
    frozen = b1.load_frozen()
    table = frozen._tables()  # coefficient data only; no frozen nonlinear helper
    weights = b1.WEIGHTS
    assert weights.shape == (4, N, N)
    checks, activity = [], []
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        dtype = fnp.float32
        const = {"off_mask": fnp.ones((N, N), dtype=dtype) - fnp.eye(N, dtype=dtype),
                 "ones_n": fnp.ones(N, dtype=dtype),
                 "wick": frozen._wick_program(table, dtype)}
        reference = (fnp.zeros(N, dtype=dtype), fnp.eye(N, dtype=dtype), None, 0.0)
        compact_state = (np.zeros(N), np.eye(N), compact.Shared(), 0.0)
        for layer, fixed in enumerate(weights):
            final = layer == len(weights) - 1
            W = fnp.array(fixed, dtype=dtype)
            Wc = fixed.astype(np.float32).T.astype(np.float64)
            mu, cov, shared, c4 = compact_state
            record(checks, layer, "incoming.mu", mu, reference[0])
            record(checks, layer, "incoming.cov", cov, reference[1])
            record(checks, layer, "incoming.c4", c4, reference[3])
            if layer:
                compare_sources(checks, layer, shared, reference[2])
            advanced = compact.transport_shared(shared, Wc, np.ones(N))
            d3, d21_raw = compact.compact_slices(advanced, need_d21=not final)
            D3 = np.zeros(N) if d3 is None else d3
            off_mask = np.ones((N, N)) - np.eye(N)
            D21 = (None if final else
                   np.zeros((N, N)) if d21_raw is None else d21_raw * off_mask)
            pre_mu = Wc @ mu
            pre_cov = Wc @ cov @ Wc.T
            var = np.maximum(np.diag(pre_cov), 1e-12)
            sigma = np.sqrt(var)
            alpha = np.clip(pre_mu / sigma, -40.0, 40.0)
            w = independent_wick(pre_mu, sigma, alpha, table)
            D4 = c4 * 4.0 * np.ones(N)
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
                       np.diag(D21), np.zeros(N))
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
                                 "D3_max_abs": float(np.max(np.abs(D3))),
                                 "D21_max_abs": None, "qr_rank": None})
                reference = ref_next
                continue
            w1, w2 = w[(1, 1)], w[(2, 1)]
            fac1 = w1[:, None] * C11
            p111_3 = (w1 ** 3) * D3
            p111_21 = ((w1 ** 2)[:, None] * w1[None, :]) * D21
            p111_21 += fac1 ** 2 * w2[None, :] * off_mask
            S3 = K[(3,)] - p111_3
            S21 = K[(2, 1)] - p111_21
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
                A=fac1, P=np.eye(N), Z=np.diag(S3) + 3.0 * S21.T,
                L=np.eye(N), s=np.zeros(N), e=np.zeros(N), w2=w2,
                dA=np.ones(N), dP=np.ones(N), c1=None, c2=None, Y=None, age=0)
            next_shared = compact.weighted_gram_join(old_scaled, newborn,
                                                      RANK, np.eye(N))
            next_cov = np.diag(K[(2,)]) + K[(1, 1)]
            lap = table["lap2_d4"]
            c4_out = projection_coefficient(N) * (
                float(lap["(4,)"]["terms"][0]["coef"])
                * float(lap["(4,)"]["int_partition_coef"])
                * float(np.sum(K[(4,)]))
                + float(lap["(2, 2)"]["terms"][0]["coef"])
                * float(lap["(2, 2)"]["int_partition_coef"])
                * float(np.sum(K[(2, 2)])))
            record(checks, layer, "post.cov", next_cov, ref_next[1])
            record(checks, layer, "post.c4", c4_out, ref_next[3])
            record(checks, layer, "birth.A_L", fac1,
                   b1.tower_sources(ref_next[2])[-1].A)
            record(checks, layer, "birth.A_D", compact.materialize(newborn),
                   b1.tower_sources(ref_next[2])[-1].Z)
            compare_sources(checks, layer, next_shared, ref_next[2])
            qr_rank = int(np.linalg.matrix_rank(next_shared.Q))
            record(checks, layer, "qr.rank", qr_rank, N)
            activity.append({"layer": layer, "ages": len(next_shared.factors),
                             "incoming_ages": len(advanced.factors),
                             "D3_max_abs": float(np.max(np.abs(D3))),
                             "D21_max_abs": float(np.max(np.abs(D21))),
                             "c4_max_abs": float(abs(c4_out)),
                             "p111_3_max_abs": float(np.max(np.abs(p111_3))),
                             "p111_21_max_abs": float(np.max(np.abs(p111_21))),
                             "qr_rank": qr_rank})
            compact_state = (mu_out, next_cov, next_shared, c4_out)
            reference = ref_next
        positive(checks, 2, "activity.off_diagonal_D21", activity[2]["D21_max_abs"])
        positive(checks, 1, "activity.scalar_c4", activity[1]["c4_max_abs"])
        positive(checks, 2, "activity.p111_3", activity[2]["p111_3_max_abs"])
        positive(checks, 2, "activity.p111_21", activity[2]["p111_21_max_abs"])
        record(checks, 2, "activity.two_ages_nonfinal",
               int(activity[2]["incoming_ages"] >= 2), 1)
        record(checks, 3, "final.output", compact_state[0], reference[0])
    return {"fixture": "B1/B2 fixed asymmetric n4/depth4; full shared/residual rank",
            "scope": "tiny full-rank nonlinear parity only; no compression/cost/score/truth",
            "tolerance": TOL,
            "normalization": "abs(got-want)/max(1,abs(want)) elementwise",
            "source_sha256": {"probe": sha(__file__), "frozen_L2": sha(b1.FROZEN),
                              "B1_probe": sha(b1.__file__),
                              "compact_reference": sha(b1.COMPACT)},
            "weights": weights.tolist(), "activity": activity, "checks": checks,
            "check_count": len(checks),
            "failed_count": sum(not row["passed"] for row in checks),
            "max_normalized_error": max(row.get("normalized_error", 0)
                                        for row in checks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"artifact": str(args.out), "checks": result["check_count"],
                      "failed": result["failed_count"],
                      "max_error": result["max_normalized_error"]}))
    if result["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
