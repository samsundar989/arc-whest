"""Direct label-free parity checks for generated compact candidates.

The independently written reference and the generated grader-facing modules
receive the same tiny float32-valued weights.  Full shared rank removes the
old-source basis approximation, so every post-first-layer prediction must agree
within the accepted Stage-1 tolerance.  Row zero is checked against the exact
float64 boundary formula that intentionally replaces the reference's internal
float32 Gaussian calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np

import harness
import lifecycle_eval as le
import p2_build_k4_arms as builder
from p2_compact_reference import metered_program


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N, DEPTH, RECENT, RANK, RESIDUAL_RANK, SEED = 8, 8, 4, 8, 8, 123
REL_TOL = 3.0e-5


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0e-300)
    return float(np.max(np.abs(a / scale - b / scale)))


def _weights():
    rng = np.random.default_rng(SEED)
    # Quantize before both programs see the values.  The lifecycle mint stores
    # float32 weights, while the grader supplies float64 arrays containing those
    # values.  raw.T is the reference program's W.
    return [(rng.normal(size=(N, N)) / math.sqrt(N)).astype(np.float32)
            for _ in range(DEPTH)]


def _candidate(path, raw_weights, *, shared_rank=None, residual_rank=None):
    flops = le._require_real_flopscope()
    fnp = le._require_real_fnp()
    module, _ = le.load_candidate(path)
    if shared_rank is not None:
        module._SHARED_RANK = int(shared_rank)
    if residual_rank is not None:
        module._RESIDUAL_RANK = int(residual_rank)
    spec = le.LifecycleSpec(width=N, depth=DEPTH,
                            residual_wall_time_limit_s=None)
    estimator = module.Estimator()
    estimator.setup(le.SetupContext(spec))
    weights = le.build_weights(raw_weights, "f64_readonly", fnp)
    mlp = harness.MLP(
        weights=weights,
        width=N, depth=DEPTH, seed=SEED)
    with flops.BudgetContext(flop_budget=100 * (2 ** 41), quiet=True) as ctx:
        prediction = estimator.predict(mlp, 2 ** 41)
        used = int(ctx.flops_used)
        operations = len(ctx.op_log)
    # Independently drive the candidate's layer kernel so the reference can be
    # compared before predict() applies its required exact-row and nonnegative
    # output boundary.  This is a test observation, not a second estimator.
    rank = min(module._SHARED_RANK, N)
    residual_rank = min(module._RESIDUAL_RANK, N)
    dtype = fnp.float32
    state = (fnp.zeros(N, dtype=dtype), fnp.eye(N, dtype=dtype),
             fnp.ones(N, dtype=dtype), [], None, None, None)
    internal = []
    states = []
    with flops.BudgetContext(flop_budget=100 * (2 ** 41), quiet=True):
        for layer, raw in enumerate(weights):
            W = raw.T if raw.dtype == dtype else raw.T.astype(dtype)
            state = module._layer(W, state, N, rank, residual_rank, dtype,
                                  layer == DEPTH - 1)
            mu = fnp.where(fnp.isfinite(state[0]), state[0], dtype(0.0))
            state = (mu, state[1], state[2], state[3], state[4],
                     state[5], state[6])
            internal.append(np.asarray(mu))
            states.append(state)
    return np.asarray(prediction), np.stack(internal), states, used, operations


def _source_values(source):
    values = list(source[:13])
    return [None if value is None else
            int(value) if i == 12 else np.asarray(value)
            for i, value in enumerate(values)]


def _candidate_sources(state):
    out = [_source_values(source) for source in state[3]]
    shared = state[4]
    if shared is not None:
        Q, factors, _gram = shared
        for factor in factors:
            FA, FP = factor[0], factor[1]
            dense = (Q @ FA, Q @ FP) + tuple(factor[2:13])
            out.append(_source_values(dense))
    return sorted(out, key=lambda values: values[12])


def _reference_sources(row):
    return sorted((_source_values(values) for _kind, values in row["sources"]),
                  key=lambda values: values[12])


def _state_error(states, reference_states, closures):
    worst = 0.0
    ages_exact = True
    for state, ref_state, closure in zip(states, reference_states, closures):
        worst = max(worst, _rel(state[0], closure["mu"]))
        if not closure["last"]:
            worst = max(worst, _rel(state[1], closure["C"]),
                        _rel(state[2], closure["w1"]))
            if state[6] is None or closure["gcore"] is None:
                if state[6] is not None or closure["gcore"] is not None:
                    return float("inf"), False
            else:
                worst = max(worst, _rel(state[6], closure["gcore"]))
        got_sources = _candidate_sources(state)
        want_sources = _reference_sources(ref_state)
        if len(got_sources) != len(want_sources):
            return float("inf"), False
        for got, want in zip(got_sources, want_sources):
            ages_exact = ages_exact and got[12] == want[12]
            # Compare the represented residual, because its QR factors are not
            # individually unique.  All other fields are representation values.
            worst = max(worst, _rel(got[2] @ got[3].T,
                                    want[2] @ want[3].T))
            for i in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11):
                if got[i] is None or want[i] is None:
                    if got[i] is not None or want[i] is not None:
                        return float("inf"), False
                else:
                    worst = max(worst, _rel(got[i], want[i]))
        got_birth, want_birth = state[5], ref_state["newborn"]
        if got_birth is None or want_birth is None:
            if got_birth is not None or want_birth is not None:
                return float("inf"), False
        else:
            worst = max(worst, _rel(np.asarray(got_birth[1])
                                    @ np.asarray(got_birth[2]).T,
                                    np.asarray(want_birth[1])
                                    @ np.asarray(want_birth[2]).T))
            for i in (0, 3, 4, 5, 6, 7, 8, 9, 10):
                if got_birth[i] is None or want_birth[i] is None:
                    if got_birth[i] is not None or want_birth[i] is not None:
                        return float("inf"), False
                else:
                    worst = max(worst, _rel(got_birth[i], want_birth[i]))
    return worst, ages_exact


def _generated_sources(lam=1.0 / 3.0):
    source, source_sha = builder._source()
    out = {}
    for name, consumers, births, blurb in builder.ARMS:
        expected = builder.build(name, consumers, births, blurb, source,
                                 source_sha, lam).encode("utf-8")
        path = os.path.join(builder.ARMS_DIR, "arm_%s.py" % name)
        actual = open(path, "rb").read()
        out[name] = {
            "path": path,
            "exact_regeneration": actual == expected,
            "sha256": hashlib.sha256(actual).hexdigest(),
            "consumers": consumers,
            "births": births,
        }
    return out


def run():
    raw_weights = [w.T for w in _weights()]
    rows = []
    generated = _generated_sources()
    for name, item in generated.items():
        reference = metered_program(
            N, DEPTH, RECENT, RANK, RESIDUAL_RANK, seed=SEED,
            consumers=item["consumers"], births=item["births"],
            lam=1.0 / 3.0, capture_prediction=True)
        (candidate, candidate_internal, _states, candidate_flops,
         candidate_operations) = _candidate(item["path"], raw_weights)
        internal = np.asarray(reference.pop("_prediction"))
        exact0 = (np.sqrt((raw_weights[0].astype(np.float64) ** 2).sum(axis=0))
                  / math.sqrt(2.0 * math.pi))
        row0_rel = _rel(candidate[0], exact0)
        internal_rel = _rel(candidate_internal, internal)
        rows.append({
            "arm": name,
            "pass": bool(item["exact_regeneration"]
                         and candidate.shape == (DEPTH, N)
                         and np.isfinite(candidate).all()
                         and row0_rel <= 2.0e-7
                         and internal_rel <= REL_TOL),
            "source_sha256": item["sha256"],
            "exact_regeneration": item["exact_regeneration"],
            "row0_exact_relative": row0_rel,
            "internal_reference_max_relative": internal_rel,
            "candidate_flops": candidate_flops,
            "reference_flops": reference["flops"],
            "output_boundary_flops": candidate_flops - reference["flops"],
            "candidate_operations": candidate_operations,
            "reference_operations": reference["operations"],
            "output_boundary_operations": (
                candidate_operations - reference["operations"]),
        })

        compressed_reference = metered_program(
            N, DEPTH, RECENT, 4, 2, seed=SEED,
            consumers=item["consumers"], births=item["births"],
            lam=1.0 / 3.0, capture_prediction=True, capture_trace=True)
        (_compressed_output, compressed_internal, compressed_states,
         _compressed_flops, _compressed_ops) = _candidate(
             item["path"], raw_weights, shared_rank=4, residual_rank=2)
        reference_internal = np.asarray(
            compressed_reference.pop("_prediction"))
        reference_states = compressed_reference.pop("_state_trace")
        reference_closures = compressed_reference.pop("_closure_trace")
        state_rel, ages_exact = _state_error(
            compressed_states, reference_states, reference_closures)
        compressed_rel = _rel(compressed_internal, reference_internal)
        rows[-1].update({
            "compressed_rank": 4,
            "compressed_residual_rank": 2,
            "compressed_internal_max_relative": compressed_rel,
            "compressed_state_max_relative": state_rel,
            "compressed_source_ages_exact": ages_exact,
        })
        rows[-1]["pass"] = bool(
            rows[-1]["pass"] and compressed_rel <= REL_TOL
            and state_rel <= REL_TOL and ages_exact)

    exact_path = generated["exact"]["path"]
    coherent_path = generated["coherent"]["path"]
    exact_output, _exact_internal, _exact_states, exact_flops, exact_ops = _candidate(
        exact_path, raw_weights)
    coherent_module, _ = le.load_candidate(coherent_path)
    coherent_module._K4_ACTIVE = False
    coherent_module._K4_CONSUMERS = False
    coherent_module._K4_BIRTHS = False
    # Exercise the same source with its master mechanism disabled, independently
    # of the generated exact arm.  Importing under a private name avoids mutation
    # of the module used above.
    flops = le._require_real_flopscope()
    fnp = le._require_real_fnp()
    spec = le.LifecycleSpec(width=N, depth=DEPTH,
                            residual_wall_time_limit_s=None)
    est = coherent_module.Estimator()
    est.setup(le.SetupContext(spec))
    mlp = harness.MLP(weights=le.build_weights(raw_weights, "f64_readonly", fnp),
                      width=N, depth=DEPTH, seed=SEED)
    with flops.BudgetContext(flop_budget=100 * (2 ** 41), quiet=True) as ctx:
        master_off = np.asarray(est.predict(mlp, 2 ** 41))
        master_off_flops = int(ctx.flops_used)
        master_off_ops = len(ctx.op_log)
    master_off = {
        "arm": "exact-vs-coherent-source-master-off",
        "pass": bool(np.array_equal(exact_output, master_off)
                     and exact_flops == master_off_flops
                     and exact_ops == master_off_ops),
        "output_identical": bool(np.array_equal(exact_output, master_off)),
        "flops": [exact_flops, master_off_flops],
        "operations": [exact_ops, master_off_ops],
    }
    failed = [row for row in rows if not row["pass"]]
    if not master_off["pass"]:
        failed.append(master_off)
    return {
        "kind": "label-free generated compact candidate/reference parity",
        "shape": [N, DEPTH], "recent": RECENT, "rank": RANK,
        "residual_rank": RESIDUAL_RANK, "seed": SEED,
        "relative_tolerance": REL_TOL,
        "arms": rows, "master_off": master_off,
        "n_checks": len(rows) + 1, "n_failed": len(failed),
        "pass": not failed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    result = run()
    for row in result["arms"]:
        print("%-18s %s rel=%.3e boundary=%d FLOPs/%d ops" % (
            row["arm"], "PASS" if row["pass"] else "FAIL",
            row["internal_reference_max_relative"],
            row["output_boundary_flops"], row["output_boundary_operations"]))
    print("%-18s %s" % ("master-off",
                               "PASS" if result["master_off"]["pass"] else "FAIL"))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
