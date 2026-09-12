"""Truth-free, first-source-only K4/K3 factor and official-meter microkernel.

This is a cost instrument for one chosen weighted discrete law.  It deliberately
does not construct later K4 increments or an estimator prediction.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np


BUDGET = 2 ** 41
FROZEN_DENSE = 703252660224


def _real_flopscope():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]
    for p in (here, root):
        if p not in sys.path:
            sys.path.append(p)
    import flopscope as flops
    import flopscope.numpy as fnp
    if "site-packages" not in os.path.abspath(flops.__file__):
        raise RuntimeError("installed flopscope required")
    return flops, fnp


class Meter:
    def __init__(self, context):
        self.ctx = context
        self.families = {}
        self.calls = {}
        self.result_bytes = {}

    def run(self, family, fn):
        before = int(self.ctx.flops_used)
        value = fn()
        self.families[family] = self.families.get(family, 0) + int(self.ctx.flops_used) - before
        self.calls[family] = self.calls.get(family, 0) + 1
        self.result_bytes[family] = max(self.result_bytes.get(family, 0),
                                        int(getattr(value, "nbytes", 0)))
        return value


def _sha():
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _external_inputs(n, rank, depth):
    """Fixed synthetic inputs, unrelated to registry seeds or challenge truth."""
    rng = np.random.default_rng(20260912)
    raw = rng.normal(size=(n, rank)).astype(np.float64)
    p = np.full(rank, 1.0 / rank, dtype=np.float64)
    source_map = rng.normal(size=(n, n)).astype(np.float64) / math.sqrt(n)
    maps = [rng.normal(size=(n, n)).astype(np.float64) / math.sqrt(n)
            for _ in range(depth - 1)]
    w1 = [0.45 + 0.08 * rng.normal(size=n) for _ in range(depth - 1)]
    w2 = [0.12 + 0.03 * rng.normal(size=n) for _ in range(depth - 1)]
    return raw, p, source_map, maps, w1, w2


def metered_program(n=1024, depth=16, rank=64, capture=False):
    """Execute exactly one first source and its declared quadratic K3 births."""
    if n < 2 or depth < 3 or rank < 2:
        raise ValueError("n>=2, depth>=3, rank>=2 required")
    flops, f = _real_flopscope()
    raw, p_in, source_map_in, maps, w1_in, w2_in = _external_inputs(n, rank, depth)
    external_bytes = (raw.nbytes + p_in.nbytes + source_map_in.nbytes + sum(x.nbytes for x in maps)
                      + sum(x.nbytes for x in w1_in + w2_in))
    schedule = {"source_maps": 1, "k4_uses": 0, "k4_full": 0,
                "k4_final": 0, "births": 0, "feed_transports": 0,
                "shared_A_transports": 0, "feed_full_D21": 0,
                "feed_final_D3": 0, "negative_shared_D21": 0}
    trace = []
    with flops.BudgetContext(flop_budget=100 * BUDGET, quiet=True) as ctx:
        m = Meter(ctx)
        p = m.run("cast_source_weights", lambda: f.array(p_in, dtype=f.float32))
        nodes = m.run("cast_source_nodes", lambda: f.array(raw, dtype=f.float32))
        center = m.run("source_weighted_center", lambda: nodes @ p)
        centered = m.run("source_center_columns", lambda: nodes - center[:, None])
        source_map = m.run("cast_source_map", lambda: f.array(source_map_in, dtype=f.float32))
        V = m.run("dense_source_map", lambda: source_map @ centered)
        births = []
        peak_payload = 0
        max_single_array = 0
        layer_rows = []
        for layer in range(1, depth):
            final = layer == depth - 1
            B = m.run("cast_layer_map", lambda layer=layer: f.array(maps[layer - 1], dtype=f.float32))
            V = m.run("dense_k4_transport", lambda: B @ V)
            schedule["k4_uses"] += 1
            if final:
                schedule["k4_final"] += 1
            else:
                schedule["k4_full"] += 1
            # A source age is transported once per layer.  Old feed ages move
            # through the same signed map before their current readouts.
            moved = []
            for birth in births:
                A, P, a, V_birth, birth_p = birth
                A = m.run("dense_shared_A_transport", lambda A=A: B @ A)
                P = m.run("dense_feed_P_transport", lambda P=P: B @ P)
                a = m.run("dense_feed_a_transport", lambda a=a: B @ a)
                moved.append((A, P, a, V_birth, birth_p))
                schedule["shared_A_transports"] += 1
                schedule["feed_transports"] += 1
            births = moved
            V2 = m.run("k4_V_square", lambda: V * V)
            c = m.run("k4_covariance_diagonal", lambda: (V2 * p[None, :]).sum(axis=1))
            D4pos = m.run("k4_D4_positive", lambda: ((V2 * V2) * p[None, :]).sum(axis=1))
            D4 = m.run("k4_D4_negative", lambda: D4pos - 3.0 * c * c)
            if not final:
                Cp = m.run("k4_covariance_weight", lambda: V * p[None, :])
                C = m.run("dense_k4_C_gram", lambda: Cp @ V.T)
                V2p = m.run("k4_D22_weight", lambda: V2 * p[None, :])
                D22pos = m.run("dense_k4_D22_positive", lambda: V2p @ V2.T)
                D22 = m.run("k4_D22_negative", lambda: D22pos - c[:, None] * c[None, :] - 2.0 * C * C)
                V3p = m.run("k4_D31_weight", lambda: (V2 * V) * p[None, :])
                D31pos = m.run("dense_k4_D31_positive", lambda: V3p @ V.T)
                D31 = m.run("k4_D31_negative", lambda: D31pos - 3.0 * c[:, None] * C)
            D3 = m.run("readout_D3_init", lambda: f.zeros(n, dtype=f.float32))
            D21 = None if final else m.run("readout_D21_init", lambda: f.zeros((n, n), dtype=f.float32))
            for A, P, a, V_birth, birth_p in births:
                Pp = m.run("feed_weighted_P", lambda P=P, birth_p=birth_p: P * birth_p[None, :])
                Q = m.run("dense_feed_Q_reconstruct", lambda Pp=Pp, V_birth=V_birth: Pp @ V_birth.T)
                Vb2 = m.run("feed_birth_V_square", lambda V_birth=V_birth: V_birth * V_birth)
                Ap = m.run("dense_feed_Aplus_reconstruct", lambda A=A, Vb2=Vb2: A @ Vb2)
                Q2 = m.run("feed_Q_square", lambda Q=Q: Q * Q)
                if final:
                    hdiag = m.run("feed_H_diagonal", lambda P=P, Pp=Pp: (P * Pp).sum(axis=1))
                else:
                    H = m.run("dense_feed_H_gram", lambda Pp=Pp, P=P: Pp @ P.T)
                    hdiag = m.run("feed_H_diagonal_extract", lambda H=H: f.diagonal(H))
                d3positive = m.run("feed_D3_positive", lambda Ap=Ap, P=P, birth_p=birth_p:
                                   1.5 * (Ap * P * P * birth_p[None, :]).sum(axis=1))
                d3negative = m.run("feed_D3_negative", lambda A=A, Q2=Q2, a=a, hdiag=hdiag:
                                   -1.5 * a * hdiag - 3.0 * (A * Q2).sum(axis=1))
                D3 = m.run("readout_D3_accumulate", lambda D3=D3: D3 + d3positive + d3negative)
                if final:
                    schedule["feed_final_D3"] += 1
                else:
                    P2p = m.run("feed_D21_positive_left_weight", lambda P=P, birth_p=birth_p:
                                P * P * birth_p[None, :])
                    plus1 = m.run("dense_feed_D21_positive_left", lambda P2p=P2p, Ap=Ap: P2p @ Ap.T)
                    APp = m.run("feed_D21_positive_right_weight", lambda Ap=Ap, Pp=Pp: Ap * Pp)
                    plus2 = m.run("dense_feed_D21_positive_right", lambda APp=APp, P=P: APp @ P.T)
                    neg1 = m.run("dense_feed_D21_negative_left", lambda Q2=Q2, A=A: Q2 @ A.T)
                    AQ = m.run("feed_D21_negative_right_weight", lambda A=A, Q=Q: A * Q)
                    neg2 = m.run("dense_feed_D21_negative_right", lambda AQ=AQ, Q=Q: AQ @ Q.T)
                    elem = m.run("feed_D21_negative_elementwise", lambda a=a, hdiag=hdiag, H=H:
                                 -0.5 * (hdiag[:, None] * a[None, :] + 2.0 * a[:, None] * H))
                    delta = m.run("feed_D21_combine", lambda plus1=plus1, plus2=plus2, neg1=neg1, neg2=neg2, elem=elem:
                                  0.5 * plus1 + plus2 - neg1 - 2.0 * neg2 + elem)
                    D21 = m.run("readout_D21_accumulate", lambda D21=D21, delta=delta: D21 + delta)
                    schedule["feed_full_D21"] += 1
                    schedule["negative_shared_D21"] += 1
            if not final:
                w1 = m.run("cast_birth_w1", lambda layer=layer: f.array(w1_in[layer - 1], dtype=f.float32))
                w2 = m.run("cast_birth_w2", lambda layer=layer: f.array(w2_in[layer - 1], dtype=f.float32))
                A = m.run("birth_A_diagonal", lambda: f.diag(w2))
                P = m.run("birth_P", lambda: w1[:, None] * V)
                a = m.run("birth_a", lambda: w2 * c)
                V_birth = m.run("birth_V_copy", lambda: f.copy(V))
                births.append((A, P, a, V_birth, p))
                schedule["births"] += 1
            if capture and n <= 8:
                trace.append({"layer": layer, "V": np.asarray(V).copy(),
                              "D4": np.asarray(D4).copy(),
                              "D22": None if final else np.asarray(D22).copy(),
                              "D31": None if final else np.asarray(D31).copy(),
                              "D3": np.asarray(D3).copy(),
                              "D21": None if final else np.asarray(D21).copy()})
            # Explicit conservative payload account: persistent factors plus
            # the largest readout expression's live array workspace.  The OS
            # peak from workerq remains the authoritative process observation.
            persistent = (len(births) * (n * n + 2 * n * rank + n)
                          + n * rank + n * n) * 4
            scratch = (10 * n * n + 6 * n * rank) * 4
            peak_payload = max(peak_payload, persistent + scratch)
            max_single_array = max(max_single_array, n * n * 4)
            layer_rows.append({"layer": layer, "births_retained": len(births),
                               "full_readout": not final})
        used = int(ctx.flops_used)
        operations = len(ctx.op_log)
    if sum(m.families.values()) != used:
        raise AssertionError("operation-family ledger does not reconcile")
    dense = {key: value for key, value in m.families.items() if key.startswith("dense_")}
    negative = sum(value for key, value in dense.items() if key.startswith("dense_feed_D21_negative"))
    expected = {"source_maps": 1, "k4_uses": depth - 1,
                "k4_full": depth - 2, "k4_final": 1,
                "births": depth - 2, "feed_transports": (depth - 2) * (depth - 1) // 2,
                "shared_A_transports": (depth - 2) * (depth - 1) // 2,
                "feed_full_D21": (depth - 3) * (depth - 2) // 2,
                "feed_final_D3": depth - 2,
                "negative_shared_D21": (depth - 3) * (depth - 2) // 2}
    if schedule != expected:
        raise AssertionError("schedule mismatch: %r != %r" % (schedule, expected))
    result = {"kind": "label-free first-source-only exact discrete K4 and declared K3 feed cost",
              "source_sha256": _sha(), "shape": [n, depth], "rank": rank,
              "flops": used, "budget_fraction": used / BUDGET,
              "operations": operations, "families": m.families,
              "family_calls": m.calls, "family_max_result_bytes": m.result_bytes,
              "dense_families": dense, "dense_subtotal": sum(dense.values()),
              "dense_frozen_prediction": FROZEN_DENSE if (n, depth, rank) == (1024, 16, 64) else None,
              "dense_relative_to_frozen": (sum(dense.values()) / FROZEN_DENSE - 1.0)
                  if (n, depth, rank) == (1024, 16, 64) else None,
              "negative_D21_flops": negative,
              "schedule": schedule, "expected_schedule": expected,
              "memory": {"external_input_bytes": external_bytes,
                         "estimated_peak_array_payload_bytes": peak_payload,
                         "max_single_array_bytes": max_single_array,
                         "basis": "conservative persistent bank plus 10n^2+6nS readout scratch; process peak from workerq"},
              "layers": layer_rows,
              "scope": "one first-layer discrete source; no later source, K3 backbone, labels, score or server residual"}
    if capture:
        result["_trace"] = trace
        result["_inputs"] = (raw, p_in, source_map_in, maps, w1_in, w2_in)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    result = metered_program(a.width, a.depth, a.rank)
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("flops=%d dense=%d negative_D21=%d operations=%d" %
          (result["flops"], result["dense_subtotal"],
           result["negative_D21_flops"], result["operations"]))


if __name__ == "__main__":
    main()
