"""Deterministic array-shape audit for the integrated first-source instrument.

No challenge networks, large arrays, flopscope context, or labels are loaded.
The schedule follows p2_k4_integrated_first_source.py at its frozen SHA. This is
an explicit array-payload and named co-live allowance, not a process peak.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "local_sim/p2_k4_integrated_first_source.py"
SHORTCUT = (ROOT / "experiments/2026-09-11_stage2_compact_p07/"
            "integrated_first_source/shortcut_result.json")
FROZEN_SHA = "3a57e5e5709a62848f8412de094f218ea9b323933b9071e04c5dabbdab8b12b3"
FIRST_RESULT_SHA = "10c7835042b581b6691f6f98963a90adeeb222ba846557da2582396443025155"
FIRST_RESULT = SHORTCUT.with_name("memory_ledger_result.json")
COLIVE_RESULT_SHA = "f7d5c67bd549d66ecb6f3d3e1f0a2be055c9e7fad2e8fd9be181e84bb1f72b1f"
COLIVE_RESULT = SHORTCUT.with_name("memory_ledger_colive_result.json")
BOUNDARY_RESULT_SHA = "3368f9e26b9fb0ff506d9241f18a9b86c774cb5224681b99d202981b97a847cf"
BOUNDARY_RESULT = SHORTCUT.with_name("memory_ledger_boundary_result.json")
BOUNDARY_SOURCE_SHA = "990eb0f61c08962b42dd8ed0a5727dea0095b7d505ef34fe9dac20b070545c74"
ARRAY_LIMIT = 4 * 1024 ** 3
PROCESS_LIMIT = 8 * 1024 ** 3
PRIOR_14 = ("C_off", "bank_covariance", "positive22", "D22",
            "positive31", "D31", "K11", "off_mask", "local21",
            "old21", "repeated", "K21", "separable", "Rres")
CURRENT_BIRTH_9 = ("C_off", "bank_covariance", "positive22", "D22",
                   "positive31", "D31", "K11", "off_mask", "D21")


@dataclass(frozen=True)
class Array:
    key: str
    shape: tuple[int, ...]
    bytes_per_element: int = 4
    backing_shape: tuple[int, ...] | None = None

    @property
    def nbytes(self) -> int:
        count = 1
        for axis in self.backing_shape or self.shape:
            count *= axis
        return count * self.bytes_per_element


class Maker:
    def __init__(self):
        self.serial = 0
        self.all_arrays: list[Array] = []

    def make(self, label: str, shape: tuple[int, ...], bpe: int = 4) -> Array:
        self.serial += 1
        array = Array(f"{label}#{self.serial}", shape, bpe)
        self.all_arrays.append(array)
        return array


@dataclass
class History:
    kind: str
    age: int
    fields: dict[str, Array]

    def arrays(self) -> list[Array]:
        return list(self.fields.values())


def inventory(arrays: list[Array]) -> dict:
    # A sliced view may carry a smaller logical shape with the same backing
    # payload identity. The largest extent wins when summing that identity.
    unique = {}
    for array in arrays:
        if array.key not in unique or array.nbytes > unique[array.key].nbytes:
            unique[array.key] = array
    return {"bytes": sum(a.nbytes for a in unique.values()),
            "objects": len(unique),
            "fields": [{"id": a.key, "shape": list(a.shape),
                        "bytes": a.nbytes} for a in unique.values()]}


def compact_inventory(arrays: list[Array]) -> dict:
    x = inventory(arrays)
    return {"bytes": x["bytes"], "objects": x["objects"]}


def source_fields(m: Maker, tag: str, kind: str, location: str,
                  n: int, s: int, rank: int, residual: int,
                  padded: bool,
                  shared_zero_one: tuple[Array, Array] | None = None) -> dict[str, Array]:
    width = n if kind != "feed_positive" or padded else s
    leg = rank if location == "shared" else n
    residual_width = (residual if kind == "ordinary" else
                      s if kind == "feed_positive" else 0)
    if kind == "feed_positive" and padded:
        loading_shape = (n, s)
    elif kind == "feed_positive":
        loading_shape = (s, s)
    else:
        loading_shape = (n, residual_width)
    fields = {
        "A" if location != "shared" else "FA": m.make(tag + ".A", (leg, width)),
        "P" if location != "shared" else "FP": m.make(tag + ".P", (leg, width)),
        "Z": m.make(tag + ".Z", (n, residual_width)),
        "L": m.make(tag + ".L", loading_shape),
    }
    if kind == "feed_positive":
        zero = m.make(tag + ".zero", (width,))
        one = m.make(tag + ".one", (width,))
        fields.update(s=zero, e=zero, w2=zero, dA=one, dP=one)
    elif kind == "feed_negative":
        zero, one = (shared_zero_one if shared_zero_one is not None else
                     (m.make(tag + ".zero", (n,)),
                      m.make(tag + ".one", (n,))))
        fields.update(s=zero, e=m.make(tag + ".negative_e", (n,)),
                      w2=zero, dA=one, dP=one)
    else:
        fields.update({field: m.make(tag + "." + field, (n,))
                       for field in ("s", "e", "w2", "dA", "dP")})
    return fields


def transport(m: Maker, source: History, n: int, rank: int,
              label: str, location: str, newborn: bool = False) -> History:
    f = dict(source.fields)
    width = f["FA" if location == "shared" else "A"].shape[1]
    if location == "shared":
        # Factor legs and source metadata are carried; only residual Z moves.
        f["Z"] = m.make(label + ".Z", f["Z"].shape)
    else:
        f["A"] = m.make(label + ".A", (n, width))
        f["P"] = m.make(label + ".P", (n, width))
        f["Z"] = m.make(label + ".Z", f["Z"].shape)
    return History(source.kind, 0 if newborn else source.age + 1, f)


def join(m: Maker, source: History, n: int, rank: int,
         label: str) -> History:
    f = dict(source.fields)
    f["FA"] = m.make(label + ".FA", (rank, f.pop("A").shape[1]))
    f["FP"] = m.make(label + ".FP", (rank, f.pop("P").shape[1]))
    return History(source.kind, source.age, f)


def rotate(m: Maker, source: History, rank: int, label: str) -> History:
    f = dict(source.fields)
    f["FA"] = m.make(label + ".FA", f["FA"].shape)
    f["FP"] = m.make(label + ".FP", f["FP"].shape)
    return History(source.kind, source.age, f)


def stage(m: Maker, name: str, permanent: list[Array],
          named: list[tuple[str, tuple[int, ...], int]],
          named_live: dict[str, Array] | None = None,
          named_none: tuple[str, ...] = ()) -> dict:
    # Each named array is a simultaneous operand/result/workspace allowance.
    named_live = {} if named_live is None else named_live
    scratch = [m.make("scratch." + name + "." + label, shape, bpe)
               for label, shape, bpe in named]
    live = inventory(permanent + list(named_live.values()) + scratch)
    seen = set()
    live_names = []
    for label, array in named_live.items():
        live_names.append({"name": label, "array_id": array.key,
                           "shape": list(array.shape), "bytes": array.nbytes,
                           "alias": array.key in seen})
        seen.add(array.key)
    return {"name": name, "bytes": live["bytes"],
            "resident_bytes": compact_inventory(permanent)["bytes"],
            "named_live": live_names,
            "named_none": list(named_none),
            "named_extra": [{"name": label, "shape": list(shape),
                             "bytes": a.nbytes}
                            for (label, shape, _), a in zip(named, scratch)],
            "unique_objects": live["objects"]}


def audit(n: int, depth: int, s: int, rank: int, recent: int,
          residual: int, padded: bool) -> dict:
    if not (n >= 3 and depth >= 3 and 2 <= s <= n and
            1 <= rank <= n and recent >= 1 and 1 <= residual <= n):
        raise ValueError("invalid integrated instrument shape")
    m = Maker()
    external = [m.make(f"external.W{j}.float64", (n, n), 8)
                for j in range(depth)]
    external += [m.make("external.raw.float64", (n, s), 8),
                 m.make("external.probability.float64", (s,), 8)]
    input_inventory = inventory(external)
    probability = m.make("probability.float32", (s,))
    nodes = m.make("nodes.float32", (n, s))
    center = m.make("center.float32", (n,))
    centered = m.make("centered.float32", (n, s))
    fixed = [probability, nodes, center, centered]
    mu = m.make("initial.mu", (n,))
    C = m.make("initial.C", (n, n))
    wick = m.make("initial.wick", (n,))
    V = None
    outputs: list[Array] = []
    young: list[History] = []
    shared: list[History] = []
    newborn: list[History] = []
    Q = gram = None
    previous_locals: dict[str, Array] = {}
    rows = []
    counters = {k: 0 for k in ("ordinary_births", "feed_births", "joins",
                              "join_ordinary", "join_feed_positive",
                              "join_feed_negative", "source_D3_uses",
                              "source_D21_uses", "ordinary_D3_uses",
                              "ordinary_D21_uses", "feed_positive_D3_uses",
                              "feed_positive_D21_uses", "feed_negative_D3_uses",
                              "feed_negative_D21_uses", "k4_uses",
                              "k4_full", "k4_final")}

    def state_arrays(hist: list[History], q: Array | None,
                     g: Array | None) -> list[Array]:
        return [a for h in hist for a in h.arrays()] + [a for a in (q, g)
                                                        if a is not None]

    for layer in range(depth):
        final = layer == depth - 1
        prefix = f"L{layer}"
        old_state = state_arrays(young + shared + newborn, Q, gram)
        old_backbone = [mu, C, wick] + outputs + ([] if V is None else [V])
        old_wick = wick
        W = m.make(prefix + ".W.float32", (n, n))
        WC = m.make(prefix + ".WC", (n, n))
        C_pre = m.make(prefix + ".C_pre", (n, n))
        mu_pre = m.make(prefix + ".mu_pre", (n,))
        var_pre = m.make(prefix + ".var_pre", (n,))
        V = m.make(prefix + ".V", (n, s))
        if layer:
            counters["k4_uses"] += 1
            counters["k4_final" if final else "k4_full"] += 1
        common = external + fixed + old_backbone + old_state
        prior_named = {"prior." + k: v for k, v in previous_locals.items()}
        early_current = {"W": W, "WC": WC, "C_pre": C_pre,
                         "mu_pre": mu_pre, "var_pre": var_pre, "V": V,
                         "nodes": nodes, "center": center,
                         "centered": centered, "probability": probability,
                         "raw": external[-2], "p_in": external[-1],
                         "wick": old_wick, "wick_prev": old_wick}
        early_named = {"current." + k: v for k, v in early_current.items()}
        stages = [stage(m, "cast_and_backbone", common +
                        [W, WC, C_pre, mu_pre, var_pre, V],
                        [("cov_matmul_result", (n, n), 4),
                         ("source_map_operand", (n, s), 4)],
                        named_live=prior_named | early_named,
                        named_none=("sources",) if layer else ())]

        ready = ([] if final else [h for h in young if h.age + 1 >= recent])
        staying = [h for h in young if h not in ready]
        new_shared = shared[:]
        new_Q, new_gram = Q, gram
        for i, source in enumerate(ready):
            tag = f"{prefix}.join{i}"
            joined = join(m, source, n, rank, tag)
            new_shared = [rotate(m, h, rank, tag + ".rotate")
                          for h in new_shared] + [joined]
            new_Q = m.make(tag + ".Q", (n, rank))
            new_gram = m.make(tag + ".gram", (rank, rank))
            counters["joins"] += 1
            counters["join_" + source.kind] += 1
        if ready or shared:
            # Shared Q and every Z are transported after all joins.
            new_Q = m.make(prefix + ".Q_transport", (n, rank))
            new_shared = [transport(m, h, n, rank, prefix + ".shared",
                                    "shared") for h in new_shared]
        next_young = [transport(m, h, n, rank, prefix + ".young", "young")
                      for h in staying]
        next_young += [transport(m, h, n, rank, prefix + ".insert", "young",
                                 newborn=True) for h in newborn]
        young, shared, newborn, Q, gram = next_young, new_shared, [], new_Q, new_gram
        next_state = state_arrays(young + shared, Q, gram)
        if ready or shared:
            stages.append(stage(m, "gram_qr_rotate_transport",
                                common + next_state +
                                [W, WC, C_pre, mu_pre, var_pre, V],
                                [("wick_A", (n, n), 4),
                                 ("wick_P", (n, n), 4),
                                 ("Q_pre", (n, rank), 4),
                                 ("sketch_Y", (n, rank), 4),
                                 ("QR_Q_full", (n, rank), 4),
                                 ("QR_R", (rank, rank), 4),
                                 ("rotation_T", (rank, rank), 4),
                                 ("gram_old_product", (rank, rank), 4),
                                 ("gram_update", (rank, rank), 4),
                                 ("factor_rotation_result", (rank, n), 4),
                                 ("transport_matmul_operand", (n, n), 4)],
                                named_live=prior_named | early_named,
                                named_none=("sources",) if layer else ()))
        stages.append(stage(m, "young_transport", common + next_state +
                            [W, WC, C_pre, mu_pre, var_pre, V],
                            [("wick_A_or_P", (n, n), 4),
                             ("matmul_result", (n, n), 4),
                             ("residual_operand", (n, residual), 4)],
                            named_live=prior_named | early_named,
                            named_none=("sources",) if layer else ()))

        for h in young + shared:
            counters["source_D3_uses"] += 1
            counters[h.kind + "_D3_uses"] += 1
            if not final:
                counters["source_D21_uses"] += 1
                counters[h.kind + "_D21_uses"] += 1
        # The readout first materializes *all* shared A/P legs, then walks rows.
        materialized = []
        for j, h in enumerate(shared):
            width = h.fields["FA"].shape[1]
            materialized += [m.make(f"{prefix}.shared{j}.A", (n, width)),
                             m.make(f"{prefix}.shared{j}.P", (n, width))]
        readout_named = [(name, (n, n), 4) for name in
                         ("D21_total", "D21_piece", "D21_residual",
                          "AP_or_PP", "R_or_MP", "LA_or_LP",
                          "contraction_result", "shared_return")]
        readout_named += [("D21_coords", (n, rank), 4),
                          ("PP_L", (n, residual), 4),
                          ("D3_total", (n,), 4)]
        if final:
            readout_named = [(name, shape, bpe) for name, shape, bpe
                             in readout_named if not name.startswith("D21")
                             and name != "shared_return"]
        stages.append(stage(m, "D3_D21_materialization" if not final else
                            "final_D3_only", common + next_state + materialized +
                            [W, WC, C_pre, mu_pre, var_pre, V], readout_named,
                            named_live=prior_named | early_named,
                            named_none=("sources",) if layer else ()))

        mu = m.make(prefix + ".mu", (n,))
        outputs.append(mu)  # Same object: output and current mean are aliases.
        derivative_names = ("safe_var", "sigma", "alpha", "phi64", "phi",
                            "Phi64", "Phi", "w2", "w3", "w4", "base",
                            "D3", "D4", "mean3", "mean4")
        derivatives = [m.make(prefix + "." + name, (n,),
                              8 if name.endswith("64") else 4)
                       for name in derivative_names]
        current_locals = {"W": W, "WC": WC, "C_pre": C_pre,
                          "mu_pre": mu_pre, "var_pre": var_pre, "V": V,
                          "mu": mu, "D3": derivatives[11],
                          "nodes": nodes, "center": center,
                          "centered": centered, "probability": probability,
                          "raw": external[-2], "p_in": external[-1],
                          "wick": old_wick, "wick_prev": old_wick,
                          "w1": derivatives[6], "Phi": derivatives[6],
                          "w2now": derivatives[7], "w3": derivatives[8],
                          "w4": derivatives[9], "base": derivatives[10]}
        current_locals.update(zip(derivative_names, derivatives))
        current_locals["C_off"] = m.make(prefix + ".local.C_off", (n, n))
        current_locals["D21"] = (m.make(prefix + ".local.D21", (n, n))
                                if not final else None)
        if layer:
            current_locals["V2"] = m.make(prefix + ".local.V2", (n, s))
            current_locals["c"] = m.make(prefix + ".local.c", (n,))
            current_locals["positive4"] = m.make(
                prefix + ".local.positive4", (n,))
        if not final:
            current_locals["D22"] = m.make(prefix + ".local.D22", (n, n))
            current_locals["D31"] = m.make(prefix + ".local.D31", (n, n))
            current_locals["K11"] = m.make(prefix + ".local.K11", (n, n))
            current_locals["off_mask"] = m.make(
                prefix + ".local.off_mask", (n, n))
            current_locals["e2"] = m.make(prefix + ".local.e2", (n,))
            current_locals["var_out"] = m.make(prefix + ".local.var_out", (n,))
            current_locals["e3"] = m.make(prefix + ".local.e3", (n,))
            current_locals["k3v"] = m.make(prefix + ".local.k3v", (n,))
            if layer:
                for name, shape in (("bank_p", (n, s)),
                                    ("bank_covariance", (n, n)),
                                    ("V2p", (n, s)),
                                    ("positive22", (n, n)),
                                    ("V3p", (n, s)),
                                    ("positive31", (n, n))):
                    current_locals[name] = m.make(
                        prefix + ".local." + name, shape)
        covariance_fields = [("K11_operand_1", (n, n), 4),
                             ("K11_operand_2", (n, n), 4),
                             ("K4_Gram_operand", (n, s), 4)]
        if final:
            covariance_fields = [("final_V2_operand", (n, s), 4)]
        else:
            # The assembled C exists at the post-C boundary. Keep the older
            # C in common too as an explicitly inclusive lifetime allowance.
            C_next = m.make(prefix + ".C", (n, n))
            current_locals["C"] = C_next
        final_none = ("bank_covariance", "D22", "D31", "D21")
        prior_after_none = {k: v for k, v in previous_locals.items()
                            if not final or k not in final_none}
        stages.append(stage(m, "nonlinear_C_W_K11" if not final else
                            "final_mean_shortcut", common + next_state +
                            [W, WC, C_pre, mu_pre, var_pre, V, mu] +
                            ([] if final else [C_next]) +
                            derivatives + outputs, covariance_fields,
                            named_live={"prior." + k: v for k, v in
                                        prior_after_none.items()} |
                                       {"current." + k: v for k, v in
                                        current_locals.items() if v is not None},
                            named_none=final_none + ("sources",) if final
                            else ("sources",)))
        if final:
            row_newborn = []
        else:
            C = C_next
            wick = derivatives[6]  # Phi = w1, an object alias.
            current_locals["wick_prev"] = wick
            ordinary = History("ordinary", 0, source_fields(
                m, prefix + ".birth.ordinary", "ordinary", "young",
                n, s, rank, residual, padded))
            newborn = [ordinary]
            counters["ordinary_births"] += 1
            if layer > 0:
                positive = History("feed_positive", 0, source_fields(
                    m, prefix + ".birth.feed_positive", "feed_positive",
                    "young", n, s, rank, residual, padded))
                shared_constants = ((positive.fields["s"], positive.fields["dA"])
                                    if padded else None)
                negative = History("feed_negative", 0, source_fields(
                    m, prefix + ".birth.feed_negative", "feed_negative",
                    "young", n, s, rank, residual, padded,
                    shared_zero_one=shared_constants))
                newborn.extend((positive, negative))
                counters["feed_births"] += 1
            row_newborn = [h.kind for h in newborn]
            # Names below are actual outer-frame bindings, rather than a
            # single anonymous scratch maximum. Ordinary a/Rr/Qr/identity
            # alias the newborn Source fields; source vectors alias likewise.
            ordinary_fields = ordinary.fields
            current_locals.update({"a": ordinary_fields["A"],
                                   "Rr": ordinary_fields["Z"],
                                   "Qr": ordinary_fields["L"],
                                   "identity": ordinary_fields["P"],
                                   "s0": ordinary_fields["s"],
                                   "e0": ordinary_fields["e"],
                                   "dA": ordinary_fields["dA"],
                                   "dP": ordinary_fields["dP"]})
            for name in ("local21", "old21", "repeated", "K21",
                         "separable", "Rres"):
                current_locals[name] = m.make(
                    prefix + ".local." + name, (n, n))
            current_locals["Y"] = m.make(prefix + ".local.Y", (n, residual))
            current_locals["Bm"] = m.make(prefix + ".local.Bm", (residual, n))
            # Om is W[:, :ordinary_residual]: a distinct view object whose
            # backing W payload is already charged. Keep its logical shape.
            current_locals["Om"] = Array(W.key, (n, residual),
                                          backing_shape=(n, n))
            if layer:
                positive, negative = newborn[1:]
                current_locals.update({
                    "Pbase": (positive.fields["P"] if not padded else
                              m.make(prefix + ".feed_helper.Pbase", (n, s))),
                    "Aplus": m.make(prefix + ".local.Aplus", (n, s)),
                    "Zplus": positive.fields["Z"],
                    "zero_A": positive.fields["A"],
                    "negative_A": negative.fields["A"],
                    "negative_P": negative.fields["P"],
                    "negative_e": negative.fields["e"],
                    "selector": positive.fields["L"]})
            birth_named = [("source_expression_operand", (n, n), 4),
                           ("K11_operand", (n, n), 4)]
            birth_named += [("residual_sketch", (n, residual), 4),
                            ("residual_QR_Q", (n, residual), 4),
                            ("residual_QR_R", (residual, residual), 4),
                            ("residual_project", (residual, n), 4),
                            ("feed_Pbase", (n, s), 4),
                            ("feed_Aplus", (n, s), 4),
                            ("feed_Zplus", (n, s), 4)]
            if padded:
                birth_named += [("positive_pad", (n, n - s), 4),
                                ("selector_eye_base", (n, n), 4)]
            stages.append(stage(m, "ordinary_and_feed_birth",
                                common + next_state +
                                state_arrays(newborn, None, None) +
                                [W, WC, C_pre, mu_pre, var_pre, V, mu, C] +
                                derivatives + outputs, birth_named,
                                named_live={"current." + k: v for k, v in
                                            current_locals.items() if v is not None},
                                named_none=("sources",)))

        # The integrated function keeps many of these locals bound after the
        # loop body. At the next readout they still have their previous-layer
        # identity. Including any already-rebound value in a later overlapping
        # stage is deliberately conservative and may overcount actual liveness.
        current_bound = {k: v for k, v in current_locals.items()
                         if v is not None}
        previous_locals = (
            {k: v for k, v in previous_locals.items()
             if k not in current_locals and k not in final_none} | current_bound
            if final else current_bound)

        current_partial = ([V, probability, mu, C] + outputs +
                           state_arrays(young + shared + newborn, Q, gram))
        resident = (external + fixed + [mu, C, wick, V] + outputs +
                    state_arrays(young + shared + newborn, Q, gram))
        kinds = {kind: [h for h in young + shared + newborn if h.kind == kind]
                 for kind in ("ordinary", "feed_positive", "feed_negative")}
        retained_fields = []
        for location, members in (("young", young), ("shared", shared),
                                  ("newborn", newborn)):
            for h in members:
                for field, array in h.fields.items():
                    retained_fields.append({"location": location,
                                            "kind": h.kind, "age": h.age,
                                            "field": field, "array_id": array.key,
                                            "shape": list(array.shape),
                                            "bytes": array.nbytes})
        rows.append({"layer": layer, "final": final, "young": len(young),
                     "shared": len(shared), "newborn": row_newborn,
                     "ages": [{"kind": h.kind, "age": h.age,
                               "location": "young" if h in young else "shared"}
                              for h in young + shared],
                     "by_kind_bytes": {kind: compact_inventory(
                         [a for h in members for a in h.arrays()])["bytes"]
                         for kind, members in kinds.items()},
                     "retained_fields": retained_fields,
                     "partial_resident_floor_bytes":
                         compact_inventory(current_partial)["bytes"],
                     "explicit_resident_bytes": compact_inventory(resident)["bytes"],
                     "stages": stages,
                     "peak_modeled_array_payload_bytes": max(x["bytes"]
                                                              for x in stages)})
    output_stack = m.make("output.stack", (depth, n))
    final_arrays = (external + fixed + [mu, C, wick, V] + outputs +
                    state_arrays(young + shared, Q, gram))
    rows[-1]["stages"].append(stage(
        m, "output_stack", final_arrays,
        [("stack_result", (depth, n), 4)],
        named_live={"frame." + k: v for k, v in previous_locals.items()},
        named_none=("bank_covariance", "D22", "D31", "D21", "sources")))
    rows[-1]["peak_modeled_array_payload_bytes"] = max(
        x["bytes"] for x in rows[-1]["stages"])
    maximum = max(m.all_arrays, key=lambda a: a.nbytes)
    return {"variant": "old_padded" if padded else "optimized_skinny",
            "shape": {"width": n, "depth": depth, "source_rank": s,
                      "common_rank": rank, "recent": recent,
                      "ordinary_residual": residual},
            "external_input": input_inventory,
            "schedule": counters, "layers": rows,
            "max_resident_array_bytes": max(r["explicit_resident_bytes"]
                                            for r in rows),
            "max_modeled_array_payload_bytes": max(
                r["peak_modeled_array_payload_bytes"] for r in rows),
            "largest_single_array": {"id": maximum.key,
                                     "shape": list(maximum.shape),
                                     "bytes": maximum.nbytes},
            "every_modeled_array_below_4GiB": all(a.nbytes < ARRAY_LIMIT
                                                  for a in m.all_arrays),
            "final_output_shape": list(output_stack.shape)}


def tiny_runtime_snapshot(n: int, source_rank: int, rank: int,
                          ordinary_residual: int) -> dict:
    """Inspect actual tiny outer-frame bindings at each sampled boundary.

    Only names/shapes/within-snapshot aliases are saved. The runtime never
    exposes protected truth and never constructs a full-width array.
    """
    spec = importlib.util.spec_from_file_location(
        "p2_k4_integrated_memory_runtime_reference", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen tiny reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines, 1)
                 if line.startswith("def integrated_metered_program("))

    def find_after(needle: str, after: int = start) -> int:
        return next(i for i, line in enumerate(lines, 1)
                    if i > after and needle in line)

    cast_line = find_after("if layer == 0:")
    join_line = find_after("shared, staying = _integrated_join_ready(")
    young_line = find_after("young = [_integrated_transport_source(")
    readout_line = find_after("D3, D21, sources = _integrated_readout(")
    cov_assemble_line = find_after('C = meter.run("backbone_cov_assemble"')
    post_c_line = find_after("if capture_tiny:", cov_assemble_line)
    output_append_line = find_after("outputs.append(mu)")
    final_mean_line = find_after("if final:", output_append_line)
    output_line = find_after('prediction = meter.run("output_stack"')
    birth_lines = [i for i, line in enumerate(lines, 1)
                   if i > start and 'layer_rows.append({"layer": layer' in line]
    if len(birth_lines) != 2:
        raise AssertionError("expected final and nonfinal row publication sites")
    sites = {(cast_line, 14): "L14_cast_after_backbone",
             (join_line, 14): "L14_join_before_call",
             (young_line, 14): "L14_young_transport_before_call",
             (readout_line, 14): "L14_readout_before_call",
             (post_c_line, 14): "L14_post_C",
             (birth_lines[-1], 14): "L14_birth_after_construction",
             (readout_line, 15): "L15_final_readout_before_call",
             (final_mean_line, 15): "L15_final_mean",
             (output_line, 15): "L15_output_stack"}
    snapshots = {}

    def tracer(frame, event, _arg):
        if frame.f_code is module.integrated_metered_program.__code__:
            key = (frame.f_lineno, frame.f_locals.get("layer"))
            if event == "line" and key in sites:
                identities: dict[int, str] = {}
                fields = {}
                none = []
                for name, value in sorted(frame.f_locals.items()):
                    if value is None:
                        none.append(name)
                    shape = getattr(value, "shape", None)
                    if shape is None:
                        continue
                    try:
                        dimensions = [int(axis) for axis in shape]
                    except (TypeError, ValueError):
                        continue
                    identity = id(value)
                    alias_of = identities.get(identity)
                    if alias_of is None:
                        identities[identity] = name
                    fields[name] = {"shape": dimensions,
                                    "alias_of": alias_of}
                nested = {}
                shared = frame.f_locals.get("shared")
                if shared is not None:
                    nested["shared.Q"] = list(shared.Q.shape)
                    nested["shared.gram"] = list(shared.gram.shape)
                    if shared.factors:
                        factor = shared.factors[-1]
                        nested["shared.last.FA"] = list(factor.FA.shape)
                        nested["shared.last.FP"] = list(factor.FP.shape)
                        nested["shared.last.Z"] = list(factor.Z.shape)
                newborn = frame.f_locals.get("newborn")
                if newborn:
                    ordinary = newborn[0]
                    nested["newborn.ordinary.Z"] = list(ordinary.Z.shape)
                    nested["newborn.ordinary.L"] = list(ordinary.L.shape)
                snapshots[sites[key]] = {"arrays": fields, "none": none,
                                         "nested": nested,
                                         "line": frame.f_lineno}
            return tracer
        return None

    previous_trace = sys.gettrace()
    try:
        sys.settrace(tracer)
        module.integrated_metered_program(
            n=n, depth=16, source_rank=source_rank, rank=rank, recent=4,
            ordinary_residual=ordinary_residual, capture_tiny=False)
    finally:
        sys.settrace(previous_trace)
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    stored = json.loads(SHORTCUT.read_text(encoding="utf-8"))
    optimized = audit(1024, 16, 64, 384, 4, 16, False)
    old = audit(1024, 16, 64, 384, 4, 16, True)
    tiny = audit(8, 16, 6, 8, 4, 8, False)
    tiny_old = audit(8, 16, 6, 8, 4, 8, True)
    thin = audit(16, 16, 8, 8, 4, 2, False)
    thin_old = audit(16, 16, 8, 8, 4, 2, True)
    checks = []

    def check(name: str, ok: bool, **details: object) -> None:
        checks.append({"name": name, "pass": bool(ok), **details})

    check("frozen_optimized_source_sha", sha == FROZEN_SHA, got=sha,
          expected=FROZEN_SHA)
    check("shortcut_source_binding", stored["optimized_source_sha256"] == sha)
    first_sha = hashlib.sha256(FIRST_RESULT.read_bytes()).hexdigest()
    check("first_incomplete_result_unchanged", first_sha == FIRST_RESULT_SHA,
          got=first_sha, expected=FIRST_RESULT_SHA)
    colive_sha = hashlib.sha256(COLIVE_RESULT.read_bytes()).hexdigest()
    check("second_incomplete_result_unchanged", colive_sha == COLIVE_RESULT_SHA,
          got=colive_sha, expected=COLIVE_RESULT_SHA)
    accepted_n8_sha = hashlib.sha256(BOUNDARY_RESULT.read_bytes()).hexdigest()
    check("accepted_n8_result_unchanged",
          accepted_n8_sha == BOUNDARY_RESULT_SHA,
          got=accepted_n8_sha, expected=BOUNDARY_RESULT_SHA)
    accepted_n8 = json.loads(BOUNDARY_RESULT.read_text(encoding="utf-8"))
    check("accepted_n8_source_binding_unchanged",
          accepted_n8["source_sha256"] == BOUNDARY_SOURCE_SHA and
          accepted_n8["frozen_integrated_source_sha256"] == FROZEN_SHA)
    check("shortcut_existing_gate_remains_red", stored["pass"] and
          stored["n8_L3_min_prevariance_new"] < 0,
          existing_n8_min_prevariance=stored["n8_L3_min_prevariance_new"])
    for label, case in (("optimized", optimized), ("old", old),
                        ("tiny", tiny), ("tiny_old", tiny_old),
                        ("thin", thin), ("thin_old", thin_old)):
        check(label + "_schedule", case["schedule"] == stored["new_schedule"])
        check(label + "_single_array_cap", case["every_modeled_array_below_4GiB"])
        check(label + "_final_shortcut", not case["layers"][-1]["final"] is False
              and not any(x["name"] == "ordinary_and_feed_birth" for x in
                          case["layers"][-1]["stages"]))
        source_width = case["shape"]["source_rank"]
        width = case["shape"]["width"]
        positive_width = (width if case["variant"] == "old_padded"
                          else source_width)
        wrong_legs = [
            (row["layer"], f["kind"], f["field"], f["shape"])
            for row in case["layers"] for f in row["retained_fields"]
            if f["field"] in ("A", "P", "FA", "FP") and
            f["shape"][1] != (positive_width if f["kind"] == "feed_positive"
                              else width)]
        check(label + "_positive_S_or_padded_negative_n_ordinary_n_legs",
              not wrong_legs, wrong_legs=wrong_legs[:8])
    # The source's partial function excludes external inputs, W/WC/C_pre and
    # derivatives. We reproduce those included fields from shapes and aliases.
    check("tiny_partial_resident_floor_crosscheck",
          max(r["partial_resident_floor_bytes"] for r in tiny["layers"])
          == stored["new_peak_resident_bytes"],
          ledger=max(r["partial_resident_floor_bytes"] for r in tiny["layers"]),
          committed=stored["new_peak_resident_bytes"])
    check("tiny_old_partial_resident_floor_crosscheck",
          max(r["partial_resident_floor_bytes"] for r in tiny_old["layers"])
          == stored["old_peak_resident_bytes"],
          ledger=max(r["partial_resident_floor_bytes"] for r in tiny_old["layers"]),
          committed=stored["old_peak_resident_bytes"])
    saved_raw_ap = 14 * 2 * 4 * 1024 * (1024 - 64)
    negative_raw_ap = 14 * 2 * 4 * 1024 * 1024
    check("positive_14_birth_raw_AP_saves_at_least_100MiB",
          saved_raw_ap >= 100 * 1024 ** 2, bytes=saved_raw_ap)
    check("negative_14_birth_raw_AP_is_112MiB",
          negative_raw_ap == 112 * 1024 ** 2, bytes=negative_raw_ap)
    check("skinny_peak_below_padded_peak",
          optimized["max_modeled_array_payload_bytes"] <
          old["max_modeled_array_payload_bytes"])
    def stage_at(case: dict, layer: int, name: str) -> dict:
        return next(s for s in case["layers"][layer]["stages"]
                    if s["name"] == name)

    for label, case in (("optimized", optimized), ("old_padded", old),
                        ("tiny_optimized", tiny), ("tiny_old", tiny_old)):
        width = case["shape"]["width"]
        readout = stage_at(case, 14, "D3_D21_materialization")
        birth = stage_at(case, 14, "ordinary_and_feed_birth")
        for site, names, prefix in (("prior_readout", PRIOR_14, "prior."),
                                    ("current_birth", CURRENT_BIRTH_9,
                                     "current.")):
            fields = readout["named_live"] if site == "prior_readout" else birth["named_live"]
            located = {row["name"]: row for row in fields}
            selected = [located.get(prefix + name) for name in names]
            check(label + "_L14_" + site + "_named_distinct_matrices",
                  all(row is not None and row["shape"] == [width, width]
                      for row in selected) and
                  len({row["array_id"] for row in selected if row is not None})
                  == len(names),
                  expected_names=[prefix + name for name in names],
                  present_names=[row["name"] for row in selected if row is not None])
    check("repaired_inclusive_peak_exceeds_rejected_peak",
          optimized["max_modeled_array_payload_bytes"] > 638803200,
          rejected_peak=638803200,
          revised_peak=optimized["max_modeled_array_payload_bytes"])
    try:
        runtime = tiny_runtime_snapshot(8, 6, 8, 8)
    except Exception as exc:
        runtime = {"error": repr(exc)}
    try:
        runtime_thin = tiny_runtime_snapshot(16, 8, 8, 2)
    except Exception as exc:
        runtime_thin = {"error": repr(exc)}
    for site, names in (("L14_readout_before_call", PRIOR_14),
                        ("L14_birth_after_construction", CURRENT_BIRTH_9)):
        actual = runtime.get(site, {}).get("arrays", {})
        check("tiny_runtime_" + site + "_distinct_matrices",
              all(name in actual and actual[name]["shape"] == [8, 8] and
                  actual[name]["alias_of"] is None for name in names),
              expected_names=list(names),
              present_names=[name for name in names if name in actual])
        modeled_stage = stage_at(
            tiny, 14, "D3_D21_materialization" if "readout" in site else
            "ordinary_and_feed_birth")
        prefix = "prior." if "readout" in site else "current."
        modeled = {row["name"][len(prefix):]: row for row in
                   modeled_stage["named_live"] if row["name"].startswith(prefix)}
        missing_or_wrong = [name for name, field in actual.items()
                            if name not in modeled or
                            field["shape"] != modeled[name]["shape"]]
        check("tiny_runtime_" + site + "_full_outer_array_shape_coverage",
              bool(actual) and not missing_or_wrong,
              missing_or_wrong=missing_or_wrong,
              actual_array_bindings=len(actual),
              modeled_array_bindings=len(modeled))
    boundaries = {
        "L14_cast_after_backbone": (14, "cast_and_backbone"),
        "L14_join_before_call": (14, "gram_qr_rotate_transport"),
        "L14_young_transport_before_call": (14, "young_transport"),
        "L14_readout_before_call": (14, "D3_D21_materialization"),
        "L14_post_C": (14, "nonlinear_C_W_K11"),
        "L14_birth_after_construction": (14, "ordinary_and_feed_birth"),
        "L15_final_readout_before_call": (15, "final_D3_only"),
        "L15_final_mean": (15, "final_mean_shortcut"),
        "L15_output_stack": (15, "output_stack"),
    }
    for shape_label, model, snapshots in (("n8", tiny, runtime),
                                          ("n16_thin", thin, runtime_thin)):
        for site, (layer, stage_name) in boundaries.items():
            snapshot = snapshots.get(site, {})
            actual = snapshot.get("arrays", {})
            actual_none = set(snapshot.get("none", []))
            modeled_stage = stage_at(model, layer, stage_name)
            modeled = {}
            for row in modeled_stage["named_live"]:
                name = row["name"].split(".", 1)[1]
                modeled[name] = row
            missing_or_wrong = [name for name, field in actual.items()
                                if name not in modeled or
                                field["shape"] != modeled[name]["shape"]]
            false_alias = [name for name, field in actual.items()
                           if field["alias_of"] is not None and
                           name in modeled and field["alias_of"] in modeled and
                           modeled[name]["array_id"] !=
                           modeled[field["alias_of"]]["array_id"]]
            modeled_groups = {}
            for name in actual:
                if name in modeled:
                    modeled_groups.setdefault(modeled[name]["array_id"], []).append(name)
            false_modeled_alias = []
            for names in modeled_groups.values():
                if len(names) < 2:
                    continue
                actual_identities = {actual[name]["alias_of"] or name
                                     for name in names}
                if len(actual_identities) > 1 and set(names) != {"Om", "W"}:
                    false_modeled_alias.append(names)
            check(shape_label + "_boundary_" + site + "_outer_array_coverage",
                  bool(actual) and not missing_or_wrong and not false_alias and
                  not false_modeled_alias,
                  actual_array_bindings=len(actual),
                  missing_or_wrong=missing_or_wrong, false_alias=false_alias,
                  false_modeled_alias=false_modeled_alias,
                  allowed_backing_exception="Om is a distinct W column-view object whose retained backing payload is W")
            check(shape_label + "_boundary_" + site + "_exact_None_bindings",
                  bool(actual) and actual_none == set(modeled_stage["named_none"]),
                  actual_none=sorted(actual_none),
                  modeled_none=sorted(modeled_stage["named_none"]))
    output_actual = runtime.get("L15_output_stack", {}).get("arrays", {})
    output_none = runtime.get("L15_output_stack", {}).get("none", [])
    output_stage = stage_at(tiny, 15, "output_stack")
    fourteen = ("W", "WC", "C_pre", "C_off", "positive22",
                "positive31", "K11", "off_mask", "local21", "old21",
                "repeated", "K21", "separable", "Rres")
    output_modeled = {row["name"].removeprefix("frame."): row
                      for row in output_stage["named_live"]}
    check("L15_output_fourteen_distinct_matrices",
          all(name in output_actual and output_actual[name]["shape"] == [8, 8]
              and name in output_modeled and
              output_modeled[name]["shape"] == [8, 8]
              for name in fourteen) and
          len({output_modeled[name]["array_id"] for name in fourteen
               if name in output_modeled}) == 14,
          expected_names=list(fourteen))
    check("L15_output_none_bindings_explicit",
          set(output_stage["named_none"]) ==
          {"bank_covariance", "D22", "D31", "D21", "sources"} and
          set(output_stage["named_none"]) == set(output_none),
          modeled_none=output_stage["named_none"],
          runtime_none=output_none)
    full_output_stage = stage_at(optimized, 15, "output_stack")
    full_om = next((row for row in full_output_stage["named_live"]
                    if row["name"] == "frame.Om"), None)
    check("L15_Om_view_keeps_prior_W_backing",
          full_om is not None and full_om["shape"] == [1024, 16] and
          full_om["bytes"] == 4 * 1024 * 1024)
    thin_readout_nested = runtime_thin.get("L14_readout_before_call", {}).get(
        "nested", {})
    thin_birth_nested = runtime_thin.get("L14_birth_after_construction", {}).get(
        "nested", {})
    thin_birth_arrays = runtime_thin.get("L14_birth_after_construction", {}).get(
        "arrays", {})
    check("n16_thin_common_rank8_observed",
          thin_readout_nested.get("shared.Q") == [16, 8] and
          thin_readout_nested.get("shared.gram") == [8, 8] and
          thin_readout_nested.get("shared.last.FA", [None])[0] == 8 and
          thin_readout_nested.get("shared.last.FP", [None])[0] == 8,
          observed=thin_readout_nested)
    check("n16_thin_ordinary_residual2_observed",
          thin_birth_nested.get("newborn.ordinary.Z") == [16, 2] and
          thin_birth_nested.get("newborn.ordinary.L") == [16, 2] and
          thin_birth_arrays.get("Om", {}).get("shape") == [16, 2] and
          thin_birth_arrays.get("Qr", {}).get("shape") == [16, 2] and
          thin_birth_arrays.get("Bm", {}).get("shape") == [2, 16] and
          thin_birth_arrays.get("Y", {}).get("shape") == [16, 2],
          observed_nested=thin_birth_nested)
    for stage_name in ("gram_qr_rotate_transport", "young_transport"):
        s14 = stage_at(optimized, 14, stage_name)
        names = {row["name"] for row in s14["named_live"]}
        check(stage_name + "_has_current_mu_pre_var_pre",
              {"current.mu_pre", "current.var_pre"}.issubset(names))
    cov_names = {row["name"] for row in
                 stage_at(optimized, 14, "nonlinear_C_W_K11")["named_live"]}
    check("L14_post_C_has_current_C", "current.C" in cov_names)
    result = {"kind": "shape-only integrated first-source array memory audit",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_integrated_source_sha256": sha,
              "committed_shortcut_result_sha256": hashlib.sha256(
                  SHORTCUT.read_bytes()).hexdigest(),
              "first_incomplete_result_sha256": first_sha,
              "second_incomplete_result_sha256": colive_sha,
              "accepted_n8_result_sha256": accepted_n8_sha,
              "tiny_runtime_outer_frame": runtime,
              "thin_runtime_outer_frame": runtime_thin,
              "checks": checks, "pass": all(x["pass"] for x in checks),
              "optimized": optimized, "old_padded_control": old,
              "tiny_optimized_crosscheck": tiny,
              "tiny_old_padded_crosscheck": tiny_old,
              "thin_optimized_crosscheck": thin,
              "thin_old_padded_crosscheck": thin_old,
              "raw_AP_control": {"positive_14_birth_saving_bytes": saved_raw_ap,
                                 "negative_14_birth_bytes": negative_raw_ap},
              "limits": {"individual_array_strict_bytes": ARRAY_LIMIT,
                         "process_cap_reference_bytes": PROCESS_LIMIT,
                         "process_cap_gate": "NOT_TESTED"},
              "scope": "Finite sampled n8 and n16 thin-shape named-boundary inventory: explicit float64 inputs, float32 retained Source fields, prior/current outer-frame bindings and named co-live arrays. Om is a view whose backing W payload is charged. The inclusive stages intentionally retain some prior values after rebinding and may overcount actual liveness. This is no rigorous whole-process upper bound: library allocator, flopscope wrappers, BLAS/QR workspaces, fragmentation and Python process overhead are unbounded by this model. Windows does not enforce the 8 GiB cap. Existing n8 variance and n4 numeric gates remain FAIL; this is no stability, cost, accuracy, or deployment claim."}
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("memory ledger %s; optimized modeled peak %d bytes; padded %d bytes" %
          ("PASS" if result["pass"] else "FAIL",
           optimized["max_modeled_array_payload_bytes"],
           old["max_modeled_array_payload_bytes"]))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
