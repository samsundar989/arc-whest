"""Deterministic array-shape audit for the integrated first-source instrument.

No challenge networks, large arrays, flopscope context, or labels are loaded.
The schedule follows p2_k4_integrated_first_source.py at its frozen SHA. This is
an explicit array-payload and named co-live allowance, not a process peak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "local_sim/p2_k4_integrated_first_source.py"
SHORTCUT = (ROOT / "experiments/2026-09-11_stage2_compact_p07/"
            "integrated_first_source/shortcut_result.json")
FROZEN_SHA = "3a57e5e5709a62848f8412de094f218ea9b323933b9071e04c5dabbdab8b12b3"
ARRAY_LIMIT = 4 * 1024 ** 3
PROCESS_LIMIT = 8 * 1024 ** 3


@dataclass(frozen=True)
class Array:
    key: str
    shape: tuple[int, ...]
    bytes_per_element: int = 4

    @property
    def nbytes(self) -> int:
        count = 1
        for axis in self.shape:
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
    unique = {array.key: array for array in arrays}
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
          named: list[tuple[str, tuple[int, ...], int]]) -> dict:
    # Each named array is a simultaneous operand/result/workspace allowance.
    scratch = [m.make("scratch." + name + "." + label, shape, bpe)
               for label, shape, bpe in named]
    live = inventory(permanent + scratch)
    return {"name": name, "bytes": live["bytes"],
            "resident_bytes": compact_inventory(permanent)["bytes"],
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
        stages = [stage(m, "cast_and_backbone", common +
                        [W, WC, C_pre, mu_pre, var_pre, V],
                        [("cov_matmul_result", (n, n), 4),
                         ("source_map_operand", (n, s), 4)])]

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
                                common + next_state + [W, WC, C_pre, V],
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
                                 ("transport_matmul_operand", (n, n), 4)]))
        stages.append(stage(m, "young_transport", common + next_state +
                            [W, WC, C_pre, V],
                            [("wick_A_or_P", (n, n), 4),
                             ("matmul_result", (n, n), 4),
                             ("residual_operand", (n, residual), 4)]))

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
                            [W, WC, C_pre, mu_pre, var_pre, V], readout_named))

        mu = m.make(prefix + ".mu", (n,))
        outputs.append(mu)  # Same object: output and current mean are aliases.
        derivative_names = ("safe_var", "sigma", "alpha", "phi64", "phi",
                            "Phi64", "Phi", "w2", "w3", "w4", "base",
                            "D3", "D4", "mean3", "mean4")
        derivatives = [m.make(prefix + "." + name, (n,),
                              8 if name.endswith("64") else 4)
                       for name in derivative_names]
        covariance_fields = [("C_off", (n, n), 4), ("bank_C", (n, n), 4),
                             ("D22", (n, n), 4), ("D31", (n, n), 4),
                             ("K11", (n, n), 4), ("off_mask", (n, n), 4),
                             ("C_new", (n, n), 4),
                             ("K11_operand_1", (n, n), 4),
                             ("K11_operand_2", (n, n), 4),
                             ("K4_Gram_operand", (n, s), 4)]
        if final:
            covariance_fields = [("C_off", (n, n), 4),
                                 ("V_squared", (n, s), 4),
                                 ("final_D4_vector", (n,), 4)]
        stages.append(stage(m, "nonlinear_C_W_K11" if not final else
                            "final_mean_shortcut", common + next_state +
                            [W, WC, C_pre, mu_pre, var_pre, V, mu] +
                            derivatives + outputs, covariance_fields))
        if final:
            row_newborn = []
        else:
            C = m.make(prefix + ".C", (n, n))
            wick = derivatives[6]  # Phi = w1, an object alias.
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
            birth_named = [(name, (n, n), 4) for name in
                           ("source_a", "local21", "old21", "repeated21",
                            "K21", "separable21", "Rres", "K11_operand")]
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
                                derivatives + outputs, birth_named))

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
    rows[-1]["stages"].append(stage(m, "output_stack", final_arrays,
                                      [("stack_result", (depth, n), 4)]))
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
    checks = []

    def check(name: str, ok: bool, **details: object) -> None:
        checks.append({"name": name, "pass": bool(ok), **details})

    check("frozen_optimized_source_sha", sha == FROZEN_SHA, got=sha,
          expected=FROZEN_SHA)
    check("shortcut_source_binding", stored["optimized_source_sha256"] == sha)
    check("shortcut_existing_gate_remains_red", stored["pass"] and
          stored["n8_L3_min_prevariance_new"] < 0,
          existing_n8_min_prevariance=stored["n8_L3_min_prevariance_new"])
    for label, case in (("optimized", optimized), ("old", old),
                        ("tiny", tiny), ("tiny_old", tiny_old)):
        check(label + "_schedule", case["schedule"] == stored["new_schedule"])
        check(label + "_single_array_cap", case["every_modeled_array_below_4GiB"])
        check(label + "_final_shortcut", not case["layers"][-1]["final"] is False
              and not any(x["name"] == "ordinary_and_feed_birth" for x in
                          case["layers"][-1]["stages"]))
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
    result = {"kind": "shape-only integrated first-source array memory audit",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_integrated_source_sha256": sha,
              "committed_shortcut_result_sha256": hashlib.sha256(
                  SHORTCUT.read_bytes()).hexdigest(),
              "checks": checks, "pass": all(x["pass"] for x in checks),
              "optimized": optimized, "old_padded_control": old,
              "tiny_optimized_crosscheck": tiny,
              "tiny_old_padded_crosscheck": tiny_old,
              "raw_AP_control": {"positive_14_birth_saving_bytes": saved_raw_ap,
                                 "negative_14_birth_bytes": negative_raw_ap},
              "limits": {"individual_array_strict_bytes": ARRAY_LIMIT,
                         "process_cap_reference_bytes": PROCESS_LIMIT,
                         "process_cap_gate": "NOT_TESTED"},
              "scope": "Explicit float64 inputs, float32 retained field identities and named co-live arrays. Library allocator, flopscope wrapper, BLAS/QR workspaces, fragmentation, Python process overhead and true peak cannot be bounded here. Windows does not enforce the 8 GiB cap. Existing n8 variance and n4 numeric gates remain FAIL; this is no stability, cost, accuracy, or deployment claim."}
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
