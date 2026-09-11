"""Tiny label-free checks for the compact four-arm reference.

These tests execute the metered compact program itself.  They use no research
labels, seed blocks, public data, or protected truth.
"""

from __future__ import annotations

import argparse
import json

from p2_compact_reference import metered_program


SHAPE = (8, 4, 2, 8, 2)
SEED = 123
DEFAULT_FLOPS = 68912
DEFAULT_OPERATIONS = 963
DEFAULT_OUTPUT_SHA256 = (
    "bc44e7ef07f9749cde9d7dd68313e7e1b6049606fb818368f1e02a2573b47ced"
)


def _run(consumers, births, lam=1.0 / 3.0, *, depth=4):
    n, _depth, recent, rank, residual_rank = SHAPE
    return metered_program(n, depth, recent, rank, residual_rank, seed=SEED,
                           consumers=consumers, births=births, lam=lam)


def _parity_ok(result):
    p = result["parity"]
    return (p["exact"] and p["tiny_dense_oracle_executed"]
            and p["tiny_dense_D3_max_relative"] <= 2e-5
            and p["tiny_dense_D21_max_relative"] <= 2e-5
            and max(p["tiny_dense_closure_max_relative"].values()) <= 3e-5
            and p["tiny_dense_shadow_max_relative"] <= 3e-5
            and p["tiny_dense_shadow_ages_exact"])


def run():
    rows = []

    coherent = _run(True, True)
    rows.append({
        "check": "default-coherent-regression",
        "pass": (coherent["flops"] == DEFAULT_FLOPS
                 and coherent["operations"] == DEFAULT_OPERATIONS
                 and coherent["lifecycle"]["output_sha256"]
                 == DEFAULT_OUTPUT_SHA256
                 and coherent["mechanisms"] == {
                     "consumers": True, "births": True,
                     "active": True, "lam": 1.0 / 3.0}),
        "flops": coherent["flops"],
        "operations": coherent["operations"],
    })

    exact_a = _run(False, False, -0.4)
    exact_b = _run(False, False, 0.73)
    forbidden = tuple(name for name in exact_a["families"]
                      if (name.startswith("k4_")
                          or "k4_birth" in name))
    rows.append({
        "check": "exact-is-master-off",
        "pass": (not exact_a["mechanisms"]["active"]
                 and exact_a["flops"] == exact_b["flops"]
                 and exact_a["operations"] == exact_b["operations"]
                 and exact_a["family_calls"] == exact_b["family_calls"]
                 and exact_a["lifecycle"]["output_sha256"]
                 == exact_b["lifecycle"]["output_sha256"]
                 and not forbidden),
        "unexpected_k4_families": list(forbidden),
    })

    lam_rows = [_run(True, True, lam) for lam in (-0.4, 0.0, 0.73)]
    rows.append({
        "check": "lambda-does-not-change-cost",
        "pass": (len({r["flops"] for r in lam_rows}) == 1
                 and len({r["operations"] for r in lam_rows}) == 1
                 and all(r["family_calls"] == lam_rows[0]["family_calls"]
                         for r in lam_rows[1:])),
        "flops": [r["flops"] for r in lam_rows],
        "operations": [r["operations"] for r in lam_rows],
    })

    exact_d1 = _run(False, False, depth=1)
    births_d1 = _run(False, True, depth=1)
    rows.append({
        "check": "depth-one-births-only-nests-exact",
        "pass": (exact_d1["lifecycle"]["output_sha256"]
                 == births_d1["lifecycle"]["output_sha256"]
                 and exact_d1["schedule"]["births"] == 0
                 and births_d1["schedule"]["births"] == 0),
    })

    exact = _run(False, False)
    consumers = _run(True, False)
    births = _run(False, True)
    layer_hashes = {name: r["lifecycle"]["layer_output_sha256"]
                    for name, r in (("exact", exact),
                                    ("coherent", coherent),
                                    ("consumers", consumers),
                                    ("births", births))}
    rows.append({
        "check": "later-depth-mechanisms-are-active",
        "pass": (len({v[0] for v in layer_hashes.values()}) == 1
                 and consumers["lifecycle"]["output_sha256"]
                 != exact["lifecycle"]["output_sha256"]
                 and births["lifecycle"]["output_sha256"]
                 != exact["lifecycle"]["output_sha256"]
                 and coherent["lifecycle"]["output_sha256"]
                 not in {exact["lifecycle"]["output_sha256"],
                         consumers["lifecycle"]["output_sha256"],
                         births["lifecycle"]["output_sha256"]}),
        "first_layer_hashes": {k: v[0] for k, v in layer_hashes.items()},
    })

    arm_results = {"exact": exact, "coherent": coherent,
                   "consumers_only": consumers, "births_only": births}
    rows.append({
        "check": "all-arms-dense-full-rank-state-and-closure",
        "pass": all(_parity_ok(result) for result in arm_results.values()),
        "closure_max_relative": {
            name: result["parity"]["tiny_dense_closure_max_relative"]
            for name, result in arm_results.items()},
        "state_max_relative": {
            name: result["parity"]["tiny_dense_shadow_max_relative"]
            for name, result in arm_results.items()},
    })

    failed = [row for row in rows if not row["pass"]]
    return {"kind": "label-free compact four-arm reference checks",
            "checks": rows, "n_checks": len(rows),
            "n_failed": len(failed), "pass": not failed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    result = run()
    for row in result["checks"]:
        print("%-48s %s" %
              (row["check"], "PASS" if row["pass"] else "FAIL"))
    print("%d/%d passed" %
          (result["n_checks"] - result["n_failed"], result["n_checks"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
