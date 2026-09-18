"""Strict read-only P15 same-closure attribution reader; no deployment verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import competition_score as cs  # noqa: E402
import design_sidecar as ds  # noqa: E402
import s18_reference_score as scorer  # noqa: E402
from seed_blocks import seeds as registered_seeds  # noqa: E402

N = 24
P15_SEEDS = registered_seeds("P15")[:N]
WIDTH, DEPTH = 1024, 16
SAMPLES = 2**24
GATE = 1.10
FIXED_MULTIPLIER = 0.1
ROLES = {"control": "C10", "unshared": "unshared",
         "full_p2": "full_p2", "combined": "unshared_full_p2"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numbers(rows):
    raw = np.asarray([r["mse"] for r in rows], dtype=np.float64)
    noise = np.asarray([r["gt_noise_averaged"] for r in rows], dtype=np.float64)
    if (not np.isfinite(raw).all() or not np.isfinite(noise).all() or
            np.any(raw < 0) or np.any(noise < 0)):
        raise ValueError("invalid raw MSE or truth noise")
    return raw, noise


def _arm(directory, role):
    directory = Path(directory).resolve(strict=True)
    m = scorer.verify(directory)
    if (m["arm"] != ROLES[role] or m["block"] != "P15" or
            (m["n"], m["width"], m["depth"], m["n_samples_per_replica"]) !=
            (N, WIDTH, DEPTH, SAMPLES) or
            m["ordered_network_seeds"] != P15_SEEDS or
            m["config"] != {"input_weight_dtype": "f64_readonly",
                            "diagnostic_budget": scorer.DIAGNOSTIC_BUDGET,
                            "official_budget": scorer.OFFICIAL_BUDGET,
                            "lambda": 0.0, "local_seconds_caps": "off",
                            "complete_endogenous": True,
                            "deployable_cost_claim": False}):
        raise ValueError("wrong P15 arm, seed order or numerical configuration")
    with np.load(m["truth"], allow_pickle=False) as z:
        seeds, width, depth, provenance = scorer._truth(z, "P15")
        if (list(map(int, seeds)) != m["ordered_network_seeds"] or
                (width, depth) != (WIDTH, DEPTH) or
                provenance != m["truth_provenance"] or
                int(z["n_samples"]) != SAMPLES):
            raise ValueError("truth numerical identity mismatch")
        target = ((z["means_r1"][:, -1].astype(np.float64) +
                   z["means_r2"][:, -1].astype(np.float64)) / 2).astype(np.float32)
        noise = z["gt_noise"].astype(np.float64) / 2
    side = json.loads(Path(m["truth"] + ".producer.json").read_text(encoding="utf-8"))
    if side != {"artifact_sha256": m["truth_sha256"], "math_provenance": provenance}:
        raise ValueError("truth producer sidecar mismatch")
    if (provenance["mint_sha256"] != m["dependency_sha256"]["local_sim/mint_provenance.py"] or
            provenance["producer_sha256"] != m["dependency_sha256"]["local_sim/p2_truth.py"] or
            m["packages"] != {x: scorer._version(x) for x in ("flopscope", "whestbench", "numpy")}):
        raise ValueError("truth producer or package identity mismatch")
    rows = [json.loads(x) for x in (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    loaded = ds.load_arm(str(directory), m["arm"], width=WIDTH, expect_n=N)
    if len(rows) != N or np.any(~np.isfinite(loaded["predictions"])):
        raise ValueError("incomplete or nonfinite predictions")
    for i, (r, idx, pred) in enumerate(zip(rows, loaded["rows"], loaded["predictions"])):
        mse = float(np.mean((pred - target[i]) ** 2))
        if (r["failed"] or idx["failed"] or r["index"] != i or
                r["input_seed"] != P15_SEEDS[i] or
                r["mse"] != mse or r["gt_noise_averaged"] != float(noise[i]) or
                idx["gt_noise"] != float(noise[i]) or
                r["scalar_materializations_total"] != 0 or
                sum(r["scalar_materializations"].values()) != 0 or
                r["last_run"]["mode"] != "full" or
                r["last_run"]["generations"] is not None or
                r["flops_used"] <= 0 or r["operation_calls"] <= 0 or
                r["flops_used"] != sum(v["flops"] for v in r["operations"].values()) or
                r["operation_calls"] != sum(v["calls"] for v in r["operations"].values()) or
                any(v["flops"] < 0 or v["calls"] < 0 for v in r["operations"].values())):
            raise ValueError(f"{role} row {i}: scored prediction, cost or status mismatch")
    return m, rows, sha(directory / ds.MANIFEST)


def _comparison(control, candidate, *, binding):
    c, nc = _numbers(control)
    a, na = _numbers(candidate)
    if not np.array_equal(nc, na):
        raise ValueError("paired arms have different truth noise")
    c, a = c * FIXED_MULTIPLIER, a * FIXED_MULTIPLIER
    paired = cs.paired_compare(c, a)
    out = {"control_mean_raw_fixed_cost": float(c.mean()),
           "candidate_mean_raw_fixed_cost": float(a.mean()),
           "paired_raw_difference": float(paired.mean_difference),
           "paired_t95": list(map(float, paired.t_ci_difference)),
           "arithmetic_gain": float(paired.arithmetic_gain),
            "noise_corrected_true_function_difference": float(np.mean((c - nc * FIXED_MULTIPLIER) -
                                                                   (a - na * FIXED_MULTIPLIER))),
           "mean_midpoint_truth_noise_floor_fixed_cost": float(np.mean(nc) * FIXED_MULTIPLIER)}
    # The same noise subtracts from both arms, so the difference is unchanged.
    if binding:
        verdict = cs.decide(c, a, gate=GATE, interval="t")
        out["isolated_accuracy_gate"] = {"passed": bool(verdict.passed),
                                          "gate": GATE, "interval": "t",
                                          "ci": list(map(float, verdict.ci)),
                                          "reasons": list(verdict.reasons)}
    return out


def synthesize(control, unshared, full_p2, combined):
    paths = dict(zip(ROLES, (control, unshared, full_p2, combined)))
    if len({str(Path(p).resolve()) for p in paths.values()}) != 4:
        raise ValueError("four distinct arm directories required")
    arms = {role: _arm(path, role) for role, path in paths.items()}
    base = arms["control"][0]
    for role, (m, _, _) in arms.items():
        for key in ("truth", "truth_sha256", "block", "ordered_network_seeds",
                    "n", "width", "depth", "n_samples_per_replica",
                    "truth_provenance", "control_sha256", "generator_sha256",
                    "dependency_sha256", "packages", "numerical_environment", "config"):
            if m[key] != base[key]:
                raise ValueError(f"{role}: cross-arm {key} mismatch")
    results = {role: _comparison(arms["control"][1], arms[role][1],
                                 binding=role != "combined")
               for role in ("unshared", "full_p2", "combined")}
    return {"artifact": "s18_reference_decide_v1", "tier": "DESIGN",
            "block": "P15", "n": N, "truth_sha256": base["truth_sha256"],
            "fixed_cost_multiplier": FIXED_MULTIPLIER,
            "raw_score_gate_only": True, "deployable_cost_claim": False,
            "arms": {role: {"source_sha256": m["source_sha256"],
                            "manifest_sha256": digest,
                            "mean_raw_mse": m["mean_raw_mse"],
                            "mean_midpoint_truth_noise_floor": m["mean_truth_noise_floor"],
                            "mean_counted_flops": float(np.mean([r["flops_used"] for r in rows])),
                            "observed_counted_cost_score": m["official_precision_score_forecast"],
                            "official_precision_counted_cost_forecast": float(np.mean([
                                cs.mlp_score(r["mse"] - r["gt_noise_averaged"] +
                                             r["gt_noise_averaged"] * (2 * SAMPLES) / 1e9,
                                             r["flops_used"], 0.0, scorer.OFFICIAL_BUDGET,
                                             lam=0.0) for r in rows]))}
                     for role, (m, rows, digest) in arms.items()},
            "comparisons": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        ap.add_argument("--" + role.replace("_", "-"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    result = synthesize(a.control, a.unshared, a.full_p2, a.combined)
    with Path(a.out).open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(json.dumps(result["comparisons"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
