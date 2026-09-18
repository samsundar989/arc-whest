"""Read-only P12 complete-package factorial synthesis.

Producer: p2_design_score.py with control C00 and candidates in this order:
C10,C01,C11,v31,sentinel. Its pred-out directory is mandatory. This reader
checks complete source-bound sidecars and uses competition_score for every
score and gate. P12 remains exposed DESIGN; no result here promotes a file.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import competition_score as cs  # noqa: E402
import design_sidecar as ds  # noqa: E402
import lifecycle_eval as le  # noqa: E402
import s15_own_fit_driver as p12  # noqa: E402

LABELS = ("C00", "C10", "C01", "C11", "v31", "sentinel")
# Reviewed, frozen role identities. A positional producer invocation alone
# cannot establish which method a score arm represents.
ROLE_PATHS = {
    "C00": "estimator_k3_v43_complete_c00.py",
    "C10": "estimator_k3_v43_complete_c10.py",
    "C01": "estimator_k3_v43_complete_c01.py",
    "C11": "estimator_k3_v43_complete_c11.py",
    "v31": "estimator_k3_v31.py",
    "sentinel": "estimator_k3_v43_fitted_uncompressed_p4d5.py",
}
ROLE_SHA256 = {
    "C00": "39d9a37a3567bb4d268b89b6e7c6913f23f254a4b52cb4191243b80eab5b0c42",
    "C10": "fd8d497db6dfbc3affcf579119c7115c3aee4d190a3e92fec4d85c366152d1bf",
    "C01": "7c0fcbcbce92c4f29e712c49e8b08d5dbe0eae2fc101b52f0cb363e79c2cdac8",
    "C11": "95cf412792abf4ec3d0dd21f33414be113e8bf5600b5714c61a301b1f53a4768",
    "v31": "6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c",
    "sentinel": "ebc0ee2ca107f39c8db06fc33d8b464b759ab09c0b3754ab99153d40c94073ca",
}
PAIRS = (("C00", "C10"), ("C00", "C01"), ("C10", "C11"),
         ("C01", "C11"), ("C00", "C11"), ("C10", "C01"))
B = 2 ** 41


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(raw, seeds):
    if len(raw) != 24 or [int(r["seed"]) for r in raw] != seeds:
        raise ValueError("incomplete or reordered P12 network rows")
    if [int(r["index"]) for r in raw] != list(range(24)):
        raise ValueError("missing or repeated network index")
    return [cs.MlpRow(mse=float(r["mse"]), flops_used=float(r["flops_used"]),
                      residual_wall_time_s=float(r["residual_wall_time_s"]),
                      budget=B,
                      failed=bool(r["failed"] or r["nonfinite"] or r["error_code"]),
                      name=f'seed:{r["seed"]}', gt_noise=float(r["gt_noise"]))
            for r in raw]


def _verdict(v):
    return {"passed": bool(v.passed), "gain": float(v.gain),
            "ci": list(v.ci), "reasons": list(v.reasons)}


def synthesize(score_path, prediction_dir, sources):
    d = json.loads(Path(score_path).read_text(encoding="utf-8"))
    if (d.get("tier") != "DESIGN" or d.get("block") != "P12" or
            d.get("n") != 24 or d.get("width") != 1024 or
            d.get("depth") != 16 or d.get("accuracy_only") is not True or
            d.get("local_seconds_caps") != "off"):
        raise ValueError("expected complete P12 DESIGN 1024x16 accuracy run")
    side = ds.verify(str(prediction_dir), root=str(ROOT),
                     require_present_sources=True, require_full_row_binding=True)
    man = side["manifest"]
    if (man.get("run_complete") is not True or man.get("block") != "P12" or
            man.get("n") != 24 or man.get("truth_sha256") != d["truth_sha256"] or
            man.get("truth_sha256") != p12.TRUTH_SHA or
            man.get("ordered_network_seeds") is None):
        raise ValueError("incomplete or mismatched prediction manifest")
    expected_spec = dataclasses.asdict(le.LifecycleSpec(
        width=1024, depth=16, flop_budget=B, lam=0.,
        residual_wall_time_limit_s=None, wall_time_limit_s=None,
        predict_timeout_s=1e9, setup_timeout_s=1e9))
    cfg = man.get("config", {})
    if (man.get("tier") != d["tier"] or man.get("width") != d["width"] or
            man.get("depth") != d["depth"] or man.get("truth") != d["truth"].replace("\\", "/") or
            man.get("n_samples_per_replica") != d.get("n_samples_per_replica") or
            man.get("n_samples_per_replica") != 2**24 or
            man.get("truth_replicas") != 2 or
            cfg.get("accuracy_only") != d["accuracy_only"] or
            cfg.get("local_seconds_caps") != d["local_seconds_caps"] or
            cfg.get("official_samples") != 1e9 or cfg.get("gate") != 1.0 or
            cfg.get("weights_mode") != le.DEFAULT_WEIGHTS_MODE or
            cfg.get("spec") != expected_spec or not cfg.get("ts") or
            man.get("control") != d.get("control") or
            man.get("candidate") != d.get("candidate")):
        raise ValueError("score/sidecar sample count, shape, protocol or configuration differs")
    if d.get("prediction_sidecars", "").replace("\\", "/") != str(
            Path(prediction_dir).resolve().relative_to(ROOT)).replace("\\", "/"):
        raise ValueError("score points to a different prediction directory")
    seeds = [int(x) for x in man["ordered_network_seeds"]]
    if seeds != list(p12.SEEDS):
        raise ValueError("bad P12 network identities")
    if set(d["arms"]) != {"control", "cand0", "cand1", "cand2", "cand3", "cand4"}:
        raise ValueError("expected exactly the four factorial arms, v31 and sentinel")
    if set(ROLE_PATHS) != set(LABELS) or set(ROLE_SHA256) != set(LABELS):
        raise ValueError("six reviewed source roles must be frozen before synthesis")
    out = {"tier": "DESIGN", "promotes_nothing": True,
           "score_artifact": str(score_path), "score_sha256": _sha(score_path),
           "prediction_manifest_sha256": _sha(Path(prediction_dir) / ds.MANIFEST),
           "arms": {}, "comparisons": {}}
    rows = {}
    for i, label in enumerate(LABELS):
        key = "control" if i == 0 else f"cand{i-1}"
        arm = d["arms"][key]
        source = Path(sources[label]).resolve()
        registered = ROOT / ROLE_PATHS[label]
        if (source != registered.resolve() or
                source != (ROOT / arm["path"]).resolve() or
                _sha(source) != ROLE_SHA256[label] or
                arm["candidate_sha256"] != ROLE_SHA256[label]):
            raise ValueError(f"{label}: source path or bytes differ")
        sarm = man["arms"].get(key)
        if (sarm is None or sarm["candidate_sha256"] != arm["candidate_sha256"] or
                sarm.get("rows_hash") != arm.get("rows_hash") or
                sarm.get("n_written") != 24 or
                sarm.get("n_failed") != arm.get("n_failed")):
            raise ValueError(f"{label}: sidecar source identity differs")
        if side["arms"][key]["seeds"] != seeds:
            raise ValueError(f"{label}: prediction identities differ")
        rr = _rows(arm["rows"], seeds)
        for j, x in enumerate(rr):
            ix = side["arms"][key]["rows"][j]
            if (ix["seed"] != seeds[j] or ix["mse"] != x.mse or
                    ix["flops_used"] != x.flops_used or ix["gt_noise"] != x.gt_noise or
                    ix["failed"] != x.failed):
                raise ValueError(f"{label}: score/sidecar row mismatch at {j}")
        rows[label] = rr
        observed = cs.score_suite(rr, lam=0.)
        official = cs.score_suite([
            dataclasses.replace(x, mse=x.mse-x.gt_noise+x.gt_noise*(2*float(d["n_samples_per_replica"])/1e9))
            for x in rr], lam=0.)
        if abs(observed.objective-float(arm["objective"])) > 1e-15:
            raise ValueError(f"{label}: saved objective differs from canonical scorer")
        out["arms"][label] = {
            "path": str(source), "sha256": arm["candidate_sha256"],
            "observed": observed.objective, "official": official.objective,
            "raw_mse": sum(x.mse for x in rr)/24,
            "truth_noise": sum(x.gt_noise for x in rr)/24,
            "mean_flops": sum(x.flops_used for x in rr)/24,
            "failed_seeds": [seeds[j] for j, x in enumerate(rr) if x.failed]}
    for control, cand in PAIRS + (("v31", "C00"), ("v31", "C10"),
                                  ("v31", "C01"), ("v31", "C11")):
        nac = cs.noise_corrected_compare(rows[control], rows[cand],
                                          local_samples=2*float(d["n_samples_per_replica"]),
                                          require_ids=True)
        # Canonical per-network suite scores, with the same precision conversion.
        def suite(label, precision):
            rr = rows[label]
            if precision == "official":
                rr = [dataclasses.replace(x, mse=x.mse-x.gt_noise+
                                     x.gt_noise*(2*float(d["n_samples_per_replica"])/1e9))
                      for x in rr]
            return cs.score_suite(rr, lam=0.).scores
        ov = cs.decide(suite(control, "observed"), suite(cand, "observed"),
                       gate=1.0, interval="t")
        fv = cs.decide(suite(control, "official"), suite(cand, "official"),
                       gate=1.0, interval="t")
        out["comparisons"][f"{control}:{cand}"] = {
            "observed": _verdict(ov), "official": _verdict(fv),
            "noise_corrected_difference": nac.true_function.mean_difference,
            "new_failed_seeds": sorted(set(out["arms"][cand]["failed_seeds"]) -
                                       set(out["arms"][control]["failed_seeds"]))}
    incumbent = "C00"
    trace = []
    for cand in ("C10", "C01", "C11"):
        def direct(control):
            if control == cand:
                return True
            a = cs.score_suite([dataclasses.replace(x, mse=x.mse-x.gt_noise+
                               x.gt_noise*(2*float(d["n_samples_per_replica"])/1e9))
                                for x in rows[control]], lam=0.).scores
            b = cs.score_suite([dataclasses.replace(x, mse=x.mse-x.gt_noise+
                               x.gt_noise*(2*float(d["n_samples_per_replica"])/1e9))
                                for x in rows[cand]], lam=0.).scores
            return cs.decide(a, b, gate=1.0, interval="t").passed
        no_new = all(not (set(out["arms"][cand]["failed_seeds"]) -
                              set(out["arms"][ctrl]["failed_seeds"]))
                     for ctrl in ("C00", incumbent))
        take = (out["arms"][cand]["official"] < out["arms"][incumbent]["official"]
                and direct("C00") and direct(incumbent) and no_new)
        trace.append({"candidate": cand, "prior": incumbent, "accepted": bool(take)})
        if take:
            incumbent = cand
    out["selection"] = {"incumbent": incumbent, "trace": trace}
    out["factorial_interaction_official"] = (
        out["arms"]["C11"]["official"] - out["arms"]["C10"]["official"]
        - out["arms"]["C01"]["official"] + out["arms"]["C00"]["official"])
    out["half_v31_targets"] = {
        "observed": out["arms"]["v31"]["observed"]/2,
        "official": out["arms"]["v31"]["official"]/2}
    out["fit_reduction_needed"] = {p: max(0., 1.-out["half_v31_targets"][p]/
                                                  out["arms"][incumbent][p])
                                   for p in ("observed", "official")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", required=True)
    ap.add_argument("--pred-out", required=True)
    for label in LABELS:
        ap.add_argument("--"+label.lower(), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sources = {label: getattr(a, label.lower()) for label in LABELS}
    result = synthesize(a.score, a.pred_out, sources)
    dest = Path(a.out)
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"selection": result["selection"],
                      "half_v31_targets": result["half_v31_targets"],
                      "fit_reduction_needed": result["fit_reduction_needed"]}, indent=2))


if __name__ == "__main__":
    main()
