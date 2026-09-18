"""Strict read-only P12 DESIGN decision for one frozen exact/approx query pair.

The source paths and SHA-256 values must be frozen by the caller before the
score run. This module opens saved scalar scores and verified prediction
sidecars, never runs an estimator, and never selects a replacement source.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import competition_score as cs  # noqa: E402
import design_sidecar as ds  # noqa: E402
import lifecycle_eval as le  # noqa: E402
import s15_own_fit_driver as p12  # noqa: E402

B = 2 ** 41
N = 24
LABELS = ("control", "candidate")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(path, digest):
    p = Path(path).resolve(strict=True)
    if p.parent != ROOT or p.suffix != ".py" or len(digest) != 64:
        raise ValueError("source must be a frozen root-level Python file and SHA-256")
    if _sha(p) != digest:
        raise ValueError(f"source bytes differ: {p.name}")
    return p


def _scored_rows(raw, seeds):
    if len(raw) != N or [int(r["index"]) for r in raw] != list(range(N)) or [int(r["seed"]) for r in raw] != seeds:
        raise ValueError("missing, repeated or reordered P12 rows")
    out = []
    for r in raw:
        for key in ("mse", "flops_used", "gt_noise", "residual_wall_time_s"):
            if not math.isfinite(float(r[key])):
                raise ValueError(f"nonfinite {key}")
        if r["flops_used"] < 0 or r["gt_noise"] < 0 or r["residual_wall_time_s"] < 0:
            raise ValueError("negative cost, truth noise or residual")
        out.append(cs.MlpRow(mse=float(r["mse"]), flops_used=float(r["flops_used"]),
                             residual_wall_time_s=float(r["residual_wall_time_s"]),
                             budget=B, gt_noise=float(r["gt_noise"]),
                             failed=bool(r["failed"]), name=f'seed:{r["seed"]}'))
    return out


def _scores(rows, local_samples, official=False):
    if official:
        rows = [dataclasses.replace(r, mse=r.mse-r.gt_noise +
                                    r.gt_noise*local_samples/1e9) for r in rows]
    return cs.score_suite(rows, lam=0.).scores


def _paired(control, candidate, gate):
    pair = cs.paired_compare(control, candidate)
    verdict = cs.decide(control, candidate, gate=gate, interval="t")
    return {"control_mean": float(pair.mean_control),
            "candidate_mean": float(pair.mean_candidate),
            "difference": float(pair.mean_difference),
            "t95": [float(x) for x in pair.t_ci_difference],
            "gain": float(pair.arithmetic_gain),
            "verdict": {"passed": bool(verdict.passed), "gate": float(verdict.gate),
                        "interval": verdict.interval, "ci": list(verdict.ci),
                        "reasons": list(verdict.reasons)}}


def synthesize(score_path, prediction_dir, exact_path, exact_sha,
               approx_path, approx_sha, *, source_root=ROOT):
    exact = _source(exact_path, exact_sha)
    approx = _source(approx_path, approx_sha)
    if exact == approx or exact_sha == approx_sha:
        raise ValueError("exact and approximate roles must be distinct")
    d = json.loads(Path(score_path).read_text(encoding="utf-8"))
    if (d.get("tier") != "DESIGN" or d.get("block") != "P12" or
            (d.get("n"), d.get("width"), d.get("depth"), d.get("n_samples_per_replica")) !=
            (N, 1024, 16, 2**24) or d.get("truth_sha256") != p12.TRUTH_SHA or
            d.get("accuracy_only") is not True or d.get("local_seconds_caps") != "off" or
            set(d.get("arms", {})) != set(LABELS)):
        raise ValueError("wrong P12 DESIGN score identity or protocol")
    root = Path(source_root).resolve()
    side = ds.verify(str(prediction_dir), root=str(root), check_sources=True,
                     require_present_sources=True, require_full_row_binding=True)
    man = side["manifest"]
    expected_spec = dataclasses.asdict(le.LifecycleSpec(
        width=1024, depth=16, flop_budget=B, lam=0.,
        residual_wall_time_limit_s=None, wall_time_limit_s=None,
        predict_timeout_s=1e9, setup_timeout_s=1e9))
    cfg = man.get("config", {})
    seeds = [int(s) for s in man.get("ordered_network_seeds", [])]
    if (man.get("run_complete") is not True or
            seeds != list(p12.SEEDS) or man.get("tier") != d["tier"] or
            any(man.get(k) != d.get(k) for k in ("block", "n", "width", "depth",
                "truth", "truth_sha256", "n_samples_per_replica", "control", "candidate")) or
            man.get("truth_replicas") != 2 or
            cfg.get("accuracy_only") is not True or cfg.get("local_seconds_caps") != "off" or
            cfg.get("official_samples") != 1e9 or cfg.get("gate") != 1.05 or
            cfg.get("weights_mode") != le.DEFAULT_WEIGHTS_MODE or
            cfg.get("spec") != expected_spec or not cfg.get("ts") or
            set(man.get("arms", {})) != set(LABELS) or
            set(side.get("arms", {})) != set(LABELS)):
        raise ValueError("score/sidecar provenance or registered configuration differs")
    if Path(ROOT / str(d.get("prediction_sidecars", ""))).resolve() != Path(prediction_dir).resolve():
        raise ValueError("score points to another prediction directory")
    sources = {"control": (exact, exact_sha), "candidate": (approx, approx_sha)}
    if Path(d["control"]).resolve() != exact or Path(d["candidate"]).resolve() != approx:
        raise ValueError("score source roles differ")
    rows = {}
    failed = {}
    for label in LABELS:
        arm = d["arms"][label]
        info = man["arms"][label]
        source, digest = sources[label]
        if (Path(arm["path"]).resolve() != source or
                Path(info["path"]).resolve() != source or
                arm.get("candidate_sha256") != digest or
                info.get("candidate_sha256") != digest or
                arm.get("rows_hash") != info.get("rows_hash") or
                info.get("n_written") != N):
            raise ValueError(f"{label}: source or sidecar hash differs")
        compact = arm["rows"]
        rr = _scored_rows(compact, seeds)
        full = info.get("evidence_rows")
        if full is None or len(full) != N or side["arms"][label]["seeds"] != seeds:
            raise ValueError(f"{label}: incomplete prediction sidecar")
        if le.rows_hash([le.EvidenceRow(**r) for r in full]) != arm["rows_hash"]:
            raise ValueError(f"{label}: full evidence rows hash differs")
        for i, (short, long, index) in enumerate(zip(compact, full, side["arms"][label]["rows"])):
            for a, b in (("index", "index"), ("seed", "input_seed"),
                         ("mse", "mse"), ("flops_used", "flops_used"),
                         ("gt_noise", "gt_noise"), ("residual_wall_time_s", "residual_wall_time_s"),
                         ("nonfinite", "nonfinite"), ("error_code", "error_code")):
                if short.get(a) != long.get(b):
                    raise ValueError(f"{label}: score/sidecar row {i} {a} differs")
            f = bool(long.get("budget_exhausted") or long.get("time_exhausted") or
                     long.get("residual_exhausted") or long.get("combined_exhausted") or
                     long.get("error_code"))
            if short["failed"] != f or index["failed"] != f or bool(short["nonfinite"]) != bool(long["nonfinite"]):
                raise ValueError(f"{label}: score/sidecar failure differs")
        if sum(r.failed for r in rr) != arm.get("n_failed") or abs(
                cs.score_suite(rr, lam=0.).objective-float(arm["objective"])) > 1e-15:
            raise ValueError(f"{label}: score summary differs")
        rows[label] = rr
        failed[label] = {r.name for r in rr if r.failed}
    cs.noise_corrected_compare(rows["control"], rows["candidate"],
                               local_samples=2*2**24, official_samples=1e9,
                               require_ids=True)
    observed = _paired(_scores(rows["control"], 2*2**24),
                       _scores(rows["candidate"], 2*2**24), 1.05)
    official = _paired(_scores(rows["control"], 2*2**24, True),
                       _scores(rows["candidate"], 2*2**24, True), 1.05)
    new_failed = sorted(failed["candidate"]-failed["control"])
    return {"artifact": "s17_query_decide", "tier": "DESIGN", "promotes_nothing": True,
            "score_sha256": _sha(score_path),
            "prediction_manifest_sha256": _sha(Path(prediction_dir) / ds.MANIFEST),
            "source_sha256": {"exact": exact_sha, "approx": approx_sha},
            "truth_sha256": d["truth_sha256"],
            "truth_noise_floor": sum(r.gt_noise for r in rows["control"])/N,
            "observed": observed, "official": official,
            "new_failed_identities": new_failed,
            "QUERY_SCORE_PASSED": bool(official["verdict"]["passed"] and not new_failed)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("score", "pred-out", "exact", "exact-sha", "approx", "approx-sha", "out"):
        ap.add_argument("--"+key, required=True)
    a = ap.parse_args()
    out = synthesize(a.score, a.pred_out, a.exact, a.exact_sha, a.approx, a.approx_sha)
    dest = Path(a.out)
    with dest.open("x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"QUERY_SCORE_PASSED": out["QUERY_SCORE_PASSED"],
                      "official": out["official"],
                      "new_failed_identities": out["new_failed_identities"]}, indent=2))


if __name__ == "__main__":
    main()
