"""Bounded, source-pinned P12 DESIGN fit of one selected complete package.

Dry-run does not open truth or execute an estimator. Execution is conditional on
the registered factorial trigger and must be submitted separately via workerq.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import sys

import s14_own_fit_controls as controls
import s15_own_fit_driver as old

ROOT = Path(__file__).resolve().parents[1]
NAMES = old.NAMES
BOUNDS = old.BOUNDS
STEPS = old.STEPS
RIDGE = old.RIDGE
TOTAL_EVALS = old.TOTAL_EVALS
MAX_EVALS = old.MAX_EVALS
FLAGS = ("_OWN_THETA_FAM", "_OWN_S_LAM", "_OWN_T_LAM", "_OWN_S_FEED")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected(raw: bytes) -> tuple[str, tuple[float, ...], bool]:
    src = raw.decode("utf-8")
    if not all(src.count(marker) == 1 for marker in
               ("_QUERY_BASIS = False", "_P3_ON = ", "_K4_PERNEURON = True")):
        raise ValueError("selected source is not a complete gram/P3/P4 package")
    values = []
    counts = []
    for flag in FLAGS:
        matches = re.findall(rf"(?m)^{flag} = ([^\n]+)$", src)
        counts.append(len(matches))
        if matches:
            try:
                values.append(float(matches[0]))
            except ValueError as exc:
                raise ValueError(f"invalid {flag} literal") from exc
    if counts == [0] * 4:
        return src, old.X0, False
    if counts != [1] * 4:
        raise ValueError("incomplete or repeated own controls")
    center = tuple(values)
    if not all(math.isfinite(v) and lo <= v <= hi
               for v, (lo, hi) in zip(center, BOUNDS)):
        raise ValueError("selected controls outside registered bounds")
    return src, center, True


def render(src: str, center: tuple[float, ...], has_controls: bool,
           point: tuple[float, ...]) -> bytes:
    if len(point) != 4 or not all(math.isfinite(v) and lo <= v <= hi
                                  for v, (lo, hi) in zip(point, BOUNDS)):
        raise ValueError("trial outside registered bounds")
    if point == center:
        return src.encode("utf-8")  # Exact selected source, including wrapper.
    if not has_controls:
        return controls.build(src, **dict(zip(NAMES, point))).encode("utf-8")
    for flag, value in zip(FLAGS, point):
        src, count = re.subn(rf"(?m)^{flag} = [^\n]+$", f"{flag} = {value!r}", src)
        if count != 1:
            raise ValueError(f"lost {flag} anchor")
    return src.encode("utf-8")


def nearby(point, coordinate, direction, step):
    trial = list(point)
    lo, hi = BOUNDS[coordinate]
    trial[coordinate] = min(hi, max(lo, trial[coordinate] + direction * step))
    return tuple(trial)


def objective(score, baseline, point, center):
    if not (math.isfinite(score) and math.isfinite(baseline) and baseline > 0):
        raise ValueError("invalid fit score")
    return score / baseline + RIDGE * sum((a-b)**2 for a, b in zip(point, center))


def coordinate_fit(evaluate, center):
    """Exactly 17 first-half evaluations, then two frozen last-half evaluations."""
    point = center
    base = evaluate("fit", point, 0)
    incumbent = objective(base, base, point, center)
    history = [("fit", point, base)]
    for steps in STEPS:
        for j, step in enumerate(steps):
            best_point, best_value = point, incumbent
            for direction in (-1, 1):
                trial = nearby(point, j, direction, step)
                score = evaluate("fit", trial, len(history))
                history.append(("fit", trial, score))
                value = objective(score, base, trial, center)
                if value < best_value:  # ties retain incumbent
                    best_point, best_value = trial, value
            point, incumbent = best_point, best_value
    held_base = evaluate("validation", center, len(history))
    history.append(("validation", center, held_base))
    held = evaluate("validation", point, len(history))
    history.append(("validation", point, held))
    assert len(history) == TOTAL_EVALS == 19 <= MAX_EVALS
    return point, base, held_base, held, history


def plan(source: Path, expected_sha: str) -> dict:
    raw = source.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError("selected source SHA-256 mismatch")
    src, center, has_controls = selected(raw)
    return {"tier": "DESIGN", "block": "P12", "selected_source": str(source.resolve()),
            "selected_source_sha256": expected_sha, "center": dict(zip(NAMES, center)),
            "source_has_controls": has_controls, "bounds": BOUNDS, "passes": STEPS,
            "coordinate_order": NAMES, "directions": (-1, 1),
            "ridge_center": "selected source controls", "ridge": RIDGE,
            "tie_rule": "retain incumbent", "scheduled_evaluations": TOTAL_EVALS,
            "max_evaluations": MAX_EVALS, "fit_seeds": old.SEEDS[:12],
            "validation_seeds": old.SEEDS[12:], "full_network_runs": 228,
            "validation_rule": "default then selected on last12 after fit freeze",
            "workerq_request": {"ram_gib": 12, "gpu_count": 0, "cpus": 4}}


def execute(source: Path, expected_sha: str, out_dir: Path, ts: str):
    import numpy as np
    import competition_score as cs
    import design_sidecar as ds
    import lifecycle_eval as le
    import truth_reads as tr

    source = source.resolve()
    out_dir = out_dir.resolve()
    manifest = plan(source, expected_sha)
    raw = source.read_bytes()
    src, center, has_controls = selected(raw)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("output directory must be empty; no automatic recovery")
    if not ts or not ts.endswith("Z"):
        raise ValueError("registered UTC truth-read timestamp required")
    if sha(old.TRUTH.read_bytes()) != old.TRUTH_SHA:
        raise ValueError("P12 truth SHA-256 mismatch")
    with tr.reserve("P12", str(old.TRUTH), consumer="local_sim/s17_complete_fit.py",
                    purpose="registered conditional complete-package first12 fit/last12 validation",
                    ts=ts, instrument=str(Path(__file__).resolve()),
                    result_artifact=str(out_dir / "result.json")) as truth_path:
        with np.load(truth_path) as z:
            if tuple(int(s) for s in z["seeds"]) != old.SEEDS:
                raise ValueError("P12 seed identity mismatch")
            if (int(z["width"]), int(z["depth"]), int(z["n_samples"])) != (1024, 16, 2**24):
                raise ValueError("P12 shape/precision mismatch")
            truth = (z["means_r1"][:, -1, :].astype(np.float64)
                     + z["means_r2"][:, -1, :].astype(np.float64)) / 2
            noise = z["gt_noise"].astype(np.float64) / 2
    spec = old.design_spec()
    manifest.update({"truth_sha256": old.TRUTH_SHA, "status": "running", "evaluations": [],
                     "instrument_sha256": sha(Path(__file__).read_bytes()),
                     "generator_sha256": sha(Path(controls.__file__).read_bytes()),
                     "dependency_sha256": {name: sha((ROOT / "local_sim" / name).read_bytes())
                         for name in ("s15_own_fit_driver.py", "lifecycle_eval.py",
                                      "competition_score.py", "truth_reads.py",
                                      "design_sidecar.py", "mint_provenance.py", "official_config.py")},
                     "environment": {"python": sys.version, "numpy": np.__version__,
                                     "platform": platform.platform(), "executable": sys.executable,
                                     "threads": {k: os.environ.get(k) for k in
                                         ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
                     "local_seconds_caps": "disabled for DESIGN; not release evidence",
                     "spec": dataclasses.asdict(spec)})
    old._atomic_json(out_dir / "manifest.json", manifest)
    baseline = [None]

    def evaluate(split, point, index):
        if index >= TOTAL_EVALS or split != ("fit" if index < 17 else "validation"):
            raise ValueError("fit evaluation order/cap violation")
        if sha(source.read_bytes()) != expected_sha:
            raise ValueError("selected source changed during fit")
        name = f"eval_{index:02d}_{split}"
        trial = out_dir / (name + ".py")
        candidate = render(src, center, has_controls, point)
        with trial.open("xb") as fh:
            fh.write(candidate)
            fh.flush(); os.fsync(fh.fileno())
        candidate_sha = sha(candidate)
        indices = list(range(12)) if split == "fit" else list(range(12, 24))
        ids = [old.SEEDS[i] for i in indices]
        rows_file = out_dir / (name + ".jsonl")
        pred_dir = out_dir / (name + "_predictions")
        writer = ds.ArmWriter(str(pred_dir), "candidate", width=1024,
                              candidate=str(trial), candidate_sha256=candidate_sha)
        preds = []
        def sink(row):
            with rows_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dataclasses.asdict(row), sort_keys=True) + "\n")
                fh.flush(); os.fsync(fh.fileno())
            writer.append(row, preds[-1])
        rows, prov = le.evaluate(str(trial), ids, truth[indices], spec=spec,
                                 gt_noise=list(noise[indices]), predictions_out=preds,
                                 evidence_sink=sink)
        if (len(rows) != 12 or [r.input_seed for r in rows] != ids
                or writer.n != 12 or any(r.failed for r in rows)):
            raise ValueError("incomplete, failed, or mismatched evaluation")
        if prov.get("candidate_sha256") != candidate_sha:
            raise ValueError("lifecycle source SHA-256 mismatch")
        official = [dataclasses.replace(r.to_row(), mse=r.mse-r.gt_noise
                    + r.gt_noise*(2*2**24/1e9)) for r in rows]
        score = cs.score_suite(official, lam=0.).objective
        rec = {"index": index, "split": split, "parameters": dict(zip(NAMES, point)),
               "candidate": trial.name, "sha256": candidate_sha,
               "rows": rows_file.name, "rows_file_sha256": sha(rows_file.read_bytes()),
               "rows_hash": le.rows_hash(rows), "prediction_directory": pred_dir.name,
               "prediction_stream_sha256": sha(Path(writer._stream).read_bytes()),
               "prediction_index_sha256": sha(Path(writer._index).read_bytes()),
               "provenance": prov, "n": 12, "n_failed": 0,
               "official_precision_score": score, "observed_score": le.score(rows, spec=spec).objective,
               "mean_flops": sum(r.flops_used for r in rows)/12,
               "gt_noise_mean": float(noise[indices].mean())}
        if index == 0:
            baseline[0] = score
            manifest["frozen_default_fit_score"] = score
        elif split == "fit":
            rec["penalized_fit_objective"] = objective(score, baseline[0], point, center)
        manifest["evaluations"].append(rec)
        old._atomic_json(out_dir / "manifest.json", manifest)
        return score

    point, base, held_base, held, history = coordinate_fit(evaluate, center)
    selected_sha = sha(render(src, center, has_controls, point))
    selected_fit = [rec["index"] for rec in manifest["evaluations"][:17]
                    if rec["sha256"] == selected_sha]
    if not selected_fit or manifest["evaluations"][18]["sha256"] != selected_sha:
        raise ValueError("selected source lacks an exact first12/last12 match")
    manifest.update({"status": "complete", "selected_parameters": dict(zip(NAMES, point)),
                     "frozen_default_fit_score": base, "heldout_default_official_score": held_base,
                     "heldout_selected_official_score": held, "evaluations_used": len(history),
                     "default_fit_evaluation_index": 0,
                     "default_validation_evaluation_index": 17,
                     "selected_fit_evaluation_index": selected_fit[-1],
                     "selected_validation_evaluation_index": 18,
                     "selected_source_sha256_after_fit": selected_sha})
    old._atomic_json(out_dir / "manifest.json", manifest)
    old._atomic_json(out_dir / "result.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument("--dry-run", action="store_true")
    choice.add_argument("--execute", action="store_true")
    ap.add_argument("--selected-source", type=Path, required=True)
    ap.add_argument("--selected-sha256", required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--ts", default="")
    args = ap.parse_args()
    if args.dry_run:
        print(json.dumps(plan(args.selected_source, args.selected_sha256), indent=2))
    else:
        if args.out_dir is None:
            ap.error("--execute requires --out-dir")
        execute(args.selected_source, args.selected_sha256, args.out_dir, args.ts)


if __name__ == "__main__":
    main()
