"""Read-only, source-bound verification of the fixed P12 own-fit output.

This consumes saved JSON and row bytes only. It never opens truth labels,
executes an estimator, decides a gate, or repairs an incomplete fit.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import tempfile

import competition_score as cs
import lifecycle_eval as le
import s14_own_fit_controls as gen
import s15_own_fit_driver as driver

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SAMPLES = 2 * 2**24
OFFICIAL_SAMPLES = 1e9


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"nonlocal output artifact path: {name!r}")
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise ValueError(f"missing output artifact: {name}")
    return path


def _same_number(actual, expected, label: str) -> None:
    if not (isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-20)):
        raise ValueError(f"{label} mismatch: {actual!r} vs {expected!r}")


def verify(directory: str | Path, *, root: Path = ROOT) -> dict:
    directory = Path(directory).resolve()
    result = json.loads(_path(directory, "result.json").read_text(encoding="utf-8"))
    manifest = json.loads(_path(directory, "manifest.json").read_text(encoding="utf-8"))
    if result != manifest or result.get("status") != "complete":
        raise ValueError("result/manifest differ or fit is incomplete")
    planned = driver.plan()
    for key, value in planned.items():
        if result.get(key) != value:
            raise ValueError(f"registered fit plan mismatch: {key}")
    if result.get("truth_sha256") != driver.TRUTH_SHA:
        raise ValueError("truth identity mismatch")
    if result.get("local_seconds_caps") != "disabled for P12 DESIGN; not release evidence":
        raise ValueError("local seconds cap declaration mismatch")
    environment = result.get("numerical_environment")
    if not isinstance(environment, dict) or not all(
        isinstance(environment.get(key), str) and environment[key]
        for key in ("python", "numpy", "platform", "executable")):
        raise ValueError("missing numerical environment identity")
    threads = environment.get("threads")
    if (not isinstance(threads, dict) or
            set(threads) != {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"} or
            not all(value is None or isinstance(value, str)
                    for value in threads.values())):
        raise ValueError("missing numerical thread environment identity")
    expected_sources = {
        "base_source_sha256": gen.BASE,
        "driver_sha256": Path(driver.__file__),
        "generator_sha256": Path(gen.__file__),
    }
    for key, source in expected_sources.items():
        if result.get(key) != _sha(source):
            raise ValueError(f"source identity mismatch: {key}")
    for name in ("competition_score.py", "lifecycle_eval.py",
                 "mint_provenance.py", "official_config.py", "truth_reads.py"):
        if result.get("dependency_sha256", {}).get(name) != _sha(root / "local_sim" / name):
            raise ValueError(f"dependency identity mismatch: {name}")
    if result.get("evaluations_used") != driver.TOTAL_EVALS:
        raise ValueError("evaluation count mismatch")
    records = result.get("evaluations")
    if not isinstance(records, list) or len(records) != driver.TOTAL_EVALS:
        raise ValueError("missing or extra evaluations")
    if [r.get("index") for r in records] != list(range(driver.TOTAL_EVALS)):
        raise ValueError("evaluation indices incomplete, duplicated or out of order")

    reconstructed = []
    seen_sources = set()
    spec = driver.design_spec()
    split_truth_hash = {}
    network_identity = {}
    for index, rec in enumerate(records):
        split = "fit" if index < 17 else "validation"
        if rec.get("split") != split:
            raise ValueError(f"evaluation {index}: split leakage")
        expected_seeds = list(driver.SEEDS[:12] if split == "fit" else driver.SEEDS[12:])
        name = f"eval_{index:02d}_{split}"
        if rec.get("candidate") != name + ".py" or rec.get("rows") != name + ".jsonl":
            raise ValueError(f"evaluation {index}: artifact naming mismatch")
        source = _path(directory, rec["candidate"])
        rows_path = _path(directory, rec["rows"])
        if source in seen_sources:
            raise ValueError("candidate source reused across evaluations")
        seen_sources.add(source)
        if _sha(source) != rec.get("sha256") or _sha(rows_path) != rec.get("rows_file_sha256"):
            raise ValueError(f"evaluation {index}: source or row byte SHA mismatch")
        provenance = rec.get("provenance", {})
        if provenance.get("candidate_sha256") != rec["sha256"]:
            raise ValueError(f"evaluation {index}: lifecycle source identity mismatch")
        truth_sha = provenance.get("truth_sha256")
        if not (isinstance(truth_sha, str) and len(truth_sha) == 64 and
                all(ch in "0123456789abcdef" for ch in truth_sha)):
            raise ValueError(f"evaluation {index}: missing split truth identity")
        if split in split_truth_hash and truth_sha != split_truth_hash[split]:
            raise ValueError(f"evaluation {index}: split truth identity changed")
        split_truth_hash.setdefault(split, truth_sha)
        if (provenance.get("n_mlps") != 12 or
                provenance.get("spec") != dataclasses.asdict(spec) or
                provenance.get("round_tag") != spec.round_tag or
                provenance.get("lifecycle") != "grader" or
                provenance.get("weights_mode") != le.DEFAULT_WEIGHTS_MODE or
                provenance.get("setup_called") is not True):
            raise ValueError(f"evaluation {index}: lifecycle spec mismatch")
        params = rec.get("parameters")
        if not isinstance(params, dict) or list(params) != list(driver.NAMES):
            raise ValueError(f"evaluation {index}: parameter names/order mismatch")
        for key, val, (lo, hi) in zip(driver.NAMES, params.values(), driver.BOUNDS):
            if not isinstance(val, (int, float)) or not math.isfinite(val) or not lo <= val <= hi:
                raise ValueError(f"evaluation {index}: invalid {key}")
        with tempfile.TemporaryDirectory(prefix="own_fit_source_verify_") as tmp:
            expected_sha = gen.emit(Path(tmp) / "generated.py", **params)
        if rec["sha256"] != expected_sha:
            raise ValueError(f"evaluation {index}: candidate differs from maintained generator")
        raw = rows_path.read_text(encoding="utf-8").splitlines()
        if len(raw) != 12 or rec.get("n") != 12:
            raise ValueError(f"evaluation {index}: incomplete 12-network rows")
        try:
            rows = [le.EvidenceRow(**json.loads(line)) for line in raw]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"evaluation {index}: malformed rows") from exc
        if [r.input_seed for r in rows] != expected_seeds or [r.index for r in rows] != list(range(12)):
            raise ValueError(f"evaluation {index}: ordered network identities mismatch")
        if (len({r.input_seed for r in rows}) != 12 or
                le.rows_hash(rows) != rec.get("rows_hash") or
                le.rows_hash(rows) != provenance.get("rows_hash")):
            raise ValueError(f"evaluation {index}: duplicate identity or rows_hash mismatch")
        for row in rows:
            if (row.candidate_sha256 != rec["sha256"] or
                    row.truth_sha256 != truth_sha or
                    row.width != spec.width or row.depth != spec.depth or
                    row.budget != spec.flop_budget):
                raise ValueError(f"evaluation {index}: row provenance mismatch")
            # These values are determined by the minted network and its
            # labels, never by the candidate. A changed value on a later arm
            # would make an apparently paired fit compare different inputs.
            fixed = (row.index, row.estimator_seed, row.truth_sha256,
                     row.gt_noise, row.zero_prediction_mse, row.width,
                     row.depth, row.budget)
            prior = network_identity.setdefault(row.input_seed, fixed)
            if fixed != prior:
                raise ValueError(f"evaluation {index}: cross-evaluation "
                                 f"network identity mismatch on seed {row.input_seed}")
            if row.nonfinite:
                raise ValueError(f"evaluation {index}: nonfinite failed seed {row.input_seed}")
            if row.failed:
                if not math.isclose(row.mse, row.zero_prediction_mse, rel_tol=1e-12):
                    raise ValueError(f"evaluation {index}: noncanonical failure fallback on seed {row.input_seed}")
                raise ValueError(f"evaluation {index}: failed seed {row.input_seed}; fit must abort")
            if not all(math.isfinite(v) for v in (row.mse, row.flops_used,
                                                   row.gt_noise, row.n_ops)):
                raise ValueError(f"evaluation {index}: nonfinite score row")
        if rec.get("n_failed") != 0:
            raise ValueError(f"evaluation {index}: false failure count")
        observed_rows = [r.to_row() for r in rows]
        official_rows = [dataclasses.replace(r, mse=r.mse-r.gt_noise+
                         r.gt_noise*LOCAL_SAMPLES/OFFICIAL_SAMPLES)
                         for r in observed_rows]
        observed = cs.score_suite(observed_rows, lam=0.).objective
        official = cs.score_suite(official_rows, lam=0.).objective
        _same_number(rec.get("observed_score"), observed, f"evaluation {index} observed")
        _same_number(rec.get("official_precision_score"), official,
                     f"evaluation {index} official")
        _same_number(rec.get("mean_flops"), sum(r.flops_used for r in rows)/12,
                     f"evaluation {index} FLOPs")
        _same_number(rec.get("gt_noise_mean"), sum(r.gt_noise for r in rows)/12,
                     f"evaluation {index} noise")
        reconstructed.append((rows, observed_rows, official_rows, official))

    if split_truth_hash.get("fit") == split_truth_hash.get("validation"):
        raise ValueError("fit and validation truth slices share one identity")

    def replay(split, x, index):
        rec = records[index]
        if rec["split"] != split or tuple(rec["parameters"][k] for k in driver.NAMES) != tuple(x):
            raise ValueError(f"evaluation {index}: deviates from frozen coordinate schedule")
        return reconstructed[index][3]
    selected, base, held_base, held, history = driver.coordinate_fit(replay)
    if result.get("selected_parameters") != dict(zip(driver.NAMES, selected)):
        raise ValueError("selected parameters differ from frozen training result")
    for key, value in (("frozen_default_fit_score", base),
                       ("heldout_default_official_score", held_base),
                       ("heldout_selected_official_score", held)):
        _same_number(result.get(key), value, key)
    for index, rec in enumerate(records[:17]):
        if index == 0:
            continue
        _same_number(rec.get("penalized_fit_objective"),
                     driver.objective(reconstructed[index][3], base,
                                      tuple(rec["parameters"][k] for k in driver.NAMES)),
                     f"evaluation {index} fit objective")
    control = reconstructed[17][1]
    candidate = reconstructed[18][1]
    paired = cs.noise_corrected_compare(control, candidate,
                                        local_samples=LOCAL_SAMPLES,
                                        official_samples=OFFICIAL_SAMPLES,
                                        lam=0., require_ids=True)
    return {
        "artifact": "own_fit_verified_summary", "tier": "DESIGN",
        "promotion_decision": None, "evaluations": 19,
        "numerical_environment": environment,
        "local_seconds_caps": result["local_seconds_caps"],
        "selected_parameters": dict(zip(driver.NAMES, selected)),
        "heldout": {
            "observed": {"control": paired.observed.mean_control,
                         "candidate": paired.observed.mean_candidate,
                         "difference": paired.observed.mean_difference,
                         "t95": list(paired.observed.t_ci_difference)},
            "official_precision": {"control": paired.forecast.mean_control,
                                   "candidate": paired.forecast.mean_candidate,
                                   "difference": paired.forecast.mean_difference,
                                   "t95": list(paired.forecast.t_ci_difference),
                                   "arithmetic_gain": paired.forecast.arithmetic_gain},
            "truth_noise_floor_mean": sum(r.gt_noise for r in reconstructed[17][0])/12,
            "control_counted_flops_mean": records[17]["mean_flops"],
            "candidate_counted_flops_mean": records[18]["mean_flops"],
            "control_failed": 0, "candidate_failed": 0,
            "ordered_seeds": list(driver.SEEDS[12:]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dir), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
