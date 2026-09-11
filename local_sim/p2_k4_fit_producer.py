"""Produce immutable coherent-K4 candidate observations through lifecycle_eval.

Every candidate and bound is supplied explicitly.  The producer writes one
candidate-specific generated source, evaluates it on the same registered
identities through the canonical lifecycle evaluator, and stores
final predictions, exact per-identity counted costs, and verified source paths
and hashes for p2_k4_fit.py.

Identity manifests may describe small synthetic fixtures, or the exact P07
design registry.  The latter is checked against seed_blocks before any network
is minted.  This producer never opens truth.

Run:
    uv run python local_sim/p2_k4_fit_producer.py \
      --candidates candidates.json --bounds bounds.json \
      --identities identities.json --out-dir candidate_observations
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import lifecycle_eval as le  # noqa: E402
import p2_build_k4_arms as builder  # noqa: E402
import p2_k4_fit as selector  # noqa: E402
import seed_blocks  # noqa: E402


class ProducerRefused(ValueError):
    pass


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256(path):
    with open(path, "rb") as fh:
        return _sha256_bytes(fh.read())


def _observed_binding(path, frozen_sha256):
    try:
        current_sha256 = _sha256(path)
        observation_error = ""
    except OSError as exc:
        current_sha256 = None
        observation_error = "%s: %s" % (type(exc).__name__, exc)
    return {
        "path": os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/"),
        "frozen_sha256": frozen_sha256,
        "current_sha256": current_sha256,
        "unchanged_after_evaluation": current_sha256 == frozen_sha256,
        "observation_error": observation_error,
    }


def _diagnostic_value(value):
    """Keep failure evidence strict JSON even when nested values contain NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    return value


def _write_failure_diagnostic(a, *, candidate_index, candidate, source_path,
                              source_hash, rows, provenance, predictions,
                              identity_manifest, width, depth, spec,
                              input_hashes, producer_binding,
                              generator_binding, failure_stage="predict"):
    row_values = []
    for row in rows:
        raw = dataclasses.asdict(row)
        row_values.append(_diagnostic_value(raw))
    try:
        current_source_sha = _sha256(source_path)
        source_observation_error = ""
    except OSError as exc:
        current_source_sha = None
        source_observation_error = "%s: %s" % (type(exc).__name__, exc)
    producer_diagnostic = _observed_binding(
        os.path.join(_ROOT, producer_binding["path"]),
        producer_binding["sha256"])
    generator_diagnostic = _observed_binding(
        os.path.join(_ROOT, generator_binding["path"]),
        generator_binding["sha256"])
    generator_diagnostic["upstream_source_sha256"] = \
        generator_binding["upstream_source_sha256"]
    generator_diagnostic["upstream_source"] = _observed_binding(
        builder.SOURCE, generator_binding["upstream_source_sha256"])
    result = {
        "schema": "arc-whest.coherent-k4-fit-producer-failure.v1",
        "immutable": True,
        "complete_candidate": False,
        "labels_read": False,
        "failure_stage": failure_stage,
        "candidate_index": candidate_index,
        "candidate": candidate,
        "candidate_source": {
            "path": os.path.relpath(source_path, _ROOT).replace("\\", "/"),
            "sha256": source_hash,
            "current_sha256": current_source_sha,
            "unchanged_after_evaluation": current_source_sha == source_hash,
            "observation_error": source_observation_error,
        },
        "inputs": {
            "candidates": _observed_binding(
                a.candidates, input_hashes["candidates_sha256"]),
            "bounds": _observed_binding(
                a.bounds, input_hashes["bounds_sha256"]),
            "identities": _observed_binding(
                a.identities, input_hashes["identities_sha256"]),
        },
        "producer": producer_diagnostic,
        "generator": generator_diagnostic,
        "identity_manifest": identity_manifest,
        "lifecycle_spec": _diagnostic_value(dataclasses.asdict(spec)),
        "lifecycle_provenance": _diagnostic_value(provenance),
        "evaluated_rows": len(rows),
        "captured_predictions": len(predictions),
        "rows_hash": provenance.get("rows_hash"),
        "lifecycle_candidate_sha256": provenance.get("candidate_sha256"),
        "rows": row_values,
    }
    path = os.path.join(a.out_dir, "failure_diagnostic.json")
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    return path


def _pre_row_provenance(source_path, source_hash, seeds, zero_target, spec,
                        exc):
    """Record what is knowable when lifecycle setup/import fails before rows."""
    backend = {"numpy": np.__version__, "python": sys.version.split()[0]}
    try:
        flops = le._require_real_flopscope()
        backend.update({
            "flopscope": getattr(flops, "__version__", "?"),
            "flopscope_path": os.path.dirname(flops.__file__ or ""),
        })
    except Exception as backend_exc:
        backend.update({
            "flopscope": "unavailable",
            "flopscope_path": "",
            "backend_error": "%s: %s" %
                             (type(backend_exc).__name__, backend_exc),
        })
    error = "%s: %s" % (type(exc).__name__, exc)
    return {
        "candidate": os.path.relpath(source_path, _ROOT).replace("\\", "/"),
        "candidate_sha256": None,
        "truth_sha256": le._array_sha(zero_target[:len(seeds)]),
        "n_mlps": len(seeds),
        "spec": dataclasses.asdict(spec),
        "round_tag": spec.round_tag,
        "round_deviations": spec.deviations_from_round(),
        "lifecycle": "grader",
        "weights_mode": le.DEFAULT_WEIGHTS_MODE,
        "weights_mode_is_grader_default": True,
        "setup_called": None,
        "setup_seconds": None,
        "setup_timeout_s": spec.setup_timeout_s,
        "setup_timed_out": "SETUP_TIMEOUT" in str(exc),
        "setup_error": error,
        "host_cold_start_s": le.HOST_COLD_START_S,
        "residual_measurement_path": "runner=local (in-process BudgetContext)",
        "residual_is_graded": False,
        "objective_is_graded": False,
        "backend": backend,
        "rows_hash": le.rows_hash([]),
    }


def _provenance_problem(provenance, *, spec, expected_path, source_hash,
                        n_mlps, truth_sha256, rows_hash):
    expected_keys = {
        "candidate", "candidate_sha256", "truth_sha256", "n_mlps", "spec",
        "round_tag", "round_deviations", "lifecycle", "weights_mode",
        "weights_mode_is_grader_default", "setup_called", "setup_seconds",
        "setup_timeout_s", "setup_timed_out", "setup_error",
        "host_cold_start_s", "residual_measurement_path",
        "residual_is_graded", "objective_is_graded", "backend", "rows_hash",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_keys:
        return "lifecycle provenance keys mismatch"
    if provenance["candidate"] != expected_path:
        return "provenance candidate path mismatch"
    if provenance["candidate_sha256"] != source_hash:
        return "provenance candidate hash mismatch"
    if provenance["truth_sha256"] != truth_sha256:
        return "provenance target hash mismatch"
    if provenance["n_mlps"] != n_mlps:
        return "provenance identity count mismatch"
    if provenance["spec"] != dataclasses.asdict(spec):
        return "provenance lifecycle spec mismatch"
    if (provenance["round_tag"] != spec.round_tag or
            provenance["round_deviations"] != spec.deviations_from_round()):
        return "provenance round mismatch"
    if (provenance["lifecycle"] != "grader" or
            provenance["weights_mode"] != le.DEFAULT_WEIGHTS_MODE or
            provenance["weights_mode_is_grader_default"] is not True):
        return "provenance grader lifecycle/weight mode mismatch"
    if (provenance["setup_called"] is not True or
            provenance["setup_timed_out"] is not False or
            provenance["setup_error"] != "" or
            provenance["setup_timeout_s"] != spec.setup_timeout_s):
        return "provenance setup outcome mismatch"
    setup_seconds = provenance["setup_seconds"]
    if (not isinstance(setup_seconds, (int, float)) or
            isinstance(setup_seconds, bool) or not math.isfinite(setup_seconds) or
            setup_seconds < 0):
        return "provenance setup duration invalid"
    if provenance["host_cold_start_s"] != le.HOST_COLD_START_S:
        return "provenance host cold-start mismatch"
    if (provenance["residual_measurement_path"] !=
            "runner=local (in-process BudgetContext)" or
            provenance["residual_is_graded"] is not False or
            provenance["objective_is_graded"] is not False):
        return "provenance grading-status mismatch"
    try:
        flops = le._require_real_flopscope()
        expected_backend = {
            "flopscope": getattr(flops, "__version__", "?"),
            "flopscope_path": os.path.dirname(flops.__file__ or ""),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        }
    except Exception as exc:
        return "cannot verify lifecycle backend: %s" % exc
    if provenance["backend"] != expected_backend:
        return "provenance backend mismatch"
    if provenance["rows_hash"] != rows_hash:
        return "provenance rows hash mismatch"
    return ""


def _lifecycle_evidence_problem(rows, predictions, provenance, seeds, spec,
                                source_path, source_hash, zero_target):
    if len(predictions) != len(seeds):
        return ("prediction count %d != identity count %d" %
                (len(predictions), len(seeds)))
    if len(rows) != len(seeds):
        return "row count %d != identity count %d" % (len(rows), len(seeds))
    expected_path = os.path.relpath(source_path, _ROOT).replace("\\", "/")
    expected_truth_sha = le._array_sha(zero_target[:len(seeds)])
    for i, prediction in enumerate(predictions):
        if (not isinstance(prediction, np.ndarray) or
                prediction.shape != (spec.width,) or
                prediction.dtype != np.dtype(np.float32) or
                not bool(np.all(np.isfinite(prediction)))):
            return "prediction %d shape/dtype/finiteness mismatch" % i
    for i, (row, seed) in enumerate(zip(rows, seeds)):
        if row.index != i or row.input_seed != seed:
            return "row %d identity/order mismatch" % i
        if (row.candidate != expected_path or
                row.candidate_sha256 != source_hash):
            return "row %d candidate binding mismatch" % i
        if row.truth_sha256 != expected_truth_sha or row.gt_noise != 0.0:
            return "row %d target/noise binding mismatch" % i
        if ((row.width, row.depth) != (spec.width, spec.depth) or
                row.budget != float(spec.flop_budget)):
            return "row %d lifecycle binding mismatch" % i
    problem = _provenance_problem(
        provenance, spec=spec, expected_path=expected_path,
        source_hash=source_hash, n_mlps=len(seeds),
        truth_sha256=expected_truth_sha, rows_hash=le.rows_hash(rows))
    if problem:
        return problem
    try:
        if _sha256(source_path) != source_hash:
            return "candidate source changed during evaluation"
    except OSError as exc:
        return "candidate source unavailable after evaluation: %s" % exc
    return ""


def _load_identities(path):
    manifest, ids, identity_sha = selector._identity_registry(
        path, return_hash=True)
    kind = manifest.get("kind")
    width, depth = int(manifest.get("width", 0)), int(manifest.get("depth", 0))
    seeds = [int(row["seed"]) for row in manifest["identities"]]
    if kind == "synthetic-identities-v1":
        if width <= 0 or depth <= 0:
            raise ProducerRefused("synthetic identity manifest width/depth must be positive")
        if len(seeds) != len(set(seeds)):
            raise ProducerRefused("synthetic identity seeds must be unique")
    elif kind == "registered-design-identities-v1":
        if manifest.get("block") != "P07":
            raise ProducerRefused("registered design identities require exact block P07")
        if (width, depth) != (1024, 16):
            raise ProducerRefused("P07 identity manifest requires width=1024 and depth=16")
        expected = seed_blocks.seeds("P07")
        if seeds != expected:
            raise ProducerRefused("identity seeds must exactly match P07 seed registry in order")
        if ids != [str(seed) for seed in expected]:
            raise ProducerRefused("identity ids must exactly match the ordered P07 seeds")
    else:
        raise ProducerRefused("identity manifest kind must be synthetic-identities-v1 or registered-design-identities-v1")
    return manifest, ids, seeds, width, depth, identity_sha


def _validate(a):
    for label, path in (("candidates", a.candidates), ("bounds", a.bounds),
                        ("identities", a.identities)):
        if not os.path.isfile(path):
            raise ProducerRefused("%s file does not exist: %s" % (label, path))
    if os.path.lexists(a.out_dir):
        raise ProducerRefused("output directory already exists: %s" % a.out_dir)
    parent = os.path.dirname(os.path.abspath(a.out_dir))
    if not os.path.isdir(parent):
        raise ProducerRefused("output parent does not exist: %s" % parent)
    try:
        names, candidates, bounds, matrix, registry_hashes = selector._registry(
            a.candidates, a.bounds, return_hashes=True)
        (identity_manifest, identities, seeds, width, depth,
         identity_sha) = _load_identities(a.identities)
    except selector.FitRefused as exc:
        raise ProducerRefused(str(exc)) from exc
    input_hashes = {
        "candidates_sha256": registry_hashes["candidates"],
        "bounds_sha256": registry_hashes["bounds"],
        "identities_sha256": identity_sha,
    }
    return (names, candidates, bounds, matrix, identity_manifest,
            identities, seeds, width, depth, input_hashes)


def produce(a):
    (names, candidates, bounds, matrix, identity_manifest,
     identities, seeds, width, depth, input_hashes) = _validate(a)
    producer_binding = {
        "path": os.path.relpath(__file__, _ROOT).replace("\\", "/"),
        "sha256": _sha256(__file__),
    }
    generator_binding = {
        "path": os.path.relpath(builder.__file__, _ROOT).replace("\\", "/"),
        "sha256": _sha256(builder.__file__),
    }
    src, upstream_sha = builder._source()
    generator_binding["upstream_source_sha256"] = upstream_sha
    coherent = next(row for row in builder.ARMS if row[0] == "coherent")
    os.mkdir(a.out_dir)
    source_paths, source_hashes = [], []
    all_predictions, all_costs, all_ops, row_hashes = [], [], [], []
    lifecycle_provenance = []
    spec = le.LifecycleSpec(width=width, depth=depth,
                            residual_wall_time_limit_s=None)
    zero_target = np.zeros((len(seeds), width), dtype=np.float32)
    for i, candidate in enumerate(candidates):
        name, consumers, births, blurb = coherent
        text = builder.build(name, consumers, births, blurb, src, upstream_sha,
                             candidate["lam"])
        source_path = os.path.join(a.out_dir, "candidate_%03d_coherent.py" % i)
        with open(source_path, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        source_hash = _sha256(source_path)
        predictions = []
        try:
            rows, provenance = le.evaluate(
                source_path, seeds, zero_target, spec=spec,
                gt_noise=[0.0] * len(seeds), predictions_out=predictions)
        except Exception as exc:
            provenance = _pre_row_provenance(
                source_path, source_hash, seeds, zero_target, spec, exc)
            diagnostic = _write_failure_diagnostic(
                a, candidate_index=i, candidate=candidate,
                source_path=source_path, source_hash=source_hash,
                rows=[], provenance=provenance, predictions=predictions,
                identity_manifest=identity_manifest, width=width, depth=depth,
                spec=spec, input_hashes=input_hashes,
                producer_binding=producer_binding,
                generator_binding=generator_binding,
                failure_stage="setup_or_lifecycle_exception")
            raise ProducerRefused(
                "candidate %d lifecycle raised %s; diagnostic: %s" %
                (i, type(exc).__name__, diagnostic)) from exc
        evidence_problem = _lifecycle_evidence_problem(
            rows, predictions, provenance, seeds, spec, source_path, source_hash,
            zero_target)
        if evidence_problem or any(row.failed for row in rows):
            failure_stage = ("incomplete_or_misbound_evidence"
                             if evidence_problem else "predict")
            diagnostic = _write_failure_diagnostic(
                a, candidate_index=i, candidate=candidate,
                source_path=source_path, source_hash=source_hash,
                rows=rows, provenance=provenance, predictions=predictions,
                identity_manifest=identity_manifest, width=width, depth=depth,
                spec=spec, input_hashes=input_hashes,
                producer_binding=producer_binding,
                generator_binding=generator_binding,
                failure_stage=failure_stage)
            raise ProducerRefused(
                "candidate %d failed canonical lifecycle%s; diagnostic: %s" %
                (i, (" (%s)" % evidence_problem) if evidence_problem else "",
                 diagnostic))
        if provenance["candidate_sha256"] != source_hash:
            raise ProducerRefused("candidate %d lifecycle/source hash mismatch" % i)
        all_predictions.append(np.stack(predictions).astype(np.float32))
        all_costs.append([int(row.flops_used) for row in rows])
        all_ops.append([int(row.n_ops) for row in rows])
        row_hashes.append(provenance["rows_hash"])
        lifecycle_provenance.append(_diagnostic_value(provenance))
        source_paths.append(os.path.relpath(source_path, _ROOT).replace("\\", "/"))
        source_hashes.append(source_hash)
    costs = np.asarray(all_costs, dtype=np.int64)
    if not np.array_equal(costs, np.repeat(costs[:1], len(candidates), axis=0)):
        raise ProducerRefused("coefficient candidates unexpectedly changed counted cost")
    manifest = {
        "kind": "coherent-K4 canonical-lifecycle observations-v1",
        "identity_role": identity_manifest["kind"],
        "producer": producer_binding,
        "generator": generator_binding,
        "candidate_registry": candidates, "bounds": bounds,
        "identity_manifest": identity_manifest,
        "lifecycle_spec": {"width": width, "depth": depth,
                           "flop_budget": int(spec.flop_budget),
                           "weights_mode": le.DEFAULT_WEIGHTS_MODE,
                           "residual_cap": None},
        "candidate_sources": [
            {"index": i, "candidate": candidates[i], "path": source_paths[i],
             "sha256": source_hashes[i]}
            for i in range(len(candidates))],
        "counted_costs": costs.tolist(),
        "row_hashes": row_hashes,
        "n_ops": all_ops,
        "lifecycle_provenance": lifecycle_provenance,
        "labels_read": False}
    output = os.path.join(a.out_dir, "predictions.npz")
    with open(output, "x+b") as fh:
        np.savez(fh, parameter_names=np.asarray(names), candidate_values=matrix,
                 identities=np.asarray(identities),
                 predictions=np.stack(all_predictions), counted_costs=costs,
                 candidate_source_paths=np.asarray(source_paths),
                 candidate_source_hashes=np.asarray(source_hashes),
                 producer_manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    result = {"schema": "arc-whest.coherent-k4-fit-producer-receipt.v1",
              "immutable": True, "predictions": output,
              "predictions_sha256": _sha256(output),
              "candidate_sources": source_paths,
              "candidate_source_hashes": source_hashes,
              "counted_costs": costs.tolist(), "manifest": manifest}
    with open(os.path.join(a.out_dir, "producer_receipt.json"), "x",
              encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--bounds", required=True)
    ap.add_argument("--identities", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        result = produce(a)
    except (ProducerRefused, OSError) as exc:
        print("REFUSED: %s" % exc)
        return 2
    print("produced %d immutable candidates at %s" %
          (len(result["candidate_sources"]), result["predictions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
