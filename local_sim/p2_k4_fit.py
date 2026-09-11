"""Bounded, receipt-producing selector for coherent-K4 fit observations.

The selector never invents candidates or bounds.  A producer evaluates the
explicitly registered candidate list through the four-arm lab and stores the
predictions and counted costs in an NPZ.  This program validates that archive,
the arm sources, the requested objective, and the unused receipt path before a
protected truth loader can reserve or open labels.

Truth-free producer verification:

    uv run python local_sim/p2_k4_fit.py --verify-only \
      --predictions predictions.npz --producer-receipt producer_receipt.json \
      --candidates candidates.json --bounds bounds.json \
      --identities identities.json \
      --objective final_layer_mse --receipt verification_receipt.json

Synthetic fit example (all input files must already exist):

    uv run python local_sim/p2_k4_fit.py --predictions predictions.npz \
      --producer-receipt producer_receipt.json \
      --truth synthetic_truth.npz --data-role synthetic \
      --candidates candidates.json --bounds bounds.json \
      --identities identities.json \
      --objective final_layer_mse --receipt receipt.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re

import numpy as np
from numpy import load as _load_npz

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ARMS = ("exact", "coherent", "consumers_only", "births_only")
_OBJECTIVES = ("final_layer_mse",)


class FitRefused(ValueError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bytes(path, label):
    """Return one immutable byte snapshot and the hash of those exact bytes."""
    try:
        with open(path, "rb") as fh:
            value = fh.read()
    except OSError as exc:
        raise FitRefused("cannot read %s: %s" % (label, exc)) from exc
    return value, hashlib.sha256(value).hexdigest()


def _read_json(path, label):
    try:
        raw, sha256 = _read_bytes(path, label)
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FitRefused("invalid %s JSON: %s" % (label, exc)) from exc
    return value, sha256


def _relative(path):
    return os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")


def _exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise FitRefused("%s keys must be exactly %s, got %s" %
                         (label, sorted(expected), got))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_json_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _repo_path(raw, label):
    if not isinstance(raw, str) or not raw:
        raise FitRefused("%s must be a nonempty path string" % label)
    path = os.path.realpath(os.path.abspath(
        raw if os.path.isabs(raw) else os.path.join(_ROOT, raw)))
    root = os.path.realpath(os.path.abspath(_ROOT))
    try:
        inside = os.path.commonpath([root, path]) == root
    except ValueError:
        inside = False
    if not inside:
        raise FitRefused("%s escapes the repository" % label)
    return path


def _same_path(left, right):
    return (os.path.normcase(os.path.realpath(os.path.abspath(left))) ==
            os.path.normcase(os.path.realpath(os.path.abspath(right))))


def _valid_int_matrix(value, rows, columns):
    return (isinstance(value, list) and len(value) == rows
            and all(isinstance(row, list) and len(row) == columns
                    and all(_is_int(x) and x >= 0 for x in row)
                    for row in value))


def _generated_constants(text, path):
    """Read the literal mechanism switches emitted by p2_build_k4_arms."""
    values = {}
    try:
        tree = ast.parse(text, filename=path)
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and
                    node.targets[0].id in {"_K4_ACTIVE", "_K4_CONSUMERS",
                                           "_K4_BIRTHS", "_K4_LAM"}):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    except (SyntaxError, ValueError) as exc:
        raise FitRefused("candidate source does not parse: %s: %s" %
                         (path, exc)) from exc
    return values


def _validate_sources():
    paths = {"fit_interface": os.path.abspath(__file__),
             "fit_producer": os.path.join(_HERE, "p2_k4_fit_producer.py"),
             "arm_generator": os.path.join(_HERE, "p2_build_k4_arms.py")}
    for arm in _ARMS:
        paths["arm_" + arm] = os.path.join(_HERE, "p2_arms",
                                            "arm_%s.py" % arm)
    hashes = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FitRefused("missing source: %s" % path)
        raw, sha256 = _read_bytes(path, "source")
        try:
            ast.parse(raw.decode("utf-8"), filename=path)
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise FitRefused("source does not parse: %s: %s" % (path, exc)) from exc
        hashes[name] = {"path": _relative(path), "sha256": sha256}
    return hashes


def _registry(candidate_path, bounds_path, *, return_hashes=False):
    raw_candidates, candidate_sha = _read_json(candidate_path, "candidate registry")
    raw_bounds, bounds_sha = _read_json(bounds_path, "bounds")
    _exact_keys(raw_candidates, {"parameters", "candidates"},
                "candidate registry")
    names = raw_candidates["parameters"]
    candidates = raw_candidates["candidates"]
    if (not isinstance(names, list) or not names or
            not all(isinstance(x, str) and re.fullmatch(r"[a-z][a-z0-9_]*", x)
                    for x in names) or len(names) != len(set(names))):
        raise FitRefused("parameters must be a nonempty unique name list")
    if names != ["lam"]:
        raise FitRefused("this repaired generator exposes exactly parameter 'lam'")
    if not isinstance(candidates, list) or not candidates:
        raise FitRefused("candidates must be a nonempty explicit list")
    if not isinstance(raw_bounds, dict) or set(raw_bounds) != set(names):
        raise FitRefused("bounds must name exactly the registered parameters")
    bounds = {}
    for name in names:
        bound = raw_bounds[name]
        if not isinstance(bound, dict) or set(bound) != {"min", "max"}:
            raise FitRefused("bound %s must contain exactly min/max" % name)
        if not _is_json_number(bound["min"]) or not _is_json_number(bound["max"]):
            raise FitRefused("bound %s min/max must be JSON numbers" % name)
        lo, hi = float(bound["min"]), float(bound["max"])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise FitRefused("invalid finite bound for %s" % name)
        bounds[name] = {"min": lo, "max": hi}
    matrix, normalized = [], []
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != set(names):
            raise FitRefused("candidate %d must name exactly the parameters" % i)
        row, item = [], {}
        for name in names:
            if not _is_json_number(candidate[name]):
                raise FitRefused("candidate %d %s must be a JSON number" %
                                 (i, name))
            value = float(candidate[name])
            if not np.isfinite(value):
                raise FitRefused("candidate %d %s is not finite" % (i, name))
            if not bounds[name]["min"] <= value <= bounds[name]["max"]:
                raise FitRefused("candidate %d %s is outside its bound" % (i, name))
            row.append(value)
            item[name] = value
        matrix.append(row)
        normalized.append(item)
    if len({tuple(row) for row in matrix}) != len(matrix):
        raise FitRefused("candidate registry contains duplicates")
    result = (names, normalized, bounds, np.asarray(matrix, dtype=np.float64))
    if return_hashes:
        return result + ({"candidates": candidate_sha, "bounds": bounds_sha},)
    return result


def _scalar_strings(value, label):
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise FitRefused("%s must be one-dimensional" % label)
    return [str(x) for x in arr.tolist()]


def _identity_registry(path, *, return_hash=False):
    raw, sha256 = _read_json(path, "identity registry")
    if not isinstance(raw, dict):
        raise FitRefused("identity manifest must be an object")
    kind = raw.get("kind")
    if kind == "synthetic-identities-v1":
        _exact_keys(raw, {"kind", "width", "depth", "identities"},
                    "synthetic identity manifest")
    elif kind == "registered-design-identities-v1":
        _exact_keys(raw, {"kind", "block", "width", "depth", "identities"},
                    "registered identity manifest")
    else:
        raise FitRefused("identity manifest kind must be synthetic-identities-v1 or registered-design-identities-v1")
    width, depth = raw["width"], raw["depth"]
    if not _is_int(width) or width <= 0 or not _is_int(depth) or depth <= 0:
        raise FitRefused("identity manifest width/depth must be positive integers")
    rows = raw["identities"]
    if not isinstance(rows, list) or not rows:
        raise FitRefused("identity registry must contain a nonempty identities list")
    ids, seeds = [], []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"id", "seed"}:
            raise FitRefused("identity %d must contain exactly id/seed" % i)
        identity, seed = row["id"], row["seed"]
        if (not isinstance(identity, str) or not identity
                or not _is_int(seed) or seed < 0):
            raise FitRefused("identity %d requires a nonempty string id and nonnegative integer seed" % i)
        ids.append(identity)
        seeds.append(seed)
    if kind == "synthetic-identities-v1":
        if len(ids) != len(set(ids)):
            raise FitRefused("identity registry contains duplicate ids")
        if len(seeds) != len(set(seeds)):
            raise FitRefused("identity registry contains duplicate seeds")
    else:
        if raw["block"] != "P07" or (width, depth) != (1024, 16):
            raise FitRefused("registered design identities require exact block P07 at width=1024 and depth=16")
        import seed_blocks
        expected = seed_blocks.seeds("P07")
        if seeds != expected or ids != [str(seed) for seed in expected]:
            raise FitRefused("registered identity seeds and ids must exactly match P07's ordered registry")
    result = (raw, ids)
    return result + (sha256,) if return_hash else result


def _validate_output_path(path):
    if not path:
        raise FitRefused("receipt path is required")
    if os.path.lexists(path):
        raise FitRefused("receipt already exists: %s" % path)
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise FitRefused("receipt parent directory does not exist: %s" % parent)


def _validate_request(a, *, verify_only=False):
    if a.objective not in _OBJECTIVES:
        raise FitRefused("objective must be explicitly one of %s" %
                         (", ".join(_OBJECTIVES)))
    for label, path in (("predictions", a.predictions),
                        ("producer receipt", a.producer_receipt),
                        ("candidates", a.candidates), ("bounds", a.bounds)):
        if not os.path.isfile(path):
            raise FitRefused("%s file does not exist: %s" % (label, path))
    if not os.path.isfile(a.identities):
        raise FitRefused("identities file does not exist: %s" % a.identities)
    _validate_output_path(a.receipt)
    if verify_only:
        forbidden = [name for name in ("truth", "data_role", "block", "ts", "note")
                     if getattr(a, name, "")]
        if forbidden:
            raise FitRefused("verify-only does not accept truth/read arguments: %s" %
                             ", ".join(forbidden))
    else:
        if a.data_role not in ("synthetic", "protected-design"):
            raise FitRefused("data role must be synthetic or protected-design")
        if a.data_role == "protected-design" and a.block != "P07":
            raise FitRefused("protected-design fitting is restricted to exact block P07")
    names, candidates, bounds, matrix, registry_hashes = _registry(
        a.candidates, a.bounds, return_hashes=True)
    identity_manifest, registered_identities, identity_sha = _identity_registry(
        a.identities, return_hash=True)
    if not verify_only:
        expected_role = ("registered-design-identities-v1"
                         if a.data_role == "protected-design"
                         else "synthetic-identities-v1")
        if identity_manifest["kind"] != expected_role:
            raise FitRefused("%s fitting requires identity role %s" %
                             (a.data_role, expected_role))
        if not a.truth or not os.path.isfile(a.truth):
            raise FitRefused("truth file does not exist: %s" % a.truth)
    source_hashes = _validate_sources()
    prediction_bytes, prediction_sha = _read_bytes(a.predictions,
                                                    "prediction archive")
    try:
        with _load_npz(io.BytesIO(prediction_bytes), allow_pickle=False) as z:
            required = {"parameter_names", "candidate_values", "identities",
                        "predictions", "counted_costs",
                        "candidate_source_paths", "candidate_source_hashes",
                        "producer_manifest_json"}
            missing = required - set(z.files)
            if missing:
                raise FitRefused("prediction archive misses %s" % sorted(missing))
            archive_names = _scalar_strings(z["parameter_names"], "parameter_names")
            values = np.asarray(z["candidate_values"], dtype=np.float64)
            identities = _scalar_strings(z["identities"], "identities")
            predictions = np.asarray(z["predictions"], dtype=np.float64)
            costs = np.asarray(z["counted_costs"])
            candidate_paths = _scalar_strings(z["candidate_source_paths"],
                                              "candidate_source_paths")
            candidate_hashes = _scalar_strings(z["candidate_source_hashes"],
                                               "candidate_source_hashes")
            producer_manifest = json.loads(str(np.asarray(
                z["producer_manifest_json"]).item()))
    except (OSError, ValueError) as exc:
        if isinstance(exc, FitRefused):
            raise
        raise FitRefused("invalid prediction archive: %s" % exc) from exc
    if archive_names != names or values.shape != matrix.shape or not np.array_equal(values, matrix):
        raise FitRefused("prediction candidates do not exactly match the registry")
    c, n = len(candidates), len(identities)
    if identities != registered_identities:
        raise FitRefused("prediction identities do not exactly match the registry")
    if predictions.ndim not in (3, 4) or predictions.shape[:2] != (c, n):
        raise FitRefused("predictions must have shape CxNxW or CxNxDxW")
    if costs.shape == (c,):
        costs = np.repeat(costs[:, None], n, axis=1)
    if costs.shape != (c, n) or costs.dtype.kind not in "iu" or np.any(costs < 0):
        raise FitRefused("counted_costs must be nonnegative integers shaped C or CxN")
    if not np.array_equal(costs, np.repeat(costs[:1], c, axis=0)):
        raise FitRefused("final_layer_mse requires value-independent candidate costs")
    if not np.isfinite(predictions).all():
        raise FitRefused("predictions contain non-finite values")
    if len(candidate_paths) != c or len(candidate_hashes) != c:
        raise FitRefused("candidate source path/hash count must equal candidate count")
    candidate_sources = []
    for i, (raw_path, expected) in enumerate(zip(candidate_paths, candidate_hashes)):
        path = _repo_path(raw_path, "candidate source %d" % i)
        if not os.path.isfile(path) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise FitRefused("candidate source %d is missing or has invalid hash" % i)
        source_bytes, actual = _read_bytes(path, "candidate source %d" % i)
        if actual != expected:
            raise FitRefused("candidate source %d hash mismatch" % i)
        try:
            constants = _generated_constants(source_bytes.decode("utf-8"), path)
        except UnicodeDecodeError as exc:
            raise FitRefused("candidate source %d cannot be read: %s" %
                             (i, exc)) from exc
        expected_constants = {"_K4_ACTIVE": True, "_K4_CONSUMERS": True,
                              "_K4_BIRTHS": True,
                              "_K4_LAM": candidates[i]["lam"]}
        if constants != expected_constants:
            raise FitRefused("candidate source %d does not encode its registered coherent configuration" % i)
        candidate_sources.append({"path": _relative(path), "sha256": actual,
                                  "verified": True})
    expected_manifest_sources = [
        {"index": i, "candidate": candidates[i],
         "path": candidate_sources[i]["path"],
         "sha256": candidate_sources[i]["sha256"]}
        for i in range(c)]
    manifest_keys = {"kind", "identity_role", "producer", "generator",
                     "candidate_registry", "bounds", "identity_manifest",
                     "input_bindings",
                     "lifecycle_spec", "candidate_sources", "counted_costs",
                     "row_hashes", "n_ops", "lifecycle_provenance",
                     "labels_read"}
    _exact_keys(producer_manifest, manifest_keys, "producer manifest")
    if (producer_manifest["kind"] !=
            "coherent-K4 canonical-lifecycle observations-v1"
            or producer_manifest["labels_read"] is not False
            or producer_manifest["identity_role"] != identity_manifest.get("kind")
            or producer_manifest["candidate_registry"] != candidates
            or producer_manifest["bounds"] != bounds
            or producer_manifest["identity_manifest"] != identity_manifest
            or producer_manifest["candidate_sources"] != expected_manifest_sources
            or producer_manifest["counted_costs"] != costs.tolist()):
        raise FitRefused("producer manifest does not match the validated observations")
    if not _valid_int_matrix(producer_manifest["counted_costs"], c, n):
        raise FitRefused("producer manifest costs must be an integer candidate-by-identity matrix")

    _exact_keys(producer_manifest["producer"], {"path", "sha256"},
                "producer manifest producer")
    if producer_manifest["producer"] != source_hashes["fit_producer"]:
        raise FitRefused("producer manifest does not match current producer bytes")
    _exact_keys(producer_manifest["generator"],
                {"path", "sha256", "upstream_source_sha256"},
                "producer manifest generator")
    generator = producer_manifest["generator"]
    if ({"path": generator["path"], "sha256": generator["sha256"]} !=
            source_hashes["arm_generator"]):
        raise FitRefused("producer manifest does not match current generator bytes")
    upstream = os.path.join(_ROOT, "estimator_k3_full.py")
    if (not _valid_hash(generator["upstream_source_sha256"])
            or not os.path.isfile(upstream)
            or generator["upstream_source_sha256"] != _sha256(upstream)):
        raise FitRefused("producer manifest upstream source hash mismatch")

    width = identity_manifest.get("width")
    depth = identity_manifest.get("depth")
    if not _is_int(width) or width <= 0 or not _is_int(depth) or depth <= 0:
        raise FitRefused("identity manifest width/depth must be positive integers")
    if predictions.ndim == 3 and predictions.shape[2] != width:
        raise FitRefused("prediction width does not match identity manifest")
    if predictions.ndim == 4 and predictions.shape[2:] != (depth, width):
        raise FitRefused("prediction depth/width do not match identity manifest")
    _exact_keys(producer_manifest["lifecycle_spec"],
                {"width", "depth", "flop_budget", "weights_mode", "residual_cap"},
                "producer lifecycle spec")
    lifecycle = producer_manifest["lifecycle_spec"]
    import lifecycle_eval as le
    expected_spec = le.LifecycleSpec(width=width, depth=depth,
                                     residual_wall_time_limit_s=None)
    expected_lifecycle = {"width": width, "depth": depth,
                          "flop_budget": int(expected_spec.flop_budget),
                          "weights_mode": le.DEFAULT_WEIGHTS_MODE,
                          "residual_cap": None}
    if lifecycle != expected_lifecycle:
        raise FitRefused("producer lifecycle spec does not match canonical lifecycle")

    row_hashes = producer_manifest["row_hashes"]
    n_ops = producer_manifest["n_ops"]
    if (not isinstance(row_hashes, list) or len(row_hashes) != c
            or not all(_valid_hash(x) for x in row_hashes)):
        raise FitRefused("producer row hashes must cover every candidate")
    if not _valid_int_matrix(n_ops, c, n):
        raise FitRefused("producer operation counts must cover every candidate and identity")
    lifecycle_provenance = producer_manifest["lifecycle_provenance"]
    if (not isinstance(lifecycle_provenance, list) or
            len(lifecycle_provenance) != c):
        raise FitRefused("producer lifecycle provenance must cover every candidate")
    import p2_k4_fit_producer as producer
    input_bindings = producer_manifest["input_bindings"]
    _exact_keys(input_bindings, {"candidates", "bounds", "identities"},
                "producer input bindings")
    expected_input_hashes = {
        "candidates": registry_hashes["candidates"],
        "bounds": registry_hashes["bounds"],
        "identities": identity_sha,
    }
    expected_input_paths = {
        "candidates": a.candidates,
        "bounds": a.bounds,
        "identities": a.identities,
    }
    for name in ("candidates", "bounds", "identities"):
        binding = input_bindings[name]
        _exact_keys(binding, {"path", "frozen_sha256", "current_sha256",
                              "unchanged_after_evaluation", "observation_error"},
                    "producer %s input binding" % name)
        expected_hash = expected_input_hashes[name]
        if (binding != {
                "path": _relative(expected_input_paths[name]),
                "frozen_sha256": expected_hash,
                "current_sha256": expected_hash,
                "unchanged_after_evaluation": True,
                "observation_error": ""}):
            raise FitRefused("producer %s input bytes do not match current input" %
                             name)
    zero_truth_sha = le._array_sha(np.zeros((n, width), dtype=np.float32))
    for i, item in enumerate(lifecycle_provenance):
        problem = producer._provenance_problem(
            item, spec=expected_spec,
            expected_path=candidate_sources[i]["path"],
            source_hash=candidate_sources[i]["sha256"], n_mlps=n,
            truth_sha256=zero_truth_sha, rows_hash=row_hashes[i])
        if problem:
            raise FitRefused("producer lifecycle provenance %d: %s" %
                             (i, problem))

    producer_receipt, producer_receipt_sha = _read_json(
        a.producer_receipt, "producer receipt")
    receipt_keys = {"schema", "immutable", "predictions", "predictions_sha256",
                    "candidate_sources", "candidate_source_hashes",
                    "counted_costs", "manifest"}
    _exact_keys(producer_receipt, receipt_keys, "producer receipt")
    receipt_prediction_path = _repo_path(producer_receipt["predictions"],
                                         "producer receipt predictions")
    if not _valid_int_matrix(producer_receipt["counted_costs"], c, n):
        raise FitRefused("producer receipt costs must be an integer candidate-by-identity matrix")
    if (producer_receipt["schema"] !=
            "arc-whest.coherent-k4-fit-producer-receipt.v1"
            or producer_receipt["immutable"] is not True
            or not _same_path(receipt_prediction_path, a.predictions)
            or not _valid_hash(producer_receipt["predictions_sha256"])
            or producer_receipt["predictions_sha256"] != prediction_sha
            or producer_receipt["candidate_sources"] != candidate_paths
            or producer_receipt["candidate_source_hashes"] != candidate_hashes
            or producer_receipt["counted_costs"] != costs.tolist()
            or producer_receipt["manifest"] != producer_manifest):
        raise FitRefused("producer receipt does not match validated observations")
    return {"parameter_names": names, "candidates": candidates,
            "bounds": bounds, "identities": identities,
            "identity_manifest": identity_manifest,
            "predictions": predictions, "counted_costs": costs.astype(np.int64),
            "candidate_sources": candidate_sources,
            "source_hashes": source_hashes,
            "producer_manifest": producer_manifest,
            "producer_receipt": producer_receipt,
            "producer_receipt_sha256": producer_receipt_sha,
            "predictions_sha256": prediction_sha,
            "registry_hashes": registry_hashes,
            "identities_sha256": identity_sha,
            "row_hashes": row_hashes, "n_ops": n_ops,
            "lifecycle_spec": lifecycle}


def _load_truth(a):
    reservation_receipt = None
    if a.data_role == "protected-design":
        import truth_reads
        block = truth_reads.resolve_block(a.truth, a.block)
        if block != "P07":
            raise FitRefused("protected-design fitting is restricted to exact block P07")
        z = truth_reads.load_truth(
            block, a.truth, consumer="local_sim/p2_k4_fit.py",
            purpose="bounded coherent-K4 coefficient fit",
            ts=a.ts, instrument=os.path.abspath(__file__),
            result_artifact=a.receipt, note=a.note)
        completed = [row for row in truth_reads.status(block)["reads"]
                     if row["consumer"] == "local_sim/p2_k4_fit.py"
                     and row["result_artifact"] == a.receipt
                     and row["status"] == "completed"]
        if not completed:
            raise FitRefused("protected read completed without a reservation receipt")
        reservation_receipt = completed[-1]
    else:
        # Synthetic fixtures are explicitly outside the protected registry.
        z = _load_npz(a.truth, allow_pickle=False)
    try:
        files = set(z.files if hasattr(z, "files") else z)
        if (a.data_role == "synthetic" and
                {"truth", "identities", "manifest_json"} <= files):
            truth = np.asarray(z["truth"], dtype=np.float64)
            identities = _scalar_strings(z["identities"], "truth identities")
            manifest = json.loads(str(np.asarray(z["manifest_json"]).item()))
            if (not isinstance(manifest, dict)
                    or manifest.get("kind") != "synthetic"
                    or manifest.get("labels_protected") is not False):
                raise FitRefused("synthetic truth requires an explicit unprotected manifest")
        elif (a.data_role == "protected-design" and
              {"means_r1", "means_r2", "seeds"} <= files):
            truth = ((np.asarray(z["means_r1"], dtype=np.float64)
                      + np.asarray(z["means_r2"], dtype=np.float64)) / 2.0)
            identities = [str(int(x)) for x in np.asarray(z["seeds"]).tolist()]
            manifest = {"kind": "protected-design", "block": a.block,
                        "replicas": 2,
                        "n_samples": (int(np.asarray(z["n_samples"]).item())
                                      if "n_samples" in files else None)}
        else:
            raise FitRefused("truth archive schema does not match the declared data role")
    finally:
        if hasattr(z, "close"):
            z.close()
    if not isinstance(manifest, dict):
        raise FitRefused("truth/synthetic manifest must be a JSON object")
    return identities, truth, manifest, reservation_receipt


def _write_receipt(path, payload):
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError as exc:
        raise FitRefused("receipt already exists: %s" % path) from exc


def verify(a):
    """Validate a producer receipt without accepting or resolving truth."""
    validated = _validate_request(a, verify_only=True)
    receipt = {
        "schema": "arc-whest.coherent-k4-producer-verification-receipt.v1",
        "immutable": True,
        "verification_only": True,
        "labels_read": False,
        "objective": a.objective,
        "inputs": {
            "producer_receipt": {"path": _relative(a.producer_receipt),
                                 "sha256": validated["producer_receipt_sha256"]},
            "predictions": {"path": _relative(a.predictions),
                            "sha256": validated["predictions_sha256"]},
            "candidates": {"path": _relative(a.candidates),
                           "sha256": validated["registry_hashes"]["candidates"]},
            "bounds": {"path": _relative(a.bounds),
                       "sha256": validated["registry_hashes"]["bounds"]},
            "identities": {"path": _relative(a.identities),
                           "sha256": validated["identities_sha256"]},
        },
        "source_hashes": validated["source_hashes"],
        "candidate_sources": validated["candidate_sources"],
        "candidate_registry": validated["candidates"],
        "bounds": validated["bounds"],
        "identity_manifest": validated["identity_manifest"],
        "lifecycle_spec": validated["lifecycle_spec"],
        "row_hashes": validated["row_hashes"],
        "n_ops": validated["n_ops"],
        "counted_costs": validated["counted_costs"].tolist(),
        "validation": {
            "producer_receipt_revalidated": True,
            "current_sources_revalidated": True,
            "candidate_count": len(validated["candidates"]),
            "identity_count": len(validated["identities"]),
            "truth_arguments_accepted": False,
        },
    }
    _write_receipt(a.receipt, receipt)
    return receipt


def fit(a, truth_loader=_load_truth):
    validated = _validate_request(a)  # receipt validation precedes truth loading
    loaded = truth_loader(a)
    if len(loaded) == 3:  # injectable synthetic test loaders
        truth_ids, truth, manifest = loaded
        reservation_receipt = None
    else:
        truth_ids, truth, manifest, reservation_receipt = loaded
    if truth_ids != validated["identities"]:
        raise FitRefused("truth identities do not exactly match prediction order")
    predictions = validated["predictions"]
    pred_final = predictions[:, :, -1, :] if predictions.ndim == 4 else predictions
    truth_final = truth[:, -1, :] if truth.ndim == 3 else truth
    if truth_final.shape != pred_final.shape[1:]:
        raise FitRefused("truth shape does not match each candidate final prediction")
    if not np.isfinite(truth).all():
        raise FitRefused("truth contains non-finite values")
    per_identity = ((pred_final - truth_final[None, :, :]) ** 2).mean(axis=2)
    objective = per_identity.mean(axis=1)
    selected = int(np.argmin(objective))  # registered order is the tie break
    receipt = {
        "schema": "arc-whest.coherent-k4-fit-receipt.v1",
        "immutable": True,
        "data_role": a.data_role,
        "objective": {"name": a.objective, "direction": "minimize",
                      "definition": "mean identity final-layer neuron MSE",
                      "validity": ("registered coefficient candidates have "
                                   "identical counted cost for every identity")},
        "parameter_names": validated["parameter_names"],
        "generator_binding": {"lam": "generated arm module constant _K4_LAM",
                              "arms": list(_ARMS)},
        "candidate_registry": validated["candidates"],
        "bounds": validated["bounds"],
        "identities": validated["identities"],
        "truth_manifest": manifest,
        "reservation_receipt": reservation_receipt,
        "inputs": {"predictions": {"path": _relative(a.predictions),
                                      "sha256": validated["predictions_sha256"]},
                   "producer_receipt": {
                       "path": _relative(a.producer_receipt),
                       "sha256": validated["producer_receipt_sha256"]},
                   "truth": {"path": _relative(a.truth),
                              "sha256": _sha256(a.truth)},
                   "candidates": {"path": _relative(a.candidates),
                                   "sha256": validated["registry_hashes"]["candidates"]},
                   "bounds": {"path": _relative(a.bounds),
                               "sha256": validated["registry_hashes"]["bounds"]},
                   "identities": {"path": _relative(a.identities),
                                   "sha256": validated["identities_sha256"]}},
        "source_hashes": validated["source_hashes"],
        "candidate_sources": validated["candidate_sources"],
        "counted_costs": validated["counted_costs"].tolist(),
        "results": [{"candidate": validated["candidates"][i],
                     "objective": float(objective[i]),
                     "per_identity": per_identity[i].tolist()}
                    for i in range(len(objective))],
        "selection": {"registered_index": selected,
                      "frozen_values": validated["candidates"][selected],
                      "objective": float(objective[selected]),
                      "tie_break": "first in explicitly registered order"},
        "validation": {"completed_before_truth_load": True,
                       "producer_receipt_revalidated_before_truth_load": True,
                       "candidate_count": len(objective),
                       "identity_count": len(truth_ids),
                       "all_finite": True}}
    # Exclusive creation is the immutability guarantee, including concurrent runs.
    _write_receipt(a.receipt, receipt)
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--producer-receipt", required=True)
    ap.add_argument("--verify-only", action="store_true",
                    help="validate producer evidence and write a receipt without accepting truth arguments")
    ap.add_argument("--truth", default="")
    ap.add_argument("--data-role", default=None,
                    choices=("synthetic", "protected-design"))
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--bounds", required=True)
    ap.add_argument("--identities", required=True)
    ap.add_argument("--objective", required=True, choices=_OBJECTIVES)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--block", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        receipt = verify(a) if a.verify_only else fit(a)
    except FitRefused as exc:
        print("REFUSED before %s: %s" %
              ("verification" if a.verify_only else "selection", exc))
        return 2
    if a.verify_only:
        print("verified immutable producer receipt without truth access")
        print("receipt: %s" % a.receipt)
        return 0
    print("selected registered candidate %d: %s" %
          (receipt["selection"]["registered_index"],
           json.dumps(receipt["selection"]["frozen_values"], sort_keys=True)))
    print("receipt: %s" % a.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
