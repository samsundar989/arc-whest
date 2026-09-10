"""Bounded, receipt-producing selector for coherent-K4 fit observations.

The selector never invents candidates or bounds.  A producer evaluates the
explicitly registered candidate list through the four-arm lab and stores the
predictions and counted costs in an NPZ.  This program validates that archive,
the arm sources, the requested objective, and the unused receipt path before a
protected truth loader can reserve or open labels.

Synthetic example (all three input files must already exist):

    uv run python local_sim/p2_k4_fit.py --predictions predictions.npz \
      --truth synthetic_truth.npz --data-role synthetic \
      --candidates candidates.json --bounds bounds.json \
      --objective final_layer_mse --receipt receipt.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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


def _read_json(path, label):
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise FitRefused("invalid %s JSON: %s" % (label, exc)) from exc
    return value


def _relative(path):
    return os.path.relpath(os.path.abspath(path), _ROOT).replace("\\", "/")


def _validate_sources():
    paths = {"fit_interface": os.path.abspath(__file__),
             "arm_generator": os.path.join(_HERE, "p2_build_k4_arms.py")}
    for arm in _ARMS:
        paths["arm_" + arm] = os.path.join(_HERE, "p2_arms",
                                            "arm_%s.py" % arm)
    hashes = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FitRefused("missing source: %s" % path)
        try:
            with open(path, encoding="utf-8") as fh:
                ast.parse(fh.read(), filename=path)
        except (OSError, SyntaxError) as exc:
            raise FitRefused("source does not parse: %s: %s" % (path, exc)) from exc
        hashes[name] = {"path": _relative(path), "sha256": _sha256(path)}
    return hashes


def _registry(candidate_path, bounds_path):
    raw_candidates = _read_json(candidate_path, "candidate registry")
    raw_bounds = _read_json(bounds_path, "bounds")
    if not isinstance(raw_candidates, dict):
        raise FitRefused("candidate registry must be an object")
    names = raw_candidates.get("parameters")
    candidates = raw_candidates.get("candidates")
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
    return names, normalized, bounds, np.asarray(matrix, dtype=np.float64)


def _scalar_strings(value, label):
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise FitRefused("%s must be one-dimensional" % label)
    return [str(x) for x in arr.tolist()]


def _validate_request(a):
    if a.objective not in _OBJECTIVES:
        raise FitRefused("objective must be explicitly one of %s" %
                         (", ".join(_OBJECTIVES)))
    if a.data_role not in ("synthetic", "protected-design"):
        raise FitRefused("data role must be synthetic or protected-design")
    for label, path in (("predictions", a.predictions), ("truth", a.truth),
                        ("candidates", a.candidates), ("bounds", a.bounds)):
        if not os.path.isfile(path):
            raise FitRefused("%s file does not exist: %s" % (label, path))
    if not a.receipt:
        raise FitRefused("receipt path is required")
    if os.path.lexists(a.receipt):
        raise FitRefused("receipt already exists: %s" % a.receipt)
    parent = os.path.dirname(os.path.abspath(a.receipt))
    if not os.path.isdir(parent):
        raise FitRefused("receipt parent directory does not exist: %s" % parent)
    if a.data_role == "protected-design" and a.block.strip().upper() != "P07":
        raise FitRefused("protected-design fitting is restricted to exact block P07")
    names, candidates, bounds, matrix = _registry(a.candidates, a.bounds)
    source_hashes = _validate_sources()
    try:
        with _load_npz(a.predictions, allow_pickle=False) as z:
            required = {"parameter_names", "candidate_values", "identities",
                        "predictions", "counted_costs",
                        "producer_candidate_source_hashes"}
            missing = required - set(z.files)
            if missing:
                raise FitRefused("prediction archive misses %s" % sorted(missing))
            archive_names = _scalar_strings(z["parameter_names"], "parameter_names")
            values = np.asarray(z["candidate_values"], dtype=np.float64)
            identities = _scalar_strings(z["identities"], "identities")
            predictions = np.asarray(z["predictions"], dtype=np.float64)
            costs = np.asarray(z["counted_costs"])
            candidate_hashes = _scalar_strings(
                z["producer_candidate_source_hashes"],
                "producer_candidate_source_hashes")
    except (OSError, ValueError) as exc:
        if isinstance(exc, FitRefused):
            raise
        raise FitRefused("invalid prediction archive: %s" % exc) from exc
    if archive_names != names or values.shape != matrix.shape or not np.array_equal(values, matrix):
        raise FitRefused("prediction candidates do not exactly match the registry")
    c, n = len(candidates), len(identities)
    if not identities or len(set(identities)) != n:
        raise FitRefused("prediction identities must be nonempty and unique")
    if predictions.ndim != 4 or predictions.shape[:2] != (c, n):
        raise FitRefused("predictions must have shape (candidates, identities, depth, width)")
    if costs.shape == (c,):
        costs = np.repeat(costs[:, None], n, axis=1)
    if costs.shape != (c, n) or costs.dtype.kind not in "iu" or np.any(costs < 0):
        raise FitRefused("counted_costs must be nonnegative integers shaped C or CxN")
    if not np.array_equal(costs, np.repeat(costs[:1], c, axis=0)):
        raise FitRefused("final_layer_mse requires value-independent candidate costs")
    if not np.isfinite(predictions).all():
        raise FitRefused("predictions contain non-finite values")
    if len(candidate_hashes) != c or any(not re.fullmatch(r"[0-9a-f]{64}", h)
                                         for h in candidate_hashes):
        raise FitRefused("producer_candidate_source_hashes must contain one lowercase SHA-256 per candidate")
    return {"parameter_names": names, "candidates": candidates,
            "bounds": bounds, "identities": identities,
            "predictions": predictions, "counted_costs": costs.astype(np.int64),
            "producer_candidate_source_hashes": candidate_hashes,
            "source_hashes": source_hashes}


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


def fit(a, truth_loader=_load_truth):
    validated = _validate_request(a)  # must precede reservation/loading
    loaded = truth_loader(a)
    if len(loaded) == 3:  # injectable synthetic test loaders
        truth_ids, truth, manifest = loaded
        reservation_receipt = None
    else:
        truth_ids, truth, manifest, reservation_receipt = loaded
    if truth_ids != validated["identities"]:
        raise FitRefused("truth identities do not exactly match prediction order")
    predictions = validated["predictions"]
    if truth.shape != predictions.shape[1:]:
        raise FitRefused("truth shape does not match each candidate prediction")
    if not np.isfinite(truth).all():
        raise FitRefused("truth contains non-finite values")
    per_identity = ((predictions[:, :, -1, :] - truth[None, :, -1, :]) ** 2).mean(axis=2)
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
                                      "sha256": _sha256(a.predictions)},
                   "truth": {"path": _relative(a.truth),
                             "sha256": _sha256(a.truth)},
                   "candidates": {"path": _relative(a.candidates),
                                  "sha256": _sha256(a.candidates)},
                   "bounds": {"path": _relative(a.bounds),
                              "sha256": _sha256(a.bounds)}},
        "source_hashes": validated["source_hashes"],
        "producer_asserted_candidate_source_hashes": validated[
            "producer_candidate_source_hashes"],
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
                       "candidate_count": len(objective),
                       "identity_count": len(truth_ids),
                       "all_finite": True}}
    # Exclusive creation is the immutability guarantee, including concurrent runs.
    try:
        with open(a.receipt, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(receipt, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except FileExistsError as exc:
        raise FitRefused("receipt already exists: %s" % a.receipt) from exc
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--data-role", required=True,
                    choices=("synthetic", "protected-design"))
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--bounds", required=True)
    ap.add_argument("--objective", required=True, choices=_OBJECTIVES)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--block", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        receipt = fit(a)
    except FitRefused as exc:
        print("REFUSED before selection: %s" % exc)
        return 2
    print("selected registered candidate %d: %s" %
          (receipt["selection"]["registered_index"],
           json.dumps(receipt["selection"]["frozen_values"], sort_keys=True)))
    print("receipt: %s" % a.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
