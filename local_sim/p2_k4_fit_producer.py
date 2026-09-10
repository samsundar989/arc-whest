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
import hashlib
import json
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


def _load_identities(path):
    manifest, ids = selector._identity_registry(path)
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
    return manifest, ids, seeds, width, depth


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
        names, candidates, bounds, matrix = selector._registry(
            a.candidates, a.bounds)
        identity_manifest, identities, seeds, width, depth = _load_identities(
            a.identities)
    except selector.FitRefused as exc:
        raise ProducerRefused(str(exc)) from exc
    return (names, candidates, bounds, matrix, identity_manifest,
            identities, seeds, width, depth)


def produce(a):
    (names, candidates, bounds, matrix, identity_manifest,
     identities, seeds, width, depth) = _validate(a)
    src, upstream_sha = builder._source()
    coherent = next(row for row in builder.ARMS if row[0] == "coherent")
    os.mkdir(a.out_dir)
    source_paths, source_hashes = [], []
    all_predictions, all_costs, all_ops, row_hashes = [], [], [], []
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
        rows, provenance = le.evaluate(
            source_path, seeds, zero_target, spec=spec,
            gt_noise=[0.0] * len(seeds), predictions_out=predictions)
        if len(predictions) != len(seeds) or any(row.failed for row in rows):
            raise ProducerRefused("candidate %d failed canonical lifecycle" % i)
        if provenance["candidate_sha256"] != source_hash:
            raise ProducerRefused("candidate %d lifecycle/source hash mismatch" % i)
        all_predictions.append(np.stack(predictions).astype(np.float32))
        all_costs.append([int(row.flops_used) for row in rows])
        all_ops.append([int(row.n_ops) for row in rows])
        row_hashes.append(provenance["rows_hash"])
        source_paths.append(os.path.relpath(source_path, _ROOT).replace("\\", "/"))
        source_hashes.append(source_hash)
    costs = np.asarray(all_costs, dtype=np.int64)
    if not np.array_equal(costs, np.repeat(costs[:1], len(candidates), axis=0)):
        raise ProducerRefused("coefficient candidates unexpectedly changed counted cost")
    manifest = {
        "kind": "coherent-K4 canonical-lifecycle observations-v1",
        "identity_role": identity_manifest["kind"],
        "producer": {"path": os.path.relpath(__file__, _ROOT).replace("\\", "/"),
                     "sha256": _sha256(__file__)},
        "generator": {"path": os.path.relpath(builder.__file__, _ROOT).replace("\\", "/"),
                      "sha256": _sha256(builder.__file__),
                      "upstream_source_sha256": upstream_sha},
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
