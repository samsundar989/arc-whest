"""Truth-free one-row lifecycle and operation ledger for a compact K4 arm."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os

import numpy as np

import lifecycle_eval as le
import p2_k4_fit as fit
import p2_build_k4_arms as builder


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(candidate, *, seed, width, depth, out=""):
    full = os.path.realpath(os.path.abspath(candidate))
    root = os.path.realpath(_ROOT)
    if os.path.commonpath([root, full]) != root or not os.path.isfile(full):
        raise ValueError("candidate must be an existing repository file")
    emitted = {
        os.path.realpath(os.path.join(builder.ARMS_DIR, "arm_%s.py" % name)):
        (name, consumers, births, blurb)
        for name, consumers, births, blurb in builder.ARMS}
    if full not in emitted:
        raise ValueError("candidate is not one of the four registered arm paths")
    arm = emitted[full]
    dependencies = [full, builder.__file__, builder.SOURCE]
    with fit._stable_files(dependencies):
        raw = open(full, "rb").read()
        source_text = raw.decode("utf-8")
        candidate_sha = hashlib.sha256(raw).hexdigest()
        constants = fit._generated_constants(source_text, full)
        expected_keys = {
            "_K4_ACTIVE", "_K4_CONSUMERS", "_K4_BIRTHS", "_K4_LAM"}
        if set(constants) != expected_keys:
            raise ValueError(
                "candidate does not contain the four literal arm constants")
        if constants["_K4_ACTIVE"] != (
                constants["_K4_CONSUMERS"] or constants["_K4_BIRTHS"]):
            raise ValueError("candidate master switch is inconsistent")
        template, template_sha = builder._source()
        expected = builder.build(*arm, template, template_sha,
                                 builder.DEFAULT_LAM)
        if source_text != expected:
            raise ValueError("candidate is not an exact registered regeneration")
        generator_sha = fit._sha256(builder.__file__)

        predictions, operation_logs = [], []
        spec = le.LifecycleSpec(
            width=width, depth=depth, residual_wall_time_limit_s=None)
        rows, provenance = le.evaluate(
            full, [int(seed)], np.zeros((1, width), dtype=np.float32), spec=spec,
            gt_noise=[0.0], predictions_out=predictions,
            operation_logs_out=operation_logs)
        if len(rows) != 1 or len(predictions) != 1 or len(operation_logs) != 1:
            raise RuntimeError(
                "one-row lifecycle did not return exactly one result")
        row = rows[0]
        if (row.candidate_sha256 != candidate_sha
                or provenance.get("candidate_sha256") != candidate_sha
                or fit._sha256(full) != candidate_sha
                or fit._sha256(builder.__file__) != generator_sha
                or fit._sha256(builder.SOURCE) != template_sha):
            raise RuntimeError("candidate or generator bytes changed during gate")

        groups = collections.defaultdict(lambda: {"calls": 0, "flops": 0})
        for rec in operation_logs[0]:
            key = (str(rec.op_name), str(rec.resolved_dtype),
                   "x".join("(" + ",".join(str(d) for d in shape) + ")"
                            for shape in rec.shapes))
            groups[key]["calls"] += 1
            groups[key]["flops"] += int(rec.flop_cost)
        operation_groups = [
            {"op": key[0], "dtype": key[1], "shapes": key[2], **value}
            for key, value in groups.items()]
        operation_groups.sort(
            key=lambda item: (-item["flops"], -item["calls"], item["op"],
                              item["shapes"]))
        group_flops = sum(item["flops"] for item in operation_groups)
        group_calls = sum(item["calls"] for item in operation_groups)
        prediction = np.asarray(predictions[0], dtype=np.float32)
        payload = {
        "schema": "arc-whest.compact-k4-arm-label-free-gate.v1",
        "labels_read": False,
        "candidate": os.path.relpath(full, _ROOT).replace("\\", "/"),
        "candidate_sha256": candidate_sha,
        "arm_constants": constants,
        "source_binding": {
            "exact_registered_regeneration": True,
            "generator": {"path": os.path.relpath(
                builder.__file__, _ROOT).replace("\\", "/"),
                "sha256": generator_sha},
            "template": {"path": os.path.relpath(
                builder.SOURCE, _ROOT).replace("\\", "/"),
                "sha256": template_sha},
            "lifecycle_candidate_sha256": provenance["candidate_sha256"],
            "closing_candidate_sha256": fit._sha256(full),
        },
        "identity": {"kind": "synthetic", "seed": int(seed),
                     "width": int(width), "depth": int(depth)},
        "lifecycle_row": dataclasses.asdict(row),
        "lifecycle_provenance": provenance,
        "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "operation_groups": operation_groups,
        "operation_group_parity": {
            "group_flops": group_flops,
            "context_flops": int(row.flops_used),
            "flops_exact": group_flops == int(row.flops_used),
            "group_calls": group_calls,
            "reported_calls": int(row.n_ops),
            "calls_exact": group_calls == int(row.n_ops),
        },
        "scope": ("label-free canonical local lifecycle with residual cap disabled; "
                  "not server residual or accuracy evidence"),
    }
        if (row.failed or not payload["operation_group_parity"]["flops_exact"]
                or not payload["operation_group_parity"]["calls_exact"]
                or prediction.shape != (width,)
                or not np.isfinite(prediction).all()):
            raise RuntimeError(
                "compact arm failed lifecycle, output, or operation parity")
        if out:
            with open(out, "x", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True,
                          allow_nan=False)
                fh.write("\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=40_910_007_000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.out):
        raise ValueError("output already exists: %s" % args.out)
    result = run(args.candidate, seed=args.seed, width=args.width,
                 depth=args.depth, out=args.out)
    print("PASS %s: %d FLOPs, %d operations" %
          (result["candidate"],
           result["operation_group_parity"]["context_flops"],
           result["operation_group_parity"]["reported_calls"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
