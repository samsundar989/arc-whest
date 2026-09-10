"""Synthetic-only regressions for p2_k4_fit's bounded receipt interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace

import numpy as np
from numpy import load as _load_npz

import p2_k4_fit as fit


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, sort_keys=True)
        fh.write("\n")


def _fixture(root):
    candidates = os.path.join(root, "candidates.json")
    bounds = os.path.join(root, "bounds.json")
    predictions = os.path.join(root, "predictions.npz")
    truth = os.path.join(root, "truth.npz")
    receipt = os.path.join(root, "receipt.json")
    registry = {"parameters": ["lam"],
                "candidates": [{"lam": 0.0}, {"lam": 0.25}, {"lam": 0.5}]}
    _write_json(candidates, registry)
    _write_json(bounds, {"lam": {"min": 0.0, "max": 0.5}})
    ids = np.asarray(["synthetic-0", "synthetic-1"])
    target = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10.0
    pred = np.stack([target + 0.3, target, target - 0.1])
    hashes = np.asarray([hashlib.sha256(("candidate-%d" % i).encode()).hexdigest()
                         for i in range(3)])
    np.savez(predictions, parameter_names=np.asarray(["lam"]),
             candidate_values=np.asarray([[0.0], [0.25], [0.5]]),
             identities=ids, predictions=pred,
             counted_costs=np.asarray([202, 202, 202], dtype=np.int64),
             producer_candidate_source_hashes=hashes)
    manifest = {"kind": "synthetic", "fixture": "affine-offset-v1",
                "labels_protected": False, "identity_count": 2}
    np.savez(truth, identities=ids, truth=target,
             manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    args = Namespace(predictions=predictions, truth=truth,
                     data_role="synthetic", candidates=candidates,
                     bounds=bounds, objective="final_layer_mse",
                     receipt=receipt, block="", ts="", note="")
    return args


def run_checks():
    rows = []

    def check(name, condition, detail):
        rows.append({"check": name, "pass": bool(condition), "detail": detail})
        print("%-48s %s  %s" % (name, "PASS" if condition else "FAIL", detail))

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-") as root:
        a = _fixture(root)
        receipt = fit.fit(a)
        check("synthetic optimum is frozen", receipt["selection"]["frozen_values"] == {"lam": 0.25},
              str(receipt["selection"]))
        required = {"source_hashes", "objective", "identities", "truth_manifest",
                    "counted_costs", "selection", "candidate_registry", "bounds"}
        check("receipt carries required provenance", required <= set(receipt),
              ",".join(sorted(required)))
        check("synthetic manifest is explicit",
              receipt["truth_manifest"]["kind"] == "synthetic"
              and receipt["data_role"] == "synthetic"
              and receipt["reservation_receipt"] is None, "synthetic only")
        check("counted costs preserve identities",
              receipt["counted_costs"] == [[202, 202], [202, 202], [202, 202]],
              str(receipt["counted_costs"]))
        before = open(a.receipt, "rb").read()
        refused = False
        try:
            fit.fit(a)
        except fit.FitRefused as exc:
            refused = "already exists" in str(exc)
        check("existing receipt is immutable", refused and open(a.receipt, "rb").read() == before,
              "exclusive create; bytes unchanged")

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-order-") as root:
        a = _fixture(root)
        _write_json(a.bounds, {"lam": {"min": 0.0, "max": 0.1}})
        calls = []
        try:
            fit.fit(a, truth_loader=lambda _a: calls.append(True))
            refused = False
        except fit.FitRefused as exc:
            refused = "outside its bound" in str(exc)
        check("invalid request precedes truth loader", refused and not calls,
              "loader calls=%d" % len(calls))

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-cost-") as root:
        a = _fixture(root)
        with _load_npz(a.predictions, allow_pickle=False) as z:
            values = {name: z[name] for name in z.files}
        values["counted_costs"] = np.asarray([202, 203, 202], dtype=np.int64)
        np.savez(a.predictions, **values)
        calls = []
        try:
            fit.fit(a, truth_loader=lambda _a: calls.append(True))
            refused = False
        except fit.FitRefused as exc:
            refused = "value-independent" in str(exc)
        check("MSE objective refuses unequal costs", refused and not calls,
              "loader calls=%d" % len(calls))

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-role-") as root:
        a = _fixture(root)
        a.data_role, a.block = "protected-design", "P01"
        calls = []
        try:
            fit.fit(a, truth_loader=lambda _a: calls.append(True))
            refused = False
        except fit.FitRefused as exc:
            refused = "exact block P07" in str(exc)
        check("protected fitting requires exact P07", refused and not calls,
              "loader calls=%d" % len(calls))

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-synthetic-role-") as root:
        a = _fixture(root)
        np.savez(a.truth, seeds=np.asarray([1, 2]),
                 means_r1=np.zeros((2, 3, 4)), means_r2=np.zeros((2, 3, 4)))
        try:
            fit.fit(a)
            refused = False
        except fit.FitRefused as exc:
            refused = "declared data role" in str(exc)
        check("synthetic role rejects protected schema", refused
              and not os.path.exists(a.receipt), "truth_reads cannot be bypassed")

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-id-") as root:
        a = _fixture(root)
        calls = []
        def wrong_ids(_a):
            calls.append(True)
            return ["wrong-0", "wrong-1"], np.zeros((2, 3, 4)), {"kind": "synthetic"}
        try:
            fit.fit(a, truth_loader=wrong_ids)
            refused = False
        except fit.FitRefused as exc:
            refused = "identities" in str(exc)
        check("identity mismatch writes no receipt", refused and len(calls) == 1
              and not os.path.exists(a.receipt), "post-load alignment refusal")

    with tempfile.TemporaryDirectory(prefix="p2-k4-fit-cli-") as root:
        a = _fixture(root)
        command = [sys.executable, os.path.join(os.path.dirname(__file__),
                   "p2_k4_fit.py"), "--predictions", a.predictions,
                   "--truth", a.truth, "--data-role", "synthetic",
                   "--candidates", a.candidates, "--bounds", a.bounds,
                   "--objective", a.objective, "--receipt", a.receipt]
        proc = subprocess.run(command, capture_output=True, text=True)
        check("documented synthetic CLI executes", proc.returncode == 0
              and os.path.isfile(a.receipt), "exit=%d receipt_created=%s" %
              (proc.returncode, os.path.isfile(a.receipt)))

    return {"kind": "synthetic coherent-K4 fitting-interface conformance",
            "labels_read": False, "checks": rows,
            "n_checks": len(rows),
            "n_failed": sum(not row["pass"] for row in rows),
            "pass": all(row["pass"] for row in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--fixture-dir", default="",
                    help="exclusively create a durable synthetic fixture and receipt")
    a = ap.parse_args(argv)
    if a.fixture_dir:
        try:
            os.mkdir(a.fixture_dir)
        except FileExistsError:
            raise SystemExit("REFUSED: fixture directory already exists: %s" %
                             a.fixture_dir)
        fixture = _fixture(a.fixture_dir)
        receipt = fit.fit(fixture)
        print("synthetic fixture selected %s" %
              json.dumps(receipt["selection"]["frozen_values"], sort_keys=True))
        print("receipt: %s" % fixture.receipt)
        return 0
    result = run_checks()
    if a.json:
        if os.path.lexists(a.json):
            raise SystemExit("REFUSED: evidence already exists: %s" % a.json)
        with open(a.json, "x", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print("%d/%d passed" % (result["n_checks"] - result["n_failed"],
                            result["n_checks"]))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
