"""Focused positive, negative, incomplete and mutation fixtures for the freezer."""
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from argparse import Namespace

import p2_freeze_k4_arms as freezer


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REGISTRY_DIR = os.path.join(_ROOT, "experiments", "2026-09-10_stage2_k4_design")


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _copy_json(source, target):
    with open(source, encoding="utf-8") as fh:
        value = json.load(fh)
    _write_json(target, value)
    return value


def _binding(path):
    return {"path": freezer._relative(path), "sha256": freezer._read_bytes(path, "fixture")[1]}


def _fixture(root, *, selected=1):
    candidates_path = os.path.join(root, "candidates.json")
    bounds_path = os.path.join(root, "bounds.json")
    identities_path = os.path.join(root, "identities.json")
    candidates_doc = _copy_json(os.path.join(_REGISTRY_DIR, "candidates.json"), candidates_path)
    bounds_doc = _copy_json(os.path.join(_REGISTRY_DIR, "bounds.json"), bounds_path)
    identities_doc = _copy_json(os.path.join(_REGISTRY_DIR, "identities.json"), identities_path)
    candidates = [{"lam": float(row["lam"])} for row in candidates_doc["candidates"]]

    predictions = os.path.join(root, "predictions.npz")
    producer_receipt = os.path.join(root, "producer_receipt.json")
    with open(predictions, "wb") as fh:
        fh.write(b"truth-free synthetic prediction binding\n")
    _write_json(producer_receipt, {"fixture": True})

    generator_raw, generator_sha = freezer._read_bytes(freezer._GENERATOR, "generator")
    template_raw, template_sha = freezer._read_bytes(freezer._TEMPLATE, "template")
    generator = freezer._load_generator(generator_raw)
    coherent = next(row for row in generator["ARMS"] if row[0] == "coherent")
    candidate_sources = []
    for index, candidate in enumerate(candidates):
        raw = generator["build"](*coherent, template_raw.decode("utf-8"),
                                 template_sha, candidate["lam"]).encode("utf-8")
        path = os.path.join(root, "candidate_%03d_coherent.py" % index)
        with open(path, "wb") as fh:
            fh.write(raw)
        candidate_sources.append({"path": freezer._relative(path),
                                  "sha256": freezer._sha256_bytes(raw),
                                  "verified": True})

    source_paths = {"fit_interface": freezer._FIT,
                    "fit_producer": freezer._PRODUCER,
                    "arm_generator": freezer._GENERATOR,
                    "arm_template": freezer._TEMPLATE}
    source_paths.update({"arm_" + name: os.path.join(freezer._DEFAULT_ARMS,
                                                      "arm_%s.py" % name)
                         for name in freezer._ARMS})
    sources = {name: _binding(path) for name, path in source_paths.items()}
    fit_path = os.path.join(root, "fit_receipt.json")
    truth_path = os.path.join(_ROOT, "experiments", "2026-09-10_slice5", "truth_P07.npz")
    truth_hash = "a" * 64  # compared as metadata; the freezer must never open truth.
    objectives = [4.0, 1.0, 2.0, 3.0]
    if selected != 1:
        objectives[selected] = 0.5
        objectives[1] = 1.0
    ids = [row["id"] for row in identities_doc["identities"]]
    reservation = {
        "block": "P07", "artifact": freezer._relative(truth_path),
        "artifact_sha256": truth_hash, "consumer": "local_sim/p2_k4_fit.py",
        "instrument_sha256": sources["fit_interface"]["sha256"],
        "purpose": "bounded coherent-K4 coefficient fit",
        "ts": "2026-09-11T00:00:00Z", "status": "completed",
        "reservation": "0123456789abcdef", "result_artifact": freezer._relative(fit_path),
        "note": "synthetic receipt fixture; no truth opened", "backfilled": False,
        "timestamp_source": "automatic_runtime",
        "receipt_created_utc": "2026-09-11T00:00:00Z",
        "receipt_completed_utc": "2026-09-11T00:00:01Z", "elapsed_seconds": 1.0}
    ledger = os.path.join(root, "truth_reads.jsonl")
    freezer._TRUTH_READS_LEDGER = ledger
    reserved = copy.deepcopy(reservation)
    reserved["status"] = "reserved"
    reserved["receipt_completed_utc"] = ""
    reserved["elapsed_seconds"] = None
    with open(ledger, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(reserved, sort_keys=True) + "\n")
        fh.write(json.dumps(reservation, sort_keys=True) + "\n")
    receipt = {
        "schema": freezer._FIT_SCHEMA, "immutable": True,
        "data_role": "protected-design",
        "objective": {"name": "final_layer_mse", "direction": "minimize",
                      "definition": "mean identity final-layer neuron MSE",
                      "validity": "registered coefficient candidates have identical counted cost for every identity"},
        "parameter_names": ["lam"],
        "generator_binding": {"lam": "generated arm module constant _K4_LAM",
                              "arms": list(freezer._ARMS)},
        "candidate_registry": candidates,
        "bounds": {"lam": {"min": float(bounds_doc["lam"]["min"]),
                            "max": float(bounds_doc["lam"]["max"])}},
        "identities": ids,
        "truth_manifest": {"kind": "protected-design", "block": "P07",
                           "replicas": 2, "n_samples": 4096},
        "reservation_receipt": reservation,
        "inputs": {"predictions": _binding(predictions),
                   "producer_receipt": _binding(producer_receipt),
                   "truth": {"path": freezer._relative(truth_path), "sha256": truth_hash},
                   "candidates": _binding(candidates_path), "bounds": _binding(bounds_path),
                   "identities": _binding(identities_path)},
        "source_hashes": sources, "candidate_sources": candidate_sources,
        "counted_costs": [[818_515_845_672] * len(ids) for _ in candidates],
        "results": [{"candidate": candidate, "objective": objectives[index],
                     "per_identity": [objectives[index]] * len(ids)}
                    for index, candidate in enumerate(candidates)],
        "selection": {"registered_index": selected,
                      "frozen_values": candidates[selected],
                      "objective": objectives[selected],
                      "tie_break": "first in explicitly registered order"},
        "validation": {"completed_before_truth_load": True,
                       "producer_receipt_revalidated_before_truth_load": True,
                       "candidate_count": len(candidates), "identity_count": len(ids),
                       "all_finite": True}}
    _write_json(fit_path, receipt)
    return fit_path, receipt


def _temp(prefix):
    return tempfile.TemporaryDirectory(prefix=prefix, dir=_ROOT)


def run_checks():
    rows = []

    def check(name, condition, detail):
        rows.append({"check": name, "pass": bool(condition), "detail": detail})
        print("%-58s %s  %s" % (name, "PASS" if condition else "FAIL", detail))

    with _temp("p2-freeze-positive-") as root:
        fit_path, _ = _fixture(root, selected=1)
        out_dir = os.path.join(root, "frozen")
        made = freezer.freeze(Namespace(fit_receipt=fit_path, out_dir=out_dir))
        receipt_path = os.path.join(out_dir, "freeze_receipt.json")
        checked = freezer.validate_freeze_receipt(receipt_path)
        check("nondefault P07 selection freezes four exact arms",
              made["selection"]["k4_lam"] == 1.0 / 6.0 and
              checked["k4_lam"] == 1.0 / 6.0 and len(checked["arm_paths"]) == 4,
              "selected index 1, lam=1/6, four strict regenerations")
        with freezer.hold_freeze_receipt(receipt_path) as held:
            closing = freezer.revalidate_held_freeze(held)
            active = closing["receipt_sha256"] == held["receipt_sha256"]
        try:
            freezer.revalidate_held_freeze(held)
            outside_refused = False
        except freezer.FreezeRefused:
            outside_refused = True
        check("held validator covers execution and closing publication",
              active and outside_refused,
              "closing check works only inside complete dependency lock")
        check("freeze receipt is published after its arm files",
              all(os.path.getmtime(path) <= os.path.getmtime(receipt_path)
                  for path in checked["arm_paths"]),
              "all arm mtimes precede/equal receipt mtime")
        before = open(receipt_path, "rb").read()
        try:
            freezer.freeze(Namespace(fit_receipt=fit_path, out_dir=out_dir))
            refused = False
        except freezer.FreezeRefused as exc:
            refused = "already exists" in str(exc)
        check("existing output directory is refused immutably",
              refused and open(receipt_path, "rb").read() == before,
              "exclusive fresh-directory contract")
        with open(checked["arm_paths"][2], "ab") as fh:
            fh.write(b"\n# mutation\n")
        try:
            freezer.validate_freeze_receipt(receipt_path)
            refused = False
        except freezer.FreezeRefused as exc:
            refused = "stale, mixed, or misbound" in str(exc)
        check("post-freeze arm mutation is detected", refused,
              "strict bytes and exact regeneration")

    mutations = (
        ("wrong fit role", lambda r: r.__setitem__("data_role", "synthetic"),
         "protected-design"),
        ("selected value mismatch", lambda r: r["selection"].__setitem__(
            "frozen_values", {"lam": 0.5}), "selected index/value/objective"),
        ("objective/per-identity mismatch", lambda r: (
            r["results"][1].__setitem__("objective", 1.1),
            r["selection"].__setitem__("objective", 1.1)), "arithmetic mean"),
        ("wrong objective", lambda r: r["objective"].__setitem__("name", "mse"),
         "objective"),
        ("missing reservation", lambda r: r.__setitem__("reservation_receipt", None),
         "reservation receipt"),
        ("incomplete reservation", lambda r: r["reservation_receipt"].__setitem__(
            "status", "reserved"), "completed P07 reservation"),
        ("forged embedded reservation", lambda r: r["reservation_receipt"].__setitem__(
            "reservation", "fedcba9876543210"), "live truth-read ledger"),
        ("wrong identity binding", lambda r: r["inputs"]["identities"].__setitem__(
            "sha256", "0" * 64), "identities hash mismatch"),
        ("stale source binding", lambda r: r["source_hashes"]["arm_generator"].__setitem__(
            "sha256", "0" * 64), "source arm_generator"),
    )
    for label, mutate, expected in mutations:
        with _temp("p2-freeze-negative-") as root:
            fit_path, receipt = _fixture(root)
            mutate(receipt)
            _write_json(fit_path, receipt)
            out_dir = os.path.join(root, "must-not-exist")
            try:
                freezer.freeze(Namespace(fit_receipt=fit_path, out_dir=out_dir))
                refused, detail = False, "accepted"
            except freezer.FreezeRefused as exc:
                refused, detail = expected in str(exc), str(exc)
            check("freezer refuses %s" % label,
                  refused and not os.path.exists(out_dir), detail)

    with _temp("p2-freeze-candidate-mutation-") as root:
        fit_path, receipt = _fixture(root)
        source = freezer._repo_path(receipt["candidate_sources"][0]["path"], "fixture")
        with open(source, "ab") as fh:
            fh.write(b"\n# changed after fit\n")
        out_dir = os.path.join(root, "must-not-exist")
        try:
            freezer.freeze(Namespace(fit_receipt=fit_path, out_dir=out_dir))
            refused = False
        except freezer.FreezeRefused as exc:
            refused = "exact registered regeneration" in str(exc)
        check("candidate mutation refuses without partial freeze",
              refused and not os.path.exists(out_dir), "pre-publication source check")

    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    rows = run_checks()
    result = {"suite": "p2-freeze-k4-arms", "passed": sum(r["pass"] for r in rows),
              "total": len(rows), "checks": rows}
    if args.output:
        _write_json(args.output, result)
    print("\n%d/%d checks passed" % (result["passed"], result["total"]))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
