"""Truth-free parent-publication checks against the real immutable freeze receipt.

The staged JSON here is a binding-layer probe, not a scored ladder report.
It is removed with its sacrificial directory and opens no P01/P07 truth arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_lab as lab  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(lab._ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = Path(lab._ROOT) / "experiments/2026-09-11_stage2_compact_p07/frozen_arms_metadata_protocol/freeze_receipt.json"
    frozen = json.loads(receipt.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="p2-publication-bindings-", dir=lab._ROOT) as td:
        root = Path(td)
        truth = root / "synthetic_truth.txt"
        truth.write_text("sacrificial publication binding\n", encoding="utf-8")
        stage = root / "stage.json"
        final = root / "final.json"
        target_sha = sha(lab.__file__)
        report = {
            "status": "complete",
            "arms": {row["name"].replace("_", "-"):
                     {"candidate_sha256": row["sha256"]} for row in frozen["arms"]},
            "k4_freeze": {
                "locked_cli_source_bindings": {
                    "p2_lab": {"path": "local_sim/p2_lab.py", "sha256": target_sha}},
                "instrument_bindings": {
                    "p2_lab": {"path": "local_sim/p2_lab.py", "sha256": target_sha},
                    "locked_bootstrap": {
                        "path": "local_sim/p2_locked_bootstrap.py",
                        "sha256": sha(lab._BOOTSTRAP)}},
                "fit_receipt": frozen["fit_receipt"],
                "freeze_receipt": {"path": rel(receipt), "sha256": sha(receipt)},
                "protected_truth": {
                    "path": rel(truth), "sha256": sha(truth),
                    "reservation_receipt": {"result_artifact": rel(final)}}}}
        stage.write_text(json.dumps(report), encoding="utf-8")
        lab._publish_audited_stage(str(stage), str(final), target_sha)
        positive = final.exists() and final.read_bytes() == stage.read_bytes()
        final.unlink()

        truth.write_text("mutated sacrificial binding\n", encoding="utf-8")
        try:
            lab._publish_audited_stage(str(stage), str(final), target_sha)
            mutation_refused = False
        except ValueError:
            mutation_refused = not final.exists()
        truth.unlink()
        try:
            lab._publish_audited_stage(str(stage), str(final), target_sha)
            missing_refused = False
        except ValueError:
            missing_refused = not final.exists()

        truth.write_text("sacrificial publication binding\n", encoding="utf-8")
        final.write_text("preexisting\n", encoding="utf-8")
        preexisting_sha = sha(final)
        try:
            lab._publish_audited_stage(str(stage), str(final), target_sha)
            preexisting_preserved = False
        except ValueError:
            preexisting_preserved = sha(final) == preexisting_sha

        result = {
            "schema": "arc-whest.p2-publication-receipt-probe.v1",
            "truth": "sacrificial text only; no P01/P07/vault truth arrays opened",
            "staged_payload": "binding-layer probe; no scored outcomes",
            "real_freeze_receipt": rel(receipt),
            "real_freeze_receipt_sha256": sha(receipt),
            "lab_sha256": target_sha,
            "bootstrap_sha256": sha(lab._BOOTSTRAP),
            "positive_real_receipt_binding": positive,
            "closing_truth_mutation_refused_without_output": mutation_refused,
            "missing_truth_binding_refused_without_output": missing_refused,
            "preexisting_output_preserved": preexisting_preserved,
        }
    Path(args.output).open("x", encoding="utf-8", newline="\n").write(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all((positive, mutation_refused, missing_refused,
                     preexisting_preserved)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
