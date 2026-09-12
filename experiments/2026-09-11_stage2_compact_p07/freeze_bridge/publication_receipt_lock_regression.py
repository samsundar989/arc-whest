"""Sacrificial regression for receipt disk replacement during locked parse."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "local_sim"))
import p2_lab as lab  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="p2-receipt-race-", dir=ROOT) as td:
        work = Path(td)
        arms = []
        for name in ("exact", "coherent", "consumers_only", "births_only"):
            arm = work / ("arm_" + name + ".py")
            arm.write_text("# sacrificial " + name + "\n", encoding="utf-8")
            arms.append({"name": name, "path": rel(arm), "sha256": sha(arm)})
        fit = work / "fit.json"
        fit.write_text("{}\n", encoding="utf-8")
        input_path = work / "input.txt"
        input_path.write_text("registered input\n", encoding="utf-8")
        truth = work / "truth.txt"
        truth.write_text("sacrificial truth\n", encoding="utf-8")
        frozen = {
            "arms": arms,
            "source_bindings": {},
            "input_bindings": {"registered": {"path": rel(input_path),
                                               "sha256": sha(input_path)}},
        }
        receipt = work / "freeze.json"
        receipt.write_text(json.dumps(frozen), encoding="utf-8")
        final = work / "final.json"
        stage = work / "stage.json"
        report = {
            "status": "pending",
            "arms": {row["name"].replace("_", "-"):
                     {"candidate_sha256": row["sha256"]} for row in arms},
            "k4_freeze": {
                "locked_cli_source_bindings": {
                    "p2_lab": {"path": "local_sim/p2_lab.py", "sha256": sha(lab.__file__)}},
                "instrument_bindings": {
                    "locked_bootstrap": {"path": "local_sim/p2_locked_bootstrap.py",
                                         "sha256": sha(lab._BOOTSTRAP)}},
                "fit_receipt": {"path": rel(fit), "sha256": sha(fit)},
                "freeze_receipt": {"path": rel(receipt), "sha256": sha(receipt)},
                "protected_truth": {
                    "path": rel(truth), "sha256": sha(truth),
                    "reservation_receipt": {"result_artifact": rel(final)},
                },
            },
        }
        stage.write_text(json.dumps(report), encoding="utf-8")
        input_path.write_text("changed after staged receipt\n", encoding="utf-8")
        baseline_refused = False
        try:
            lab._publish_audited_stage(str(stage), str(final), sha(lab.__file__))
        except ValueError as exc:
            baseline_refused = "report binding changed" in str(exc) and not final.exists()

        actual_loads = json.loads
        original_bytes = receipt.read_bytes()
        transient = dict(frozen, input_bindings={})
        attempted_swap = []

        def attempted_parse_swap(raw, *a, **kw):
            if raw == original_bytes:
                denied = False
                try:
                    receipt.write_text(json.dumps(transient), encoding="utf-8")
                except OSError:
                    denied = True
                finally:
                    if not denied:
                        receipt.write_bytes(original_bytes)
                attempted_swap.append(denied)
            return actual_loads(raw, *a, **kw)

        with patch.object(lab.json, "loads", attempted_parse_swap):
            try:
                lab._publish_audited_stage(str(stage), str(final), sha(lab.__file__))
                guarded = False
            except ValueError as exc:
                guarded = "report binding changed" in str(exc) and not final.exists()
        result = {
            "schema": "arc-whest.publication-receipt-lock-regression.v1",
            "scope": "disposable receipt, arms, input and truth; no protected arrays opened",
            "lab_sha256": sha(lab.__file__),
            "bootstrap_sha256": sha(lab._BOOTSTRAP),
            "baseline_changed_input_refused": baseline_refused,
            "receipt_swap_attempted_during_parse": len(attempted_swap) == 1,
            "receipt_swap_denied_by_lock_on_windows": attempted_swap == [True],
            "changed_registered_input_refused_after_attempt": guarded,
            "complete_output_absent": not final.exists(),
            "receipt_on_disk_unchanged": sha(receipt) == report["k4_freeze"]["freeze_receipt"]["sha256"],
            "registered_input": rel(input_path),
        }
    Path(args.output).open("x", encoding="utf-8", newline="\n").write(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all((result["baseline_changed_input_refused"],
                     result["receipt_swap_attempted_during_parse"],
                     result["receipt_swap_denied_by_lock_on_windows"],
                     result["changed_registered_input_refused_after_attempt"],
                     result["complete_output_absent"],
                     result["receipt_on_disk_unchanged"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
