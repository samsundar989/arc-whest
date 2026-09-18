"""Focused same-run label receipt refusal checks; no estimator or truth run."""

import json
import os
import tempfile
from pathlib import Path

import test_enforcement_labels as labels


def refused():
    try:
        labels.measure("official_config")
    except ValueError:
        return True
    return False


with tempfile.TemporaryDirectory(prefix="arc_whest_receipt_test_") as root:
    os.environ["ARC_WHEST_LABEL_RECEIPTS"] = root
    receipt = Path(root) / "official_config.json"
    assert refused(), "missing receipt accepted"
    assert labels.emit_receipt("official_config") == 0
    assert labels.measure("official_config") == (96, 96, 0)
    # The live ledger is never modified: a sacrificial copy exercises the
    # dependency identity that the same-run receipt binds.
    ledger = Path(root) / "ledger.jsonl"
    ledger.write_bytes((Path(labels._ROOT) / "experiments" /
                        "seed_ledger.jsonl").read_bytes())
    before = labels._receipt_identity("seed_ledger_refusals", ledger)
    with ledger.open("ab") as fh:
        fh.write(b"\n")
    assert labels._receipt_identity("seed_ledger_refusals", ledger) != before
    original = receipt.read_text(encoding="utf-8")
    data = json.loads(original)
    data["passed"] = 1
    receipt.write_text(json.dumps(data), encoding="utf-8")
    assert refused(), "tampered count accepted"
    receipt.write_text(original, encoding="utf-8")
    data = json.loads(original)
    data["identity"] = "0" * 64
    receipt.write_text(json.dumps(data), encoding="utf-8")
    assert refused(), "wrong source identity accepted"
    receipt.write_text(original, encoding="utf-8")
    assert labels.measure("official_config") == (96, 96, 0)

print("preflight receipt: missing, count tamper and source mismatch refused")
