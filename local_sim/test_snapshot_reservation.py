"""Sacrificial host register shared by two copied worker snapshots."""

import tempfile
import json
import subprocess
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import truth_reads as tr

SNAPSHOT_ID = "a" * 40


def refused(fn):
    try:
        fn()
    except (tr.ReadRefused, FileNotFoundError, ValueError):
        return True
    return False


with tempfile.TemporaryDirectory(prefix="arc_whest_snapshot_test_") as temp:
    root = Path(temp)
    register = root / "host_reads.jsonl"
    ledger = root / "host_ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    host_truth = root / "truth_P08.npz"
    host_truth.write_bytes(b"sacrificial truth bytes; never loaded")
    host_instrument = root / "reader.py"
    host_instrument.write_bytes(b"same reader bytes")
    token = tr.prepare_snapshot_reservation(
        "P08", str(host_truth), consumer="test", purpose="two snapshots",
        instrument=str(host_instrument), snapshot_id=SNAPSHOT_ID,
        path=str(register),
        ledger_path=str(ledger))
    assert tr.spent("P08", str(register)) == 1
    assert refused(lambda: tr.prepare_snapshot_reservation(
        "P08", str(host_truth), consumer="second", purpose="duplicate",
        instrument=str(host_instrument), snapshot_id=SNAPSHOT_ID,
        path=str(register),
        ledger_path=str(ledger)))
    snapshots = []
    for n in range(2):
        snap = root / f"snapshot{n}"
        snap.mkdir()
        truth = snap / "truth_P08.npz"
        reader = snap / "reader.py"
        truth.write_bytes(host_truth.read_bytes())
        reader.write_bytes(host_instrument.read_bytes())
        snapshots.append((truth, reader))
    bad_truth = root / "truth_P08_changed.npz"
    bad_truth.write_bytes(b"different")
    # A context manager must fail on entry before exposing labels.
    try:
        with tr.consume_snapshot_reservation(
                token, "P08", str(bad_truth), instrument=str(host_instrument),
                consumer="test", purpose="two snapshots", snapshot_id=SNAPSHOT_ID,
                path=str(register), ledger_path=str(ledger)):
            raise AssertionError("changed truth exposed")
    except tr.ReadRefused:
        pass
    for changed in (
            {"consumer": "other consumer"},
            {"purpose": "other purpose"},
            {"snapshot_id": "b" * 40}):
        fields = {"consumer": "test", "purpose": "two snapshots",
                  "snapshot_id": SNAPSHOT_ID}
        fields.update(changed)
        try:
            with tr.consume_snapshot_reservation(
                    token, "P08", str(host_truth),
                    instrument=str(host_instrument), path=str(register),
                    ledger_path=str(ledger), **fields):
                raise AssertionError("changed token identity exposed truth")
        except tr.ReadRefused:
            pass
    copied_register = root / "snapshot_register.jsonl"
    copied_ledger = root / "snapshot_ledger.jsonl"
    copied_register.write_bytes(register.read_bytes())
    copied_ledger.write_bytes(ledger.read_bytes())
    try:
        with tr.consume_snapshot_reservation(
                token, "P08", str(host_truth),
                instrument=str(host_instrument), consumer="test",
                purpose="two snapshots", snapshot_id=SNAPSHOT_ID,
                path=str(copied_register), ledger_path=str(copied_ledger)):
            raise AssertionError("copied snapshot-local authority exposed truth")
    except tr.ReadRefused:
        pass
    try:
        with tr.consume_snapshot_reservation(
                "missing-token", "P08", str(host_truth),
                instrument=str(host_instrument), path=str(register),
                consumer="test", purpose="two snapshots", snapshot_id=SNAPSHOT_ID,
                ledger_path=str(ledger)):
            raise AssertionError("missing token exposed truth")
    except tr.ReadRefused:
        pass
    bad_reader = root / "other_reader.py"
    bad_reader.write_bytes(b"different reader")
    try:
        with tr.consume_snapshot_reservation(
                token, "P08", str(host_truth), instrument=str(bad_reader),
                consumer="test", purpose="two snapshots", snapshot_id=SNAPSHOT_ID,
                path=str(register), ledger_path=str(ledger)):
            raise AssertionError("changed instrument exposed truth")
    except tr.ReadRefused:
        pass
    barrier = Barrier(2)

    def worker(index):
        truth, reader = snapshots[index]
        barrier.wait()
        try:
            with tr.consume_snapshot_reservation(
                    token, "P08", str(truth), instrument=str(reader),
                    consumer="test", purpose="two snapshots", snapshot_id=SNAPSHOT_ID,
                    path=str(register), ledger_path=str(ledger)) as exposed:
                assert Path(exposed) == truth
                return "accepted"
        except tr.ReadRefused:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(worker, (0, 1)))
    assert sorted(outcomes) == ["accepted", "refused"], outcomes
    assert tr.spent("P08", str(register)) == 1
    assert [r.status for r in tr.reads(str(register))].count("activated") == 1
    assert [r.status for r in tr.reads(str(register))].count("completed") == 1
    assert refused(lambda: tr.prepare_snapshot_reservation(
        "P08", str(host_truth), consumer="third", purpose="after completion",
        instrument=str(host_instrument), snapshot_id=SNAPSHOT_ID,
        path=str(register),
        ledger_path=str(ledger)))

    # A second sacrificial authority demonstrates that an exception after
    # activation records abort and still consumes the one look.
    aborted_register = root / "aborted_reads.jsonl"
    aborted_ledger = root / "aborted_ledger.jsonl"
    aborted_ledger.write_text("", encoding="utf-8")
    aborted_token = tr.prepare_snapshot_reservation(
        "P08", str(host_truth), consumer="aborting worker", purpose="failure",
        instrument=str(host_instrument), snapshot_id=SNAPSHOT_ID,
        path=str(aborted_register),
        ledger_path=str(aborted_ledger))
    try:
        with tr.consume_snapshot_reservation(
                aborted_token, "P08", str(host_truth),
                instrument=str(host_instrument), path=str(aborted_register),
                consumer="aborting worker", purpose="failure",
                snapshot_id=SNAPSHOT_ID,
                ledger_path=str(aborted_ledger)):
            raise RuntimeError("simulated worker failure")
    except RuntimeError:
        pass
    assert tr.spent("P08", str(aborted_register)) == 1
    assert [r.status for r in tr.reads(str(aborted_register))][-1] == "aborted"

    cli_register = root / "cli_reads.jsonl"
    cli_ledger = root / "cli_ledger.jsonl"
    cli_ledger.write_text("", encoding="utf-8")
    cli = subprocess.run([
        sys.executable, str(Path(tr.__file__)), "--prepare-snapshot",
        "--block", "P08", "--truth", str(host_truth),
        "--instrument", str(host_instrument), "--consumer", "cli test",
        "--purpose", "no-label preparation", "--snapshot-id", SNAPSHOT_ID,
        "--authority-register",
        str(cli_register), "--authority-ledger", str(cli_ledger)],
        text=True, capture_output=True, check=True)
    cli_data = json.loads(cli.stdout)
    assert cli_data["block"] == "P08" and cli_data["reservation"]
    assert tr.spent("P08", str(cli_register)) == 1

    # The production loader must activate before np.load, deriving the worker
    # snapshot ID from its own checkout rather than accepting a caller string.
    loader_register = root / "loader_reads.jsonl"
    loader_ledger = root / "loader_ledger.jsonl"
    loader_ledger.write_text("", encoding="utf-8")
    loader_truth = root / "loader_truth_P08.npz"
    np.savez(loader_truth, tiny=np.array([1, 2], dtype=np.int32))
    current_head = subprocess.check_output(
        ["git", "-C", tr._ROOT, "rev-parse", "HEAD"], text=True).strip()
    loader_token = tr.prepare_snapshot_reservation(
        "P08", str(loader_truth), consumer="loader", purpose="one real load",
        instrument=str(host_instrument), snapshot_id=current_head,
        path=str(loader_register), ledger_path=str(loader_ledger))
    try:
        tr.load_snapshot_truth("missing", "P08", str(loader_truth),
                               instrument=str(host_instrument), consumer="loader",
                               purpose="one real load", path=str(loader_register),
                               ledger_path=str(loader_ledger))
        raise AssertionError("unreserved loader opened truth")
    except tr.ReadRefused:
        pass
    loaded = tr.load_snapshot_truth(
        loader_token, "P08", str(loader_truth),
        instrument=str(host_instrument), consumer="loader",
        purpose="one real load", path=str(loader_register),
        ledger_path=str(loader_ledger))
    assert loaded["tiny"].tolist() == [1, 2]
    assert tr.spent("P08", str(loader_register)) == 1

print("snapshot reservation: host pre-reserve; two-snapshot race, bytes, purpose, consumer, snapshot, copied authority and unreserved loader refused")
