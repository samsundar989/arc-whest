"""Sacrificial host register shared by two copied worker snapshots."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import truth_reads as tr


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
        instrument=str(host_instrument), path=str(register),
        ledger_path=str(ledger))
    assert tr.spent("P08", str(register)) == 1
    assert refused(lambda: tr.prepare_snapshot_reservation(
        "P08", str(host_truth), consumer="second", purpose="duplicate",
        instrument=str(host_instrument), path=str(register),
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
                path=str(register), ledger_path=str(ledger)):
            raise AssertionError("changed truth exposed")
    except tr.ReadRefused:
        pass
    try:
        with tr.consume_snapshot_reservation(
                "missing-token", "P08", str(host_truth),
                instrument=str(host_instrument), path=str(register),
                ledger_path=str(ledger)):
            raise AssertionError("missing token exposed truth")
    except tr.ReadRefused:
        pass
    bad_reader = root / "other_reader.py"
    bad_reader.write_bytes(b"different reader")
    try:
        with tr.consume_snapshot_reservation(
                token, "P08", str(host_truth), instrument=str(bad_reader),
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
        instrument=str(host_instrument), path=str(register),
        ledger_path=str(ledger)))

    # A second sacrificial authority demonstrates that an exception after
    # activation records abort and still consumes the one look.
    aborted_register = root / "aborted_reads.jsonl"
    aborted_ledger = root / "aborted_ledger.jsonl"
    aborted_ledger.write_text("", encoding="utf-8")
    aborted_token = tr.prepare_snapshot_reservation(
        "P08", str(host_truth), consumer="aborting worker", purpose="failure",
        instrument=str(host_instrument), path=str(aborted_register),
        ledger_path=str(aborted_ledger))
    try:
        with tr.consume_snapshot_reservation(
                aborted_token, "P08", str(host_truth),
                instrument=str(host_instrument), path=str(aborted_register),
                ledger_path=str(aborted_ledger)):
            raise RuntimeError("simulated worker failure")
    except RuntimeError:
        pass
    assert tr.spent("P08", str(aborted_register)) == 1
    assert [r.status for r in tr.reads(str(aborted_register))][-1] == "aborted"

print("snapshot reservation: host pre-reserve, two-snapshot race, byte swap and repeat refused")
