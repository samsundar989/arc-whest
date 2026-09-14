"""Exclusive, fsynced scorer evidence; diagnostic recovery, never permission to resume.

Each row is appended and synced before evaluation continues. A tiny pipe guardian
publishes INTERRUPTED if the scorer dies without a terminal manifest. Machine or
guardian death can still leave no terminal: verification refuses that state.
Publication is a synced temporary file plus an exclusive atomic hard link (NTFS
and POSIX), never overwrite/replace. Hardware power-loss durability of directory
entries is filesystem-dependent; this is process-crash safety, not a backup.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _safe(value):
    # Missing telemetry must survive a failed row; do not emit nonstandard JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _atomic_new(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with tmp.open("xb") as stream:
        stream.write(_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)  # fails if target exists, including a competing writer
    finally:
        tmp.unlink()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scan(directory):
    """Preserve every intact prefix row; a torn/corrupt tail is never COMPLETE."""
    directory = Path(directory)
    header = _read(directory / "header.json")
    previous = _digest(header)
    records = []
    error = None
    with (directory / "rows.jsonl").open("rb") as stream:
        for line in stream:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("torn journal record")
                record = json.loads(line)
                digest = record.pop("sha256")
                if record["previous"] != previous or _digest(record) != digest:
                    raise ValueError("journal record hash/chain mismatch")
                previous = digest
                records.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                break
    return header, records, previous, error


def _snapshot(directory, status, reason, report=None):
    header, records, chain, error = _scan(directory)
    rows = [r["data"] for r in records if r["kind"] == "row"]
    return {
        "schema": 1, "status": status, "reason": reason,
        "header_sha256": _digest(header), "chain_sha256": chain,
        "binding_sha256": sha_file(Path(directory) / "binding.json")
        if (Path(directory) / "binding.json").exists() else None,
        "expected_rows": header["expected_rows"], "completed_rows": len(rows),
        "row_identities": [[r["arm"], r["row"]["input_seed"]] for r in rows],
        "failed_rows": [r for r in rows if r["failure_categories"]],
        "events": [r["data"] for r in records if r["kind"] == "event"],
        "journal_error": error, "report": report,
        "resume_authorized": False, "protected_reread_authorized": False,
        "complete_is_not_promotion_or_deployment_certification": True,
    }


def _guardian(directory):
    sys.stdin.buffer.read()  # EOF also arrives after abrupt parent process death
    terminal = Path(directory) / "manifest.json"
    if not terminal.exists():
        try:
            _atomic_new(terminal, _snapshot(directory, "INTERRUPTED",
                        "writer disappeared before terminal publication"))
        except FileExistsError:
            pass


def failure_categories(row):
    flags = ("budget_exhausted", "time_exhausted", "residual_exhausted",
             "combined_exhausted", "nonfinite")
    categories = [key for key in flags if row.get(key)]
    if row.get("error_code"):
        categories.append("estimator_error:" + row["error_code"])
    return categories


class EvidenceJournal:
    def __init__(self, output, *, candidates, instrument_files, truth_files,
                 expected_per_arm, config):
        self.output = Path(output).resolve()
        self.directory = Path(str(self.output) + ".journal")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(self.output):
            raise FileExistsError(self.output)
        self.directory.mkdir()  # exclusive reservation before any truth read
        self.guardian = None
        self.header = {
            "schema": 1, "output": str(self.output),
            "candidates": candidates,
            "instrument_hashes": {str(Path(p).resolve()): sha_file(p)
                                  for p in instrument_files},
            "truth_files": truth_files, "config": config,
            "expected_per_arm": expected_per_arm,
            "expected_rows": expected_per_arm * len(candidates),
        }
        _atomic_new(self.directory / "header.json", self.header)
        with (self.directory / "rows.jsonl").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        self.chain = _digest(self.header)
        self.seen = set()
        self.binding = None
        self.stage = "truth_load"
        self.guardian = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--watch", str(self.directory)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if not (self.directory / "manifest.json").exists():
                status = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED"
                reason = {"stage": self.stage,
                          "exception": exc_type.__name__ if exc_type else "MissingPublication",
                          "message": str(exc) if exc else "run exited without publication"}
                self.finish(status, reason)
        finally:
            self.guardian.stdin.close()
            self.guardian.wait(timeout=15)
        return False

    def _append(self, kind, data):
        if (self.directory / "manifest.json").exists():
            raise ValueError("terminal journal is immutable")
        body = {"previous": self.chain, "kind": kind, "data": _safe(data)}
        digest = _digest(body)
        with (self.directory / "rows.jsonl").open("ab") as stream:
            stream.write(_bytes({**body, "sha256": digest}))
            stream.flush()
            os.fsync(stream.fileno())
        self.chain = digest

    def bind(self, seeds, truth_sha256, truth_provenance):
        if len(seeds) != self.header["expected_per_arm"] or len(set(seeds)) != len(seeds):
            raise ValueError("unexpected or duplicate seed identities")
        files = truth_provenance.get("files", [])
        if len(files) != len(self.header["truth_files"]):
            raise ValueError("truth file identity mismatch")
        for actual, expected in zip(files, self.header["truth_files"]):
            if Path(actual.get("snapshot_path", actual["path"])).resolve() != Path(expected["path"]).resolve():
                raise ValueError("truth file path mismatch")
            digest = actual["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("missing truth file hash")
            if expected.get("sha256") is not None and digest != expected["sha256"]:
                raise ValueError("truth file hash mismatch")
        self.binding = {"seeds": seeds, "truth_sha256": truth_sha256,
                        "truth_provenance": truth_provenance}
        _atomic_new(self.directory / "binding.json", self.binding)

    def event(self, category, **details):
        self._append("event", {"stage": self.stage, "category": category, **details})

    def append_row(self, arm, row):
        row = dataclasses.asdict(row) if dataclasses.is_dataclass(row) else dict(row)
        if self.binding is None:
            raise ValueError("rows require immutable truth/identity binding")
        try:
            _validate_row(self.header, self.binding, arm, row, self.seen)
        except (ValueError, KeyError, TypeError) as exc:
            self.event("rejected_row", arm=arm, row=row, reason=str(exc))
            raise
        self._append("row", {"arm": arm, "row": row,
                             "failure_categories": failure_categories(row)})
        self.seen.add((arm, row["input_seed"]))

    def assert_complete(self):
        _validate_complete(self.directory)
        # Content bindings are also rechecked against disk immediately before
        # inference/publication. Truth files are hashed, never reopened as arrays.
        for path, digest in self.header["instrument_hashes"].items():
            if sha_file(path) != digest:
                raise ValueError("instrument hash mismatch")
        for item in self.header["candidates"].values():
            if sha_file(item["path"]) != item["sha256"]:
                raise ValueError("candidate hash mismatch")

    def finish(self, status, reason, report=None):
        if status not in {"COMPLETE", "FAILED", "INTERRUPTED"}:
            raise ValueError("invalid terminal status")
        if status == "COMPLETE":
            self.assert_complete()
        _atomic_new(self.directory / "manifest.json",
                    _snapshot(self.directory, status, reason, report))

    def publish(self, payload):
        self.stage = "publication"
        self.event("publication_started")
        self.assert_complete()
        _atomic_new(self.output, payload)
        self.finish("COMPLETE", "all expected rows validated and report published",
                    {"path": str(self.output), "sha256": sha_file(self.output)})


def _validate_row(header, binding, arm, row, seen):
    # Identity alone is not evidence: absent flags and unavailable numerical
    # telemetry must never be interpreted as a successful observation.
    required = {
        "index", "input_seed", "estimator_seed", "candidate", "candidate_sha256",
        "truth_sha256", "width", "depth", "budget", "flops_used", "n_ops", "mse",
        "zero_prediction_mse", "gt_noise", "residual_wall_time_s", "wall_time_s",
        "budget_exhausted", "time_exhausted", "residual_exhausted",
        "combined_exhausted", "error_code", "nonfinite",
    }
    if set(row) != required:
        raise ValueError("incomplete or unknown EvidenceRow schema")
    for key in ("budget_exhausted", "time_exhausted", "residual_exhausted",
                "combined_exhausted", "nonfinite"):
        if type(row[key]) is not bool:
            raise ValueError("failure flags require exact booleans")
    for key in ("index", "input_seed", "estimator_seed", "width", "depth"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError("row identity/shape requires nonnegative integers")
    if row['width'] == 0 or row['depth'] == 0:
        raise ValueError("empty row shape")
    if not isinstance(row['candidate'], str) or not isinstance(row['error_code'], str):
        raise ValueError("row candidate/error requires strings")
    if not failure_categories(row):
        for key in ("budget", "flops_used", "n_ops", "mse", "zero_prediction_mse",
                    "gt_noise", "residual_wall_time_s", "wall_time_s"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("successful row requires finite nonnegative " + key)
    if arm not in header["candidates"]:
        raise ValueError("unknown arm")
    seed = row["input_seed"]
    if (arm, seed) in seen:
        raise ValueError("duplicate row identity")
    if seed not in binding["seeds"] or row["index"] != binding["seeds"].index(seed):
        raise ValueError("unexpected row identity/index")
    if row["candidate_sha256"] != header["candidates"][arm]["sha256"]:
        raise ValueError("candidate hash mismatch in row")
    if row["truth_sha256"] != binding["truth_sha256"]:
        raise ValueError("truth hash mismatch in row")


def _validate_complete(directory):
    header, records, chain, error = _scan(directory)
    if error:
        raise ValueError(error)
    binding = _read(Path(directory) / "binding.json")
    seen = set()
    for record in records:
        if record["kind"] == "event":
            if record["data"]["category"] in {
                "rejected_row", "setup_failure", "evaluation_error", "evaluation_interrupted"
            }:
                raise ValueError("failed event prohibits score decision")
            continue
        if record["kind"] != "row":
            raise ValueError("unknown record type")
        data = record["data"]
        row, arm = data["row"], data["arm"]
        _validate_row(header, binding, arm, row, seen)
        if data["failure_categories"] or failure_categories(row):
            raise ValueError("failed row prohibits score decision")
        seen.add((arm, row["input_seed"]))
    expected = {(arm, seed) for arm in header["candidates"] for seed in binding["seeds"]}
    if seen != expected or len(seen) != header["expected_rows"]:
        raise ValueError("incomplete evidence prohibits score decision")
    return header, records, chain


def verify_complete(directory):
    """Consumers must require this, not merely a report file or exit status 0."""
    directory = Path(directory)
    terminal = _read(directory / "manifest.json")
    if terminal["status"] != "COMPLETE":
        raise ValueError("non-COMPLETE evidence prohibits score decision")
    header, records, chain = _validate_complete(directory)
    if terminal != _snapshot(directory, "COMPLETE", terminal["reason"], terminal["report"]):
        raise ValueError("terminal manifest content/hash mismatch")
    report = terminal["report"]
    if not report or report["path"] != header["output"] or sha_file(report["path"]) != report["sha256"]:
        raise ValueError("published report hash mismatch")
    return terminal


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--watch":
        _guardian(sys.argv[2])
    else:
        raise SystemExit("internal guardian: --watch JOURNAL_DIRECTORY")
