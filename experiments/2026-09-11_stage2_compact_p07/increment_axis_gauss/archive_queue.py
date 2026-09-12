"""Retain and byte-verify terminal live-worktree preflight job 991."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


JOB = 991
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
QUEUE = Path.home() / ".local" / "state" / "gpuq"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    receipt_bytes = subprocess.run(["workerq", "show", str(JOB), "--json"],
                                   check=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE).stdout
    receipt = json.loads(receipt_bytes)
    if receipt["state"] != "SUCCEEDED" or receipt["exit_code"] != 0:
        raise RuntimeError("archive only terminal successful preflight")
    paths = {
        "job991_live_preflight.log": QUEUE / "logs" / f"job-{JOB:06d}.log",
        "job991_live_manifest.json": QUEUE / "jobs" / str(JOB) / "manifest.json",
        "job991_live_environment.json": QUEUE / "jobs" / str(JOB) / "environment.json",
        "job991_live_result.json": QUEUE / "jobs" / str(JOB) / "result.json",
    }
    archive = {}
    for name, original in paths.items():
        target = HERE / name
        shutil.copyfile(original, target)
        source_bytes = original.read_bytes()
        if target.read_bytes() != source_bytes:
            raise AssertionError(f"copy differs from original: {name}")
        archive[name] = {"original": str(original), "bytes": len(source_bytes),
                         "sha256": digest(source_bytes)}
    (HERE / "job991_terminal_show.json").write_bytes(receipt_bytes)
    archive["job991_terminal_show.json"] = {
        "original": "workerq show 991 --json terminal stdout",
        "bytes": len(receipt_bytes), "sha256": digest(receipt_bytes),
    }
    zip_path = HERE / "job991_raw.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zipped:
        for name in archive:
            zipped.write(HERE / name, arcname=name)
    with zipfile.ZipFile(zip_path) as zipped:
        for name, info in archive.items():
            member = zipped.read(name)
            if digest(member) != info["sha256"] or member != (HERE / name).read_bytes():
                raise AssertionError(f"ZIP member differs from retained original: {name}")
    source = ROOT / "local_sim" / "test_p2_k4_increment_axis_gauss.py"
    result = HERE / "result.json"
    before = json.loads((HERE / "byte_state_start.json").read_text(encoding="utf-8"))
    after = json.loads((HERE / "byte_state_end.json").read_text(encoding="utf-8"))
    if before != after:
        raise AssertionError("HEAD, source/result or Python bytes drifted during preflight")
    if before["source_sha256"] != digest(source.read_bytes()) or before["result_sha256"] != digest(result.read_bytes()):
        raise AssertionError("focused source or result differs from preflight-bound bytes")
    verification = {
        "job_id": JOB, "terminal_state": receipt["state"],
        "exit_code": receipt["exit_code"], "peak_ram_mib": receipt["peak_ram_mib"],
        "source_sha256": digest(source.read_bytes()),
        "fixture_result_sha256": digest(result.read_bytes()),
        "byte_states_identical": True, "tested_head": before["head"],
        "python_file_count": len(before["python_hashes"]),
        "zip_sha256": digest(zip_path.read_bytes()), "members": archive,
    }
    (HERE / "queue_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
