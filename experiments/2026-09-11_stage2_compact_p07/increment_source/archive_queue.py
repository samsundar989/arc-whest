"""Archive terminal workerq #983 originals and verify each retained ZIP member."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


JOB = 983
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
QUEUE = Path.home() / ".local" / "state" / "gpuq"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    receipt_bytes = subprocess.run(
        ["workerq", "show", str(JOB), "--json"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    receipt = json.loads(receipt_bytes)
    if receipt["state"] != "SUCCEEDED" or receipt["exit_code"] != 0:
        raise RuntimeError("archive only the terminal successful queue receipt")
    paths = {
        "job983_live_preflight.log": QUEUE / "logs" / f"job-{JOB:06d}.log",
        "job983_live_manifest.json": QUEUE / "jobs" / str(JOB) / "manifest.json",
        "job983_live_environment.json": QUEUE / "jobs" / str(JOB) / "environment.json",
        "job983_live_result.json": QUEUE / "jobs" / str(JOB) / "result.json",
    }
    archive = {}
    for name, original in paths.items():
        copy = HERE / name
        shutil.copyfile(original, copy)
        source_bytes, copy_bytes = original.read_bytes(), copy.read_bytes()
        if source_bytes != copy_bytes:
            raise AssertionError(f"copy differs from original: {name}")
        archive[name] = {"original": str(original), "bytes": len(source_bytes),
                         "sha256": digest(source_bytes)}
    (HERE / "job983_terminal_show.json").write_bytes(receipt_bytes)
    archive["job983_terminal_show.json"] = {
        "original": "workerq show 983 --json terminal stdout",
        "bytes": len(receipt_bytes), "sha256": digest(receipt_bytes),
    }
    zip_path = HERE / "job983_raw.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zipped:
        for name in archive:
            zipped.write(HERE / name, arcname=name)
    with zipfile.ZipFile(zip_path) as zipped:
        for name, info in archive.items():
            data = zipped.read(name)
            if digest(data) != info["sha256"] or data != (HERE / name).read_bytes():
                raise AssertionError(f"ZIP member differs from original: {name}")
    source = ROOT / "local_sim" / "test_p2_k4_increment.py"
    result = HERE / "result.json"
    verification = {
        "job_id": JOB,
        "terminal_state": receipt["state"], "exit_code": receipt["exit_code"],
        "peak_ram_mib": receipt["peak_ram_mib"],
        "source_sha256": digest(source.read_bytes()),
        "fixture_result_sha256": digest(result.read_bytes()),
        "zip_sha256": digest(zip_path.read_bytes()),
        "members": archive,
    }
    (HERE / "queue_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
