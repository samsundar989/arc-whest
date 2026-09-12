"""Copy terminal workerq preflight originals with a byte-checked ZIP."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
QUEUE = Path.home() / ".local" / "state" / "gpuq"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=int)
    args = parser.parse_args()
    job = args.job
    show = subprocess.run(["workerq", "show", str(job), "--json"],
                          check=True, capture_output=True).stdout
    receipt = json.loads(show)
    if receipt["state"] != "SUCCEEDED" or receipt["exit_code"] != 0:
        raise AssertionError("terminal successful preflight required")
    originals = {
        f"job{job}_log.log": QUEUE / "logs" / f"job-{job:06d}.log",
        f"job{job}_manifest.json": QUEUE / "jobs" / str(job) / "manifest.json",
        f"job{job}_environment.json": QUEUE / "jobs" / str(job) / "environment.json",
        f"job{job}_result.json": QUEUE / "jobs" / str(job) / "result.json",
    }
    members = {}
    for name, path in originals.items():
        copy = HERE / name
        shutil.copyfile(path, copy)
        if copy.read_bytes() != path.read_bytes():
            raise AssertionError("copied bytes differ from " + str(path))
        members[name] = {"original": str(path), "bytes": copy.stat().st_size,
                         "sha256": sha(copy.read_bytes())}
    show_path = HERE / f"job{job}_show.json"
    show_path.write_bytes(show)
    members[show_path.name] = {"original": f"workerq show {job} --json stdout",
                               "bytes": len(show), "sha256": sha(show)}
    zip_path = HERE / f"job{job}_raw.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in members:
            z.write(HERE / name, name)
    with zipfile.ZipFile(zip_path) as z:
        for name, meta in members.items():
            if sha(z.read(name)) != meta["sha256"]:
                raise AssertionError("ZIP member mismatch: " + name)
    verified = {"job_id": job, "state": receipt["state"],
                "exit_code": receipt["exit_code"],
                "peak_ram_mib": receipt["peak_ram_mib"],
                "requested_ram_mib": receipt["requested_ram_mib"],
                "requested_cpus": receipt["requested_cpus"],
                "requested_gpu_count": receipt["requested_gpu_count"],
                "snapshot_mode": receipt["snapshot_mode"],
                "zip_sha256": sha(zip_path.read_bytes()), "members": members}
    (HERE / f"job{job}_verification.json").write_text(
        json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
