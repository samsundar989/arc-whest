"""Compare retained workerq evidence against terminal queue originals bytewise."""

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
QUEUE = Path.home() / ".local" / "state" / "gpuq"
NAMES = {
    976: ("live.log", "live_manifest.json", "live_environment.json", "live_result.json"),
    977: ("live_preflight.log", "live_manifest.json", "live_environment.json", "live_result.json"),
    979: ("live.log", "live_manifest.json", "live_environment.json", "live_result.json"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    records = {}
    for job, suffixes in NAMES.items():
        originals = (QUEUE / "logs" / f"job-{job:06}.log",
                     QUEUE / "jobs" / str(job) / "manifest.json",
                     QUEUE / "jobs" / str(job) / "environment.json",
                     QUEUE / "jobs" / str(job) / "result.json")
        archive = HERE / f"job{job}_raw.zip"
        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            expected = {f"job{job}_{suffix}" for suffix in suffixes}
            assert names == expected, (job, names, expected)
            entries = {}
            for suffix, original in zip(suffixes, originals):
                copy = HERE / f"job{job}_{suffix}"
                raw = original.read_bytes()
                assert copy.read_bytes() == raw, copy
                assert zipped.read(copy.name) == raw, copy
                entries[copy.name] = {"sha256": digest(raw), "bytes": len(raw),
                                      "queue_original": str(original)}
        records[str(job)] = {"archive_sha256": digest(archive.read_bytes()),
                             "files": entries}
    (HERE / "queue_verification.json").write_text(
        json.dumps(records, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
