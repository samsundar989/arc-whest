"""Preserve workerq #986 originals and verify the frozen cost contract."""

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOB = 986
QUEUE = Path.home() / ".local" / "state" / "gpuq"
SOURCE = ROOT / "local_sim" / "p2_k4_first_source_kernel.py"
TEST = ROOT / "local_sim" / "test_p2_k4_first_source_kernel.py"
FROZEN_DENSE = 703252660224


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    show = subprocess.run(["workerq", "show", str(JOB), "--json"],
                          check=True, capture_output=True).stdout
    receipt = json.loads(show)
    if receipt["state"] != "SUCCEEDED" or receipt["exit_code"] != 0:
        raise AssertionError("job is not terminal successful")
    paths = {
        "job986_live.log": QUEUE / "logs" / "job-000986.log",
        "job986_manifest.json": QUEUE / "jobs" / "986" / "manifest.json",
        "job986_environment.json": QUEUE / "jobs" / "986" / "environment.json",
        "job986_result.json": QUEUE / "jobs" / "986" / "result.json",
    }
    members = {}
    for name, original in paths.items():
        copy = HERE / name
        shutil.copyfile(original, copy)
        if copy.read_bytes() != original.read_bytes():
            raise AssertionError("queue original/copy mismatch: " + name)
        members[name] = {"original": str(original), "bytes": copy.stat().st_size,
                         "sha256": sha(copy.read_bytes())}
    shown = HERE / "job986_terminal_show.json"
    shown.write_bytes(show)
    members[shown.name] = {"original": "workerq show 986 --json stdout",
                           "bytes": len(show), "sha256": sha(show)}
    zipped = HERE / "job986_raw.zip"
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_STORED) as z:
        for name in members:
            z.write(HERE / name, name)
    with zipfile.ZipFile(zipped) as z:
        for name, meta in members.items():
            if sha(z.read(name)) != meta["sha256"]:
                raise AssertionError("ZIP mismatch: " + name)
    meter_bytes = (HERE / "meter.json").read_bytes()
    meter = json.loads(meter_bytes)
    checks = {
        "source_hash_matches_meter": sha(SOURCE.read_bytes()) == meter["source_sha256"],
        "shape_S64": meter["shape"] == [1024, 16] and meter["rank"] == 64,
        "ledger_equals_context": sum(meter["families"].values()) == meter["flops"],
        "dense_families_equal_subtotal": sum(meter["dense_families"].values()) == meter["dense_subtotal"],
        "dense_within_five_percent": abs(meter["dense_subtotal"] / FROZEN_DENSE - 1) < .05,
        "negative_D21_over_350B": meter["negative_D21_flops"] > 350000000000,
        "process_peak_under_8GiB": receipt["peak_ram_mib"] < 8192,
        "queue_resources": (receipt["requested_ram_mib"] == 8192
                            and receipt["requested_cpus"] == 2
                            and receipt["requested_gpu_count"] == 0
                            and receipt["snapshot_mode"] == "live"),
        "schedule_exact": meter["schedule"] == meter["expected_schedule"],
        "single_array_below_4GiB": meter["memory"]["max_single_array_bytes"] < 4 * 2**30,
    }
    result = {
        "job_id": JOB, "checks": checks, "pass": all(checks.values()),
        "source_sha256": sha(SOURCE.read_bytes()),
        "test_sha256": sha(TEST.read_bytes()),
        "meter_sha256": sha(meter_bytes), "raw_zip_sha256": sha(zipped.read_bytes()),
        "raw_members": members,
        "frozen_dense_conventional": FROZEN_DENSE,
        "counted_dense": meter["dense_subtotal"],
        "dense_relative_difference": meter["dense_subtotal"] / FROZEN_DENSE - 1,
        "counted_total": meter["flops"],
        "counted_nondense": meter["flops"] - meter["dense_subtotal"],
        "negative_D21": meter["negative_D21_flops"],
        "operations": meter["operations"],
        "peak_ram_mib": receipt["peak_ram_mib"],
        "separate_supplied_core_baseline_flops": 818494496296,
        "baseline_note": "Different executable. Overlap is unreconciled; do not add totals.",
        "memory_limit_note": "Windows does not enforce the challenge 8 GiB RLIMIT_AS; observed workerq peak only.",
        "scope_note": "Only first-source discrete K4 and declared quadratic K3 feed; one A per birth, so multi-age same-birth sharing is untested.",
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["pass"]:
        raise AssertionError("registered gate failed: " + repr(checks))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
