"""Run default preflight on live bytes and record a before/after source receipt.

Submit only through workerq with --live-worktree --gpus 0 and declared RAM/CPU.
This runner is verification-only; it does not read challenge truth.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/2026-09-12_causal_signed_physical"
TARGET = "local_sim/test_p2_k4_causal_signed_physical.py"
EXPECTED_TARGET_SHA = "dffd6aa6d4232827082f77f6d72da5f2781dfb93c374674192a88556b315416a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def source_state():
    paths = sorted(os.fsdecode(p).replace("\\", "/") for p in
                   git("ls-files", "-z", "--", "*.py").split(b"\0") if p)
    hashes = {p: sha((ROOT / p).read_bytes()) for p in paths}
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "python_hashes": hashes,
            "python_tree_sha256": sha(json.dumps(hashes, sort_keys=True,
                                              separators=(",", ":")).encode()),
            "estimator_sha256": sha((ROOT / "estimator.py").read_bytes()),
            "target_sha256": hashes[TARGET]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = source_state()
    log = OUT / "preflight_live_controlled.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if before["target_sha256"] != EXPECTED_TARGET_SHA:
        code = 100
    else:
        code = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", "scripts/preflight.ps1", "-Log", str(log)],
            cwd=ROOT, env=env, check=False).returncode
    after = source_state()
    receipt = {"kind": "live-byte default preflight controlled receipt",
               "before": before, "after": after,
               "preflight_exit_code": code,
               "source_unchanged": before == after,
               "target_expected": before["target_sha256"] == EXPECTED_TARGET_SHA,
               "transcript": str(log.relative_to(ROOT)).replace("\\", "/"),
               "transcript_sha256": sha(log.read_bytes()) if log.exists() else None,
               "pass": code == 0 and before == after and
                       before["target_sha256"] == EXPECTED_TARGET_SHA}
    path = OUT / "preflight_live_controlled_receipt.json"
    path.write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    print("live-byte preflight", "PASS" if receipt["pass"] else "FAIL",
          "exit", code, "Python paths", len(before["python_hashes"]),
          "unchanged", receipt["source_unchanged"], flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
