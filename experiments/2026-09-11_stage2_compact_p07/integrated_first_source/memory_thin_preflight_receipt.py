"""Controlled live-byte default preflight for the symbolic memory ledger."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TARGET = "local_sim/p2_k4_integrated_memory_ledger.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def snapshot():
    paths = sorted(os.fsdecode(x).replace("\\", "/") for x in
                   git("ls-files", "-z", "--", "*.py").split(b"\0") if x)
    hashes = {p: sha((ROOT / p).read_bytes()) for p in paths}
    head = git("rev-parse", "HEAD").decode().strip()
    return {"head": head, "python_hashes": hashes,
            "python_tree_sha256": sha(json.dumps(hashes, sort_keys=True,
                                               separators=(",", ":")).encode()),
            "target_worktree_sha256": hashes[TARGET],
            "target_index_sha256": sha(git("show", head + ":" + TARGET)),
            "estimator_sha256": sha((ROOT / "estimator.py").read_bytes())}


def main():
    before = snapshot()
    log = OUT / "memory_thin_preflight.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if before["target_worktree_sha256"] != before["target_index_sha256"]:
        code = 100
    else:
        code = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", "scripts/preflight.ps1", "-Log", str(log)],
            cwd=ROOT, env=env, check=False).returncode
    after = snapshot()
    receipt = {"kind": "thin-boundary symbolic memory ledger live-byte default preflight",
               "before": before, "after": after,
               "preflight_exit_code": code,
               "same_source": before == after,
               "target_committed": (before["target_worktree_sha256"] ==
                                    before["target_index_sha256"]),
               "transcript_sha256": sha(log.read_bytes()) if log.exists() else None,
               "pass": code == 0 and before == after and
                       before["target_worktree_sha256"] ==
                       before["target_index_sha256"]}
    (OUT / "memory_thin_preflight_receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    print("memory ledger live preflight", "PASS" if receipt["pass"] else "FAIL",
          "exit", code, "Python paths", len(before["python_hashes"]),
          "unchanged", receipt["same_source"], flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
