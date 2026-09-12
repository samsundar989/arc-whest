"""Generate and hash full-width synthetic inputs only, through workerq."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "local_sim"))
from p2_k4_integrated_first_source import _integrated_inputs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    weights, raw, probability, digest = _integrated_inputs(1024, 16, 64)
    payload = {
        "kind": "generator-only full-width input hash; no flopscope context",
        "input_sha256": digest,
        "source_sha256": sha(ROOT / "local_sim" /
                             "p2_k4_integrated_first_source.py"),
        "script_sha256": sha(__file__),
        "input_manifest_sha256": sha(Path(__file__).with_name(
            "input_manifest.json")),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True).strip(),
        "shape": [1024, 16],
        "source_rank": 64,
        "map_dtype": str(weights[0].dtype),
        "raw_dtype": str(raw.dtype),
        "probability_dtype": str(probability.dtype),
        "input_array_bytes": sum(x.nbytes for x in weights)
        + raw.nbytes + probability.nbytes,
    }
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("input_sha256=" + digest)


if __name__ == "__main__":
    main()
