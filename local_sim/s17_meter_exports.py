"""Meter frozen complete-candidate exports with the existing grader-arm anchor.

This is an orchestration wrapper around p2_flop_anchor._run_arm, not a new
FLOP model. It runs one source at a time and writes a single complete artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import p2_flop_anchor as anchor


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if len(set(args.source)) != len(args.source):
        raise ValueError("duplicate source")
    result = {
        "meter": "local_sim/p2_flop_anchor.py::_run_arm(grader,f64_readonly)",
        "input_weight_dtype": "f64_readonly",
        "meter_sha256": hashlib.sha256(Path(anchor.__file__).read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": {"width": anchor.WIDTH, "depth": anchor.DEPTH},
        "budget": anchor.BUDGET,
        "sources": [],
    }
    for name in args.source:
        source = ROOT / name
        if source.parent != ROOT or not source.is_file():
            raise ValueError(f"source must be a root-level file: {name}")
        result["sources"].append({
            "name": name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "grader": anchor._run_arm(name, "grader", input_weight_dtype="f64_readonly"),
        })
    result["status"] = "complete"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
