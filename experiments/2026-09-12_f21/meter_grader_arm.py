"""One-arm CLI for the accepted official-stack P2 FLOP anchor instrument.

This selects its existing ``grader`` arm with float64 loader weights, which
uses flopscope-backed weights and in-window WhestBench materialisation.
It does not read labels.
The stored one-row official report remains the separate external check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "local_sim"))

from p2_flop_anchor import _run_arm  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source = (ROOT / a.candidate).resolve(strict=True)
    if not source.is_relative_to(ROOT):
        p.error("candidate must be inside this repository")
    out = Path(a.out).resolve()
    if not out.is_relative_to(ROOT / "experiments") or out.exists():
        p.error("output must be new and under live experiments")
    before = sha256(source)
    arm = _run_arm(a.candidate, "grader", input_weight_dtype="f64_readonly")
    after = sha256(source)
    if before != after:
        raise RuntimeError("candidate source changed during metering")
    payload = {
        "instrument": "local_sim/p2_flop_anchor.py::_run_arm",
        "instrument_sha256": sha256(ROOT / "local_sim" / "p2_flop_anchor.py"),
        "candidate": a.candidate,
        "candidate_sha256": before,
        "arm": "grader",
        "weights_mode": "f64_readonly",
        "shape": {"width": 1024, "depth": 16},
        "truth_or_seed_block_read": False,
        "result": arm,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"{a.candidate}: {arm['total']:,.0f} counted FLOPs, {len(arm['ops'])} op families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
