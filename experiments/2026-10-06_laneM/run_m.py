"""Lane M1+M2 in one queued job: label the width-override nets, then run the variants.

Two interpreters (the labeller needs `.venv-gpu`, the ARC reference `.venv-kprop`), so
this driver only sequences subprocesses. Labels are resumable; M2 writes one JSON per
width to an ABSOLUTE path in the live worktree (a relative path would land in the
worker-q snapshot and be deleted).

    python experiments/2026-10-06_laneM/run_m.py --widths 256,512 --nets 0:8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MAIN = Path("C:/Users/samsu/Documents/arc-whest")
LIVE = Path("C:/Users/samsu/Documents/arc-whest-lane-m")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GPU_PY = MAIN / ".venv-gpu" / "Scripts" / "python.exe"
KPROP_PY = MAIN / ".venv-kprop" / "Scripts" / "python.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="256,512")
    ap.add_argument("--nets", default="0:8")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    env = dict(os.environ, ARC_FRESH_CACHE=str(MAIN / "cache" / "fresh"),
               PYTHONPATH=str(MAIN / "external" / "mlp_cumulant_propagation" / "src"))
    lo, hi = a.nets.split(":")
    widths = [int(w) for w in a.widths.split(",")]
    for w in widths:
        subprocess.run([str(GPU_PY), "-u", str(ROOT / "tools" / "fresh" / "mc.py"),
                        "--lane-m", f"{w}:{lo}:{hi}", "--n", "1e8"], env=env, check=True)
    for w in widths:
        out = LIVE / "experiments" / "2026-10-06_laneM" / f"m2_w{w}{a.tag}.json"
        cmd = [str(KPROP_PY), "-u", str(HERE / "m2_variants.py"), "--width", str(w),
               "--nets", a.nets, "--out", str(out)]
        if a.variants:
            cmd += ["--variants", a.variants]
        subprocess.run(cmd, env=env, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
