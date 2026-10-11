"""Queue launcher: run a Lane M2 script under `.venv-kprop` with ARC on the path.

    python experiments/2026-10-07_laneM2/run_kprop.py m5_ablate.py --mini --nets 0:16 --out <abs.json>

Outputs must be ABSOLUTE paths into the live worktree (a relative write lands in the
worker-q snapshot and is deleted).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAIN = Path("C:/Users/samsu/Documents/arc-whest")
HERE = Path(__file__).resolve().parent


def main() -> int:
    env = dict(os.environ, ARC_FRESH_CACHE=str(MAIN / "cache" / "fresh"),
               PYTHONPATH=str(MAIN / "external" / "mlp_cumulant_propagation" / "src"))
    script, *rest = sys.argv[1:]
    return subprocess.run([str(MAIN / ".venv-kprop" / "Scripts" / "python.exe"), "-u", str(HERE / script), *rest],
                          env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
