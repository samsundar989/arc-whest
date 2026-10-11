"""Submit Lane M3's GPU work one job at a time, only when no Lane P job is waiting.

Same rule as Lane M2's gatekeeper (worker-q admitted low-priority Lane M jobs ahead of Lane P's
high-priority jobs on 2026-10-07): the next job goes in only when no high/critical job from
another checkout is QUEUED or RUNNING and the previous Lane M job has finished. Stdlib only.

    python experiments/2026-10-08_laneM3/gatekeeper.py [start] [stop]
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gatekeeper_m2", Path(__file__).resolve().parent.parent / "2026-10-07_laneM2" / "gatekeeper.py")
gk2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gk2)

LANE_M = gk2.LANE_M
O = "C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-08_laneM3"
RUN = ["python", "experiments/2026-10-08_laneM3/run_kprop.py"]
BIG = ["--vram", "26", "--ram", "28", "--cpus", "2"]

JOBS = [(f"M8 drop old kappa3 (bE + a, A in inf/8/5/3/2/1), nets {lo}:{lo + 4}", "25m", BIG,
         ["m8_drop.py", "--nets", f"{lo}:{lo + 4}", "--out", f"{O}/m8_{lo}.json"]) for lo in (0, 4, 8, 12)]


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stop = int(sys.argv[2]) if len(sys.argv) > 2 else len(JOBS)
    for k, (desc, eta, res, args) in enumerate(JOBS[start:stop], start):
        while gk2.others_waiting():
            gk2.time.sleep(60)
        cmd = ["workerq", "submit", "--project", "arc-whest", "--node", "local", "--priority", "low", *res,
               "--describe", f"Lane M3 [{k}] {desc}", "--blocks", "Lane P Oct 10 decision (M8)", "--eta", eta,
               "--", *RUN, *args]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=LANE_M)
        m = re.search(r"job #(\d+) submitted", r.stdout + r.stderr)
        if not m:
            gk2.log(f"[{k}] SUBMIT FAILED: {(r.stdout + r.stderr)[-400:]}")
            return 1
        jid = m.group(1)
        gk2.log(f"[{k}] submitted #{jid}: {desc}")
        w = subprocess.run(["workerq", "wait", jid], capture_output=True, text=True)
        gk2.log(f"[{k}] #{jid} finished, exit {w.returncode}")
        if w.returncode != 0:
            gk2.log(f"[{k}] FAILED #{jid}; stopping (resume with: gatekeeper.py {k + 1})")
            return 1
    gk2.log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
