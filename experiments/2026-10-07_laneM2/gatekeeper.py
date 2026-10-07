"""Submit Lane M2's GPU work one short job at a time, only when no Lane P job is waiting.

worker-q's dispatcher admitted queued low-priority Lane M jobs ahead of Lane P's high-priority
exclusive jobs twice on 2026-10-07 (stale free-VRAM readings), against the handoff's ordering
(Lane P twin jobs first). So Lane M work is held out of the queue: the next job is submitted
only when (a) no high/critical job from another checkout is QUEUED or RUNNING and (b) the
previous Lane M job has finished. Jobs are ~5-15 min, so a Lane P job submitted mid-run waits
at most one of them. Stdlib only; log lines on stdout.

    python experiments/2026-10-07_laneM2/gatekeeper.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime

LANE_M = r"C:\Users\samsu\Documents\arc-whest-lane-m"
O = "C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-07_laneM2"
RUN = ["python", "experiments/2026-10-07_laneM2/run_kprop.py"]
BIG = ["--vram", "26", "--ram", "28", "--cpus", "2"]
A12 = "b,1_simple_legs,2_dressed,2c_asym,2d_dressed_s0.5,2e_dressed_s0.7,2f_dressed_s0.3,3a_memoryless,3b_gain,3c_memoryless+gain,4_no211,5_noK211"
A5 = "45_no_k3k4_loop,145_simple_legs_no_loop,3d_gain_from6,3e_memoryless+gain_from6,3f_memoryless_from6"

JOBS = [("M5 asym + dressing s, nets 0:4", "8m", BIG,
         ["m5_ablate.py", "--mini", "--nets", "0:4", "--arms", "2c_asym,2d_dressed_s0.5,2e_dressed_s0.7,2f_dressed_s0.3",
          "--out", f"{O}/m5_mini1024_part2.json"])]
JOBS.append(("M5 12 arms, nets 4:7", "14m", BIG,
             ["m5_ablate.py", "--mini", "--nets", "4:7", "--arms", A12, "--out", f"{O}/m5_mini1024_part1_4.json"]))
# Lane P's question (2026-10-07): sym dressing with the spurious H⊗H subtracted on the top-f hubs only
JOBS.append(("M5 dressed + top-hub H⊗H correction f=0.05/0.15/0.4, nets 0:8", "10m", BIG,
             ["m5_ablate.py", "--mini", "--nets", "0:8", "--arms",
              "2g_dressed_corr0.05,2h_dressed_corr0.15,2i_dressed_corr0.4",
              "--out", f"{O}/m5_mini1024_part4.json"]))
for lo in (7, 10, 13):
    JOBS.append((f"M5 12 arms, nets {lo}:{lo + 3}", "14m", BIG,
                 ["m5_ablate.py", "--mini", "--nets", f"{lo}:{lo + 3}", "--arms", A12,
                  "--out", f"{O}/m5_mini1024_part1_{lo}.json"]))
for lo in (0, 8):
    JOBS.append((f"M5 loop/late-gain arms, nets {lo}:{lo + 8}", "14m", BIG,
                 ["m5_ablate.py", "--mini", "--nets", f"{lo}:{lo + 8}", "--arms", A5,
                  "--out", f"{O}/m5_mini1024_part3_{lo}.json"]))
for lo in (0, 3):
    JOBS.append((f"M6 dense oracle, nets {lo}:{lo + 3}", "20m", ["--vram", "27", "--ram", "28", "--cpus", "2"],
                 ["m6_oracle.py", "--nets", f"{lo}:{lo + 3}", "--out", f"{O}/m6_oracle_{lo}.json"]))
for lo in (0, 4, 8, 12):
    JOBS.append((f"M6 e2e E21 forms, nets {lo}:{lo + 4}", "12m", BIG,
                 ["m6_e2e.py", "--nets", f"{lo}:{lo + 4}", "--out", f"{O}/m6_e2e_{lo}.json"]))
for lo in (0, 2, 4, 6):
    JOBS.append((f"M7 MC N=1e9, nets {lo}:{lo + 2}", "12m", ["--vram", "4", "--ram", "8", "--cpus", "2"],
                 ["m7_mc.py", "--state", f"{O}/m7_state.npz", "--n", "1e9", "--nets", f"{lo}:{lo + 2}",
                  "--out", f"{O}/m7_mc_{lo}.npz"]))


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def others_waiting() -> list[int]:
    out = subprocess.run(["workerq", "status", "--json"], capture_output=True, text=True).stdout
    jobs = json.loads(out).get("jobs", [])
    return [j["id"] for j in jobs if j["state"] in ("QUEUED", "RUNNING")
            and ((j["priority"] in ("high", "critical") and j.get("repo_root") != LANE_M)
                 or j.get("repo_root") == LANE_M)]   # also never stack on our own unfinished job


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for k, (desc, eta, res, args) in enumerate(JOBS[start:], start):
        while True:
            busy = others_waiting()
            if not busy:
                break
            time.sleep(60)
        cmd = ["workerq", "submit", "--project", "arc-whest", "--node", "local", "--priority", "low", *res,
               "--describe", f"Lane M2 [{k}] {desc}", "--blocks", "Lane M2 report", "--eta", eta, "--", *RUN, *args]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=LANE_M)
        m = re.search(r"job #(\d+) submitted", r.stdout + r.stderr)
        if not m:
            log(f"[{k}] SUBMIT FAILED: {(r.stdout + r.stderr)[-400:]}")
            return 1
        jid = m.group(1)
        log(f"[{k}] submitted #{jid}: {desc}")
        w = subprocess.run(["workerq", "wait", jid], capture_output=True, text=True)
        log(f"[{k}] #{jid} finished, exit {w.returncode}")
        if w.returncode != 0:
            log(f"[{k}] FAILED #{jid}; stopping (resume with: gatekeeper.py {k + 1} after fixing)")
            return 1
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
