"""Local wall A/B (PLAN §5.8 evidence): the REAL meter on one DEV net, candidates alternated.

    uv run python tools/wall_ab.py <a.py> <b.py> [...] [--net 3] [--reps 2] [--out <txt>]

Each run is a fresh `tools/_cost_worker.py` process (real flopscope, f64 read-only), so the files
are timed exactly as `tools/ev.py`'s cost step times them. Prints `file wall ops F/B` per run (the
format of `experiments/2026-10-10_p5/v82*_wall_ab.txt`) and the median ratio of each file to the
first. Run nothing else heavy alongside: a wall A/B is only as good as the machine's quiet.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(cand: str, net: int) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "tools/_cost_worker.py"), cand, "dev", str(net)],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode != 0 or not line:
        raise SystemExit(f"{cand}: cost worker failed\n{out.stderr[-3000:]}")
    return json.loads(line[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--net", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    walls = {f: [] for f in a.files}
    lines = []
    for _ in range(a.reps):
        for f in a.files:
            r = one(f, a.net)
            wall = r.get("wall_s", r.get("wall"))
            walls[f].append(wall)
            ln = f"{Path(f).name} {wall} {float(r['ops'])} {r['flops'] / 2 ** 41}"
            print(ln, flush=True)
            lines.append(ln)
    base = statistics.median(walls[a.files[0]])
    for f in a.files:
        ln = f"# {Path(f).name}: median {statistics.median(walls[f]):.2f} s, ratio vs first {statistics.median(walls[f]) / base:.4f}"
        print(ln)
        lines.append(ln)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
