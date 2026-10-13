"""Lane O: unrefit VAL-100 (`hold100`) screens of built files in the main tree's twin (`--raw`).
    python run_val.py <variants dir> tag[,tag...]     -> <variants dir>/../val_raw_<tag>.json
Paths are absolute (lane-m worktree), so outputs land in the live tree, not the job snapshot."""
import subprocess
import sys
from pathlib import Path

MAIN = Path("C:/Users/samsu/Documents/arc-whest")   # holds the full-split weight memmaps (cache/twin)
LANE = Path("C:/Users/samsu/Documents/arc-whest-lane-m")
V = LANE / sys.argv[1]
tw = [sys.executable, "-u", str(MAIN / "tools/twin/twin.py"), "eval", "--panel", "hold100", "--raw"]
for t in sys.argv[2].split(","):
    js = V.parent / f"val_raw_{t}.json"
    if js.exists():
        continue
    print("raw", t, flush=True)
    subprocess.run(tw + ["--cand", str(V / f"est_{t}.py"), "--json", str(js)], check=True, cwd=MAIN)
