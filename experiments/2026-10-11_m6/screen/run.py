"""M6.2 step 1: unrefit VAL-100 (hold100) screen of the price.py variant files in the twin.
    python run.py name[,name...]       -> val_raw_<tag>.json here"""
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]
# twin.py is identical on main and lane-m; main holds the full-split weight memmaps (cache/twin)
MAIN = Path("C:/Users/samsu/Documents/arc-whest")
OUT = Path(__file__).resolve().parent
tw = [sys.executable, "-u", str(MAIN / "tools/twin/twin.py"), "eval", "--panel", "hold100"]
for nm in sys.argv[1].split(","):
    tag = nm.replace("+", "_")
    js = OUT / f"val_raw_{tag}.json"
    if js.exists():
        continue
    print("raw", nm, flush=True)
    subprocess.run(tw + ["--raw", "--cand", str(OUT / f"variants/est_{tag}.py"), "--json", str(js)],
                   check=True, cwd=MAIN)
