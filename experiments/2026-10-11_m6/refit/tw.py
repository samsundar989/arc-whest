"""Lane M6 launcher (copy of the M5 one, one folder deeper): tools/twin/twin.py with the main checkout's data cache (worktree has none)."""
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("TWIN_CACHE", "C:/Users/samsu/Documents/arc-whest/cache/twin")
ROOT = Path(__file__).resolve().parents[3]
sys.argv = [str(ROOT / "tools" / "twin" / "twin.py")] + sys.argv[1:]
sys.path.insert(0, str(ROOT / "tools" / "twin"))
runpy.run_path(sys.argv[0], run_name="__main__")
