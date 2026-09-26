"""Twin DEV-100 screen for the Step 1 arms (baked files, --raw), control first."""
import subprocess, sys
from pathlib import Path
D = Path(__file__).resolve().parent
arms = [("off", "estimator_k3_v68_wino.py")] + [
    (a, f"experiments/2026-09-27_harm/arms/estimator_k3_v68_wino__harm_{a}.py")
    for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["H5c", "H5d", "H4c", "H3c"])]
for name, cand in arms:
    out = D / f"dev_{name}.json"
    if out.exists():
        continue
    subprocess.run([sys.executable, "tools/twin/twin.py", "eval", "--raw", "--cand", cand,
                    "--panel", "dev", "--json", str(out)], check=True)
