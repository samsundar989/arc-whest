import json, glob, os
import numpy as np
base = None
p = "experiments/2026-10-10_p5/d2/val_ctl.json"
if os.path.exists(p):
    base = np.array(json.load(open(p))["per_net"])
for f in sorted(glob.glob("experiments/2026-10-10_p5/d2/val_*.json")):
    d = json.load(open(f)); x = np.array(d["per_net"])
    s = f"{os.path.basename(f):22s} {d['raw_mse']:.6e} vs v81d-twin {100*(d['raw_mse']/1.945887e-08-1):+.2f}%"
    if base is not None and len(x) == len(base):
        r = x / base; s += f"  paired vs ctl {100*(x.mean()/base.mean()-1):+.2f}% (worse on {int((x>base).sum())}/{len(x)})"
    print(s)
