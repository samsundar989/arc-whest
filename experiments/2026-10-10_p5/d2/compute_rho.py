"""P5.2 step 1: per-layer d21e energy ratio rho_l = E[sum_all D21^2] / E[sum_kept D21^2]
(kept rows, 1/sigma^4 weighted as the corrector's feature is), from the control chain's
`--mask-stats` run on FIT-100. Writes rho.json next to this file."""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
d = json.loads(src.read_text())
acc = {}
for net, layers in d["stats"].items():
    for st in layers:
        if "e_all" in st:
            a, k = acc.get(st["l"], (0.0, 0.0))
            acc[st["l"]] = (a + st["e_all"], k + st["e_kept"])
depth = 16
rho = [1.0] * depth
for l, (a, k) in sorted(acc.items()):
    rho[l] = a / k if k > 0 else 1.0
out = src.parent / "rho.json"
out.write_text(json.dumps({"rho": rho, "source": str(src), "nets": len(d["stats"])}, indent=1))
for l, r in enumerate(rho):
    print(l, round(r, 4))
