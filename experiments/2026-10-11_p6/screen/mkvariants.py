"""P6.1 step 2: the strategist's price.py variants, patched from v82c_r2 (R2 constants) instead of
v82c_r1, written to ./variants and metered (prices.json in the tool folder is left untouched)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]
TOOL = R / "docs/research/2026-10-11_p6price"
sys.path.insert(0, str(TOOL))
import price  # noqa: E402

price.SRC = R / "estimator_k3_v82c_r2.py"
price.OUT = Path(__file__).resolve().parent / "variants"
names = sys.argv[1].split(",")
res = {}
for nm in names:
    fn = price.build(nm)
    F, ops = price.price(fn)
    res[nm] = {"file": fn.name, "flops": F, "F/B": F / price.B, "ops": ops}
    print(f"{nm:20s} F {F:16,d}  F/B {F / price.B:.5f}  ops {ops:6,d}", flush=True)
(price.OUT / "prices_r2.json").write_text(json.dumps(res, indent=1))
