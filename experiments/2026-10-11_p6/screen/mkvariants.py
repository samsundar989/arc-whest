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
# Oct 11 salvage: dead-only H by count (current alpha ranking, no always-on drop), H row = kink keep(tau)
KEEP20 = [1024, 1024, 992, 960, 928, 896, 896, 864, 832, 832, 800, 768, 768, 768, 736, 736]


def hdead(table):
    return lambda s: price.rep(s, price.H_OLD, f'          "H": {table!r}}}')


KEEP175 = [1024, 1024, 992, 928, 896, 864, 832, 832, 800, 800, 768, 768, 736, 736, 704, 704]
KEEP15 = [1024, 1024, 960, 896, 864, 832, 800, 800, 768, 768, 736, 736, 704, 704, 672, 672]
price.VARIANTS["fold+Hd20"] = [price.fold, hdead(KEEP20)]
price.VARIANTS["fold+Hd175"] = [price.fold, hdead(KEEP175)]
price.VARIANTS["fold+Hd15"] = [price.fold, hdead(KEEP15)]
price.OUT = Path(__file__).resolve().parent / "variants"
names = sys.argv[1].split(",")
res = {}
for nm in names:
    fn = price.build(nm)
    F, ops = price.price(fn)
    res[nm] = {"file": fn.name, "flops": F, "F/B": F / price.B, "ops": ops}
    print(f"{nm:20s} F {F:16,d}  F/B {F / price.B:.5f}  ops {ops:6,d}", flush=True)
pj = price.OUT / "prices_r2.json"
old = json.loads(pj.read_text()) if pj.exists() else {}
old.update(res)
pj.write_text(json.dumps(old, indent=1))
