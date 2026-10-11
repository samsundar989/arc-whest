"""M6.2: the strategist's price.py spend levers, patched from v82c_r2 (R2 constants) as P6.1's
screen does, on two bases: fold+Habs (the handoff's) and fold (near equilibrium unrefit; fold+Habs
unrefit is +65.6 % raw on VAL-100, P6.1 screen). Writes ./variants and meters every file.
    python mkvariants.py fold+Habs+J3,fold+J3,..."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3]
TOOL = R / "docs/research/2026-10-11_p6price"
sys.path.insert(0, str(TOOL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import price  # noqa: E402

price.SRC = R / "estimator_k3_v82c_r2.py"
price.OUT = Path(__file__).resolve().parent / "variants"
V = price.VARIANTS
import m6levers  # noqa: E402  (this folder)
for _L0 in (9, 11, 13):
    V.setdefault(f"fold+lateage{_L0}", [price.fold, m6levers.lateage(_L0)])
for lever, patch in (("J3", price.J(3)), ("J4", price.J(4)), ("age6", price.age(6)),
                     ("jsvd", price.join_svd), ("p2svd", price.p2svd)):
    V.setdefault(f"fold+{lever}", [price.fold, patch])
names = sys.argv[1].split(",")
out = price.OUT / "prices_m6.json"
res = json.loads(out.read_text()) if out.exists() else {}
for nm in names:
    fn = price.build(nm)
    F, ops = price.price(fn)
    res[nm] = {"file": fn.name, "flops": F, "F/B": F / price.B, "ops": ops}
    print(f"{nm:20s} F {F:16,d}  F/B {F / price.B:.5f}  ops {ops:6,d}", flush=True)
    out.write_text(json.dumps(res, indent=1))
