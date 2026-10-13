"""Lane O: paired VAL-100 comparison of screen arms against the v84 control (M6 table conventions).
    python compare.py <step dir> tag[,tag...]      (control: O1/val_raw_v84.json; prices: <step>/variants/prices.json)
break-even Δraw = 1/(1+ΔF) - 1; net = (1+Δraw)(1+ΔF) - 1 (the score change at λ = 0)."""
import json
import math
import sys
from pathlib import Path

O = Path(__file__).resolve().parent
step = O / sys.argv[1]
if len(sys.argv) > 3:                      # O4: control = a tag in the same step, prices from Lane K's meter text
    ctl = json.loads((step / f"val_raw_{sys.argv[3]}.json").read_text())
    prices = {ln.split(" | ")[0].strip(): {"flops": int(ln.split("TOTAL ")[1].split()[0])}
              for ln in (step / "variants/prices.txt").read_text().splitlines() if "TOTAL" in ln}
    F0 = prices[sys.argv[3]]["flops"]
else:
    ctl = json.loads((O / "O1/val_raw_v84.json").read_text())
    F0 = json.loads((O / "O1/variants/prices.json").read_text())["v84"]["flops"]
    prices = json.loads((step / "variants/prices.json").read_text())
c = ctl["per_net"]
print(f"control v84 raw {ctl['raw_mse']:.6e} (n {ctl['n']})\n")
print("| arm | raw MSE | Δraw | break-even | ΔF | **net score** | per-net Δ ± se | wins |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for t in sys.argv[2].split(","):
    p = step / f"val_raw_{t}.json"
    if not p.exists():
        print(f"| {t} | (not run) |")
        continue
    a = json.loads(p.read_text())["per_net"]
    assert len(a) == len(c)
    raw = sum(a) / len(a)
    d = raw / ctl["raw_mse"] - 1
    r = [x / y - 1 for x, y in zip(a, c)]
    m = sum(r) / len(r)
    se = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1) / len(r))
    dF = prices[t]["flops"] / F0 - 1
    be = 1 / (1 + dF) - 1
    net = (1 + d) * (1 + dF) - 1
    w = sum(x < y for x, y in zip(a, c))
    print(f"| {t} | {raw:.5e} | {d:+.2%} | {be:+.2%} | {dF:+.2%} | **{net:+.2%}** | {m:+.2%} ± {se:.2%} | {w}/{len(a)} |")
