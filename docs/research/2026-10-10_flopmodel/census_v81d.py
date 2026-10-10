import sys, os, json
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/scratch"))
import run_census as rc, fakefs
path = sys.argv[1] if len(sys.argv) > 1 else "estimator_k3_v81d_starall2.py"
mod, tot, ops = rc.run(path)
U = 2 * 1024**3 - 1024**2   # one dense 1024^3 product, billed
print("TOTAL", tot + 32768, "ops", ops + 1, "U", U, "total u", (tot+32768)/U)
# per-site matmul summary
g = defaultdict(lambda: dict(flops=0, leaf=0, ops=0, calls=0, layers=[]))
for m in rc.MM:
    k = (m["caller"], m["line"])
    g[k]["flops"] += m["flops"]; g[k]["leaf"] += m["leaf"]; g[k]["ops"] += m["ops"]; g[k]["calls"] += 1
    g[k]["layers"].append((m["layer"], m["A"], m["B"], m["depth"], m["flops"]))
mmtot = sum(v["flops"] for v in g.values())
print("matmul (_mm) total", mmtot, "%.2f%%" % (100*mmtot/tot), "leaf", sum(v["leaf"] for v in g.values()), "glue", mmtot - sum(v["leaf"] for v in g.values()))
for k, v in sorted(g.items(), key=lambda kv: -kv[1]["flops"]):
    print("%-24s %5d  %16d  %6.2f u  %5.2f%%  leaf %5.1f%%  calls %3d ops %5d" % (k[0], k[1], v["flops"], v["flops"]/U, 100*v["flops"]/tot, 100*v["leaf"]/max(v["flops"],1), v["calls"], v["ops"]))
json.dump(rc.MM, open(os.path.basename(path) + ".mm.json", "w"))
# non-mm ledger by site
g2 = defaultdict(lambda: [0, 0])
for op, fl, layer, key, shp in fakefs.LEDGER:
    if key[0] in ("_mm", "_sb"): continue
    g2[(key[1], key[2])][0] += fl; g2[(key[1], key[2])][1] += 1
rest = sum(v[0] for v in g2.values())
print("non-_mm total", rest, "%.2f%%" % (100*rest/tot), "ops", sum(v[1] for v in g2.values()))
for k, v in sorted(g2.items(), key=lambda kv: -kv[1][0])[:60]:
    print("   %-24s %5d %14d %6.3f u %5.2f%% ops %d" % (k[0], k[1], v[0], v[0]/U, 100*v[0]/tot, v[1]))
json.dump([list(map(lambda x: list(x) if isinstance(x, tuple) else x, r)) for r in fakefs.LEDGER], open(os.path.basename(path) + ".ledger.json", "w"))
