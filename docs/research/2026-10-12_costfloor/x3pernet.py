"""Per-network FLOP variability if keep counts were ADAPTIVE (each net's own class counts, nearest 32)
instead of fixed q50 tables.   python3 x3pernet.py run [i n]   /   python3 x3pernet.py report"""
import sys, json, os
import numpy as np
import x3price as P
from x3grid import C1, C2, C3, C4

COMBOS = {"S0": {}, "C1": C1, "C2": C2, "C3": C3, "C4": C4,
          "C1+A6noold": dict(C1, age=6, old=False), "C3+A6noold": dict(C3, age=6, old=False),
          "C3+A8noold+gout+od8": dict(C3, age=8, old=False, glue="out", depth=("opt", 8))}
taus, dead, on = P.load_stats()
NN = dead.shape[1]


def one(cn, i):
    T = P.Tabs(dead, on, taus, i)
    return P.run_cfg(P.mk(COMBOS[cn], T), "pn_%s_%03d" % (cn.replace("+", "_"), i), keep_json=False)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        i0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        for cn in COMBOS:
            for i in range(NN):
                if i % n == i0:
                    try:
                        one(cn, i)
                    except Exception as e:
                        print("FAIL", cn, i, str(e)[-400:], flush=True)
            print("done", cn, flush=True)
    else:
        print("%-22s %9s %9s %7s %9s %9s %9s %9s %9s" % ("combo (adaptive)", "fixed u", "mean u", "CV %", "p10 u", "p50 u", "p90 u", "min u", "max u"))
        for cn, spec in COMBOS.items():
            fx = P.price(spec)["units"]
            v = []
            for i in range(NN):
                f = os.path.join(P.RES, "pn_%s_%03d.json" % (cn.replace("+", "_"), i))
                if os.path.exists(f):
                    v.append(json.load(open(f))["units"])
            if not v: continue
            v = np.array(v)
            print("%-22s %9.2f %9.2f %7.2f %9.2f %9.2f %9.2f %9.2f %9.2f  (n=%d)  F/B mean %.5f p10 %.5f p90 %.5f" % (
                cn, fx, v.mean(), 100 * v.std() / v.mean(), np.percentile(v, 10), np.median(v), np.percentile(v, 90),
                v.min(), v.max(), len(v), v.mean() * P.U / P.B, np.percentile(v, 10) * P.U / P.B, np.percentile(v, 90) * P.U / P.B))
