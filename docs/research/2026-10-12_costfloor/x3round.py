"""Rounding study: what the multiples-of-32 convention costs.
 (1) F with q50 counts rounded floor-32 / nearest-32 / ceil-32 / nearest-16 / nearest-8 / raw (the trap: non-32
     multiples lose Strassen levels), at the shipped depth rule and at per-shape optimal depth.
 (2) finite differences: +32 on one table at one layer -> dF; F_cont = F_near + sum grad*(raw-near)/32 is the
     'smooth' price at the raw medians, so (F_near - F_cont) is what nearest-32 rounding costs (+) or saves (-).
   python3 x3round.py run [i n]  /  python3 x3round.py report"""
import sys, json, os
import numpy as np
import x3price as P, x3lib
from x3grid import C1, C2, C3

COMBOS = {"S0": {}, "C1": C1, "C2": C2, "C3": C3}
T = P.q50tabs()
T.adaptive_tables = True
MODES = [("floor", 32), ("near", 32), ("ceil", 32), ("near", 16), ("near", 8), ("raw", 1)]
TABS = ["R", "Dr", "Dc", "H", "O"]


class TA(P.Tabs):
    pass


def full_cfg(spec, g=32, mode="near"):
    """cfg with every table explicit (so each can be bumped)."""
    t = P.Tabs(T.dead, T.on, T.taus, "q50"); t.adaptive = True     # forces explicit R/D/H tables
    cfg = P.mk(spec, t, g, mode)
    cfg.setdefault("O", [1024] * 16)
    return cfg


def jobs():
    J = []
    for cn, spec in COMBOS.items():
        for mode, g in MODES:
            for dp in ("std", ("opt", 8)):
                J.append(("rd_%s_%s%d_%s" % (cn, mode, g, "std" if dp == "std" else "od8"), dict(full_cfg(spec, g, mode), depth=dp)))
        base = full_cfg(spec)
        for tb in TABS:
            for l in range(16):
                if tb in ("R", "Dr", "Dc", "H") and l == 15: continue
                if base[tb][l] >= 1024: sgn = -1
                else: sgn = +1
                c = json.loads(json.dumps(base)); c[tb][l] += 32 * sgn
                if tb == "H" and "HC" in c: pass
                J.append(("fd_%s_%s_%02d_%s" % (cn, tb, l, "p" if sgn > 0 else "m"), c))
    return J


if __name__ == "__main__":
    J = jobs()
    if sys.argv[1] == "run":
        i0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        for j, (nm, cfg) in enumerate(J):
            if j % n == i0:
                try:
                    P.run_cfg(cfg, nm, keep_json=False)
                except Exception as e:
                    print("FAIL", nm, str(e)[-500:], flush=True)
        print("done", flush=True)
    else:
        R = lambda nm: json.load(open(os.path.join(P.RES, nm + ".json")))
        print("(1) rounding mode x depth rule  [units ; F/B]")
        print("%-5s %-6s" % ("combo", "depth") + "".join("%16s" % ("%s-%d" % m) for m in MODES))
        for cn in COMBOS:
            for dp in ("std", "od8"):
                row = []
                for mode, g in MODES:
                    try:
                        d = R("rd_%s_%s%d_%s" % (cn, mode, g, dp)); row.append("%8.2f %.5f" % (d["units"], d["fb"]))
                    except Exception: row.append("   --")
                print("%-5s %-6s" % (cn, dp) + "".join("%16s" % x for x in row))
        print("\n(2) sensitivities: dF (units) for +32 kept on ONE table, summed over layers / and per layer")
        for cn, spec in COMBOS.items():
            base = full_cfg(spec); f0 = R("rd_%s_near32_std" % cn)["units"]
            t = P.Tabs(T.dead, T.on, T.taus, "q50"); t.adaptive = True
            raw = P.mk(spec, t, 1, "exact"); raw.setdefault("O", [1024] * 16)
            # exact raw medians (unrounded floats)
            def rawtab(tb):
                C = lambda a, b=None: t.count(a, b)
                if tb == "R": return C(spec["s3"][1]) if spec.get("s3") else C(2.0)
                if tb == "Dc": return C(2.5)
                if tb == "Dr": return C(*P._ab(spec["s1"])) if spec.get("s1") is not None else C(2.5)
                if tb == "O": return C(spec["s3"][0]) if spec.get("s3") and spec["s3"][0] is not None else [1024.0] * 16
                if tb == "H":
                    if spec.get("s2") is not None: return C(*P._ab(spec["s2"]))
                    if spec.get("s2b") is not None:
                        a = C(*P._ab(spec["s2b"])); m = C(2.5); return [(x + y) / 2 for x, y in zip(a, m)]
                    return C(2.5)
            corr = 0.0
            print("combo %s  F_near32 = %.2f u" % (cn, f0))
            for tb in TABS:
                g = []; rw = rawtab(tb)
                for l in range(16):
                    nm = [x for x in ("fd_%s_%s_%02d_p" % (cn, tb, l), "fd_%s_%s_%02d_m" % (cn, tb, l)) if os.path.exists(os.path.join(P.RES, x + ".json"))]
                    if not nm: g.append(0.0); continue
                    d = R(nm[0])["units"] - f0
                    if nm[0].endswith("_m"): d = -d
                    g.append(d)
                    corr += d * (rw[l] - base[tb][l]) / 32.0
                print("   %-3s sum %+6.2f u per +32 on all layers | per layer: %s" % (tb, sum(g), " ".join("%+.2f" % x for x in g)))
            print("   smooth price at the raw medians F_cont = %.2f u  -> nearest-32 rounding costs %+.2f u (%+.2f%%)" % (
                f0 + corr, -corr, -100 * corr / f0))
