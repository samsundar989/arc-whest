"""Tables for the x3 report (reads res/*.json; prices anything missing)."""
import sys, json
import x3price as P
from x3grid import BASES, AGES, GLUES, C1, C2, C3, C4
U, B = P.U, P.B
GROUPS = [("recent transport", ["recent transport"]),
          ("recent D21 readout", ["recent D21 readout"]),
          ("cov sandwich", ["cov sandwich"]),
          ("old rebuild", ["old rebuild"]),
          ("old D21 readout", ["old D21 readout"]),
          ("old join", ["old join"]),
          ("old transport (Q)", ["old transport (Q)"]),
          ("closure", ["closure"]),
          ("thin (Z,Y) transport+readout", ["thin transport", "thin D21 readout"]),
          ("leg elementwise (P2,hubs,D3,D21 glue,fold,star)", ["P2 recon", "hubs", "D3 readout", "D21 glue", "fold (elementwise)", "star D4", "tower bookkeeping", "transport glue", "KAS select/gathers"]),
          ("everything else", None)]


def comp(d):
    c = {k: sum(v.values()) for k, v in d["comp"].items()}
    out = []; used = set()
    for nm, ks in GROUPS:
        if ks is None:
            out.append((nm, sum(v for k, v in c.items() if k not in used)))
        else:
            out.append((nm, sum(c.get(k, 0) for k in ks))); used |= set(ks)
    return out


def layers(d):
    L = [0.0] * 17
    for k, v in d["comp"].items():
        for l, x in v.items():
            L[int(l) + 1] += x
    return L


def show_split(names_specs):
    ds = [(nm, P.price(s)) for nm, s in names_specs]
    print("%-50s" % "component (units)" + "".join("%11s" % nm for nm, _ in ds))
    cs = [comp(d) for _, d in ds]
    for i, (g, _) in enumerate(GROUPS):
        print("%-50s" % g + "".join("%11.2f" % (c[i][1] / U) for c in cs))
    print("%-50s" % "TOTAL" + "".join("%11.2f" % d["units"] for _, d in ds))
    print("%-50s" % "  of which Strassen leaves / glue" + "".join("%11s" % ("%.1f/%.1f" % (d["leaf"] / U, d["glue"] / U)) for _, d in ds))
    print("%-50s" % "F/B" + "".join("%11.5f" % d["fb"] for _, d in ds))
    print("%-50s" % "ops" + "".join("%11d" % d["ops"] for _, d in ds))
    print()
    print("%-50s" % "layer (units)" + "".join("%11s" % nm for nm, _ in ds))
    Ls = [layers(d) for _, d in ds]
    for l in range(17):
        print("%-50s" % ("pre/consts" if l == 0 else "L%d" % (l - 1)) + "".join("%11.2f" % (x[l] / U) for x in Ls))


def tab(specs, ref=P.F0):
    for nm, s in specs:
        d = P.price(s)
        print("%-40s %8.2f %9.5f %+8.2f%% %+8.2f %7d" % (nm, d["units"], d["fb"], 100 * (d["flops"] / ref - 1), (d["flops"] - ref) / U, d["ops"]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    hdr = "%-40s %8s %9s %9s %8s %7s" % ("scenario", "units", "F/B", "dF", "d units", "ops")
    if what in ("all", "base"):
        print("== S0 base and main combos: component and layer split ==")
        show_split([("S0", {}), ("C1", C1), ("C2", C2), ("C3", C3), ("C4", C4)])
    if what in ("all", "single"):
        print("\n== S1 / S2 / S3 singles =="); print(hdr)
        tab([("S0 base", {})])
        tab([("S1 rows |a|<=%s" % (t,), dict(s1=t)) for t in (2.0, 2.5, 3.0, 3.5)])
        tab([("S1a rows -%g<=a<=%g" % t, dict(s1=t)) for t in ((2.5, 2.0), (2.5, 3.0), (2.5, 3.5), (2.0, 2.5), (2.0, 3.0))])
        tab([("S2 H |a|<=%s" % (t,), dict(s2=t)) for t in (2.0, 2.5, 3.0)])
        tab([("S2a H -%g<=a<=%g" % t, dict(s2=t)) for t in ((2.5, 2.0), (2.5, 3.0), (2.0, 2.5), (2.0, 3.0))])
        tab([("S2b A-leg |a|<=%s, M-leg non-dead" % (t,), dict(s2b=t)) for t in (2.0, 2.5, 3.0)])
        tab([("S3 out>-%s in>-%s" % (to if to else "inf", ti), dict(s3=(to, ti))) for to in (None, 2.5, 3.0, 3.5, 4.0) for ti in (2.0, 2.5)])
        tab([("S3 out>-%s in>-2.0, last layer all rows" % to, dict(s3=(to, 2.0), olast=False)) for to in (2.5, 3.0, 3.5, 4.0)])
        tab([("S1(2.5)+S2(2.5)", dict(s1=2.5, s2=2.5)), ("S1(2.5)+S3(3.5/2.0)", dict(s1=2.5, s3=(3.5, 2.0))), ("S2(2.5)+S3(3.5/2.0)", dict(s2=2.5, s3=(3.5, 2.0)))])
        print("\n== S4 combos =="); print(hdr)
        tab([("C1 = S1(2.5)+S2(2.5)+S3(3.5/2.0)", C1), ("C2 = S1(3.0)+S2b(2.5)+S3(4.0/2.0)", C2), ("C3 = S1(2.0)+S2(2.0)+S3(3.0/2.0)", C3), ("C4 = S1(2.0)+S2(2.0)+S3(2.5/2.0)", C4)])
    if what in ("all", "ages"):
        gl = [("std", "std"), ("std", ("opt", 16)), ("std", ("opt", 8)), ("out", "std"), ("out", ("opt", 16)), ("out", ("opt", 8)), ("min", ("opt", 8))]
        gn = ["8.25 ship", "8.25 od16", "8.25 od8", "5.25 ship", "5.25 od16", "5.25 od8", "3.75 od8"]
        for bn, b in BASES.items():
            for out in ("fb", "units", "ops"):
                print("\n== S5/S6 on %s : %s  (rows = exact ages A; old tier rank-256 vs none; columns = glue/depth) ==" % (bn, out))
                print("%-14s" % "A / old tier" + "".join("%11s" % g for g in gn))
                for old in (True, False):
                    for A in AGES + ([9, 11, 12, 13, 14, 15] if not old else []):
                        row = []
                        for g, dp in gl:
                            try:
                                import os
                                s = dict(b, age=A, old=old, glue=g, depth=dp)
                                if not os.path.exists(os.path.join(P.RES, P.name_of(s) + ".json")):
                                    row.append("%11s" % "--"); continue
                                d = P.price(s)
                                row.append("%11.5f" % d["fb"] if out == "fb" else ("%11.2f" % d["units"] if out == "units" else "%11d" % d["ops"]))
                            except Exception:
                                row.append("%11s" % "--")
                        print("%-14s" % ("A=%d %s" % (A, "old256" if old else "no-old")) + "".join(row))
    if what in ("all", "last"):
        print("\n== S7 last layer: only the K youngest lineages transported into L15 ==")
        print("%-34s %9s %9s %9s %9s %9s %9s" % ("base config", "all: F", "L15", "K=4: F", "L15", "K=2: F", "L15"))
        for bn, b in BASES.items():
            for extra, en in (({}, ""), (dict(glue="out", depth=("opt", 8)), "+g5.25/od8"), (dict(old=False, age=6), "+A6 no-old"), (dict(old=False, age=6, glue="out", depth=("opt", 8)), "+A6 no-old+g5.25/od8")):
                row = []
                for K in (None, 4, 2):
                    s = dict(b, **extra)
                    if K: s["lastk"] = K
                    d = P.price(s)
                    row += ["%9.2f" % d["units"], "%9.2f" % (layers(d)[16] / U)]
                print("%-34s " % (bn + en) + " ".join(row))
