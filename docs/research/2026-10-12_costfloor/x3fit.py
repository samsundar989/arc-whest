"""How many exact ages fit under a F/B threshold, per combo and case."""
import os, json
import x3price as P
from x3grid import BASES
CASES = [("(a) pruning only, old tier r256", True, "std", "std"),
         ("(b) + glue 5.25, shipped depth, old tier", True, "out", "std"),
         ("(b) + glue 5.25, od16, old tier", True, "out", ("opt", 16)),
         ("(b) + glue 5.25, od8, old tier", True, "out", ("opt", 8)),
         ("(a') pruning only, NO old tier", False, "std", "std"),
         ("(c) glue 5.25 shipped depth, NO old tier", False, "out", "std"),
         ("(c) glue 5.25 od16, NO old tier", False, "out", ("opt", 16)),
         ("(c) glue 5.25 od8, NO old tier", False, "out", ("opt", 8))]
THR = [0.105, 0.115, 0.125]
AS = list(range(1, 16))
for bn, b in BASES.items():
    print("\n== %s ==   F/B by number of exact ages A" % bn)
    print("%-42s" % "case" + "".join("%7d" % a for a in AS) + "   | max A under " + " / ".join("%.3f" % t for t in THR))
    for cn, old, gl, dp in CASES:
        row = []; fit = {t: 0 for t in THR}; vals = {}
        for A in AS:
            f = os.path.join(P.RES, P.name_of(dict(b, age=A, old=old, glue=gl, depth=dp)) + ".json")
            if os.path.exists(f):
                v = json.load(open(f))["fb"]; vals[A] = v; row.append("%7.4f" % v)
            else:
                row.append("%7s" % "")
        for t in THR:
            ok = [A for A, v in vals.items() if v <= t]
            fit[t] = max(ok) if ok else 0
        print("%-42s" % cn + "".join(row) + "   | " + " / ".join("%5d" % fit[t] for t in THR))
