"""All scenario specs (S0..S7). `python3 x3grid.py run [i n]` prices them (cached in res/); import SPECS elsewhere."""
import sys
import x3price as P

C1 = dict(s1=2.5, s2=2.5, s3=(3.5, 2.0))          # S1(2.5)+S2(2.5)+S3(3.5/2.0)
C2 = dict(s1=3.0, s2b=2.5, s3=(4.0, 2.0))         # S1(3.0)+S2b(2.5)+S3(4.0/2.0)
C3 = dict(s1=2.0, s2=2.0, s3=(3.0, 2.0))          # S1(2.0)+S2(2.0)+S3(3.0/2.0)  most aggressive plausible
C4 = dict(s1=2.0, s2=2.0, s3=(2.5, 2.0))          # extra: output rows = the D21 column set (floor of S3)
BASES = {"S0": {}, "C1": C1, "C2": C2, "C3": C3, "C4": C4}
AGES = [4, 5, 6, 7, 8, 10]
GLUES = [("std", "std"), ("out", "std"), ("out", ("opt", 16)), ("out", ("opt", 8)), ("std", ("opt", 16)), ("std", ("opt", 8)),
         ("min", ("opt", 8))]


def specs():
    S = []
    # S1 / S2 / S2b / S3 singles
    for t in (2.0, 2.5, 3.0, 3.5, (2.5, 2.0), (2.5, 3.0), (2.5, 3.5), (2.0, 2.5), (2.0, 3.0)):
        S.append(dict(s1=t))
    for t in (2.0, 2.5, 3.0, (2.5, 2.0), (2.5, 3.0), (2.0, 2.5), (2.0, 3.0)):
        S.append(dict(s2=t))
    for t in (2.0, 2.5, 3.0):
        S.append(dict(s2b=t))
    for to in (None, 2.5, 3.0, 3.5, 4.0):
        for ti in (2.0, 2.5):
            S.append(dict(s3=(to, ti)))
    for to in (2.5, 3.0, 3.5, 4.0):
        S.append(dict(s3=(to, 2.0), olast=False))
    # pairs (to see additivity)
    S += [dict(s1=2.5, s2=2.5), dict(s1=2.5, s3=(3.5, 2.0)), dict(s2=2.5, s3=(3.5, 2.0))]
    # S4 combos x S5 ages x old on/off x S6 glue/depth
    for bn, b in BASES.items():
        for A in AGES:
            for old in (True, False):
                for gl, dp in GLUES:
                    S.append(dict(b, age=A, old=old, glue=gl, depth=dp))
    # S7: last layer K youngest
    for bn, b in BASES.items():
        for K in (2, 4):
            S.append(dict(b, lastk=K))
            S.append(dict(b, lastk=K, glue="out", depth=("opt", 8)))
            S.append(dict(b, lastk=K, old=False, age=6))
            S.append(dict(b, lastk=K, old=False, age=6, glue="out", depth=("opt", 8)))
    # extra ages without an old tier (how many exact ages fit)
    for bn, b in BASES.items():
        for A in (9, 11, 12, 13, 14, 15):
            for gl, dp in (("std", "std"), ("out", "std"), ("out", ("opt", 8))):
                S.append(dict(b, age=A, old=False, glue=gl, depth=dp))
    seen = set(); out = []
    for s in S:
        nm = P.name_of(s)
        if nm not in seen:
            seen.add(nm); out.append(s)
    return out


if __name__ == "__main__":
    S = specs()
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        for j, s in enumerate(S):
            if j % n != i:
                continue
            try:
                print(P.line(P.price(s)), flush=True)
            except Exception as e:
                print("FAIL", P.name_of(s), str(e)[-600:], flush=True)
    else:
        print(len(S), "specs")
