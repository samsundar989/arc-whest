"""Extra age points (A = 1,2,3,9 ...) so 'how many exact ages fit' can be read without interpolation."""
import sys
import x3price as P
from x3grid import BASES
S = []
for bn, b in BASES.items():
    for A in (1, 2, 3, 9, 11, 12):
        for old in (True, False):
            for gl, dp in (("std", "std"), ("out", "std"), ("out", ("opt", 16)), ("out", ("opt", 8))):
                S.append(dict(b, age=A, old=old, glue=gl, depth=dp))
if __name__ == "__main__":
    i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for j, s in enumerate(S):
        if j % n == i:
            try:
                P.price(s)
            except Exception as e:
                print("FAIL", P.name_of(s), str(e)[-300:].replace("\n", " | "), flush=True)
    print("done", flush=True)
