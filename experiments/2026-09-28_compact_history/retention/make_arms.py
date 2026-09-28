"""Build the E2 information-retention diagnostic arms from the frozen v72 teacher.

Not candidates. Each arm is v72 with ONE source change:
  age3_r64 / age3_r32 : lineages older than 3 births projected onto the existing shared
                        range-finder basis at rank 64 / 32 (A_L, M legs only; P2/hub parts
                        stay exact) -- an optimistic bound on what a q-dim core can retain.
  age3_zero           : sharing off, keep only the 3 youngest lineages after each birth
                        (the zero-compressed-history deletion control of PLAN E2).
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "estimator_k3_v72_gt.py"
src = SRC.read_text(encoding="utf-8")
assert hashlib.sha256(SRC.read_bytes()).hexdigest().startswith("3b6cc397")


def sub(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b, 1)


arms = {
    "age3_r64": [("_SHARED_AGE = 5\n", "_SHARED_AGE = 3\n"), ("_SHARED_RANK = 256\n", "_SHARED_RANK = 64\n")],
    "age3_r32": [("_SHARED_AGE = 5\n", "_SHARED_AGE = 3\n"), ("_SHARED_RANK = 256\n", "_SHARED_RANK = 32\n")],
    "age3_zero": [("_SHARED_ON = True\n", "_SHARED_ON = False\n"),
                  ("    out_tw = _truncate(out_tw, gens, n)\n", "    out_tw = _truncate(out_tw, 3, n)\n")],
}
for name, subs in arms.items():
    s = src
    for a, b in subs:
        s = sub(s, a, b)
    p = Path(__file__).parent / f"arm_{name}.py"
    p.write_text(s, encoding="utf-8", newline="\n")
    print(name, hashlib.sha256(p.read_bytes()).hexdigest()[:8])
