"""
Inject the Sobol direction-number blob into estimator_v1_1.py, replacing the
`_SOBOL_B64 = "@@BLOB@@"` placeholder (or a previously injected blob).

Keeps the submission a single self-contained file while leaving the table's
provenance reproducible: run this after make_sobol_blob.py changes.

Run: uv run python local_sim/inject_sobol_blob.py [target.py]
"""

import base64
import os
import re
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_sobol_blob import BITS, DIM, direction_numbers, sobol_uint  # noqa: E402

TARGET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "estimator_v1_1.py"
)

vt = direction_numbers()

# provenance check before we write anything into a submission file
from scipy.stats import qmc  # noqa: E402

n = 4096
ref = np.round(qmc.Sobol(d=DIM, scramble=False, seed=0).random(n) * (2.0**BITS)).astype(np.uint32)
assert np.array_equal(sobol_uint(n, vt), ref), "generator != scipy Sobol"
print(f"verified generator == scipy Sobol (n={n}, d={DIM})")

blob = base64.b64encode(zlib.compress(vt.astype("<u4").tobytes(), 9)).decode("ascii")
lines = [f'    "{blob[i : i + 100]}"' for i in range(0, len(blob), 100)]
replacement = "_SOBOL_B64 = (\n" + "\n".join(lines) + "\n)"

src = open(TARGET, encoding="utf-8").read()
pattern = re.compile(r"^_SOBOL_B64 = (?:\"@@BLOB@@\"|\((?:\s*\"[^\"]*\")+\s*\))", re.M)
if not pattern.search(src):
    raise SystemExit("no _SOBOL_B64 placeholder/blob found in " + TARGET)
open(TARGET, "w", encoding="utf-8").write(pattern.sub(replacement, src, count=1))

print(f"injected {len(blob)} b64 chars ({len(lines)} lines) into {os.path.basename(TARGET)}")

# round-trip: decode straight back out of the written file
ns = {}
exec(compile(open(TARGET, encoding="utf-8").read().split("import base64")[0], "x", "exec"), ns)
raw = zlib.decompress(base64.b64decode(pattern.search(open(TARGET, encoding="utf-8").read()).group(0).split("=", 1)[1].strip().strip("()").replace('"', "").replace("\n", "").replace(" ", "")))
back = np.frombuffer(raw, dtype="<u4").reshape(BITS, DIM)
assert np.array_equal(back, vt), "round-trip mismatch"
print("round-trip OK: embedded blob decodes to the exact direction numbers")
