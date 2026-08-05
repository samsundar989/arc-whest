"""
Emit the Sobol direction-number table that estimator_v1_1.py embeds.

Why embedded rather than shipped as a .npz: `whest submit --estimator <file>`
packages exactly one file, and CLAUDE.md requires the submission stay
self-contained. The table is 30 x 256 uint32 = 30,720 bytes raw, which
zlib+base64 compresses to ~20 KB of source text — trivial next to the 50 MiB
submission cap, and it keeps single-file submission intact.

The numbers themselves are the Joe-Kuo direction numbers that scipy ships
(BSD-licensed, same ones scipy.stats.qmc.Sobol uses); we extract them via
scipy's own initializer so there is no chance of transcribing the recurrence
wrongly, then verify our pure-integer generator reproduces scipy's point set
exactly.

Run:  uv run python local_sim/make_sobol_blob.py > /tmp/blob.txt
"""

import base64
import sys
import zlib

import numpy as np
from scipy.stats import _sobol
from scipy.stats import qmc

BITS = 30
DIM = 256


def direction_numbers(dim=DIM, bits=BITS):
    v = np.zeros((dim, bits), dtype=np.uint32)
    _sobol._initialize_v(v, dim, bits)
    return np.ascontiguousarray(v.T)  # (bits, dim), bit-plane major


def sobol_uint(n, vt, bits=BITS):
    """point[i, j] = XOR over set bits k of gray(i) of vt[k, j] — pure integer
    ops, exactly the sequence estimator_v1_1 runs under fnp."""
    d = vt.shape[1]
    i = np.arange(n, dtype=np.uint32)
    i = i ^ (i >> np.uint32(1))  # Gray code
    i = i[:, None]
    acc = np.zeros((n, d), dtype=np.uint32)
    for k in range(bits):
        bit = (i >> np.uint32(k)) & np.uint32(1)
        acc = np.bitwise_xor(acc, bit * vt[k][None, :])
    return acc


def main():
    vt = direction_numbers()

    # verify against scipy before emitting anything
    n = 4096
    ref = np.round(qmc.Sobol(d=DIM, scramble=False, seed=0).random(n) * (2.0**BITS)).astype(np.uint32)
    mine = sobol_uint(n, vt)
    assert np.array_equal(mine, ref), "generator does not reproduce scipy's Sobol points"
    print(f"# verified: pure-integer generator == scipy Sobol for n={n}, d={DIM}", file=sys.stderr)

    raw = vt.astype("<u4").tobytes()
    blob = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    print(f"# raw {len(raw)} B -> compressed+b64 {len(blob)} B", file=sys.stderr)

    print(f"_SOBOL_BITS = {BITS}")
    print(f"_SOBOL_DIM = {DIM}")
    print("_SOBOL_B64 = (")
    for i in range(0, len(blob), 100):
        print(f'    "{blob[i : i + 100]}"')
    print(")")

    np.savez_compressed(__file__.replace("make_sobol_blob.py", "sobol_dirnum.npz"), v=vt, bits=np.int32(BITS))


if __name__ == "__main__":
    main()
