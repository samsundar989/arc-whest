"""Asserted rank-32 old-D21 readout transform for the gram-join package.

The source representation, exact reconstructed legs, column-energy weights and
gram join are untouched.  Only the old aggregate D21 product is approximated.
No truth, seed block or estimator execution is involved in this generator.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANCHOR = """    FF = fnp.concatenate([tw.FA, tw.FM], axis=1)
    # QUERY BASIS: `X_old` is this product's inner half -- the OLD lineages'
    # actual contribution to the consumed contraction in the current layer's
    # coordinates. It is stashed, not recomputed: the join reads it instead of
    # a carried Gram of stale weights.
    Xold = _mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T)
    tw._qold = Xold
    old = _mm(Xold, tw.Q.T)
    return old + rec"""

REPLACEMENT = """    FF = fnp.concatenate([tw.FA, tw.FM], axis=1)
    LL = fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1)
    # The gram join never consumes _qold. Avoid forming its n-by-r Xold stash.
    tw._qold = None
    r = tw.Q.shape[1]
    q = min(_OLD_QUERY_RANK, n)
    if q >= r:
        # Rank alone does not prove sketch coverage. This path is exact.
        old = _mm(_mm(LL, FF.T), tw.Q.T)
    else:
        # Fixed columns avoid sketch randomness, norm ties and any neuron
        # ordering dependency. Q is transported and need not be orthonormal.
        Omega = tw.Q[:, :q]
        Y = _mm(LL, _mm(FF.T, _mm(tw.Q.T, Omega)))
        U, R = fnp.linalg.qr(Y)
        diagonal = fnp.abs(fnp.diag(R))
        # Zero/deficient sketches use the exact query. A nearly singular
        # sketch is also unsafe because QR may return arbitrary extra axes.
        if float(fnp.max(diagonal)) <= 0.0 or float(fnp.min(diagonal)) <= 1e-10 * float(fnp.max(diagonal)):
            old = _mm(_mm(LL, FF.T), tw.Q.T)
        else:
            old = _mm(U, _mm(_mm(_mm(U.T, LL), FF.T), tw.Q.T))
    return old + rec"""


def build(src: str, rank: int = 32) -> str:
    if rank < 1:
        raise ValueError("query rank must be positive")
    if src.count("_QUERY_BASIS = False") != 1:
        raise ValueError("query approximation requires the gram join")
    if src.count(ANCHOR) != 1:
        raise ValueError(f"old-query anchor count {src.count(ANCHOR)}, expected one")
    if "_OLD_QUERY_RANK" in src:
        raise ValueError("already patched")
    src = src.replace(ANCHOR, REPLACEMENT, 1)
    flag = "_OLD_QUERY_RANK = %d\n" % rank
    at = "_QUERY_BASIS = False\n"
    src = src.replace(at, at + flag, 1)
    return src


def emit(base: Path, output: Path, rank: int = 32) -> str:
    raw = base.read_bytes()
    body = build(raw.decode("utf-8"), rank)
    header = ('"""%s: rank-%d two-pass old-D21 query on the gram package.\n'
              'Generated from %s, SHA-256 %s by local_sim/s17_query_cost.py.\n'
              'DESIGN only; not approved for submission.\n"""\n\n'
              % (output.name, rank, base.name, hashlib.sha256(raw).hexdigest()))
    output.write_bytes((header + body).encode("utf-8"))
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()
    print(emit(args.base, args.output, args.rank))
