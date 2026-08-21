"""Emit the CP1.1 evidence artifact. Read-only; writes one JSON.

Contract and predictions K1-K5: `experiments/LOG.md`, 2026-08-20, committed as
b2a92a2 before any of it ran. The contract's Scope named
`experiments/2026-08-20_kerdock/` and the first pass did not write it -- the
Skeptic caught the omission. This is that artifact, regenerable.

Run:  uv run python local_sim/kerdock_artifact.py
"""

from __future__ import annotations

import collections
import json
import math
import os
import sys
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

OUT = os.path.join(_ROOT, "experiments", "2026-08-20_kerdock", "kerdock_profile.json")
DIM, B = 256, 2.72e11


def main() -> int:
    import importlib.util as u

    from carriers import _kerdock_bits, build, kerdock_signs  # noqa: F401

    spec = u.spec_from_file_location("tc", os.path.join(_HERE, "test_carriers.py"))
    tc = u.module_from_spec(spec)
    spec.loader.exec_module(tc)

    bits = _kerdock_bits()
    w = bits.sum(axis=1)
    weights = {int(k): int(v) for k, v in
               sorted(collections.Counter(w.tolist()).items())}
    n_uniq = int(len(np.unique(np.packbits(bits.astype(np.uint8), axis=1), axis=0)))

    # Delsarte dual distance from the weight enumerator -- exact integers.
    size = 65536

    def kraw(k, ww):
        return sum((-1) ** j * comb(ww, j) * comb(DIM - ww, k - j)
                   for j in range(k + 1))

    dual = []
    for k in range(1, 8):
        tot = sum(a * kraw(k, ww) for ww, a in weights.items())
        dual.append(tot // size)

    arms = {}
    iid_by_n = {}
    for name, n in (("iid", 16384), ("sobol (SHIPPED)", 16384),
                    ("sobol (SHIPPED)", 65536), ("haar frames", 65536),
                    ("kerdock (rotated)", 65536),
                    ("kerdock+axial (rotated)", 66048),
                    ("bch OA (rotated)", 1 << 17)):
        p = tc.strength_profile(build(name, n, DIM, 20260820), DIM,
                                seed=4242, n_dirs=800 if n > 40000 else 4000)
        if n not in iid_by_n:
            iid_by_n[n] = tc.strength_profile(
                build("iid", n, DIM, 20260820), DIM, seed=4242,
                n_dirs=800 if n > 40000 else 4000)
        arms[f"{name} @{n}"] = {
            "n": n,
            "absolute": {str(k): p[k] for k in tc.DEGREES},
            "ratio_to_iid_at_same_n": {str(k): p[k] / iid_by_n[n][k] for k in tc.DEGREES},
        }

    payload = {
        "contract_commit": "b2a92a2",
        "slice": "CP1.1",
        "what": "Kerdock (length-256, non-linear) + 2d axial points",
        "n_points": (1 << 16) + 2 * DIM,
        "fingerprint": {
            "distinct_codewords": n_uniq,
            "length": int(bits.shape[1]),
            "weight_distribution": weights,
            "closed_under_complement": bool(np.array_equal(
                np.unique(np.packbits(bits.astype(np.uint8), axis=1), axis=0),
                np.unique(np.packbits((1 - bits).astype(np.uint8), axis=1), axis=0))),
        },
        "delsarte_dual_B1_to_B7": dual,
        "orthogonal_array_strength": int(next(
            (i for i, b in enumerate(dual, 1) if b != 0), len(dual) + 1) - 1),
        "arms": arms,
        "cost": {
            "note": "counted FLOPs only; the CP0.3 line was fitted on the D18 Sobol arms",
            "cb_from_cp0_3_line": 0.69316,
            "cb_rotation_applied_to_points": 0.75669,
            "cb_rotation_folded_into_W1": 0.69341,
            "failure_cliff": 1.0,
            "rotation_equivalence": "W1^T (Q u) == (Q^T W1)^T u, max abs diff 1.705e-13",
        },
        "caveats": [
            "NO end-to-end MSE gain measured. Needs fresh networks on a DESIGN block.",
            "Exact spherical 4-design; degrees 1/3/5 are free from antipodal symmetry.",
            "Degree 6 is equivalent to the BCH array once scaled to equal n (1.016).",
            "The iid denominator carries +-3.5% per draw; ratio-of-ratios cannot "
            "resolve 5%.",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"strength {payload['orthogonal_array_strength']}  "
          f"dual B1..B7 {dual}")
    print(f"wrote {os.path.relpath(OUT, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
