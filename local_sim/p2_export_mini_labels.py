"""Export the organisers' `mini` split labels so a torch-only venv can score against them.

Contract: `experiments/LOG.md`, "CONTRACT (Rule 1): P-A-CONFIRM".

## Why an export step exists at all

`local_sim/real_data.py` needs the real `whestbench` to read the dataset, and
`whestbench` lives in `.venv`. The kprop arm needs torch, which lives in
`.venv-gpu`. Nothing sensible installs one into the other, so this module runs in
`.venv` and writes what the GPU venv needs.

## Why the WEIGHTS are not exported

They do not have to be. `mint_provenance.canonical_mint` is bit-identical to the
organisers' published `v2-phase2` weights, 16/16
(`local_sim/phase2_mint_parity.py`), so a seed reproduces a row exactly. Exporting
the weights would be 100 x 67 MB = 6.7 GB; exporting seeds and labels is 13 MB.

**That identity is re-asserted here rather than cited.** `--verify N` mints N rows
from their seeds and requires **bit-identity** against the row's own stored
weights. If that fails, nothing is written: a label set whose weights we cannot
reproduce is not usable, and the failure would otherwise surface as a mysterious
accuracy result rather than as what it is.

## READ-ONLY

This module reads the public split and writes labels. **It fits nothing, selects
nothing, and orders nothing.** `VALIDATION.md` permits exactly this use of a
CONFIRM-tier split and no other.

Run (the whestbench venv, NOT the GPU venv):
    .venv\\Scripts\\python.exe local_sim/p2_export_mini_labels.py \\
        --split mini --verify 3 --out experiments/2026-08-27_pa_confirm/mini_labels.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import real_data  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--revision", default="v2-phase2",
                    help="dataset revision. real_data.REVISION still defaults to "
                         "the PHASE-1 'v1-phase1', so this must be explicit.")
    ap.add_argument("--expect-width", type=int, default=1024)
    ap.add_argument("--expect-depth", type=int, default=16)
    ap.add_argument("--verify", type=int, default=3,
                    help="mint this many rows from their seeds and require "
                         "BIT-IDENTITY against the row's own stored weights")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # THE ROUND IS PINNED AND ASSERTED, not inherited. real_data.REVISION is
    # 'v1-phase1' -- a Phase-1 default that silently hands back 256x32 labels for
    # a 'mini' split that looks entirely normal. Scoring a Phase-2 arm against
    # them would produce a plausible, meaningless number. This happened on the
    # first run of this module and is the reason the assertion below exists.
    real_data.REVISION = a.revision
    n = real_data.n_mlps(a.split)
    if a.limit:
        n = min(n, a.limit)
    print(f"  dataset  {real_data.DATASET}@{a.revision}   split {a.split}: {n} MLPs")
    print(f"  expecting labels of shape ({a.expect_depth}, {a.expect_width})")

    seeds, names, labels, avgvar = [], [], [], []
    verified = 0
    for i in range(n):
        w, truth, meta = real_data.get(i, a.split, dtype=np.float32)
        if truth.shape != (a.expect_depth, a.expect_width):
            raise SystemExit(
                f"REFUSED: row {i} label is {truth.shape}, expected "
                f"({a.expect_depth}, {a.expect_width}). Wrong dataset revision "
                f"({a.revision!r})? Nothing written."
            )
        if i < a.verify:
            depth, width = w.shape[0], w.shape[1]
            wm, _e, _s = canonical_mint(width, depth, int(meta["mlp_seed"]))
            wm = np.stack([np.asarray(x, dtype=np.float32) for x in wm])
            same = bool(np.array_equal(wm, np.asarray(w, dtype=np.float32)))
            print(f"    row {i:>3} {meta['mlp_name']:<24} seed {meta['mlp_seed']}  "
                  f"mint bit-identical: {same}")
            if not same:
                raise SystemExit(
                    "REFUSED: canonical_mint does not reproduce this row's weights "
                    "bit-for-bit. Nothing written. A label set whose networks we "
                    "cannot reproduce is not usable."
                )
            verified += 1
        seeds.append(int(meta["mlp_seed"]))
        names.append(str(meta["mlp_name"]))
        labels.append(np.asarray(truth, dtype=np.float64))
        avgvar.append(float(meta["avg_variance"]))

    out = os.path.abspath(os.path.join(_ROOT, a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"REFUSED: {out} exists.")
    np.savez_compressed(
        out, seeds=np.asarray(seeds, dtype=np.int64),
        names=np.asarray(names), labels=np.stack(labels),
        avg_variance=np.asarray(avgvar, dtype=np.float64),
        split=a.split, revision=a.revision, n_mlps=np.int64(n),
        width=np.int64(a.expect_width), depth=np.int64(a.expect_depth),
        mint_rows_verified=np.int64(verified),
    )
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"\n  wrote {os.path.relpath(out, _ROOT)}  ({os.path.getsize(out):,} B)")
    print(f"  sha256 {sha}")
    print(f"  mint bit-identity verified on {verified}/{n} rows")
    print(f"  label shape {np.stack(labels).shape}, mean avg_variance "
          f"{np.mean(avgvar):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
