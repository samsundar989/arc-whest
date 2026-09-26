"""Exact pre-activation labels for FIT nets, in the chain's own names, float32, for the twin.

    .venv/Scripts/python.exe tools/twin/labels_cache.py [--n 100]

HANDOFF 2026-09-27 Step 2 (refit ceilings). Same definitions as `tools/anatomy2.py`
`load_truth` (keenanpepper full1000-N1e9 via `tools/labels/cumulants.py`), per layer l:

    mu, var, D3 = kappa_iii, dG = kappa_iiii / 2 (metric m = 2)          (16, n)
    C11 (diag 0), D21 = kappa_iij, D22 = kappa_iijj, Q31 = kappa_iiij    (16, n, n)

Q31 is RAW kappa_iiij; the twin normalises it with the chain's own sigma, as anatomy2's
`Ctl` does. One uncompressed npz per net under cache/twin_labels/ (~256 MB each).
Resumable: an existing file is skipped. CPU only (numpy, float64 conversions).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tools" / "labels"))
from cumulants import NetLabels  # noqa: E402

OUT = ROOT / "cache" / "twin_labels"
METRIC = 2.0


def build(g: int) -> None:
    p = OUT / f"fit_{g:03d}.npz"
    if p.exists():
        return
    L = NetLabels(g, np)
    mg = L.pre_marginals()
    depth, n = mg["mu"].shape
    dense = {k: np.zeros((depth, n, n), np.float32) for k in ("C11", "D21", "D22", "Q31")}
    for l in range(depth):
        d = L.layer(l)
        C = np.array(d["C"], dtype=np.float64)
        np.fill_diagonal(C, 0.0)
        dense["C11"][l] = C
        dense["D21"][l] = d["k3_iij"]
        dense["D22"][l] = d["k4_iijj"]
        dense["Q31"][l] = d["k4_iiij"]
        del d, C
    tmp = OUT / f"fit_{g:03d}.tmp.npz"
    np.savez(tmp, mu=mg["mu"].astype(np.float32), var=mg["var"].astype(np.float32),
             D3=mg["k3"].astype(np.float32), dG=(mg["k4"] / METRIC).astype(np.float32),
             name=np.array(L.name), **dense)
    os.replace(tmp, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prog = os.environ.get("WORKERQ_PROGRESS")
    for g in range(a.n):
        build(g)
        print(f"fit {g} done", flush=True)
        if prog:
            Path(prog).write_text(str((g + 1) / a.n))


if __name__ == "__main__":
    main()
