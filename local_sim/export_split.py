"""
Dump a dataset split's weights + 1e9-sample ground truth to a plain .npz so the
GPU environment can read it.

`.venv-gpu` deliberately has only numpy/scipy/torch — no whestbench, no
huggingface_hub — because it must never be able to influence an estimator file.
That means GPU experiments cannot load the real dataset directly. This exports
what they need, once, to the scratchpad.

Run (main venv):
    uv run python local_sim/export_split.py --split mini --n 100 --out <path.npz>
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = min(args.n, rd.n_mlps(args.split) - args.start)
    W, T, seeds, names = [], [], [], []
    for j in range(n):
        i = args.start + j
        w, truth, meta = rd.get(i, split=args.split, dtype=np.float32)
        W.append(w)
        T.append(truth)
        seeds.append(int(meta["mlp_seed"]))
        names.append(str(meta["mlp_name"]))
        if (j + 1) % 25 == 0:
            print(f"  {j + 1}/{n}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        weights=np.stack(W).astype(np.float32),
        truth=np.stack(T).astype(np.float64),
        seeds=np.asarray(seeds, dtype=np.int64),
        names=np.asarray(names),
        split=args.split,
        start=args.start,
    )
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB): "
          f"{n} MLPs from split={args.split} start={args.start}")


if __name__ == "__main__":
    main()
