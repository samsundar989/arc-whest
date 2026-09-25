"""Export panel weights + labels to flat float32 memmaps the torch venv can read.

    python tools/twin/export_weights.py            (official .venv; needs `datasets`)

`.venv-gpu` has torch but not the dataset reader, and the official `.venv` has the
reader but not torch; neither is changed. This writes, once:

    cache/twin/full_w.f32    (1000, 16, 1024, 1024) float32   the `full` split, row order
    cache/twin/mini_w.f32    ( 100, 16, 1024, 1024) float32   the `mini` split (DEV)
    cache/twin/{full,mini}_truth.npy   (N, 16, 1024) float64   all_layer_means
    cache/twin/{full,mini}_meta.json   names and seeds, for identity checks

The weights are the SAME float32 values `real_data.get(..., dtype=np.float32)`
returns (the grader's float64 weights cast once, as the chain's own opening cast
does). Resumable: a row whose done-marker exists is skipped.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "local_sim"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import real_data as rd  # noqa: E402

OUT = ROOT / "cache" / "twin"


def export(split: str) -> None:
    n = rd.n_mlps(split)
    w0, _, _ = rd.get(0, split, dtype=np.float32)
    shape = (n,) + w0.shape
    path = OUT / f"{split}_w.f32"
    # PLAIN FILE WRITES, NOT A WRITABLE MEMMAP. A memmap's dirty pages count
    # against this process's working set (36 GiB observed on a 4 GiB
    # declaration, worker-q 2134), which starved the whole queue of host RAM.
    # Seek-and-write goes through the reclaimable file cache instead.
    row_bytes = int(np.prod(shape[1:])) * 4
    if not path.exists() or path.stat().st_size != n * row_bytes:
        with open(path, "ab") as f:
            f.truncate(n * row_bytes)
    fh = open(path, "r+b")
    tpath = OUT / f"{split}_truth.npy"
    truth = np.load(tpath) if tpath.exists() else np.zeros((n,) + w0.shape[:1] + w0.shape[2:])
    mpath = OUT / f"{split}_meta.json"
    meta = json.loads(mpath.read_text()) if mpath.exists() else {"names": [None] * n,
                                                                 "seeds": [None] * n,
                                                                 "done": [False] * n}
    for i in range(n):
        if meta["done"][i]:
            continue
        w, tr, m = rd.get(i, split, dtype=np.float32)
        fh.seek(i * row_bytes)
        fh.write(np.ascontiguousarray(w, dtype=np.float32).tobytes())
        truth[i] = tr
        meta["names"][i], meta["seeds"][i] = m["mlp_name"], int(m["mlp_seed"])
        meta["done"][i] = True
        if i % 50 == 49 or i == n - 1:
            fh.flush()
            os.fsync(fh.fileno())
            np.save(tpath, truth)
            mpath.write_text(json.dumps(meta))
            print(f"{split}: {i + 1}/{n}", flush=True)
            p = os.environ.get("WORKERQ_PROGRESS")
            if p:
                Path(p).write_text(str((i + 1) / n))
    fh.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export("mini")
    export("full")
