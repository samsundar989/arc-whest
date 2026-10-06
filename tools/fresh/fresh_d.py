"""FRESH-D -- a third, independent fresh panel from keenanpepper's d8b corpus. `.venv-gpu`.

PLAN rev 18 / handoff V4: the twin trains on d8b nets 0..2047 (`tools/twin/twin.py`,
`train+d8b`); the corpus holds 12,000 more (indices 2048..14047, `shard_ext*`) that
nothing in this repository has read. FRESH-D = the first 100 rows of `shard_ext00.npz`
(the ext shards are strided: shard_ext00 holds idx 2048 + 8k, so idx 2048..2840). Independent of FRESH-A/LOCK in BOTH generator (torch CPU Philox,
`manual_seed(770000 + idx)`, float32 randn x sqrt(2/n), per the dataset card) and
labeller (their H200 MC, N = 1e8; ours is not involved).

Weights are staged as float32 files (the shimmed evaluator has no torch), hashed, and
pinned in `experiments/2026-10-06_laneV/fresh_d.json`. Labels are converted to our
format: means = m1, var = m2 - m1^2 (raw moments, per layer).

    P=$(.venv/Scripts/python.exe -c "from huggingface_hub import hf_hub_download as h;         print(h('keenanpepper/arc-whestbench-p2-d8b-corpus-14k','shard_ext00.npz',repo_type='dataset'))")
    .venv-gpu/Scripts/python.exe tools/fresh/fresh_d.py "$P"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import gen  # noqa: E402

REPO = "keenanpepper/arc-whestbench-p2-d8b-corpus-14k"
W_SEED0 = 770000
N_NETS = 100
PIN = gen.ROOT / "experiments" / "2026-10-06_laneV" / "fresh_d.json"


def weights_path(idx: int) -> Path:
    return gen.cache_dir() / "fresh_d_weights" / f"{idx}.f32"


def gen_weights(idx: int) -> np.ndarray:
    g = torch.Generator(device="cpu").manual_seed(W_SEED0 + idx)
    w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * (2.0 / 1024) ** 0.5
    return w.numpy()


def main() -> int:
    if PIN.exists():
        raise SystemExit(f"{PIN} exists; FRESH-D is pinned")
    if len(sys.argv) > 1:   # .venv-gpu has no huggingface_hub: fetch with .venv, pass the path
        p = sys.argv[1]
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        p = hf_hub_download(REPO, "shard_ext00.npz", repo_type="dataset")
    d = np.load(p)
    idx_all = np.asarray(d["idx"])
    n_samples = int(d["n_samples"])
    moments, final_mean = d["moments"], d["final_mean"]   # read once: NpzFile re-reads per access
    seeds, hashes = [], []
    for k in range(N_NETS):
        idx = int(idx_all[k])
        m = np.asarray(moments[k], dtype=np.float64)       # (4, 16, 1024) raw m1..m4
        means, var = m[0], np.maximum(m[1] - m[0] ** 2, 0.0)
        if not np.allclose(means[-1], final_mean[k], rtol=0, atol=1e-6):
            raise SystemExit(f"idx {idx}: moments m1 != final_mean")
        w = gen_weights(idx)
        h = gen.sha256_weights(w)
        wp = weights_path(idx)
        wp.parent.mkdir(parents=True, exist_ok=True)
        w.tofile(wp)
        lp = gen.label_path("fresh_d", idx, n_samples)
        lp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(lp, means=means, var=var, n=np.int64(n_samples), input_seed=np.int64(idx),
                 width=np.int64(1024), weight_sha256=h,
                 meta=json.dumps({"source": f"hf://{REPO}/shard_ext00.npz", "row": k}))
        seeds.append(idx)
        hashes.append(h)
    PIN.write_text(json.dumps({
        "panel": "fresh_d", "source": f"hf://datasets/{REPO} shard_ext00.npz",
        "generator": "torch CPU manual_seed(770000 + idx), randn f32 x sqrt(2/1024)",
        "label_n": n_samples, "seeds": seeds, "weight_sha256": hashes}, indent=1) + "\n",
        encoding="utf-8")
    print(f"FRESH-D: {len(seeds)} nets, N = {n_samples:.0e}, pinned {PIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
