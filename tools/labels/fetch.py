"""Fetch keenanpepper's public MC label files (PLAN rev 13 decision 3).

    python tools/labels/fetch.py full1000 --panel fit --n 20   dense pre-activation labels, N=1e9
    python tools/labels/fetch.py minimarg                      bakev2 marg.npz for the 100 DEV nets
    python tools/labels/fetch.py benchmarg                     bakev2 marg.npz for FIT (full rows 0..99)

Files land under cache/hf/<repo>/ (gitignored). `full1000` rows are aicrowd config
`full` row `global_index`, which is exactly our panel index; checked on load
(`global_index`, `mlp_name`) against data/panels/*.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FULL = "keenanpepper/arc-whestbench-p2-full1000-N1e9"
MINI = "keenanpepper/whest-p2-bakev2-mini-supp"
#: bench 0..99 = aicrowd config `full` rows 0..99 = our FIT panel (card, "B-full")
BENCH = "keenanpepper/whest-p2-bakev2-bench-supp"


def local_dir(repo: str) -> Path:
    return ROOT / "cache" / "hf" / repo.split("/")[1]


def fetch_full(panel: str, n: int) -> None:
    idx = json.loads((ROOT / "data" / "panels" / f"{panel}.json").read_text())["indices"][:n]
    for i in idx:
        p = hf_hub_download(FULL, f"moments/mlp_{i:05d}.npz", repo_type="dataset",
                            local_dir=local_dir(FULL))
        print("ok", i, p, flush=True)


def fetch_marg(repo: str) -> None:
    files = [f.rfilename for f in HfApi().dataset_info(repo).siblings
             if f.rfilename.endswith("marg.npz")]
    for f in sorted(files):
        hf_hub_download(repo, f, repo_type="dataset", local_dir=local_dir(repo))
        print("ok", f, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["full1000", "minimarg", "benchmarg"])
    ap.add_argument("--panel", default="fit")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    if a.what == "full1000":
        fetch_full(a.panel, a.n)
    else:
        fetch_marg(MINI if a.what == "minimarg" else BENCH)
