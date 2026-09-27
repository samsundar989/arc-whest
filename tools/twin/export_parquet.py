"""Stream the organizer parquet shards into the twin's float32 memmaps (for a RunPod box).

    python tools/twin/export_parquet.py --split mini
    python tools/twin/export_parquet.py --split full --rows 100     # the first 100 rows only
    python tools/twin/export_parquet.py --split full                # all 1000

Same output as `export_weights.py` (cache/twin/<split>_w.f32, _truth.npy, _meta.json) but
without whestbench / `datasets`: each shard of `aicrowd/arc-whestbench-public-2026@v2-phase2`
is downloaded, its rows written at offset `mlp_id`, and the shard DELETED, so the disk never
holds the parquet and the memmap together (full = 75 GB parquet, 64 GB memmap). Resumable.
Weights are the stored float32 (input, output) arrays -- `real_data.get(..., np.float32)`.
Check: `--verify <sha.json>` compares sha256 of chosen rows against the Windows box's export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache" / "twin"
REPO, REV = "aicrowd/arc-whestbench-public-2026", "v2-phase2"
N = {"mini": 100, "full": 1000}
PER = 16 * 1024 * 1024


def export(split: str, rows: int) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    n = N[split]
    wp, tp, mp = OUT / f"{split}_w.f32", OUT / f"{split}_truth.npy", OUT / f"{split}_meta.json"
    if not wp.exists():
        with open(wp, "wb") as f:
            f.truncate(n * PER * 4)
    truth = np.load(tp) if tp.exists() else np.zeros((n, 16, 1024))
    meta = json.loads(mp.read_text()) if mp.exists() else {
        "names": [None] * n, "seeds": [None] * n, "done": [False] * n}
    shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
                    if f.startswith(f"data/{split}-"))
    fh = open(wp, "r+b")
    for sh in shards:
        if all(meta["done"][:rows]):
            break
        local = hf_hub_download(REPO, sh, repo_type="dataset", revision=REV,
                                local_dir=str(ROOT / "cache" / "parquet"))
        pf = pq.ParquetFile(local)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                i = int(r["mlp_id"])
                if meta["done"][i]:
                    continue
                w = np.asarray(r["weights"], dtype=np.float32)
                assert w.shape == (16, 1024, 1024)
                fh.seek(i * PER * 4)
                fh.write(np.ascontiguousarray(w).tobytes())
                truth[i] = np.asarray(r["all_layer_means"], dtype=np.float64)
                meta["names"][i], meta["seeds"][i] = r["mlp_name"], int(r["mlp_seed"])
                meta["done"][i] = True
        fh.flush()
        os.fsync(fh.fileno())
        np.save(tp, truth)
        mp.write_text(json.dumps(meta))
        os.remove(local)
        print(f"{split}: {sh} -> {sum(meta['done'])}/{n} rows", flush=True)
    fh.close()


def sha_rows(split: str, idx) -> dict:
    out = {}
    with open(OUT / f"{split}_w.f32", "rb") as f:
        for i in idx:
            f.seek(i * PER * 4)
            out[f"{split}:{i}"] = hashlib.sha256(f.read(PER * 4)).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(N))
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--sha", default="", help="write sha256 of rows e.g. full:0,99,999;mini:0")
    ap.add_argument("--verify", default="", help="compare against a --sha json")
    a = ap.parse_args()
    if a.sha:
        res = {}
        for part in a.sha.split(";"):
            sp, ix = part.split(":")
            res.update(sha_rows(sp, [int(x) for x in ix.split(",")]))
        print(json.dumps(res))
        return
    if a.verify:
        ref = json.loads(Path(a.verify).read_text())
        bad = 0
        for k, v in ref.items():
            sp, i = k.split(":")
            meta = json.loads((OUT / f"{sp}_meta.json").read_text())
            if not meta["done"][int(i)]:
                print(k, "not exported yet")
                continue
            ok = sha_rows(sp, [int(i)])[k] == v
            bad += not ok
            print(k, "OK" if ok else "MISMATCH")
        raise SystemExit(1 if bad else 0)
    export(a.split, a.rows or N[a.split])


if __name__ == "__main__":
    main()
