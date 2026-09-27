#!/bin/bash
# RunPod data lane 2: exact labels for FIT 0..99 (full1000 moments, label cache, bench-supp).
cd /workspace/arc && export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_SYMLINKS_WARNING=1
python tools/labels/fetch.py full1000 --panel fit --n 100 > /workspace/log_fetch_full.txt 2>&1 && \
python tools/twin/labels_cache.py --n 100 && touch /workspace/READY_LABELS
python tools/labels/fetch.py benchmarg > /workspace/log_fetch_bmarg.txt 2>&1
python - <<'PY'
from huggingface_hub import hf_hub_download
for g in range(100):
    for l in range(16):
        hf_hub_download("keenanpepper/whest-p2-bakev2-bench-supp", f"s00/net_{g:05d}/layer_{l:02d}.npz",
                        repo_type="dataset", local_dir="cache/hf/whest-p2-bakev2-bench-supp")
    print("bench net", g, flush=True)
PY
touch /workspace/READY_BENCH
