#!/bin/bash
# RunPod data lane 1: organizer weights (twin memmaps). Flags in /workspace.
cd /workspace/arc && export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_SYMLINKS_WARNING=1
python tools/twin/export_parquet.py --split mini && \
python tools/twin/export_parquet.py --split full --rows 100 && \
python tools/twin/export_parquet.py --verify /workspace/arcwhest_sha.json && touch /workspace/READY_W100
python tools/twin/export_parquet.py --split full && touch /workspace/READY_FULL
python tools/twin/export_parquet.py --verify /workspace/arcwhest_sha.json
