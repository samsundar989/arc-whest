#!/bin/sh
# usage: run_dev.sh <candidate.py> <lo> <hi> <tag>
cd /c/Users/samsu/Documents/arc-whest
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=experiments/2026-10-10_p5/v82/dev
mkdir -p $D
uv run python tools/_predict_worker.py "$1" dev $2 $3 "$(pwd -W)/$D/$4_$2_$3.npz" > $D/$4_$2_$3.out 2> $D/$4_$2_$3.err
