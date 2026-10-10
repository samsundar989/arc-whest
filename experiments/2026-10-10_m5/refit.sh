#!/bin/bash
# M5 refits: 1 epoch TRAIN-900, lr 1e-4, cosine, from refit_starall2 (v81d's params).
# usage: refit.sh <tag> <stage> [extra twin args...]   (control: refit.sh ctl S2ts)
PY=C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe
O=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/refit_$1
KT=C:/Users/samsu/Documents/arc-whest/experiments/2026-10-06_mask/ktab/aggr_q50.json
P=C:/Users/samsu/Documents/arc-whest/experiments/2026-10-08_p3/refit_starall2/params_best.npz
tag=$1; stage=$2; shift 2
exec $PY -u experiments/2026-10-10_m5/tw.py train --cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py \
  --mask R=2.0,Hd=2.5,Dz=2.5,K=$KT --dress all --init $P --epochs 1 --accum 8 --lr 1e-4 \
  --eval-every 28 --patience 3 --hold-panel hold100 --bank star,starall --stage $stage --out $O "$@"
