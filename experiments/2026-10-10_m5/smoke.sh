#!/bin/bash
# identity checks + one live fold/rad read on 3 VAL nets
PY=C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe
O=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/smoke
mkdir -p $O
B="--cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py --bank star,starall --mask R=2.0,Hd=2.5,Dz=2.5,K=C:/Users/samsu/Documents/arc-whest/experiments/2026-10-06_mask/ktab/aggr_q50.json --dress all --params C:/Users/samsu/Documents/arc-whest/experiments/2026-10-08_p3/refit_starall2/params_best.npz --panel hold100 --n 3"
set -x
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank --json $O/base.json
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+fold+rad --json $O/inert.json
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+fold --fold scale=0 --json $O/fold0.json
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+rad --rad all,scale=0 --json $O/rad0.json || true
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+fold --fold tau=2,mode=ns,stats=1 --json $O/fold_ns.json
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+fold --fold tau=2,mode=st --json $O/fold_st.json
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+fold --fold tau=2,mode=rec --json $O/fold_rec.json
$PY -u experiments/2026-10-10_m5/tw.py eval $B --arm mask+dress+bank+rad --rad all --json $O/rad.json
