#!/bin/bash
# M5.1 / M5.2 / M5.4-step-0: VAL-100 unrefit reads on the v81d twin chain (refit_starall2 params)
PY=C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe
O=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/val
mkdir -p $O
KT=C:/Users/samsu/Documents/arc-whest/experiments/2026-10-06_mask/ktab/aggr_q50.json
KRD=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5/ktab_RD.json
P=C:/Users/samsu/Documents/arc-whest/experiments/2026-10-08_p3/refit_starall2/params_best.npz
C="--cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py --bank star,starall --dress all --params $P --panel hold100"
B="$C --mask R=2.0,Hd=2.5,Dz=2.5,K=$KT"
E="$PY -u experiments/2026-10-10_m5/tw.py eval"
run() { local tag=$1; shift; [ -f $O/$tag.json ] || $E "$@" --json $O/$tag.json; }
run base      $B --arm mask+dress+bank
run fold_ns   $B --arm mask+dress+bank+fold --fold tau=2,mode=ns,stats=1
run fold_st   $B --arm mask+dress+bank+fold --fold tau=2,mode=st
run fold_rec  $B --arm mask+dress+bank+fold --fold tau=2,mode=rec
run rad_all   $B --arm mask+dress+bank+rad --rad all
run rad_q     $B --arm mask+dress+bank+rad --rad q=1,d22=0,d4=0
run rad_d22   $B --arm mask+dress+bank+rad --rad q=0,d22=1,d4=0
run rad_d4    $B --arm mask+dress+bank+rad --rad q=0,d22=0,d4=1
# M5.4 step 0: H by threshold (the K table keeps R and D only)
run h_d25     $C --arm mask+dress+bank --mask R=2.0,Hd=2.5,Dz=2.5,K=$KRD
run h_kh25    $C --arm mask+dress+bank --mask R=2.0,Hkh=2.5,Dz=2.5,K=$KRD
run h_lit25   $C --arm mask+dress+bank --mask R=2.0,H=2.5,Dz=2.5,K=$KRD
