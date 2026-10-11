#!/usr/bin/env bash
# M6.1 compensation localisation (PREDICTIONS.md addendum). One job, four knockouts, nets 0-3.
set -e
PY=C:/Users/samsu/Documents/arc-whest/.venv/Scripts/python.exe
C=C:/Users/samsu/Documents/arc-whest/estimator_k3_v82c_r1.py
O=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-11_m6_anatomy
A=(--panel fit --n 4 --only D21,k3,nomu,all)
$PY tools/anatomy2.py $C "${A[@]}" --set _CORR_ON=False --out $O/loc_corr.json
$PY tools/anatomy2.py $C "${A[@]}" --set _DRESS_C2=0.0 --set _DRESS_D21=0.0 --set _DRESS_D22=0.0 --out $O/loc_dress.json
$PY tools/anatomy2.py $C "${A[@]}" --set _KAS_ON=False --out $O/loc_kas.json
$PY tools/anatomy2.py $C "${A[@]}" --set _CORR_ON=False --set _DRESS_C2=0.0 --set _DRESS_D21=0.0 --set _DRESS_D22=0.0 --set _KAS_ON=False --out $O/loc_alloff.json
