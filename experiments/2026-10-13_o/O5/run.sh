#!/usr/bin/env bash
# O5: anatomy v2 on v84 (R8), KAS off (M6.1's interpretable base), 20 FIT nets; verify first.
set -e
PY=C:/Users/samsu/Documents/arc-whest/.venv/Scripts/python.exe
C=C:/Users/samsu/Documents/arc-whest/estimator_k3_v84_age6.py
O=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-13_o/O5
cd C:/Users/samsu/Documents/arc-whest-lane-m   # its tools/gpu_shim has the cupy block shim (M6.1); cache/hf is a junction
$PY tools/anatomy2.py $C --verify --out $O/verify_v84.json
$PY tools/anatomy2.py $C --panel fit --n 20 --set _KAS_ON=False --out $O/fit20_v84_kasoff.json
$PY tools/anatomy2.py --report $O/fit20_v84_kasoff.json > $O/fit20_v84_kasoff.md
echo O5_DONE
