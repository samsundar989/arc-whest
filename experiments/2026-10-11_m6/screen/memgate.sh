# M6.2 hand-over: VmPeak of age 6 vs the fold control (WSL RLIMIT_AS 8 GiB gate)
set -e
PY=C:/Users/samsu/Documents/arc-whest/.venv/Scripts/python.exe
D=experiments/2026-10-11_m6/screen
for v in fold fold_age6; do
  $PY local_sim/linux_memory_gate.py --candidate $D/variants/est_$v.py --repeats 3 --json $D/memgate_$v.json
done
