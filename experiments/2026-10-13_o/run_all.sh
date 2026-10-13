#!/usr/bin/env bash
# Lane O screens, run directly (AGENTS.md rule 6: candidate evaluations do not go through workerq), sequentially.
set -e
GPY=C:/Users/samsu/Documents/arc-whest/.venv-gpu/Scripts/python.exe
O=C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-13_o
cd C:/Users/samsu/Documents/arc-whest
$GPY $O/O1/o1_energy.py $O/O1/variants/est_v84.py $O/O1/energy_v84.json 0,1
$GPY $O/run_val.py experiments/2026-10-13_o/O1/variants v84,o1,o1_224,o1_192,o1_128
$GPY $O/run_val.py experiments/2026-10-13_o/O2/variants o2_256x128,o2_128x256,o2_256x192,o2_192x256,o2_256x256
$GPY $O/run_val.py experiments/2026-10-13_o/O3/variants o3_8,o3_9,o3_7,o3_6
echo ALL_DONE
