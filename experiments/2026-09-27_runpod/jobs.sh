#!/bin/bash
# RunPod GPU lanes. Each job writes where the local queue job would have written.
cd /workspace/arc
w() { until [ -f /workspace/$1 ]; do sleep 30; done; }
T="python tools/twin/twin.py train --stage S3 --cand estimator_k3_v62_s1.py --init experiments/2026-09-25_twin/S3/params_best.npz --train-panel fit80 --hold-panel fit20h --epochs 22 --accum 8 --lr 3e-4 --zero-scale 0.3 --eval-every 20 --patience 3"
ceil() { $T --out experiments/2026-09-28_ceil/$1 ${2:+--subst $2} > /workspace/log_ceil_$1.txt 2>&1; }
laneA() { w READY_W100; w READY_LABELS; ceil ctrl; ceil k3; ceil all; }
laneB() { w READY_W100; w READY_LABELS; ceil cov; ceil k4; }
laneC() { w READY_W100
  python experiments/2026-09-27_harm/run_dev.py H5r,H5d,H5e,H5z > /workspace/log_harm.txt 2>&1
  w READY_LABELS; w READY_BENCH
  python tools/closure_bench.py --nets 0-99 --out experiments/2026-09-29_closure/bench > /workspace/log_bench.txt 2>&1
  w READY_FULL
  python tools/twin/twin.py train --stage S3 --cand estimator_k3_v62_s1.py --init experiments/2026-09-25_twin/S3/params_best.npz --out experiments/2026-09-27_S2t3 --epochs 2 --accum 8 --lr 1.5e-4 --zero-scale 0.3 --eval-every 46 --patience 3 --train-panel train+d8b > /workspace/log_s2t3.txt 2>&1; }
laneA & laneB & laneC & wait
touch /workspace/ALL_DONE
