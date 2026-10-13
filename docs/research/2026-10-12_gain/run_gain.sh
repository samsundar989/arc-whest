#!/bin/bash
cd $HOME/x2_scratch
for g in "$@"; do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 gain.py $g >> log_gain_$1.txt 2>&1; done
