#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/sandbox
python3 mc.py 256 8 0 2e7
python3 mc.py 512 8 0 1e7
python3 mc.py 96 8 0 1e8
python3 mc.py 64 8 2 1e8
