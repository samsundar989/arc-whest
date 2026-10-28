#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/sandbox
python3 mc.py 64 8 0 1e8
python3 mc.py 64 8 1 1e8
python3 mc.py 64 16 0 1e8
python3 mc.py 128 8 1 5e7
python3 mc.py 64 8 0 1e8 sphere
