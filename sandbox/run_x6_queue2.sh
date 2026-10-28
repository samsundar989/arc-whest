#!/bin/bash
cd /home/claude/sandbox
python3 x6_chains.py 64 16 0 full,age3,dp,k2 > results/x6_queue2.log 2>&1
python3 x6_chains.py 64 16 1 full,age3,dp,k2 >> results/x6_queue2.log 2>&1
python3 x6_chains.py 256 16 0 full,dp,age3 >> results/x6_queue2.log 2>&1
echo QUEUE_DONE >> results/x6_queue2.log
