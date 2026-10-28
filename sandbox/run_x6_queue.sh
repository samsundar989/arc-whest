#!/bin/bash
cd /home/claude/sandbox
python3 x6_chains.py 64 8 0 full,age2,age3,age4,ind3,dp,gate,k2 > results/x6_queue.log 2>&1
python3 x6_chains.py 64 8 1 full,age2,age3,age4,ind3,dp,gate,k2 >> results/x6_queue.log 2>&1
python3 x6_chains.py 128 16 0 full,age3,age4,ind3,dp,k2 >> results/x6_queue.log 2>&1
python3 x6_chains.py 256 8 0 full,age3,age4,dp,k2 >> results/x6_queue.log 2>&1
echo QUEUE_DONE >> results/x6_queue.log
