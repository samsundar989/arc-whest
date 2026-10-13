#!/bin/bash
# run a cached job list with 2 workers for at most ~160 s (background jobs do not survive a tool call here)
cd $HOME/x3_scratch
timeout ${2:-100} python3 $1 run 0 2 >> $1.0.log 2>&1 &
timeout ${2:-100} python3 $1 run 1 2 >> $1.1.log 2>&1 &
wait
rm -f res/*.tmp_* res/*.part variants/est_*.py 2>/dev/null
ls res | wc -l
grep -h FAIL $1.*.log | sort | uniq | head -5
