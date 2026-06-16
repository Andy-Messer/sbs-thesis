#!/bin/bash
cd /Users/ai-krotov/Claude/Projects/sbs-project-thesis/sbs-thesis/code/experiments
# wait for chain1 (faithful+analyze+variance) to finish
until grep -qa "ALLDONE" chain.log 2>/dev/null; do sleep 45; done
echo "=== chain1 done, starting improve ==="
/tmp/sbsvenv/bin/python3 improve_judge.py > improve_run.txt 2>&1
echo "--- improve_judge done ---"; tail -3 improve_run.txt
/tmp/sbsvenv/bin/python3 analyze_improve.py > improve_analysis.txt 2>&1
echo "--- analyze_improve done ---"; tail -25 improve_analysis.txt
echo "ALLDONE2"
