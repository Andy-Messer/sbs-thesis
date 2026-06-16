#!/bin/bash
cd /Users/ai-krotov/Claude/Projects/sbs-project-thesis/sbs-thesis/code/experiments
# 1) wait for faithful judge run to finish
until grep -qa "Анализ: analyze_faithful" faithful_run.log 2>/dev/null || ! pgrep -f faithful_judge_run.py >/dev/null; do sleep 30; done
echo "=== faithful done: $(wc -l < staged_results/votes_faithful.jsonl)/1200 ==="
# 2) analysis (per-criterion + train/test verdicts)
/tmp/sbsvenv/bin/python3 analyze_faithful.py > faithful_analysis.txt 2>&1
echo "--- analyze_faithful done ---"; tail -20 faithful_analysis.txt
# 3) multimodal variance experiment (vision model, RAM now free)
/tmp/sbsvenv/bin/python3 multimodal_variance.py > variance_run.txt 2>&1
echo "--- variance done ---"; tail -5 variance_run.txt
echo "ALLDONE"
