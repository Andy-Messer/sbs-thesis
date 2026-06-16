"""Анализ текущих результатов: B_faithful + C_multiprompt + Design."""
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

VOTES = Path("/Users/ai-krotov/Claude/Projects/sbs-project-thesis/sbs-thesis-backup/code/experiments/staged_results/full_evaluation_votes.jsonl")
SAMPLE = Path("/Users/ai-krotov/Claude/Projects/sbs-project-thesis/sbs-thesis-backup/code/experiments/staged_results/sample_v4_200.json")
THETA = 0.7
B = 10000

lines = [json.loads(l) for l in VOTES.read_text().splitlines() if l.strip()]
sample = {str(r["query_id"]): r for r in json.loads(SAMPLE.read_text())}
qids_with_data = set(d["qid"] for d in lines if d.get("vote") in ("S", "C", "draw"))
qids_valid = [q for q in qids_with_data if q in sample]
av = {"ydx_win": "S", "gpt_win": "C", "draw": "draw"}
assessor = {q: av[sample[q]["verdict"]] for q in qids_valid}

def verdict(vlist):
    vs = [d["vote"] for d in vlist if d.get("vote") in ("S", "C", "draw")]
    if not vs: return None
    ws = sum(v == "S" for v in vs)
    wc = sum(v == "C" for v in vs)
    t = THETA * len(vs)
    if ws >= t and ws > wc: return "S"
    if wc >= t and wc > ws: return "C"
    return "draw"

def sbs(verd, qids):
    n = len(qids)
    if not n: return 0.0
    return (sum(verd.get(q) == "S" for q in qids) - sum(verd.get(q) == "C" for q in qids)) / n * 100

def agreement(jv, av, qids):
    valid = [q for q in qids if jv.get(q) is not None]
    if not valid: return 0.0, 0
    return sum(1 for q in valid if jv[q] == av[q]) / len(valid), len(valid)

# Content analysis
for meas in ["B_faithful", "C_multiprompt"]:
    votes_by_q = defaultdict(list)
    for d in lines:
        if d.get("measurement") == meas and d.get("vote") in ("S", "C", "draw"):
            votes_by_q[d["qid"]].append(d)
    jv = {q: verdict(votes_by_q.get(q, [])) for q in qids_valid}
    ag, nv = agreement(jv, assessor, qids_valid)
    sj = sbs(jv, qids_valid)
    print(f"{meas:20} n={nv:3} agree={ag:.0%} SbS={sj:+.1f}%")

# Design analysis
print("\nC.R.A.P. (средние баллы):")
for side in ["YandexAI", "Gpt"]:
    side_designs = [d for d in lines if d.get("eval_type") == "design" and d.get("side") == side]
    for crit in ["contrast", "repetition", "alignment", "proximity"]:
        vals = [d[crit] for d in side_designs if d.get(crit) is not None]
        mean = sum(vals)/len(vals) if vals else 0
        print(f"  {side:10} {crit:12} mean={mean:.2f} (n={len(vals)})")
    totals = [d["design_total"] for d in side_designs if d.get("design_total") is not None]
    print(f"  {side:10} TOTAL      mean={sum(totals)/len(totals):.1f}/8 (n={len(totals)})")
    print()
