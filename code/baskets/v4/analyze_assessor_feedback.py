#!/usr/bin/env python3
"""Analyze assessor SBS feedback (output1) joined to basket prod_classifier_agent.

For each prod_classifier_agent we compute, for the *Yandex AI* model:
  - win / draw / loss rate vs GPT (from models_ranks)
  - satisfaction rate per criterion (readability / completeness / accuracy / relevance)
    where a checked box == criterion satisfied, missing box == weakness
  - GPT satisfaction rate per criterion for comparison
  - representative assessor comments on losing cases
"""
import json
import re
from collections import Counter, defaultdict

BASKET = "baskets/v4/20260506_basket_yandex_ai_app_v4_validate"
OUTPUT = "baskets/v4/output1_43368eda-68e8-47ca-b1e6-622aff1eec29"
CRITERIA = ["readability", "completeness", "accuracy", "relevance"]

# 1. query_id -> prod_classifier_agent
qid2agent = {}
with open(BASKET) as f:
    for line in f:
        d = json.loads(line)
        qid2agent[d["id"]] = d.get("prod_classifier_agent")

# Per-agent accumulators
stats = defaultdict(lambda: {
    "n": 0,
    "ya_win": 0, "ya_draw": 0, "ya_loss": 0,
    "ya_sat": Counter(),      # criterion -> count satisfied (Yandex AI)
    "gpt_sat": Counter(),     # criterion -> count satisfied (GPT)
    "ya_loss_comments": Counter(),  # comment text -> freq on losing cases
})

# global checkbox-vs-rank sanity
rank_ckbox = []

with open(OUTPUT) as f:
    for line in f:
        d = json.loads(line)
        iv = d.get("inputValues", {})
        ov = d.get("outputValues", {})
        if not ov:
            continue
        qid = iv.get("meta", {}).get("query_id")
        agent = qid2agent.get(qid)
        names = iv.get("models_names", [])
        ranks = ov.get("models_ranks")
        cbs = ov.get("models_checkboxes")
        if not (names and ranks and cbs) or len(names) < 2:
            continue
        try:
            ya_i = names.index("Yandex AI")
            gpt_i = names.index("Gpt")
        except ValueError:
            continue
        try:
            ya_rank = int(ranks[ya_i]); gpt_rank = int(ranks[gpt_i])
        except (ValueError, TypeError, IndexError):
            continue

        s = stats[agent]
        s["n"] += 1
        # lower rank number == better
        if ya_rank < gpt_rank:
            s["ya_win"] += 1
        elif ya_rank == gpt_rank:
            s["ya_draw"] += 1
        else:
            s["ya_loss"] += 1

        ya_cb = (cbs[ya_i] if ya_i < len(cbs) else {}) or {}
        gpt_cb = (cbs[gpt_i] if gpt_i < len(cbs) else {}) or {}
        ya_checks = sum(1 for c in CRITERIA if ya_cb.get(c))
        rank_ckbox.append((ya_rank, gpt_rank, ya_checks,
                           sum(1 for c in CRITERIA if gpt_cb.get(c))))
        for c in CRITERIA:
            if ya_cb.get(c):
                s["ya_sat"][c] += 1
            if gpt_cb.get(c):
                s["gpt_sat"][c] += 1

        # collect comment on losing cases for Yandex AI
        if ya_rank > gpt_rank:
            cm = (ov.get("ranking_comment") or ya_cb.get("comment") or "").strip()
            if cm:
                s["ya_loss_comments"][cm] += 1

# sanity: avg checks of winner vs loser
win_checks = [yc for yr, gr, yc, gc in rank_ckbox if yr < gr]
loss_checks = [yc for yr, gr, yc, gc in rank_ckbox if yr > gr]
print("=== SANITY: checkbox==satisfied semantics ===")
print(f"Yandex avg checks when WINNING: {sum(win_checks)/max(len(win_checks),1):.2f} (n={len(win_checks)})")
print(f"Yandex avg checks when LOSING : {sum(loss_checks)/max(len(loss_checks),1):.2f} (n={len(loss_checks)})")
print()

# Order agents by volume
order = sorted([a for a in stats if a], key=lambda a: -stats[a]["n"])

rows = []
for a in order:
    s = stats[a]
    n = s["n"]
    wr = 100*s["ya_win"]/n
    dr = 100*s["ya_draw"]/n
    lr = 100*s["ya_loss"]/n
    ya = {c: 100*s["ya_sat"][c]/n for c in CRITERIA}
    gpt = {c: 100*s["gpt_sat"][c]/n for c in CRITERIA}
    rows.append((a, n, wr, dr, lr, ya, gpt, s["ya_loss_comments"]))

# Print summary table
print("=== PER-AGENT: Yandex AI vs GPT ===")
hdr = f"{'agent':<24}{'n':>6}{'win%':>7}{'draw%':>7}{'loss%':>7} | " + "".join(f"{c[:4]+'YA':>9}{c[:4]+'GP':>9}" for c in CRITERIA)
print(hdr)
for a, n, wr, dr, lr, ya, gpt, _ in rows:
    line = f"{a:<24}{n:>6}{wr:>7.1f}{dr:>7.1f}{lr:>7.1f} | "
    for c in CRITERIA:
        line += f"{ya[c]:>9.1f}{gpt[c]:>9.1f}"
    print(line)

# Dump structured data for the markdown report
report = {"agents": []}
for a, n, wr, dr, lr, ya, gpt, comments in rows:
    top_comments = [{"text": t, "freq": f} for t, f in comments.most_common(8)]
    report["agents"].append({
        "agent": a, "n": n,
        "ya_win_pct": round(wr, 1), "ya_draw_pct": round(dr, 1), "ya_loss_pct": round(lr, 1),
        "ya_sat": {c: round(ya[c], 1) for c in CRITERIA},
        "gpt_sat": {c: round(gpt[c], 1) for c in CRITERIA},
        "gap": {c: round(gpt[c]-ya[c], 1) for c in CRITERIA},
        "top_loss_comments": top_comments,
    })
with open("baskets/v4/assessor_feedback_analysis.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\nWrote baskets/v4/assessor_feedback_analysis.json")
