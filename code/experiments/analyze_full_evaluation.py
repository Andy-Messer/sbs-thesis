"""Анализ полной оценки: контент (A/B/C/D) + дизайн (C.R.A.P.).

Вход: full_evaluation_votes.jsonl + sample_v4_200.json
Выход: full_analysis.json + figures/full-eval-*.jpeg
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VOTES = ROOT / "code/experiments/staged_results/full_evaluation_votes.jsonl"
SAMPLE = ROOT / "code/experiments/staged_results/sample_v4_200.json"
OUT = ROOT / "code/experiments/staged_results"
FIG = ROOT / "figures"
THETA = 0.7
DELTA = 10.0
B = 10000
CRITS = ["relevance", "accuracy", "completeness", "readability"]
DESIGN_CRITS = ["contrast", "repetition", "alignment", "proximity"]


def load():
    content_votes = defaultdict(lambda: defaultdict(list))
    design_votes = defaultdict(lambda: defaultdict(list))
    for line in VOTES.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["eval_type"] == "content":
            content_votes[d["qid"]][d["measurement"]].append(d)
        elif d["eval_type"] == "design":
            design_votes[d["qid"]][d["side"]].append(d)
    sample = {str(r["query_id"]): r for r in json.loads(SAMPLE.read_text())}
    return content_votes, design_votes, sample


def content_verdict(vlist, theta=THETA):
    vs = [d["vote"] for d in vlist if d.get("vote") in ("S", "C", "draw")]
    if not vs:
        return None
    ws = sum(v == "S" for v in vs)
    wc = sum(v == "C" for v in vs)
    t = theta * len(vs)
    if ws >= t and ws > wc:
        return "S"
    if wc >= t and wc > ws:
        return "C"
    return "draw"


def assessor_verdict(r):
    return {"ydx_win": "S", "gpt_win": "C", "draw": "draw"}[r["verdict"]]


def sbs(verd, qids):
    n = len(qids)
    if not n:
        return 0.0
    return (sum(verd.get(q) == "S" for q in qids)
            - sum(verd.get(q) == "C" for q in qids)) / n * 100


def agreement(jv, av, qids):
    valid = [q for q in qids if jv.get(q) is not None]
    if not valid:
        return 0.0, 0
    a = sum(1 for q in valid if jv[q] == av[q]) / len(valid)
    return a, len(valid)


def bootstrap_ci(jv, av, qids):
    rng = random.Random(123)
    qids = list(qids)
    n = len(qids)
    dj, dd = [], []
    for _ in range(B):
        s = [qids[rng.randrange(n)] for _ in range(n)]
        sj = sbs(jv, s)
        sa = sbs(av, s)
        dj.append(sj)
        dd.append(sj - sa)
    dj.sort(); dd.sort()
    ci = lambda d: (round(d[int(.025*B)], 1), round(d[int(.975*B)], 1))
    return ci(dj), ci(dd)


def main():
    content_votes, design_votes, sample = load()
    qids = [q for q in content_votes if q in sample]
    av = {q: assessor_verdict(sample[q]) for q in qids}
    sbs_ass = sbs(av, qids)
    print(f"N={len(qids)}, SbS(assessors)={sbs_ass:+.1f}%")

    # === CONTENT ANALYSIS ===
    measurements = ["A_baseline", "B_faithful", "C_multiprompt", "D_consensus"]
    content_results = {}
    for meas in measurements:
        jv = {q: content_verdict(content_votes[q].get(meas, [])) for q in qids}
        ag, nv = agreement(jv, av, qids)
        sj = sbs(jv, qids)
        ci_j, ci_d = bootstrap_ci(jv, av, qids)
        equiv = ci_d[0] > -DELTA and ci_d[1] < DELTA
        content_results[meas] = {
            "n": nv, "agreement": round(ag, 3), "sbs_nps": round(sj, 1),
            "ci_judge": list(ci_j), "delta_ci": list(ci_d), "equiv": equiv,
        }
        print(f"  {meas:20} n={nv:3} agree={ag:.0%} SbS={sj:+.1f}% "
              f"CI{ci_j} d{ci_d} {'ЭКВИВ' if equiv else 'не экв'}")

    # === DESIGN ANALYSIS (C.R.A.P.) ===
    print("\nC.R.A.P. дизайн-оценки:")
    design_results = {}
    for side in ["YandexAI", "Gpt"]:
        side_data = defaultdict(list)
        for qid in qids:
            for d in design_votes[qid].get(side, []):
                if d.get("design_total") is not None:
                    side_data[d["rerun"]].append(d["design_total"])

        all_scores = []
        for rerun_scores in side_data.values():
            all_scores.extend(rerun_scores)

        if all_scores:
            mean_score = sum(all_scores) / len(all_scores)
            per_crit = {}
            for crit in DESIGN_CRITS:
                vals = [d.get(crit) for d in design_votes[qid].get(side, [])
                        for qid in qids if d.get(crit) is not None]
                per_crit[crit] = round(sum(vals)/len(vals), 2) if vals else 0
            design_results[side] = {
                "mean_total": round(mean_score, 2),
                "per_criterion": per_crit,
                "n": len(all_scores),
            }
            print(f"  {side:10} mean={mean_score:.1f}/8, per-crit={per_crit}")

    # === CHARTS ===
    # 1. Ablation ladder
    labels = ["A: Baseline", "B: Anti-bias", "C: +Multi-prompt", "D: +Consensus"]
    ags = [content_results[m]["agreement"] * 100 for m in measurements]
    dlt = [abs(content_results[m]["sbs_nps"] - sbs_ass) for m in measurements]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(4), ags, color="#1d9e75", alpha=0.85)
    for i, a in enumerate(ags):
        ax1.text(i, a + 1, f"{a:.0f}%", ha="center")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Agreement с асессорами, %")
    ax1.set_title("Ablation study: согласие по замерам")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(range(4), dlt, color="#d85a30", alpha=0.85)
    for i, d in enumerate(dlt):
        ax2.text(i, d + 0.5, f"{d:.1f}", ha="center")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("|Δ SbS-NPS| к асессорам, п.п.")
    ax2.set_title("Отклонение от эталона (меньше — лучше)")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "full-eval-ablation.jpeg", dpi=130)
    plt.close(fig)

    # 2. C.R.A.P. comparison
    if design_results:
        sides = list(design_results.keys())
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(DESIGN_CRITS))
        width = 0.35
        for i, side in enumerate(sides):
            vals = [design_results[side]["per_criterion"].get(c, 0) for c in DESIGN_CRITS]
            ax.bar(x + i*width, vals, width, label=side, alpha=0.85)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([c.capitalize() for c in DESIGN_CRITS])
        ax.set_ylabel("Средний балл (0–2)")
        ax.set_title("C.R.A.P. дизайн-оценки по сторонам")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "full-eval-crap.jpeg", dpi=130)
        plt.close(fig)

    # Save results
    out = {
        "n": len(qids), "sbs_assessors": round(sbs_ass, 1),
        "content": content_results, "design": design_results,
    }
    (OUT / "full_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {OUT / 'full_analysis.json'}")
    print(f"Графики: full-eval-ablation.jpeg, full-eval-crap.jpeg")


if __name__ == "__main__":
    main()
