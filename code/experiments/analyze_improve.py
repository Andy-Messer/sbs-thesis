"""Графики итеративных улучшений судьи (доказательство выводов диссертации).

Совмещает faithful (агрегационная лестница) и improve (промпт-инжиниринг)
прогоны; считает на TEST согласие с асессорами и |Δ SbS-NPS| для каждой
конфигурации-итерации и строит:
  improve-ladder.jpeg   -- согласие с асессорами по итерациям;
  improve-percrit.jpeg  -- покритериальное согласие base vs verbosity;
  improve-delta.jpeg    -- |Δ SbS-NPS| к асессорам по итерациям.
"""
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_staged as A

OUT = Path(__file__).parent / "staged_results"
CACHE = OUT / "cache_v2.json"
GOLD = Path(__file__).resolve().parents[2] / "code/baskets/golden-set/golden_set_full.xlsx"
FIG = Path(__file__).resolve().parents[2] / "figures"
CRITS = ["relevance", "accuracy", "completeness", "readability"]
FP = ["c1_compact", "c2_stoprule", "c3_reason"]
THETA = 0.7


def read_votes(path):
    v = defaultdict(dict); rows = defaultdict(list)
    if not path.exists():
        return v, rows
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            key = d.get("prompt")
            v[d["qid"]][key if "model" not in d else (d["model"], key)] = d["vote"]
            rows[d["qid"]].append(d)
    return v, rows


def verdict(vlist, th=THETA):
    vs = [x for x in vlist if x in ("S", "C", "draw")]
    if not vs:
        return None
    ws = sum(x == "S" for x in vs); wc = sum(x == "C" for x in vs)
    t = th * len(vs)
    return "S" if (ws >= t and ws > wc) else "C" if (wc >= t and wc > ws) else "draw"


def main():
    fv = defaultdict(dict)
    for line in (OUT / "votes_faithful.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line); fv[d["qid"]][(d["model"], d["prompt"])] = d["vote"]
    iv, irows = read_votes(OUT / "votes_improve.jsonl")
    cache = json.loads(CACHE.read_text())
    df = pd.read_excel(GOLD).drop_duplicates("query_id").set_index("query_id")
    test = [q for q in cache if cache[q].get("split") == "test"]
    av = {q: A.assessor_verdict({"verdict": cache[q]["verdict"]}) for q in test}

    def agree(verd):
        nv = sum(verd[q] is not None for q in test)
        ok = sum(verd[q] is not None and verd[q] == av[q] for q in test)
        return (ok / nv * 100) if nv else 0

    def sbs(verd):
        n = len(test)
        return (sum(verd.get(q) == "S" for q in test)
                - sum(verd.get(q) == "C" for q in test)) / n * 100 if n else 0

    sbs_ass = sbs(av)

    # конфигурации-итерации
    configs = {}
    configs["faithful\nsingle"] = {q: verdict([fv[q].get(("qwen2.5:7b", "c2_stoprule"))]) for q in test}
    configs["+ multi-\nprompt"] = {q: verdict([fv[q].get(("qwen2.5:7b", p)) for p in FP]) for q in test}
    configs["+ consensus\nl=4"] = {q: verdict([fv[q].get((m, p)) for m in A.CONS4 for p in FP]) for q in test}
    if iv:
        configs["+ verbosity\ncorr."] = {q: verdict([iv[q].get("verbosity")]) for q in test}
        configs["+ few-shot"] = {q: verdict([iv[q].get("fewshot")]) for q in test}

    names = list(configs)
    agr = [agree(configs[n]) for n in names]
    dlt = [abs(sbs(configs[n]) - sbs_ass) for n in names]

    # 1) лестница согласия
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(range(len(names)), agr, color="#1d9e75", alpha=0.85)
    for i, a in enumerate(agr):
        ax.text(i, a + 1, f"{a:.0f}%", ha="center")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("согласие с асессорами на test, %")
    ax.set_title("Итеративные улучшения судьи: согласие с асессорами")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "improve-ladder.jpeg", dpi=130); plt.close(fig)

    # 2) |Δ SbS-NPS| к асессорам
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(names)), dlt, color="#d85a30", alpha=0.85)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("|Δ SbS-NPS| к асессорам, п.п.")
    ax.set_title("Отклонение от асессорского замера по итерациям (меньше — лучше)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "improve-delta.jpeg", dpi=130); plt.close(fig)

    # 3) покритериально base vs verbosity (consensus-мажор по критерию)
    def percrit(rows_src, variant):
        agg = {c: [0, 0] for c in CRITS}
        for qid in test:
            gid = int(qid)
            if gid not in df.index:
                continue
            recs = [r for r in rows_src.get(qid, []) if r.get("prompt") == variant]
            for side, model, key in (("S", "YandexAI", "s_checks"), ("C", "Gpt", "c_checks")):
                for ci, cr in enumerate(CRITS):
                    gv = df.loc[gid].get(f"{model}_{cr}")
                    if pd.isna(gv):
                        continue
                    vals = [r[key][ci] for r in recs if r.get(key)]
                    if not vals:
                        continue
                    maj = sum(vals) >= len(vals) / 2
                    agg[cr][0] += (maj == bool(gv)); agg[cr][1] += 1
        return [agg[c][0] / agg[c][1] * 100 if agg[c][1] else 0 for c in CRITS]

    out = {"sbs_assessors": round(sbs_ass, 1),
           "configs": {n: {"agreement": round(agr[i], 1), "abs_delta": round(dlt[i], 1)}
                       for i, n in enumerate(names)}}
    if iv:
        pc_base = percrit(irows, "base"); pc_verb = percrit(irows, "verbosity")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = range(len(CRITS)); w = 0.38
        ax.bar([i - w/2 for i in x], pc_base, w, label="base", color="#888")
        ax.bar([i + w/2 for i in x], pc_verb, w, label="+verbosity corr.", color="#1d9e75")
        ax.set_xticks(list(x)); ax.set_xticklabels([c.capitalize()[:4] for c in CRITS])
        ax.set_ylabel("покритериальное согласие с golden, %")
        ax.set_title("Эффект коррекции verbosity по критериям")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "improve-percrit.jpeg", dpi=130); plt.close(fig)
        out["percrit_base"] = [round(v, 1) for v in pc_base]
        out["percrit_verbosity"] = [round(v, 1) for v in pc_verb]

    (OUT / "improve.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("configs:", json.dumps(out["configs"], ensure_ascii=False))
    print("saved improve-ladder/-delta/-percrit.jpeg")


if __name__ == "__main__":
    main()
