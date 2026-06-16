"""Анализ честного judge: пер-критериальное согласие с golden + вердикты.

1) Покритериальное согласие (Relevance/Accuracy/Completeness/Readability)
   консенсуса судей с реальными ассессорскими чекбоксами golden set, по S и C.
2) Вердиктные метрики на train/test: theta калибруется на train, отчёт на test
   (agreement, SbS-NPS, bootstrap-CI, H1/H2/TOST).
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

import analyze_staged as A

OUT = Path(__file__).parent / "staged_results"
VOTES = OUT / "votes_faithful.jsonl"
CACHE = OUT / "cache_v2.json"
GOLD = Path(__file__).resolve().parents[2] / "code/baskets/golden-set/golden_set_full.xlsx"
CRITS = ["relevance", "accuracy", "completeness", "readability"]
DELTA = 10.0
THETAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load():
    rows = defaultdict(list)         # qid -> list of records
    votes = defaultdict(dict)        # qid -> {(model,prompt): vote}
    for line in VOTES.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["qid"]].append(d)
            votes[d["qid"]][(d["model"], d["prompt"])] = d["vote"]
    cache = json.loads(CACHE.read_text())
    return rows, votes, cache


def per_criterion_agreement(rows, cache, df):
    """Согласие консенсуса (мажоритар по критерию) с golden, по S и C."""
    agg = {side: {c: [0, 0] for c in CRITS} for side in ("S", "C")}  # [ok, tot]
    for qid, recs in rows.items():
        gid = int(qid)
        if gid not in df.index:
            continue
        grow = df.loc[gid]
        for side, model in (("S", "YandexAI"), ("C", "Gpt")):
            key = "s_checks" if side == "S" else "c_checks"
            for ci, crit in enumerate(CRITS):
                gv = grow.get(f"{model}_{crit}")
                if pd.isna(gv):
                    continue
                gv = bool(gv)
                votes_ci = [r[key][ci] for r in recs if r.get(key)]
                if not votes_ci:
                    continue
                maj = sum(votes_ci) >= len(votes_ci) / 2
                agg[side][crit][0] += (maj == gv)
                agg[side][crit][1] += 1
    return agg


def verdict(vlist, theta):
    vs = [v for v in vlist if v in ("S", "C", "draw")]
    if not vs:
        return None
    ws = sum(v == "S" for v in vs); wc = sum(v == "C" for v in vs)
    th = theta * len(vs)
    if ws >= th and ws > wc:
        return "S"
    if wc >= th and wc > ws:
        return "C"
    return "draw"


def sbs(verd, qids):
    n = len(qids)
    if not n:
        return 0.0
    return (sum(verd.get(q) == "S" for q in qids)
            - sum(verd.get(q) == "C" for q in qids)) / n * 100


def boot(jv, av, qids, B=10000):
    rng = random.Random(5); n = len(qids); dj = []; dd = []
    for _ in range(B):
        s = [qids[rng.randrange(n)] for _ in range(n)]
        x = sbs(jv, s); y = sbs(av, s); dj.append(x); dd.append(x - y)
    dj.sort(); dd.sort()
    ci = lambda d: (round(d[int(.025*B)], 1), round(d[int(.975*B)], 1))
    return ci(dj), ci(dd)


def main():
    rows, votes, cache = load()
    df = pd.read_excel(GOLD).drop_duplicates("query_id").set_index("query_id")
    qids = list(votes.keys())
    av = {q: A.assessor_verdict({"verdict": cache[q]["verdict"]}) for q in qids}
    train = [q for q in qids if cache[q].get("split") == "train"]
    test = [q for q in qids if cache[q].get("split") == "test"]
    print(f"N={len(qids)} (train={len(train)}, test={len(test)})\n")

    print("Пер-критериальное согласие консенсуса с golden (ассессорские чекбоксы):")
    agg = per_criterion_agreement(rows, cache, df)
    for side in ("S", "C"):
        line = "  " + side + ": " + "  ".join(
            f"{c[:4].capitalize()}={agg[side][c][0]}/{agg[side][c][1]}"
            f"({(agg[side][c][0]/agg[side][c][1]*100 if agg[side][c][1] else 0):.0f}%)"
            for c in CRITS)
        print(line)

    # калибровка theta на train (по вердиктному согласию consensus l=4)
    def c4(q, th):
        return verdict([votes[q].get((m, p)) for m in A.CONS4
                        for p in faithful_prompts()], th)
    best_t, best_a = 0.7, -1
    for t in THETAS:
        ok = tot = 0
        for q in train:
            jv = c4(q, t)
            if jv is not None:
                tot += 1; ok += (jv == av[q])
        a = ok / tot if tot else 0
        print(f"  theta={t}: train agreement(consensus)={a:.0%}")
        if a > best_a:
            best_a, best_t = a, t
    print(f"Выбран theta={best_t}\n")

    sbs_ass = sbs(av, test)
    print(f"TEST (theta={best_t}): SbS-NPS(асессоры)={sbs_ass:+.1f}%")
    out = {"theta": best_t, "n_test": len(test), "sbs_assessors": round(sbs_ass, 1),
           "per_criterion": {s: {c: agg[s][c] for c in CRITS} for s in ("S", "C")},
           "schemes": {}}
    schemes = {
        "Классическая": lambda q: [votes[q].get(("qwen2.5:7b", "c2_stoprule"))],
        "Multi-prompt": lambda q: [votes[q].get(("qwen2.5:7b", p)) for p in faithful_prompts()],
        "Consensus l=4": lambda q: [votes[q].get((m, p)) for m in A.CONS4 for p in faithful_prompts()],
    }
    for nm, fn in schemes.items():
        jv = {q: verdict(fn(q), best_t) for q in test}
        nv = sum(jv[q] is not None for q in test)
        ok = sum(jv[q] is not None and jv[q] == av[q] for q in test)
        s = sbs(jv, test); cij, cid = boot(jv, av, test)
        equiv = cid[0] > -DELTA and cid[1] < DELTA
        out["schemes"][nm] = {"n": nv, "agreement": round(ok/nv, 3) if nv else 0,
            "sbs_nps": round(s, 1), "ci": cij, "delta_ci": cid, "equiv": equiv}
        print(f"  {nm:15} n={nv:2} agree={ok/nv if nv else 0:.0%} SbS={s:+.1f} "
              f"CI{cij} d{cid} {'ЭКВИВ' if equiv else 'не экв'}")
    (OUT / "analysis_faithful.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nСохранено: analysis_faithful.json")


def faithful_prompts():
    return ["c1_compact", "c2_stoprule", "c3_reason"]


if __name__ == "__main__":
    main()
