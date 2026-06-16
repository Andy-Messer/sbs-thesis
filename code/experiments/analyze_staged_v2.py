"""Анализ v2: починенное извлечение + train/test + калибровка порога.

Калибрует порог голосования theta на train (по согласию с асессорами),
применяет к test и считает agreement, SbS-NPS, bootstrap-CI, H1/H2/TOST.
Печатает сравнение v1 (битый вход) vs v2 (починенный) на test.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import analyze_staged as A

OUT = Path(__file__).parent / "staged_results"
VOTES = OUT / "votes_v2.jsonl"
SAMPLE = OUT / "sample_v2.json"
DELTA = 10.0
THETAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load():
    votes = defaultdict(dict)
    for line in VOTES.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            votes[d["qid"]][(d["model"], d["prompt"])] = d["vote"]
    sample = {str(r["query_id"]): r for r in json.loads(SAMPLE.read_text())}
    return votes, sample


def verdict(votes_list, theta):
    vs = [v for v in votes_list if v in ("S", "C", "draw")]
    if not vs:
        return None
    ws = sum(1 for v in vs if v == "S"); wc = sum(1 for v in vs if v == "C")
    th = theta * len(vs)
    if ws >= th and ws > wc:
        return "S"
    if wc >= th and wc > ws:
        return "C"
    return "draw"


def cons4_votes(votes, qid):
    return [votes[qid].get((m, p)) for m in A.CONS4 for p in A.PROMPTS]


def agree(votes, qids, av, theta):
    ok = tot = 0
    for q in qids:
        jv = verdict(cons4_votes(votes, q), theta)
        if jv is None:
            continue
        tot += 1; ok += (jv == av[q])
    return ok / tot if tot else 0.0, tot


def sbs(verdicts, qids):
    n = len(qids)
    if not n:
        return 0.0
    s = sum(1 for q in qids if verdicts.get(q) == "S")
    c = sum(1 for q in qids if verdicts.get(q) == "C")
    return (s - c) / n * 100


def boot(jv, av, qids, B=10000):
    rng = random.Random(11); n = len(qids); dj = []; dd = []
    for _ in range(B):
        samp = [qids[rng.randrange(n)] for _ in range(n)]
        sj = sbs(jv, samp); sa = sbs(av, samp)
        dj.append(sj); dd.append(sj - sa)
    dj.sort(); dd.sort()
    ci = lambda d: (round(d[int(0.025*B)], 1), round(d[int(0.975*B)], 1))
    return ci(dj), ci(dd)


def main():
    votes, sample = load()
    qids = [q for q in votes if q in sample]
    av = {q: A.assessor_verdict(sample[q]) for q in qids}
    train = [q for q in qids if sample[q]["split"] == "train"]
    test = [q for q in qids if sample[q]["split"] == "test"]
    print(f"N={len(qids)} (train={len(train)}, test={len(test)})")

    # калибровка theta на train
    best_t, best_a = 0.7, -1
    for t in THETAS:
        a, _ = agree(votes, train, av, t)
        print(f"  theta={t}: train agreement={a:.0%}")
        if a > best_a:
            best_a, best_t = a, t
    print(f"Выбран theta={best_t} (train agreement={best_a:.0%})")

    # схемы на test при выбранном theta
    def scheme_verdicts(keyfn):
        return {q: verdict([votes[q].get(k) for k in keyfn], best_t) for q in test}
    classic = {q: verdict([votes[q].get(A.CLASSIC)], best_t) for q in test}
    mp = scheme_verdicts([(A.MP_MODEL, p) for p in A.PROMPTS])
    c4 = scheme_verdicts([(m, p) for m in A.CONS4 for p in A.PROMPTS])

    sbs_ass = sbs(av, test)
    print(f"\nTEST (theta={best_t}): SbS-NPS(асессоры)={sbs_ass:+.1f}%")
    out = {"sbs_assessors_test": round(sbs_ass, 1), "theta": best_t, "schemes": {}}
    for nm, jv in [("Классическая", classic), ("Multi-prompt", mp), ("Consensus l=4", c4)]:
        ag = sum(1 for q in test if jv[q] is not None and jv[q] == av[q])
        nv = sum(1 for q in test if jv[q] is not None)
        s = sbs(jv, test); cij, cid = boot(jv, av, test)
        equiv = cid[0] > -DELTA and cid[1] < DELTA
        out["schemes"][nm] = {"agreement": round(ag/nv, 3) if nv else 0, "n": nv,
                              "sbs_nps": round(s, 1), "ci": cij, "delta_ci": cid,
                              "equiv": equiv}
        print(f"  {nm:16} n={nv:2} agree={ag/nv if nv else 0:.0%} "
              f"SbS={s:+.1f} CI{cij} d{cid} "
              f"{'ЭКВИВ' if equiv else 'не экв'}")
    (OUT / "analysis_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nСохранено: analysis_v2.json")


if __name__ == "__main__":
    main()
