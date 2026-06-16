"""Анализ результатов staged SbS-эксперимента.

Читает staged_results/votes.jsonl и sample.json, собирает схемы
(classic / multi-prompt / consensus l=3 / consensus l=4), считает
согласие с асессорами и SbS-NPS с bootstrap-CI. Печатает готовые
числа для таблиц главы результатов и сохраняет analysis.json.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

OUT = Path(__file__).parent / "staged_results"
VOTES = OUT / "votes.jsonl"
SAMPLE = OUT / "sample.json"
CACHE = OUT / "responses_cache.json"

PROMPTS = ["p1_compact", "p2_criteria", "p3_reason"]
CLASSIC = ("qwen2.5:7b", "p2_criteria")
MP_MODEL = "qwen2.5:7b"
CONS3 = ["qwen2.5:3b", "qwen2.5:7b", "qwen3:1.7b"]
CONS4 = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen3:1.7b"]
THETA = 0.7
DELTA_MARGIN = 10.0  # порог эквивалентности (п.п.) для TOST, eq:tost


def load():
    votes = defaultdict(dict)  # qid -> {(model,prompt): vote}
    for line in VOTES.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        votes[d["qid"]][(d["model"], d["prompt"])] = d["vote"]
    sample = {str(r["query_id"]): r for r in json.loads(SAMPLE.read_text())}
    return votes, sample


def verdict_from_votes(vote_list):
    """Пороговое голосование eq:verdict при theta=0.7*|valid|."""
    vs = [v for v in vote_list if v in ("S", "C", "draw")]
    if not vs:
        return None
    ws = sum(1 for v in vs if v == "S")
    wc = sum(1 for v in vs if v == "C")
    th = THETA * len(vs)
    if ws >= th and ws > wc:
        return "S"
    if wc >= th and wc > ws:
        return "C"
    return "draw"


def assessor_verdict(row):
    return {"ydx_win": "S", "gpt_win": "C", "draw": "draw"}[row["verdict"]]


def scheme_verdicts(votes, scheme):
    out = {}
    for qid, vd in votes.items():
        if scheme == "classic":
            vl = [vd.get(CLASSIC)]
        elif scheme == "multiprompt":
            vl = [vd.get((MP_MODEL, p)) for p in PROMPTS]
        elif scheme == "cons3":
            vl = [vd.get((m, p)) for m in CONS3 for p in PROMPTS]
        elif scheme == "cons4":
            vl = [vd.get((m, p)) for m in CONS4 for p in PROMPTS]
        out[qid] = verdict_from_votes(vl)
    return out


def sbs_nps(verdicts, qids):
    n = len(qids)
    if n == 0:
        return 0.0
    s = sum(1 for q in qids if verdicts.get(q) == "S")
    c = sum(1 for q in qids if verdicts.get(q) == "C")
    return (s - c) / n * 100


def agreement(jv, av, qids):
    valid = [q for q in qids if jv.get(q) is not None]
    if not valid:
        return 0.0, 0
    a = sum(1 for q in valid if jv[q] == av[q]) / len(valid)
    return a, len(valid)


def bootstrap(qids, jv, av, B=10000):
    rng = random.Random(123)
    qids = list(qids)
    n = len(qids)
    djudge, dass, ddelta = [], [], []
    for _ in range(B):
        samp = [qids[rng.randrange(n)] for _ in range(n)]
        sj = sbs_nps(jv, samp)
        sa = sbs_nps(av, samp)
        djudge.append(sj); dass.append(sa); ddelta.append(sj - sa)
    def ci(d):
        d = sorted(d)
        return d[int(0.025 * B)], d[int(0.975 * B)]
    return ci(djudge), ci(dass), ci(ddelta)


def paired_agreement_gain(qids, jvA, jvB, av, B=10000):
    """Парный bootstrap прироста согласия с асессорами: agree(A) - agree(B)."""
    rng = random.Random(99)
    qids = [q for q in qids if jvA.get(q) is not None and jvB.get(q) is not None]
    n = len(qids)
    if n == 0:
        return (0.0, 0.0), 0.0
    diffs = []
    for _ in range(B):
        samp = [qids[rng.randrange(n)] for _ in range(n)]
        aA = sum(1 for q in samp if jvA[q] == av[q]) / n
        aB = sum(1 for q in samp if jvB[q] == av[q]) / n
        diffs.append((aA - aB) * 100)
    diffs.sort()
    return (diffs[int(0.025 * B)], diffs[int(0.975 * B)]), sum(diffs) / len(diffs)


def pct(x):
    return f"{x:+.1f}"


def main():
    votes, sample = load()
    qids = [q for q in votes if q in sample]
    av = {q: assessor_verdict(sample[q]) for q in qids}

    schemes = {
        "classic": scheme_verdicts(votes, "classic"),
        "multiprompt": scheme_verdicts(votes, "multiprompt"),
        "cons3": scheme_verdicts(votes, "cons3"),
        "cons4": scheme_verdicts(votes, "cons4"),
    }
    sbs_ass = sbs_nps(av, qids)
    res = {"n_queries": len(qids), "sbs_nps_assessors": round(sbs_ass, 1),
           "schemes": {}}
    names = {"classic": "Классический (1 судья, 1 промпт)",
             "multiprompt": "Multi-prompt (1 судья, 3 промпта)",
             "cons3": "Consensus (3 судьи x 3 промпта)",
             "cons4": "Consensus (4 судьи x 3 промпта)"}
    print(f"N = {len(qids)} запросов; SbS-NPS(асессоры) = {sbs_ass:+.1f}%\n")
    print(f"{'Схема':<34} {'Agree':>6} {'SbS-NPS':>8} {'CI(SbS)':>16} {'dDelta CI vs ass':>20}")
    for key, jv in schemes.items():
        ag, nv = agreement(jv, av, qids)
        sj = sbs_nps(jv, qids)
        ci_j, ci_a, ci_d = bootstrap(qids, jv, av)
        res["schemes"][key] = {
            "name": names[key], "agreement": round(ag, 3), "n_valid": nv,
            "sbs_nps": round(sj, 1),
            "ci_judge": [round(ci_j[0], 1), round(ci_j[1], 1)],
            "ci_delta_vs_assessors": [round(ci_d[0], 1), round(ci_d[1], 1)],
        }
        h1 = "H1: значимо!=0" if not (ci_j[0] <= 0 <= ci_j[1]) else "H1: не отлич. от 0"
        equiv = (ci_d[0] > -DELTA_MARGIN and ci_d[1] < DELTA_MARGIN)
        res["schemes"][key]["h1_reject_zero"] = not (ci_j[0] <= 0 <= ci_j[1])
        res["schemes"][key]["equivalent_to_assessors"] = equiv
        print(f"{names[key]:<34} {ag*100:5.0f}% {sj:+7.1f}% "
              f"[{ci_j[0]:+.1f},{ci_j[1]:+.1f}]  "
              f"d[{ci_d[0]:+.1f},{ci_d[1]:+.1f}] "
              f"{h1}; {'ЭКВИВ (TOST)' if equiv else 'не экв.'}")

    # H2: парные приросты согласия между соседними схемами + classic vs assessors
    print("\nH2 (парный bootstrap прироста согласия, 95% CI):")
    transitions = [("multiprompt", "classic"), ("cons3", "multiprompt"),
                   ("cons4", "cons3"), ("cons4", "classic")]
    res["h2_pairs"] = {}
    for a, b in transitions:
        ci, mean = paired_agreement_gain(qids, schemes[a], schemes[b], av)
        sig = not (ci[0] <= 0 <= ci[1])
        res["h2_pairs"][f"{a}_vs_{b}"] = {
            "mean_gain_pp": round(mean, 1),
            "ci": [round(ci[0], 1), round(ci[1], 1)], "significant": sig}
        print(f"  agree({a}) - agree({b}) = {mean:+.1f} п.п. "
              f"[{ci[0]:+.1f},{ci[1]:+.1f}] {'ЗНАЧИМО' if sig else 'не значимо'}")
    res["delta_margin_pp"] = DELTA_MARGIN

    # verbosity bias: связь длины ответа с предпочтением судьи
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
        longer_pref = 0; tot = 0
        for q in qids:
            jvv = schemes["cons4"][q]
            if jvv not in ("S", "C"):
                continue
            ce = cache.get(q)
            if not ce:
                continue
            longer = "S" if len(ce["ydx"]) >= len(ce["gpt"]) else "C"
            tot += 1
            if jvv == longer:
                longer_pref += 1
        res["verbosity_longer_pref_rate"] = round(longer_pref / tot, 3) if tot else None
        print(f"\nVerbosity: судья (cons4) предпочёл более длинный ответ в "
              f"{longer_pref}/{tot} = {(longer_pref/tot*100 if tot else 0):.0f}% неничейных случаев")

    (OUT / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {OUT/'analysis.json'}")


if __name__ == "__main__":
    main()
