"""Честный judge: ассессорский Step-1 протокол (чекбоксы по порядку, стоп
на первом No, каждый ответ независимо) -> вердикт из сравнения c_S vs c_C.

Отличия от winner-промпта (v1/v2):
  * судья НЕ сравнивает напрямую, а заполняет 4 чекбокса по КАЖДОМУ ответу
    в порядке Relevance->Accuracy->Completeness->Readability со «стопом»;
  * вердикт выводится из числа пройденных чекбоксов, как у ассессоров;
  * сохраняются пер-критериальные чекбоксы -> сверка с golden по критериям.

Вход -- cache_v2.json (починенное селекторное извлечение). L=4, M=3.
Голоса -> votes_faithful.jsonl.
"""
import sys
import json
import time
import random
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from monitoring import SystemMonitor

OUT = Path(__file__).parent / "staged_results"
CACHE = OUT / "cache_v2.json"
VOTES = OUT / "votes_faithful.jsonl"
OLLAMA = "http://localhost:11434/api/generate"
MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen3:1.7b"]
CRITS = ["relevance", "accuracy", "completeness", "readability"]

_BASE = """Запрос: "{q}"

Ответ A:
{a}

Ответ B:
{b}

Критерии (строго в этом порядке):
- relevance: реально отвечает на запрос (или уместно уточняет / объясняет отказ); истинность здесь не важна.
- accuracy: фактически верно, без явных ошибок, галлюцинаций и противоречий; источники надёжны, если они есть.
- completeness: покрывает запрос полностью — ключевые пункты, нужный формат, язык, единицы, актуальность.
- readability: легко найти суть — структура, списки/заголовки, читаемые формулы; не свалка ссылок и эмодзи.

Ответь СТРОГО JSON:
{{"A":{{"relevance":true,"accuracy":true,"completeness":true,"readability":true}},"B":{{"relevance":true,"accuracy":true,"completeness":true,"readability":true}}}}"""

# Три эквивалентные формулировки протокола (multi-prompt M=3)
PROMPTS = {
    "c1_compact": (
        "Ты ассессор. Оцени КАЖДЫЙ из двух ответов НЕЗАВИСИМО по 4 критериям "
        "строго по порядку. Как только по ответу ставишь false — все последующие "
        "критерии этого ответа тоже false. Не сравнивай ответы между собой.\n\n" + _BASE),
    "c2_stoprule": (
        "Ты выступаешь как асессор разметки. Для каждого ответа (A и B) пройди "
        "4 чекбокса В ПОРЯДКЕ relevance→accuracy→completeness→readability. Правило "
        "стопа: первый false обнуляет все последующие критерии этого же ответа. "
        "Каждый ответ оценивается сам по себе, независимо от другого.\n\n" + _BASE),
    "c3_reason": (
        "Ты опытный асессор. По каждому ответу отдельно кратко подумай по 4 "
        "критериям в порядке, поставь true/false (после первого false — остальные "
        "false), и верни итог. Оценивай ответы независимо друг от друга.\n\n" + _BASE),
}


def call(model, prompt):
    try:
        r = requests.post(OLLAMA, json={"model": model, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.0, "num_predict": 250}},
            timeout=300)
        return r.json().get("response", "")
    except Exception as e:
        return f"[err {e}]"


def parse(text):
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        d = json.loads(text[s:e])
        def checks(side):
            sd = d.get(side, {})
            return [bool(sd.get(c)) for c in CRITS]
        return checks("A"), checks("B")
    except Exception:
        return None, None


def count_stop(bools):
    c = 0
    for b in bools:
        if b:
            c += 1
        else:
            break
    return c


def done_keys():
    done = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["qid"], d["model"], d["prompt"]))
    return done


def main():
    cache = json.loads(CACHE.read_text())
    qids = list(cache.keys())
    done = done_keys()
    total = len(qids) * len(MODELS) * len(PROMPTS)
    print(f"Честный judge: {len(qids)} запросов × {len(MODELS)} × {len(PROMPTS)} = {total}; "
          f"сделано {len(done)}")
    mon = SystemMonitor(str(OUT / "monitoring_faithful"))
    mon.snapshot("start")
    n = 0
    with open(VOTES, "a") as f:
        for qid in qids:
            ce = cache[qid]
            rnd = random.Random(int(qid) & 0xffffffff)
            swap = rnd.random() < 0.5
            a_txt, b_txt, a_side, b_side = (
                (ce["gpt"], ce["ydx"], "C", "S") if swap
                else (ce["ydx"], ce["gpt"], "S", "C"))
            for model in MODELS:
                for pname, ptmpl in PROMPTS.items():
                    n += 1
                    if (qid, model, pname) in done:
                        continue
                    t0 = time.time()
                    raw = call(model, ptmpl.format(q=ce["query"][:300],
                                                   a=a_txt[:2000], b=b_txt[:2000]))
                    ca, cb = parse(raw)
                    if ca is None:
                        s_checks = c_checks = None; vote = None; cs = cc = None
                    else:
                        s_checks = ca if a_side == "S" else cb
                        c_checks = cb if a_side == "S" else ca
                        cs = count_stop(s_checks); cc = count_stop(c_checks)
                        vote = "S" if cs > cc else "C" if cc > cs else "draw"
                    f.write(json.dumps({"qid": qid, "model": model, "prompt": pname,
                        "c_s": cs, "c_c": cc, "s_checks": s_checks,
                        "c_checks": c_checks, "vote": vote,
                        "sec": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
                    f.flush()
                    snap = mon.snapshot(f"{model}:{pname}"); mon.heartbeat(0, model, n)
                    if n % 20 == 0:
                        print(f"  {n}/{total} {model}/{pname} vote={vote} "
                              f"cs={cs} cc={cc} ({round(time.time()-t0,1)}s) "
                              f"CPU={snap['cpu_percent']:.0f}%", flush=True)
    mon.snapshot("end"); mon.save_plot("faithful_load.jpeg")
    print("Готово. Анализ: analyze_faithful.py")


if __name__ == "__main__":
    main()
