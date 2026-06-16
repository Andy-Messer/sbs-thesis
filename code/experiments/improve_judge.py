"""Итеративные улучшения судьи методологиями диссертации.

Прогоняет qwen2.5:7b на корзине (cache_v2, train/test) с тремя
прогрессивно улучшаемыми формулировками чек-лист-протокола:
  base       -- ассессорский чек-лист (как в faithful);
  verbosity  -- base + явная коррекция verbosity-смещения
                (не награждать длину; лишние ссылки/эмодзи -- минус);
  fewshot    -- base + few-shot примеры разметки из train.

Каждая -- пер-ответные чекбоксы по порядку со «стопом», независимо.
Голоса -> votes_improve.jsonl (model, prompt-вариант, c_s, c_c, checks).
"""
import sys
import json
import time
import random
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import faithful_judge_run as FJ
from monitoring import SystemMonitor

OUT = Path(__file__).parent / "staged_results"
CACHE = OUT / "cache_v2.json"
VOTES = OUT / "votes_improve.jsonl"
MODEL = "qwen2.5:7b"

_HEAD = {
    "base": "Ты выступаешь как асессор разметки. Для каждого ответа (A и B) "
            "пройди 4 чекбокса В ПОРЯДКЕ relevance→accuracy→completeness→readability. "
            "Первый false обнуляет последующие критерии этого ответа. Оценивай "
            "ответы независимо друг от друга.",
    "verbosity": "Ты выступаешь как асессор разметки. Для каждого ответа (A и B) "
            "пройди 4 чекбокса В ПОРЯДКЕ relevance→accuracy→completeness→readability "
            "(первый false обнуляет последующие). ВАЖНО: НЕ награждай ответ за длину "
            "и объём — короткий, но точный и полный ответ НЕ хуже длинного; лишние "
            "ссылки, повторы и эмодзи — это минус к readability, а не плюс. Конкретный "
            "точный ответ предпочтительнее общего «обзорного». Оценивай ответы "
            "независимо друг от друга.",
    "fewshot": "Ты выступаешь как асессор разметки. Пройди по каждому ответу 4 "
            "чекбокса в порядке relevance→accuracy→completeness→readability (первый "
            "false обнуляет последующие), независимо для A и B.\n"
            "Примеры разметки:\n"
            "1) Запрос «сколько длится фильм X?». Ответ «около часа» вместо точного "
            "«1 ч 33 мин» → relevance=true, accuracy=false (далее false): неточно.\n"
            "2) Запрос «погода в Стамбуле на неделю, таблицей». Ответ даёт прогноз "
            "сплошным текстом без таблицы → relevance=true, accuracy=true, "
            "completeness=false (нет запрошенного формата), readability=false.\n"
            "3) Полный структурированный ответ строго по запросу → все true.",
}


def prompt(variant, q, a, b):
    return (_HEAD[variant] + "\n\n" + FJ._BASE.format(q=q, a=a, b=b))


def done_keys():
    d = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if line.strip():
                r = json.loads(line); d.add((r["qid"], r["prompt"]))
    return d


def main():
    cache = json.loads(CACHE.read_text())
    qids = list(cache.keys())
    done = done_keys()
    variants = list(_HEAD)
    total = len(qids) * len(variants)
    print(f"Improve: {len(qids)} запросов × {len(variants)} вариантов = {total}; "
          f"модель {MODEL}; сделано {len(done)}")
    mon = SystemMonitor(str(OUT / "monitoring_improve"))
    n = 0
    with open(VOTES, "a") as f:
        for qid in qids:
            ce = cache[qid]
            rnd = random.Random(int(qid) & 0xffffffff)
            swap = rnd.random() < 0.5
            a_txt, b_txt, a_side, b_side = (
                (ce["gpt"], ce["ydx"], "C", "S") if swap
                else (ce["ydx"], ce["gpt"], "S", "C"))
            for variant in variants:
                n += 1
                if (qid, variant) in done:
                    continue
                t0 = time.time()
                raw = FJ.call(MODEL, prompt(variant, ce["query"][:300],
                                            a_txt[:2000], b_txt[:2000]))
                ca, cb = FJ.parse(raw)
                if ca is None:
                    rec = {"qid": qid, "prompt": variant, "c_s": None, "c_c": None,
                           "s_checks": None, "c_checks": None, "vote": None}
                else:
                    sc = ca if a_side == "S" else cb
                    cc_ = cb if a_side == "S" else ca
                    cs = FJ.count_stop(sc); ccnt = FJ.count_stop(cc_)
                    rec = {"qid": qid, "prompt": variant, "c_s": cs, "c_c": ccnt,
                           "s_checks": sc, "c_checks": cc_,
                           "vote": "S" if cs > ccnt else "C" if ccnt > cs else "draw"}
                rec["sec"] = round(time.time() - t0, 1)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if n % 20 == 0:
                    snap = mon.snapshot()
                    print(f"  {n}/{total} {variant} vote={rec['vote']} "
                          f"({rec['sec']}s) CPU={snap['cpu_percent']:.0f}%", flush=True)
    print("Готово. Анализ: analyze_improve.py")


if __name__ == "__main__":
    main()
