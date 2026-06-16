"""Полная оценка N=200: контент (A/B/C/D) + дизайн (C.R.A.P.).

Два типа оценок:
1. Контентная (ассессорский чек-лист): relevance/accuracy/completeness/readability
   Замеры: A_baseline, B_faithful, C_multiprompt, D_consensus
2. Дизайнерская (C.R.A.P.): contrast/repetition/alignment/proximity
   По шкале {0,1,2} для каждого принципа, суммарно {0..8}

Вход: sample_v4_200.json + design_shots_v4/*.png
Выход: full_evaluation_votes.jsonl
"""
import sys
import json
import time
import base64
import random
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "code/experiments/staged_results/sample_v4_200.json"
SHOTS = ROOT / "code/experiments/design_shots_v4"
OUT = ROOT / "code/experiments/staged_results"
VOTES = OUT / "full_evaluation_votes.jsonl"
OLLAMA = "http://localhost:11434/api/generate"
TEMP = 0.0
RERUNS = 3

UI_EXCLUDE = (
    "ВАЖНО: Оценивай ТОЛЬКО содержание ответа (область с текстом/виджетом "
    "справа). Сайдбар, навигация, системный промпт, название модели и "
    "элементы интерфейса платформы НЕ учитываются при оценке.\n\n"
)

# --- Контентные промпты (A/B/C/D) ---
CONTENT_PROMPTS = {
    "A_baseline": (
        "Compare these two responses. Which is better? "
        "Reply ONLY with: A, B, or DRAW.\n\n"
        "Query: {q}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
    ),
    "B_faithful": (
        UI_EXCLUDE +
        "Ты ассессор. Для КАЖДОГО ответа (A и B) пройди 4 чекбокса "
        "В ПОРЯДКЕ relevance→accuracy→completeness→readability. "
        "Первый false обнуляет последующие. НЕ награждай за длину. "
        "Ответь СТРОГО JSON:\n"
        '{{"A":{{"relevance":true,...}},"B":{{"relevance":true,...}}}}\n\n'
        "Query: {q}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
    ),
    "C_multiprompt_compact": (
        UI_EXCLUDE +
        "Ты ассессор. Оцени КАЖДЫЙ ответ по 4 критериям строго по порядку. "
        "Как только false — все последующие false для этого ответа.\n\n"
        "Query: {q}\n\nA:\n{a}\n\nB:\n{b}\n\n"
        'Ответь JSON: {{"A":{{"relevance":...}},"B":{{"relevance":...}}}}'
    ),
    "C_multiprompt_reason": (
        UI_EXCLUDE +
        "Ты опытный асессор. По каждому ответу подумай по 4 критериям "
        "(relevance, accuracy, completeness, readability), поставь true/false "
        "(после false — остальные false).\n\n"
        "Query: {q}\n\nA:\n{a}\n\nB:\n{b}\n\n"
        'JSON: {{"A":{{...}},"B":{{...}}}}'
    ),
}

# --- Дизайнерский промпт (C.R.A.P.) ---
DESIGN_PROMPT = (
    UI_EXCLUDE +
    "Оцени визуальное качество ответа по 4 принципам дизайна. "
    "Каждый принцип оценивается по шкале: 0 (плохо), 1 (нормально), 2 (хорошо).\n\n"
    "Принципы:\n"
    "- Contrast (контраст): различимы ли элементы разной роли (заголовки, текст, акценты)\n"
    "- Repetition (повторение): единый стиль (буллеты, отступы, заголовки)\n"
    "- Alignment (выравнивание): элементы выровнены, таблицы ровные\n"
    "- Proximity (близость): связанные элементы сгруппированы\n\n"
    "Reply ONLY as JSON:\n"
    '{{"contrast":0-2,"repetition":0-2,"alignment":0-2,"proximity":0-2}}\n\n'
    "Query: {q}\n\nResponse to evaluate:\n{screenshot}"
)


def call_ollama(model, prompt, images=None):
    payload = {
        "model": model, "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMP, "num_predict": 250}
    }
    if images:
        payload["images"] = images
    try:
        r = requests.post(OLLAMA, json=payload, timeout=120)
        return r.json().get("response", "")
    except Exception as e:
        return f"[err {e}]"


def parse_content(text):
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        d = json.loads(text[s:e])
        def checks(side):
            sd = d.get(side, {})
            return [bool(sd.get(c)) for c in
                    ["relevance", "accuracy", "completeness", "readability"]]
        return checks("A"), checks("B")
    except Exception:
        return None, None


def parse_design(text):
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        d = json.loads(text[s:e])
        return {k: int(d.get(k, 0)) for k in
                ["contrast", "repetition", "alignment", "proximity"]}
    except Exception:
        return None


def count_stop(bools):
    c = 0
    for b in bools:
        if b: c += 1
        else: break
    return c


def verdict_from_checks(ca, cb, swap):
    cs = count_stop(ca if not swap else cb)
    cc = count_stop(cb if not swap else ca)
    if cs > cc: return "S"
    elif cc > cs: return "C"
    return "draw"


def done_keys():
    done = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["qid"], d["eval_type"], d.get("rerun", 0),
                          d.get("variant", "")))
    return done


def main():
    sample = json.loads(SAMPLE.read_text())
    print(f"Sample: {len(sample)} queries")

    done = done_keys()
    # Content: 4 measurements × 3 reruns × 200 = 2400
    # Design: 2 sides × 3 reruns × 200 = 1200
    total_content = len(sample) * 4 * RERUNS
    total_design = len(sample) * 2 * RERUNS
    total = total_content + total_design
    print(f"Total calls: {total} (content={total_content}, design={total_design})")
    print(f"Already done: {len(done)}")

    n = 0
    with open(VOTES, "a") as f:
        for qi, q in enumerate(sample):
            qid = str(q["query_id"])
            shot_gpt = SHOTS / f"{qid}_gpt.png"
            shot_ydx = SHOTS / f"{qid}_ydx.png"
            if not shot_gpt.exists() or not shot_ydx.exists():
                continue

            rng = random.Random(int(qid) & 0xffffffff)
            swap = rng.random() < 0.5
            shot_a = str(shot_ydx if swap else shot_gpt)
            shot_b = str(shot_gpt if swap else shot_ydx)
            gpt_img = base64.b64encode(shot_gpt.read_bytes()).decode()
            ydx_img = base64.b64encode(shot_ydx.read_bytes()).decode()

            # === КОНТЕНТНАЯ ОЦЕНКА (A/B/C/D) ===
            content_configs = [
                ("A_baseline", ["A_baseline"], ["qwen2.5vl:7b"]),
                ("B_faithful", ["B_faithful"], ["qwen2.5vl:7b"]),
                ("C_multiprompt", ["C_multiprompt_compact", "C_multiprompt_reason"], ["qwen2.5vl:7b"]),
                ("D_consensus", ["C_multiprompt_compact", "C_multiprompt_reason"],
                 ["qwen2.5:3b", "qwen2.5:7b", "qwen3:1.7b"]),
            ]

            for meas_name, variants, models in content_configs:
                for variant in variants:
                    for model in models:
                        for rerun in range(RERUNS):
                            key = (qid, "content", rerun, f"{meas_name}:{variant}:{model}")
                            if key in done:
                                continue

                            prompt = CONTENT_PROMPTS[variant].format(
                                q=q["last_query"][:300],
                                a="[screenshot A]", b="[screenshot B]")
                            t0 = time.time()
                            raw = call_ollama(model, prompt, [ydx_img, gpt_img])
                            sec = round(time.time() - t0, 1)

                            ca, cb = parse_content(raw)
                            vote = None
                            if ca is not None:
                                vote = verdict_from_checks(ca, cb, swap)

                            rec = {
                                "qid": qid, "verdict_gold": q["verdict"],
                                "eval_type": "content", "measurement": meas_name,
                                "variant": variant, "model": model, "rerun": rerun,
                                "vote": vote,
                                "c_s": count_stop(ca if not swap else cb) if ca else None,
                                "c_c": count_stop(cb if not swap else ca) if cb else None,
                                "sec": sec,
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                            n += 1
                            if n % 100 == 0:
                                print(f"  [{n}/{total}] {meas_name}/{model} vote={vote} ({sec}s)")

            # === ДИЗАЙНЕРСКАЯ ОЦЕНКА (C.R.A.P.) ===
            for side, img_b64, side_name in [
                ("A", ydx_img, "YandexAI"), ("B", gpt_img, "Gpt")]:
                for rerun in range(RERUNS):
                    key = (qid, "design", rerun, side)
                    if key in done:
                        continue

                    prompt = DESIGN_PROMPT.format(
                        q=q["last_query"][:300],
                        screenshot=f"[screenshot {side}]")
                    t0 = time.time()
                    raw = call_ollama("qwen2.5vl:7b", prompt, [img_b64])
                    sec = round(time.time() - t0, 1)

                    design = parse_design(raw)
                    total_score = sum(design.values()) if design else None

                    rec = {
                        "qid": qid, "verdict_gold": q["verdict"],
                        "eval_type": "design", "side": side_name,
                        "rerun": rerun,
                        "contrast": design.get("contrast") if design else None,
                        "repetition": design.get("repetition") if design else None,
                        "alignment": design.get("alignment") if design else None,
                        "proximity": design.get("proximity") if design else None,
                        "design_total": total_score,
                        "sec": sec,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    if n % 100 == 0:
                        print(f"  [{n}/{total}] design/{side_name} score={total_score} ({sec}s)")

            if (qi + 1) % 20 == 0:
                print(f"--- Query {qi+1}/{len(sample)} done ---")

    print(f"\nГотово: {n} записей в {VOTES}")


if __name__ == "__main__":
    main()
