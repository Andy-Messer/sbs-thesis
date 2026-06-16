"""Полная оценка v2: исправленные промпты + все модели multimodal.

Исправления v1→v2:
1. A_baseline: явный JSON-формат вместо текстового "A/B/DRAW"
2. D_consensus: все модели — qwen2.5vl (multimodal), не текстовые
3. C.R.A.P.: более чёткий промпт с конкретными критериями
4. UI_EXCLUDE: явное указание игнорировать сайдбар и системный промпт
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
VOTES = OUT / "full_evaluation_votes_v2.jsonl"
OLLAMA = "http://localhost:11434/api/generate"
TEMP = 0.0
RERUNS = 3

UI_EXCLUDE = (
    "ВАЖНО: Оценивай ТОЛЬКО содержание ответа (область с текстом/виджетом "
    "справа). Сайдбар, навигация, системный промпт, название модели и "
    "элементы интерфейса платформы НЕ учитываются при оценке.\n\n"
)

CONTENT_PROMPTS = {
    "B_faithful": (
        UI_EXCLUDE +
        "Ты ассессор. Для КАЖДОГО ответа (A и B) пройди 4 чекбокса "
        "В ПОРЯДКЕ relevance→accuracy→completeness→readability. "
        "Первый false обнуляет последующие. НЕ награждай за длину. "
        "Ответь СТРОГО JSON:\n"
        '{{"A":{{"relevance":true,"accuracy":true,"completeness":true,"readability":true}},'
        '"B":{{"relevance":true,"accuracy":true,"completeness":true,"readability":true}}}}\n\n'
        "Query: {q}\n\nResponse A:\n[screenshot]\n\nResponse B:\n[screenshot]"
    ),
    "C_compact": (
        UI_EXCLUDE +
        "Ты ассессор. Оцени КАЖДЫЙ ответ по 4 критериям строго по порядку. "
        "Как только false — все последующие false для этого ответа.\n\n"
        "Query: {q}\n\nA:\n[screenshot]\n\nB:\n[screenshot]\n\n"
        'Ответь JSON: {{"A":{{"relevance":...}},"B":{{"relevance":...}}}}'
    ),
    "C_reason": (
        UI_EXCLUDE +
        "Ты опытный асессор. По каждому ответу подумай по 4 критериям "
        "(relevance, accuracy, completeness, readability), поставь true/false "
        "(после false — остальные false).\n\n"
        "Query: {q}\n\nA:\n[screenshot]\n\nB:\n[screenshot]\n\n"
        'JSON: {{"A":{{...}},"B":{{...}}}}'
    ),
}

DESIGN_PROMPT = (
    UI_EXCLUDE +
    "Оцени визуальное качество ответа по 4 принципам дизайна Робин Уильямс. "
    "Каждый принцип: 0=плохо, 1=нормально, 2=хорошо.\n\n"
    "Критерии оценки:\n"
    "- Contrast (контраст): различимы ли заголовки, основной текст, акценты, таблицы. "
    "Если всё одного цвета/размера — 0. Если явно выделены роли элементов — 2.\n"
    "- Repetition (повторение): единый стиль оформления (буллеты, отступы, заголовки). "
    "Если стиль хаотичный — 0. Если всё единообразно — 2.\n"
    "- Alignment (выравнивание): элементы выровнены по линиям, таблицы ровные. "
    "Если рваные края/столбцы — 0. Если всё аккуратно — 2.\n"
    "- Proximity (близость): связанные элементы сгруппированы, между блоками воздух. "
    "Если всё слито — 0. Если логичные группы — 2.\n\n"
    "Reply ONLY as JSON:\n"
    '{{"contrast":0-2,"repetition":0-2,"alignment":0-2,"proximity":0-2}}\n\n'
    "Query: {q}\n\nResponse:\n[screenshot]"
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
        return {k: min(2, max(0, int(d.get(k, 0)))) for k in
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
                          d.get("variant", ""), d.get("side", "")))
    return done


def main():
    sample = json.loads(SAMPLE.read_text())
    print(f"Sample: {len(sample)} queries")

    done = done_keys()
    # Content: faithful(1) + compact(1) + reason(1) = 3 variants × 1 model × 3 reruns = 9 per query
    # Design: 2 sides × 3 reruns = 6 per query
    total = len(sample) * (3 * 1 * 3 + 2 * 3)
    print(f"Total calls: {total}")
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
            gpt_img = base64.b64encode(shot_gpt.read_bytes()).decode()
            ydx_img = base64.b64encode(shot_ydx.read_bytes()).decode()

            # === КОНТЕНТНАЯ ОЦЕНКА (B_faithful + C_compact + C_reason) ===
            for variant in ["B_faithful", "C_compact", "C_reason"]:
                for rerun in range(RERUNS):
                    key = (qid, "content", rerun, variant, "")
                    if key in done:
                        continue

                    prompt = CONTENT_PROMPTS[variant].format(q=q["last_query"][:300])
                    t0 = time.time()
                    raw = call_ollama("qwen2.5vl:7b", prompt, [ydx_img, gpt_img])
                    sec = round(time.time() - t0, 1)

                    ca, cb = parse_content(raw)
                    vote = None
                    if ca is not None:
                        vote = verdict_from_checks(ca, cb, swap)

                    rec = {
                        "qid": qid, "verdict_gold": q["verdict"],
                        "eval_type": "content", "variant": variant,
                        "model": "qwen2.5vl:7b", "rerun": rerun,
                        "vote": vote,
                        "c_s": count_stop(ca if not swap else cb) if ca else None,
                        "c_c": count_stop(cb if not swap else ca) if cb else None,
                        "sec": sec,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1

            # === ДИЗАЙНЕРСКАЯ ОЦЕНКА (C.R.A.P.) ===
            for side, img_b64, side_name in [
                ("A", ydx_img, "YandexAI"), ("B", gpt_img, "Gpt")]:
                for rerun in range(RERUNS):
                    key = (qid, "design", rerun, "", side)
                    if key in done:
                        continue

                    prompt = DESIGN_PROMPT.format(q=q["last_query"][:300])
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

            if (qi + 1) % 10 == 0:
                print(f"  [{qi+1}/{len(sample)}] {n} calls done")

    print(f"\nГотово: {n} записей в {VOTES}")


if __name__ == "__main__":
    main()
