"""Шаг 2: Ablation Study — 4 замера на N=200 мультимодальных скриншотах.

Замеры:
  A: Raw Baseline — winner-промпт, 1 модель
  B: Faithful Single — ассессорский чек-лист + anti-bias, 1 модель
  C: Multi-prompt — 3 промпта, 1 модель
  D: Consensus — 3 модели × 3 промпта = 9 голосов

Вход: sample_v4_200.json + design_shots_v4/*.webp
Выход: ablation_results.json + ablation_votes.jsonl
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
VOTES = OUT / "ablation_votes.jsonl"
RESULTS = OUT / "ablation_results.json"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"
TEMP = 0.0
RERUNS = 3  # для замера остаточной стохастичности

CRITS = ["relevance", "accuracy", "completeness", "readability"]

# --- Промпты ---
# ВАЖНО: судья оценивает ТОЛЬКО содержание ответа (область справа).
# Сайдбар, навигация, системный промпт, элементы интерфейса платформы
# НЕ учитываются при оценке.
UI_EXCLUDE = (
    "ВАЖНО: Оценивай ТОЛЬКО содержание ответа (область с текстом/виджетом "
    "справа). Сайдбар, навигация, системный промпт, название модели и "
    "элементы интерфейса платформы НЕ учитываются при оценке.\n\n"
)
PROMPTS = {
    "winner": (
        UI_EXCLUDE +
        "Compare these two responses. Which is better? "
        "Reply ONLY with: A, B, or DRAW.\n\n"
        "Query: {q}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
    ),
    "faithful": (
        UI_EXCLUDE +
        "Ты ассессор. Для КАЖДОГО ответа (A и B) пройди 4 чекбокса "
        "В ПОРЯДКЕ relevance→accuracy→completeness→readability. "
        "Первый false обнуляет последующие. НЕ награждай за длину. "
        "Ответь СТРОГО JSON:\n"
        '{{"A":{{"relevance":true,...}},"B":{{"relevance":true,...}}}}\n\n'
        "Query: {q}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
    ),
    "compact": (
        UI_EXCLUDE +
        "Ты ассессор. Оцени КАЖДЫЙ ответ по 4 критериям строго по порядку. "
        "Как только false — все последующие false для этого ответа.\n\n"
        "Query: {q}\n\nA:\n{a}\n\nB:\n{b}\n\n"
        'Ответь JSON: {{"A":{{"relevance":...}},"B":{{"relevance":...}}}}'
    ),
    "reason": (
        UI_EXCLUDE +
        "Ты опытный асессор. По каждому ответу подумай по 4 критериям "
        "(relevance, accuracy, completeness, readability), поставь true/false "
        "(после false — остальные false).\n\n"
        "Query: {q}\n\nA:\n{a}\n\nB:\n{b}\n\n"
        'JSON: {{"A":{{...}},"B":{{...}}}}'
    ),
}


def call_ollama(model, prompt, image_path=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMP, "num_predict": 250},
    }
    if image_path:
        img = base64.b64encode(Path(image_path).read_bytes()).decode()
        payload["images"] = [img]
    for attempt in range(2):
        try:
            r = requests.post(OLLAMA, json=payload, timeout=(10, 60))
            return r.json().get("response", "")
        except Exception as e:
            if attempt == 0:
                time.sleep(1)
                continue
            return f"[err {e}]"


def parse_checkboxes(text):
    try:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        s = t.find("{")
        e = t.rfind("}") + 1
        d = json.loads(t[s:e])
        KEY_MAP = {"helpfulness": "readability", "readability": "readability"}
        def checks(side):
            sd = d.get(side, {})
            if isinstance(sd, bool):
                return [sd] * len(CRITS)
            if not isinstance(sd, dict):
                return None
            return [bool(sd.get(KEY_MAP.get(c, c))) for c in CRITS]
        a, b = checks("A"), checks("B")
        if a is None or b is None:
            return None, None
        return a, b
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


def parse_winner(text):
    t = text.strip().upper()
    if "DRAW" in t or "TIE" in t:
        return "draw"
    if "A" in t[:5] and "B" not in t[:5]:
        return "A"
    if "B" in t[:5] and "A" not in t[:5]:
        return "B"
    return None


def verdict_from_checks(ca, cb, swap):
    cs = count_stop(ca if not swap else cb)
    cc = count_stop(cb if not swap else ca)
    if cs > cc:
        return "S"
    elif cc > cs:
        return "C"
    return "draw"


def run_single(query, shot_a, shot_b, prompt_key, swap, model=MODEL):
    pt = PROMPTS[prompt_key]
    q = query["last_query"][:300]
    if prompt_key == "winner":
        prompt = pt.format(q=q, a="[screenshot A]", b="[screenshot B]")
    else:
        prompt = pt.format(q=q, a="[screenshot A]", b="[screenshot B]")
    # For multimodal: use image of the response
    # Actually we need to send the screenshot as image
    raw = call_ollama(model, prompt, shot_a)
    if raw.startswith("[err"):
        return {"vote": None, "raw": raw[:200]}
    if prompt_key == "winner":
        w = parse_winner(raw)
        vote = "S" if (w == "A" and not swap) or (w == "B" and swap) else \
               "C" if (w == "B" and not swap) or (w == "A" and swap) else "draw"
        return {"vote": vote, "raw": raw[:200]}
    else:
        ca, cb = parse_checkboxes(raw)
        if ca is None:
            return {"vote": None, "raw": raw[:200]}
        vote = verdict_from_checks(ca, cb, swap)
        return {"vote": vote, "c_s": count_stop(ca if not swap else cb),
                "c_c": count_stop(cb if not swap else ca), "raw": raw[:200]}


def load_done():
    done = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["qid"], d["measurement"], d.get("rerun", 0)))
    return done


def main():
    sample = json.loads(SAMPLE.read_text())
    print(f"Sample: {len(sample)} queries")

    done = load_done()
    total = len(sample) * (1 + 1 + 1 + 9) * RERUNS  # A(1) + B(1) + C(3) + D(9)
    print(f"Total calls: {total}, done: {len(done)}")

    measurements = [
        ("A_baseline", ["winner"], [MODEL]),
        ("B_faithful", ["faithful"], [MODEL]),
        ("C_multiprompt", ["compact", "faithful", "reason"], [MODEL]),
        ("D_consensus", ["compact", "faithful", "reason"],
         ["qwen2.5vl:7b", "qwen2.5vl:3b", "moondream:latest"]),
    ]

    n = 0
    with open(VOTES, "a") as f:
        for q in sample:
            qid = str(q["query_id"])
            shot_gpt = SHOTS / f"{qid}_gpt.png"
            shot_ydx = SHOTS / f"{qid}_ydx.png"
            if not shot_gpt.exists() or not shot_ydx.exists():
                continue

            rng = random.Random(int(qid) & 0xffffffff)
            swap = rng.random() < 0.5
            shot_a = str(shot_ydx if swap else shot_gpt)
            shot_b = str(shot_gpt if swap else shot_ydx)

            for meas_name, prompts, models in measurements:
                for prompt_key in prompts:
                    for model in models:
                        for rerun in range(RERUNS):
                            n += 1
                            key = (qid, meas_name, rerun)
                            if key in done:
                                continue
                            t0 = time.time()
                            res = run_single(q, shot_a, shot_b, prompt_key, swap, model)
                            rec = {
                                "qid": qid, "verdict_gold": q["verdict"],
                                "measurement": meas_name, "model": model,
                                "prompt": prompt_key, "rerun": rerun,
                                "vote": res["vote"],
                                "c_s": res.get("c_s"), "c_c": res.get("c_c"),
                                "raw": res.get("raw", ""),
                                "sec": round(time.time() - t0, 1),
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                            if n % 50 == 0:
                                print(f"  [{n}/{total}] {meas_name}/{model}/{prompt_key} "
                                      f"vote={res['vote']} ({rec['sec']}s)")

    print(f"Готово: {n} записей в {VOTES}")


if __name__ == "__main__":
    main()
