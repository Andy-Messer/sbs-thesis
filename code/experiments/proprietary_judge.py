"""Шаг 3: Capacity Test — сравнение open-source vs proprietaryjudge.

Прогоняет идентичный пайплайн (ассессорский чек-лист) на:
  - Open-source: qwen2.5vl:7b (локальный Ollama)
  - Proprietary: GPT-4o-mini (OpenAI API)

На N=50 случайных скриншотах из sample_v4_200.
Сравнивает: agreement, SbS-NPS, coverage, latency, cost.
"""
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
VOTES = OUT / "capacity_votes.jsonl"
RESULTS = OUT / "capacity_analysis.json"

OLLAMA = "http://localhost:11434/api/generate"
OPENAI_API = "https://api.openai.com/v1/chat/completions"
# OPENAI_KEY must be set in environment or .env

N = 50
SEED = 77
CRITS = ["relevance", "accuracy", "completeness", "readability"]

PROMPT = (
    "Ты ассессор. Для КАЖДОГО ответа (A и B) пройди 4 чекбокса "
    "В ПОРЯДКЕ relevance→accuracy→completeness→readability. "
    "Первый false обнуляет последующие. НЕ награждай ответ за длину. "
    "Ответь СТРОГО JSON:\n"
    '{{"A":{{"relevance":true,"accuracy":true,"completeness":true,"readability":true}},'
    '"B":{{"relevance":true,"accuracy":true,"completeness":true,"readability":true}}}}\n\n'
    "Query: {q}\n\nResponse A:\n{a}\n\nResponse B:\n{b}"
)


def call_ollama(prompt, image_path):
    img = base64.b64encode(Path(image_path).read_bytes()).decode()
    r = requests.post(OLLAMA, json={
        "model": "qwen2.5vl:7b", "prompt": prompt, "images": [img],
        "stream": False, "options": {"temperature": 0.0, "num_predict": 250}},
        timeout=120)
    return r.json().get("response", "")


def call_openai(prompt, image_path):
    import os
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return "[err: no OPENAI_API_KEY]"
    img = base64.b64encode(Path(image_path).read_bytes()).decode()
    r = requests.post(OPENAI_API, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img}"}}
            ]}],
            "temperature": 0.0, "max_tokens": 250},
        timeout=60)
    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")


def parse_checkboxes(text):
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
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


def verdict_from_checks(ca, cb, swap):
    cs = count_stop(ca if not swap else cb)
    cc = count_stop(cb if not swap else ca)
    if cs > cc:
        return "S"
    elif cc > cs:
        return "C"
    return "draw"


def done_keys():
    done = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["qid"], d["provider"]))
    return done


def main():
    import os
    sample = json.loads(SAMPLE.read_text())
    rng = random.Random(SEED)
    subset = rng.sample(sample, min(N, len(sample)))
    print(f"Capacity test: {len(subset)} queries")

    done = done_keys()
    providers = [
        ("ollama", call_ollama),
        ("openai", call_openai),
    ]

    with open(VOTES, "a") as f:
        for q in subset:
            qid = str(q["query_id"])
            shot_gpt = SHOTS / f"{qid}_gpt.png"
            shot_ydx = SHOTS / f"{qid}_ydx.png"
            if not shot_gpt.exists() or not shot_ydx.exists():
                continue

            rng2 = random.Random(int(qid) & 0xffffffff)
            swap = rng2.random() < 0.5
            shot_a = str(shot_ydx if swap else shot_gpt)
            shot_b = str(shot_gpt if swap else shot_ydx)

            for prov_name, call_fn in providers:
                if (qid, prov_name) in done:
                    continue
                q_text = q["last_query"][:300]
                prompt = PROMPT.format(q=q_text, a="[screenshot]", b="[screenshot]")
                t0 = time.time()
                raw = call_fn(prompt, shot_a)
                sec = round(time.time() - t0, 1)
                ca, cb = parse_checkboxes(raw)
                vote = None
                if ca is not None:
                    vote = verdict_from_checks(ca, cb, swap)
                rec = {
                    "qid": qid, "verdict_gold": q["verdict"],
                    "provider": prov_name, "vote": vote,
                    "c_s": count_stop(ca if not swap and ca else cb if cb else None),
                    "c_c": count_stop(cb if not swap and cb else ca if ca else None),
                    "sec": sec,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {prov_name:8} qid={qid[:12]} vote={vote} ({sec}s)")

    print(f"Готово: {VOTES}")


if __name__ == "__main__":
    main()
