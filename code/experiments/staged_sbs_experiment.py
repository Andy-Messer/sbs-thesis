"""Реальный staged SbS-эксперимент на golden set.

Сравнивает схемы LLM-as-a-Judge (classic / multi-prompt / consensus) с
асессорской разметкой по метрике SbS-NPS с bootstrap-CI.

Данные: code/baskets/golden-set/golden_set_full.xlsx -- 200 запросов, для
каждого есть ссылки на ответы двух систем (S = YandexAI, C = Gpt),
асессорские ранги и вердикт (ydx_win / gpt_win / draw).

Пайплайн:
  Фаза A. Для каждого запроса вытащить текст ответа обеих систем (Playwright),
          закэшировать в responses_cache.json.
  Фаза B. Для каждой пары (модель-судья a, промпт p) получить SbS-голос
          v_{a,p}(q) in {S, C, draw}. Голоса пишутся инкрементально в
          votes.jsonl (устойчиво к прерыванию).
  Фаза C. Из голосов собираются схемы и считаются метрики (отдельный
          скрипт analyze_staged.py).

L = 4 модели-судьи, M = 3 формулировки промпта.
Порядок ответов A/B рандомизируется на запрос (seed=query_id), чтобы снять
систематическое позиционное смещение, и фиксируется для всех судей.
"""
import sys
import json
import time
import random
from pathlib import Path

import requests
import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from monitoring import SystemMonitor

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_XLSX = ROOT / "code/baskets/golden-set/golden_set_full.xlsx"
OUT = Path(__file__).parent / "staged_results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "responses_cache.json"
VOTES = OUT / "votes.jsonl"

OLLAMA = "http://localhost:11434/api/generate"
MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen3:1.7b"]
N_QUERIES = 50

# Три семантически эквивалентные формулировки промпта судьи (M=3),
# различающиеся глубиной явного рассуждения (см. раздел 2 обзора).
PROMPTS = {
    "p1_compact": (
        "Ты оцениваешь, какой из двух ответов на запрос пользователя лучше.\n"
        "Запрос: \"{q}\"\n\nОтвет A:\n{a}\n\nОтвет B:\n{b}\n\n"
        "Кто отвечает лучше по релевантности, точности, полноте и читаемости? "
        "Ответь СТРОГО JSON без пояснений: {{\"winner\": \"A\"|\"B\"|\"draw\"}}"
    ),
    "p2_criteria": (
        "Сравни два ответа на запрос пользователя по 4 критериям СТРОГО В ПОРЯДКЕ: "
        "Relevance, Accuracy, Completeness, Readability.\n"
        "Запрос: \"{q}\"\n\nОтвет A:\n{a}\n\nОтвет B:\n{b}\n\n"
        "Для каждого критерия определи, какой ответ сильнее, затем вынеси общий вердикт. "
        "Ответь СТРОГО JSON: {{\"winner\": \"A\"|\"B\"|\"draw\"}}"
    ),
    "p3_reason": (
        "Ты — внимательный асессор. Сначала кратко опиши сильные и слабые стороны "
        "каждого ответа, затем реши, какой лучше.\n"
        "Запрос: \"{q}\"\n\nОтвет A:\n{a}\n\nОтвет B:\n{b}\n\n"
        "После рассуждения дай финальную строку СТРОГО в формате JSON: "
        "{{\"winner\": \"A\"|\"B\"|\"draw\"}}"
    ),
}


def load_sample():
    df = pd.read_excel(GOLDEN_XLSX)

    def httpok(s):
        return df[s].astype(str).str.startswith("http")

    mask = (httpok("ydx_url") & httpok("gpt_url")
            & df["YandexAI_rank"].notna() & df["Gpt_rank"].notna()
            & df["verdict"].notna())
    sub = df[mask].drop_duplicates("query_id").copy()
    # стратифицированно по вердикту асессоров
    rows = []
    per = {"draw": N_QUERIES // 2, "gpt_win": N_QUERIES // 3,
           "ydx_win": N_QUERIES - N_QUERIES // 2 - N_QUERIES // 3}
    for v, n in per.items():
        cand = sub[sub["verdict"] == v]
        rows.append(cand.sample(min(n, len(cand)), random_state=42))
    s = pd.concat(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    return s


def extract_text(page, url, query):
    try:
        page.goto(url, timeout=30000)
        page.wait_for_timeout(2200)
        try:
            txt = page.inner_text("main")
        except Exception:
            txt = page.inner_text("body")
        if not txt or len(txt) < 50:
            txt = page.inner_text("body")
        # вырезаем окно вокруг последнего вхождения запроса (отсекаем навигацию)
        key = query[:25]
        idx = txt.rfind(key)
        if idx > 0:
            txt = txt[idx:]
        return " ".join(txt.split())[:2500]
    except Exception as e:
        return f"[extract error: {e}]"


def phase_a_extract(sample):
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    todo = [r for _, r in sample.iterrows() if str(r["query_id"]) not in cache]
    if not todo:
        print(f"[A] кэш полон: {len(cache)} запросов")
        return cache
    print(f"[A] извлекаю тексты для {len(todo)} запросов...")
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1100, "height": 1500})
        for i, r in enumerate(todo):
            qid = str(r["query_id"])
            q = str(r["last_query"])
            a = extract_text(page, r["ydx_url"], q)   # S = YandexAI
            c = extract_text(page, r["gpt_url"], q)   # C = Gpt
            cache[qid] = {"query": q, "ydx": a, "gpt": c,
                          "verdict": r["verdict"],
                          "ydx_rank": float(r["YandexAI_rank"]),
                          "gpt_rank": float(r["Gpt_rank"])}
            if (i + 1) % 5 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"  [A] {i+1}/{len(todo)} (ydx={len(a)} gpt={len(c)} симв.)")
        b.close()
    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[A] готово: {len(cache)} запросов в кэше")
    return cache


def parse_winner(text):
    try:
        s = text.rfind("{"); e = text.rfind("}") + 1
        d = json.loads(text[s:e])
        w = str(d.get("winner", "")).strip().upper()
        return w if w in ("A", "B", "DRAW") else None
    except Exception:
        return None


def call_judge(model, prompt):
    try:
        r = requests.post(OLLAMA, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": 400},
        }, timeout=300)
        return r.json().get("response", "")
    except Exception as e:
        return f"[error {e}]"


def done_keys():
    done = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            done.add((d["qid"], d["model"], d["prompt"]))
    return done


def phase_b_judge(cache, sample, monitor):
    done = done_keys()
    total = len(sample) * len(MODELS) * len(PROMPTS)
    print(f"[B] всего пар (запрос×модель×промпт): {total}, уже сделано: {len(done)}")
    n = 0
    with open(VOTES, "a") as f:
        for _, r in sample.iterrows():
            qid = str(r["query_id"])
            if qid not in cache:
                continue
            ce = cache[qid]
            # фиксированный для запроса порядок A/B (снимаем позиционное смещение)
            rnd = random.Random(int(r["query_id"]) & 0xffffffff)
            swap = rnd.random() < 0.5
            if swap:   # A = Gpt(C), B = YandexAI(S)
                a_txt, b_txt = ce["gpt"], ce["ydx"]
                a_side, b_side = "C", "S"
            else:      # A = YandexAI(S), B = Gpt(C)
                a_txt, b_txt = ce["ydx"], ce["gpt"]
                a_side, b_side = "S", "C"
            for model in MODELS:
                for pname, ptmpl in PROMPTS.items():
                    n += 1
                    if (qid, model, pname) in done:
                        continue
                    prompt = ptmpl.format(q=ce["query"][:400],
                                          a=a_txt, b=b_txt)
                    t0 = time.time()
                    raw = call_judge(model, prompt)
                    w = parse_winner(raw)
                    # переводим A/B обратно в S/C/draw
                    if w == "A":
                        vote = a_side
                    elif w == "B":
                        vote = b_side
                    elif w == "DRAW":
                        vote = "draw"
                    else:
                        vote = None
                    rec = {"qid": qid, "model": model, "prompt": pname,
                           "vote": vote, "winner_ab": w,
                           "sec": round(time.time() - t0, 1)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    snap = monitor.snapshot(label=f"{model}:{pname}")
                    monitor.heartbeat(0, model, n)
                    if n % 10 == 0:
                        print(f"  [B] {n}/{total} {model}/{pname} qid={qid} "
                              f"vote={vote} ({rec['sec']}s) "
                              f"CPU={snap['cpu_percent']:.0f}% RAM={snap['ram_percent']:.0f}%",
                              flush=True)
    print("[B] голосование завершено")


def main():
    print("=" * 64)
    print("  Staged SbS experiment: classic / multi-prompt / consensus vs assessors")
    print("=" * 64)
    sample = load_sample()
    print(f"Выборка: {len(sample)} запросов; вердикты: "
          f"{sample['verdict'].value_counts().to_dict()}")
    cache = phase_a_extract(sample)
    monitor = SystemMonitor(str(OUT / "monitoring"))
    monitor.snapshot(label="start")
    phase_b_judge(cache, sample, monitor)
    monitor.snapshot(label="end")
    plot = monitor.save_plot("staged_load.jpeg")
    sample[["query_id", "verdict", "YandexAI_rank", "Gpt_rank"]].to_json(
        OUT / "sample.json", orient="records", force_ascii=False)
    print(f"График: {plot}")
    print(f"Голоса: {VOTES}")
    print("Готово. Анализ: analyze_staged.py")


if __name__ == "__main__":
    main()
