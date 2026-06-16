"""Staged SbS-эксперимент v2: ПОЧИНЕННОЕ извлечение ответов.

Отличия от v1 (staged_sbs_experiment.py):
  * извлечение по DOM-селекторам, а не по rfind-эвристике:
      Yandex AI: .FuturisDialogPair-MessageResponse
      GPT:       [data-message-author-role="assistant"]
    (см. диагностику: rfind хватал не тот фрагмент, ответ S недоизвлекался);
  * N = 100 запросов со стратифицированным train/test-сплитом (50/50):
    порог draw калибруется на train, отчёт -- на test;
  * имена кэша/голосов с суффиксом _v2.

Фазы и судьи (L=4, M=3, температура 0) -- как в v1.
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
import staged_sbs_experiment as v1   # переиспользуем промпты и парсер

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_XLSX = ROOT / "code/baskets/golden-set/golden_set_full.xlsx"
OUT = Path(__file__).parent / "staged_results"
CACHE = OUT / "cache_v2.json"
VOTES = OUT / "votes_v2.jsonl"
SAMPLE = OUT / "sample_v2.json"

MODELS = v1.MODELS
PROMPTS = v1.PROMPTS
N_TOTAL = 100

YDX_SEL = ".FuturisDialogPair-MessageResponse"
GPT_SEL = '[data-message-author-role="assistant"]'


def load_sample():
    df = pd.read_excel(GOLDEN_XLSX)

    def httpok(s):
        return df[s].astype(str).str.startswith("http")

    mask = (httpok("ydx_url") & httpok("gpt_url")
            & df["YandexAI_rank"].notna() & df["Gpt_rank"].notna()
            & df["verdict"].notna())
    sub = df[mask].drop_duplicates("query_id").copy()
    per = {"draw": 50, "gpt_win": 30, "ydx_win": 20}
    parts = []
    for v, n in per.items():
        cand = sub[sub["verdict"] == v].sample(min(n, len(sub[sub["verdict"] == v])),
                                               random_state=42)
        cand = cand.reset_index(drop=True)
        cand["split"] = ["train" if i % 2 == 0 else "test" for i in range(len(cand))]
        parts.append(cand)
    s = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
    return s


def extract(pg, url, q):
    pg.goto(url, timeout=30000)
    pg.wait_for_timeout(2200)
    low = url.lower()
    sel = YDX_SEL if ("yandex." in low and "conversation" in low) else GPT_SEL
    try:
        loc = pg.locator(sel).last
        if loc.count() if hasattr(loc, "count") else True:
            t = loc.inner_text(timeout=4000)
        else:
            t = ""
    except Exception:
        t = ""
    if not t or len(t) < 20:
        try:
            t = pg.inner_text("body")
        except Exception:
            t = ""
    t = " ".join(t.split())
    # У Yandex AI ответ-элемент содержит сверху блок источников
    # ("Yanıtın içeriği ... Daha fazla arama sonucu göster") -- это
    # метаданные, а не ответ; судья штрафует за них S. Отрезаем преамбулу.
    if "yandex." in low:
        marker = "Daha fazla arama sonucu göster"
        if marker in t:
            t = t.split(marker, 1)[1].strip()
        elif t.startswith("Yanıtın içeriği"):
            t = t[len("Yanıtın içeriği"):].strip()
    return t[:2500]


def phase_a(sample):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [r for _, r in sample.iterrows() if str(r["query_id"]) not in cache]
    if not todo:
        print(f"[A] кэш полон: {len(cache)}")
        return cache
    print(f"[A] селекторное извлечение для {len(todo)} запросов...")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1100, "height": 1500})
        for i, r in enumerate(todo):
            qid = str(r["query_id"]); q = str(r["last_query"])
            a = extract(pg, r["ydx_url"], q)
            c = extract(pg, r["gpt_url"], q)
            cache[qid] = {"query": q, "ydx": a, "gpt": c, "verdict": r["verdict"],
                          "split": r["split"]}
            if (i + 1) % 5 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"  [A] {i+1}/{len(todo)} ydx={len(a)} gpt={len(c)}")
        b.close()
    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"[A] готово: {len(cache)}")
    return cache


def done_keys():
    done = set()
    if VOTES.exists():
        for line in VOTES.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["qid"], d["model"], d["prompt"]))
    return done


def phase_b(cache, sample, monitor):
    done = done_keys()
    total = len(sample) * len(MODELS) * len(PROMPTS)
    print(f"[B] всего пар: {total}, сделано: {len(done)}")
    n = 0
    with open(VOTES, "a") as f:
        for _, r in sample.iterrows():
            qid = str(r["query_id"])
            if qid not in cache:
                continue
            ce = cache[qid]
            rnd = random.Random(int(r["query_id"]) & 0xffffffff)
            swap = rnd.random() < 0.5
            if swap:
                a_txt, b_txt, a_side, b_side = ce["gpt"], ce["ydx"], "C", "S"
            else:
                a_txt, b_txt, a_side, b_side = ce["ydx"], ce["gpt"], "S", "C"
            for model in MODELS:
                for pname, ptmpl in PROMPTS.items():
                    n += 1
                    if (qid, model, pname) in done:
                        continue
                    prompt = ptmpl.format(q=ce["query"][:400], a=a_txt, b=b_txt)
                    t0 = time.time()
                    raw = v1.call_judge(model, prompt)
                    w = v1.parse_winner(raw)
                    vote = (a_side if w == "A" else b_side if w == "B"
                            else "draw" if w == "DRAW" else None)
                    f.write(json.dumps({"qid": qid, "model": model, "prompt": pname,
                                        "vote": vote, "winner_ab": w,
                                        "sec": round(time.time() - t0, 1)},
                                       ensure_ascii=False) + "\n")
                    f.flush()
                    snap = monitor.snapshot(label=f"{model}:{pname}")
                    monitor.heartbeat(0, model, n)
                    if n % 20 == 0:
                        print(f"  [B] {n}/{total} {model}/{pname} vote={vote} "
                              f"({round(time.time()-t0,1)}s) CPU={snap['cpu_percent']:.0f}%",
                              flush=True)
    print("[B] голосование завершено")


def main():
    print("=" * 64)
    print("  Staged SbS v2: селекторное извлечение + train/test")
    print("=" * 64)
    sample = load_sample()
    print(f"Выборка: {len(sample)}; вердикты {sample['verdict'].value_counts().to_dict()}; "
          f"split {sample['split'].value_counts().to_dict()}")
    sample[["query_id", "verdict", "split", "YandexAI_rank", "Gpt_rank"]].to_json(
        SAMPLE, orient="records", force_ascii=False)
    cache = phase_a(sample)
    # быстрая проверка качества извлечения
    lens = [(len(c["ydx"]), len(c["gpt"])) for c in cache.values()]
    bad = sum(1 for a, _ in lens if a < 40)
    print(f"[A] медиана длины S={sorted(a for a,_ in lens)[len(lens)//2]}, "
          f"коротких S(<40)={bad}/{len(lens)}")
    monitor = SystemMonitor(str(OUT / "monitoring_v2"))
    monitor.snapshot(label="start")
    phase_b(cache, sample, monitor)
    monitor.snapshot(label="end")
    monitor.save_plot("staged_v2_load.jpeg")
    print("Готово. Анализ: analyze_staged_v2.py")


if __name__ == "__main__":
    main()
