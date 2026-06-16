"""Снижение дисперсии мультимодального судьи за счёт агрегации.

Демонстрирует ключевое свойство multi-prompt/consensus: дисперсия
агрегированной оценки убывает ~ σ²/n с числом голосов n.

Схема: для нескольких скриншотов ответов запускаем мультимодального судью
(qwen2.5vl) R раз при температуре > 0 (источник стохастичности — сама
модель). Каждый прогон даёт число пройденных чекбоксов c∈{0..4}. Для
каждого n=1..R оцениваем дисперсию среднего из n прогонов (усреднение по
случайным подвыборкам и по скриншотам) и сравниваем с теоретической σ²/n.

Скриншот — элемента-ответа на полную высоту (фикс inner-scroll).
Результат: figures/variance-reduction.jpeg + variance.json.
"""
import sys
import json
import base64
import time
import random
from pathlib import Path
from itertools import combinations

import requests
import pandas as pd
from playwright.sync_api import sync_playwright
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "code/baskets/golden-set/golden_set_full.xlsx"
SHOTS = Path(__file__).parent / "design_shots"
FIG = ROOT / "figures"
OUT = Path(__file__).parent / "staged_results"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"
GPT_SEL = '[data-message-author-role="assistant"]'
YDX_SEL = ".FuturisDialogPair-MessageResponse"
R = 8
TEMP = 0.8
N_SHOTS = 5

VIS_PROMPT = """На скриншоте — ответ модели на запрос пользователя.
Запрос: "{q}"
Оцени ответ по 4 критериям СТРОГО ПО ПОРЯДКУ (как только ставишь false —
все последующие тоже false): relevance, accuracy, completeness, readability.
Ответь СТРОГО JSON: {{"relevance":true,"accuracy":true,"completeness":true,"readability":true}}"""


def shoot(pg, url, out):
    low = url.lower()
    sel = YDX_SEL if ("yandex." in low and "conversation" in low) else GPT_SEL
    pg.goto(url, timeout=30000); pg.wait_for_timeout(2000)
    pg.locator(sel).last.screenshot(path=str(out))


def count_stop(d):
    c = 0
    for k in ["relevance", "accuracy", "completeness", "readability"]:
        if bool(d.get(k)):
            c += 1
        else:
            break
    return c


def judge(img_path, q):
    img = base64.b64encode(Path(img_path).read_bytes()).decode()
    try:
        r = requests.post(OLLAMA, json={"model": MODEL,
            "prompt": VIS_PROMPT.format(q=q[:300]), "images": [img],
            "stream": False, "options": {"temperature": TEMP, "num_predict": 120}},
            timeout=600)
        t = r.json().get("response", "")
        s = t.find("{"); e = t.rfind("}") + 1
        return count_stop(json.loads(t[s:e]))
    except Exception:
        return None


def variance_of_mean(scores, n, samples=200):
    """Дисперсия среднего из n прогонов (по случайным подвыборкам)."""
    rng = random.Random(0)
    means = []
    for _ in range(samples):
        pick = rng.sample(scores, n)
        means.append(sum(pick) / n)
    m = sum(means) / len(means)
    return sum((x - m) ** 2 for x in means) / len(means)


def main():
    df = pd.read_excel(GOLDEN)

    def httpok(s):
        return df[s].astype(str).str.startswith("http")
    sub = df[httpok("gpt_url") & df["Gpt_relevance"].notna()].drop_duplicates("query_id")
    sample = sub.sample(N_SHOTS, random_state=3)
    SHOTS.mkdir(exist_ok=True)

    per_shot = {}   # qid -> list of R scores
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1000, "height": 1400})
        for _, row in sample.iterrows():
            qid = int(row["query_id"]); q = str(row["last_query"])
            shot = SHOTS / f"var_{qid}.png"
            try:
                shoot(pg, row["gpt_url"], shot)
            except Exception as e:
                print("shot err", e); continue
            scores = []
            for i in range(R):
                c = judge(shot, q)
                if c is not None:
                    scores.append(c)
                print(f"  qid={qid} run {i+1}/{R} c={c}", flush=True)
            if len(scores) >= 4:
                per_shot[qid] = scores
        b.close()

    # дисперсия среднего из n прогонов, усреднённая по скриншотам
    ns = list(range(1, R + 1))
    var_n = []
    for n in ns:
        vs = [variance_of_mean(s, n) for s in per_shot.values() if len(s) >= n]
        var_n.append(sum(vs) / len(vs) if vs else 0)
    sigma2 = var_n[0]    # дисперсия одиночного судьи
    theo = [sigma2 / n for n in ns]

    import math
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(ns, [math.sqrt(v) for v in var_n], "o-", color="#1d9e75",
            label="эмпирическая std среднего из n судей")
    ax.plot(ns, [math.sqrt(v) for v in theo], "k--", alpha=0.6,
            label=r"теоретическая $\sigma/\sqrt{n}$")
    ax.set_xlabel("число агрегируемых судей n (multi-prompt / consensus)")
    ax.set_ylabel("std оценки c (чекбоксы)")
    ax.set_title(f"Снижение дисперсии мультимодального судьи ({MODEL}, {len(per_shot)} скриншотов)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "variance-reduction.jpeg", dpi=130)

    data = {"model": MODEL, "R": R, "temp": TEMP, "n_shots": len(per_shot),
            "per_shot": per_shot, "std_by_n": [round(math.sqrt(v), 3) for v in var_n],
            "sigma_single": round(math.sqrt(sigma2), 3)}
    (OUT / "variance.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print("std by n:", data["std_by_n"])
    print("saved figures/variance-reduction.jpeg")


if __name__ == "__main__":
    main()
