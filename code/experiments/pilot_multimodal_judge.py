"""Пилотный эксперимент: мультимодальные LLM-судьи (Qwen2.5-VL) поверх
скриншотов ответов из golden set, с мониторингом живости процесса.

Шаги:
  1. Берём небольшую стратифицированную выборку запросов из
     code/baskets/golden-set/golden_set_full.xlsx (по полю gpt_checks,
     посчитанному из ассессорских чекбоксов Gpt_*).
  2. Для каждого запроса рендерим скриншот страницы с ответом
     (Playwright) -- это и есть "мультимодальный" вход судьи.
  3. Прогоняем скриншот + текст запроса через qwen2.5vl:3b и
     qwen2.5vl:7b (Ollama, потоковый режим) и просим выставить
     чекбоксы Relevance/Accuracy/Completeness/Readability по тем же
     правилам, что даются ассессорам.
  4. Сравниваем число выставленных чекбоксов с ассессорской разметкой
     (golden), считаем agreement/MAE.
  5. Во время прогона SystemMonitor пишет CPU/RAM и "heartbeats"
     (получение очередного чанка токенов) -- по логу и графику видно,
     что judge не висит, а генерирует ответ постепенно.

Результаты сохраняются в code/experiments/pilot_results/.
"""
import sys
import json
import time
import base64
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from monitoring import SystemMonitor

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_XLSX = ROOT / "code/baskets/golden-set/golden_set_full.xlsx"
OUT_DIR = Path(__file__).parent / "pilot_results"
SHOTS_DIR = OUT_DIR / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHOTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["qwen2.5vl:3b", "qwen2.5vl:7b"]

JUDGE_PROMPT = """Ты выступаешь в роли ассессора, оценивающего ответ модели на
запрос пользователя. На скриншоте показан экран с запросом и ответом.

Запрос пользователя: "{query}"

Оцени ответ по 4 критериям, СТРОГО В ЭТОМ ПОРЯДКЕ: Relevance, Accuracy,
Completeness, Readability. Критерий = true/false. Если по какому-то
критерию ставишь false, все последующие критерии тоже считаются false
(не оценивай их отдельно, просто верни false).

- Relevance: модель реально отвечает на вопрос пользователя (или по
  делу объясняет, почему не может ответить).
- Accuracy: в ответе нет явных фактических ошибок и противоречий.
- Completeness: ответ покрывает запрос полностью (формат, детали,
  единицы измерения и т.п.).
- Readability: ответ хорошо структурирован, легко найти суть.

Ответь СТРОГО в формате JSON без пояснений вокруг:
{{"relevance": true/false, "accuracy": true/false,
  "completeness": true/false, "readability": true/false,
  "comment": "1-2 предложения по-русски, почему"}}"""


def load_sample(n_per_bucket=(3, 6, 6)):
    """Стратифицированная выборка по числу ассессорских чекбоксов (1/2/4)."""
    df = pd.read_excel(GOLDEN_XLSX)
    sub = df[(df["is_dialog"] == False)
             & df["Gpt_relevance"].notna()
             & df["gpt_url"].notna()
             & (df["gpt_url"].astype(str).str.startswith("http"))]
    sub = sub.drop_duplicates("query_id").copy()
    checks = sub[["Gpt_relevance", "Gpt_accuracy",
                   "Gpt_completeness", "Gpt_readability"]]
    sub["gpt_checks"] = checks.sum(axis=1).astype(int)

    rows = []
    for bucket, n in zip((1, 2, 4), n_per_bucket):
        cand = sub[sub["gpt_checks"] == bucket]
        rows.append(cand.sample(min(n, len(cand)), random_state=42))
    sample = pd.concat(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    return sample


def take_screenshot(url: str, out_path: Path) -> bool:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1000, "height": 1400})
            page.goto(url, timeout=30000)
            page.wait_for_timeout(800)
            page.screenshot(path=str(out_path))
            browser.close()
        return True
    except Exception as e:
        print(f"    [screenshot error] {e}")
        return False


def call_judge_streaming(model: str, query: str, image_path: Path,
                          monitor: SystemMonitor, query_idx: int) -> dict:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    prompt = JUDGE_PROMPT.format(query=query[:500])

    t0 = time.time()
    chunks = []
    n_tokens = 0
    try:
        with requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt, "images": [img_b64],
            "stream": True,
            "options": {"temperature": 0.0, "num_predict": 300},
        }, timeout=600, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                piece = d.get("response", "")
                chunks.append(piece)
                n_tokens += 1
                monitor.heartbeat(query_idx, model, n_tokens)
                if n_tokens % 10 == 0:
                    snap = monitor.snapshot(label=f"q{query_idx}:{model}")
                    print(f"      ... жива, токенов={n_tokens:>4d}, "
                          f"t={time.time() - t0:5.1f}s, "
                          f"CPU={snap['cpu_percent']:.0f}% RAM={snap['ram_percent']:.0f}%")
                if d.get("done"):
                    break
    except Exception as e:
        return {"error": str(e), "elapsed_sec": time.time() - t0}

    text = "".join(chunks)
    elapsed = time.time() - t0
    parsed = parse_verdict(text)
    parsed["raw_response"] = text
    parsed["elapsed_sec"] = round(elapsed, 1)
    parsed["n_tokens"] = n_tokens
    return parsed


def parse_verdict(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        d = json.loads(text[start:end])
        rel, acc, comp, read = (bool(d.get("relevance")), bool(d.get("accuracy")),
                                 bool(d.get("completeness")), bool(d.get("readability")))
        checks = 0
        for v in (rel, acc, comp, read):
            if v:
                checks += 1
            else:
                break
        return {"relevance": rel, "accuracy": acc, "completeness": comp,
                "readability": read, "checks": checks,
                "comment": d.get("comment", "")}
    except Exception as e:
        return {"relevance": None, "accuracy": None, "completeness": None,
                "readability": None, "checks": None, "comment": f"parse error: {e}",
                "parse_error": str(e)}


def main():
    print("=" * 60)
    print("  Пилот: мультимодальные судьи Qwen2.5-VL + мониторинг живости")
    print("=" * 60)

    sample = load_sample()
    print(f"\nВыборка: {len(sample)} запросов, "
          f"распределение gpt_checks: {sample['gpt_checks'].value_counts().to_dict()}")

    monitor = SystemMonitor(str(OUT_DIR / "monitoring"))
    monitor.snapshot(label="start")

    print("\nШаг 1/2: рендер скриншотов...")
    shots = []
    for i, row in sample.iterrows():
        path = SHOTS_DIR / f"q{i:02d}.png"
        if not path.exists():
            ok = take_screenshot(row["gpt_url"], path)
        else:
            ok = True
        shots.append(path if ok else None)
        print(f"  [{i+1:>2}/{len(sample)}] query_id={row['query_id']} -> "
              f"{'ok' if ok else 'FAIL'}")
        monitor.snapshot(label=f"screenshot {i}")

    results = {m: [] for m in MODELS}
    print("\nШаг 2/2: прогон судей...")
    for model in MODELS:
        print(f"\n--- модель: {model} ---")
        for i, row in sample.iterrows():
            if shots[i] is None:
                continue
            print(f"  [{i+1:>2}/{len(sample)}] query_id={row['query_id']}: "
                  f"запускаю {model}...")
            v = call_judge_streaming(model, row["last_query"], shots[i], monitor, i)
            v["query_id"] = int(row["query_id"])
            v["gpt_checks_golden"] = int(row["gpt_checks"])
            v["scenario_tag"] = row.get("scenario_tag", "")
            results[model].append(v)
            ok_checks = v.get("checks")
            print(f"      -> checks_llm={ok_checks} checks_golden={v['gpt_checks_golden']} "
                  f"({v.get('elapsed_sec', '?')}s, {v.get('n_tokens', '?')} tok)")

    monitor.snapshot(label="end")
    plot_path = monitor.save_plot("pilot_load.jpeg")
    print(f"\nГрафик нагрузки: {plot_path}")

    summary = {}
    for model in MODELS:
        rs = [r for r in results[model] if r.get("checks") is not None]
        if not rs:
            continue
        agree = sum(1 for r in rs if r["checks"] == r["gpt_checks_golden"]) / len(rs)
        mae = sum(abs(r["checks"] - r["gpt_checks_golden"]) for r in rs) / len(rs)
        avg_time = sum(r["elapsed_sec"] for r in rs) / len(rs)
        summary[model] = {
            "n": len(rs), "agreement": round(agree, 3),
            "mae": round(mae, 3), "avg_elapsed_sec": round(avg_time, 1),
        }
        print(f"\n{model}: n={len(rs)} agreement={agree:.0%} "
              f"MAE={mae:.2f} avg_time={avg_time:.1f}s")

    summary["monitor"] = monitor.summary()
    summary["sample_size"] = len(sample)
    summary["checks_distribution"] = sample["gpt_checks"].value_counts().to_dict()

    (OUT_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "details": results}, indent=2,
                    ensure_ascii=False, default=str))
    print(f"\nРезультаты: {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
