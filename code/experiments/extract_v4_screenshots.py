"""Шаг 1: Стратифицированная выборка N=200 + скриншоты через Playwright.

Берёт 200 запросов из massive_val_v4.json (stratified по verdict),
рендерит GPT и YDX ответы в скриншоты (WEBP), сохраняет в design_shots/.
Выход: sample_v4_200.json + design_shots/{qid}_{side}.webp
"""
import json
import random
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "code/baskets/v4/massive_val_v4.json"
OUT_DIR = ROOT / "code/experiments/design_shots_v4"
SAMPLE_OUT = ROOT / "code/experiments/staged_results/sample_v4_200.json"
N = 200
TARGET = {"draw": 74, "gpt_win": 66, "ydx_win": 60}
SEED = 42
VIEWPORT = {"width": 1000, "height": 1400}


def main():
    data = json.loads(INPUT.read_text())
    rng = random.Random(SEED)

    # Stratified sampling
    by_verdict = {}
    for r in data:
        v = r["verdict"]
        if v not in by_verdict:
            by_verdict[v] = []
        by_verdict[v].append(r)

    sample = []
    for verdict, count in TARGET.items():
        candidates = by_verdict.get(verdict, [])
        if len(candidates) < count:
            print(f"WARNING: {verdict} has only {len(candidates)} < {count}")
            count = len(candidates)
        chosen = rng.sample(candidates, count)
        sample.extend(chosen)

    rng.shuffle(sample)
    print(f"Sample: {len(sample)} queries")

    OUT_DIR.mkdir(exist_ok=True)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport=VIEWPORT)

        for i, r in enumerate(sample):
            qid = r["query_id"]
            for side, url_key in [("gpt", "gpt_url"), ("ydx", "ydx_url")]:
                url = r.get(url_key)
                if not url:
                    continue
                out_path = OUT_DIR / f"{qid}_{side}.png"
                if out_path.exists():
                    continue
                try:
                    pg.goto(url, timeout=30000)
                    pg.wait_for_timeout(2200)
                    pg.screenshot(path=str(out_path), full_page=False, type="png")
                except Exception as e:
                    print(f"  ERR {qid}_{side}: {e}")

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(sample)}] done")
                SAMPLE_OUT.write_text(json.dumps(sample, ensure_ascii=False, indent=2))

        b.close()

    SAMPLE_OUT.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    shots = list(OUT_DIR.glob("*.webp"))
    print(f"Gотово: {len(sample)} запросов, {len(shots)} скриншотов")
    print(f"Sample: {SAMPLE_OUT}")
    print(f"Screenshots: {OUT_DIR}")


if __name__ == "__main__":
    main()
