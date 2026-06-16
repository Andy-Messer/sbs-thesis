"""Извлечение ответов из URL через Playwright для massive val set v4.

Читает massive_val_v4.json, для каждого запроса открывает GPT и Yandex URL
в headless Chromium, извлекает текст ответа по CSS-селекторам.
Результат — massive_val_cache.json (resume-совместимый).
"""
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "baskets/v4/massive_val_v4.json"
CACHE = ROOT / "baskets/v4/massive_val_cache.json"

GPT_SEL = '[data-message-author-role="assistant"]'
YDX_SEL = ".FuturisDialogPair-MessageResponse"


def extract(pg, url, query):
    """Извлечение текста ответа по URL."""
    try:
        pg.goto(url, timeout=30000)
        pg.wait_for_timeout(2200)
        low = url.lower()
        sel = YDX_SEL if ("yandex." in low and "conversation" in low) else GPT_SEL
        try:
            loc = pg.locator(sel).last
            t = loc.inner_text(timeout=4000)
        except Exception:
            t = ""
        if not t or len(t) < 20:
            try:
                t = pg.inner_text("body")
            except Exception:
                t = ""
        t = " ".join(t.split())
        # Убираем преамбулу Yandex AI
        if "yandex." in low:
            marker = "Daha fazla arama sonucu göster"
            if marker in t:
                t = t.split(marker, 1)[1].strip()
            elif t.startswith("Yanıtın içeriği"):
                t = t[len("Yanıtın içeriği"):]
        return t[:2500]
    except Exception as e:
        return f"[err {e}]"


def main():
    data = json.loads(INPUT.read_text())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [r for r in data if r["query_id"] not in cache]
    print(f"Всего: {len(data)}, в кэше: {len(cache)}, осталось: {len(todo)}")
    if not todo:
        print("Всё извлечено!")
        return

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1100, "height": 1500})
        for i, r in enumerate(todo):
            qid = r["query_id"]
            gpt = extract(pg, r["gpt_url"], r["last_query"])
            ydx = extract(pg, r["ydx_url"], r["last_query"])
            cache[qid] = {"gpt": gpt, "ydx": ydx}
            if (i + 1) % 10 == 0:
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"  [{i+1}/{len(todo)}] gpt={len(gpt)} ydx={len(ydx)}")
        b.close()

    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"Готово: {len(cache)} извлечено, сохранено в {CACHE}")


if __name__ == "__main__":
    main()
