"""Дизайн-судья: визуальная оценка скриншота ответа по принципам C.R.A.P.

Расширяет процесс судейства второй осью -- визуально-дизайнерской, в
дополнение к контентному чек-листу (Relevance/Accuracy/Completeness/
Readability). Оценка опирается на четыре принципа дизайна из книги
Robin Williams, "The Non-Designer's Design Book": Contrast, Repetition,
Alignment, Proximity.

Вход -- скриншот элемента-ответа (полная высота через element.screenshot,
а не page full_page, см. диагностику scroll-контейнера). Судья --
мультимодальная модель (qwen2.5vl) через Ollama.
"""
import sys
import json
import base64
import time
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434/api/generate"

# Селекторы элемента-ответа (полная высота, без обрезки inner-scroll)
YDX_SEL = ".FuturisDialogPair-MessageResponse"
GPT_SEL = '[data-message-author-role="assistant"]'

DESIGN_PROMPT = """Ты оцениваешь ВИЗУАЛЬНО-ДИЗАЙНЕРСКОЕ качество ответа на скриншоте
по четырём принципам дизайна Робин Уильямс (книга "The Non-Designer's Design
Book"): Контраст, Повторение, Выравнивание, Близость (C.R.A.P.).

Оцени КАЖДЫЙ принцип по шкале 0/1/2 (0 — нарушен, 1 — частично, 2 — соблюдён):
- contrast (Контраст): визуально различимы элементы разной роли (заголовки,
  основной текст, акценты, таблицы); нет однородной «серой каши».
- repetition (Повторение): единый повторяющийся стиль (одинаковые буллеты,
  отступы, оформление заголовков и иконок) — визуальная согласованность.
- alignment (Выравнивание): элементы выровнены по общим линиям; нет рваных
  краёв, таблицы и списки ровные.
- proximity (Близость): связанные элементы сгруппированы, между смысловыми
  блоками есть воздух; группировка отражает структуру ответа.

Ответь СТРОГО JSON без пояснений вокруг:
{"contrast":0,"repetition":0,"alignment":0,"proximity":0,"comment":"1-2 предложения по-русски"}"""


def evaluate_design(model: str, image_path: Path) -> dict:
    img = base64.b64encode(Path(image_path).read_bytes()).decode()
    t0 = time.time()
    try:
        r = requests.post(OLLAMA, json={
            "model": model, "prompt": DESIGN_PROMPT, "images": [img],
            "stream": False, "options": {"temperature": 0.0, "num_predict": 300},
        }, timeout=600)
        raw = r.json().get("response", "")
    except Exception as e:
        return {"error": str(e)}
    out = _parse(raw)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    out["raw"] = raw
    return out


def _parse(text: str) -> dict:
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        d = json.loads(text[s:e])
        keys = ["contrast", "repetition", "alignment", "proximity"]
        vals = {k: max(0, min(2, int(d.get(k, 0)))) for k in keys}
        vals["design_score"] = sum(vals.values())   # 0..8
        vals["comment"] = d.get("comment", "")
        return vals
    except Exception as ex:
        return {"contrast": None, "repetition": None, "alignment": None,
                "proximity": None, "design_score": None,
                "comment": f"parse error: {ex}"}


def shoot_answer(page, url: str, out_path: Path) -> bool:
    """Скриншот ЭЛЕМЕНТА-ответа на полную высоту (обходит inner-scroll)."""
    low = url.lower()
    sel = YDX_SEL if ("yandex." in low and "conversation" in low) else GPT_SEL
    try:
        page.goto(url, timeout=30000)
        page.wait_for_timeout(2200)
        page.locator(sel).last.screenshot(path=str(out_path))
        return True
    except Exception as e:
        print(f"  [shot err] {e}")
        return False


if __name__ == "__main__":
    # Демонстрация: один скриншот -> дизайн-оценка
    import pandas as pd
    from playwright.sync_api import sync_playwright
    ROOT = Path(__file__).resolve().parents[2]
    df = pd.read_excel(ROOT / "code/baskets/golden-set/golden_set_full.xlsx")
    qid = int(sys.argv[1]) if len(sys.argv) > 1 else 6366152190423976500
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5vl:3b"
    row = df[df["query_id"] == qid].iloc[0]
    shots = Path(__file__).parent / "design_shots"
    shots.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1000, "height": 1400})
        for tag, url in [("ydx", row["ydx_url"]), ("gpt", row["gpt_url"])]:
            out = shots / f"{qid}_{tag}.png"
            ok = shoot_answer(pg, url, out)
            print(f"{tag}: shot={'ok' if ok else 'FAIL'} ({out.stat().st_size if ok else 0} B)")
            if ok:
                res = evaluate_design(model, out)
                print(f"  design[{model}]:", {k: res.get(k) for k in
                      ["contrast", "repetition", "alignment", "proximity", "design_score"]})
                print("  comment:", res.get("comment"))
        b.close()
