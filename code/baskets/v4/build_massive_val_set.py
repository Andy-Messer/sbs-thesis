"""Сборка Massive Val Set из v4 баскета и двух пулов ассессоров.

Мержит assignments_from_pool_91796413 и 91859764, извлекает
query_id, verdict, покритериальные чекбоксы, URL ответов.
Для каждого запроса делает majority-vote по вердикту и пер-критериальному
согласию ассессоров.

Выход: massive_val_v4.json — список словарей с полями:
  query_id, last_query, verdict, gpt_url, ydx_url,
  gpt_checkboxes, ydx_checkboxes, n_assessors
"""
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
V4_BASKET = ROOT / "baskets/v4/20260506_basket_yandex_ai_app_v4_validate"
POOLS = [
    ROOT / "baskets/v4/assignments_from_pool_91796413__15-06-2026.tsv",
    ROOT / "baskets/v4/assignments_from_pool_91859764__15-06-2026.tsv",
]
OUT = ROOT / "baskets/v4/massive_val_v4.json"
CRITS = ["relevance", "accuracy", "completeness", "readability"]


def parse_checkboxes(cb_raw):
    """Парсинг OUTPUT:models_checkboxes → (cb_model1, cb_model2)."""
    if not cb_raw or cb_raw.strip() in ("", ","):
        return None, None
    parts = cb_raw.split("},{")
    if len(parts) != 2:
        return None, None
    try:
        cb1 = json.loads(parts[0].replace("\\", "") + "}")
        cb2 = json.loads("{" + parts[1].replace("\\", ""))
        return cb1, cb2
    except Exception:
        return None, None


def parse_verdict(ranks_str):
    """OUTPUT:models_ranks '1,2' → verdict (gpt_win/ydx_win/draw)."""
    if not ranks_str or ranks_str.strip() == "":
        return None
    parts = [p.strip() for p in ranks_str.split(",")]
    if len(parts) != 2:
        return None
    try:
        r1, r2 = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if r1 < r2:
        return "gpt_win"
    elif r2 < r1:
        return "ydx_win"
    else:
        return "draw"


def bool_check(cb, crit):
    """Извлечение булевого значения критерия из чекбокса."""
    if cb is None:
        return None
    v = cb.get(crit)
    if isinstance(v, bool):
        return v
    return None


def load_assignments():
    """Загрузка и мерж обоих пулов."""
    records = []  # (query_id, verdict, gpt_cb, ydx_cb, query_text, links)
    for pool_path in POOLS:
        if not pool_path.exists():
            print(f"  SKIP: {pool_path.name}")
            continue
        with open(pool_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        print(f"  {pool_path.name}: {len(rows)} assignments")
        for r in rows:
            try:
                meta = json.loads(r.get("INPUT:meta", "{}"))
                qid = meta.get("query_id")
            except Exception:
                continue
            if not qid:
                continue
            verdict = parse_verdict(r.get("OUTPUT:models_ranks", ""))
            cb1, cb2 = parse_checkboxes(r.get("OUTPUT:models_checkboxes", ""))
            query = r.get("INPUT:query", "")
            links = r.get("INPUT:models_links", "")
            link_parts = [l.strip() for l in links.split(",")] if links else []
            gpt_url = link_parts[0] if len(link_parts) > 0 else None
            ydx_url = link_parts[1] if len(link_parts) > 1 else None
            records.append({
                "query_id": str(qid),
                "verdict": verdict,
                "gpt_cb": cb1,
                "ydx_cb": cb2,
                "query": query,
                "gpt_url": gpt_url,
                "ydx_url": ydx_url,
            })
    return records


def aggregate_per_query(records):
    """Агрегация по query_id: majority vote по вердикту, consensus по чекбоксам."""
    by_qid = defaultdict(list)
    for r in records:
        by_qid[r["query_id"]].append(r)

    results = []
    for qid, recs in by_qid.items():
        # Majority vote по вердикту
        verdicts = [r["verdict"] for r in recs if r["verdict"]]
        if not verdicts:
            continue
        verdict = Counter(verdicts).most_common(1)[0][0]

        # Majority vote по чекбоксам для GPT
        gpt_crits = {}
        for crit in CRITS:
            vals = [bool_check(r["gpt_cb"], crit) for r in recs]
            vals = [v for v in vals if v is not None]
            if vals:
                gpt_crits[crit] = sum(vals) > len(vals) / 2

        # Majority vote по чекбоксам для YandexAI
        ydx_crits = {}
        for crit in CRITS:
            vals = [bool_check(r["ydx_cb"], crit) for r in recs]
            vals = [v for v in vals if v is not None]
            if vals:
                ydx_crits[crit] = sum(vals) > len(vals) / 2

        # Берём URL из первого валидного рекорда
        gpt_url = next((r["gpt_url"] for r in recs if r.get("gpt_url")), None)
        ydx_url = next((r["ydx_url"] for r in recs if r.get("ydx_url")), None)
        query = next((r["query"] for r in recs if r.get("query")), "")

        results.append({
            "query_id": qid,
            "last_query": query,
            "verdict": verdict,
            "gpt_url": gpt_url,
            "ydx_url": ydx_url,
            "Gpt_relevance": gpt_crits.get("relevance"),
            "Gpt_accuracy": gpt_crits.get("accuracy"),
            "Gpt_completeness": gpt_crits.get("completeness"),
            "Gpt_readability": gpt_crits.get("readability"),
            "YandexAI_relevance": ydx_crits.get("relevance"),
            "YandexAI_accuracy": ydx_crits.get("accuracy"),
            "YandexAI_completeness": ydx_crits.get("completeness"),
            "YandexAI_readability": ydx_crits.get("readability"),
            "n_assessors": len(recs),
        })
    return results


def main():
    print("=== Massive Val Set v4 ===")
    print("Загрузка assignment'ов из пулов...")
    records = load_assignments()
    print(f"Всего записей: {len(records)}")

    print("Агрегация по query_id...")
    results = aggregate_per_query(records)
    print(f"Уникальных запросов: {len(results)}")

    # Статистика
    verdicts = Counter(r["verdict"] for r in results)
    with_urls = sum(1 for r in results if r["gpt_url"] and r["ydx_url"])
    with_checkboxes = sum(1 for r in results if r["Gpt_relevance"] is not None)
    print(f"Вердикты: {dict(verdicts)}")
    print(f"С URL: {with_urls}")
    print(f"С чекбоксами: {with_checkboxes}")

    # Фильтрация: только с URL и чекбоксами
    valid = [r for r in results if r["gpt_url"] and r["ydx_url"]
             and r["Gpt_relevance"] is not None]
    print(f"Валидных для judge: {len(valid)}")

    OUT.write_text(json.dumps(valid, ensure_ascii=False, indent=2))
    print(f"Сохранено: {OUT}")


if __name__ == "__main__":
    main()
