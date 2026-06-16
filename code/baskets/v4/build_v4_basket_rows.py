"""
Build v4 'basket in detail' rows from raw assessor output, matching the schema of
//home/bali/yandex-ai-chat/sbs_stream_1k_classified (basket='stream1k').

Output rows get basket='v4' and are appended to the same YT table so the existing
DataLens charts work unchanged once a `basket` selector is added.

Key rules (mirrors baskets/stream-1k pipeline + user instruction):
- Deduplicate raw judgments by (query_id, workerId) — the source file triplicates
  every assignment, so we keep one judgment per worker per query.
- Per-query aggregation across unique workers:
    * <Model>_<criterion> = SUM of ticked checkboxes (raw counts)
    * <Model>_rank        = AVG assessor rank (lower = better)
    * ydx/draw/gpt _win_cnt = rank comparisons
    * <Model>_not_open    = count of model_not_open==True (single flag → both models)
    * *_comment / ranking_comment = unique texts joined by ' || '
    * verdict_ru / improve_categories via classify_report logic
- Metadata (last_query, period, region, urls, is_dialog, scenario_tag) from the
  basket file + inputValues. scenario_tag := prod_classifier_agent (the agent dim).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# reuse the exact verdict + category rules from the stream-1k pipeline
sys.path.insert(0, str(Path("baskets/stream-1k").resolve()))
from classify_report import find_categories  # noqa: E402

BASKET = "baskets/v4/20260506_basket_yandex_ai_app_v4_validate"
OUTPUT = "baskets/v4/output1_43368eda-68e8-47ca-b1e6-622aff1eec29"
OUT_JSONL = "baskets/v4/sbs_v4_basket_rows.jsonl"

BASKET_TAG = "v4"
BASKET_DATE = "2026-05-06"  # basket creation date (from filename 20260506)
CRITERIA = ["relevance", "accuracy", "completeness", "readability"]


def to_int64(v) -> int | None:
    """Reinterpret a possibly-uint64 query_id into the signed int64 range so it
    fits the existing YT `query_id int64` column while preserving uniqueness
    (two's-complement wrap). The value is only used as an identifier dimension."""
    if v is None:
        return None
    s = str(v)
    if not s.lstrip("-").isdigit():
        return None
    n = int(s)
    if n >= 2 ** 63:
        n -= 2 ** 64
    return n


def verdict_from_counts(ydx: int, draw: int, gpt: int, rk_y: float, rk_g: float) -> str:
    """Same thresholds as classify_report.verdict but on aggregated counts."""
    total = ydx + draw + gpt
    if total <= 0:
        return ""
    net = ydx - gpt
    if net >= 3:
        return "хороший"
    if net <= -3:
        return "плохой"
    if rk_y and rk_g:
        diff = rk_g - rk_y  # >0 => Yandex ranked better
        if diff >= 0.5 and net >= 0:
            return "хороший"
        if diff <= -0.5 and net <= 0:
            return "плохой"
    return "сомнительный"


def join_unique(texts: list[str]) -> str:
    seen, out = set(), []
    for t in texts:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return " || ".join(out)


def main() -> None:
    # 1. basket metadata by query_id
    meta = {}
    for line in open(BASKET):
        d = json.loads(line)
        meta[d["id"]] = {
            "scenario_tag": d.get("prod_classifier_agent"),
            "is_dialog": bool(d.get("is_dialog")),
        }

    # 2. group raw judgments, dedup by (query_id, workerId)
    #    acc[qid] = dict with accumulators
    acc: dict = defaultdict(lambda: {
        "workers": set(),
        "ya_crit": {c: 0 for c in CRITERIA},
        "gpt_crit": {c: 0 for c in CRITERIA},
        "ya_rank_sum": 0.0, "gpt_rank_sum": 0.0, "rank_n": 0,
        "ydx_win": 0, "draw": 0, "gpt_win": 0,
        "not_open": 0,
        "ya_comments": [], "gpt_comments": [], "ranking_comments": [],
        "last_query": None, "period": None, "region": None,
        "ydx_url": None, "gpt_url": None,
    })

    seen_pairs = set()
    rows_read = 0
    for line in open(OUTPUT):
        d = json.loads(line)
        rows_read += 1
        iv = d.get("inputValues", {})
        ov = d.get("outputValues") or {}
        qid = iv.get("meta", {}).get("query_id")
        worker = d.get("workerId")
        if qid is None or worker is None:
            continue
        pair = (qid, worker)
        if pair in seen_pairs:
            continue  # dedup triplication
        seen_pairs.add(pair)

        a = acc[qid]
        a["workers"].add(worker)

        names = iv.get("models_names", [])
        links = iv.get("models_links", [])
        # capture metadata once
        if a["last_query"] is None:
            a["last_query"] = iv.get("query")
            a["period"] = iv.get("period")
            a["region"] = iv.get("region")
            try:
                yi, gi = names.index("Yandex AI"), names.index("Gpt")
                a["ydx_url"] = links[yi] if yi < len(links) else None
                a["gpt_url"] = links[gi] if gi < len(links) else None
            except ValueError:
                pass

        try:
            ya_i, gpt_i = names.index("Yandex AI"), names.index("Gpt")
        except ValueError:
            continue

        # not_open (single flag → counts toward both models, matching stream1k)
        if ov.get("model_not_open") is True:
            a["not_open"] += 1

        cbs = ov.get("models_checkboxes")
        if cbs and ya_i < len(cbs) and gpt_i < len(cbs):
            ya_cb = cbs[ya_i] or {}
            gpt_cb = cbs[gpt_i] or {}
            for c in CRITERIA:
                if ya_cb.get(c):
                    a["ya_crit"][c] += 1
                if gpt_cb.get(c):
                    a["gpt_crit"][c] += 1
            if ya_cb.get("comment"):
                a["ya_comments"].append(ya_cb["comment"])
            if gpt_cb.get("comment"):
                a["gpt_comments"].append(gpt_cb["comment"])

        if ov.get("ranking_comment"):
            a["ranking_comments"].append(ov["ranking_comment"])

        ranks = ov.get("models_ranks")
        if ranks and ya_i < len(ranks) and gpt_i < len(ranks):
            try:
                yr, gr = int(ranks[ya_i]), int(ranks[gpt_i])
            except (ValueError, TypeError):
                yr = gr = None
            if yr is not None and gr is not None:
                a["ya_rank_sum"] += yr
                a["gpt_rank_sum"] += gr
                a["rank_n"] += 1
                if yr < gr:
                    a["ydx_win"] += 1
                elif yr == gr:
                    a["draw"] += 1
                else:
                    a["gpt_win"] += 1

    print(f"read {rows_read} rows, {len(seen_pairs)} unique (qid,worker), "
          f"{len(acc)} queries", file=sys.stderr)

    # 3. emit aggregated rows
    n_written = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for qid, a in acc.items():
            rk_y = round(a["ya_rank_sum"] / a["rank_n"], 4) if a["rank_n"] else None
            rk_g = round(a["gpt_rank_sum"] / a["rank_n"], 4) if a["rank_n"] else None
            ya_cmt = join_unique(a["ya_comments"])
            rank_cmt = join_unique(a["ranking_comments"])
            blob = (ya_cmt + " || " + rank_cmt).lower()
            v = verdict_from_counts(a["ydx_win"], a["draw"], a["gpt_win"],
                                    rk_y or 0, rk_g or 0)
            cats = find_categories(blob)
            if v == "хороший" and cats == ["gpt_refused"]:
                cats = ["none"]
            improve = ",".join(dict.fromkeys(cats)) if cats else (
                "none" if v == "хороший" else "—")
            m = meta.get(qid, {})

            rec = {
                "ydx_url": a["ydx_url"],
                "gpt_url": a["gpt_url"],
                "query_id": to_int64(qid),
                "last_query": a["last_query"],
                "period": a["period"],
                "is_dialog": bool(m.get("is_dialog", False)),
                "region": a["region"],
                "scenario_tag": m.get("scenario_tag"),
                "ydx_win_cnt": a["ydx_win"],
                "draw_cnt": a["draw"],
                "gpt_win_cnt": a["gpt_win"],
                "YandexAI_rank": rk_y,
                "Gpt_rank": rk_g,
                "Gpt_not_open": a["not_open"],
                "YandexAI_not_open": a["not_open"],
                "YandexAI_relevance": a["ya_crit"]["relevance"],
                "YandexAI_accuracy": a["ya_crit"]["accuracy"],
                "YandexAI_completeness": a["ya_crit"]["completeness"],
                "YandexAI_readability": a["ya_crit"]["readability"],
                "Gpt_relevance": a["gpt_crit"]["relevance"],
                "Gpt_accuracy": a["gpt_crit"]["accuracy"],
                "Gpt_completeness": a["gpt_crit"]["completeness"],
                "Gpt_readability": a["gpt_crit"]["readability"],
                "YandexAI_comment": ya_cmt,
                "Gpt_comment": join_unique(a["gpt_comments"]),
                "ranking_comment": rank_cmt,
                "verdict_ru": v,
                "improve_categories": improve,
                "sbs_net": a["ydx_win"] - a["gpt_win"],
                "basket": BASKET_TAG,
                "date": BASKET_DATE,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"wrote {n_written} rows -> {OUT_JSONL}", file=sys.stderr)

    # tiny sanity summary
    from collections import Counter
    vc = Counter()
    for line in open(OUT_JSONL):
        vc[json.loads(line)["verdict_ru"]] += 1
    print("verdict_ru:", dict(vc), file=sys.stderr)


if __name__ == "__main__":
    main()
