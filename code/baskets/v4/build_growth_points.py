"""Generate per-agent growth-points markdown block for the BALI-1479 report."""
import json
import re
from collections import Counter, defaultdict

ROWS = "baskets/v4/sbs_v4_basket_rows.jsonl"
SUMMARY = "baskets/v4/nps_significance_summary.jsonl"
OUT = "/tmp/growth_points.md"
WIN = 7

CAT = {
    "incomplete": "неполнота ответа", "irrelevant": "релевантность",
    "context_lost": "удержание контекста", "hallucination": "достоверность фактов",
    "readability": "читаемость/структура", "broken_links": "рабочие ссылки",
    "only_links": "текст вместо «только ссылок»", "media_broken": "рабочие медиа",
    "image_request_unmet": "выдача картинок", "language_mismatch": "язык ответа",
    "time_window_mismatch": "актуальность данных", "illegal_or_unsafe": "безопасный контент",
}


def oc(r):
    y, g = r.get("ydx_win_cnt") or 0, r.get("gpt_win_cnt") or 0
    return "win" if (y > WIN and y > g) else ("loss" if (g > WIN and g > y) else "draw")


def labs(r):
    ic = r.get("improve_categories") or ""
    return [x.strip() for x in ic.split(",") if x.strip() and x.strip() not in ("—", "none", "gpt_refused")]


def cl(s, n=85):
    if not s:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip()
    return t[:n] + ("…" if len(t) > n else "")


def best_comment(r):
    cs = [c.strip() for c in str(r.get("ranking_comment") or "").split("||")
          if c.strip() and not re.fullmatch(r"[\d\s.\-]+", c.strip())]
    cs.sort(key=len, reverse=True)
    return cs[0] if cs else ""


def main():
    rows = [json.loads(l) for l in open(ROWS)]
    by = defaultdict(list)
    for r in rows:
        by[r.get("scenario_tag") or "unknown"].append(r)

    summ = {}
    for l in open(SUMMARY):
        r = json.loads(l)
        if (r["basket"] == "v4" and r["filter"] == "all"
                and r["hypothesis"] == "H1_vs_gpt" and r["dimension"] == "agent"):
            summ[r["value"]] = r

    agents = sorted([a for a in by if a != "unknown"],
                    key=lambda a: summ.get(a, {}).get("point", 0))

    out = []
    out.append("| Агент | NPS | Точка роста №1 | Возврат | Пример (проигрыш) |")
    out.append("|---|--:|---|--:|---|")
    detail = []
    for a in agents:
        recs = by[a]
        N = len(recs)
        losses = [r for r in recs if oc(r) == "loss"]
        # main growth defect = most frequent defect among losses
        cnt = Counter()
        for r in losses:
            for c in labs(r):
                cnt[c] += 1
        if not cnt:
            continue
        top, _ = cnt.most_common(1)[0]
        nloss_top = sum(1 for r in losses if top in labs(r))
        uplift = 100 * nloss_top / N
        s = summ.get(a, {})
        nps = f"{s.get('point', 0):+.1f}%"
        sig = "✅" if s.get("significant") else ""
        ex = [r for r in losses if top in labs(r)]
        ex.sort(key=lambda r: (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0))
        e = ex[0] if ex else None
        q = cl(e.get("last_query"), 55) if e else "—"
        out.append(f"| **{a}** {sig} | {nps} | {CAT.get(top, top)} | +{uplift:.1f} пп | «{q}» |")
        # detail bullet with links + comment
        if e:
            lk = []
            if e.get("ydx_url"):
                lk.append(f"[Yandex]({e['ydx_url']})")
            if e.get("gpt_url"):
                lk.append(f"[GPT]({e['gpt_url']})")
            cm = best_comment(e)
            d = (f"- **{a}** → растить «{CAT.get(top, top)}»: «{cl(e.get('last_query'),70)}» "
                 f"(YA/GPT {e.get('ydx_win_cnt')}/{e.get('gpt_win_cnt')})")
            if cm:
                d += f" — 💬 {cl(cm, 120)}"
            if lk:
                d += " · " + " · ".join(lk)
            detail.append(d)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
        f.write("\n\n### Точки роста с примерами\n\n")
        f.write("\n".join(detail))
    print("wrote", OUT, "agents:", len(agents))


if __name__ == "__main__":
    main()
