"""
Build a detailed per-agent NPS decomposition report for v4.

For every agent (scenario_tag) it combines:
  - statistical result: NPS point + 95% CI + significance (H1 vs GPT, threshold metric)
    from nps_significance_summary.jsonl
  - outcome mix: Yandex win / draw / GPT win shares (>7 votes threshold)
  - assessor labels: improve_categories frequency (what assessors flagged)
  - per-criterion satisfaction (relevance/accuracy/completeness/readability) YA vs GPT
  - concrete examples: best (Yandex clearly wins) and worst (GPT clearly wins) queries
    with the query text, links, assessor comments and labels

Output: measurements/v4-2026-05/AGENT_NPS_DECOMPOSITION.md
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

ROWS = "baskets/v4/sbs_v4_basket_rows.jsonl"
SUMMARY = "baskets/v4/nps_significance_summary.jsonl"
OUT = "measurements/v4-2026-05/AGENT_NPS_DECOMPOSITION.md"

WIN_THR = 7
CRITERIA = ["relevance", "accuracy", "completeness", "readability"]

# Human labels for improve_categories codes
CAT_RU = {
    "broken_links": "битые ссылки",
    "only_links": "только ссылки без текста",
    "illegal_or_unsafe": "небезопасный/запрещённый контент",
    "hallucination": "галлюцинации/выдуманные факты",
    "time_window_mismatch": "устаревшая/несвоевременная инфа",
    "language_mismatch": "ответ не на том языке",
    "context_lost": "потерян контекст диалога",
    "irrelevant": "нерелевантно",
    "incomplete": "неполный ответ",
    "readability": "плохая читаемость",
    "media_broken": "битые медиа",
    "image_request_unmet": "не выдал запрошенную картинку",
    "gpt_refused": "GPT отказался (драйвер побед)",
    "none": "без претензий",
}


def clean(s: str, n: int = 240) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:n] + ("…" if len(s) > n else "")


def split_comments(s: str) -> list[str]:
    if not s:
        return []
    return [c.strip() for c in str(s).split("||") if c.strip() and c.strip() != "."]


def load_summary():
    """agent -> {nps, ci_low, ci_high, p, sig, n} for H1_vs_gpt, filter=all, dim=agent."""
    res = {}
    for line in open(SUMMARY):
        r = json.loads(line)
        if (r["basket"] == "v4" and r["filter"] == "all"
                and r["dimension"] == "agent" and r["hypothesis"] == "H1_vs_gpt"):
            res[r["value"]] = r
        if (r["basket"] == "v4" and r["filter"] == "all"
                and r["dimension"] == "overall" and r["hypothesis"] == "H1_vs_gpt"):
            res["_overall_"] = r
    return res


def main():
    rows = [json.loads(l) for l in open(ROWS)]
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r.get("scenario_tag") or "unknown"].append(r)

    summ = load_summary()

    # order agents by NPS desc (those present in summary)
    agents = sorted(by_agent.keys(),
                    key=lambda a: summ.get(a, {}).get("point", -999), reverse=True)

    out = []
    ov = summ.get("_overall_", {})
    out.append("# v4 — декомпозиция NPS по агентам\n")
    out.append(
        f"Источник: {len(rows)} запросов v4 (per-query, дедуп по workerId). "
        f"Метрика — **SBS NPS (H1 vs GPT)**: (победы YA − победы GPT)/N при пороге "
        f"**>7 голосов**. Значимость — 95% bootstrap CI.\n")
    if ov:
        out.append(
            f"**Итог по корзинке v4:** NPS = **{ov['point']:+.1f}%** "
            f"(CI [{ov['ci_low']:+.1f}, {ov['ci_high']:+.1f}], "
            f"{'значимо' if ov['significant'] else 'не значимо'}, n={ov['n_queries']}).\n")
    out.append("Ниже — разложение этого результата по агентам: что тянет NPS вверх, "
               "что вниз, с метками асессоров, комментариями и конкретными примерами.\n")

    # ---- summary table ----
    out.append("## Сводная таблица\n")
    out.append("| Агент | n | NPS % | 95% CI | Значимо | YA win | draw | GPT win |")
    out.append("|---|--:|--:|---|:--:|--:|--:|--:|")
    for a in agents:
        s = summ.get(a)
        recs = by_agent[a]
        n = len(recs)
        w = Counter()
        for r in recs:
            y = r.get("ydx_win_cnt") or 0
            g = r.get("gpt_win_cnt") or 0
            if y > WIN_THR and y > g:
                w["ya"] += 1
            elif g > WIN_THR and g > y:
                w["gpt"] += 1
            else:
                w["draw"] += 1
        if s:
            ci = f"[{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]"
            sig = "✅" if s["significant"] else "—"
            nps = f"{s['point']:+.1f}"
        else:
            ci, sig, nps = "—", "—", "—"
        out.append(f"| {a} | {n} | {nps} | {ci} | {sig} | "
                   f"{100*w['ya']/n:.0f}% | {100*w['draw']/n:.0f}% | {100*w['gpt']/n:.0f}% |")
    out.append("")

    # ---- per agent detail ----
    for a in agents:
        s = summ.get(a, {})
        recs = by_agent[a]
        n = len(recs)

        # outcome mix + criteria sat + labels
        w = Counter()
        ya_sat = Counter()
        gpt_sat = Counter()
        labels = Counter()
        votes_tot = 0
        for r in recs:
            y = r.get("ydx_win_cnt") or 0
            g = r.get("gpt_win_cnt") or 0
            d = r.get("draw_cnt") or 0
            votes_tot += (y + g + d)
            if y > WIN_THR and y > g:
                w["ya"] += 1
            elif g > WIN_THR and g > y:
                w["gpt"] += 1
            else:
                w["draw"] += 1
            for c in CRITERIA:
                ya_sat[c] += r.get(f"YandexAI_{c}") or 0
                gpt_sat[c] += r.get(f"Gpt_{c}") or 0
            ic = r.get("improve_categories") or ""
            for code in [x.strip() for x in ic.split(",") if x.strip() and x.strip() not in ("—", "none")]:
                labels[code] += 1

        out.append(f"## {a}\n")
        if s:
            verdict = ("**значимое преимущество**" if s["significant"] and s["point"] > 0
                       else "**значимое отставание**" if s["significant"] and s["point"] < 0
                       else "без статзначимого смещения")
            out.append(
                f"- **NPS = {s['point']:+.1f}%**, CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}], "
                f"{verdict} (n={n}).")
        else:
            out.append(f"- n={n} (нет в сводке значимости).")
        out.append(
            f"- Исходы: Yandex win **{100*w['ya']/n:.0f}%** · draw {100*w['draw']/n:.0f}% · "
            f"GPT win {100*w['gpt']/n:.0f}%.")

        # criteria gap (avg ticks per query)
        crit_line = []
        for c in CRITERIA:
            ya = ya_sat[c] / n
            gp = gpt_sat[c] / n
            crit_line.append(f"{c[:4]} {ya:.1f}/{gp:.1f} ({ya-gp:+.1f})")
        out.append(f"- Критерии (галочек/запрос YA/GPT, Δ): " + " · ".join(crit_line) + ".")

        # assessor labels
        if labels:
            top = labels.most_common(6)
            lab_str = ", ".join(f"{CAT_RU.get(k, k)} ×{v}" for k, v in top)
            out.append(f"- Метки асессоров (частые недочёты): {lab_str}.")
        else:
            out.append("- Метки асессоров: критичных не отмечено.")

        # interpretation: strengths / weaknesses from criteria deltas
        strengths = [c for c in CRITERIA if (ya_sat[c]-gpt_sat[c])/n > 0.3]
        weaks = [c for c in CRITERIA if (gpt_sat[c]-ya_sat[c])/n > 0.3]
        if strengths:
            out.append(f"- 💪 Сильные стороны: {', '.join(strengths)} (выше GPT).")
        if weaks:
            out.append(f"- ⚠️ Слабые стороны: {', '.join(weaks)} (ниже GPT).")

        # examples: best (YA wins big) and worst (GPT wins big)
        scored = []
        for r in recs:
            y = r.get("ydx_win_cnt") or 0
            g = r.get("gpt_win_cnt") or 0
            scored.append((y - g, r))
        best = sorted(scored, key=lambda x: -x[0])[:2]
        worst = sorted(scored, key=lambda x: x[0])[:2]

        def fmt_example(net, r, kind):
            q = clean(r.get("last_query"), 120)
            cmts = split_comments(r.get("ranking_comment"))
            yc = split_comments(r.get("YandexAI_comment"))
            ic = r.get("improve_categories") or ""
            lines = [f"  - **{kind}** (net {net:+d}, голоса YA/GPT "
                     f"{r.get('ydx_win_cnt')}/{r.get('gpt_win_cnt')}): «{q}»"]
            if ic and ic not in ("—", "none"):
                human = ", ".join(CAT_RU.get(x.strip(), x.strip()) for x in ic.split(",") if x.strip())
                lines.append(f"    - метки: {human}")
            for c in (yc[:1] + cmts[:2]):
                lines.append(f"    - 💬 {clean(c, 200)}")
            ydx_url = r.get("ydx_url")
            if ydx_url:
                lines.append(f"    - [ответ Yandex AI]({ydx_url})")
            return "\n".join(lines)

        out.append("- **Примеры:**")
        for net, r in best:
            if net > 0:
                out.append(fmt_example(net, r, "✅ хорошо (Yandex выигрывает)"))
        for net, r in worst:
            if net < 0:
                out.append(fmt_example(net, r, "❌ плохо (GPT выигрывает)"))
        out.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT} ({len(out)} lines, {len(agents)} agents)")


if __name__ == "__main__":
    main()
