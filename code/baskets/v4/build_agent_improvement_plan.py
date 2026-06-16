"""
Per-agent FULL status + every NPS driver (down to the smallest) with examples — v4.

METHODOLOGY (same for every agent)
----------------------------------
1. Outcome per query (>7 votes): WIN / LOSS / DRAW. NPS_a = (W_a − L_a)/N_a.
2. Roll-up: NPS_total = Σ_a w_a·NPS_a, w_a = N_a/N (volume share).
3. Drivers = assessor labels (improve_categories). We enumerate ALL of them, no frequency
   threshold ("mark the smallest drivers"), separately on:
     - LOSS queries  → realised loss drivers (recoverable uplift = L^(ℓ)/N).
     - DRAW queries  → near-miss drivers (potential, where Yandex almost lost / under-performs).
   We also report WIN drivers (why the agent wins) incl. fragile wins (gpt_refused).
4. Each driver gets one concrete example (query, votes, assessor comment, link).

Output: measurements/v4-2026-05/AGENT_STATUS_FULL.md
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

ROWS = "baskets/v4/sbs_v4_basket_rows.jsonl"
SUMMARY = "baskets/v4/nps_significance_summary.jsonl"
OUT = "measurements/v4-2026-05/AGENT_STATUS_FULL.md"
WIN_THR = 7
CRITERIA = ["relevance", "accuracy", "completeness", "readability"]

CAT_RU = {
    "broken_links": "битые ссылки", "only_links": "только ссылки без текста",
    "illegal_or_unsafe": "небезопасный контент", "hallucination": "галлюцинации/выдуманные факты",
    "time_window_mismatch": "устаревшая инфа", "language_mismatch": "ответ не на том языке",
    "context_lost": "потерян контекст диалога", "irrelevant": "нерелевантно",
    "incomplete": "неполный ответ", "readability": "плохая читаемость",
    "media_broken": "битые медиа", "image_request_unmet": "не выдал запрошенную картинку",
    "gpt_refused": "GPT отказался",
}
CAT_FIX = {
    "incomplete": "полный ответ: прямой ответ первым, покрыть все части запроса",
    "irrelevant": "точнее матчить интент/регион",
    "context_lost": "удерживать контекст диалога",
    "hallucination": "строгий грунтинг, не выдумывать факты",
    "readability": "структура/форматирование, убрать лишние ссылки",
    "broken_links": "валидировать открытие ссылок",
    "only_links": "добавлять текст, а не только ссылки",
    "media_broken": "проверять открытие медиа",
    "image_request_unmet": "выдавать запрошенную картинку",
    "language_mismatch": "отвечать на языке пользователя",
    "time_window_mismatch": "следить за актуальностью данных",
    "illegal_or_unsafe": "контроль безопасного контента",
}


def clean(s, n=170):
    if not s:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip()
    return t[:n] + ("…" if len(t) > n else "")


def split_c(s):
    return [c.strip() for c in str(s or "").split("||") if c.strip() and c.strip() != "."]


def outcome(r):
    y = r.get("ydx_win_cnt") or 0
    g = r.get("gpt_win_cnt") or 0
    if y > WIN_THR and y > g:
        return "win"
    if g > WIN_THR and g > y:
        return "loss"
    return "draw"


def labels_of(r):
    ic = r.get("improve_categories") or ""
    return [x.strip() for x in ic.split(",") if x.strip() and x.strip() not in ("—", "none")]


def best_comment(r):
    """Pick the most informative assessor comment (longest non-noise)."""
    cs = split_c(r.get("ranking_comment")) + split_c(r.get("YandexAI_comment"))
    cs = [c for c in cs if not re.fullmatch(r"[\d\s.\-]+", c)]
    cs.sort(key=len, reverse=True)
    return cs[0] if cs else ""


def ex_line(r, indent="    "):
    net = (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0)
    q = clean(r.get("last_query"), 100)
    c = best_comment(r)
    s = f"{indent}- пример net {net:+d} (YA/GPT {r.get('ydx_win_cnt')}/{r.get('gpt_win_cnt')}): «{q}»"
    if c:
        s += f" — 💬 {clean(c, 150)}"
    links = []
    if r.get("ydx_url"):
        links.append(f"[Yandex AI]({r['ydx_url']})")
    if r.get("gpt_url"):
        links.append(f"[GPT]({r['gpt_url']})")
    if links:
        s += " · " + " · ".join(links)
    return s


def load_summary():
    res = {}
    for line in open(SUMMARY):
        r = json.loads(line)
        if r["basket"] == "v4" and r["filter"] == "all" and r["hypothesis"] == "H1_vs_gpt":
            if r["dimension"] == "agent":
                res[r["value"]] = r
            elif r["dimension"] == "overall":
                res["_overall_"] = r
    return res


def driver_block(title, recs, scope_rows, total_n, kind):
    """Enumerate ALL labels on scope_rows, sorted by count; one example each."""
    out = []
    cat_rows = defaultdict(list)
    for r in scope_rows:
        for c in labels_of(r):
            if kind == "weak" and c == "gpt_refused":
                continue
            cat_rows[c].append(r)
    if not cat_rows:
        out.append(f"  - {title}: меток не выставлено.")
        return out
    out.append(f"  - {title}:")
    for code, rs in sorted(cat_rows.items(), key=lambda kv: -len(kv[1])):
        uplift = 100 * len(rs) / total_n
        fix = CAT_FIX.get(code, "")
        head = (f"    - **{CAT_RU.get(code, code)}** ×{len(rs)}"
                + (f" · потенциал +{uplift:.1f} пп" if kind in ("loss", "draw") else "")
                + (f" · фикс: {fix}" if fix and kind in ("loss", "draw") else ""))
        out.append(head)
        # example: most decisive row for this label
        rs2 = sorted(rs, key=lambda r: (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0))
        out.append(ex_line(rs2[0], indent="      "))
    return out


def main():
    rows = [json.loads(l) for l in open(ROWS)]
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r.get("scenario_tag") or "unknown"].append(r)
    summ = load_summary()
    ov = summ.get("_overall_", {})
    N_all = len(rows)

    agents = sorted([a for a in by_agent if a != "unknown"],
                    key=lambda a: summ.get(a, {}).get("point", 0))

    out = []
    out.append("# v4 — статус по каждому агенту: все драйверы NPS с примерами\n")
    out.append("## Подход\n")
    out.append(
        "Для каждого агента — полный статус и **все драйверы NPS, включая мельчайшие** "
        "(без порога частоты). Драйверы — это метки асессоров (`improve_categories`), "
        "разложенные по типам исходов:\n"
        "- **на проигрышах** — реализованные потери (возврат NPS = доля таких запросов);\n"
        "- **на ничьих** — near-miss: где Yandex недотянул до победы (потенциал);\n"
        "- **на победах** — за счёт чего выигрывает (+ хрупкие победы из-за отказа GPT).\n\n"
        "Формула: $NPS_a=(W_a-L_a)/N_a$, итог $NPS_{total}=\\sum_a w_a NPS_a$, $w_a=N_a/N$. "
        "Возврат рычага $\\ell$: $\\Delta NPS_a=L_a^{(\\ell)}/N_a$ (loss→draw).\n")
    if ov:
        out.append(f"Итог v4: **NPS {ov['point']:+.1f}%**. Агенты — от худшего к лучшему.\n")

    for a in agents:
        recs = by_agent[a]
        N = len(recs)
        wins = [r for r in recs if outcome(r) == "win"]
        losses = [r for r in recs if outcome(r) == "loss"]
        draws = [r for r in recs if outcome(r) == "draw"]
        fragile = [r for r in wins if "gpt_refused" in labels_of(r)]
        s = summ.get(a, {})
        w_a = N / N_all
        contrib = w_a * s.get("point", 0) if s else 0

        verdict = ("🔴 значимое отставание" if s.get("significant") and s.get("point", 0) < 0
                   else "🟢 значимое преимущество" if s.get("significant") and s.get("point", 0) > 0
                   else "⚪ без статзначимого смещения")

        out.append(f"## {a} — {verdict}\n")
        out.append(
            f"- **NPS {s.get('point', 0):+.1f}%** (CI [{s.get('ci_low',0):+.1f}, "
            f"{s.get('ci_high',0):+.1f}], n={N}, вес в корзинке {100*w_a:.1f}%, "
            f"вклад в общий NPS {contrib:+.2f} пп).")
        out.append(
            f"- Исходы: WIN {len(wins)} ({100*len(wins)/N:.0f}%) · "
            f"DRAW {len(draws)} ({100*len(draws)/N:.0f}%) · "
            f"LOSS {len(losses)} ({100*len(losses)/N:.0f}%).")
        # criteria deltas (all four, always)
        crit = []
        for c in CRITERIA:
            ya = sum(r.get(f"YandexAI_{c}") or 0 for r in recs) / N
            gp = sum(r.get(f"Gpt_{c}") or 0 for r in recs) / N
            mark = "🟢" if ya - gp > 0.3 else ("🔴" if gp - ya > 0.3 else "▪️")
            crit.append(f"{mark}{c[:4]} {ya-gp:+.1f}")
        out.append(f"- Критерии vs GPT (Δ галочек/запрос): " + " · ".join(crit) + ".")
        if wins:
            out.append(
                f"- ⚠️ Хрупкие победы (отказ GPT): {len(fragile)}/{len(wins)} "
                f"({100*len(fragile)/len(wins):.0f}%); «честный» NPS без них "
                f"{100*(len(wins)-len(fragile)-len(losses))/N:+.1f}%.")

        # ALL drivers
        out += driver_block("🔻 Драйверы потерь (проигрыши)", recs, losses, N, "loss")
        out += driver_block("🟡 Near-miss драйверы (ничьи)", recs, draws, N, "draw")
        out += driver_block("🟢 За счёт чего выигрывает (победы)", recs, wins, N, "win")
        out.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT} ({len(agents)} agents, {len(out)} lines)")


if __name__ == "__main__":
    main()
