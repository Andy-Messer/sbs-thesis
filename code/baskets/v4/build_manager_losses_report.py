"""
Manager report — где и из-за каких факторов агенты ПРОИГРЫВАЮТ (v4).

Только проигрыши (GPT победил, >7 голосов). Для менеджеров:
  A. Матрица «агент × фактор проигрыша»: сколько проигрышей каждого агента вызвано
     каждым дефектом (метка асессоров).
  B. По каждому агенту — его факторы проигрыша (доля от проигрышей агента) + пример.
  C. Кросс-срез: по каждому фактору — какие агенты теряют на нём больше всего.

Output: measurements/v4-2026-05/MANAGER_AGENT_LOSSES.md
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

ROWS = "baskets/v4/sbs_v4_basket_rows.jsonl"
OUT = "measurements/v4-2026-05/MANAGER_AGENT_LOSSES.md"
WIN_THR = 7

CAT_RU = {
    "broken_links": "битые ссылки", "only_links": "только ссылки без текста",
    "illegal_or_unsafe": "небезопасный контент", "hallucination": "галлюцинации/выдумки",
    "time_window_mismatch": "устаревшая инфа", "language_mismatch": "ответ не на том языке",
    "context_lost": "потерян контекст диалога", "irrelevant": "нерелевантно",
    "incomplete": "неполный ответ", "readability": "плохая читаемость",
    "media_broken": "битые медиа", "image_request_unmet": "не выдал картинку",
    "gpt_refused": "GPT отказался",
}
FIX = {
    "incomplete": "полный ответ, покрыть все части запроса",
    "irrelevant": "точнее матчить интент/регион",
    "context_lost": "удерживать контекст диалога",
    "hallucination": "не выдумывать факты, грунтинг",
    "readability": "структура/форматирование",
    "broken_links": "валидировать ссылки", "only_links": "добавлять текст к ссылкам",
    "media_broken": "проверять медиа", "image_request_unmet": "выдавать картинку",
    "language_mismatch": "язык пользователя", "time_window_mismatch": "актуальность данных",
}


def clean(s, n=100):
    if not s:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip()
    return t[:n] + ("…" if len(t) > n else "")


def split_c(s):
    return [c.strip() for c in str(s or "").split("||") if c.strip() and c.strip() != "."]


def best_comment(r):
    cs = split_c(r.get("ranking_comment"))
    cs = [c for c in cs if not re.fullmatch(r"[\d\s.\-]+", c)]
    cs.sort(key=len, reverse=True)
    return cs[0] if cs else ""


def outcome(r):
    y, g = r.get("ydx_win_cnt") or 0, r.get("gpt_win_cnt") or 0
    if y > WIN_THR and y > g:
        return "win"
    if g > WIN_THR and g > y:
        return "loss"
    return "draw"


def labels(r):
    ic = r.get("improve_categories") or ""
    return [x.strip() for x in ic.split(",") if x.strip() and x.strip() not in ("—", "none", "gpt_refused")]


def ex(r):
    net = (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0)
    c = best_comment(r)
    s = f"«{clean(r.get('last_query'))}» (YA/GPT {r.get('ydx_win_cnt')}/{r.get('gpt_win_cnt')})"
    if c:
        s += f" — 💬 {clean(c, 130)}"
    lk = []
    if r.get("ydx_url"):
        lk.append(f"[Yandex AI]({r['ydx_url']})")
    if r.get("gpt_url"):
        lk.append(f"[GPT]({r['gpt_url']})")
    if lk:
        s += " · " + " · ".join(lk)
    return s


def main():
    rows = [json.loads(l) for l in open(ROWS)]
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r.get("scenario_tag") or "unknown"].append(r)

    # losses per agent + factor
    agent_losses = {a: [r for r in rs if outcome(r) == "loss"] for a, rs in by_agent.items()}
    agent_factor = {a: Counter() for a in by_agent}
    factor_agent = defaultdict(Counter)
    for a, ls in agent_losses.items():
        for r in ls:
            for c in labels(r):
                agent_factor[a][c] += 1
                factor_agent[c][a] += 1

    agents = sorted([a for a in by_agent if a != "unknown"],
                    key=lambda a: -len(agent_losses[a]))

    out = []
    out.append("# v4 — где и почему агенты проигрывают (для менеджеров)\n")
    out.append(
        "Анализ **только проигрышей** (GPT победил, >7 голосов). Цель — показать, какие "
        "конкретные дефекты приводят к поражениям у каждого агента, с примерами. "
        "Метки — из разметки асессоров (`improve_categories`).\n")
    total_loss = sum(len(v) for v in agent_losses.values())
    out.append(f"Всего проигрышей в корзинке: **{total_loss}** из {len(rows)} запросов "
               f"({100*total_loss/len(rows):.0f}%).\n")

    # ---- A. matrix ----
    all_factors = [c for c, _ in Counter(
        {c: sum(factor_agent[c].values()) for c in factor_agent}).most_common()]
    out.append("## A. Матрица: проигрыши агента по факторам\n")
    head = "| Агент | Проигр. | " + " | ".join(CAT_RU.get(c, c) for c in all_factors) + " |"
    out.append(head)
    out.append("|---|--:|" + "|".join(["--:"] * len(all_factors)) + "|")
    for a in agents:
        if not agent_losses[a]:
            continue
        cells = [str(agent_factor[a].get(c, "") or "") for c in all_factors]
        out.append(f"| {a} | {len(agent_losses[a])} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- B. per agent ----
    out.append("## B. По каждому агенту — факторы проигрышей с примерами\n")
    for a in agents:
        ls = agent_losses[a]
        if not ls:
            continue
        out.append(f"### {a} — {len(ls)} проигрышей\n")
        if not agent_factor[a]:
            out.append("- Проигрыши без выставленных меток (общая слабость ответа).")
        for c, cnt in agent_factor[a].most_common():
            share = 100 * cnt / len(ls)
            fx = FIX.get(c, "")
            out.append(f"- **{CAT_RU.get(c, c)}** — {cnt} из {len(ls)} проигрышей "
                       f"({share:.0f}%)" + (f". Фикс: {fx}." if fx else "."))
            # one example for this factor
            exrows = [r for r in ls if c in labels(r)]
            exrows.sort(key=lambda r: (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0))
            if exrows:
                out.append(f"  - пример: {ex(exrows[0])}")
        out.append("")

    # ---- C. factor -> agents ----
    out.append("## C. Кросс-срез: какой фактор бьёт по каким агентам\n")
    for c in all_factors:
        ag = factor_agent[c]
        tot = sum(ag.values())
        top = " · ".join(f"{a} ({n})" for a, n in ag.most_common(6))
        out.append(f"- **{CAT_RU.get(c, c)}** ({tot} проигрышей всего): {top}.")
    out.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
