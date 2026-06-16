"""Generate report blocks for BALI-1479: significance table (H1+H2) and rich agent signals."""
import json
import re
from collections import Counter, defaultdict

ROWS = "baskets/v4/sbs_v4_basket_rows.jsonl"
SUMMARY = "baskets/v4/nps_significance_summary.jsonl"
WIN = 7
CAT = {
    "incomplete": "неполнота", "irrelevant": "релевантность", "context_lost": "контекст диалога",
    "hallucination": "достоверность", "readability": "читаемость", "broken_links": "ссылки",
    "only_links": "только ссылки", "media_broken": "медиа", "image_request_unmet": "картинки",
    "language_mismatch": "язык", "time_window_mismatch": "актуальность", "illegal_or_unsafe": "безопасность",
}


def oc(r):
    y, g = r.get("ydx_win_cnt") or 0, r.get("gpt_win_cnt") or 0
    return "win" if (y > WIN and y > g) else ("loss" if (g > WIN and g > y) else "draw")


def labs(r):
    ic = r.get("improve_categories") or ""
    return [x.strip() for x in ic.split(",") if x.strip() and x.strip() not in ("—", "none", "gpt_refused")]


def cl(s, n=70):
    if not s:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip()
    return t[:n] + ("…" if len(t) > n else "")


def bc(r):
    cs = [c.strip() for c in str(r.get("ranking_comment") or "").split("||")
          if c.strip() and not re.fullmatch(r"[\d\s.\-]+", c.strip())]
    cs.sort(key=len, reverse=True)
    return cs[0] if cs else ""


def links(r):
    lk = []
    if r.get("ydx_url"):
        lk.append(f"[Yandex]({r['ydx_url']})")
    if r.get("gpt_url"):
        lk.append(f"[GPT]({r['gpt_url']})")
    return " · ".join(lk)


def example(r, prefix=""):
    return (f"{prefix}«{cl(r.get('last_query'))}» ({r.get('ydx_win_cnt')}:{r.get('gpt_win_cnt')})"
            + (f" — 💬 {cl(bc(r), 110)}" if bc(r) else "") + (f" · {links(r)}" if links(r) else ""))


def main():
    rows = [json.loads(l) for l in open(ROWS)]
    by = defaultdict(list)
    for r in rows:
        by[r.get("scenario_tag") or "unknown"].append(r)

    # significance: H1 + H2, per agent + overall
    sig = defaultdict(dict)
    for l in open(SUMMARY):
        r = json.loads(l)
        if r["basket"] != "v4" or r["filter"] != "all":
            continue
        key = "_TOTAL_" if r["dimension"] == "overall" else (r["value"] if r["dimension"] == "agent" else None)
        if key is None:
            continue
        sig[key][r["hypothesis"]] = r

    agents = sorted([a for a in by if a != "unknown"],
                    key=lambda a: sig.get(a, {}).get("H1_vs_gpt", {}).get("point", 0))

    # ---- BLOCK 1: significance table ----
    out = ["## Приложение. Статзначимость по агентам и в целом\n",
           "Две гипотезы, бутстрап по запросам (5000 реплик), 95% ДИ.\n",
           "- **H1 — смещение относительно GPT**: пороговый NPS = (победы YA − победы GPT)/N. "
           "Значимо ⇔ ДИ не включает 0.\n"
           "- **H2 — смещение относительно 0**: непрерывный NPS = среднее `(голоса_YA − голоса_GPT)/всего`. "
           "Значимо ⇔ ДИ не включает 0.\n",
           "| Срез | n | H1: NPS vs GPT | 95% ДИ | знач. | H2: NPS vs 0 | 95% ДИ | знач. |",
           "|---|--:|--:|---|:--:|--:|---|:--:|"]
    def row(name, key):
        h1 = sig.get(key, {}).get("H1_vs_gpt", {})
        h2 = sig.get(key, {}).get("H2_vs_zero", {})
        return (f"| {name} | {h1.get('n_queries','—')} | {h1.get('point',0):+.1f}% | "
                f"[{h1.get('ci_low',0):+.1f}; {h1.get('ci_high',0):+.1f}] | "
                f"{'✅' if h1.get('significant') else '—'} | "
                f"{h2.get('point',0):+.1f}% | [{h2.get('ci_low',0):+.1f}; {h2.get('ci_high',0):+.1f}] | "
                f"{'✅' if h2.get('significant') else '—'} |")
    out.append(row("**ИТОГО v4**", "_TOTAL_"))
    for a in agents:
        out.append(row(a, a))
    open("/tmp/block_sig.md", "w", encoding="utf-8").write("\n".join(out))

    # ---- BLOCK 2: rich signals per agent ----
    out2 = ["## 4. Сигналы по агентам (тонкие закономерности с примерами)\n",
            "Для каждого проблемного агента — конкретный сигнал, что именно ломается, "
            "и 2 примера с ссылками на оба ответа.\n"]
    # focus on agents with NPS<=0 or notable
    focus = [a for a in agents if sig.get(a, {}).get("H1_vs_gpt", {}).get("point", 0) <= 2]
    for a in focus:
        recs = by[a]
        N = len(recs)
        losses = [r for r in recs if oc(r) == "loss"]
        draws = [r for r in recs if oc(r) == "draw"]
        s = sig.get(a, {}).get("H1_vs_gpt", {})
        # dialog effect
        dlg = [r for r in recs if r.get("is_dialog")]
        nd = [r for r in recs if not r.get("is_dialog")]
        def nps(rs):
            if not rs:
                return None
            w = sum(1 for r in rs if oc(r) == "win"); ls = sum(1 for r in rs if oc(r) == "loss")
            return 100 * (w - ls) / len(rs)
        # top defects on loss+draw
        cnt = Counter()
        for r in losses + draws:
            for c in labs(r):
                cnt[c] += 1
        top = cnt.most_common(3)

        out2.append(f"### {a} — NPS {s.get('point',0):+.1f}% "
                    f"({'значимо' if s.get('significant') else 'не значимо'})\n")
        # signal lines — EACH with an explicit example
        # 1) dialog-worse signal + example dialog loss
        nd_nps, dl_nps = nps(nd), nps(dlg)
        if dl_nps is not None and nd_nps is not None and (nd_nps - dl_nps) > 8:
            dlg_loss = sorted([r for r in dlg if oc(r) == "loss"],
                              key=lambda r: (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0))
            out2.append(f"- 🔎 **В диалогах хуже**: NPS {dl_nps:+.0f}% против {nd_nps:+.0f}% на одиночных "
                        f"(теряет нить разговора).")
            for r in dlg_loss[:2]:
                out2.append(f"  - {example(r)}")
        # 2) worst-criterion signal + example where Yandex is worst on that criterion
        crit = {}
        for c in ["relevance", "accuracy", "completeness", "readability"]:
            ya = sum(r.get(f"YandexAI_{c}") or 0 for r in recs) / N
            gp = sum(r.get(f"Gpt_{c}") or 0 for r in recs) / N
            crit[c] = ya - gp
        worst = min(crit, key=crit.get)
        if crit[worst] < -0.3:
            cand = sorted(recs, key=lambda r: (r.get(f"YandexAI_{worst}") or 0) - (r.get(f"Gpt_{worst}") or 0))
            out2.append(f"- 🔎 **Сильнее всего проседает {worst}** (Δ{crit[worst]:+.1f} галочки/запрос).")
            for r in cand[:2]:
                gy = r.get(f"YandexAI_{worst}") or 0
                gg = r.get(f"Gpt_{worst}") or 0
                out2.append(f"  - {worst} YA/GPT {gy}/{gg} · {example(r)}")
        # group examples BY named defect; for each defect give 2+ examples
        defect_rows = defaultdict(list)
        for r in losses + draws:
            for c in labs(r):
                defect_rows[c].append(r)
        # order defects by how often they cost a loss
        loss_defect = Counter()
        for r in losses:
            for c in labs(r):
                loss_defect[c] += 1
        ordered = [c for c, _ in loss_defect.most_common()]
        # include draw-only defects after
        for c in defect_rows:
            if c not in ordered:
                ordered.append(c)
        for c in ordered[:4]:  # top defects of the agent
            rs = sorted(defect_rows[c],
                        key=lambda r: (r.get("ydx_win_cnt") or 0) - (r.get("gpt_win_cnt") or 0))
            nloss = sum(1 for r in rs if oc(r) == "loss")
            out2.append(f"- **Дефект «{CAT.get(c, c)}»** ({len(rs)} запросов, из них проигрышей {nloss}):")
            # 2-3 examples, prefer losses first
            seen_q = set()
            shown = 0
            for r in rs:
                q = r.get("last_query")
                if q in seen_q:
                    continue
                seen_q.add(q)
                out2.append(f"  - {example(r)}")
                shown += 1
                if shown >= 3:
                    break
        out2.append("")
    open("/tmp/block_signals.md", "w", encoding="utf-8").write("\n".join(out2))

    print("wrote /tmp/block_sig.md and /tmp/block_signals.md")
    print("TOTAL H1:", sig["_TOTAL_"]["H1_vs_gpt"]["point"], "H2:", sig["_TOTAL_"]["H2_vs_zero"]["point"])


if __name__ == "__main__":
    main()
