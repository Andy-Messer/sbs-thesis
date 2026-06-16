"""
Build a redesigned, concise SBS dashboard into the NEW entry 1jmesyf49xzil.
Reuses existing shared charts (no breakage of the original o5xwsapa9af88).

Tabs (laconic, analysis-friendly):
  1. 📊 Обзор            — KPI + распределение исходов + NPS-карпет (общая картина, basket-куб)
  2. 🤖 Агенты           — сводка по агентам, качество vs GPT, win-rate, что улучшать (basket-куб)
  3. 🔬 Значимость        — bootstrap NPS значимость по корзинам/агентам (sig-куб)
  4. 🔎 Детали           — детальная таблица проблемных запросов (basket-куб)
  5. 🟣 Perplexity        — Yandex vs Perplexity (как было)

Each data-tab carries its own selectors from dataset vcv7i3vsdmjqf (basket cube):
  📦 Корзинка, 🗓 Дата, 🎯 Агент, 💬 Диалог, ✅ Убрать критичные, 🏷 Вердикт.
The significance tab keeps its own selectors (sig cube).
"""
import json

DASH_ID = "1jmesyf49xzil"
DS = "vcv7i3vsdmjqf"  # basket cube

# selector field guids (basket cube)
F_AGENT = "96a6a88b-fd67-4d41-97de-6fe8ac8d290d"
F_DIALOG = "12338cca-3fa3-4dd3-914d-abeecfe78cfe"
F_VERDICT = "5b22e51e-6a2b-4a09-bad4-e6b71e527e35"
F_CRIT = "a40f6c80-5e68-11f1-a3b7-11453f43639a"
F_BASKET = "basket_666s"
F_DATE = "date_w7w1"


def gc(item_id, ctrl_id, field, title, default, tab, multi=True):
    dv = [default] if default not in (None, "") else ""
    return {"id": item_id, "data": {"group": [{
        "id": ctrl_id, "title": title, "width": "",
        "source": {"datasetId": DS, "fieldType": "string", "showTitle": True,
                   "elementType": "select", "datasetFieldId": field,
                   "titlePlacement": "top", "multiselectable": multi,
                   "datasetFieldType": "DIMENSION",
                   **({"defaultValue": dv} if dv != "" else {})},
        "defaults": {field: dv},
        "namespace": "default", "impactType": "currentTab", "sourceType": "dataset",
        "impactTabsIds": [tab], "placementMode": "auto"}],
        "autoHeight": False, "buttonApply": False, "buttonReset": False,
        "showGroupName": False, "updateControlsOnChange": False},
        "type": "group_control", "namespace": "default"}


def widget(item_id, wtab, chart, title, hide_title=False):
    return {"id": item_id, "data": {"tabs": [{
        "id": wtab, "title": title, "params": {}, "chartId": chart, "isDefault": True,
        "autoHeight": False, "background": {"color": "like-chart-bg"}, "description": ""}],
        "hideTitle": hide_title, "background": {"color": "transparent"}},
        "type": "widget", "namespace": "default"}


def text(item_id, txt, auto=True):
    return {"id": item_id, "data": {"text": txt, "autoHeight": auto,
            "background": {"color": "transparent"}}, "type": "text", "namespace": "default"}


def basket_selectors(prefix, tab, with_critical=True, with_verdict=False):
    """Standard selector row for basket-cube tabs."""
    items = [
        gc(f"{prefix}_bsk", f"{prefix}_bsk_c", F_BASKET, "📦 Корзинка", "v4", tab, multi=False),
        gc(f"{prefix}_dt", f"{prefix}_dt_c", F_DATE, "🗓 Дата", "", tab),
        gc(f"{prefix}_ag", f"{prefix}_ag_c", F_AGENT, "🎯 Агент", "", tab),
        gc(f"{prefix}_dl", f"{prefix}_dl_c", F_DIALOG, "💬 Диалог", "", tab, multi=False),
    ]
    if with_critical:
        items.append(gc(f"{prefix}_cr", f"{prefix}_cr_c", F_CRIT, "✅ Без критичных", "True", tab))
    if with_verdict:
        items.append(gc(f"{prefix}_vd", f"{prefix}_vd_c", F_VERDICT, "🏷 Вердикт", "", tab))
    return items


def selector_layout(prefix, ids, y=3):
    """Spread selector items across the 36-grid header row."""
    n = len(ids)
    w = max(6, 36 // n)
    out = []
    x = 0
    for i in ids:
        out.append({"i": i, "x": x, "y": y, "w": w, "h": 3, "parent": "__fixHead"})
        x += w
    return out


def main():
    tabs = []

    # ---------- TAB 1: 📊 Обзор ----------
    T = "ov"
    sels = basket_selectors("ov", T, with_critical=True)
    sel_ids = [s["id"] for s in sels]
    items = sels + [
        text("ov_h", "# 📊 Обзор корзинки\nКлючевые метрики Yandex AI vs GPT по выбранной "
             "корзинке. Управление — селекторами выше (корзинка, дата, агент, диалог, фильтр)."),
        widget("ov_q", "ov_q_t", "j0hbfzaw4ycu3", "Запросов всего", hide_title=True),
        widget("ov_nps", "ov_nps_t", "bs938yqqoqxqv", "SBS-like NPS", hide_title=True),
        widget("ov_yw", "ov_yw_t", "9q715qrgap9at", "Yandex wins", hide_title=True),
        widget("ov_dr", "ov_dr_t", "r8pjobox9w86b", "Draws", hide_title=True),
        widget("ov_gw", "ov_gw_t", "9q716vzd4eaqt", "GPT wins", hide_title=True),
        text("ov_h2", "## 🗳 Как голосуют ассессоры"),
        widget("ov_wr", "ov_wr_t", "taqbh1c6t2gqd", "Win-rate %"),
        widget("ov_rank", "ov_rank_t", "vcsdmt97oovmf", "Δ rank"),
    ]
    layout = selector_layout("ov", sel_ids)
    layout += [
        {"i": "ov_h", "x": 0, "y": 0, "w": 36, "h": 3},
        {"i": "ov_q", "x": 0, "y": 6, "w": 8, "h": 5},
        {"i": "ov_nps", "x": 8, "y": 6, "w": 8, "h": 5},
        {"i": "ov_yw", "x": 16, "y": 6, "w": 7, "h": 5},
        {"i": "ov_dr", "x": 23, "y": 6, "w": 6, "h": 5},
        {"i": "ov_gw", "x": 29, "y": 6, "w": 7, "h": 5},
        {"i": "ov_h2", "x": 0, "y": 11, "w": 36, "h": 1},
        {"i": "ov_wr", "x": 0, "y": 12, "w": 18, "h": 9},
        {"i": "ov_rank", "x": 18, "y": 12, "w": 18, "h": 9},
    ]
    tabs.append({"id": T, "title": "📊 Обзор", "items": items, "layout": layout,
                 "connections": [], "aliases": {}})

    # ---------- TAB 2: 🤖 Агенты ----------
    T = "ag"
    sels = basket_selectors("ag", T, with_critical=True)
    sel_ids = [s["id"] for s in sels]
    items = sels + [
        text("ag_h", "# 🤖 Анализ по агентам\nКачество ответов Yandex AI против GPT в разрезе "
             "агентов: сводка, средние оценки по критериям, win-rate и что улучшать."),
        widget("ag_sum", "ag_sum_t", "8p88jbjioomws", "Сводка по агентам"),
        text("ag_h2", "## 🧪 Качество vs GPT (Yandex avg · GPT avg · Δ)"),
        widget("ag_ya", "ag_ya_t", "vcsdj8zw6452f", "Yandex AI: средняя оценка"),
        widget("ag_gp", "ag_gp_t", "l2i390f8epui5", "ChatGPT: средняя оценка"),
        widget("ag_d", "ag_d_t", "l2i3cikvg4665", "Δ Yandex − GPT"),
        text("ag_h3", "## 🎯 Что улучшать (топ-причин) и вердикты ассессоров"),
        widget("ag_top", "ag_top_t", "0h032wqq3glok", "Топ-причин проблем"),
        widget("ag_vd", "ag_vd_t", "l2i39xakzkgm5", "Вердикты ассессоров"),
    ]
    layout = selector_layout("ag", sel_ids)
    layout += [
        {"i": "ag_h", "x": 0, "y": 0, "w": 36, "h": 3},
        {"i": "ag_sum", "x": 0, "y": 6, "w": 36, "h": 10},
        {"i": "ag_h2", "x": 0, "y": 16, "w": 36, "h": 1},
        {"i": "ag_ya", "x": 0, "y": 17, "w": 12, "h": 9},
        {"i": "ag_gp", "x": 12, "y": 17, "w": 12, "h": 9},
        {"i": "ag_d", "x": 24, "y": 17, "w": 12, "h": 9},
        {"i": "ag_h3", "x": 0, "y": 26, "w": 36, "h": 1},
        {"i": "ag_top", "x": 0, "y": 27, "w": 18, "h": 10},
        {"i": "ag_vd", "x": 18, "y": 27, "w": 18, "h": 10},
    ]
    tabs.append({"id": T, "title": "🤖 Агенты", "items": items, "layout": layout,
                 "connections": [], "aliases": {}})

    # ---------- TAB 3: 🔬 Значимость (reuse existing Sig4 tab as-is) ----------
    sig = json.load(open("/tmp/tab_Sig4.json"))
    sig["title"] = "🔬 Значимость"
    tabs.append(sig)

    # ---------- TAB 4: 🔎 Детали ----------
    T = "dt"
    sels = basket_selectors("dt", T, with_critical=False, with_verdict=True)
    sel_ids = [s["id"] for s in sels]
    items = sels + [
        text("dt_h", "# 🔎 Детальные запросы\nВсе запросы с категорией проблемы, голосами по 4 "
             "критериям, кликабельными ссылками (Yandex/GPT) и полными комментариями асессоров. "
             "Фильтруйте по агенту, вердикту, диалогу."),
        widget("dt_tbl", "dt_tbl_t", "dudmvbg5v98yx", "Запросы (ссылки и критерии)"),
    ]
    layout = selector_layout("dt", sel_ids)
    layout += [
        {"i": "dt_h", "x": 0, "y": 0, "w": 36, "h": 3},
        {"i": "dt_tbl", "x": 0, "y": 6, "w": 36, "h": 20},
    ]
    tabs.append({"id": T, "title": "🔎 Детали", "items": items, "layout": layout,
                 "connections": [], "aliases": {}})

    # ---------- TAB 5: 🟣 Perplexity (reuse vM) ----------
    vm = json.load(open("/tmp/tab_vM.json"))
    vm["title"] = "🟣 Perplexity"
    tabs.append(vm)

    # ---------- assemble data ----------
    base = json.load(open("/tmp/dash.json"))["entry"]["data"]
    data = {
        "counter": base.get("counter", 200) + 1,
        "salt": base.get("salt", "0.5"),
        "schemeVersion": base.get("schemeVersion", 8),
        "settings": base.get("settings", {}),
        "tabs": tabs,
    }
    json.dump(data, open("/tmp/redesign_data.json", "w"), ensure_ascii=False)
    print("tabs:", [(t["id"], t["title"]) for t in tabs])
    print("wrote /tmp/redesign_data.json")


if __name__ == "__main__":
    main()
