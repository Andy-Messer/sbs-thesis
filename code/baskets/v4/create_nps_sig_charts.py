#!/usr/bin/env python3
"""Create all charts for the universal NPS-significance tab.

Datasets:
  summary  4mpuxtvm0qw0o  (point/CI/p/significant per basket×filter×slice, +agent)
  dist     ygjjoct9uneci  (bootstrap histogram bins for bell curves)
  mix      asv0325bmoxwu  (win/draw/loss share per slice)

Charts:
  1 bell      — area/bin bell curves of bootstrap-NPS, one series per basket
  2 ci_col    — sorted column of point NPS, colored by significance
  3 table     — full stats table
  4 pval      — sorted column of p-value (visual 0.05 reference via color)
  5 mix       — 100% stacked column win/draw/loss per slice
  6 scorecard — flat table styled as KPI (basket × NPS × CI × sig) for current selection
"""
import json
import os

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TOKEN = open("oauth_token.txt").read().strip()
LOGIN = "ai-krotov"
BASE = "https://api.datalens.yandex.net"
HEAD = {"accept": "application/json", "x-dl-api-version": "1",
        "Authorization": f"OAuth {TOKEN}", "Content-Type": "application/json",
        "x-dl-org-id": "111"}

DS_SUM = "4mpuxtvm0qw0o"
DS_DIST = "ygjjoct9uneci"
DS_MIX = "asv0325bmoxwu"
SUM = json.load(open("/tmp/ds_sum.json"))["dataset"]["result_schema"]
DIST = json.load(open("/tmp/ds_dist.json"))["dataset"]["result_schema"]
MIX = json.load(open("/tmp/ds_mix.json"))["dataset"]["result_schema"]
SUMF = {f["title"]: f for f in SUM}
DISTF = {f["title"]: f for f in DIST}
MIXF = {f["title"]: f for f in MIX}


def fref(F, ds_id, title, *, type=None, agg=None, cast=None, alias=None, direction=None):
    f = F[title]
    if type == "DIMENSION" and agg is None:
        agg = "none"
    item = {"guid": f["guid"], "title": alias or f["title"], "type": type or f["type"],
            "data_type": f["data_type"], "initial_data_type": f["initial_data_type"],
            "cast": cast or f["cast"], "aggregation": agg or f["aggregation"],
            "calc_mode": "direct", "formula": "", "guid_formula": "", "datasetId": ds_id,
            "source": f["source"], "description": "", "valid": True, "hidden": False,
            "virtual": False, "managed_by": "user", "autoaggregated": False,
            "has_auto_aggregation": False, "aggregation_locked": False,
            "lock_aggregation": False, "avatar_id": f.get("avatar_id")}
    if direction:
        item["direction"] = direction
    return item


VIZ = {
    "area": {"id": "area", "type": "area", "name": "Area chart"},
    "column": {"id": "column", "type": "column", "name": "Column chart"},
    "flatTable": {"id": "flatTable", "type": "flatTable", "name": "Flat table"},
}
for v in VIZ.values():
    v.update({"allowFilters": True, "allowColors": True, "allowSort": True, "allowLabels": True})


def data(ds_id, schema, viz, title, colors=None, sort=None, labels=None, stack=None):
    partial = [{"guid": f["guid"], "title": f["title"]} for f in schema]
    extra = {"legendMode": "show", "title": title, "titleMode": "show"}
    if stack:
        extra["barMode"] = stack
    return {"title": title, "type": "datalens", "version": 12, "datasetsIds": [ds_id],
            "datasetsPartialFields": [partial], "visualization": viz,
            "colors": colors or [], "colorsConfig": {}, "extraSettings": extra,
            "filters": [], "geopointsConfig": {}, "hierarchies": [], "labels": labels or [],
            "links": [], "sort": sort or [], "tooltips": [], "updates": [], "shapes": [],
            "shapesConfig": {}}


def create(name, payload_data):
    r = requests.post(BASE + "/rpc/createWizardChart", headers=HEAD,
                      json={"template": "datalens", "key": f"Users/{LOGIN}/{name}",
                            "data": payload_data}, verify=False)
    if r.status_code != 200:
        print(f"  FAIL {name}: {r.status_code} {r.text[:300]}")
        return None
    cid = r.json().get("entryId") or r.json().get("entry", {}).get("entryId")
    print(f"  ✓ {name} -> {cid}")
    return cid


def main():
    out = {}

    # 1) bell curves — AREA chart, bins on X, density on Y, color by basket
    v = dict(VIZ["area"])
    v["placeholders"] = [
        {"id": "x", "items": [fref(DISTF, DS_DIST, "bin_center", type="DIMENSION", cast="float")]},
        {"id": "y", "items": [fref(DISTF, DS_DIST, "density", agg="sum")]},
        {"id": "y2", "items": []}]
    v["colorsCapacity"] = 10
    out["nps_bellcurves"] = create("nps_bellcurves", data(
        DS_DIST, DIST, v, "🔔 Распределения bootstrap-NPS (бины) по корзинам",
        colors=[fref(DISTF, DS_DIST, "basket")]))

    # 2) CI column — point NPS by value, sorted desc, color by significance
    v = dict(VIZ["column"])
    v["placeholders"] = [
        {"id": "x", "items": [fref(SUMF, DS_SUM, "value")]},
        {"id": "y", "items": [fref(SUMF, DS_SUM, "point", agg="sum", alias="NPS, %")]},
        {"id": "y2", "items": []}]
    v["colorsCapacity"] = 1
    out["nps_ci_column"] = create("nps_ci_column", data(
        DS_SUM, SUM, v, "📊 NPS по срезам (сортировка, цвет = значимость)",
        colors=[fref(SUMF, DS_SUM, "significant")],
        sort=[fref(SUMF, DS_SUM, "point", agg="sum", alias="NPS, %", direction="DESC")],
        labels=[fref(SUMF, DS_SUM, "point", agg="sum", alias="NPS, %")]))

    # 3) full stats table
    v = dict(VIZ["flatTable"])
    cols = [fref(SUMF, DS_SUM, "basket"), fref(SUMF, DS_SUM, "value"),
            fref(SUMF, DS_SUM, "n_queries", agg="sum"),
            fref(SUMF, DS_SUM, "point", agg="sum", alias="NPS, %"),
            fref(SUMF, DS_SUM, "ci_low", agg="sum", alias="CI low"),
            fref(SUMF, DS_SUM, "ci_high", agg="sum", alias="CI high"),
            fref(SUMF, DS_SUM, "p_value", agg="sum", alias="p-value"),
            fref(SUMF, DS_SUM, "significant")]
    v["placeholders"] = [{"id": "flat-table-columns", "items": cols}]
    v["colorsCapacity"] = 1
    out["nps_sig_table"] = create("nps_sig_table", data(
        DS_SUM, SUM, v, "📋 Значимость NPS — таблица",
        sort=[fref(SUMF, DS_SUM, "point", agg="sum", alias="NPS, %", direction="DESC")]))

    # 4) p-value column — sorted asc, color by significance (significant = p<0.05)
    v = dict(VIZ["column"])
    v["placeholders"] = [
        {"id": "x", "items": [fref(SUMF, DS_SUM, "value")]},
        {"id": "y", "items": [fref(SUMF, DS_SUM, "p_value", agg="sum", alias="p-value")]},
        {"id": "y2", "items": []}]
    v["colorsCapacity"] = 1
    out["nps_pvalue"] = create("nps_pvalue", data(
        DS_SUM, SUM, v, "🎯 p-value по срезам (зелёный = значимо, p<0.05)",
        colors=[fref(SUMF, DS_SUM, "significant")],
        sort=[fref(SUMF, DS_SUM, "p_value", agg="sum", alias="p-value", direction="ASC")],
        labels=[fref(SUMF, DS_SUM, "p_value", agg="sum", alias="p-value")]))

    # 5) outcome mix — 100% stacked column, x=value, y=share, color=outcome
    v = dict(VIZ["column"])
    v["placeholders"] = [
        {"id": "x", "items": [fref(MIXF, DS_MIX, "value")]},
        {"id": "y", "items": [fref(MIXF, DS_MIX, "share", agg="sum", alias="Доля, %")]},
        {"id": "y2", "items": []}]
    v["colorsCapacity"] = 3
    out["nps_outcome_mix"] = create("nps_outcome_mix", data(
        DS_MIX, MIX, v, "🟦 Структура исходов (Yandex win / draw / GPT win), %",
        colors=[fref(MIXF, DS_MIX, "outcome")],
        labels=[fref(MIXF, DS_MIX, "share", agg="sum", alias="Доля, %")],
        stack="percent"))

    # 6) KPI scorecard table — compact basket-level
    v = dict(VIZ["flatTable"])
    cols = [fref(SUMF, DS_SUM, "basket"),
            fref(SUMF, DS_SUM, "point", agg="sum", alias="NPS, %"),
            fref(SUMF, DS_SUM, "ci_low", agg="sum", alias="CI low"),
            fref(SUMF, DS_SUM, "ci_high", agg="sum", alias="CI high"),
            fref(SUMF, DS_SUM, "significant")]
    v["placeholders"] = [{"id": "flat-table-columns", "items": cols}]
    v["colorsCapacity"] = 1
    out["nps_scorecard"] = create("nps_scorecard", data(
        DS_SUM, SUM, v, "🏁 NPS по корзинам (текущий срез)"))

    json.dump(out, open("baskets/v4/nps_sig_chart_ids.json", "w"), indent=2)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
