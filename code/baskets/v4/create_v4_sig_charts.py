#!/usr/bin/env python3
"""Create wizard charts for the v4 significance tab on dataset xfhe9l5tin6ah."""
import json
import os
import uuid

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TOKEN_FILE = "oauth_token.txt"
DATASET_ID = "xfhe9l5tin6ah"
LOGIN = "ai-krotov"
ORG_ID = "111"
BASE_URL = "https://api.datalens.yandex.net"

token = open(TOKEN_FILE).read().strip()
headers = {
    "accept": "application/json",
    "x-dl-api-version": "1",
    "Authorization": f"OAuth {token}",
    "Content-Type": "application/json",
    "x-dl-org-id": ORG_ID,
}

dataset = json.load(open("/tmp/ds_v4boot.json"))["dataset"]
result_schema = dataset["result_schema"]
F = {f["title"]: f for f in result_schema}


def field(title, *, type=None, agg=None, cast=None, alias=None):
    f = F[title]
    return {
        "guid": f["guid"], "title": alias or f["title"],
        "type": type or f["type"], "data_type": f["data_type"],
        "initial_data_type": f["initial_data_type"], "cast": cast or f["cast"],
        "aggregation": agg or f["aggregation"], "calc_mode": "direct",
        "formula": "", "guid_formula": "", "datasetId": DATASET_ID,
        "source": f["source"], "description": "", "valid": True, "hidden": False,
        "virtual": False, "managed_by": "user", "autoaggregated": False,
        "has_auto_aggregation": False, "aggregation_locked": False,
        "lock_aggregation": False, "avatar_id": f.get("avatar_id"),
    }


VIZ = {
    "flatTable": {"id": "flatTable", "type": "flatTable", "name": "Flat table",
                  "allowFilters": True, "allowColors": True, "allowSort": True, "allowLabels": True},
    "column": {"id": "column", "type": "column", "name": "Column chart",
               "allowFilters": True, "allowColors": True, "allowSort": True, "allowLabels": True},
}


def viz_flat(cols):
    v = dict(VIZ["flatTable"])
    v["placeholders"] = [{"id": "flat-table-columns", "items": cols}]
    v["colorsCapacity"] = 1
    return v


def viz_column(dims, measures):
    v = dict(VIZ["column"])
    v["placeholders"] = [{"id": "x", "items": dims},
                         {"id": "y", "items": measures},
                         {"id": "y2", "items": []}]
    v["colorsCapacity"] = 1
    return v


def chart_data(viz, title, colors=None, sort=None, labels=None):
    partial = [{"guid": f["guid"], "title": f["title"]} for f in result_schema]
    return {
        "title": title, "type": "datalens", "version": 12,
        "datasetsIds": [DATASET_ID], "datasetsPartialFields": [partial],
        "visualization": viz, "colors": colors or [], "colorsConfig": {},
        "extraSettings": {"legendMode": "show", "title": title, "titleMode": "show"},
        "filters": [], "geopointsConfig": {}, "hierarchies": [],
        "labels": labels or [], "links": [], "sort": sort or [], "tooltips": [],
        "updates": [], "shapes": [], "shapesConfig": {},
    }


def create(name, data):
    payload = {"template": "datalens", "key": f"Users/{LOGIN}/{name}", "data": data}
    r = requests.post(BASE_URL + "/rpc/createWizardChart", headers=headers, json=payload, verify=False)
    if r.status_code != 200:
        print(f"  FAIL {name}: {r.status_code} {r.text[:400]}")
        return None
    cid = r.json().get("entryId") or r.json().get("entry", {}).get("entryId")
    print(f"  ✓ {name} -> {cid}")
    return cid


def main():
    results = {}

    # 1) Flat table: slice value + n + point + CI + p + significant
    cols = [
        field("dimension"), field("value"),
        field("n_queries", agg="sum"),
        field("point", agg="sum", alias="NPS, %"),
        field("ci_low", agg="sum", alias="CI low"),
        field("ci_high", agg="sum", alias="CI high"),
        field("p_value", agg="sum", alias="p-value"),
        field("significant"),
    ]
    results["v4_sig_table"] = create(
        "v4_sig_table", chart_data(viz_flat(cols), "Значимость v4 — таблица"))

    # 2) Column chart: NPS point by value, colored by significance
    dims = [field("value")]
    measures = [field("point", agg="sum", alias="NPS, %")]
    colors = [field("significant")]
    labels = [field("point", agg="sum", alias="NPS, %")]
    results["v4_sig_column"] = create(
        "v4_sig_column",
        chart_data(viz_column(dims, measures), "NPS по срезам (цвет = значимость)",
                   colors=colors, labels=labels))

    json.dump(results, open("baskets/v4/v4_sig_chart_ids.json", "w"), indent=2)
    print("\nSaved:", results)


if __name__ == "__main__":
    main()
