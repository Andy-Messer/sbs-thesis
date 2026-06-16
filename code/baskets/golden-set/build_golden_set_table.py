"""
Build a consolidated golden-set analysis table.

For every query (from input.xlsx) collect rows of three rater types:
  - assessor  (up to 10 unique workers per query, from asessors-*-pool.tsv)
  - manager   (from managers.xlsx Table1..Table4 sheets)
  - corrected (from disagreements.xlsx new_verdict, when present)

Each row carries:
  * query metadata
  * verdict produced by that rater
  * per-model (Yandex AI / Gpt / Perplexity) checkbox flags, comments, rank, score
  * free-form ranking_comment

Output: baskets/golden-set/golden_set_full.xlsx
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("baskets/golden-set")
INPUT_PATH = ROOT / "input.xlsx"
MANAGERS_PATH = ROOT / "managers.xlsx"
DISAGREE_PATH = ROOT / "disagreements.xlsx"
ASR_POOLS = [
    ROOT / "asessors-nondialog-pool.tsv",
    ROOT / "asessors-dialog-pool.tsv",
]
OUT_PATH = ROOT / "golden_set_full.xlsx"

MODELS = ["YandexAI", "Gpt", "Perplexity"]
CHECKBOX_FIELDS = ["not_open", "relevance", "accuracy", "completeness", "readability"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalize_verdict(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in {"ydx", "ydx_win", "yandex", "yandex ai", "yandex_ai", "yandexai"}:
        return "ydx_win"
    if s in {"gpt", "gpt_win", "chatgpt"}:
        return "gpt_win"
    if s in {"draw", "tie"}:
        return "draw"
    return s


def verdict_from_ranks(ydx_rank, gpt_rank) -> str | None:
    try:
        y = float(ydx_rank)
        g = float(gpt_rank)
    except (TypeError, ValueError):
        return None
    if pd.isna(y) or pd.isna(g):
        return None
    if y < g:
        return "ydx_win"
    if g < y:
        return "gpt_win"
    return "draw"


def parse_checkboxes(raw: str) -> list[dict]:
    """The OUTPUT:models_checkboxes column contains a comma-separated list of
    JSON objects with backslash-escaped commas inside, e.g.

        {"relevance":true\\,"accuracy":true},{"relevance":true\\,...}

    Returns one dict per model (in the order of INPUT:models_names).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).replace("\\,", ",")
    # split on "},{" boundaries
    parts = re.split(r"\}\s*,\s*\{", text.strip())
    if not parts:
        return []
    parts = [p if p.startswith("{") else "{" + p for p in parts]
    parts = [p if p.endswith("}") else p + "}" for p in parts]
    out = []
    for p in parts:
        try:
            out.append(json.loads(p))
        except Exception:
            out.append({})
    return out


def empty_model_block() -> dict:
    return {
        "rank": None,
        "score": None,
        "not_open": None,
        "relevance": None,
        "accuracy": None,
        "completeness": None,
        "readability": None,
        "comment": None,
    }


# ---------------------------------------------------------------------------
# load input.xlsx (drop schema row)
# ---------------------------------------------------------------------------

print("Loading input.xlsx ...")
inp = pd.read_excel(INPUT_PATH)
# drop the first row (which holds yt-style type definitions)
inp = inp[inp["query_id"].astype(str).str.match(r"^\d+$")].copy()
inp["query_id"] = inp["query_id"].astype(str)
print(f"  queries in input: {len(inp)}")

input_meta_cols = [
    "query_id", "last_query", "scenario_tag", "is_dialog",
    "region_id", "region_name", "geo", "device", "source",
    "dialog_id", "shared_dialog_id", "shared_dialog_url",
    "ydx_url", "gpt_url", "ppx_url",
]
input_meta_cols = [c for c in input_meta_cols if c in inp.columns]
inp_meta = inp[input_meta_cols].drop_duplicates("query_id").set_index("query_id")


# ---------------------------------------------------------------------------
# load assessor pools
# ---------------------------------------------------------------------------

print("Loading assessor pools ...")
asr_frames = []
for path in ASR_POOLS:
    df = pd.read_csv(path, sep="\t")
    df["pool_file"] = path.name
    asr_frames.append(df)
asr = pd.concat(asr_frames, ignore_index=True)
asr = asr[asr["ASSIGNMENT:status"] == "APPROVED"].copy()
print(f"  approved assessor assignments: {len(asr)}")


def asr_to_rows(row) -> list[dict]:
    meta_raw = row.get("INPUT:meta")
    if pd.isna(meta_raw):
        return []
    try:
        meta = json.loads(meta_raw)
    except Exception:
        return []
    qid = str(meta.get("query_id"))
    names_raw = row.get("INPUT:models_names")
    ranks_raw = row.get("OUTPUT:models_ranks")
    if pd.isna(names_raw) or pd.isna(ranks_raw):
        return []
    names = [n.strip() for n in str(names_raw).split(",")]
    ranks = [r.strip() for r in str(ranks_raw).split(",")]
    checkboxes = parse_checkboxes(row.get("OUTPUT:models_checkboxes"))
    model_blocks = {m: empty_model_block() for m in MODELS}
    for i, name in enumerate(names):
        # map "Yandex AI" -> "YandexAI"
        key = name.replace(" ", "")
        if key not in model_blocks:
            continue
        rank = ranks[i] if i < len(ranks) else None
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = None
        cb = checkboxes[i] if i < len(checkboxes) else {}
        block = model_blocks[key]
        block["rank"] = rank
        block["relevance"] = cb.get("relevance")
        block["accuracy"] = cb.get("accuracy")
        block["completeness"] = cb.get("completeness")
        block["readability"] = cb.get("readability")
        block["comment"] = cb.get("comment")
    # not_open is a single boolean per task
    not_open = row.get("OUTPUT:model_not_open")
    for m in MODELS:
        model_blocks[m]["not_open"] = not_open

    verdict = verdict_from_ranks(model_blocks["YandexAI"]["rank"],
                                 model_blocks["Gpt"]["rank"])

    return [{
        "query_id": qid,
        "rater_type": "assessor",
        "rater_id": row.get("ASSIGNMENT:worker_id"),
        "verdict": verdict,
        "ranking_comment": row.get("OUTPUT:ranking_comment"),
        "assignment_id": row.get("ASSIGNMENT:assignment_id"),
        "assignment_link": row.get("ASSIGNMENT:link"),
        "pool_file": row.get("pool_file"),
        "started_at": row.get("ASSIGNMENT:started"),
        "_models": model_blocks,
    }]


asr_rows: list[dict] = []
for _, r in asr.iterrows():
    asr_rows.extend(asr_to_rows(r))
print(f"  assessor rows after parsing: {len(asr_rows)}")

# Cap to <=10 unique workers per query (keep earliest assignment per worker, then first 10 workers)
asr_df = pd.DataFrame(asr_rows)
asr_df["started_at"] = pd.to_datetime(asr_df["started_at"], errors="coerce")
asr_df = asr_df.sort_values(["query_id", "started_at"])
asr_df = asr_df.drop_duplicates(["query_id", "rater_id"], keep="first")
asr_df = asr_df.groupby("query_id", group_keys=False).head(10).reset_index(drop=True)
print(f"  assessor rows after dedup + cap(10): {len(asr_df)}")


# ---------------------------------------------------------------------------
# load managers
# ---------------------------------------------------------------------------

print("Loading managers.xlsx ...")
manager_sheets = ["Table1", "Table2", "Table3", "Table4"]
mgr_parts = []
for s in manager_sheets:
    try:
        m = pd.read_excel(MANAGERS_PATH, sheet_name=s)
        m["sheet"] = s
        mgr_parts.append(m)
    except Exception as exc:
        print(f"  WARN: cannot read {s}: {exc}")
mgr = pd.concat(mgr_parts, ignore_index=True)
mgr = mgr[mgr["has_verdict"] == 1].copy()
print(f"  manager rows with verdict: {len(mgr)}")

# match managers to query_id via Yandex AI url (== input.shared_dialog_url) or fallback by Query text
url_to_qid = {}
for qid, row in inp_meta.iterrows():
    for col in ("shared_dialog_url", "ydx_url"):
        u = row.get(col)
        if isinstance(u, str) and u:
            url_to_qid[u] = qid
last_query_to_qid = {str(row["last_query"]).strip(): qid
                     for qid, row in inp_meta.iterrows()
                     if isinstance(row.get("last_query"), str)}


def mgr_qid(row) -> str | None:
    yurl = row.get("Yandex AI_link") or row.get("YandexAI")
    if isinstance(yurl, str) and yurl in url_to_qid:
        return url_to_qid[yurl]
    q = row.get("Query")
    if isinstance(q, str) and q.strip() in last_query_to_qid:
        return last_query_to_qid[q.strip()]
    return None


mgr["query_id"] = mgr.apply(mgr_qid, axis=1)
missing = mgr["query_id"].isna().sum()
print(f"  manager rows without query_id match: {missing}")

mgr_rows = []
for _, r in mgr.iterrows():
    qid = r["query_id"]
    if qid is None:
        continue
    blocks = {m: empty_model_block() for m in MODELS}
    blocks["YandexAI"].update({
        "not_open":     r.get("yazeka_not_open"),
        "relevance":    r.get("yazeka_relevance"),
        "accuracy":     r.get("yazeka_accuracy"),
        "completeness": r.get("yazeka_completeness"),
        "readability":  r.get("yazeka_readability"),
        "comment":      r.get("yazeka_comment"),
        "rank":         r.get("yazeka_rating"),
        "score":        r.get("yazeka_score"),
    })
    blocks["Gpt"].update({
        "not_open":     r.get("gpt_not_open"),
        "relevance":    r.get("gpt_relevance"),
        "accuracy":     r.get("gpt_accuracy"),
        "completeness": r.get("gpt_completeness"),
        "readability":  r.get("gpt_readability"),
        "comment":      r.get("gpt_comment"),
        "rank":         r.get("gpt_rating"),
        "score":        r.get("gpt_score"),
    })
    blocks["Perplexity"].update({
        "not_open":     r.get("perplexity_not_open"),
        "relevance":    r.get("perplexity_relevance"),
        "accuracy":     r.get("perplexity_accuracy"),
        "completeness": r.get("perplexity_completeness"),
        "readability":  r.get("perplexity_readability"),
        "comment":      r.get("perplexity_comment"),
        "rank":         r.get("perplexity_rating"),
        "score":        r.get("perplexity_score"),
    })
    mgr_rows.append({
        "query_id": qid,
        "rater_type": "manager",
        "rater_id": r.get("reviewer"),
        "verdict": normalize_verdict(r.get("verdict")),
        "ranking_comment": r.get("ranking_comment"),
        "assignment_id": None,
        "assignment_link": None,
        "pool_file": r.get("sheet"),
        "started_at": None,
        "_models": blocks,
    })
mgr_df = pd.DataFrame(mgr_rows)
print(f"  manager rows usable: {len(mgr_df)}")


# ---------------------------------------------------------------------------
# load disagreements -> 'corrected' rows
# ---------------------------------------------------------------------------

print("Loading disagreements.xlsx ...")
dis = pd.read_excel(DISAGREE_PATH, sheet_name="Disagreements")
dis["query_id"] = dis["query_id"].astype(str)
dis_corrections = dis[dis["new_verdict"].notna()].copy()
print(f"  disagreements with new_verdict: {len(dis_corrections)}")

dis_lookup: dict[str, dict] = {}
corrected_rows = []
for _, r in dis_corrections.iterrows():
    qid = r["query_id"]
    dis_lookup[qid] = {
        "new_verdict": normalize_verdict(r.get("new_verdict")),
        "mgr_verdict": normalize_verdict(r.get("mgr_verdict")),
        "asr_verdict": normalize_verdict(r.get("asr_verdict")),
        "disagree_type": r.get("disagree_type"),
        "reason": r.get("reason"),
    }
    blocks = {m: empty_model_block() for m in MODELS}
    corrected_rows.append({
        "query_id": qid,
        "rater_type": "corrected",
        "rater_id": "review_committee",
        "verdict": normalize_verdict(r.get("new_verdict")),
        "ranking_comment": r.get("reason"),
        "assignment_id": None,
        "assignment_link": None,
        "pool_file": "disagreements",
        "started_at": None,
        "_models": blocks,
    })
corr_df = pd.DataFrame(corrected_rows)


# ---------------------------------------------------------------------------
# override manager verdicts when corrections exist
# ---------------------------------------------------------------------------

if not mgr_df.empty:
    mgr_df["original_mgr_verdict"] = mgr_df["verdict"]
    def patch(row):
        c = dis_lookup.get(row["query_id"])
        if c and c["new_verdict"]:
            return c["new_verdict"]
        return row["verdict"]
    mgr_df["verdict"] = mgr_df.apply(patch, axis=1)
    mgr_df["verdict_corrected_flag"] = (
        mgr_df["verdict"] != mgr_df["original_mgr_verdict"]
    )
else:
    mgr_df["original_mgr_verdict"] = []
    mgr_df["verdict_corrected_flag"] = []


# ---------------------------------------------------------------------------
# flatten _models into wide columns
# ---------------------------------------------------------------------------

def flatten_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, r in df.iterrows():
        flat = {k: v for k, v in r.items() if k != "_models"}
        for m in MODELS:
            blk = r["_models"][m]
            for k, v in blk.items():
                flat[f"{m}_{k}"] = v
        rows.append(flat)
    return pd.DataFrame(rows)


all_df = pd.concat(
    [flatten_models(asr_df), flatten_models(mgr_df), flatten_models(corr_df)],
    ignore_index=True, sort=False,
)


# ---------------------------------------------------------------------------
# attach query metadata + disagreement context to every row
# ---------------------------------------------------------------------------

all_df = all_df.merge(inp_meta, left_on="query_id", right_index=True, how="left")

# disagreement context columns for *every* row in the query
ctx = pd.DataFrame.from_dict(dis_lookup, orient="index").reset_index().rename(
    columns={"index": "query_id"})
if not ctx.empty:
    ctx = ctx.rename(columns={
        "new_verdict":   "corrected_verdict",
        "mgr_verdict":   "disagreement_mgr_verdict",
        "asr_verdict":   "disagreement_asr_verdict",
        "disagree_type": "disagreement_type",
        "reason":        "disagreement_reason",
    })
    all_df = all_df.merge(ctx, on="query_id", how="left")
else:
    for c in ("corrected_verdict", "disagreement_mgr_verdict",
              "disagreement_asr_verdict", "disagreement_type",
              "disagreement_reason"):
        all_df[c] = None

# convenience flag
all_df["has_correction"] = all_df["corrected_verdict"].notna()


# ---------------------------------------------------------------------------
# column ordering
# ---------------------------------------------------------------------------

meta_cols = [
    "query_id", "last_query", "scenario_tag", "is_dialog",
    "region_name", "region_id", "geo", "device", "source",
    "ydx_url", "gpt_url", "ppx_url", "shared_dialog_url",
]
meta_cols = [c for c in meta_cols if c in all_df.columns]

rater_cols = [
    "rater_type", "rater_id", "verdict", "original_mgr_verdict",
    "verdict_corrected_flag", "has_correction",
    "corrected_verdict", "disagreement_mgr_verdict",
    "disagreement_asr_verdict", "disagreement_type", "disagreement_reason",
    "ranking_comment",
    "assignment_id", "assignment_link", "pool_file", "started_at",
]
rater_cols = [c for c in rater_cols if c in all_df.columns]

model_cols = []
for m in MODELS:
    for f in ["rank", "score", "not_open", "relevance", "accuracy",
              "completeness", "readability", "comment"]:
        col = f"{m}_{f}"
        if col in all_df.columns:
            model_cols.append(col)

ordered = meta_cols + rater_cols + model_cols
remaining = [c for c in all_df.columns if c not in ordered]
all_df = all_df[ordered + remaining]

# sort: query, rater_type order, rater_id
rater_order = {"assessor": 0, "manager": 1, "corrected": 2}
all_df["_rt_order"] = all_df["rater_type"].map(rater_order)
all_df = all_df.sort_values(
    ["query_id", "_rt_order", "rater_id"], na_position="last"
).drop(columns="_rt_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# also produce a per-query summary sheet
# ---------------------------------------------------------------------------

def majority(series: pd.Series) -> str | None:
    s = series.dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()


summary = all_df.groupby("query_id").apply(lambda g: pd.Series({
    "last_query":            g["last_query"].iloc[0] if "last_query" in g else None,
    "scenario_tag":          g["scenario_tag"].iloc[0] if "scenario_tag" in g else None,
    "is_dialog":             g["is_dialog"].iloc[0] if "is_dialog" in g else None,
    "n_assessors":           (g["rater_type"] == "assessor").sum(),
    "n_managers":            (g["rater_type"] == "manager").sum(),
    "assessor_verdict_top":  majority(g.loc[g["rater_type"] == "assessor", "verdict"]),
    "manager_verdict_top":   majority(g.loc[g["rater_type"] == "manager", "original_mgr_verdict"]),
    "corrected_verdict":     g["corrected_verdict"].iloc[0],
    "has_correction":        bool(g["has_correction"].iloc[0]),
    "final_verdict":         (g["corrected_verdict"].iloc[0]
                              if pd.notna(g["corrected_verdict"].iloc[0])
                              else majority(g.loc[g["rater_type"] == "manager", "verdict"])),
})).reset_index()


# ---------------------------------------------------------------------------
# write workbook
# ---------------------------------------------------------------------------

print(f"Writing {OUT_PATH} (total rows: {len(all_df)}) ...")
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    all_df.to_excel(writer, sheet_name="long", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print("Done.")
print("Row counts per rater_type:")
print(all_df["rater_type"].value_counts().to_dict())
print("Queries covered:", all_df["query_id"].nunique())
