"""
Bootstrap statistical-significance of v4 SBS NPS for two hypotheses, per slice.

Metric under test = **SBS NPS** (the dashboard metric), evaluated two ways:

  H1 — bias vs GPT (threshold NPS):
       NPS_thr = (#queries Yandex wins - #queries GPT wins) / N
       where a "win" needs > 7 votes for that model (and more than the other).
       Since the NPS is itself the YA-minus-GPT contrast, "significant bias vs GPT"
       == 95% bootstrap CI of NPS_thr does not contain 0.

  H2 — bias vs 0 (continuous NPS):
       NPS_cont = mean over queries of (ydx_win_cnt - gpt_win_cnt) / total_votes.
       "significant vs 0" == 95% bootstrap CI of NPS_cont does not contain 0.

Resampling unit = query (per-query rows already deduplicated by worker).
Slices (each value plus a `_total_`):
  - dimension = agent  (scenario_tag)
  - dimension = dialog (is_dialog)
  - filter    = all | correct_stream  (correct_stream = no broken_links / time_window /
                media_broken / gpt_refused — same as dataset "Filter Correct Stream")

Output: one row per (filter, dimension, value, hypothesis) with point estimate, CI,
p-value (share of bootstrap replicas on the opposite side of 0, two-sided), n_queries,
and a boolean `significant`. Written to YT for a DataLens dataset.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

SRC = "baskets/v4/sbs_v4_basket_rows.jsonl"
OUT_JSONL = "baskets/v4/v4_bootstrap_significance.jsonl"
N_BOOT = 5000
WIN_THRESHOLD = 7          # > 7 votes => win (i.e. >= 8)
ALPHA = 0.05
SEED = 20260605

CRITICAL_CATS = {"broken_links", "time_window_mismatch", "media_broken", "gpt_refused"}


def is_correct_stream(improve: str | None) -> bool:
    if not improve:
        return True
    cats = {c.strip() for c in improve.split(",")}
    return not (cats & CRITICAL_CATS)


def load_rows():
    rows = []
    for line in open(SRC):
        r = json.loads(line)
        y = r.get("ydx_win_cnt") or 0
        d = r.get("draw_cnt") or 0
        g = r.get("gpt_win_cnt") or 0
        tot = y + d + g
        rows.append({
            "agent": r.get("scenario_tag") or "unknown",
            "dialog": "dialog" if r.get("is_dialog") else "non_dialog",
            "correct": is_correct_stream(r.get("improve_categories")),
            "ydx": y, "draw": d, "gpt": g, "tot": tot,
            # per-query threshold outcome: +1 YA win, -1 GPT win, 0 draw
            "thr": (1 if (y > WIN_THRESHOLD and y > g)
                    else (-1 if (g > WIN_THRESHOLD and g > y) else 0)),
            # per-query continuous contribution (NaN if no votes)
            "cont": ((y - g) / tot) if tot > 0 else np.nan,
        })
    return rows


def bootstrap_metric(values: np.ndarray, rng) -> tuple[float, float, float, float]:
    """Return (point, lo, hi, p_two_sided_vs_0) for mean(values) via bootstrap."""
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    point = float(values.mean())
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = values[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * ALPHA / 2))
    hi = float(np.percentile(boot, 100 * (1 - ALPHA / 2)))
    # two-sided p vs 0: 2 * min(share<=0, share>=0)
    share_le = float((boot <= 0).mean())
    share_ge = float((boot >= 0).mean())
    p = min(1.0, 2 * min(share_le, share_ge))
    return (point, lo, hi, p)


def emit(rng, out, filt_name, dim_name, value, subset):
    thr = np.array([r["thr"] for r in subset], dtype=float)        # NPS_thr units: per-query in {-1,0,1}; mean == NPS
    cont = np.array([r["cont"] for r in subset], dtype=float)
    n = len(subset)
    if n == 0:
        return
    # H1 vs GPT  (threshold NPS; the metric itself is the YA-GPT contrast)
    p1, lo1, hi1, pv1 = bootstrap_metric(thr, rng)
    out.append({
        "filter": filt_name, "dimension": dim_name, "value": value,
        "hypothesis": "H1_vs_gpt", "metric": "nps_threshold",
        "n_queries": n,
        "point": round(100 * p1, 4),
        "ci_low": round(100 * lo1, 4), "ci_high": round(100 * hi1, 4),
        "p_value": round(pv1, 5),
        "significant": bool(not (lo1 <= 0 <= hi1)),
    })
    # H2 vs 0  (continuous NPS)
    p2, lo2, hi2, pv2 = bootstrap_metric(cont, rng)
    out.append({
        "filter": filt_name, "dimension": dim_name, "value": value,
        "hypothesis": "H2_vs_zero", "metric": "nps_continuous",
        "n_queries": int((~np.isnan(cont)).sum()),
        "point": round(100 * p2, 4),
        "ci_low": round(100 * lo2, 4), "ci_high": round(100 * hi2, 4),
        "p_value": round(pv2, 5),
        "significant": bool(not (lo2 <= 0 <= hi2)),
    })


def main():
    rng = np.random.default_rng(SEED)
    rows = load_rows()
    out = []

    for filt_name, filt in [("all", lambda r: True),
                            ("correct_stream", lambda r: r["correct"])]:
        base = [r for r in rows if filt(r)]

        # dimension = _overall_ (total)
        emit(rng, out, filt_name, "overall", "_total_", base)

        # dimension = agent
        by_agent = defaultdict(list)
        for r in base:
            by_agent[r["agent"]].append(r)
        for ag, sub in by_agent.items():
            emit(rng, out, filt_name, "agent", ag, sub)

        # dimension = dialog
        by_dialog = defaultdict(list)
        for r in base:
            by_dialog[r["dialog"]].append(r)
        for dv, sub in by_dialog.items():
            emit(rng, out, filt_name, "dialog", dv, sub)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summary
    print(f"wrote {len(out)} rows -> {OUT_JSONL}")
    for rec in out:
        if rec["dimension"] == "overall":
            print(f"  [{rec['filter']:<14}] {rec['hypothesis']:<10} "
                  f"NPS={rec['point']:+.2f}%  CI[{rec['ci_low']:+.2f},{rec['ci_high']:+.2f}]  "
                  f"p={rec['p_value']:.4f}  sig={rec['significant']}  n={rec['n_queries']}")


if __name__ == "__main__":
    main()
