"""
Universal NPS significance tool — bootstrap over ALL baskets.

Reads the canonical per-query table //home/bali/yandex-ai-chat/sbs_stream_1k_classified
(any basket appended there is picked up automatically) and produces TWO YT tables that
feed a single DataLens "NPS significance" tab:

1) sbs_nps_significance_summary  — one row per
   (basket, filter, dimension, value, hypothesis): point estimate, 95% CI, p-value,
   n_queries, significant.

2) sbs_nps_significance_dist      — bootstrap distribution (bell curve) per
   (basket, filter, hypothesis): histogram bins (bin_center, density) of the bootstrap
   NPS replicas, for the overall slice. Lets us overlay basket distributions and see
   where each one sits relative to 0.

Metric = SBS NPS, two hypotheses:
  H1_vs_gpt   (nps_threshold)  : NPS = (#YA wins - #GPT wins)/N, win = > 7 votes.
  H2_vs_zero  (nps_continuous) : mean over queries of (ydx_win_cnt - gpt_win_cnt)/total.

Resampling unit = query. Significant = 95% bootstrap CI excludes 0.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, Counter

import numpy as np
import yt.wrapper as yt

SRC = "//home/bali/yandex-ai-chat/sbs_stream_1k_classified"
DST_SUMMARY = "//home/bali/yandex-ai-chat/sbs_nps_significance_summary"
DST_DIST = "//home/bali/yandex-ai-chat/sbs_nps_significance_dist"
DST_MIX = "//home/bali/yandex-ai-chat/sbs_nps_outcome_mix"

N_BOOT = 5000
WIN_THRESHOLD = 7
ALPHA = 0.05
SEED = 20260605
DIST_BINS = 60
CRITICAL_CATS = {"broken_links", "time_window_mismatch", "media_broken", "gpt_refused"}


def is_correct_stream(improve):
    if not improve:
        return True
    cats = {c.strip() for c in str(improve).split(",")}
    return not (cats & CRITICAL_CATS)


def boot_means(values, rng):
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return None, None
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return values, values[idx].mean(axis=1)


def stat_from_boot(values, boot):
    point = float(values.mean())
    lo = float(np.percentile(boot, 100 * ALPHA / 2))
    hi = float(np.percentile(boot, 100 * (1 - ALPHA / 2)))
    p = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
    return point, lo, hi, p


def main():
    yt.config["proxy"]["url"] = "hahn"
    yt.config["token"] = open(os.path.expanduser("~/.yt/token")).read().strip()
    rng = np.random.default_rng(SEED)

    # load per-query rows grouped by basket
    by_basket = defaultdict(list)
    for r in yt.read_table(SRC):
        y = r.get("ydx_win_cnt") or 0
        d = r.get("draw_cnt") or 0
        g = r.get("gpt_win_cnt") or 0
        tot = y + d + g
        by_basket[r.get("basket") or "unknown"].append({
            "agent": r.get("scenario_tag") or "unknown",
            "dialog": "dialog" if r.get("is_dialog") else "non_dialog",
            "correct": is_correct_stream(r.get("improve_categories")),
            "thr": (1.0 if (y > WIN_THRESHOLD and y > g)
                    else (-1.0 if (g > WIN_THRESHOLD and g > y) else 0.0)),
            "cont": ((y - g) / tot) if tot > 0 else np.nan,
        })
    print("baskets:", {b: len(v) for b, v in by_basket.items()})

    summary = []
    dist = []
    mix = []

    HYP = [("H1_vs_gpt", "nps_threshold", "thr"),
           ("H2_vs_zero", "nps_continuous", "cont")]

    for basket, rows in by_basket.items():
        for filt_name, filt in [("all", lambda r: True),
                                ("correct_stream", lambda r: r["correct"])]:
            base = [r for r in rows if filt(r)]
            if not base:
                continue

            # ----- slices -----
            slices = [("overall", "_total_", base)]
            by_agent = defaultdict(list)
            for r in base:
                by_agent[r["agent"]].append(r)
            slices += [("agent", a, s) for a, s in by_agent.items()]
            by_dialog = defaultdict(list)
            for r in base:
                by_dialog[r["dialog"]].append(r)
            slices += [("dialog", dv, s) for dv, s in by_dialog.items()]

            for dim, value, subset in slices:
                for hyp, metric, key in HYP:
                    vals = np.array([r[key] for r in subset], dtype=float)
                    values, boot = boot_means(vals, rng)
                    if boot is None:
                        continue
                    point, lo, hi, p = stat_from_boot(values, boot)
                    summary.append({
                        "basket": basket, "filter": filt_name,
                        "dimension": dim, "value": value,
                        "agent": value if dim == "agent" else "_all_",
                        "hypothesis": hyp, "metric": metric,
                        "n_queries": int(len(values)),
                        "point": round(100 * point, 4),
                        "ci_low": round(100 * lo, 4),
                        "ci_high": round(100 * hi, 4),
                        "p_value": round(p, 5),
                        "significant": bool(not (lo <= 0 <= hi)),
                    })
                    # distribution only for the overall slice (bell curves)
                    if dim == "overall":
                        b100 = 100 * boot
                        counts, edges = np.histogram(b100, bins=DIST_BINS, density=True)
                        centers = (edges[:-1] + edges[1:]) / 2
                        for c, dns in zip(centers, counts):
                            dist.append({
                                "basket": basket, "filter": filt_name,
                                "hypothesis": hyp,
                                "bin_center": round(float(c), 4),
                                "density": round(float(dns), 6),
                            })

                # outcome mix (win/draw/loss share) — only need it once per slice,
                # computed on the threshold framing (>7 votes). Long format for a
                # 100%-stacked bar showing how the win/draw/loss distribution shifts.
                thr_vals = [r["thr"] for r in subset]
                n = len(thr_vals)
                wins = sum(1 for v in thr_vals if v > 0)
                draws = sum(1 for v in thr_vals if v == 0)
                losses = sum(1 for v in thr_vals if v < 0)
                for outcome, cnt in [("Yandex win", wins), ("draw", draws), ("GPT win", losses)]:
                    mix.append({
                        "basket": basket, "filter": filt_name,
                        "dimension": dim, "value": value,
                        "agent": value if dim == "agent" else "_all_",
                        "outcome": outcome,
                        "cnt": int(cnt),
                        "share": round(100 * cnt / n, 4) if n else 0.0,
                    })

    # write summary
    sum_schema = [
        {"name": "basket", "type": "string"},
        {"name": "filter", "type": "string"},
        {"name": "dimension", "type": "string"},
        {"name": "value", "type": "string"},
        {"name": "agent", "type": "string"},
        {"name": "hypothesis", "type": "string"},
        {"name": "metric", "type": "string"},
        {"name": "n_queries", "type": "int64"},
        {"name": "point", "type": "double"},
        {"name": "ci_low", "type": "double"},
        {"name": "ci_high", "type": "double"},
        {"name": "p_value", "type": "double"},
        {"name": "significant", "type": "boolean"},
    ]
    dist_schema = [
        {"name": "basket", "type": "string"},
        {"name": "filter", "type": "string"},
        {"name": "hypothesis", "type": "string"},
        {"name": "bin_center", "type": "double"},
        {"name": "density", "type": "double"},
    ]
    mix_schema = [
        {"name": "basket", "type": "string"},
        {"name": "filter", "type": "string"},
        {"name": "dimension", "type": "string"},
        {"name": "value", "type": "string"},
        {"name": "agent", "type": "string"},
        {"name": "outcome", "type": "string"},
        {"name": "cnt", "type": "int64"},
        {"name": "share", "type": "double"},
    ]
    for path, schema, data in [(DST_SUMMARY, sum_schema, summary),
                              (DST_DIST, dist_schema, dist),
                              (DST_MIX, mix_schema, mix)]:
        if yt.exists(path):
            yt.remove(path)
        yt.create("table", path, attributes={"schema": schema, "optimize_for": "scan"})
        yt.write_table(path, data, format=yt.JsonFormat(attributes={"encode_utf8": False}))
        print(f"wrote {len(data)} rows -> {path}")

    # local copies for inspection
    with open("baskets/v4/nps_significance_summary.jsonl", "w") as f:
        for r in summary:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nOverall significance per basket:")
    for r in summary:
        if r["dimension"] == "overall":
            print(f"  {r['basket']:<10} {r['filter']:<14} {r['hypothesis']:<10} "
                  f"NPS={r['point']:+.2f}% CI[{r['ci_low']:+.2f},{r['ci_high']:+.2f}] "
                  f"p={r['p_value']:.4f} sig={r['significant']} n={r['n_queries']}")


if __name__ == "__main__":
    main()
