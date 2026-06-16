"""
Fix the non-deduplicated v4 point in //home/bali/yandex-ai-chat/sbs-auto/baskets_Dashboard.

The last point (basket='v', date='2026-06-03') was loaded WITHOUT deduplication:
6164 rows for 1532 distinct queries (each query ~4x, nine queries 8x), which skews the
"SbS like NPS" time series on the "Yandex VS Gpt" tab.

Fix: collapse that partition to exactly one row per query.
- winner = majority vote across the duplicate rows (verified to match the authoritative
  >7-votes winner in 100% of cross-checked queries).
- all other fields taken from the first occurrence (duplicates are otherwise identical).
All other partitions are passed through unchanged.

Writes the cleaned table back (full rewrite) after backing up.
"""
import json
import os
from collections import defaultdict, Counter

import yt.wrapper as yt

CLUSTER = "hahn"
T = "//home/bali/yandex-ai-chat/sbs-auto/baskets_Dashboard"
BAK = T + "_bak_20260605"
FIX_BASKET = "v"
FIX_DATE = "2026-06-03"


def main():
    yt.config["proxy"]["url"] = CLUSTER
    yt.config["token"] = open(os.path.expanduser("~/.yt/token")).read().strip()

    schema = yt.get(T + "/@schema")

    passthrough = []
    fix_groups = defaultdict(list)  # query -> list of rows
    total = 0
    for r in yt.read_table(T):
        total += 1
        if r.get("basket") == FIX_BASKET and r.get("date") == FIX_DATE:
            fix_groups[r.get("query")].append(r)
        else:
            passthrough.append(r)

    deduped = []
    for q, rows in fix_groups.items():
        winners = Counter(r.get("yandex_vs_gpt_winner") for r in rows)
        majority = winners.most_common(1)[0][0]
        base = dict(rows[0])
        base["yandex_vs_gpt_winner"] = majority
        deduped.append(base)

    out = passthrough + deduped
    print(f"read {total} rows; passthrough {len(passthrough)}; "
          f"v4 {sum(len(v) for v in fix_groups.values())} -> {len(deduped)} deduped queries")

    w = Counter(r["yandex_vs_gpt_winner"] for r in deduped)
    tot = sum(w.values())
    nps = 100 * (w.get("yandex ai", 0) - w.get("gpt", 0)) / tot
    print(f"v4 deduped winners: {dict(w)}  NPS={nps:.3f}%")

    # backup then rewrite preserving schema
    if not yt.exists(BAK):
        yt.copy(T, BAK)
        print(f"backup -> {BAK}")
    else:
        print(f"backup already exists -> {BAK}")

    yt.write_table(
        yt.TablePath(T, schema=schema),
        out,
        format=yt.JsonFormat(attributes={"encode_utf8": False}),
    )
    print(f"rewrote {T}: {len(out)} rows total")


if __name__ == "__main__":
    main()
