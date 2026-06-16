"""Графики результатов staged-эксперимента для главы 5.

Генерирует (в figures/):
  results-sbs-stages.jpeg   -- SbS-NPS схем + CI, линия асессоров;
  results-confusion.jpeg    -- 3x3 матрица: вердикт LLM (consensus) vs асессоры;
  results-nps-density.jpeg  -- bootstrap-плотности SbS-NPS: асессоры vs consensus;
  results-pareto.jpeg       -- согласие с асессорами против относительной стоимости.

Запускать после analyze_staged.py (использует votes.jsonl и кэш).
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_staged as A

FIG = Path(__file__).resolve().parents[2] / "figures"
FIG.mkdir(exist_ok=True)
COST = {"classic": 1, "multiprompt": 3, "cons3": 9, "cons4": 12}
SHORT = {"classic": "classic\n(1x)", "multiprompt": "multi-prompt\n(3x)",
         "cons3": "consensus l=3\n(9x)", "cons4": "consensus l=4\n(12x)"}


def boot_dist(verd, qids, B=10000):
    rng = random.Random(7)
    n = len(qids)
    out = []
    for _ in range(B):
        samp = [qids[rng.randrange(n)] for _ in range(n)]
        out.append(A.sbs_nps(verd, samp))
    return np.array(out)


def main():
    votes, sample = A.load()
    qids = [q for q in votes if q in sample]
    av = {q: A.assessor_verdict(sample[q]) for q in qids}
    schemes = {k: A.scheme_verdicts(votes, k)
               for k in ["classic", "multiprompt", "cons3", "cons4"]}

    # 1. SbS-NPS схем + CI + линия асессоров
    sbs_ass = A.sbs_nps(av, qids)
    keys = ["classic", "multiprompt", "cons3", "cons4"]
    pts, los, his = [], [], []
    for k in keys:
        d = boot_dist(schemes[k], qids)
        pts.append(d.mean()); los.append(np.percentile(d, 2.5)); his.append(np.percentile(d, 97.5))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(keys))
    err = [np.array(pts) - np.array(los), np.array(his) - np.array(pts)]
    ax.bar(x, pts, yerr=err, capsize=5, color="#1d9e75", alpha=0.85)
    ax.axhline(sbs_ass, ls="--", color="#444", label=f"асессоры: {sbs_ass:+.1f}%")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in keys])
    ax.set_ylabel("SbS-NPS (S vs C), %")
    ax.set_title("SbS-NPS по схемам с 95% bootstrap-CI (N={})".format(len(qids)))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "results-sbs-stages.jpeg", dpi=130); plt.close(fig)

    # 2. 3x3 confusion (cons4 vs assessors), row-normalized
    labels = ["S", "draw", "C"]
    M = np.zeros((3, 3))
    cj = schemes["cons4"]
    for q in qids:
        jv = cj[q]
        if jv is None:
            continue
        M[labels.index(jv)][labels.index(av[q])] += 1
    Mn = M / M.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(Mn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("Вердикт асессоров"); ax.set_ylabel("Вердикт consensus-судьи")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                    color="white" if Mn[i, j] > 0.5 else "black")
    ax.set_title("Согласованность вердиктов (row-normalized)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(FIG / "results-confusion.jpeg", dpi=130); plt.close(fig)

    # 3. bootstrap-плотности SbS-NPS: асессоры (control) vs consensus (test)
    d_ass = boot_dist(av, qids); d_con = boot_dist(schemes["cons4"], qids)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(min(d_ass.min(), d_con.min()), max(d_ass.max(), d_con.max()), 50)
    ax.hist(d_ass, bins=bins, density=True, alpha=0.5, color="#888",
            label=f"асессоры (control), {d_ass.mean():+.1f}%")
    ax.hist(d_con, bins=bins, density=True, alpha=0.5, color="#1d9e75",
            label=f"consensus (test), {d_con.mean():+.1f}%")
    ax.axvline(d_ass.mean(), ls="--", color="#444")
    ax.axvline(d_con.mean(), ls="--", color="#0f6e56")
    ax.set_xlabel("SbS-NPS, %"); ax.set_ylabel("плотность (bootstrap)")
    ax.set_title("Сдвиг SbS-NPS: асессоры против consensus-судьи")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "results-nps-density.jpeg", dpi=130); plt.close(fig)

    # 4. Pareto: согласие с асессорами против относительной стоимости
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k in keys:
        ag, _ = A.agreement(schemes[k], av, qids)
        ax.scatter(COST[k], ag * 100, s=80, color="#185fa5")
        ax.annotate(SHORT[k].replace("\n", " "), (COST[k], ag * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Относительная стоимость (число вызовов на вердикт)")
    ax.set_ylabel("Согласие с асессорами, %")
    ax.set_title("Качество против стоимости")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "results-pareto.jpeg", dpi=130); plt.close(fig)

    print("Charts saved to", FIG)


if __name__ == "__main__":
    main()
