"""
The data figures of Section 2 of the report, Figs. 4 and 9-12, redrawn from the runs in
results/section2_pssa.

    python fig_section2.py
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

from figstyle import (BLUE, COL, GREY, INK, ORANGE, PAGE, SEQ, axes, dot, figure,
                      heatmap_frame, hgrid, ink_on, percent, plt, save, SMALL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "section2_pssa")


def gibbs():
    """Fig. 4: the embedding threshold with uniform and Gibbs-weighted moves."""
    s = json.load(open(os.path.join(RES, "embedding_threshold_summary.json")))
    Ls = sorted(int(k) for k in s)
    g = lambda m: [s[str(L)][m] for L in Ls]
    fig = figure(COL, 55)
    ax = axes(fig, 12, 3, 55, 41)
    hgrid(ax)
    dot(ax, Ls, g("uniform"), GREY)
    dot(ax, Ls, g("uniform_time"), GREY, ls="--", hollow=True)
    dot(ax, Ls, g("gibbs"), BLUE, lw=1.4)
    kw = dict(va="center", fontsize=SMALL, clip_on=False)
    ax.text(12.35, g("gibbs")[-1], "Gibbs", color=BLUE, **kw)
    ax.text(12.35, g("uniform_time")[-1] + 0.8, "uniform,\nmatched time", color=GREY, **kw)
    ax.text(12.35, g("uniform")[-1] - 1.0, "uniform", color=GREY, **kw)
    ax.set_xticks(Ls)
    ax.set_xlim(5.7, 12.3)
    ax.set_ylim(16, 30)
    ax.set_xlabel("hardware size $L$")
    ax.set_ylabel(r"embedding threshold $\bar V(L)$")
    save(fig, "fig04_gibbs_threshold")


def density():
    """Fig. 9: the smallest L that embeds a 20-vertex Erdős–Rényi graph, by its edges."""
    rows = list(csv.DictReader(open(os.path.join(RES, "density_limit_summary.csv"))))
    E = [int(r["edges"]) for r in rows]
    Lmin = [int(r["min_L"]) for r in rows]
    fig = figure(COL, 52)
    ax = axes(fig, 12, 3, 72, 38)
    hgrid(ax)
    ax.axhline(19, ls="--", lw=1.0, color=INK)
    ax.text(30, 18.4, "complete graph embedding of $K_{20}$ ($L = 19$)", va="top", ha="left",
            fontsize=SMALL, color=INK)
    dot(ax, E, Lmin, BLUE)
    ax.set_xlim(29, 51)
    ax.set_ylim(4, 20.5)
    ax.set_xticks(range(30, 51, 4))
    ax.set_yticks([5, 10, 15, 20])
    ax.set_xlabel("edges $|E(I)|$ of a 20-vertex Erdős–Rényi graph")
    ax.set_ylabel("smallest $L$ that embeds it")
    save(fig, "fig09_density_limit")


def border_fraction():
    """Fig. 10: the success rate of Mode 1 and Mode 2 by border nodes and L."""
    LS, KS = [5, 6, 7, 8, 9], [0, 2, 5, 10, 15]
    rows = [(int(r["L"]), int(r["border_nodes"]), r["mode"], int(r["success"]))
            for r in csv.DictReader(open(os.path.join(RES, "number_of_border_nodes.csv")))]
    fig = figure(PAGE, 50)
    w = 72
    panels = [axes(fig, 12, 5, w, 33), axes(fig, 12 + w + 6, 5, w, 33)]
    cax = axes(fig, 12 + 2 * w + 6 + 4, 5, 2.5, 33)
    for ax, mode, name in zip(panels, ("fixed", "permute"),
                              ("Mode 1, fixed border nodes", "Mode 2, free border nodes")):
        M = np.array([[np.mean([r[3] for r in rows if r[0] == L and r[1] == k
                                and r[2] == ("none" if k == 0 else mode)]) for k in KS] for L in LS])
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(LS)):
            for j in range(len(KS)):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=SMALL,
                        color=ink_on(SEQ, M[i, j]))
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels([f"{k} ({k / 20:.0%})" for k in KS])
        ax.set_xlabel("border nodes")
        ax.set_title(name, loc="left")
        heatmap_frame(ax)
    panels[0].set_yticks(range(len(LS)))
    panels[0].set_yticklabels(LS)
    panels[0].set_ylabel("hardware size $L$")
    panels[1].set_yticks([])
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    percent(cb.ax.yaxis)
    cb.set_label("success rate (20 seeds)")
    save(fig, "fig10_border_fraction")


def degree():
    """Fig. 11: the success rate of Mode 2 with the 5 border nodes chosen by degree."""
    ok, n = defaultdict(int), defaultdict(int)
    for r in csv.DictReader(open(os.path.join(RES, "border_node_degree_many_graphs.csv"))):
        key = (int(r["L"]), r["border_choice"])
        ok[key] += int(r["success"])
        n[key] += 1
    Ls = sorted({L for L, _ in n})
    fig = figure(COL, 55)
    ax = axes(fig, 13, 3, 72, 41)
    for k, (ch, c, name) in enumerate((("high", ORANGE, "highest degree"), ("random", GREY, "random"),
                                       ("low", BLUE, "lowest degree"))):
        ax.bar([L + (k - 1) * 0.27 for L in Ls], [ok[(L, ch)] / n[(L, ch)] for L in Ls],
               width=0.26, color=c, label=name, zorder=3)
    ax.set_xticks(Ls)
    ax.set_ylim(0, 1.25)                        # room for the legend above the bars
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1)
    percent(ax.yaxis)
    ax.set_xlabel("hardware size $L$")
    ax.set_ylabel("Mode 2 success rate")
    ax.legend(title="5 border nodes, the", loc="upper left", labelspacing=0.25, alignment="left")
    save(fig, "fig11_border_degree")


def cluster_embedding_limit():
    """Fig. 12: how many of 5 clusters embed, by cluster size and mean degree."""
    SIZES, DEGREES = [10, 15, 20, 25, 30, 40], [2.5, 3, 4, 6]
    rows = [(int(r["size"]), float(r["mean_degree"]), float(r["border_frac"]),
             int(r["smallest_L"]), int(r["success"]))
            for r in csv.DictReader(open(os.path.join(RES, "cluster_embedding_limit.csv")))]
    panels = [(0.0, "Unconstrained PSSA")] + [(f, f"PSSA wBC, {f:.0%} border nodes") for f in (0.3, 0.5, 0.7)]
    fig = figure(PAGE, 92)
    w, h = 72, 33                                   # Fig. 10's heatmap size
    for p, (frac, name) in enumerate(panels):
        r, c = divmod(p, 2)
        ax = axes(fig, 12 + c * (w + 6), 5 + r * (h + 13), w, h)
        rate = np.zeros((len(DEGREES), len(SIZES)))
        medL = np.full((len(DEGREES), len(SIZES)), np.nan)
        for i, d in enumerate(DEGREES):
            for j, s in enumerate(SIZES):
                got = [x for x in rows if x[0] == s and x[1] == d and x[2] == frac]
                rate[i, j] = np.mean([x[4] for x in got])
                Ls = [x[3] for x in got if x[4]]
                if Ls:
                    medL[i, j] = np.median(Ls)
        im = ax.imshow(rate, cmap=SEQ, vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(DEGREES)):
            for j in range(len(SIZES)):
                txt = f"{rate[i, j]:.0%}"
                if not np.isnan(medL[i, j]):
                    txt += f"\n$L = {medL[i, j]:g}$"
                ax.text(j, i, txt, ha="center", va="center", fontsize=SMALL, linespacing=1.15,
                        color=ink_on(SEQ, rate[i, j]))
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES)
        ax.set_title(name, loc="left")
        heatmap_frame(ax)
        if r == 1:
            ax.set_xlabel("cluster size $|V(I)|$")
        if c == 0:
            ax.set_yticks(range(len(DEGREES)))
            ax.set_yticklabels([f"{d:g}" for d in DEGREES])
            ax.set_ylabel("mean degree")
        else:
            ax.set_yticks([])
    cax = axes(fig, 12 + 2 * w + 6 + 4, 5, 2.5, 2 * h + 13)
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    percent(cb.ax.yaxis)
    cb.set_label("clusters embedded (of 5)")
    save(fig, "fig12_cluster_embedding_limit")


if __name__ == "__main__":
    gibbs()
    density()
    border_fraction()
    degree()
    cluster_embedding_limit()
