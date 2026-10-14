"""
The data figures of Sections 4 and 5 of the report, Figs. 17, 18, 19 and 22, redrawn from
the runs in results/.

    python fig_section4.py
"""
import collections
import csv
import os
import statistics
import sys

import numpy as np

from figstyle import (BLUE, COL, GREY, INK, ORANGE, PAGE, PURPLE, axes, dot, figure, hgrid,
                      letter, mt, save, SMALL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "section5_discussion"))
import dnc_vs_fallback as DC                    # noqa: E402

RES = os.path.join(ROOT, "results", "section4_fallback")
C_ROW = 38.0                                    # the row height c at s = 50 (Section 4.2)


def stacked():
    """Figs. 17 and 18: 70 mm wide, panel a over panel b."""
    fig = figure(70, 116)
    a = axes(fig, 15, 4, 52, 42)
    b = axes(fig, 15, 62, 52, 42)
    letter(fig, "a", 1, 2)
    letter(fig, "b", 1, 60)
    return fig, a, b


def nfmt(ax, minor=True):
    """1k, 10k, 100k, 1M on a log axis."""
    ax.set_xscale("log")
    names = {1e2: "100", 1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M"}
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _p: names.get(v, "")))
    if not minor:
        ax.xaxis.set_minor_locator(mt.NullLocator())


def size():
    """Fig. 17: the side L and the area of the fallback embedding, random cubic, s = 50."""
    pts = {int(r["n"]): (int(r["T"]), int(r["L"]))
           for r in csv.DictReader(open(os.path.join(RES, "embedding_sizes.csv"), newline=""))
           if r["family"] == "random_cubic"}
    n = np.array(sorted(pts), float)
    T = np.array([pts[int(x)][0] for x in n], float)
    L = np.array([pts[int(x)][1] for x in n], float)
    Lf = T + C_ROW * np.sqrt(n / 50)
    fig, a, b = stacked()

    nfmt(a)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(mt.FuncFormatter(
        lambda v, _p: {1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M"}.get(v, "")))
    a.plot(n, n - 1, ls="--", color=INK, lw=1.0, label="complete graph embedding, $n-1$")
    a.plot(n, Lf, color=BLUE, lw=1.4, alpha=0.5, label=r"$T + c\sqrt{n/s}$")
    a.plot(n, L, "o", color=BLUE, mec="white", mew=0.5, label="measured $L$")
    a.legend(loc="upper left", handlelength=1.8, borderaxespad=0.3)
    a.set_ylim(500, 5e6)
    a.set_ylabel("lattice side $L$")
    a.set_xlabel("vertices $n$")

    nfmt(b)
    hgrid(b)
    area = (L / (n - 1)) ** 2
    b.plot(n, (Lf / (n - 1)) ** 2, color=BLUE, lw=1.4, alpha=0.5)
    b.plot(n, area, "o", color=BLUE, mec="white", mew=0.5)
    b.text(n[-1], area[-1] - 0.015, f"{1 / area[-1]:.2f}" + r"$\times$ less at 1M", ha="right", va="top",
           fontsize=SMALL, color=INK)
    b.set_ylim(0.32, 0.56)
    b.set_ylabel("area against the complete\ngraph embedding")
    b.set_xlabel("vertices $n$")
    for ax in (a, b):
        ax.set_xlim(700, 1.5e6)
        ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    save(fig, "fig17_fallback_size")


def discovery():
    """Fig. 18: minimizing the cut against minimizing T, with the same partitioner."""
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(RES, "cut_vs_border_nodes.csv"), newline="")):
        d[(r["objective"], int(r["n"]))].append((float(r["T_over_n"]), float(r["cut_over_n"])))
    ns = sorted({k[1] for k in d})
    fig, a, b = stacked()
    for ax, j, ylab in ((a, 0, "border nodes $T/n$"), (b, 1, "cut edges per vertex")):
        nfmt(ax)
        hgrid(ax)
        for obj, col in (("cut", ORANGE), ("T", BLUE)):
            dot(ax, ns, [np.mean([v[j] for v in d[(obj, x)]]) for x in ns], col, lw=1.4)
        ax.set_xlim(800, 1.25e5)
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.set_xlabel("vertices $n$")
        ax.set_ylabel(ylab)
    last = lambda obj, j: np.mean([v[j] for v in d[(obj, ns[-1])]])
    a.set_ylim(0.5, 0.85)
    b.set_ylim(0.35, 0.65)
    a.text(ns[-1], last("cut", 0) + 0.015, "minimizing the cut", color=ORANGE, fontsize=SMALL, ha="right", va="bottom")
    a.text(ns[-1], last("T", 0) - 0.02, "minimizing $T$", color=BLUE, fontsize=SMALL, ha="right", va="top")
    b.text(ns[-1], last("T", 1) + 0.012, "minimizing $T$", color=BLUE, fontsize=SMALL, ha="right", va="bottom")
    b.text(ns[-1], last("cut", 1) - 0.03, "minimizing the cut", color=ORANGE, fontsize=SMALL, ha="right", va="top")
    save(fig, "fig18_discovery")


def plateau():
    """Fig. 19: T/n of random cubic graphs at s = 50, the median of five graphs, bars their range."""
    best = {}
    for r in csv.DictReader(open(os.path.join(RES, "t_over_n_plateau.csv"), newline="")):
        k = (int(r["n"]), int(r["seed"]))
        best[k] = min(float(r["T_over_n"]), best.get(k, 9.0))
    by = collections.defaultdict(list)
    for (n, _), v in best.items():
        by[n].append(v)
    ns = np.array(sorted(by), float)
    med = np.array([np.median(by[n]) for n in sorted(by)])
    lo = med - [min(by[n]) for n in sorted(by)]
    hi = [max(by[n]) for n in sorted(by)] - med
    level = med[ns >= 1e4].mean()                 # the plateau, from 10,000 vertices on
    fig = figure(COL, 56)
    ax = axes(fig, 13, 3, 57, 41)
    nfmt(ax)
    ax.axhline(level, color=GREY, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(820, level + 0.003, "plateau, about 0.6", color=GREY, fontsize=SMALL, ha="left", va="bottom")
    ax.errorbar(ns, med, yerr=[lo, hi], fmt="none", ecolor=BLUE, elinewidth=0.8, capsize=1.6,
                capthick=0.8, alpha=0.6, zorder=2)
    dot(ax, ns, med, BLUE, lw=1.4, zorder=3)
    ax.set_xlim(700, 1.5e6)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax.set_ylim(0.52, 0.62)
    ax.set_xlabel("vertices $n$")
    ax.set_ylabel("border nodes $T/n$")
    # the area the fallback embedding tends to, (T/n)^2 (Section 4.4)
    floor = ax.secondary_yaxis("right", functions=(lambda x: x ** 2, lambda y: np.sqrt(np.clip(y, 0, None))))
    floor.set_yticks([0.28, 0.30, 0.32, 0.34, 0.36, 0.38])
    floor.set_ylabel(r"area floor $(T/n)^2$")
    save(fig, "fig19_plateau")


def big_picture():
    """Fig. 22: the four methods on random cubic graphs, area against the complete graph embedding."""
    fb = collections.defaultdict(list)            # the fallback, s = 30 to 2,000 (ten graphs each)
    for (n, _), e in DC.fallback_entries("fallback_s30.csv", 30).items():
        fb[n].append(e["L"])
    fb_r = {n: v for n, v in fb.items() if len(v) == 10 and n <= 2000}
    fb_L = {n: statistics.median(v) for n, v in fb_r.items()}
    for r in csv.DictReader(open(os.path.join(RES, "embedding_sizes.csv"))):   # and s = 50 from 3,000
        if r["family"] == "random_cubic" and int(r["n"]) >= 3000:
            fb_L[int(r["n"])] = int(r["L"])
    dnc = DC.dnc_entries()
    dnc_r = collections.defaultdict(list)         # Divide and Conquer, the graphs it embedded
    for (n, _), e in dnc.items():
        if e["success"] and n in range(100, 1001, 100):
            dnc_r[n].append(e["L"])
    dnc_L = {n: statistics.median(v) for n, v in dnc_r.items()}
    for n in (1100, 1200, 1400):                  # one graph each beyond 1,000
        dnc_L[n] = dnc[(n, 1)]["L"]
    sugie_L = {int(r["threshold_V"]): int(r["L"]) for r in csv.DictReader(open(
        os.path.join(ROOT, "results", "section5_discussion", "sugie2021_fig6a_improved_pssa_random_cubic.csv")))}
    f = lambda n, L: DC.area(L, n)
    fig = figure(PAGE, 70)
    ax = axes(fig, 20, 3, 156, 56)
    hgrid(ax)

    def line(d, c, lw=1.2):
        ns = sorted(d)
        dot(ax, ns, [f(n, d[n]) for n in ns], c, lw=lw, ms=2.6, zorder=4)

    def bars(d, c):
        ns = sorted(d)
        ax.vlines(ns, [min(f(n, L) for L in d[n]) for n in ns], [max(f(n, L) for L in d[n]) for n in ns],
                  color=c, lw=0.7, alpha=0.45, zorder=2)

    ax.plot([60, 1.2e6], [1, 1], color=INK, lw=1.2, zorder=3)
    line(sugie_L, PURPLE)
    line(dnc_L, ORANGE)
    line(fb_L, BLUE, 1.4)
    bars(dnc_r, ORANGE)
    bars(fb_r, BLUE)
    for x, lab, ha in ((2000, "$s = 30$", "right"), (3000, "$s = 50$", "left")):
        ax.plot([x, x], [f(x, fb_L[x]) + 0.02, 0.68], ls=(0, (1, 1.6)), color=GREY, lw=0.6, zorder=1)
        ax.text(x / 1.07 if ha == "right" else x * 1.07, 0.68, lab, color=GREY, ha=ha, va="top", fontsize=SMALL)
    xs, xd = max(sugie_L), max(dnc_L)
    ax.text(1.2e6, 0.965, "best known complete graph embedding", color=INK, ha="right", va="top", fontsize=SMALL)
    ax.text(1e6, f(1e6, fb_L[int(1e6)]) + 0.03, "fallback method", color=BLUE, ha="right", va="bottom", fontsize=SMALL)
    ax.text(xs * 1.13, f(xs, sugie_L[xs]) - 0.03, "Sugie's PSSA, to 1,000", color=PURPLE, ha="left", va="center",
            fontsize=SMALL)
    ax.text(xd * 1.13, f(xd, dnc_L[xd]) + 0.04, "Divide and Conquer,\nnone beyond 1,400", color=ORANGE,
            ha="left", va="center", fontsize=SMALL)
    nfmt(ax)
    ax.set_xlim(55, 1.6e6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("vertices $n$")
    ax.set_ylabel("area against the best known\ncomplete graph embedding")
    save(fig, "fig22_big_picture")


if __name__ == "__main__":
    size()
    discovery()
    plateau()
    big_picture()
