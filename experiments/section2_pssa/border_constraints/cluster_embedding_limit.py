"""
Figure 12 of the report: how large a cluster PSSA wBC can embed. For each cluster
size s and mean degree d, 5 Erdos-Renyi graphs with a share of their vertices
(30, 50, 70 or 90%) chosen at random as border nodes, embedded as in the fallback
method, from the smallest L that holds the cluster up to L0 + 4 with 2 seeds each.
Every cluster is also run without the border constraint, which separates PSSA's own
limit from the cost of the border constraint.
"""
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np

from common import OUT, TEXTWIDTH, run, save

SIZES = [10, 15, 20, 25, 30, 40]
DEGREES = [2.5, 3, 4, 6]
CLUSTERS = 5
FRACS = [0.3, 0.5, 0.7, 0.9]          # shares of border nodes tried
SHOW = [0.3, 0.5, 0.7]  # constrained panels shown
SLACK = 4
SEEDS = (1, 2)
TMAX = 400_000


def make_cluster(s, d, seed, frac):
    rng = random.Random(seed)
    m = min(round(d * s / 2), s * (s - 1) // 2)
    G = nx.gnm_random_graph(s, m, seed=seed)
    G = nx.relabel_nodes(G, {v: v + 1 for v in G.nodes()})
    k = round(frac * s)
    border = set(rng.sample(sorted(G.nodes()), k))
    return G, border


def min_L(s, b):
    L = 3
    while L * L - 1 < s or 4 * L - 4 < b:
        L += 1
    return L


def task(args):
    s, d, c, frac = args
    mode = "none" if frac == 0 else "permute"
    I, border = make_cluster(s, d, 1000 * s + 10 * int(2 * d) + c, frac)
    L0 = min_L(s, len(border))
    for L in range(L0, L0 + SLACK + 1):
        for seed in SEEDS:
            if mode == "none":
                res, _ = run(I, L, None, None, seed, tmax=TMAX)
            else:
                res, _ = run(I, L, border, "permute", seed, tmax=TMAX)
            if res is not None and res.success:
                return (s, d, c, frac, I.number_of_edges(), len(border), L0, L, 1)
    return (s, d, c, frac, I.number_of_edges(), len(border), L0, -1, 0)


def load():
    path = os.path.join(OUT, "cluster_embedding_limit.csv")
    if not os.path.exists(path):
        return []
    rows = []
    for r in csv.DictReader(open(path)):
        if "border_frac" in r:
            frac = float(r["border_frac"])
        else:
            frac = 0.0 if r["mode"] == "none" else 0.3
        rows.append((int(r["size"]), float(r["mean_degree"]), int(r["cluster"]), frac,
                     int(r["edges"]), int(r["border_nodes"]), int(r["L0"]),
                     int(r["smallest_L"]), int(r["success"])))
    return rows


def main():
    old = load()
    done = {(r[0], r[1], r[2], r[3]) for r in old}
    jobs = [(s, d, c, f) for s in SIZES for d in DEGREES for c in range(CLUSTERS)
            for f in [0.0] + FRACS if (s, d, c, f) not in done]
    jobs.sort(key=lambda j: -j[0] * j[1])          # hardest first
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = old + list(ex.map(task, jobs))
    with open(os.path.join(OUT, "cluster_embedding_limit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "mean_degree", "cluster", "border_frac", "edges", "border_nodes",
                    "L0", "smallest_L", "success"])
        w.writerows(sorted(rows))
    summary(rows)
    plot(rows)


def summary(rows):
    for f in [0.0] + FRACS:
        print(f"--- border {f:.0%}")
        for d in DEGREES:
            line = []
            for s in SIZES:
                got = [r[8] for r in rows if r[0] == s and r[1] == d and r[3] == f]
                line.append(f"{sum(got)}/{len(got)}")
            print(f"  d={d:<4g}", "  ".join(line))


def plot(rows):
    import matplotlib.pyplot as plt
    panels = [(0.0, "Unconstrained PSSA")] + [
        (f, f"PSSA wBC, {f:.0%} border nodes") for f in SHOW]
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.22)
    for ax, (frac, name) in zip(axes.ravel(), panels):
        rate = np.zeros((len(DEGREES), len(SIZES)))
        medL = np.full((len(DEGREES), len(SIZES)), np.nan)
        for i, d in enumerate(DEGREES):
            for j, s in enumerate(SIZES):
                got = [r for r in rows if r[0] == s and r[1] == d and r[3] == frac]
                rate[i, j] = np.mean([r[8] for r in got])
                Ls = [r[7] for r in got if r[8]]
                if Ls:
                    medL[i, j] = np.median(Ls)
        im = ax.imshow(rate, cmap="YlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(DEGREES)):
            for j in range(len(SIZES)):
                txt = f"{rate[i, j]:.0%}"
                if not np.isnan(medL[i, j]):
                    txt += f"\n$L$ = {medL[i, j]:g}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if rate[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES)
        ax.set_title(name)
    for ax in axes[1]:
        ax.set_xlabel("cluster size $|V(I)|$")
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(DEGREES)))
        ax.set_yticklabels([f"{d:g}" for d in DEGREES])
        ax.set_ylabel("mean degree")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(f"fraction of {CLUSTERS} clusters embedded")
    save(fig, "fig_H_stage2_feasibility.png")


def replot():
    rows = load()
    summary(rows)
    plot(rows)


if __name__ == "__main__":
    import sys
    replot() if "--plot" in sys.argv else main()
