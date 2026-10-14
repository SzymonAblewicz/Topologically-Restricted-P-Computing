"""
Figure 10 of the report: the success rate of Mode 1 and Mode 2 against the share of
border nodes and the hardware size L, on the Erdos-Renyi graph with 20 vertices and
30 edges, with 20 seeds per cell and the border nodes drawn at random per seed.
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import networkx as nx

from common import OUT, TEXTWIDTH, one_based, pick_border, run, save, C_M1, C_M2

I = one_based(nx.gnm_random_graph(20, 30, seed=42))
LS = [5, 6, 7, 8, 9]
KS = [0, 2, 5, 10, 15, 18]      # 0, 10, 25, 50, 75, 90 % of 20 nodes
PKS = KS[:-1]  # the 18-border-node column is in the caption, not drawn
SEEDS = list(range(1, 21))


def task(args):
    L, k, mode, seed = args
    border = pick_border(I, k, seed)
    res, _ = run(I, L, border if k else None, mode if k else None, seed)
    if res is None:
        return (L, k, mode, seed, 0, -1, 1)
    return (L, k, mode, seed, int(res.success), res.eemb, 0)


def main():
    jobs = []
    for L in LS:
        for seed in SEEDS:
            jobs.append((L, 0, "none", seed))
            for k in KS[1:]:
                for mode in ("fixed", "permute"):
                    jobs.append((L, k, mode, seed))
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(task, jobs, chunksize=4))
    with open(os.path.join(OUT, "number_of_border_nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "border_nodes", "mode", "seed", "success", "eemb", "no_corridor"])
        w.writerows(rows)
    plot(rows)


def plot(rows):
    import matplotlib.pyplot as plt
    rate = {}
    for mode in ("fixed", "permute"):
        M = np.zeros((len(LS), len(PKS)))
        for i, L in enumerate(LS):
            for j, k in enumerate(PKS):
                m = "none" if k == 0 else mode
                got = [r[4] for r in rows if r[0] == L and r[1] == k and r[2] == m]
                M[i, j] = np.mean(got)
        rate[mode] = M
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, mode, name, col in zip(axes, ("fixed", "permute"),
                                   ("Mode 1 (fixed border nodes)", "Mode 2 (free border nodes)"),
                                   (C_M1, C_M2)):
        M = rate[mode]
        im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(LS)):
            for j in range(len(PKS)):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(PKS)))
        ax.set_xticklabels([f"{k}\n({k / 20:.0%})" for k in PKS])
        ax.set_xlabel("border nodes")
        ax.set_title(name, color=col)
    axes[0].set_yticks(range(len(LS)))
    axes[0].set_yticklabels(LS)
    axes[0].set_ylabel("hardware size $L$")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("success rate (20 seeds)")
    save(fig, "fig_D_border_fraction.png")


def replot():
    rows = [(int(r["L"]), int(r["border_nodes"]), r["mode"], int(r["seed"]), int(r["success"]),
             int(r["eemb"]), int(r["no_corridor"]))
            for r in csv.DictReader(open(os.path.join(OUT, "number_of_border_nodes.csv")))]
    plot(rows)


if __name__ == "__main__":
    import sys
    replot() if "--plot" in sys.argv else main()
