"""
Figure 11 of the report: does choosing the border nodes by degree matter? 20
Erdos-Renyi graphs with 20 vertices and 30 edges, 5 seeds each, with the five
highest-degree, five random or the five lowest-degree vertices as border nodes,
in Mode 2 at L = 5 to 9.
"""
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx

from common import OUT, TEXTWIDTH, one_based, pick_border, run, save, C_M1, C_M2, C_UNC

GRAPHS = list(range(1, 21))
SEEDS = list(range(1, 6))
LS = [5, 6, 7, 8, 9]
CHOICES = ["high", "random", "low"]


def task(args):
    g, L, how, seed = args
    I = one_based(nx.gnm_random_graph(20, 30, seed=g))
    border = pick_border(I, 5, 100 * g + seed, how)
    res, _ = run(I, L, border, "permute", seed)
    ok = int(res is not None and res.success)
    return (g, L, how, seed, ok, int(res is None))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    path = os.path.join(OUT, "border_node_degree_many_graphs.csv")
    old = []
    if os.path.exists(path):
        old = [(int(r["graph"]), int(r["L"]), r["border_choice"], int(r["seed"]),
                int(r["success"]), int(r["no_corridor"])) for r in csv.DictReader(open(path))]
    done = {(r[0], r[1], r[2], r[3]) for r in old}
    jobs = [(g, L, how, s) for g in GRAPHS for L in LS for how in CHOICES for s in SEEDS
            if (g, L, how, s) not in done]
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = old + list(ex.map(task, jobs, chunksize=4))
    with open(os.path.join(OUT, "border_node_degree_many_graphs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "L", "border_choice", "seed", "success", "no_corridor"])
        w.writerows(rows)
    report(rows)


def report(rows):
    from scipy.stats import fisher_exact
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.75 * TEXTWIDTH, 2.6))
    cols = {"high": "#d62728", "random": "#7f7f7f", "low": "#1f77b4"}
    names = {"high": "5 highest-degree vertices", "random": "5 random vertices",
             "low": "5 lowest-degree vertices"}
    w = 0.26
    for i, how in enumerate(CHOICES):
        for j, L in enumerate(LS):
            got = [r[4] for r in rows if r[1] == L and r[2] == how]
            k, n = sum(got), len(got)
            lo, hi = wilson(k, n)
            x = j + (i - 1) * w
            ax.bar(x, k / n, w, color=cols[how], label=names[how] if j == 0 else None)
            pass
            print(f"L={L} {how:6s}: {k}/{n} = {k / n:.0%}  (95% CI {lo:.0%}-{hi:.0%})")
        # in how many graphs is highest-degree below random?
    for L in LS:
        a = [r[4] for r in rows if r[1] == L and r[2] == "high"]
        b = [r[4] for r in rows if r[1] == L and r[2] == "random"]
        c = [r[4] for r in rows if r[1] == L and r[2] == "low"]
        p1 = fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]).pvalue
        p2 = fisher_exact([[sum(c), len(c) - sum(c)], [sum(b), len(b) - sum(b)]]).pvalue
        worse = sum(
            sum(r[4] for r in rows if r[0] == g and r[1] == L and r[2] == "high")
            < sum(r[4] for r in rows if r[0] == g and r[1] == L and r[2] == "random")
            for g in GRAPHS)
        better = sum(
            sum(r[4] for r in rows if r[0] == g and r[1] == L and r[2] == "high")
            > sum(r[4] for r in rows if r[0] == g and r[1] == L and r[2] == "random")
            for g in GRAPHS)
        print(f"L={L}: Fisher high vs random p={p1:.2g}; low vs random p={p2:.2g}; "
              f"graphs where high < random: {worse}, high > random: {better}")
    ax.set_xticks(range(len(LS)))
    ax.set_xticklabels([str(L) for L in LS])
    ax.set_xlabel("hardware size $L$")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mode 2 success rate")
    ax.legend(title="border nodes", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=3, handlelength=1.2, columnspacing=0.9, handletextpad=0.5)
    save(fig, "fig_E2_degree_many_graphs.png")


def replot():
    rows = [(int(r["graph"]), int(r["L"]), r["border_choice"], int(r["seed"]),
             int(r["success"]), int(r["no_corridor"]))
            for r in csv.DictReader(open(os.path.join(OUT, "border_node_degree_many_graphs.csv")))]
    report(rows)


if __name__ == "__main__":
    import sys
    replot() if "--plot" in sys.argv else main()
