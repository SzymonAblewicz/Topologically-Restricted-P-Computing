"""
Figure 18 of the report: minimizing the cut against minimizing T, with the same
Mt-KaHyPar call on the same random cubic graphs and only the nets changed, one per
edge (cut-net = the cut) or one per vertex, its closed neighbourhood (cut-net = T),
without the size repair or T-FM, so each objective is measured on its own.

    python cut_vs_border_nodes.py            # writes cut_vs_border_nodes.csv
"""
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "divide_and_conquer", "fallback", "benchmark_graphs")]
import mtkahypar_partitioner as KB                           # noqa: E402
from graphs import build_random_cubic      # noqa: E402

OUT = os.path.join(ROOT, "results", "section4_fallback", "cut_vs_border_nodes.csv")
S = 50
SIZES = [1000, 3000, 10000, 30000, 100000]
SEEDS = [1, 2, 3]
THREADS = 4


def measure(G, labels):
    T = KB.border_count(G, labels)
    cut = sum(1 for u, v in G.edges() if labels[u] != labels[v])
    return T, cut


def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r["n"], r["seed"], r["objective"]) for r in csv.DictReader(open(OUT))}
    for n in SIZES:
        for seed in SEEDS:
            G = build_random_cubic(n=n, seed=seed)[0]
            k = math.ceil(n / S)
            for objective, model in (("cut", "edge"), ("T", "closed_neighbourhood")):
                if (str(n), str(seed), objective) in done:
                    continue
                t0 = time.time()
                labels = KB.partition_hypergraph(G, k, eps=0.03, threads=THREADS,
                                                 verify=(model == "closed_neighbourhood"),
                                                 net_model=model)
                T, cut = measure(G, labels)
                row = dict(n=n, seed=seed, s=S, k=k, objective=objective, T=T,
                           T_over_n=round(T / n, 4), cut=cut, cut_over_n=round(cut / n, 4),
                           secs=round(time.time() - t0, 1))
                print(row, flush=True)
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)


if __name__ == "__main__":
    main()
