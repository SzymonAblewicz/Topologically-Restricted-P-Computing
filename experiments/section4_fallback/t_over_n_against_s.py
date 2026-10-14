"""
T/n against the cluster size s (Section 4.4 of the report), on a random cubic graph
with 100,000 vertices, partitioned by Mt-KaHyPar in one call at every s from 30 to 10,000.

    python -B t_over_n_against_s.py     # writes t_over_n_against_s.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("fallback", "benchmark_graphs")]

from mtkahypar_partitioner import partition_graph_kahypar  # noqa: E402
from graphs import build_random_cubic                       # noqa: E402

OUT = os.path.join(ROOT, "results", "section4_fallback", "t_over_n_against_s.csv")
N, SEED = 100000, 1
SS = [30, 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000]


def main():
    G, _, _ = build_random_cubic(n=N, seed=SEED)
    m = G.number_of_edges()
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "seed", "s", "k", "T_over_n", "cut_over_m"])
        for s in SS:
            part = partition_graph_kahypar(G, max_cluster_size=s, threads=3, levels=1)
            lab = part.labels
            cut = sum(1 for u, v in G.edges() if lab[u] != lab[v])
            w.writerow([N, SEED, s, part.k, round(part.border_count / N, 4), round(cut / m, 4)])
            fh.flush()
            print(f"s={s}: k={part.k}, T/n={part.border_count / N:.4f}", flush=True)


if __name__ == "__main__":
    main()
