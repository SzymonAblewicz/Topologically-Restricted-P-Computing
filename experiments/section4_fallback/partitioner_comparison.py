"""
The comparison of the two partitioners (Section 4.3.3 of the report), Mt-KaHyPar and
GGP + T-FM on random cubic graphs, both called exactly as the fallback calls them.

    python partitioner_comparison.py     # writes partitioner_comparison.csv
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "divide_and_conquer", "fallback", "benchmark_graphs")]
from graphs import build_random_cubic      # noqa: E402
from border_node_partitioning import partition_graph           # noqa: E402

OUT = os.path.join(ROOT, "results", "section4_fallback", "partitioner_comparison.csv")
SIZES = [1000, 3000, 10000, 30000, 100000]
CAPS = [10, 30, 50, 70, 300]
SEEDS = [1, 2, 3]
METHODS = {"mtkahypar": "Mt-KaHyPar", "csr": "GGP + T-FM"}


def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r["n"], r["seed"], r["s"], r["method"]) for r in csv.DictReader(open(OUT))}
    for n in SIZES:
        for seed in SEEDS:
            G = build_random_cubic(n=n, seed=seed)[0]
            for s in CAPS:
                for method in METHODS:
                    if (str(n), str(seed), str(s), method) in done:
                        continue
                    t0 = time.perf_counter()
                    part = partition_graph(G, max_cluster_size=s, method=method, seed=7, verbose=False)
                    row = dict(n=n, seed=seed, s=s, method=method, k=part.k, T=part.border_count,
                               T_over_n=round(part.border_count / n, 4), cut=part.cut_size,
                               secs=round(time.perf_counter() - t0, 2), used=part.method)
                    print(row, flush=True)
                    new = not os.path.exists(OUT)
                    with open(OUT, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row))
                        if new:
                            w.writeheader()
                        w.writerow(row)


if __name__ == "__main__":
    main()
