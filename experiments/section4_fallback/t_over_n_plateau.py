"""
Figure 19 of the report: the plateau of T/n. Random cubic graphs at s = 50 with a hard
cap of 50, five graphs at each size from 1,000 to 1,000,000 vertices, partitioned only,
by Mt-KaHyPar directly (up to 100,000) and in two levels (from 10,000), keeping the lower T/n.

    python -B t_over_n_plateau.py     # writes t_over_n_plateau.csv
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "divide_and_conquer", "fallback", "benchmark_graphs")]

from mtkahypar_partitioner import partition_graph_kahypar  # noqa: E402
from graphs import build_random_cubic                       # noqa: E402

RES = os.path.join(ROOT, "results", "section4_fallback")
OUT = os.path.join(RES, "t_over_n_plateau.csv")
LOG = os.path.join(RES, "log_t_over_n_plateau.txt")
SIZES = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
SEEDS = [1, 2, 3, 4, 5]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def main():
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["n", "seed", "call", "K1", "k", "T", "T_over_n", "max_size", "secs"])
        for n in SIZES:
            for seed in SEEDS:
                G, _, _ = build_random_cubic(n=n, seed=seed)
                calls = []
                if n <= 100000:
                    calls.append(("direct", 1))
                if n >= 10000:
                    calls.append(("two-level", max(2, round(n / 2500))))
                for call, K1 in calls:
                    t0 = time.perf_counter()
                    r = partition_graph_kahypar(G, 50, levels=1 if call == "direct" else 2)
                    dt = time.perf_counter() - t0
                    mx = max(len(g) for g in r.subgraphs.values())
                    w.writerow([n, seed, call, K1, r.k, r.border_count, round(r.border_count / n, 5), mx, round(dt, 1)])
                    fh.flush()
                    log(f"n={n} seed={seed} {call} K1={K1}: T/n={r.border_count / n:.4f} max {mx} in {dt:.0f} s")
    log("ALL DONE")


if __name__ == "__main__":
    main()
