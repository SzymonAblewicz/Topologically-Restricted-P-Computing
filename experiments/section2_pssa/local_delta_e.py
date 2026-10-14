"""
Table 4 of the report: the time of one annealing step with the old C++ core, which
recounts Eemb at every iteration (translation_to_cpp/pssa_core), against the new one, which
computes Delta E locally (cheap_pssa_core), on random cubic graphs with n ~ L^2 / 4,
so the super vertices stay at around four hardware vertices each.
"""
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pssa", "improved_pssa"))

import networkx as nx
import improved_pssa as P
import cheap_pssa_core

sys.path.insert(0, os.path.join(ROOT, "experiments", "section2_pssa", "translation_to_cpp"))
import pssa_core as old_core  # recounts Eemb in full every step

OUT = os.path.join(ROOT, "results", "section2_pssa")
TMAX = 50_000
LS = [6, 10, 20, 40, 80]
SEEDS = [1, 2, 3]


def graph_for(L):
    n = max(8, (L * L) // 4)
    if n % 2:
        n += 1
    G = nx.random_regular_graph(3, n, seed=42)
    return nx.relabel_nodes(G, {v: v + 1 for v in G.nodes()})


def main():
    rows = []
    for L in LS:
        I = graph_for(L)
        H = P.build_hardware_graph(L)
        gp = P.build_guiding_pattern(H, L)
        for label, core in (("full recount", old_core), ("cheap delta", cheap_pssa_core)):
            P._pssa_core_ext = core
            ts = []
            for sd in SEEDS:
                sched = P.KingsGraphSchedule(tmax=TMAX, L=L)
                t0 = time.perf_counter()
                P._pssa_cpp(I, H, gp, sched, seed=sd, gibbs=True)
                ts.append(time.perf_counter() - t0)
            us = 1e6 * min(ts) / TMAX
            rows.append(dict(L=L, n=I.number_of_nodes(), hw_edges=H.number_of_edges(),
                             core=label, tmax=TMAX, best_time_s=min(ts), us_per_step=us))
            print(f"L={L:3d} n={I.number_of_nodes():5d} |E(H)|={H.number_of_edges():6d} "
                  f"{label:13s}: {us:8.2f} us/step", flush=True)
    with open(os.path.join(OUT, "local_delta_e.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
