"""
Table 3 of the report: the improved PSSA with Gibbs-weighted moves switched on and off,
everything else identical. Each graph is run at the smallest L where either version
succeeds on at least 2 of 6 seeds, so the comparison is made where it is hard, with
20 seeds per version, and the uniform version is also run with the same annealing time.
"""
import csv
import json
import math
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pssa", "improved_pssa"))

import networkx as nx
import improved_pssa as P

OUT = os.path.join(ROOT, "results", "section2_pssa")
TMAX = 300_000
SEEDS_CAL = list(range(1, 7))
SEEDS = list(range(1, 21))


def one_based(G):
    return nx.relabel_nodes(G, {n: n + 1 for n in G.nodes()})


GRAPHS = {
    "random 20 nodes 30 edges": one_based(nx.gnm_random_graph(20, 30, seed=42)),
    "random 20 nodes 40 edges": one_based(nx.gnm_random_graph(20, 40, seed=42)),
    "Barabasi-Albert 30 nodes m=2": one_based(nx.barabasi_albert_graph(30, 2, seed=42)),
    "random cubic 24 nodes": one_based(nx.random_regular_graph(3, 24, seed=42)),
}


def run_once(I, L, tmax, gibbs, seed):
    H = P.build_hardware_graph(L)
    gp = P.build_guiding_pattern(H, L)
    sched = P.KingsGraphSchedule(tmax=tmax, L=L)
    t0 = time.perf_counter()
    phi, e_anneal = P._pssa_cpp(I, H, gp, sched, seed=seed, gibbs=gibbs)
    t_anneal = time.perf_counter() - t0
    phi_f = P.terminal_search(phi, I, H)
    e_final = P.eemb(phi_f, I, H, P.invert(phi_f))
    m = I.number_of_edges()
    ok = e_final == m and P.is_valid_embedding(phi_f, I, H)
    return dict(e_anneal=e_anneal, e_final=e_final, m=m, success=int(ok),
                t_anneal=t_anneal, qubits=sum(len(c) for c in phi_f.values()))


def main():
    rows = []
    summary = {}
    for name, I in GRAPHS.items():
        n, m = I.number_of_nodes(), I.number_of_edges()
        L = 3
        while L * L - 1 < n:
            L += 1
        L = max(L, 4)
        chosen = None
        for L_try in range(L, L + 10):
            s = {}
            for g in (False, True):
                s[g] = sum(run_once(I, L_try, TMAX, g, sd)["success"] for sd in SEEDS_CAL)
            print(f"[cal] {name} L={L_try}: blind {s[False]}/6, gibbs {s[True]}/6", flush=True)
            if max(s.values()) >= 2:
                chosen = L_try
                break
        if chosen is None:
            print(f"[cal] {name}: no L found, skipped", flush=True)
            continue

        res = {}
        for g in (False, True):
            rs = []
            for sd in SEEDS:
                r = run_once(I, chosen, TMAX, g, sd)
                r.update(graph=name, n=n, L=chosen, tmax=TMAX, gibbs=int(g), seed=sd, phase="equal steps")
                rows.append(r)
                rs.append(r)
            res[g] = rs
            print(f"[B] {name} L={chosen} gibbs={g}: "
                  f"{sum(r['success'] for r in rs)}/20, "
                  f"mean t={statistics.mean(r['t_anneal'] for r in rs):.2f}s", flush=True)

        t_g = statistics.mean(r["t_anneal"] for r in res[True])
        t_b = statistics.mean(r["t_anneal"] for r in res[False])
        tmax_eq = int(TMAX * t_g / t_b)
        rs_eq = []
        for sd in SEEDS:
            r = run_once(I, chosen, tmax_eq, False, sd)
            r.update(graph=name, n=n, L=chosen, tmax=tmax_eq, gibbs=0, seed=sd, phase="equal time")
            rows.append(r)
            rs_eq.append(r)
        print(f"[C] {name} blind at tmax={tmax_eq}: {sum(r['success'] for r in rs_eq)}/20, "
              f"mean t={statistics.mean(r['t_anneal'] for r in rs_eq):.2f}s", flush=True)

        def agg(rs):
            return dict(
                success=sum(r["success"] for r in rs), runs=len(rs),
                mean_e_anneal=statistics.mean(r["e_anneal"] for r in rs),
                mean_e_final=statistics.mean(r["e_final"] for r in rs),
                mean_t_anneal=statistics.mean(r["t_anneal"] for r in rs),
                tmax=rs[0]["tmax"],
            )
        summary[name] = dict(n=n, m=m, L=chosen, blind=agg(res[False]),
                             gibbs=agg(res[True]), blind_equal_time=agg(rs_eq))

    with open(os.path.join(OUT, "gibbs_weighted_moves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "gibbs_weighted_moves_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
