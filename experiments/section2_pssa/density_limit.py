"""
Figure 9 of the report: the smallest L on which the improved PSSA embeds a 20-vertex
graph as its edges grow from 30 to 50, with RUNS_PER_L runs at each L. Resumable by
edge count, writing density_limit.csv and density_limit_summary.csv.
"""
import csv
import os
import time

import sys

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pssa", "improved_pssa"))
from improved_pssa import ImprovedPSSA

N = 20
EDGE_MIN = 30
EDGE_MAX = 50
RUNS_PER_L = 3
TMAX = 1_000_000
L_START = 5
L_MAX = 30
GRAPH_SEED = 42
SWEEP_DEADLINE_S = 20 * 60  # hard wall-clock budget for this process

RESULTS_CSV = os.path.join(ROOT, "results", "section2_pssa", "density_limit_summary.csv")
RUNS_CSV = os.path.join(ROOT, "results", "section2_pssa", "density_limit.csv")

RUN_FIELDS = ["edges", "L", "run_idx", "success", "eemb", "m_I", "qubits_used", "wall_time"]
RESULT_FIELDS = ["edges", "min_L", "successes", "runs_at_min_L", "best_qubits", "level_wall_time_s"]


def make_graph(m):
    g = nx.gnm_random_graph(N, m, seed=GRAPH_SEED)
    mapping = {n: n + 1 for n in g.nodes()}
    return nx.relabel_nodes(g, mapping)


def qubits_used(phi):
    return sum(len(chain) for chain in phi.values())


def load_done_edges():
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add(int(row["edges"]))
    return done


def append_row(path, fields, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    done_edges = load_done_edges()
    sweep_t0 = time.perf_counter()
    budget_exceeded = False

    for m in range(EDGE_MIN, EDGE_MAX + 1):
        if m in done_edges:
            print(f"[skip] edges={m} already completed", flush=True)
            continue

        if time.perf_counter() - sweep_t0 > SWEEP_DEADLINE_S:
            print(f"[TIME BUDGET] {SWEEP_DEADLINE_S}s exceeded before starting edges={m}; "
                  f"stopping sweep here.", flush=True)
            budget_exceeded = True
            break

        graph = make_graph(m)
        print(f"\n=== edges={m} (source graph: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges) ===", flush=True)

        L = L_START
        found = False

        while L <= L_MAX and not found:
            if time.perf_counter() - sweep_t0 > SWEEP_DEADLINE_S:
                print(f"[TIME BUDGET] {SWEEP_DEADLINE_S}s exceeded mid-sweep at edges={m} L={L}; "
                      f"stopping sweep here (this edge count left incomplete).", flush=True)
                budget_exceeded = True
                break

            successes = 0
            best_qubits = None
            level_t0 = time.perf_counter()

            for run_idx in range(RUNS_PER_L):
                embedder = ImprovedPSSA(L=L, tmax=TMAX, gibbs=True, seed=-1, verbose=False)
                result = embedder.run(graph)
                used = qubits_used(result.phi) if result.success else None

                append_row(RUNS_CSV, RUN_FIELDS, {
                    "edges": m, "L": L, "run_idx": run_idx,
                    "success": result.success, "eemb": result.eemb, "m_I": result.m_I,
                    "qubits_used": used if used is not None else "",
                    "wall_time": f"{result.wall_time:.3f}",
                })
                print(f"  edges={m} L={L} run={run_idx + 1}/{RUNS_PER_L} "
                      f"success={result.success} eemb={result.eemb}/{result.m_I} "
                      f"time={result.wall_time:.2f}s", flush=True)

                if result.success:
                    successes += 1
                    if best_qubits is None or used < best_qubits:
                        best_qubits = used

            level_dt = time.perf_counter() - level_t0
            print(f"[level] edges={m} L={L}: {successes}/{RUNS_PER_L} succeeded "
                  f"({level_dt:.1f}s)", flush=True)

            if successes > 0:
                found = True
                append_row(RESULTS_CSV, RESULT_FIELDS, {
                    "edges": m, "min_L": L, "successes": successes,
                    "runs_at_min_L": RUNS_PER_L, "best_qubits": best_qubits,
                    "level_wall_time_s": f"{level_dt:.1f}",
                })
                print(f"[done] edges={m}: minimum working L = {L} "
                      f"({successes}/{RUNS_PER_L} runs succeeded, best {best_qubits} qubits)",
                      flush=True)
            else:
                L += 1

        if budget_exceeded:
            break

        if not found:
            append_row(RESULTS_CSV, RESULT_FIELDS, {
                "edges": m, "min_L": "", "successes": 0,
                "runs_at_min_L": RUNS_PER_L, "best_qubits": "",
                "level_wall_time_s": "",
            })
            print(f"[GIVE UP] edges={m}: no embedding found up to L={L_MAX}", flush=True)

    total_dt = time.perf_counter() - sweep_t0
    if budget_exceeded:
        print(f"\nSweep stopped early at {total_dt:.1f}s (20-minute budget). "
              f"Re-run this script to resume from where it left off.", flush=True)
    else:
        print(f"\nSweep complete in {total_dt:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
