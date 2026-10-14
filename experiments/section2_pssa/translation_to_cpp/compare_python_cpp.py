"""
The translation to C++ (Section 2.3.4 of the report): the same anneal run by the pure
Python loop and by the C++ loop, on a random graph with 20 vertices and 30 edges on
KG(6) with tmax = 3 x 10^5. comparison_output.txt is the run behind the report's
"about 30 times faster".
"""
import copy
import time

import networkx as nx

from pssa import (
    ImprovedPSSA, build_hardware_graph, build_guiding_pattern,
    KingsGraphSchedule, pssa, _pssa_cpp, terminal_search, invert, eemb,
    is_valid_embedding,
)


def make_test_graph():
    source_graph = nx.gnm_random_graph(20, 30, seed=42)
    mapping = {n: n + 1 for n in source_graph.nodes()}
    return nx.relabel_nodes(source_graph, mapping)


def run_one(label, fn, I, H, gp, schedule, seed, gibbs, weighted):
    t0 = time.perf_counter()
    phi_raw, _ = fn(I, H, gp, schedule, weighted=weighted, seed=seed,
                     verbose=False, deadline=None, gibbs=gibbs)
    phi_final = terminal_search(phi_raw, I, H)
    inv_final = invert(phi_final)
    e_final = eemb(phi_final, I, H, inv_final)
    m_I = I.number_of_edges()
    wall = time.perf_counter() - t0
    valid = is_valid_embedding(phi_final, I, H)
    success = (e_final == m_I) and valid

    print(f"[{label}]")
    print(f"  T start -> end : {schedule.temperature(0):.4f} -> {schedule.temperature(schedule.tmax):.4f}")
    print(f"  wall time      : {wall:.3f}s")
    print(f"  eemb           : {e_final}/{m_I}  (coverage={e_final/m_I:.3f})")
    print(f"  valid embedding: {valid}")
    print(f"  success        : {success}")
    print()
    return wall, e_final, success


def main():
    L = 6
    I = make_test_graph()
    seed = 7  # the same seed for both
    gibbs = True
    weighted = False

    H = build_hardware_graph(L)
    gp = build_guiding_pattern(H, L)
    schedule = KingsGraphSchedule(tmax=300_000, L=L)
    print(schedule.summary())
    V = I.number_of_nodes()
    E = I.number_of_edges()
    density = (2 * E) / (V * (V - 1)) if V > 1 else 0.0
    print(f"Test graph: |V(I)|={V}, |E(I)|={E}, density={density:.4f}")
    print()

    wall_cpp, e_cpp, ok_cpp = run_one("C++ pssa_core", _pssa_cpp, I, H, gp, schedule, seed, gibbs, weighted)
    wall_py, e_py, ok_py = run_one("pure-Python pssa()", pssa, I, H, gp, schedule, seed, gibbs, weighted)

    print("=" * 60)
    print(f"Speedup (python_time / cpp_time): {wall_py / wall_cpp:.1f}x")
    print(f"Both valid embeddings found     : {ok_py and ok_cpp}")
    print(f"Coverage comparable              : python={e_py}, cpp={e_cpp}")


if __name__ == "__main__":
    main()