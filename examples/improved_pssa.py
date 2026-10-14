"""
The improved PSSA on a random graph with 300 vertices and 300 edges on KG(46), with 10
independent runs keeping the valid embedding that uses the fewest hardware vertices,
drawn next to the guiding pattern and the initial placement.
"""
import os
import sys

import networkx as nx
import matplotlib.pyplot as plt
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pssa/improved_pssa")]

from improved_pssa import ImprovedPSSA, build_hardware_graph
from draw_embedding import draw_guiding_pattern, draw_initial_placement, draw_final_result

def make_test_graph():
    source_graph = nx.gnm_random_graph(300, 300, seed=42)

    # or eg. nx.complete_bipartite_graph(5, 5) or nx.barabasi_albert_graph(50, 3, seed=13)

    mapping = {n: n + 1 for n in source_graph.nodes()}
    return nx.relabel_nodes(source_graph, mapping)


def qubits_used(phi):
    return sum(len(chain) for chain in phi.values())

def main():
    L = 46
    source_graph = make_test_graph()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    draw_guiding_pattern(L=L, ax=axes[0], show=False)
    draw_initial_placement(source_graph, L=L, ax=axes[1], show=False)

    best_result = None
    best_qubits = None
    best_fail_result = None
    best_fail_embedder = None

    for run_idx in range(10):
        print(f"\n=== Run {run_idx + 1}/10 ===")
        embedder = ImprovedPSSA(L=L, tmax=100_000, gibbs=True, seed=-1, verbose=True)
        result = embedder.run(source_graph, deadline=None)

        if result.success:
            used = qubits_used(result.phi)
            print(f"  Success — {used} qubits used ({result.wall_time:.2f}s)")
            if best_qubits is None or used < best_qubits:
                best_result = result
                best_qubits = used
                best_embedder = embedder
        else:
            print(f"  Failed — {result.eemb}/{result.m_I} edges ({result.wall_time:.2f}s)")
            if best_fail_result is None or result.eemb > best_fail_result.eemb:
                best_fail_result = result
                best_fail_embedder = embedder

    if best_result is not None:
        print(f"\nBest embedding: {best_qubits} qubits used")
        draw_final_result(best_result.phi, source_graph, best_embedder.H, L=L,
                           success=True, ax=axes[2], show=False)
    elif best_fail_result is not None:
        print(f"\nNo successful embedding found. Showing best failed attempt: "
              f"{best_fail_result.eemb}/{best_fail_result.m_I} edges")
        draw_final_result(best_fail_result.phi, source_graph, best_fail_embedder.H, L=L,
                           success=False, ax=axes[2], show=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()