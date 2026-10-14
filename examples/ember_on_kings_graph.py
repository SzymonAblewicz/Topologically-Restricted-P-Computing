"""
The Ember reconstruction of PSSA on the King's graph, on a random graph with 20 vertices
and 30 edges on KG(7), drawn from the guiding pattern and the initial placement to the
final embedding.
"""
import os
import sys

import networkx as nx
import matplotlib.pyplot as plt
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pssa/ember_on_kings_graph")]

from pssa import ImprovedPSSA, build_hardware_graph
from visualize_guiding_pattern import draw_guiding_pattern
from visualize_initial_placement import draw_initial_placement
from visualize_final_result import draw_final_result

def main():
    L = 7

    source_graph = nx.gnm_random_graph(20, 30, seed=42)
    # or eg. nx.circulant_graph(12, [1, 2, 8])

    # vertex labels from 1
    mapping = {n: n + 1 for n in source_graph.nodes()}
    source_graph = nx.relabel_nodes(source_graph, mapping)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    draw_guiding_pattern(L=L, ax=axes[0], show=False)
    draw_initial_placement(source_graph, L=L, ax=axes[1], show=False)

    embedder = ImprovedPSSA(L=L, tmax=1_000_000, verbose=False)
    print("Running PSSA...")
    result = embedder.run(source_graph, deadline=None)

    if result.success:
        print(f"Successfully found an embedding in {result.wall_time:.2f} seconds!")
        for logical_node, physical_chain in result.phi.items():
            print(f"Logical Node {logical_node} -> Physical Qubits {physical_chain}")
    else:
        print(f"Did not find a complete embedding ({result.eemb}/{result.m_I} edges) "
              f"in {result.wall_time:.2f} seconds.")

    draw_final_result(result.phi, source_graph, embedder.H, L=L,
                       success=result.success, ax=axes[2], show=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()