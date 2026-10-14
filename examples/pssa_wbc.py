"""
PSSA wBC in Mode 2 on a random graph with 20 vertices and 30 edges on KG(6), with a
random tenth of the vertices as border nodes, drawn from the guiding pattern to the
final embedding, with a check that every border node ends on the perimeter.
"""
import os
import random
import sys
import networkx as nx
import matplotlib.pyplot as plt

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pssa/pssa_wbc")]

from pssa_wbc import ImprovedPSSA, _perimeter_nodes
from draw_embedding import (
    draw_guiding_pattern, draw_initial_placement, draw_final_result,
    draw_node_degree_list,
)

# ring the top half of the vertices by degree red and the bottom half green
BIPOLAR_DEGREES = True


def select_border_nodes(I: nx.Graph, frac: float = 0.25, seed: int = 42):
    rng = random.Random(seed)
    nodes = list(I.nodes())
    k = max(1, round(frac * len(nodes)))
    return set(rng.sample(nodes, k=k))


def main():
    I = nx.gnm_random_graph(20, 30, seed=42)
    # or eg. nx.complete_bipartite_graph(5, 5)

    border_nodes = select_border_nodes(I, frac=0.1, seed=42)
    print(f"Border nodes ({len(border_nodes)}/{I.number_of_nodes()}): "
          f"{sorted(border_nodes)}")

    L = 6
    degree_ncols = 5
    degree_nrows = -(-I.number_of_nodes() // degree_ncols)  # ceil division
    fig = plt.figure(figsize=(21, 7 + 0.6 * degree_nrows + 1))
    gs = fig.add_gridspec(2, 3, height_ratios=[7, 0.6 * degree_nrows + 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_degrees = fig.add_subplot(gs[1, :])

    draw_guiding_pattern(L=L, ax=axes[0], show=False)
    draw_initial_placement(I, L=L, ax=axes[1], show=False, border_nodes=border_nodes,
                            bipolar_degrees=BIPOLAR_DEGREES)

    embedder = ImprovedPSSA(
        L=L, tmax=1_000_000, gibbs=True, seed=-1, verbose=True,
        border_nodes=border_nodes, border_mode="permute",
    )
    result = embedder.run(I)

    print(f"\n{result}")
    print(f"Initial representative touch points: {embedder.last_anchors}")

    perimeter = _perimeter_nodes(L)
    ok = True
    final_touch_nodes = set()
    for i in sorted(border_nodes):
        touches = [u for u in result.phi[i] if u in perimeter]
        if not touches:
            ok = False
            print(f"  VIOLATION: node {i} has ZERO perimeter-touching hw "
                  f"nodes in final chain {result.phi[i]}")
        else:
            print(f"  node {i}: {len(touches)} perimeter touch(es) -> {touches}")
            final_touch_nodes.update(touches)
    if ok:
        print("\nEvery border super-vertex retained a perimeter touch for "
              "the entire run. Mode-2 constraint OK.")
    else:
        print("\nBorder perimeter-touch constraint VIOLATED -- see above.")

    draw_final_result(result.phi, I, embedder.H, L=L, success=result.success,
                       ax=axes[2], anchor_hw_nodes=final_touch_nodes,
                       show=False, bipolar_degrees=BIPOLAR_DEGREES)

    draw_node_degree_list(ax_degrees, I, ncols=degree_ncols, bipolar_degrees=BIPOLAR_DEGREES)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
