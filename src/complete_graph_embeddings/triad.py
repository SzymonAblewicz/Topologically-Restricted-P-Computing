"""
My reconstruction of the TRIAD construction of Choi (2011) (Section 1.3 of the report),
which lays out K_n as a triangle of chains of n - 1 vertices each.
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque
from typing import Dict, List, Tuple
from matplotlib.collections import LineCollection

def build_triad(n: int) -> Dict[int, List[Tuple[int, int]]]:
    """The TRIAD embedding of K_n, {vertex: [(x, y), ...]}."""
    chains = {i: [] for i in range(1, n + 1)}

    for r in range(n - 1):
        for k in range(r + 1):
            u = k + 1
            v = n - r + k
            y = r
            x_u = 2 * k - r
            x_v = 2 * k + 1 - r

            chains[u].append((x_u, y))
            chains[v].append((x_v, y))

    return chains

def king_adjacent(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    """Whether two points are adjacent on the King's graph."""
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) == 1 and p1 != p2

def is_connected(chain_nodes: List[Tuple[int, int]]) -> bool:
    """Whether a chain is connected on the King's graph."""
    if not chain_nodes: 
        return True

    chain_set = set(chain_nodes)
    visited = {chain_nodes[0]}
    queue = deque([chain_nodes[0]])

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in chain_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(chain_nodes)

def verify_embedding(chains: Dict[int, List[Tuple[int, int]]], n: int) -> bool:
    """Check that the chains are disjoint, connected and n - 1 long, and that every edge of K_n is realized."""
    print(f"\n--- Verifying TRIAD K_{n} ---")
    all_passed = True

    # 1. Disjoint
    all_coords = [node for nodes in chains.values() for node in nodes]
    if len(set(all_coords)) != len(all_coords):
        print("  [FAIL] Node collision! Multiple virtual nodes share a coordinate.")
        return False

    # 2. Connected, and n - 1 long
    for v, nodes in chains.items():
        if not is_connected(nodes):
            print(f"  [FAIL] Chain {v} is NOT contiguous!")
            all_passed = False
        if len(nodes) != n - 1:
            print(f"  [FAIL] Chain {v} has length {len(nodes)}, expected {n-1}")
            all_passed = False

    if all_passed:
        print(f"  [PASS] All {n} chains are contiguous and exact length {n-1}.")

    # 3. Every edge of K_n
    grid = {node: v for v, nodes in chains.items() for node in nodes}
    logical_edges_found = set()
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for (x, y), u in grid.items():
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in grid:
                v = grid[neighbor]
                if u != v:
                    logical_edges_found.add(tuple(sorted((u, v))))

    missing_edges = [(i, j) for i in range(1, n) for j in range(i + 1, n + 1) if (i, j) not in logical_edges_found]

    if not missing_edges:
        print(f"  [PASS] All {n*(n-1)//2} logical K_{n} edges are validly represented.")
    else:
        print(f"  [FAIL] Missing logical edges: {missing_edges}")
        all_passed = False

    if all_passed: print(f"  => SUCCESS: K_{n} verified.")
    return all_passed

def get_color(v: int, n: int):
    """The palette for n <= 8, interpolated for larger n."""
    palette = {
        1: '#F59D56', 2: '#C5E063', 3: '#71C754', 4: '#62D99A', 
        5: '#60C4E6', 6: '#4960D8', 7: '#7A4CE6', 8: '#D849D8'
    }
    if n <= 8:
        return palette.get(v)
    return cm.turbo(v / n)


def draw_triad(chains: Dict[int, List[Tuple[int, int]]], n: int, ax=None):
    """Draw the TRIAD embedding, scaled to n."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    def to_visual(x, y): return x, -y

    # sizes scale with n
    m_size = max(1.0, 70.0 / n)
    f_size = max(1.0, 40.0 / n)
    lw_intra = max(0.3, 20.0 / n)
    lw_inter = max(0.05, 8.0 / n)

    draw_labels = n <= 25
    edge_width = max(0.1, 4.0 / n) if n <= 40 else 0

    grid = {node: v for v, nodes in chains.items() for node in nodes}

    # 1. Edges between chains
    inter_segments = []
    for (x, y), u in grid.items():
        neighbor = (x + 1, y) 
        if neighbor in grid and grid[neighbor] != u and (x + y) % 2 == 0:
            X1, Y1 = to_visual(x, y)
            X2, Y2 = to_visual(*neighbor)
            inter_segments.append([(X1, Y1), (X2, Y2)])
    if inter_segments:
        lc = LineCollection(inter_segments, colors='black', linewidths=lw_inter, alpha=0.6, zorder=1)
        ax.add_collection(lc)

    # 2. The chains
    for v, nodes in chains.items():
        color = get_color(v, n)
        nodes_set = set(nodes)
        for x, y in nodes:
            for dx, dy in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor in nodes_set:
                    X1, Y1 = to_visual(x, y)
                    X2, Y2 = to_visual(*neighbor)
                    ax.plot([X1, X2], [Y1, Y2], color=color, linewidth=lw_intra, alpha=0.8, zorder=2)

    # 3. Vertices and labels
    for v, nodes in chains.items():
        color = get_color(v, n)
        for node in nodes:
            X, Y = to_visual(*node)
            ax.plot(X, Y, 'o', markersize=m_size, markerfacecolor=color, 
                    markeredgecolor='black', markeredgewidth=edge_width, zorder=3)

            if draw_labels and f_size >= 2:
                ax.text(X, Y, str(v), ha='center', va='center', 
                        fontsize=f_size, color='black', weight='bold', zorder=4)

    ax.set_aspect('equal')
    ax.axis('off')

    # a margin, so large nodes at the edge aren't clipped
    ax.margins(0.15) 

    ax.set_title(f"TRIAD $K_{{{n}}}$", fontsize=14, weight='bold')

if __name__ == "__main__":
    test_cases = [4, 5, 11, 25]

    # 1. Verification
    for n in test_cases:
        chains = build_triad(n)
        verify_embedding(chains, n)

    # 2. Drawing
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, n in enumerate(test_cases):
        chains = build_triad(n)
        draw_triad(chains, n, ax=axes[idx])

    plt.tight_layout()
    plt.savefig("triad_examples.png", dpi=150)
    plt.show()