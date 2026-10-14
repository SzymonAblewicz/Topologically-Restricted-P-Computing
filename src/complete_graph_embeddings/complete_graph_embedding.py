"""
My reconstruction of the best known complete graph embedding of Okuyama et al. (2016),
as described by Hung and Roychowdhury (2024) (Section 1.3 of the report), which embeds
K_n on KG(n-1), with vertex 1 along the first column and every other vertex a chain
running diagonally across the square. It is the guiding pattern of PSSA.
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque
from typing import Dict, List, Tuple
from matplotlib.collections import LineCollection

def build_square_triad(n: int) -> Dict[int, List[Tuple[int, int]]]:
    """The complete graph embedding of K_n on KG(n-1), {vertex: [(x, y), ...]}."""
    L = n - 1
    chains = {i: [] for i in range(1, n + 1)}

    # 1. Vertex 1 takes the first column
    for y in range(L):
        chains[1].append((0, y))

    # 2. Vertices 2, ..., n
    for i in range(2, n + 1):
        x = 1
        y = i - 2  # starts at row i - 1
        chains[i].append((x, y))

        # even vertices go down-right, odd ones up-right
        dy = 1 if i % 2 == 0 else -1

        # until the last column
        while x < L - 1:
            # at the top or bottom edge
            if (dy == 1 and y == L - 1) or (dy == -1 and y == 0):
                # bounce, one step right and then reverse
                x += 1
                chains[i].append((x, y))
                dy = -dy
            else:
                # diagonal step
                x += 1
                y += dy
                chains[i].append((x, y))

    return chains

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Verification (as in triad.py)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def king_adjacent(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) == 1 and p1 != p2

def is_connected(chain_nodes: List[Tuple[int, int]]) -> bool:
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
    print(f"\n--- Verifying Square K_{n} ---")
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Drawing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_color(v: int, n: int):
    palette = {
        1: '#F59D56', 2: '#C5E063', 3: '#71C754', 4: '#62D99A',
        5: '#60C4E6', 6: '#4960D8', 7: '#7A4CE6', 8: '#D849D8'
    }
    if n <= 8: return palette.get(v)
    return cm.turbo(v / n)

def draw_square_triad(chains: Dict[int, List[Tuple[int, int]]], n: int, ax=None):
    """Draw the complete graph embedding."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    def to_visual(x, y): return x, -y

    m_size = max(1.0, 70.0 / n)
    f_size = max(1.0, 40.0 / n)
    lw_intra = max(0.3, 20.0 / n)
    lw_inter = max(0.05, 8.0 / n)

    draw_labels = n <= 25
    edge_width = max(0.1, 4.0 / n) if n <= 40 else 0
    grid = {node: v for v, nodes in chains.items() for node in nodes}

    # 1. The hardware edges between different chains
    inter_segments = []
    drawn_edges = set()
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for (x, y), u in grid.items():
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in grid:
                v = grid[(nx, ny)]
                if u != v:
                    edge = tuple(sorted(((x, y), (nx, ny))))
                    if edge not in drawn_edges:
                        drawn_edges.add(edge)
                        X1, Y1 = to_visual(x, y)
                        X2, Y2 = to_visual(nx, ny)
                        inter_segments.append([(X1, Y1), (X2, Y2)])

    if inter_segments:
        lc = LineCollection(inter_segments, colors='black', linewidths=lw_inter, alpha=0.35, zorder=1)
        ax.add_collection(lc)

    # 2. The chains
    for v, nodes in chains.items():
        color = get_color(v, n)
        # chains are in path order
        for i in range(len(nodes) - 1):
            X1, Y1 = to_visual(*nodes[i])
            X2, Y2 = to_visual(*nodes[i+1])
            ax.plot([X1, X2], [Y1, Y2], color=color, linewidth=lw_intra, alpha=0.9, zorder=2)

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
    ax.margins(0.15)
    ax.set_title(f"Square Embedding $K_{{{n}}}$ on $KG_{{{n-1},{n-1}}}$", fontsize=14, weight='bold')


if __name__ == "__main__":
    test_cases = [5, 7, 9, 12, 20]

    # 1. Verification
    for n in test_cases:
        chains = build_square_triad(n)
        verify_embedding(chains, n)

    # 2. Drawing, for n = 5 and 7
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    draw_square_triad(build_square_triad(5), 5, ax=axes[0])
    draw_square_triad(build_square_triad(7), 7, ax=axes[1])

    plt.tight_layout()
    plt.show()