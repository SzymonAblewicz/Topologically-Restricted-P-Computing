"""
My visualization of the final embedding (Section 2.2 of the report), with the edges of
I drawn black if embedded and dashed red if missing.

    draw_final_result(result.phi, I, algo.H, L=8, success=result.success)
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection

from pssa import oneD2twoD
from kings_viz_common import _set_fixed_kings_extent


def get_color(v, n: int):
    palette = {
        1: '#F59D56', 2: '#C5E063', 3: '#71C754', 4: '#62D99A',
        5: '#60C4E6', 6: '#4960D8', 7: '#7A4CE6', 8: '#D849D8',
    }
    if n <= 8:
        return palette.get(v)

    # the colours on 0.15-0.9 of the colormap, without its darkest and faintest ends
    low, high = 0.15, 0.9
    scaled_v = low + (v / max(n - 1, 1)) * (high - low)
    return cm.turbo(scaled_v)


def _chain_centroid(chain, coord, to_visual):
    xs, ys = [], []
    for u in chain:
        r, c = coord[u]
        X, Y = to_visual(r, c)
        xs.append(X)
        ys.append(Y)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _adjacent_pair_or_centroids(chain_i, chain_j, H, coord, to_visual):
    """One adjacent pair of hardware vertices between two super vertices, so an edge is drawn along the lattice."""
    for u in chain_i:
        for w in H.neighbors(u):
            if w in chain_j:
                r1, c1 = coord[u]
                r2, c2 = coord[w]
                return to_visual(r1, c1), to_visual(r2, c2)
    return (_chain_centroid(chain_i, coord, to_visual),
            _chain_centroid(chain_j, coord, to_visual))


def draw_final_result(phi, I: nx.Graph, H: nx.Graph, L: int, success=None,
                       ax=None, show=True):
    """The final embedding on KG(L)."""
    nodes_I = sorted(I.nodes())
    n = len(nodes_I)
    color_idx = {v: idx + 1 for idx, v in enumerate(nodes_I)}

    coord = {u: oneD2twoD(u, L) for u in H.nodes()}
    owner = {u: v for v, chain in phi.items() for u in chain}

    should_show_at_end = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        should_show_at_end = True

    def to_visual(row, col):
        return col, -row

    # sizes set by L
    m_size = max(1.0, 70.0 / L)
    f_size = max(1.0, 40.0 / L)
    lw_intra = max(0.3, 20.0 / L)
    lw_inter = max(0.05, 8.0 / L)
    lw_logic_base = max(0.4, 6.0 / L) 
    draw_labels = n <= 25
    edge_width = max(0.1, 4.0 / L) if L <= 40 else 0

    # 1. The faint lattice
    inter_segments = []
    for u, v in H.edges():
        r1, c1 = coord[u]
        r2, c2 = coord[v]
        X1, Y1 = to_visual(r1, c1)
        X2, Y2 = to_visual(r2, c2)
        inter_segments.append([(X1, Y1), (X2, Y2)])
    if inter_segments:
        lc = LineCollection(inter_segments, colors='black', linewidths=lw_inter,
                             alpha=0.25, zorder=1)
        ax.add_collection(lc)

    # 2. The super vertices, every adjacent pair drawn, as after the terminal search
    #    a super vertex need not be a path in list order
    for v, chain in phi.items():
        color = get_color(color_idx[v], n)
        chain_set = set(chain)
        drawn = set()
        for u in chain:
            for w in H.neighbors(u):
                if w in chain_set and w != u:
                    edge = tuple(sorted((u, w)))
                    if edge in drawn:
                        continue
                    drawn.add(edge)
                    r1, c1 = coord[u]
                    r2, c2 = coord[w]
                    X1, Y1 = to_visual(r1, c1)
                    X2, Y2 = to_visual(r2, c2)
                    ax.plot([X1, X2], [Y1, Y2], color=color, linewidth=lw_intra,
                             alpha=0.9, zorder=2)

    # 3. Vertices and labels, only those phi uses
    for u, v in owner.items():
        r, c = coord[u]
        X, Y = to_visual(r, c)
        color = get_color(color_idx[v], n)
        ax.plot(X, Y, 'o', markersize=m_size, markerfacecolor=color,
                 markeredgecolor='black', markeredgewidth=edge_width, zorder=3)
        if draw_labels and f_size >= 2:
            ax.text(X, Y, str(v), ha='center', va='center',
                     fontsize=f_size, color='black', weight='bold', zorder=4)

    # 4. The edges of I, black if embedded, red if missing
    lw_logical = max(0.4, 6.0 / L)
    any_missing = False
    for i, j in I.edges():
        chain_i = phi.get(i, [])
        chain_j = phi.get(j, [])
        (Xi, Yi), (Xj, Yj) = _adjacent_pair_or_centroids(chain_i, chain_j, H, coord, to_visual)
        embedded = any(H.has_edge(u, w) for u in chain_i for w in chain_j)
        if embedded:
            ax.plot([Xi, Xj], [Yi, Yj], color="#000000", linewidth=lw_logical * 3,
                     alpha=0.7, zorder=1.5, linestyle='-')
        else:
            any_missing = True
            ax.plot([Xi, Xj], [Yi, Yj], color='#d62728', linewidth=lw_logical * 3,
                     alpha=0.9, zorder=1.6, linestyle='--')

    legend_handles = [
        mlines.Line2D([], [], color="#020702", linewidth=2, label='embedded edge'),
        mlines.Line2D([], [], color='#d62728', linewidth=2, linestyle='--', label='missing edge'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)

    ax.axis('off')
    _set_fixed_kings_extent(ax, L)

    used_qubits = sum(len(chain) for chain in phi.values())
    total_qubits = L * L
    status = 'SUCCESS' if success else 'INCOMPLETE'
    ax.set_title(f"Final Embedding ({status}): {used_qubits}/{total_qubits} qubits used",
                 fontsize=13, weight='bold')

    if show and should_show_at_end:
        plt.tight_layout()
        plt.show()

    return ax