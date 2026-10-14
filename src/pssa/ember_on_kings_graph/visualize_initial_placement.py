"""
My visualization of the initial placement (Section 2.2 of the report), the guiding
pattern split into one super vertex per vertex of I before any annealing.

    draw_initial_placement(I, L=14)
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

from pssa import build_hardware_graph, build_guiding_pattern, initial_placement, oneD2twoD
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


def draw_initial_placement(I: nx.Graph, L: int, ax=None, show=True):
    """The initial placement of I on KG(L)."""
    H = build_hardware_graph(L)
    gp = build_guiding_pattern(H, L)
    phi = initial_placement(I, gp, L)

    nodes_I = sorted(I.nodes())
    n = len(nodes_I)
    # a fixed colour per vertex of I
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

    # 2. The super vertices as thick coloured paths
    for v, chain in phi.items():
        color = get_color(color_idx[v], n)
        for i in range(len(chain) - 1):
            r1, c1 = coord[chain[i]]
            r2, c2 = coord[chain[i + 1]]
            X1, Y1 = to_visual(r1, c1)
            X2, Y2 = to_visual(r2, c2)
            ax.plot([X1, X2], [Y1, Y2], color=color, linewidth=lw_intra,
                     alpha=0.9, zorder=2)

    # 3. Vertices and labels
    for u in H.nodes():
        v = owner.get(u)
        if v is None:
            continue
        r, c = coord[u]
        X, Y = to_visual(r, c)
        color = get_color(color_idx[v], n)
        ax.plot(X, Y, 'o', markersize=m_size, markerfacecolor=color,
                 markeredgecolor='black', markeredgewidth=edge_width, zorder=3)
        if draw_labels and f_size >= 2:
            ax.text(X, Y, str(v), ha='center', va='center',
                     fontsize=f_size, color='black', weight='bold', zorder=4)

    ax.axis('off')
    _set_fixed_kings_extent(ax, L)
    ax.set_title(f"Initial Placement: {I.number_of_nodes()} logical vertices on King's Graph $L={L}$",
                 fontsize=13, weight='bold')

    if show and should_show_at_end:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    I_demo = nx.complete_graph(9)
    draw_initial_placement(I_demo, L=8)
