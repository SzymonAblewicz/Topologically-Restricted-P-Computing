"""
My visualization of a PSSA wBC embedding (Section 2.2 of the report), with the colours
of the Section 2 figures. It draws the King's graph as a faint lattice and each super
vertex as a thick coloured path through it, with the border anchors ringed.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection

from pssa_wbc import (
    build_hardware_graph,
    build_guiding_pattern,
    initial_placement,
    oneD2twoD,
    assign_border_anchors,
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Shared
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _set_fixed_kings_extent(ax, L: int, pad: float = 0.5):
    """Fix the axes to the whole L x L lattice, row 0 at the top, so every figure has the same scale."""
    ax.set_xlim(-pad, (L - 1) + pad)
    ax.set_ylim(-((L - 1) + pad), pad)
    ax.set_aspect('equal')


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bipolar_degrees=True rings the top half of the vertices by degree red and the bottom half green
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BIPOLAR_HIGH_COLOR = '#FF0000'   # top half by degree
BIPOLAR_LOW_COLOR = '#00C800'    # bottom half by degree

# border anchors get a black ring, outside the degree ring
ANCHOR_RING_COLOR = '#000000'


def _bipolar_degree_roles(I: nx.Graph):
    """'high' for the top half of the vertices by degree, 'low' for the bottom half."""
    ranked = sorted(I.nodes(), key=lambda v: (-I.degree(v), v))
    n = len(ranked)
    half = n // 2
    roles = {v: None for v in ranked}
    for v in ranked[:half]:
        roles[v] = 'high'
    for v in ranked[n - half:]:
        roles[v] = 'low'
    return roles


def _bipolar_ring_geometry(marker_size):
    """Size and line width of the degree ring for a marker size."""
    ring_size = marker_size * 1.45
    ring_width = max(2.2, marker_size * 0.22)
    return ring_size, ring_width


def _draw_bipolar_border(ax, X, Y, role, marker_size, zorder=3.5):
    """A hollow ring around a vertex marker."""
    if role is None:
        return
    color = BIPOLAR_HIGH_COLOR if role == 'high' else BIPOLAR_LOW_COLOR
    ring_size, ring_width = _bipolar_ring_geometry(marker_size)
    ax.plot(X, Y, 'o', markersize=ring_size, markerfacecolor='none',
             markeredgecolor=color, markeredgewidth=ring_width, zorder=zorder)


def _bipolar_legend_handles():
    return [
        mlines.Line2D([], [], marker='o', color='none', markerfacecolor='none',
                       markeredgecolor=BIPOLAR_HIGH_COLOR, markeredgewidth=2.5, markersize=10,
                       label='high degree (top half)'),
        mlines.Line2D([], [], marker='o', color='none', markerfacecolor='none',
                       markeredgecolor=BIPOLAR_LOW_COLOR, markeredgewidth=2.5, markersize=10,
                       label='low degree (bottom half)'),
    ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Guiding pattern
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_guiding_pattern(L: int = 8, ax=None, show=True):
    """The guiding pattern on KG(L), the complete graph embedding of K_{L+1} of Okuyama et al. (2016)."""
    H = build_hardware_graph(L)
    gp = build_guiding_pattern(H, L)
    n = L + 1  # the K_n being embedded

    # hardware vertex -> (row, col), and -> its super vertex
    coord = {u: oneD2twoD(u, L) for u in H.nodes()}
    owner = {u: sv for sv, chain in gp.items() for u in chain}

    # show the figure only if this call made it
    should_show_at_end = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        should_show_at_end = True

    def to_visual(row, col):
        # row 0 at the top
        return col, -row

    # sizes set by L
    m_size = max(1.0, 70.0 / L)
    f_size = max(1.0, 40.0 / L)
    lw_intra = max(0.3, 20.0 / L)
    lw_inter = max(0.05, 8.0 / L)
    lw_logic_base = max(0.4, 6.0 / L)
    draw_labels = n <= 1000
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
    for sv, chain in gp.items():
        color = get_color(sv, n)
        for i in range(len(chain) - 1):
            r1, c1 = coord[chain[i]]
            r2, c2 = coord[chain[i + 1]]
            X1, Y1 = to_visual(r1, c1)
            X2, Y2 = to_visual(r2, c2)
            ax.plot([X1, X2], [Y1, Y2], color=color, linewidth=lw_intra,
                     alpha=0.9, zorder=2)

    # 3. Vertices and labels
    for u in H.nodes():
        sv = owner.get(u)
        if sv is None:
            continue
        r, c = coord[u]
        X, Y = to_visual(r, c)
        color = get_color(sv, n)
        ax.plot(X, Y, 'o', markersize=m_size, markerfacecolor=color,
                 markeredgecolor='black', markeredgewidth=edge_width, zorder=3)
        if draw_labels and f_size >= 2:
            ax.text(X, Y, str(sv), ha='center', va='center',
                     fontsize=f_size, color='black', weight='bold', zorder=4)

    ax.axis('off')
    _set_fixed_kings_extent(ax, L)
    ax.set_title(f"Guiding Pattern $K_{{{n}}}$ on King's Graph $L={L}$ ({L*L} qubits)",
                 fontsize=13, weight='bold')

    if show and should_show_at_end:
        plt.tight_layout()
        plt.show()

    return ax


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initial placement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_initial_placement(I: nx.Graph, L: int, ax=None, show=True, border_nodes=None,
                            bipolar_degrees=False):
    """The initial placement of I on KG(L), with the anchors of the border nodes if given."""
    H = build_hardware_graph(L)
    gp = build_guiding_pattern(H, L)
    phi = initial_placement(I, gp, L)
    bipolar_roles = _bipolar_degree_roles(I) if bipolar_degrees else {}

    anchor_hw_nodes = set()
    if border_nodes:
        H_adj = {u: list(H.neighbors(u)) for u in H.nodes()}
        anchors = assign_border_anchors(phi, set(border_nodes), L, H_adj)
        anchor_hw_nodes = set(anchors.values())

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
    draw_labels = n <= 1000
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
        if bipolar_degrees:
            _draw_bipolar_border(ax, X, Y, bipolar_roles.get(v), m_size)
        if draw_labels and f_size >= 2:
            ax.text(X, Y, str(v), ha='center', va='center',
                     fontsize=f_size, color='black', weight='bold', zorder=4)

    # 3b. Anchor rings
    if anchor_hw_nodes:
        if bipolar_degrees:
            bp_ring_size, bp_ring_width = _bipolar_ring_geometry(m_size)
            anchor_edge_width = bp_ring_width
            anchor_size = bp_ring_size + 2 * bp_ring_width
        else:
            anchor_size = m_size * 1.8
            anchor_edge_width = max(1.5, edge_width * 2.5)
        for u in anchor_hw_nodes:
            r, c = coord[u]
            X, Y = to_visual(r, c)
            ax.plot(X, Y, 'o', markersize=anchor_size, markerfacecolor='none',
                     markeredgecolor=ANCHOR_RING_COLOR, markeredgewidth=anchor_edge_width, zorder=5)

    ax.axis('off')
    _set_fixed_kings_extent(ax, L)
    ax.set_title(f"Initial Placement: {I.number_of_nodes()} logical vertices on King's Graph $L={L}$",
                 fontsize=13, weight='bold')

    legend_handles = []
    if anchor_hw_nodes:
        legend_handles.append(mlines.Line2D([], [], marker='o', color='none',
                                              markerfacecolor='none', markeredgecolor=ANCHOR_RING_COLOR,
                                              markeredgewidth=2, markersize=10,
                                              label='border anchor (fixed)'))
    if bipolar_degrees:
        legend_handles.extend(_bipolar_legend_handles())
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)

    if show and should_show_at_end:
        plt.tight_layout()
        plt.show()

    return ax


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Final embedding
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


def _closest_pair_coords(chain_i, chain_j, coord, to_visual):
    """The closest pair of hardware vertices between two super vertices."""
    if not chain_i or not chain_j:
        return (0, 0), (0, 0)
    min_dist_sq = float('inf')
    best_pair = (chain_i[0], chain_j[0])
    for u in chain_i:
        r1, c1 = coord[u]
        for w in chain_j:
            r2, c2 = coord[w]
            dist_sq = (r1 - r2)**2 + (c1 - c2)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_pair = (u, w)
    return to_visual(*coord[best_pair[0]]), to_visual(*coord[best_pair[1]])


def draw_final_result(phi, I: nx.Graph, H: nx.Graph, L: int, success=None,
                       ax=None, show=True, anchor_hw_nodes=None,
                       bipolar_degrees=False):
    """The final embedding on KG(L), with the embedded edges in black and the missing ones dashed red."""
    anchor_hw_nodes = set(anchor_hw_nodes) if anchor_hw_nodes else set()
    nodes_I = sorted(I.nodes())
    n = len(nodes_I)
    color_idx = {v: idx + 1 for idx, v in enumerate(nodes_I)}
    bipolar_roles = _bipolar_degree_roles(I) if bipolar_degrees else {}

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
    draw_labels = n <= 1000
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
        if bipolar_degrees:
            _draw_bipolar_border(ax, X, Y, bipolar_roles.get(v), m_size)
        if draw_labels and f_size >= 2:
            ax.text(X, Y, str(v), ha='center', va='center',
                     fontsize=f_size, color='black', weight='bold', zorder=4)

    # 3b. Anchor rings
    if anchor_hw_nodes:
        if bipolar_degrees:
            bp_ring_size, bp_ring_width = _bipolar_ring_geometry(m_size)
            anchor_edge_width = bp_ring_width
            anchor_size = bp_ring_size + 2 * bp_ring_width
        else:
            anchor_size = m_size * 1.8
            anchor_edge_width = max(1.5, edge_width * 2.5)
        for u in anchor_hw_nodes:
            r, c = coord[u]
            X, Y = to_visual(r, c)
            ax.plot(X, Y, 'o', markersize=anchor_size, markerfacecolor='none',
                     markeredgecolor=ANCHOR_RING_COLOR, markeredgewidth=anchor_edge_width, zorder=5)

    # 4. The edges of I
    lw_logical = max(0.4, 6.0 / L)
    any_missing = False
    for i, j in I.edges():
        chain_i = phi.get(i, [])
        chain_j = phi.get(j, [])
        if not chain_i or not chain_j: continue

        embedded = any(H.has_edge(u, w) for u in chain_i for w in chain_j)

        if embedded:
            (Xi, Yi), (Xj, Yj) = _adjacent_pair_or_centroids(chain_i, chain_j, H, coord, to_visual)
            ax.plot([Xi, Xj], [Yi, Yj], color="#000000", linewidth=lw_logical * 3,
                     alpha=0.7, zorder=1.5, linestyle='-')
        else:
            any_missing = True
            # missing, drawn between the closest hardware vertices
            (Xi, Yi), (Xj, Yj) = _closest_pair_coords(chain_i, chain_j, coord, to_visual)
            ax.plot([Xi, Xj], [Yi, Yj], color='#d62728', linewidth=lw_logical * 3,
                     alpha=0.9, zorder=1.6, linestyle='--')

    legend_handles = [
        mlines.Line2D([], [], color="#020702", linewidth=2, label='embedded edge'),
        mlines.Line2D([], [], color='#d62728', linewidth=2, linestyle='--', label='missing edge'),
    ]
    if anchor_hw_nodes:
        legend_handles.append(
            mlines.Line2D([], [], marker='o', color='none', markerfacecolor='none',
                           markeredgecolor=ANCHOR_RING_COLOR, markeredgewidth=2, markersize=10,
                           label='border anchor (fixed)')
        )
    if bipolar_degrees:
        legend_handles.extend(_bipolar_legend_handles())
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Node degree list, for a panel under the three drawings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_node_degree_list(ax, I: nx.Graph, ncols: int = 5, bipolar_degrees=False):
    """Every vertex of I with its degree, drawn with the same coloured markers as the embedding."""
    nodes_I = sorted(I.nodes())
    n = len(nodes_I)
    color_idx = {v: idx + 1 for idx, v in enumerate(nodes_I)}
    degrees = dict(I.degree())
    bipolar_roles = _bipolar_degree_roles(I) if bipolar_degrees else {}

    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceil division

    ax.set_xlim(0, ncols)
    ax.set_ylim(-nrows, 0)
    ax.set_aspect('auto')
    ax.axis('off')

    marker_size = 22
    node_font_size = 9
    label_font_size = 11

    for i, v in enumerate(nodes_I):
        row, col = divmod(i, ncols)
        cy = -(row + 0.5)
        x_prefix = col + 0.05
        x_marker = col + 0.30
        x_suffix = col + 0.50

        color = get_color(color_idx[v], n)

        ax.text(x_prefix, cy, "deg(", ha='left', va='center',
                 fontsize=label_font_size, family='monospace', zorder=4)
        ax.plot(x_marker, cy, 'o', markersize=marker_size,
                 markerfacecolor=color, markeredgecolor='black',
                 markeredgewidth=1.0, zorder=3, clip_on=False)
        if bipolar_degrees:
            _draw_bipolar_border(ax, x_marker, cy, bipolar_roles.get(v),
                                     marker_size, zorder=3.5)
        ax.text(x_marker, cy, str(v), ha='center', va='center',
                 fontsize=node_font_size, color='black', weight='bold', zorder=5)
        ax.text(x_suffix, cy, f") = {degrees[v]}", ha='left', va='center',
                 fontsize=label_font_size, family='monospace', zorder=4)

    ax.set_title("Input Graph Node Degrees", fontsize=13, weight='bold')
    return ax


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Demos
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    # the guiding pattern at two sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_guiding_pattern(L=4, ax=axes[0], show=False)
    draw_guiding_pattern(L=8, ax=axes[1], show=False)
    plt.tight_layout()
    plt.show()

    # the initial placement of K_9 on L = 8
    I_demo = nx.complete_graph(9)
    draw_initial_placement(I_demo, L=8)