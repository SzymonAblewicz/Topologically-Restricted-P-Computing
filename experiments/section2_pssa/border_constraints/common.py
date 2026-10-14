"""Shared helpers for the PSSA wBC experiments and figures of Section 2.4 of the report."""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src", "pssa", "pssa_wbc"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.collections import LineCollection
import networkx as nx

import pssa_wbc as P
import draw_embedding as V

OUT = os.path.join(ROOT, "results", "section2_pssa")
TMAX = 300_000

# Colours shared by every figure
C_UNC = "#7f7f7f"     # unconstrained PSSA
C_M1 = "#d95f02"      # Mode 1
C_M2 = "#1b9e77"      # Mode 2
C_BAD = "#d62728"  # buried border node or missing edge
C_BORDER = "#111111"  # border super vertex outline

# Figures are drawn at their printed size, so 9 pt text matches the report's text
TEXTWIDTH = 6.3
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.titleweight": "bold",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def lattice_unit(panel_width_in, L, pad=0.6):
    """Points per lattice step for a KG(L) panel of the given printed width."""
    return panel_width_in * 72.0 / (L - 1 + 2 * pad)


def connected_gnm(n, m, seed):
    """The first connected G(n, m) graph from seed upwards."""
    while not nx.is_connected(G := nx.gnm_random_graph(n, m, seed=seed)):
        seed += 1
    return G


def one_based(G):
    return nx.relabel_nodes(G, {n: n + 1 for n in G.nodes()})


def pick_border(I, k, seed, how="random"):
    nodes = sorted(I.nodes())
    if k <= 0:
        return set()
    if how == "random":
        return set(random.Random(seed).sample(nodes, k))
    by_deg = sorted(nodes, key=lambda v: (I.degree(v), v), reverse=(how == "high"))
    return set(by_deg[:k])


def run(I, L, border=None, mode="permute", seed=1, tmax=TMAX, gibbs=True):
    """One PSSA wBC run, mode None for no border constraint: (result, algo), or (None, None) if the constraint can't be met at this L."""
    kw = dict(L=L, tmax=tmax, gibbs=gibbs, seed=seed)
    if border and mode is not None:
        kw.update(border_nodes=set(border), border_mode=mode)
    algo = P.ImprovedPSSA(**kw)
    try:
        res = algo.run(I)
    except ValueError:
        return None, None
    return res, algo


def perimeter(L):
    return P._perimeter_nodes(L)


def touches(chain, L):
    per = perimeter(L)
    return [u for u in chain if u in per]


def rc(u, L):
    return P.oneD2twoD(u, L)


def xy(u, L):
    r, c = rc(u, L)
    return c, -r


def draw_placement(ax, phi, I, L, border=(), highlight=(), bad=(), rings=(),
                   title=None, labels=True, pad=0.6, unit=None):
    """Draw a placement on KG(L), with the border super vertices outlined and given hardware vertices haloed or ringed."""
    H = P.build_hardware_graph(L)
    nodes_I = sorted(I.nodes())
    n = len(nodes_I)
    cidx = {v: i + 1 for i, v in enumerate(nodes_I)}
    if unit is None:
        fig = ax.figure
        bb = ax.get_position()
        w = bb.width * fig.get_figwidth() * 72.0
        h = bb.height * fig.get_figheight() * 72.0
        unit = min(w, h) / (L - 1 + 2 * pad)
    m_size = 0.55 * unit
    f_size = 0.30 * unit
    lw_intra = 0.16 * unit
    lw_mesh = 0.03 * unit
    ring_w = max(0.8, 0.07 * unit)

    segs = [[xy(u, L), xy(v, L)] for u, v in H.edges()]
    ax.add_collection(LineCollection(segs, colors="black", linewidths=lw_mesh,
                                     alpha=0.25, zorder=1))
    for v, chain in phi.items():
        col = V.get_color(cidx[v], n)
        cs = set(chain)
        for u in chain:
            for w in H.neighbors(u):
                if w in cs and u < w:
                    (x1, y1), (x2, y2) = xy(u, L), xy(w, L)
                    ax.plot([x1, x2], [y1, y2], color=col, lw=lw_intra,
                            alpha=0.9, zorder=2, solid_capstyle="round")
    halo = set(highlight)
    for v, chain in phi.items():
        col = V.get_color(cidx[v], n)
        edge_col, edge_w = "black", 0.02 * unit
        if v in bad:
            edge_col, edge_w = C_BAD, ring_w * 1.2
        elif v in border:
            edge_col, edge_w = C_BORDER, ring_w
        for u in chain:
            x, y = xy(u, L)
            if u in halo:
                ax.plot(x, y, "o", ms=m_size * 1.9, mfc="#ffd400", mec="none",
                        alpha=0.85, zorder=2.5)
            ax.plot(x, y, "o", ms=m_size, mfc=col, mec=edge_col, mew=edge_w, zorder=3)
            if labels and f_size >= 4:
                ax.text(x, y, str(v), ha="center", va="center", fontsize=f_size,
                        weight="bold", zorder=4)
    for u in rings:
        x, y = xy(u, L)
        ax.plot(x, y, "o", ms=m_size * 1.55, mfc="none", mec=C_BORDER,
                mew=ring_w, zorder=5)
    ax.set_xlim(-pad, L - 1 + pad)
    ax.set_ylim(-(L - 1 + pad), pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def outward(u, L):
    """The unit direction off the lattice from a perimeter vertex."""
    r, c = rc(u, L)
    dx = -1 if c == 0 else (1 if c == L - 1 else 0)
    dy = 1 if r == 0 else (-1 if r == L - 1 else 0)
    return dx, dy


def draw_exit_arrows(ax, points, L, color=C_BORDER, length=0.75, lw=1.2):
    for u in points:
        x, y = xy(u, L)
        dx, dy = outward(u, L)
        norm = (dx * dx + dy * dy) ** 0.5 or 1
        ax.annotate("", xy=(x + length * dx / norm, y + length * dy / norm),
                    xytext=(x + 0.28 * dx / norm, y + 0.28 * dy / norm),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=8),
                    zorder=6)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")    # vector copy: text stays sharp in the report
    plt.close(fig)
    print("saved", path, flush=True)


def draw_missing(ax, phi, I, L):
    """A red dashed line for every edge of I not embedded."""
    H = P.build_hardware_graph(L)
    n_missing = 0
    for i, j in I.edges():
        ci, cj = phi.get(i, []), phi.get(j, [])
        if not ci or not cj:
            continue
        if any(H.has_edge(u, w) for u in ci for w in cj):
            continue
        n_missing += 1
        best = min(((u, w) for u in ci for w in cj),
                   key=lambda p: sum((a - b) ** 2 for a, b in zip(rc(p[0], L), rc(p[1], L))))
        (x1, y1), (x2, y2) = xy(best[0], L), xy(best[1], L)
        ax.plot([x1, x2], [y1, y2], color=C_BAD, lw=1.1, ls="--", alpha=0.9, zorder=1.6)
    return n_missing


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Drawing at native size with my own visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NATIVE_PANEL = 6.5  # inches per lattice panel


def native_fig(n_panels, height_ratio=1.12):
    """A figure at the size of my visualization, (fig, axes, s) with s = printed width / native width."""
    w = NATIVE_PANEL * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(w, NATIVE_PANEL * height_ratio))
    return fig, axes, TEXTWIDTH / w


def own_final(ax, phi, I, H, L, s, title, rings=(), success=True):
    """My draw_final_result, with a title at print size."""
    with plt.rc_context({"font.family": "sans-serif"}):
        V.draw_final_result(phi, I, H, L, success=success, ax=ax, show=False,
                            anchor_hw_nodes=set(rings))
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_title(title, fontsize=9 / s)


def own_initial(ax, I, L, s, title, border_nodes=None):
    """My draw_initial_placement, with a title at print size."""
    with plt.rc_context({"font.family": "sans-serif"}):
        V.draw_initial_placement(I, L, ax=ax, show=False, border_nodes=border_nodes)
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_title(title, fontsize=9 / s)


def own_ring(ax, hw_nodes, L, color=C_BAD):
    """A ring in the style of the anchor rings, in another colour."""
    m_size = max(1.0, 70.0 / L)
    edge_width = max(0.1, 4.0 / L)
    for u in hw_nodes:
        x, y = xy(u, L)
        ax.plot(x, y, "o", ms=m_size * 1.8, mfc="none", mec=color,
                mew=max(1.5, edge_width * 2.5), zorder=5)


def own_halo(ax, hw_nodes, L):
    m_size = max(1.0, 70.0 / L)
    for u in hw_nodes:
        x, y = xy(u, L)
        ax.plot(x, y, "o", ms=m_size * 1.9, mfc="#ffd400", mec="none", alpha=0.85, zorder=2.5)


def own_arrows(ax, points, L, length=0.8):
    for u in points:
        x, y = xy(u, L)
        dx, dy = outward(u, L)
        n = (dx * dx + dy * dy) ** 0.5 or 1
        ax.annotate("", xy=(x + length * dx / n, y + length * dy / n),
                    xytext=(x + 0.3 * dx / n, y + 0.3 * dy / n),
                    arrowprops=dict(arrowstyle="-|>", color=C_BORDER, lw=2.5, mutation_scale=22),
                    zorder=6)


def own_extent(ax, L, pad):
    ax.set_xlim(-pad, L - 1 + pad)
    ax.set_ylim(-(L - 1 + pad), pad)


def legend_handles(s, items):
    """Legend entries matching the markers of my visualization, at print size."""
    import matplotlib.lines as mlines
    out = []
    for kind, label in items:
        if kind == "embedded":
            out.append(mlines.Line2D([], [], color="#020702", lw=2 / s * 0.45, label=label))
        elif kind == "missing":
            out.append(mlines.Line2D([], [], color="#d62728", lw=2 / s * 0.45, ls="--", label=label))
        elif kind == "ring":
            out.append(mlines.Line2D([], [], marker="o", color="none", mfc="none", mec="black",
                                     mew=2 / s * 0.45, ms=10 / s * 0.45, label=label))
        elif kind == "redring":
            out.append(mlines.Line2D([], [], marker="o", color="none", mfc="none", mec=C_BAD,
                                     mew=2 / s * 0.45, ms=10 / s * 0.45, label=label))
        elif kind == "border":
            out.append(mlines.Line2D([], [], marker="o", color="none", mfc="white", mec=C_BORDER,
                                     mew=2.2 / s * 0.45, ms=8 / s * 0.45, label=label))
        elif kind == "bad":
            out.append(mlines.Line2D([], [], marker="o", color="none", mfc="white", mec=C_BAD,
                                     mew=2.6 / s * 0.45, ms=8 / s * 0.45, label=label))
        elif kind == "halo":
            out.append(mlines.Line2D([], [], marker="o", color="none", mfc="#ffd400", mec="none",
                                     alpha=0.85, ms=12 / s * 0.45, label=label))
        elif kind == "arrow":
            out.append(_ArrowHandle(label=label))
    return out


class _ArrowHandle(mlines.Line2D):
    """A legend entry drawn as the arrow of own_arrows."""
    def __init__(self, label):
        super().__init__([], [], label=label)


class _ArrowHandler(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        from matplotlib.patches import FancyArrowPatch
        y = yd + height / 2
        return [FancyArrowPatch((xd, y), (xd + width, y), arrowstyle="-|>", color=C_BORDER,
                                lw=2.5, mutation_scale=22, transform=trans)]


Legend.update_default_handler_map({_ArrowHandle: _ArrowHandler()})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Report font: Latin Modern
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from matplotlib import font_manager as _fm

_LM = r"C:\texlive\2025\texmf-dist\fonts\opentype\public\lm"
for _f in ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf",
           "lmroman10-bolditalic.otf"):
    _p = os.path.join(_LM, _f)
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titleweight": "normal",
    "axes.unicode_minus": False,
})


def draw_first_look(ax, phi, I, L, s, title, border=(), highlight=(), bad=(), rings=(), edges=False, labels=True):
    """The layout of the first version of the figures, at native panel size."""
    H = P.build_hardware_graph(L)
    nodes_I = sorted(I.nodes())
    n = len(nodes_I)
    cidx = {v: i + 1 for i, v in enumerate(nodes_I)}
    m_size = max(1.0, 70.0 / L)
    f_size = max(1.0, 40.0 / L)
    lw_intra = max(0.3, 20.0 / L)
    lw_mesh = max(0.05, 8.0 / L)
    edge_width = max(0.1, 4.0 / L)
    segs = [[xy(u, L), xy(v, L)] for u, v in H.edges()]
    ax.add_collection(LineCollection(segs, colors="black", linewidths=lw_mesh, alpha=0.25, zorder=1))
    for v, chain in phi.items():
        col = V.get_color(cidx[v], n)
        cs = set(chain)
        for u in chain:
            for w in H.neighbors(u):
                if w in cs and u < w:
                    (x1, y1), (x2, y2) = xy(u, L), xy(w, L)
                    ax.plot([x1, x2], [y1, y2], color=col, lw=lw_intra, alpha=0.9, zorder=2)
    halo = set(highlight)
    for v, chain in phi.items():
        col = V.get_color(cidx[v], n)
        ec, ew = "black", edge_width
        if v in bad:
            ec, ew = C_BAD, 2.6
        elif v in border:
            ec, ew = C_BORDER, 2.2
        for u in chain:
            x, y = xy(u, L)
            if u in halo:
                ax.plot(x, y, "o", ms=m_size * 1.9, mfc="#ffd400", mec="none", alpha=0.85, zorder=2.5)
            ax.plot(x, y, "o", ms=m_size, mfc=col, mec=ec, mew=ew, zorder=3)
            if labels and f_size >= 2:
                ax.text(x, y, str(v), ha="center", va="center", fontsize=f_size,
                        weight="bold", zorder=4, family="DejaVu Sans")
    for u in rings:
        x, y = xy(u, L)
        ax.plot(x, y, "o", ms=m_size * 1.8, mfc="none", mec=C_BORDER, mew=2.0, zorder=5)
    # embedded input edges, final embeddings only, in draw_final_result's style
    lw_logic = max(0.4, 6.0 / L) * 3
    for i, j in (I.edges() if edges else ()):
        pair = next(((u, w) for u in phi.get(i, []) for w in phi.get(j, []) if H.has_edge(u, w)), None)
        if pair:
            (x1, y1), (x2, y2) = xy(pair[0], L), xy(pair[1], L)
            ax.plot([x1, x2], [y1, y2], color="#000000", lw=lw_logic, alpha=0.7, zorder=1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9 / s)
