"""
Drawings of large fallback embeddings. Below mesh_limit the embedding is drawn
element by element as in cluster_embedding.py, and above it every occupied hardware
vertex is binned to a pixel, with the lanes drawn over the patches so that a lane one
vertex wide still shows.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from kings_lattice import rc_of
from pssa_wbc import build_hardware_graph, oneD2twoD
from draw_embedding import get_color
from cluster_embedding import (
    _node_colors,
    _shade,
    draw_compact_embedding,
    draw_exploded_clusters,
)

Node = Any

# Below this L every element is drawn.
MESH_LIMIT = 1100

# Above this many clusters the captions are left out.
LABEL_LIMIT = 120

# Above this many clusters the patch lattices and perimeter rings are left out.
MESH_LIMIT_K = 300

# Codes in the raster; higher wins the pixel.
_EMPTY = 0
_PATCH = 1          # + cluster index
_ROUTE = 1 << 20  # the lanes


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The whole embedding
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_compact_embedding_fast(
    G, partition, result, ax=None, show: bool = True,
    title: Optional[str] = None, mesh_limit: int = MESH_LIMIT,
    max_px: int = 2400,
):
    """cluster_embedding.draw_compact_embedding, rasterized above mesh_limit."""
    L = result.L
    if L <= mesh_limit:
        return draw_compact_embedding(G, partition, result, ax=ax, show=show,
                                      title=title)

    should_show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))

    k = partition.k
    px = _target_px(ax, L, max_px)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # route cells, the lanes appended to the super vertices
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    route_cells = [np.asarray(p, dtype=np.int64).ravel()
                   for p in result.routes.values()]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # rasterize
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img = np.zeros((px, px), dtype=np.int32)
    cluster_index = {cid: i for i, cid in enumerate(sorted(partition.communities))}

    by_cluster: Dict[int, List[np.ndarray]] = {}
    for v, chain in result.phi.items():
        ids = np.asarray(chain, dtype=np.int64).ravel()
        if ids.size:
            by_cluster.setdefault(result.cluster_of.get(v, -1), []).append(ids)
    for cid in sorted(by_cluster):                  # sorted -> deterministic
        _scatter(img, np.concatenate(by_cluster[cid]), L, px,
                 _PATCH + cluster_index.get(cid, 0))

    if route_cells:
        _scatter(img, np.concatenate(route_cells), L, px, _ROUTE)

    ax.imshow(_colorize(img, partition, k, cluster_index),
              extent=(-0.5, L - 0.5, -(L - 0.5), 0.5),
              interpolation="nearest", zorder=1, aspect="equal")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # vector overlay
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for p in result.placements:
        base = get_color(p.cid + 1, k)
        ax.add_patch(Rectangle(
            (p.col0 - 0.5, -(p.row0 + p.L - 1) - 0.5), p.L, p.L,
            facecolor="none", edgecolor=_shade(base, 0.35),
            linewidth=0.6, zorder=2))
    if len(result.placements) <= LABEL_LIMIT:
        for p in result.placements:
            base = get_color(p.cid + 1, k)
            ax.text(p.col0 + p.L / 2.0 - 0.5, -(p.row0 - 0.9),
                    f"c{p.cid}", ha="center", va="bottom", fontsize=9,
                    weight="bold", color=_shade(base, -0.4), zorder=3)

    _frame(ax, result, L)
    ax.legend(handles=[
        mlines.Line2D([], [], color="#12263a", lw=2,
                      label="inter-cluster routed corridor"),
    ], loc="upper right", fontsize=11, framealpha=0.92)

    status = "VALID" if result.valid else "INVALID"
    ax.set_title(
        title or (f"Stage 3 — compact embedding  [{status}]  "
                  f"{result.eemb}/{result.m_total} edges,  "
                  f"{result.qubits}/{L*L} qubits ({result.utilization:.0%}),  L={L}"),
        fontsize=12, weight="bold", pad=10)

    if show and should_show:
        plt.tight_layout()
        plt.show()
    return ax


def _target_px(ax, L: int, max_px: int) -> int:
    """The side of the raster in pixels, twice what the axes show, at most max_px."""
    want = max_px
    try:
        fig = ax.get_figure()
        w, h = fig.get_size_inches()
        want = int(max(w, h) * fig.dpi)
    except Exception:
        pass
    return max(256, min(L, max_px, max(want, 256) * 2))


def _scatter(img: np.ndarray, ids: np.ndarray, L: int, px: int, code: int) -> None:
    """Bin hardware vertices to pixels and stamp code into each, the later write winning."""
    rows, cols = rc_of(ids, L)
    img[(rows * px) // L, (cols * px) // L] = code


def _colorize(img: np.ndarray, partition, k: int, cluster_index) -> np.ndarray:
    """Codes -> RGB, in the cluster colours."""
    h, w = img.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)      # white background

    is_route = img >= _ROUTE
    is_patch = (img >= _PATCH) & ~is_route

    if is_patch.any():
        idx = (img[is_patch] - _PATCH).astype(np.int64)
        lut = np.array(
            [get_color(cid + 1, k) for cid in sorted(partition.communities)],
            dtype=np.float32)
        if lut.size == 0:
            lut = np.zeros((1, 3), dtype=np.float32)
        rgb[is_patch] = lut[np.clip(idx, 0, len(lut) - 1)]

    rgb[is_route] = np.array((0.071, 0.149, 0.227), dtype=np.float32)  # #12263a
    return rgb


def _frame(ax, result, L: int) -> None:
    """Centre the occupied region in a square view."""
    r0 = c0 = None
    r1 = c1 = -1
    for chain in result.phi.values():
        ids = np.asarray(chain, dtype=np.int64).ravel()
        if ids.size == 0:
            continue
        r, c = rc_of(ids, L)
        lo_r, hi_r, lo_c, hi_c = int(r.min()), int(r.max()), int(c.min()), int(c.max())
        r0 = lo_r if r0 is None else min(r0, lo_r)
        c0 = lo_c if c0 is None else min(c0, lo_c)
        r1, c1 = max(r1, hi_r), max(c1, hi_c)
    if r0 is None:
        r0 = c0 = 0
        r1 = c1 = L - 1
    pad_r = (L - (r1 - r0 + 1)) / 2.0
    pad_c = (L - (c1 - c0 + 1)) / 2.0
    ax.set_xlim(c0 - 0.5 - pad_c, c1 + 0.5 + pad_c)
    ax.set_ylim(-(r1 + 0.5 + pad_r), -(r0 - 0.5 - pad_r))
    ax.set_aspect("equal")
    ax.axis("off")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every cluster on its own patch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_cached_hardware_graph = lru_cache(maxsize=64)(build_hardware_graph)


def draw_exploded_clusters_fast(
    partition, embeddings, ax=None, show: bool = True,
    title: Optional[str] = None, cluster_limit: int = 60,
    max_chords: int = 2500, seed: int = 0,
):
    """cluster_embedding.draw_exploded_clusters with batched artists above cluster_limit, and a sample of the cut edges past max_chords."""
    if partition.k <= cluster_limit:
        return draw_exploded_clusters(partition, embeddings, ax=ax, show=show,
                                      title=title)

    should_show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))

    k = partition.k
    cids = sorted(embeddings)
    node_color = _node_colors(partition, k)
    origin = _ring_origins(partition, embeddings, cids)

    # heavy detail only while it can still be seen
    draw_mesh = len(cids) <= MESH_LIMIT_K
    draw_rings = len(cids) <= MESH_LIMIT_K

    coord_of: Dict[int, Dict[int, Tuple[float, float]]] = {}
    mesh_segs: List[Any] = []
    chain_segs: List[Any] = []
    chain_cols: List[Any] = []
    qx: List[float] = []
    qy: List[float] = []
    qc: List[Any] = []
    ring_x: List[float] = []
    ring_y: List[float] = []
    perimeter_touch: Dict[Node, List[Tuple[float, float]]] = {}

    L_ref = float(np.median([embeddings[c].L for c in cids]))
    lw = max(0.8, 22.0 / L_ref)
    msize = max(2.0, 190.0 / L_ref)

    for cid in cids:
        emb = embeddings[cid]
        L = emb.L
        r_off, c_off = origin[cid]
        H_local = _cached_hardware_graph(L)

        local = {}
        for u in H_local.nodes():
            r, c = oneD2twoD(u, L)
            local[u] = (c + c_off, -(r + r_off))
        coord_of[cid] = local

        if draw_mesh:
            for u, v in H_local.edges():
                mesh_segs.append([local[u], local[v]])

        for v, chain in emb.phi.items():
            col = node_color[v]
            cs = set(chain)
            for u in chain:
                qx.append(local[u][0]); qy.append(local[u][1]); qc.append(col)
                for w in H_local.neighbors(u):
                    if w in cs and w > u:
                        chain_segs.append([local[u], local[w]])
                        chain_cols.append(col)

        perim = _perimeter_ids(L)
        for v in emb.border_nodes:
            pts = [local[u] for u in emb.phi.get(v, ()) if u in perim]
            perimeter_touch[v] = pts
            if draw_rings:
                for x, y in pts:
                    ring_x.append(x); ring_y.append(y)

    if mesh_segs:
        ax.add_collection(LineCollection(mesh_segs, colors=["#c9c9c9"],
                                         linewidths=0.35, alpha=0.45, zorder=0))
    if chain_segs:
        ax.add_collection(LineCollection(chain_segs, colors=chain_cols,
                                         linewidths=lw, alpha=0.95, zorder=2,
                                         capstyle="round"))
    if qx:
        ax.scatter(qx, qy, s=msize, c=qc, marker="o", linewidths=0,
                   zorder=3)
    if ring_x:
        ax.scatter(ring_x, ring_y, s=msize * 2.6, facecolors="none",
                   edgecolors="black", linewidths=0.7, zorder=6)

    cross = list(partition.cross_edges)
    n_cut = len(cross)
    sampled = n_cut > max_chords
    if sampled:
        # a fixed stride sample, so the figure is the same every run
        step = n_cut / float(max_chords)
        cross = [cross[int(i * step)] for i in range(max_chords)]

    cut_segs = []
    for u, v, _ca, _cb in cross:
        pu = perimeter_touch.get(u) or []
        pv = perimeter_touch.get(v) or []
        if not pu or not pv:
            continue
        a, b = min(((a, b) for a in pu for b in pv),
                   key=lambda ab: math.dist(ab[0], ab[1]))
        cut_segs.append([a, b])
    if cut_segs:
        ax.add_collection(LineCollection(
            cut_segs, colors=["#1a1a1a"], linewidths=0.4,
            alpha=0.35 if not sampled else 0.18, zorder=5))
    cut_label = (f"inter-cluster edge ({len(cut_segs):,} of {n_cut:,} shown)"
                 if sampled else "inter-cluster edge (cut edge)")

    # captions only while they fit round the ring
    if len(cids) <= LABEL_LIMIT:
        for cid in cids:
            emb = embeddings[cid]
            r_off, c_off = origin[cid]
            cxp, cyp = c_off + emb.L / 2.0, -(r_off + emb.L / 2.0)
            lbl_r = math.hypot(cxp, cyp) + emb.L * 0.62 + 3.0
            ang = math.atan2(cyp, cxp)
            ax.text(lbl_r * math.cos(ang), lbl_r * math.sin(ang),
                    f"cluster {cid}\nL={emb.L}  n={emb.n_I}",
                    ha="center", va="center", fontsize=8, weight="bold",
                    color=_shade(get_color(cid + 1, k), -0.35), zorder=8)

    ax.set_aspect("equal")
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    pad = 0.07 * (y1 - y0)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.axis("off")
    handles = [mlines.Line2D([], [], color="#1a1a1a", lw=1.8, label=cut_label)]
    if draw_rings:
        handles.append(mlines.Line2D(
            [], [], marker="o", color="none", markerfacecolor="none",
            markeredgecolor="black", markeredgewidth=1.7, markersize=11,
            label="border qubit on patch perimeter"))
    ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.92)

    n_ok = sum(1 for e in embeddings.values() if e.success)
    ax.set_title(
        title or f"Stage 2 — per-cluster PSSA embeddings ({n_ok}/{len(embeddings)} complete)",
        fontsize=12, weight="bold", pad=10)

    if show and should_show:
        plt.tight_layout()
        plt.show()
    return ax


@lru_cache(maxsize=64)
def _perimeter_ids(L: int) -> frozenset:
    """The perimeter of a patch, as local hardware ids."""
    out = set()
    for r in range(L):
        for c in range(L):
            if r in (0, L - 1) or c in (0, L - 1):
                out.add(r * L + c + 1)
    return frozenset(out)


def _ring_origins(partition, embeddings, cids):
    """The patches on a ring, spaced in proportion to their size."""
    Ls = [embeddings[c].L for c in cids]
    total = sum(L * 1.9 + 4.0 for L in Ls)
    R = max(total / (2 * math.pi), 10.0)
    origin = {}
    ang = 0.0
    for cid, L in zip(cids, Ls):
        step = (L * 1.9 + 4.0) / total * 2 * math.pi
        a = ang + step / 2.0
        origin[cid] = (-R * math.sin(a) - L / 2.0, R * math.cos(a) - L / 2.0)
        ang += step
    return origin
