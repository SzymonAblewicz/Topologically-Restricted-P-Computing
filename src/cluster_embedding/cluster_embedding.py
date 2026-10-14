"""
The embedding of every cluster on its own patch by PSSA wBC, shared by both methods,
and Divide and Conquer's placing and routing of the clusters (Section 3.2 of the
report). A cluster is tried at the smallest L that holds it, then at larger L, and
should PSSA wBC fail it falls back to the complete graph embedding of K_s on KG(s-1).
"""

import colorsys
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from pssa_wbc import (
    ImprovedPSSA,
    build_guiding_pattern,
    build_hardware_graph,
    oneD2twoD,
    twoD2oneD,
    _perimeter_nodes,
    _bfs_path,
    is_valid_embedding,
    invert,
    eemb,
)
from draw_embedding import get_color
from place_and_route import (
    Placement,
    _transform_rc,
    floorplan_anneal_cpp,
    route_negotiated_cpp,
)

Node = Any
NodeH = int


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Embedding every cluster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class ClusterEmbedding:
    cid: int
    L: int
    phi: Dict[Node, List[NodeH]]  # logical node -> hardware vertices of the patch (1..L*L)
    success: bool
    eemb: int
    m_I: int
    n_I: int
    border_nodes: Set[Node]
    seed_used: int
    wall_time: float

    @property
    def qubits(self) -> int:
        return sum(len(c) for c in self.phi.values())

    def __str__(self) -> str:
        flag = "ok  " if self.success else "FAIL"
        return (f"cluster {self.cid:2d}: {flag} L={self.L:2d}  "
                f"{self.n_I:3d} nodes  {self.eemb:3d}/{self.m_I:3d} edges  "
                f"{self.qubits:4d} qubits  {self.wall_time:5.1f}s")


def _min_feasible_L(n_nodes: int, n_border: int) -> int:
    """The smallest L with n_nodes <= L^2 - 1 and at most 4L - 4 border nodes."""
    L = 3
    while (L * L - 1) < n_nodes or (4 * L - 4) < n_border:
        L += 1
    return L


import concurrent.futures

def _embed_single_worker(args):
    """One cluster, from L0 up to L0 + L_slack with several seeds at each L, until the first success."""
    cid, I, border, L0, L_slack, tmax, tmax_hard, seeds, gibbs = args
    n = I.number_of_nodes()
    t0 = time.perf_counter()
    best = None

    for L in range(L0, L0 + L_slack + 1):
        steps = tmax_hard if L >= L0 + L_slack - 1 else tmax
        for seed in seeds:
            try:
                algo = ImprovedPSSA(
                    L=L, tmax=steps, gibbs=gibbs, seed=seed, verbose=False,
                    border_nodes=border, border_mode="permute",
                )
                res = algo.run(I)
            except ValueError:
                continue

            cand = ClusterEmbedding(
                cid=cid, L=L, phi=res.phi, success=res.success,
                eemb=res.eemb, m_I=res.m_I, n_I=n,
                border_nodes=set(border), seed_used=seed,
                wall_time=time.perf_counter() - t0,
            )
            if best is None or cand.eemb > best.eemb:
                best = cand
            if res.success:
                break
        if best is not None and best.success:
            break

    if best is None or not best.success:
        best = _complete_graph_fallback(cid, I, border, t0)

    return cid, best

def _complete_graph_fallback(cid, I, border, t0):
    """The complete graph embedding of K_n on KG(n-1), which embeds any cluster with every super vertex on the perimeter."""
    n = I.number_of_nodes()
    L = max(n - 1, 2)
    H = build_hardware_graph(L)
    phi = {v: list(c) for v, c in zip(sorted(I.nodes()), build_guiding_pattern(H, L).values())}
    return ClusterEmbedding(
        cid=cid, L=L, phi=phi, success=is_valid_embedding(phi, I, H),
        eemb=eemb(phi, I, H, invert(phi)), m_I=I.number_of_edges(), n_I=n,
        border_nodes=set(border), seed_used=-1, wall_time=time.perf_counter() - t0,
    )

def embed_clusters(
    partition,
    L_slack: int = 15,
    tmax: int = 400_000,
    tmax_hard: int = 1_500_000,
    seeds: Sequence[int] = (1, 2, 3),
    gibbs: bool = True,
    verbose: bool = True,
) -> Dict[int, ClusterEmbedding]:
    """PSSA wBC on every cluster in parallel, largest first."""
    embeddings: Dict[int, ClusterEmbedding] = {}

    order = sorted(partition.communities,
                   key=lambda c: -partition.subgraphs[c].number_of_nodes())

    tasks = []
    for cid in order:
        # a copy, so a worker is sent the cluster and not the whole graph
        I = nx.Graph(partition.subgraphs[cid])
        border = partition.border_nodes[cid]
        n = I.number_of_nodes()
        L0 = _min_feasible_L(n, len(border))
        tasks.append((cid, I, border, L0, L_slack, tmax, tmax_hard, seeds, gibbs))

    if verbose:
        print(f"  Launching {len(tasks)} parallel PSSA embedding jobs...")

    executor = concurrent.futures.ProcessPoolExecutor()
    try:
        futures = {executor.submit(_embed_single_worker, t): t for t in tasks}

        for future in concurrent.futures.as_completed(futures):
            cid, best = future.result()
            embeddings[cid] = best
            if verbose:
                print(f"  {best}")
    except BaseException:
        # on any error, stop at once instead of waiting for the other clusters
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    return embeddings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Divide and Conquer: placing and routing the clusters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class CompactResult:
    phi: Dict[Node, List[NodeH]]
    H: "nx.Graph"
    L: int
    placements: List[Placement]
    cluster_of: Dict[Node, int]
    routes: Dict[Tuple[Node, Node], List[NodeH]]
    unrouted: List[Tuple[Node, Node]]
    valid: bool
    eemb: int
    m_total: int
    orientations: Dict[int, int] = field(default_factory=dict)

    @property
    def qubits(self) -> int:
        return sum(len(c) for c in self.phi.values())

    @property
    def utilization(self) -> float:
        return self.qubits / (self.L * self.L)

    def summary(self) -> str:
        chains = [len(c) for c in self.phi.values()]
        return (
            f"CompactResult: King's graph L={self.L} ({self.L * self.L} qubits)\n"
            f"  logical nodes embedded : {len(self.phi)}\n"
            f"  edges embedded         : {self.eemb}/{self.m_total}"
            f"{'  (COMPLETE)' if self.eemb == self.m_total else ''}\n"
            f"  valid minor embedding  : {self.valid}\n"
            f"  qubits used            : {self.qubits} "
            f"({self.utilization:.1%} of lattice)\n"
            f"  chain length           : min={min(chains)} "
            f"mean={sum(chains)/len(chains):.1f} max={max(chains)}\n"
            f"  inter-cluster routes   : {len(self.routes)} "
            f"(unrouted: {len(self.unrouted)})"
        )


def compact_embedding(
    G: "nx.Graph",
    partition,
    embeddings: Dict[int, ClusterEmbedding],
    gap: int = 2,
    verbose: bool = True,
    auto_gap: bool = True,
    max_gap: int = 5,
    sa_seed: int = 0,
    n_sa_tries: int = 3,
    pad: bool = False,
    n_moves: int = 6000,
    n_restarts: int = 4,
    n_reserve: int = 2,
    max_rounds: int = 20,
) -> CompactResult:
    """
    Place the patches with the B*-tree floorplanner and route every cut edge with
    PathFinder, widening the corridors until every cut edge is routed.
    """
    if auto_gap:
        last = None
        for g in range(gap, max_gap + 1):
            res = compact_embedding(G, partition, embeddings, gap=g,
                                     verbose=True, auto_gap=False,
                                     sa_seed=sa_seed, n_sa_tries=n_sa_tries,
                                     pad=pad, n_moves=n_moves, n_restarts=n_restarts,
                                     n_reserve=n_reserve, max_rounds=max_rounds)
            last = res
            if not res.unrouted:
                break
        if verbose:
            print("\n" + last.summary())
        return last

    cut_edges = [(u, v) for u, v, _, _ in partition.cross_edges]
    cluster_of: Dict[Node, int] = {}
    for cid, nodes in partition.communities.items():
        for v in nodes:
            cluster_of[v] = cid

    # the floorplanner's cost doesn't see whether a layout routes, so several are
    # routed and the best kept
    cands = []
    for k in range(n_sa_tries):
        placements, orient, L_global = floorplan_anneal_cpp(
            embeddings, partition, gap=gap, seed=sa_seed * 1009 + k,
            n_restarts=n_restarts, n_moves=n_moves,
        )
        if pad:
            for p in placements:
                p.row0 += gap
                p.col0 += gap
            L_global += 2 * gap

        H = build_hardware_graph(L_global)
        phi: Dict[Node, List[NodeH]] = {}
        for p in placements:
            emb = embeddings[p.cid]
            t = orient[p.cid]
            for v, chain in emb.phi.items():
                gchain = []
                for u in chain:
                    r, c = oneD2twoD(u, emb.L)
                    rr, cc = _transform_rc(r, c, emb.L, t)
                    gchain.append(twoD2oneD(rr + p.row0, cc + p.col0, L_global))
                phi[v] = gchain

        free0: Set[NodeH] = set(H.nodes()) - {u for c in phi.values() for u in c}
        centers = {p.cid: p.center for p in placements}

        def _sep(u, v, _c=centers) -> float:
            a, b = _c[cluster_of[u]], _c[cluster_of[v]]
            return math.hypot(a[0] - b[0], a[1] - b[1])

        routes, unrouted, phi_out = route_negotiated_cpp(
            H, phi, cut_edges, free0, cluster_of, placements, L_global, _sep,
            n_reserve=n_reserve, max_rounds=max_rounds,
        )
        route_qubits = sum(len(p) for p in routes.values())
        max_chain = max(len(c) for c in phi_out.values())
        cands.append(((len(unrouted), L_global, max_chain, route_qubits),
                       phi_out, H, L_global, placements, routes, unrouted, orient))

    ok = [c for c in cands if c[0][0] == 0] or cands
    best_L = min(c[0][1] for c in ok)
    pool = [c for c in ok if c[0][1] <= best_L + 1]
    best = min(pool, key=lambda c: (c[0][0], c[0][2], c[0][3], c[0][1]))
    _, phi, H, L_global, placements, routes, unrouted, orient = best

    # trim to the occupied box, a translation, so every adjacency is kept
    occupied = [oneD2twoD(u, L_global) for c in phi.values() for u in c]
    r_min = min(r for r, _ in occupied); r_max = max(r for r, _ in occupied)
    c_min = min(c for _, c in occupied); c_max = max(c for _, c in occupied)
    L_new = max(r_max - r_min + 1, c_max - c_min + 1)
    if L_new < L_global:
        remap = {}
        for u in {u for c in phi.values() for u in c}:
            r, c = oneD2twoD(u, L_global)
            remap[u] = twoD2oneD(r - r_min, c - c_min, L_new)
        phi = {v: [remap[u] for u in ch] for v, ch in phi.items()}
        routes = {e: [remap[u] for u in p] for e, p in routes.items()}
        for p in placements:
            p.row0 -= r_min
            p.col0 -= c_min
        H = build_hardware_graph(L_new)
        L_global = L_new

    inv = invert(phi)
    e_total = eemb(phi, G, H, inv)
    valid = is_valid_embedding(phi, G, H)

    result = CompactResult(
        phi=phi, H=H, L=L_global, placements=placements,
        cluster_of=cluster_of, routes=routes, unrouted=unrouted,
        valid=valid, eemb=e_total, m_total=G.number_of_edges(),
        orientations=orient,
    )
    if verbose:
        print("\n" + result.summary())
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Shared drawing helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _shade(color, amount: float):
    """Blend a colour towards white (amount > 0) or black (amount < 0)."""
    r, g, b = mcolors.to_rgb(color)
    if amount >= 0:
        f = amount
        return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
    f = -amount
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def _node_colors(partition, k: int) -> Dict[Node, Any]:
    """A colour per vertex, in its cluster's hue."""
    out: Dict[Node, Any] = {}
    HUE_JITTER_DEG = 25.0  # max hue deviation from the cluster's hue
    for cid in sorted(partition.communities):
        base = get_color(cid + 1, k)
        h0, _, _ = colorsys.rgb_to_hsv(*base)
        nodes = sorted(partition.communities[cid], key=str)
        for idx, v in enumerate(nodes):
            h = (h0 + (((idx * 0.6180339887) % 1.0) - 0.5)
                 * (2 * HUE_JITTER_DEG / 360.0)) % 1.0
            s = 0.55 + ((idx * 0.3819660113) % 1.0) * 0.45   # 0.55–1.00
            val = 0.42 + ((idx * 0.2469796037) % 1.0) * 0.53  # 0.42–0.95
            out[v] = colorsys.hsv_to_rgb(h, s, val)
    return out


def _mesh_segments(L: int, r_off: float = 0.0, c_off: float = 0.0):
    """The King's graph lattice as line segments, offset by (r_off, c_off)."""
    segs = []
    for row in range(L):
        for col in range(L):
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < L and 0 <= c2 < L:
                    segs.append([
                        (col + c_off, -(row + r_off)),
                        (c2 + c_off, -(r2 + r_off)),
                    ])
    return segs


def _draw_logical_edges(ax, edges, phi, H, coord_of, lw, color="#3a3a3a",
                         alpha=0.55, zorder=1.5):
    """Every embedded edge, between the two adjacent hardware vertices that realize it."""
    segs = []
    for i, j in edges:
        ci, cj = phi.get(i), phi.get(j)
        # len() rather than `not ci`: a chain can be a numpy array
        if ci is None or cj is None or len(ci) == 0 or len(cj) == 0:
            continue
        best = None
        for a in ci:
            for b in cj:
                if H.has_edge(a, b):
                    best = (a, b)
                    break
            if best:
                break
        if best:
            segs.append([coord_of[best[0]], coord_of[best[1]]])
    if segs:
        ax.add_collection(LineCollection(segs, colors=[color], linewidths=lw,
                                          alpha=alpha, zorder=zorder))
    return len(segs)


def _draw_chains(ax, phi, H, coord_of, node_color, lw, msize,
                  edge_lw=0.0, zorder=2, alpha=0.95):
    """The super vertices as thick paths, with a marker per hardware vertex."""
    owner = {u: v for v, chain in phi.items() for u in chain}
    for v, chain in phi.items():
        color = node_color[v]
        chain_set = set(chain)
        drawn = set()
        segs = []
        for u in chain:
            for w in H.neighbors(u):
                if w in chain_set and w != u:
                    key = (min(u, w), max(u, w))
                    if key in drawn:
                        continue
                    drawn.add(key)
                    segs.append([coord_of[u], coord_of[w]])
        if segs:
            ax.add_collection(LineCollection(segs, colors=[color], linewidths=lw,
                                              alpha=alpha, zorder=zorder,
                                              capstyle="round"))
    xs = [coord_of[u][0] for u in owner]
    ys = [coord_of[u][1] for u in owner]
    cs = [node_color[owner[u]] for u in owner]
    ax.scatter(xs, ys, s=msize, c=cs, marker="o",
                edgecolors="black" if edge_lw > 0 else "none",
                linewidths=edge_lw, zorder=zorder + 1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Drawing: every cluster on its own patch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_exploded_clusters(
    partition, embeddings: Dict[int, ClusterEmbedding], ax=None,
    show: bool = True, title: Optional[str] = None,
):
    """Every cluster's patch on a ring, with each cut edge drawn between its two border nodes."""
    should_show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))

    k = partition.k
    node_color = _node_colors(partition, k)
    cids = sorted(embeddings)

    # ring order: a walk of the cluster graph, so connected clusters sit side by side
    CG = nx.Graph()
    CG.add_nodes_from(cids)
    for _, _, ca, cb in partition.cross_edges:
        w = CG.get_edge_data(ca, cb, {}).get("weight", 0)
        CG.add_edge(ca, cb, weight=w + 1)
    ring: List[int] = []
    seen: Set[int] = set()
    for comp in sorted(nx.connected_components(CG), key=len, reverse=True):
        start = max(comp, key=lambda c: CG.degree(c, weight="weight"))
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur); ring.append(cur)
            for z in sorted(CG.neighbors(cur),
                             key=lambda z: CG[cur][z].get("weight", 1)):
                if z not in seen:
                    stack.append(z)
    ring += [c for c in cids if c not in seen]

    # ring spacing in proportion to patch size
    spans = [embeddings[c].L + 4.0 for c in ring]
    total = sum(spans)
    circumference = total * 1.32
    R = max(circumference / (2 * math.pi),
            max(embeddings[c].L for c in cids) * 1.55)

    origin: Dict[int, Tuple[float, float]] = {}
    acc = 0.0
    for cid, span in zip(ring, spans):
        L = embeddings[cid].L
        angle = 2 * math.pi * (acc + span / 2.0) / total - math.pi / 2
        acc += span
        cx, cy = R * math.cos(angle), R * math.sin(angle)
        # (r_off, c_off) chosen so the patch is centered on (cx, cy)
        origin[cid] = (-cy - L / 2.0, cx - L / 2.0)

    cids = ring

    # one set of sizes for every patch, from the median patch side
    L_ref = float(np.median([embeddings[c].L for c in cids]))
    lw = max(1.4, 22.0 / L_ref)
    msize = max(9.0, 190.0 / L_ref)
    logical_lw = max(0.8, 11.0 / L_ref)

    coord_of: Dict[Tuple[int, NodeH], Tuple[float, float]] = {}
    perimeter_touch: Dict[Node, List[Tuple[float, float]]] = {}

    for cid in cids:
        emb = embeddings[cid]
        L = emb.L
        r_off, c_off = origin[cid]

        ax.add_collection(LineCollection(
            _mesh_segments(L, r_off, c_off), colors=["#c9c9c9"],
            linewidths=0.35, alpha=0.45, zorder=0))

        H_local = build_hardware_graph(L)
        local_coord = {}
        for u in H_local.nodes():
            r, c = oneD2twoD(u, L)
            local_coord[u] = (c + c_off, -(r + r_off))
        coord_of[cid] = local_coord

        sub = partition.subgraphs[cid]
        _draw_logical_edges(ax, sub.edges(), emb.phi, H_local, local_coord,
                             lw=logical_lw)
        _draw_chains(ax, emb.phi, H_local, local_coord, node_color,
                      lw=lw, msize=msize, edge_lw=0.3)

        # ring diameter, from the marker size
        ring_pt = 2.1 * math.sqrt(msize)
        perim = _perimeter_nodes(L)
        for v in emb.border_nodes:
            touches = [u for u in emb.phi.get(v, []) if u in perim]
            perimeter_touch[v] = [local_coord[u] for u in touches]
            for u in touches:
                x, y = local_coord[u]
                ax.plot(x, y, "o", markersize=ring_pt,
                         markerfacecolor="none", markeredgecolor="black",
                         markeredgewidth=1.5, zorder=6)

        # cluster caption just outside the ring
        cxp, cyp = c_off + L / 2.0, -(r_off + L / 2.0)
        lbl_r = math.hypot(cxp, cyp) + L * 0.62 + 3.0
        ang = math.atan2(cyp, cxp)
        ax.text(lbl_r * math.cos(ang), lbl_r * math.sin(ang),
                 f"cluster {cid}\nL={L}  n={emb.n_I}",
                 ha="center", va="center", fontsize=11, weight="bold",
                 color=_shade(get_color(cid + 1, k), -0.35),
                 zorder=8)

    # the cut edges, border node to border node
    for u, v, ca, cb in partition.cross_edges:
        pu = perimeter_touch.get(u) or []
        pv = perimeter_touch.get(v) or []
        if not pu or not pv:
            continue
        best = min(((a, b) for a in pu for b in pv),
                    key=lambda ab: math.dist(ab[0], ab[1]))
        ax.add_patch(FancyArrowPatch(
            best[0], best[1], connectionstyle="arc3,rad=0.13",
            arrowstyle="-", linewidth=1.7, color="#1a1a1a", alpha=0.72,
            zorder=5, mutation_scale=1))

    ax.set_aspect("equal")
    ax.autoscale_view()
    # room for the captions outside the ring
    y0, y1 = ax.get_ylim()
    pad = 0.07 * (y1 - y0)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.axis("off")
    ax.legend(handles=[
        mlines.Line2D([], [], color="#1a1a1a", lw=1.8,
                       label="inter-cluster edge (cut edge)"),
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=1.7,
                       markersize=11, label="border qubit on patch perimeter"),
    ], loc="upper right", fontsize=11, framealpha=0.92)

    n_ok = sum(1 for e in embeddings.values() if e.success)
    ax.set_title(
        title or f"Stage 2 — per-cluster PSSA embeddings ({n_ok}/{len(embeddings)} complete)",
        fontsize=12, weight="bold", pad=10)

    if show and should_show:
        plt.tight_layout()
        plt.show()
    return ax


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Drawing: the whole embedding
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_compact_embedding(
    G: "nx.Graph", partition, result: CompactResult, ax=None,
    show: bool = True, title: Optional[str] = None,
):
    """The whole embedding, every patch on one lattice and the routes between them in dark blue."""
    should_show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))

    L = result.L
    H = result.H
    k = partition.k
    node_color = _node_colors(partition, k)

    coord_of = {}
    for u in H.nodes():
        r, c = oneD2twoD(u, L)
        coord_of[u] = (c, -r)

    ax.add_collection(LineCollection(
        _mesh_segments(L), colors=["#d5d5d5"], linewidths=0.22,
        alpha=0.4, zorder=0))

    # a faint tint behind each patch
    from matplotlib.patches import Rectangle
    for p in result.placements:
        base = get_color(p.cid + 1, k)
        ax.add_patch(Rectangle(
            (p.col0 - 0.5, -(p.row0 + p.L - 1) - 0.5), p.L, p.L,
            facecolor=_shade(base, 0.9), edgecolor=_shade(base, 0.7),
            linewidth=0.9, zorder=0.2))
        ax.text(p.col0 + p.L / 2.0 - 0.5, -(p.row0 - 0.9),
                 f"c{p.cid}", ha="center", va="bottom", fontsize=9,
                 weight="bold", color=_shade(base, -0.4), zorder=0.3)

    lw = max(0.9, 30.0 / L)
    msize = max(3.0, 330.0 / L)
    intra = [(i, j) for i, j in G.edges()
              if result.cluster_of[i] == result.cluster_of[j]]
    _draw_logical_edges(ax, intra, result.phi, H, coord_of,
                         lw=max(0.5, 14.0 / L))
    _draw_chains(ax, result.phi, H, coord_of, node_color, lw=lw, msize=msize,
                  edge_lw=0.0)

    # the routes
    route_nodes = {w for path in result.routes.values() for w in path}
    if route_nodes:
        segs = []
        for (u, v), path in result.routes.items():
            pts = [coord_of[w] for w in path]
            # join each end of the route to the nearest hardware vertex of either super vertex
            path_set = set(path)
            # list() first: chains can be numpy arrays
            both = list(result.phi[u]) + list(result.phi[v])
            ends = [a for a in both if a not in path_set] or both
            a0 = min(ends, key=lambda a: math.dist(coord_of[a], pts[0]))
            b0 = min(ends, key=lambda a: math.dist(coord_of[a], pts[-1]))
            chain_pts = [coord_of[a0]] + pts + [coord_of[b0]]
            for a, b in zip(chain_pts, chain_pts[1:]):
                segs.append([a, b])
        ax.add_collection(LineCollection(
            segs, colors=["#12263a"], linewidths=max(0.8, 20.0 / L),
            alpha=0.8, zorder=4))
        xs = [coord_of[w][0] for w in route_nodes]
        ys = [coord_of[w][1] for w in route_nodes]
        ax.scatter(xs, ys, s=max(2.0, 160.0 / L), c="#12263a",
                    marker="s", zorder=5, linewidths=0)

    # centre the occupied box, which need not be square, in the view
    occ = [oneD2twoD(u, L) for c in result.phi.values() for u in c]
    r0, r1 = min(r for r, _ in occ), max(r for r, _ in occ)
    c0, c1 = min(c for _, c in occ), max(c for _, c in occ)
    pad_r = (L - (r1 - r0 + 1)) / 2.0
    pad_c = (L - (c1 - c0 + 1)) / 2.0
    ax.set_xlim(c0 - 0.5 - pad_c, c1 + 0.5 + pad_c)
    ax.set_ylim(-(r1 + 0.5 + pad_r), -(r0 - 0.5 - pad_r))
    ax.set_aspect("equal")
    ax.axis("off")
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

