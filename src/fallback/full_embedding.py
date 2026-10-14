"""
The whole embedding of the fallback method, built from the partition and the patches
by the construction and checked as a minor-embedding of the whole graph (--validate).
With fast=True it never builds the King's graph or the chains of the routing structure,
and gives the same layout, cell for cell.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

from kings_lattice import KingsLattice, rc_of, validate_embedding
from pssa_wbc import build_hardware_graph, oneD2twoD, twoD2oneD, invert, eemb, is_valid_embedding
from cluster_embedding import CompactResult, ClusterEmbedding, embed_clusters
from construction import guaranteed_plan, guaranteed_floorplan, guaranteed_route
from construction_fast import guaranteed_plan_fast, guaranteed_route_fast

Node = Any


def compact_embedding_guaranteed(
    G: "nx.Graph",
    partition,
    embeddings: Dict[int, ClusterEmbedding],
    verbose: bool = True,
    fast: bool = True,
) -> CompactResult:
    """The patches and the construction into one embedding of G, checked against G, as a CompactResult."""
    if fast:
        return _compact_embedding_fast(G, partition, embeddings, verbose)

    t0 = time.perf_counter()
    plan = guaranteed_plan(embeddings, partition)
    placements, orient, L_global = guaranteed_floorplan(embeddings, partition, plan=plan)

    H = build_hardware_graph(L_global)
    phi: Dict[Node, list] = {}
    for p in placements:
        emb = embeddings[p.cid]
        for v, chain in emb.phi.items():
            gchain = []
            for u in chain:
                r, c = oneD2twoD(u, emb.L)
                gchain.append(twoD2oneD(r + p.row0, c + p.col0, L_global))
            phi[v] = gchain

    cut_edges = [(u, v) for u, v, _, _ in partition.cross_edges]
    cluster_of: Dict[Node, int] = {}
    for cid, nodes in partition.communities.items():
        for v in nodes:
            cluster_of[v] = cid

    routes, unrouted, phi = guaranteed_route(H, phi, cut_edges, plan)

    # trim to the occupied box
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
        s = plan.stats
        print(f"\n{result.summary()}")
        print(f"  guaranteed construction : {time.perf_counter() - t0:.1f}s, "
              f"{s['T']} lanes, {s['n_top']} top-side, funnel depth {s['funnel']}, "
              f"longest lane {s['max_lane']}")
        print(_layout_budget(plan.stats, G.number_of_nodes()))
    return result


def _compact_embedding_fast(
    G: "nx.Graph",
    partition,
    embeddings: Dict[int, ClusterEmbedding],
    verbose: bool = True,
) -> CompactResult:
    """compact_embedding_guaranteed without building anything of size L^2."""
    t0 = time.perf_counter()
    plan = guaranteed_plan_fast(embeddings, partition)
    placements, orient, L_global = plan.placements, plan.orient, plan.L

    H = KingsLattice(L_global)
    phi: Dict[Node, np.ndarray] = {}
    for p in placements:
        emb = embeddings[p.cid]
        for v, chain in emb.phi.items():
            r, c = rc_of(np.asarray(chain, dtype=np.int64), emb.L)
            phi[v] = (r + p.row0) * L_global + (c + p.col0) + 1

    cut_edges = [(u, v) for u, v, _, _ in partition.cross_edges]
    cluster_of: Dict[Node, int] = {}
    for cid, nodes in partition.communities.items():
        for v in nodes:
            cluster_of[v] = cid

    routes, unrouted, phi = guaranteed_route_fast(H, phi, cut_edges, plan)

    # trim to the occupied box, chain by chain
    r_min = c_min = None
    r_max = c_max = -1
    for arr in phi.values():
        r, c = rc_of(arr, L_global)
        lo_r, hi_r = int(r.min()), int(r.max())
        lo_c, hi_c = int(c.min()), int(c.max())
        r_min = lo_r if r_min is None else min(r_min, lo_r)
        c_min = lo_c if c_min is None else min(c_min, lo_c)
        r_max = max(r_max, hi_r)
        c_max = max(c_max, hi_c)
    L_new = max(r_max - r_min + 1, c_max - c_min + 1)
    if L_new < L_global:
        def _remap(arr):
            r, c = rc_of(arr, L_global)
            return (r - r_min) * L_new + (c - c_min) + 1
        phi = {v: _remap(a) for v, a in phi.items()}
        routes = {e: _remap(a) for e, a in routes.items()}
        for p in placements:
            p.row0 -= r_min
            p.col0 -= c_min
        H = KingsLattice(L_new)
        L_global = L_new

    valid, e_total = validate_embedding(phi, G, H)

    result = CompactResult(
        phi=phi, H=H, L=L_global, placements=placements,
        cluster_of=cluster_of, routes=routes, unrouted=unrouted,
        valid=valid, eemb=e_total, m_total=G.number_of_edges(),
        orientations=orient,
    )
    if verbose:
        s = plan.stats
        print(f"\n{result.summary()}")
        print(f"  guaranteed construction : {time.perf_counter() - t0:.1f}s, "
              f"{s['T']} lanes, {s['n_top']} top-side, funnel depth {s['funnel']}, "
              f"longest lane {s['max_lane']}")
        print(_layout_budget(plan.stats, G.number_of_nodes()))
    return result


def _layout_budget(s: Dict[str, int], n: int) -> str:
    """L, term by term: width = patch columns + corridors, height = bands + patch rows + funnel + routing structure."""
    T = s["T"]
    L = max(s["width"], s["height"])
    excess = L - T
    return (
        f"  layout: grid {s['Gr']}x{s['Gc']}, "
        f"width = {s['patch_cols']} patch + {s['corridor_cols']} corridor = {s['width']}\n"
        f"          height = {s['band_rows']} band + {s['patch_rows']} patch + "
        f"{s['funnel']} funnel + {s['plaza_depth']} plaza = {s['height']}\n"
        f"          L = {L} (pre-trim) = {L / T:.3f}*T, "
        f"excess {excess} over T = {excess / math.sqrt(n):.2f}*sqrt(n)"
    )


def run_pipeline_guaranteed(
    G: "nx.Graph",
    max_cluster_size: int,
    min_cluster_size: int = 1,
    seed: Optional[int] = 7,
    verbose: bool = True,
    fast: bool = True,
):
    """The fallback method end to end: partition, every cluster by PSSA wBC, then the construction."""
    from border_node_partitioning import partition_graph

    if verbose:
        print(f"Input network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        print("--- Stage 1: partition ---")
    part = partition_graph(
        G, max_cluster_size=max_cluster_size, min_cluster_size=min_cluster_size,
        seed=seed, verbose=verbose,
    )
    if verbose:
        print(f"  k = {part.k} clusters, T = {part.border_count} border vertices "
              f"(T/n = {part.border_fraction_total():.3f}) -> plaza "
              f"{part.border_count}x{part.border_count - 1}, cut = {part.cut_size} edges "
              f"({part.cut_fraction(G):.1%})\n")
        print("--- Stage 2: border-constrained PSSA per cluster ---")

    embeddings = embed_clusters(part, verbose=verbose)

    if verbose:
        print("\n--- Stage 3 (guaranteed fallback): place + route ---")
    result = compact_embedding_guaranteed(G, part, embeddings, verbose=verbose,
                                          fast=fast)
    return part, embeddings, result