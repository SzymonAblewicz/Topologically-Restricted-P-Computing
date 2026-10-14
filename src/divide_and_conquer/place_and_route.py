"""
Placing and routing the clusters in Divide and Conquer (Section 3.2 of the report).
A B*-tree floorplanner (Chang et al. 2000) places the patches by simulated annealing,
turning and mirroring them where it helps, and PathFinder (McMurchie and Ebeling 1995)
routes every cut edge along a path of free hardware vertices. The _cpp versions give
the same results, with the hot loops in C++ (floorplan_core.cpp).
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from pssa_wbc import oneD2twoD, twoD2oneD, build_hardware_graph, _perimeter_nodes

if TYPE_CHECKING:
    from cluster_embedding import ClusterEmbedding

Node = Any
NodeH = int
Edge = Tuple[Node, Node]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Placement, one patch's position on the lattice
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class Placement:
    cid: int
    L: int
    row0: int
    col0: int

    @property
    def center(self) -> Tuple[float, float]:
        return (self.row0 + self.L / 2.0, self.col0 + self.L / 2.0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The 8 orientations of a square patch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _transform_rc(r: int, c: int, L: int, t: int) -> Tuple[int, int]:
    """One of the 8 symmetries of an L x L square, each of which keeps a patch valid."""
    if t == 0: return (r, c)
    if t == 1: return (c, L - 1 - r)
    if t == 2: return (L - 1 - r, L - 1 - c)
    if t == 3: return (L - 1 - c, r)
    if t == 4: return (r, L - 1 - c)
    if t == 5: return (L - 1 - r, c)
    if t == 6: return (c, r)
    return (L - 1 - c, L - 1 - r)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The B*-tree
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class BNode:
    cid: int
    L: int
    parent: Optional[int] = None
    left: Optional[int] = None
    right: Optional[int] = None


class BStarTree:
    __slots__ = ("nodes", "root")

    def __init__(self, nodes: Dict[int, BNode], root: Optional[int]):
        self.nodes = nodes
        self.root = root

    def copy(self) -> "BStarTree":
        return BStarTree(
            {cid: BNode(n.cid, n.L, n.parent, n.left, n.right)
             for cid, n in self.nodes.items()},
            self.root,
        )


def validate(tree: BStarTree) -> None:
    """Check the tree: every node reached once from the root, and no cycles."""
    if tree.root is None:
        assert not tree.nodes
        return
    assert tree.nodes[tree.root].parent is None
    seen: Set[int] = set()
    stack = [tree.root]
    while stack:
        cid = stack.pop()
        assert cid not in seen, f"cycle or duplicate at cid={cid}"
        seen.add(cid)
        n = tree.nodes[cid]
        if n.left is not None:
            assert tree.nodes[n.left].parent == cid
            stack.append(n.left)
        if n.right is not None:
            assert tree.nodes[n.right].parent == cid
            stack.append(n.right)
    assert seen == set(tree.nodes), "tree does not span all nodes"


def build_caterpillar(sizes: Sequence[Tuple[int, int]]) -> BStarTree:
    """A first tree, all the patches in one row."""
    nodes: Dict[int, BNode] = {}
    root: Optional[int] = None
    prev: Optional[int] = None
    for cid, L in sizes:
        nodes[cid] = BNode(cid=cid, L=L)
        if prev is None:
            root = cid
        else:
            nodes[prev].left = cid
            nodes[cid].parent = prev
        prev = cid
    return BStarTree(nodes, root)


def build_random_tree(sizes: Sequence[Tuple[int, int]], rng: random.Random) -> BStarTree:
    """A random first tree, for the restarts."""
    order = list(sizes)
    rng.shuffle(order)
    nodes: Dict[int, BNode] = {}
    cid0, L0 = order[0]
    nodes[cid0] = BNode(cid=cid0, L=L0)
    root = cid0
    placed = [cid0]
    for cid, L in order[1:]:
        nodes[cid] = BNode(cid=cid, L=L)
        candidates = [c for c in placed if nodes[c].left is None or nodes[c].right is None]
        parent = rng.choice(candidates)
        pn = nodes[parent]
        if pn.left is None and (pn.right is not None or rng.random() < 0.5):
            pn.left = cid
        else:
            pn.right = cid
        nodes[cid].parent = parent
        placed.append(cid)
    return BStarTree(nodes, root)


def decode(tree: BStarTree, gap: int) -> Tuple[Dict[int, Placement], int, int]:
    """The layout of a tree, every patch in a cell gap wider than it so that a corridor is left between neighbours: (placements, W, H)."""
    if tree.root is None:
        return {}, 0, 0

    contour: List[Tuple[int, int, int]] = []  # (col_start, col_end, row_bottom)

    def query(a: int, b: int) -> int:
        best = 0
        for s, e, h in contour:
            if s < b and e > a and h > best:
                best = h
        return best

    def update(a: int, b: int, new_h: int) -> None:
        nonlocal contour
        nc: List[Tuple[int, int, int]] = []
        for s, e, h in contour:
            if e <= a or s >= b:
                nc.append((s, e, h))
                continue
            if s < a:
                nc.append((s, a, h))
            if e > b:
                nc.append((b, e, h))
        nc.append((a, b, new_h))
        nc.sort()
        merged: List[Tuple[int, int, int]] = []
        for seg in nc:
            if merged and merged[-1][1] == seg[0] and merged[-1][2] == seg[2]:
                merged[-1] = (merged[-1][0], seg[1], seg[2])
            else:
                merged.append(seg)
        contour = merged

    placements: Dict[int, Placement] = {}
    bounds = [0, 0]  # [W, H]

    def visit(cid: int, col0: int) -> None:
        node = tree.nodes[cid]
        Lp = node.L + gap
        row0 = query(col0, col0 + Lp)
        placements[cid] = Placement(cid=cid, L=node.L, row0=row0, col0=col0)
        update(col0, col0 + Lp, row0 + Lp)
        bounds[0] = max(bounds[0], col0 + Lp)
        bounds[1] = max(bounds[1], row0 + Lp)
        if node.left is not None:
            visit(node.left, col0 + Lp)
        if node.right is not None:
            visit(node.right, col0)

    visit(tree.root, 0)
    return placements, bounds[0], bounds[1]


def _set_parent_child(tree: BStarTree, parent_cid: Optional[int],
                       old_child: int, new_child: Optional[int]) -> None:
    if parent_cid is None:
        return
    p = tree.nodes[parent_cid]
    if p.left == old_child:
        p.left = new_child
    elif p.right == old_child:
        p.right = new_child


def _subtree_cids(tree: BStarTree, cid: Optional[int]) -> Set[int]:
    if cid is None:
        return set()
    out = {cid}
    n = tree.nodes[cid]
    if n.left is not None:
        out |= _subtree_cids(tree, n.left)
    if n.right is not None:
        out |= _subtree_cids(tree, n.right)
    return out


def swap_positions(tree: BStarTree, a: int, b: int) -> None:
    """Exchange the positions of two nodes in the tree."""
    na, nb = tree.nodes[a], tree.nodes[b]
    if na.parent == b or nb.parent == a:
        raise ValueError("swap_positions requires non-adjacent nodes")
    pa, la, ra = na.parent, na.left, na.right
    pb, lb, rb = nb.parent, nb.left, nb.right

    _set_parent_child(tree, pa, a, b)
    _set_parent_child(tree, pb, b, a)

    if la is not None:
        tree.nodes[la].parent = b
    if ra is not None:
        tree.nodes[ra].parent = b
    if lb is not None:
        tree.nodes[lb].parent = a
    if rb is not None:
        tree.nodes[rb].parent = a

    if tree.root == a:
        tree.root = b
    elif tree.root == b:
        tree.root = a

    na.parent, na.left, na.right = pb, lb, rb
    nb.parent, nb.left, nb.right = pa, la, ra


def delete_node(tree: BStarTree, cid: int, rng: random.Random) -> None:
    """Take a node out of the tree, keeping the rest a valid tree."""
    n = tree.nodes[cid]
    parent, left, right = n.parent, n.left, n.right

    if left is not None and right is not None:
        promoted, orphan = left, right
    elif left is not None:
        promoted, orphan = left, None
    elif right is not None:
        promoted, orphan = right, None
    else:
        promoted, orphan = None, None

    _set_parent_child(tree, parent, cid, promoted)
    if promoted is not None:
        tree.nodes[promoted].parent = parent
    if parent is None:
        tree.root = promoted
    del tree.nodes[cid]
    n.parent = n.left = n.right = None

    if orphan is not None:
        tree.nodes[orphan].parent = None
        excluded = _subtree_cids(tree, orphan)
        candidates: List[Tuple[int, str]] = []
        for t, node in tree.nodes.items():
            if t in excluded:
                continue
            if node.left is None:
                candidates.append((t, "left"))
            if node.right is None:
                candidates.append((t, "right"))
        pt, side = rng.choice(candidates)
        _attach(tree, orphan, pt, side)

    return n


def _attach(tree: BStarTree, cid: int, parent_cid: Optional[int], side: str) -> None:
    """Attach a detached node as parent_cid's side child."""
    n = tree.nodes[cid]
    assert n.parent is None
    if parent_cid is None:
        assert tree.root is None
        tree.root = cid
        return
    p = tree.nodes[parent_cid]
    assert getattr(p, side) is None
    setattr(p, side, cid)
    n.parent = parent_cid


def insert_node(tree: BStarTree, node: BNode, parent_cid: Optional[int], side: str) -> None:
    """Put a node back into the tree as parent_cid's side child."""
    assert node.cid not in tree.nodes, "node is already in the tree"
    assert node.parent is None and node.left is None and node.right is None
    tree.nodes[node.cid] = node
    _attach(tree, node.cid, parent_cid, side)


def move_node(tree: BStarTree, cid: int, new_parent_cid: int, side: str,
              rng: random.Random) -> None:
    node = delete_node(tree, cid, rng)
    insert_node(tree, node, new_parent_cid, side)


def rotate_node(tree: BStarTree, cid: int) -> None:
    """A tree rotation that lifts cid above its parent."""
    n = tree.nodes[cid]
    p_cid = n.parent
    if p_cid is None:
        return
    p = tree.nodes[p_cid]
    g_cid = p.parent

    if p.left == cid:
        b = n.right
        p.left = b
        if b is not None:
            tree.nodes[b].parent = p_cid
        n.right = p_cid
    else:
        assert p.right == cid
        b = n.left
        p.right = b
        if b is not None:
            tree.nodes[b].parent = p_cid
        n.left = p_cid

    p.parent = cid
    n.parent = g_cid

    if g_cid is None:
        tree.root = cid
    else:
        g = tree.nodes[g_cid]
        if g.left == p_cid:
            g.left = cid
        else:
            g.right = cid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The floorplanner, simulated annealing over the B*-tree and the orientations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _border_pos_from_pts(pts: List[Tuple[int, int]], L: int, t: int,
                          placement: Placement) -> Tuple[float, float]:
    transformed = [_transform_rc(r, c, L, t) for r, c in pts]
    world = [(placement.row0 + rr, placement.col0 + cc) for rr, cc in transformed]
    return world[0] if len(world) == 1 else min(world)


def _random_move(tree: BStarTree, orient: Dict[int, int], cids: List[int],
                  rng: random.Random) -> None:
    kind = rng.random()
    if kind < 0.5 or len(tree.nodes) < 3:
        # reorient, the most common and cheapest move
        c = rng.choice(cids)
        orient[c] = rng.randrange(8)
        return
    if kind < 0.70:
        for _ in range(5):
            a, b = rng.sample(cids, 2)
            na, nb = tree.nodes[a], tree.nodes[b]
            if na.parent == b or nb.parent == a:
                continue
            swap_positions(tree, a, b)
            return
        return
    if kind < 0.85:
        non_root = [c for c in cids if c != tree.root]
        if not non_root:
            return
        c = rng.choice(non_root)
        node = delete_node(tree, c, rng)
        targets: List[Tuple[int, str]] = []
        for t, nd in tree.nodes.items():
            if nd.left is None:
                targets.append((t, "left"))
            if nd.right is None:
                targets.append((t, "right"))
        pt, side = rng.choice(targets)
        insert_node(tree, node, pt, side)
        return
    non_root = [c for c in cids if c != tree.root]
    if not non_root:
        return
    rotate_node(tree, rng.choice(non_root))


def floorplan_anneal(
    embeddings: Dict[int, "ClusterEmbedding"],
    partition,
    gap: int = 2,
    n_restarts: int = 4,
    n_moves: int = 6000,
    seed: int = 0,
    lam: float = 2.0,
) -> Tuple[List[Placement], Dict[int, int], int]:
    """The B*-tree floorplanner, placing and turning the patches to minimize the area plus the length of the cut edges: (placements, orientation, L)."""
    cids = sorted(embeddings)
    if not cids:
        return [], {}, 1
    sizes = [(c, embeddings[c].L) for c in cids]
    sizes_desc = sorted(sizes, key=lambda t: -t[1])

    perim_of_L: Dict[int, Set[NodeH]] = {}
    for _, L in sizes:
        if L not in perim_of_L:
            perim_of_L[L] = _perimeter_nodes(L)

    # per cut edge, computed once
    wire_terms: List[Tuple[int, int, List[Tuple[int, int]], int, List[Tuple[int, int]], int]] = []
    for (u, v, ca, cb) in partition.cross_edges:
        if ca not in embeddings or cb not in embeddings:
            continue
        emb_a, emb_b = embeddings[ca], embeddings[cb]
        pu = [w for w in emb_a.phi.get(u, []) if w in perim_of_L[emb_a.L]] or emb_a.phi.get(u, [])
        pv = [w for w in emb_b.phi.get(v, []) if w in perim_of_L[emb_b.L]] or emb_b.phi.get(v, [])
        if not pu or not pv:
            continue
        pts_u = [oneD2twoD(w, emb_a.L) for w in pu]
        pts_v = [oneD2twoD(w, emb_b.L) for w in pv]
        wire_terms.append((ca, cb, pts_u, emb_a.L, pts_v, emb_b.L))

    def wire_cost(placements: Dict[int, Placement], orient: Dict[int, int]) -> float:
        total = 0.0
        for ca, cb, pts_u, La, pts_v, Lb in wire_terms:
            a = _border_pos_from_pts(pts_u, La, orient[ca], placements[ca])
            b = _border_pos_from_pts(pts_v, Lb, orient[cb], placements[cb])
            cheb = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            total += max(0.0, cheb - 1.0)
        return total

    def area_and_wire(tree: BStarTree, orient: Dict[int, int]):
        placements, W, H = decode(tree, gap)
        L_global = max(W, H, 1)
        return float(L_global) ** 2, wire_cost(placements, orient), placements, L_global

    best_cost = float("inf")
    best_placements: Optional[Dict[int, Placement]] = None
    best_orient: Optional[Dict[int, int]] = None
    best_L = 1

    rng_master = random.Random(seed)

    for restart in range(n_restarts):
        rng = random.Random(rng_master.randrange(1 << 30))
        tree = build_caterpillar(sizes_desc) if restart == 0 else build_random_tree(sizes, rng)
        orient = {c: 0 for c in cids}

        cur_area, cur_wire, cur_pl, cur_L = area_and_wire(tree, orient)

        # scale the wire length to weigh about the same as the area per move;
        # lam > 1 favours short wiring, < 1 compactness
        n_probe = min(60, max(10, n_moves // 10))
        probe_tree, probe_orient = tree.copy(), dict(orient)
        probe_area, probe_wire = cur_area, cur_wire
        area_deltas: List[float] = []
        wire_deltas: List[float] = []
        for _ in range(n_probe):
            _random_move(probe_tree, probe_orient, cids, rng)
            new_area, new_wire, _, _ = area_and_wire(probe_tree, probe_orient)
            area_deltas.append(abs(new_area - probe_area))
            wire_deltas.append(abs(new_wire - probe_wire))
            probe_area, probe_wire = new_area, new_wire
        mean_area_delta = (sum(area_deltas) / len(area_deltas)) if area_deltas else 1.0
        mean_wire_delta = (sum(wire_deltas) / len(wire_deltas)) if wire_deltas else 0.0
        lam_eff = lam * (mean_area_delta / mean_wire_delta) if mean_wire_delta > 1e-9 else lam

        def cost_of(area: float, wire: float, _lam: float = lam_eff) -> float:
            return area + _lam * wire

        cur_cost = cost_of(cur_area, cur_wire)
        if cur_cost < best_cost:
            best_cost, best_placements, best_orient, best_L = cur_cost, cur_pl, dict(orient), cur_L

        # T0 and T_end from the same burn-in deltas
        combined_deltas = [ad + lam_eff * wd for ad, wd in zip(area_deltas, wire_deltas)]
        mean_delta = (sum(combined_deltas) / len(combined_deltas)) if combined_deltas else 1.0
        T0 = max(mean_delta * 2.0, 1.0)
        T_end = max(T0 / 200.0, 1e-3)

        for i in range(n_moves):
            frac = i / max(1, n_moves - 1)
            T = T0 * (T_end / T0) ** frac
            backup_tree, backup_orient = tree.copy(), dict(orient)
            _random_move(tree, orient, cids, rng)
            new_area, new_wire, new_pl, new_L = area_and_wire(tree, orient)
            new_cost = cost_of(new_area, new_wire)
            delta = new_cost - cur_cost
            accept = delta <= 0 or rng.random() < math.exp(-delta / T)
            if accept:
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost, best_placements, best_orient, best_L = new_cost, new_pl, dict(orient), new_L
            else:
                tree, orient = backup_tree, backup_orient

    assert best_placements is not None and best_orient is not None
    result_placements = [best_placements[c] for c in cids]
    return result_placements, best_orient, best_L


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Routing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _astar_route(
    H: "nx.Graph", L: int, src_chain: List[NodeH], dst_chain: List[NodeH],
    free: Set[NodeH], reserved_by: Dict[NodeH, Node],
    u: Node, v: Node,
    price: Optional[Dict[NodeH, float]] = None,
) -> Optional[List[NodeH]]:
    """A* search for a path of free hardware vertices from src_chain to dst_chain, at the given prices."""
    source, target = set(src_chain), set(dst_chain)
    tgt_rc = [oneD2twoD(w, L) for w in target]

    def h(w: NodeH) -> int:
        r, c = oneD2twoD(w, L)
        return max(0, min(max(abs(r - tr), abs(c - tc)) for tr, tc in tgt_rc) - 1)

    parent: Dict[NodeH, Optional[NodeH]] = {}
    g: Dict[NodeH, float] = {}
    heap: List[Tuple[float, int, NodeH]] = []
    tie = 0
    for w in source:
        g[w] = 0.0
        parent[w] = None
        heapq.heappush(heap, (h(w), tie, w)); tie += 1

    closed: Set[NodeH] = set()
    while heap:
        _, _, a = heapq.heappop(heap)
        if a in closed:
            continue
        closed.add(a)
        for b in H.neighbors(a):
            if b in target:
                path: List[NodeH] = []
                cur = a
                while cur not in source:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            if b in closed or b not in free:
                continue
            owner = reserved_by.get(b)
            if owner is not None and owner not in (u, v):
                continue
            step = price[b] if price is not None else 1.0
            ng = g[a] + step
            if ng < g.get(b, float("inf")):
                g[b] = ng
                parent[b] = a
                heapq.heappush(heap, (ng + h(b), tie, b)); tie += 1
    return None


def _outside_patch(w: NodeH, cid: int, patch_of: Dict[int, Placement], L_global: int) -> bool:
    p = patch_of[cid]
    r, c = oneD2twoD(w, L_global)
    return not (p.row0 <= r < p.row0 + p.L and p.col0 <= c < p.col0 + p.L)


def _reserve_escapes(
    H: "nx.Graph", work: Dict[Node, List[NodeH]], endpoints: Set[Node],
    cluster_of: Dict[Node, int], placements: List[Placement], L_global: int,
    free: Set[NodeH], n_reserve: int,
) -> Dict[NodeH, Node]:
    """Each border node reserves up to n_reserve free hardware vertices next to its super vertex before routing."""
    patch_of = {p.cid: p for p in placements}
    reserved_by: Dict[NodeH, Node] = {}
    for x in sorted(endpoints, key=str):
        cid = cluster_of[x]
        cands = [w for a in work[x] for w in H.neighbors(a) if w in free and w not in reserved_by]
        cands.sort(key=lambda w: (not _outside_patch(w, cid, patch_of, L_global), w))
        for w in cands[:n_reserve]:
            reserved_by[w] = x
    return reserved_by


def _route_all(
    H: "nx.Graph", phi: Dict[Node, List[NodeH]], cut_edges: List[Edge],
    free0: Set[NodeH], cluster_of: Dict[Node, int], placements: List[Placement],
    L_global: int, sep_fn, n_reserve: int = 2,
    reserved_by: Optional[Dict[NodeH, Node]] = None,
):
    """Route every cut edge in several orders, ripping up and retrying, and keep the best."""
    endpoints = {x for e in cut_edges for x in e}

    orderings = [
        sorted(cut_edges, key=lambda e: sep_fn(*e)),
        sorted(cut_edges, key=lambda e: -sep_fn(*e)),
    ]
    for s in (0, 1, 2, 3, 4):
        rng = random.Random(s)
        o = list(cut_edges)
        rng.shuffle(o)
        orderings.append(o)

    best = None
    for order in orderings:
        free = set(free0)
        work = {k: list(val) for k, val in phi.items()}

        if reserved_by is not None:
            rb = dict(reserved_by)
        else:
            rb = _reserve_escapes(H, work, endpoints, cluster_of, placements, L_global, free, n_reserve)

        routes: Dict[Edge, List[NodeH]] = {}
        unrouted: List[Edge] = []

        for u, v in order:
            if any(H.has_edge(a, b) for a in work[u] for b in work[v]):
                continue
            path = _astar_route(H, L_global, work[u], work[v], free, rb, u, v)
            owner = u
            if path is None:
                path = _astar_route(H, L_global, work[v], work[u], free, rb, v, u)
                owner = v
            if path is None:
                unrouted.append((u, v))
                continue
            for w in path:
                work[owner].append(w)
                free.discard(w)
                rb.pop(w, None)
            routes[(u, v)] = path

        total_len = sum(len(p) for p in routes.values())
        key = (len(unrouted), total_len)
        if best is None or key < best[0]:
            best = (key, routes, unrouted, work)

    _, routes, unrouted, work = best
    return routes, unrouted, work


class _PriceOnDemand(dict):
    """PathFinder's prices, computed only for the hardware vertices A* reaches."""

    def __init__(self, price_fn):
        super().__init__()
        self._fn = price_fn

    def __missing__(self, w):
        self[w] = p = self._fn(w)
        return p


def route_negotiated(
    H: "nx.Graph", phi: Dict[Node, List[NodeH]], cut_edges: List[Edge],
    free0: Set[NodeH], cluster_of: Dict[Node, int], placements: List[Placement],
    L_global: int, sep_fn, n_reserve: int = 2, max_rounds: int = 20,
    hist_step: float = 1.0, pres_fac0: float = 1.0, pres_fac_mult: float = 1.35,
):
    """PathFinder, where in every round the hardware vertices that several routes compete for become more expensive, until every cut edge has a path of its own: (routes, unrouted, phi_out)."""
    work = {k: list(v) for k, v in phi.items()}
    free = set(free0)

    remaining: List[Edge] = []
    for (u, v) in cut_edges:
        if not any(H.has_edge(a, b) for a in work[u] for b in work[v]):
            remaining.append((u, v))

    endpoints = {x for e in cut_edges for x in e}
    reserved_by = _reserve_escapes(H, work, endpoints, cluster_of, placements, L_global, free, n_reserve)

    if not remaining:
        return {}, [], work

    hist: Dict[NodeH, float] = {}
    route_of: Dict[Edge, List[NodeH]] = {}
    owner_of: Dict[Edge, Node] = {}
    claim: Dict[NodeH, Set[Edge]] = {}
    pres_fac = pres_fac0
    pending: Set[Edge] = set(remaining)

    def release(e: Edge) -> None:
        path = route_of.pop(e, None)
        owner_of.pop(e, None)
        if not path:
            return
        for w in path:
            s = claim.get(w)
            if s is not None:
                s.discard(e)
                if not s:
                    del claim[w]

    def price_fn(w: NodeH) -> float:
        # occ counts the other routes on w
        occ = len(claim.get(w, ()))
        return (1.0 + hist.get(w, 0.0)) * (1.0 + pres_fac * occ)

    for _round in range(max_rounds):
        for e in sorted(pending, key=lambda e: sep_fn(*e)):
            release(e)
            u, v = e
            # priced only where A* reaches
            price = _PriceOnDemand(price_fn)
            path = _astar_route(H, L_global, work[u], work[v], free, reserved_by, u, v, price=price)
            owner = u
            if path is None:
                path = _astar_route(H, L_global, work[v], work[u], free, reserved_by, v, u, price=price)
                owner = v
            if path is None:
                continue
            route_of[e] = path
            owner_of[e] = owner
            for w in path:
                claim.setdefault(w, set()).add(e)

        conflicted = {w for w, s in claim.items() if len(s) > 1}
        if not conflicted:
            break
        for w in conflicted:
            hist[w] = hist.get(w, 0.0) + hist_step
        pres_fac *= pres_fac_mult
        pending = set()
        for w in conflicted:
            pending |= claim[w]

    conflicted = {w for w, s in claim.items() if len(s) > 1}
    unresolved: Set[Edge] = set()
    for w in conflicted:
        unresolved |= claim[w]
    for e in remaining:
        if e not in route_of:
            unresolved.add(e)

    resolved_routes: Dict[Edge, List[NodeH]] = {}
    for e in remaining:
        if e in unresolved or e not in route_of:
            continue
        path, owner = route_of[e], owner_of[e]
        for w in path:
            work[owner].append(w)
            free.discard(w)
            reserved_by.pop(w, None)
        resolved_routes[e] = path

    if unresolved:
        fb_routes, fb_unrouted, work = _route_all(
            H, work, list(unresolved), free, cluster_of, placements, L_global, sep_fn,
            n_reserve=n_reserve, reserved_by=reserved_by,
        )
        resolved_routes.update(fb_routes)
        unrouted = fb_unrouted
    else:
        unrouted = []

    return resolved_routes, unrouted, work


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The same in C++ (floorplan_core)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    import numpy as np
    import floorplan_core as _core
    _HAVE_FLOORPLAN_CORE = True
except ImportError:
    _core = None
    _HAVE_FLOORPLAN_CORE = False


def _require_core() -> None:
    if not _HAVE_FLOORPLAN_CORE:
        raise RuntimeError(
            "floorplan_core is not built: run `python build.py` in the repo root, "
            "or use the pure-Python floorplan_anneal() / route_negotiated()."
        )


def floorplan_anneal_cpp(
    embeddings: Dict[int, "ClusterEmbedding"],
    partition,
    gap: int = 2,
    n_restarts: int = 4,
    n_moves: int = 6000,
    seed: int = 0,
    lam: float = 2.0,
) -> Tuple[List[Placement], Dict[int, int], int]:
    """floorplan_anneal() with the annealing in C++."""
    _require_core()
    cids = sorted(embeddings)
    if not cids:
        return [], {}, 1
    index = {c: i for i, c in enumerate(cids)}

    perim_of_L: Dict[int, Set[NodeH]] = {}
    for c in cids:
        if embeddings[c].L not in perim_of_L:
            perim_of_L[embeddings[c].L] = _perimeter_nodes(embeddings[c].L)

    wire_terms = []
    for (u, v, ca, cb) in partition.cross_edges:
        if ca not in embeddings or cb not in embeddings:
            continue
        emb_a, emb_b = embeddings[ca], embeddings[cb]
        pu = [w for w in emb_a.phi.get(u, []) if w in perim_of_L[emb_a.L]] or emb_a.phi.get(u, [])
        pv = [w for w in emb_b.phi.get(v, []) if w in perim_of_L[emb_b.L]] or emb_b.phi.get(v, [])
        if not pu or not pv:
            continue
        pts_u = [oneD2twoD(w, emb_a.L) for w in pu]
        pts_v = [oneD2twoD(w, emb_b.L) for w in pv]
        wire_terms.append((index[ca], index[cb], pts_u, pts_v))

    # random.Random(seed) seeds with abs(seed)
    rows, cols, orient, L_global = _core.anneal(
        [embeddings[c].L for c in cids], wire_terms, gap, n_restarts, n_moves, abs(seed), lam)
    placements = [Placement(cid=c, L=embeddings[c].L, row0=r, col0=q)
                  for c, r, q in zip(cids, rows, cols)]
    return placements, dict(zip(cids, orient)), L_global


def _free_array(free: Set[NodeH], L: int) -> "np.ndarray":
    """The free hardware vertices as an array, 1 = free."""
    arr = np.zeros(L * L + 1, dtype=np.uint8)
    arr[list(free)] = 1
    return arr


def _owner_array(reserved_by: Dict[NodeH, Node], node_id: Dict[Node, int], L: int) -> "np.ndarray":
    """The reservations as an array, the owner's id or -1."""
    arr = np.full(L * L + 1, -1, dtype=np.int64)
    for w, x in reserved_by.items():
        arr[w] = node_id[x]
    return arr


def _route_all_cpp(
    H: "nx.Graph", phi: Dict[Node, List[NodeH]], cut_edges: List[Edge],
    free0: Set[NodeH], cluster_of: Dict[Node, int], placements: List[Placement],
    L_global: int, sep_fn, n_reserve: int = 2,
    reserved_by: Optional[Dict[NodeH, Node]] = None,
):
    """_route_all() with each A* search in C++."""
    _require_core()
    endpoints = {x for e in cut_edges for x in e}
    node_id = {x: i for i, x in enumerate(phi)}

    orderings = [
        sorted(cut_edges, key=lambda e: sep_fn(*e)),
        sorted(cut_edges, key=lambda e: -sep_fn(*e)),
    ]
    for s in (0, 1, 2, 3, 4):
        rng = random.Random(s)
        o = list(cut_edges)
        rng.shuffle(o)
        orderings.append(o)

    best = None
    for order in orderings:
        free = set(free0)
        work = {k: list(val) for k, val in phi.items()}

        if reserved_by is not None:
            rb = dict(reserved_by)
        else:
            rb = _reserve_escapes(H, work, endpoints, cluster_of, placements, L_global, free, n_reserve)
        free_arr = _free_array(free, L_global)
        rb_arr = _owner_array(rb, node_id, L_global)

        routes: Dict[Edge, List[NodeH]] = {}
        unrouted: List[Edge] = []

        for u, v in order:
            if any(H.has_edge(a, b) for a in work[u] for b in work[v]):
                continue
            path = _core.astar(L_global, list(set(work[u])), work[v], free_arr, rb_arr,
                               node_id[u], node_id[v], None, None, 0.0)
            owner = u
            if path is None:
                path = _core.astar(L_global, list(set(work[v])), work[u], free_arr, rb_arr,
                                   node_id[v], node_id[u], None, None, 0.0)
                owner = v
            if path is None:
                unrouted.append((u, v))
                continue
            for w in path:
                work[owner].append(w)
                free.discard(w)
                free_arr[w] = 0
                rb.pop(w, None)
                rb_arr[w] = -1
            routes[(u, v)] = path

        total_len = sum(len(p) for p in routes.values())
        key = (len(unrouted), total_len)
        if best is None or key < best[0]:
            best = (key, routes, unrouted, work)

    _, routes, unrouted, work = best
    return routes, unrouted, work


def route_negotiated_cpp(
    H: "nx.Graph", phi: Dict[Node, List[NodeH]], cut_edges: List[Edge],
    free0: Set[NodeH], cluster_of: Dict[Node, int], placements: List[Placement],
    L_global: int, sep_fn, n_reserve: int = 2, max_rounds: int = 20,
    hist_step: float = 1.0, pres_fac0: float = 1.0, pres_fac_mult: float = 1.35,
):
    """route_negotiated() with each A* search in C++."""
    _require_core()
    work = {k: list(v) for k, v in phi.items()}
    free = set(free0)

    remaining: List[Edge] = []
    for (u, v) in cut_edges:
        if not any(H.has_edge(a, b) for a in work[u] for b in work[v]):
            remaining.append((u, v))

    endpoints = {x for e in cut_edges for x in e}
    reserved_by = _reserve_escapes(H, work, endpoints, cluster_of, placements, L_global, free, n_reserve)

    if not remaining:
        return {}, [], work

    node_id = {x: i for i, x in enumerate(work)}
    free_arr = _free_array(free, L_global)
    reserved_arr = _owner_array(reserved_by, node_id, L_global)
    hist = np.zeros(L_global * L_global + 1)
    occ = np.zeros(L_global * L_global + 1, dtype=np.int32)
    route_of: Dict[Edge, List[NodeH]] = {}
    owner_of: Dict[Edge, Node] = {}
    claim: Dict[NodeH, Set[Edge]] = {}
    pres_fac = pres_fac0
    pending: Set[Edge] = set(remaining)

    def release(e: Edge) -> None:
        path = route_of.pop(e, None)
        owner_of.pop(e, None)
        if not path:
            return
        for w in path:
            s = claim.get(w)
            if s is not None:
                s.discard(e)
                occ[w] = len(s)
                if not s:
                    del claim[w]

    for _round in range(max_rounds):
        for e in sorted(pending, key=lambda e: sep_fn(*e)):
            release(e)
            u, v = e
            path = _core.astar(L_global, list(set(work[u])), work[v], free_arr, reserved_arr,
                               node_id[u], node_id[v], hist, occ, pres_fac)
            owner = u
            if path is None:
                path = _core.astar(L_global, list(set(work[v])), work[u], free_arr, reserved_arr,
                                   node_id[v], node_id[u], hist, occ, pres_fac)
                owner = v
            if path is None:
                continue
            route_of[e] = path
            owner_of[e] = owner
            for w in path:
                s = claim.setdefault(w, set())
                s.add(e)
                occ[w] = len(s)

        conflicted = {w for w, s in claim.items() if len(s) > 1}
        if not conflicted:
            break
        for w in conflicted:
            hist[w] += hist_step
        pres_fac *= pres_fac_mult
        pending = set()
        for w in conflicted:
            pending |= claim[w]

    conflicted = {w for w, s in claim.items() if len(s) > 1}
    unresolved: Set[Edge] = set()
    for w in conflicted:
        unresolved |= claim[w]
    for e in remaining:
        if e not in route_of:
            unresolved.add(e)

    resolved_routes: Dict[Edge, List[NodeH]] = {}
    for e in remaining:
        if e in unresolved or e not in route_of:
            continue
        path, owner = route_of[e], owner_of[e]
        for w in path:
            work[owner].append(w)
            free.discard(w)
            reserved_by.pop(w, None)
        resolved_routes[e] = path

    if unresolved:
        fb_routes, fb_unrouted, work = _route_all_cpp(
            H, work, list(unresolved), free, cluster_of, placements, L_global, sep_fn,
            n_reserve=n_reserve, reserved_by=reserved_by,
        )
        resolved_routes.update(fb_routes)
        unrouted = fb_unrouted
    else:
        unrouted = []

    return resolved_routes, unrouted, work
