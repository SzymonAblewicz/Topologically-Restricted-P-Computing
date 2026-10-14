"""
The construction of the fallback method (Section 4.1 of the report). The patches
of the clusters are placed in a grid above the routing structure, and every
border node runs down its own lane to its own chain of the routing structure,
where every cut edge is realized, with no search at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from pssa_wbc import build_guiding_pattern, oneD2twoD, twoD2oneD
from place_and_route import Placement

Node = Any
NodeH = int
Edge = Tuple[Node, Node]
RC = Tuple[int, int]

# The sides of a patch, in the order a border node prefers to leave through them.
_BOTTOM, _LEFT, _RIGHT, _TOP = 0, 1, 2, 3


@dataclass
class _Lane:
    """One border node's lane, from its patch to its chain of the routing structure."""
    node: Node
    cid: int
    side: int
    dock: RC                 # its perimeter vertex, already in its super vertex
    turn_row: int            # row the horizontal run uses (patch row, or top-band row)
    gap_key: Optional[Tuple[int, str]] = None
    turn_col: int = 0        # column the vertical descent uses
    cells: List[RC] = field(default_factory=list)   # from the patch to the entrance of the routing structure
    stripe: int = -1
    trunk: List[RC] = field(default_factory=list)   # its chain, cut after its last contact


@dataclass
class GuaranteedPlan:
    placements: List[Placement]
    orient: Dict[int, int]
    L: int
    lanes: Dict[Node, _Lane]
    contacts: Dict[Edge, Tuple[int, int]]   # cut edge -> (index in trunk[u], index in trunk[v])
    stats: Dict[str, int]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The perimeter vertex each border node leaves through
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _perimeter_side(r: int, c: int, L: int) -> Optional[int]:
    """The side of an L x L patch that cell (r, c) lies on (a corner goes to the cheaper side)."""
    if L == 1:
        return _BOTTOM
    if r == L - 1:
        return _BOTTOM
    if c == 0:
        return _LEFT
    if c == L - 1:
        return _RIGHT
    if r == 0:
        return _TOP
    return None


def _pick_docks(embeddings, partition) -> Dict[int, List[Tuple[Node, int, RC]]]:
    """For every cluster, the perimeter vertex each border node leaves its patch through, on its cheapest side."""
    out: Dict[int, List[Tuple[Node, int, RC]]] = {}
    for cid in sorted(embeddings):
        emb = embeddings[cid]
        L = emb.L
        picks: List[Tuple[Node, int, RC]] = []
        for u in sorted(partition.border_nodes.get(cid, ()), key=str):
            best: Optional[Tuple[int, RC]] = None
            for w in emb.phi.get(u, ()):
                rc = oneD2twoD(w, L)
                side = _perimeter_side(rc[0], rc[1], L)
                if side is None:
                    continue
                if best is None or (side, rc) < (best[0], best[1]):
                    best = (side, rc)
            if best is None:
                raise RuntimeError(
                    f"border node {u!r} of cluster {cid} has no perimeter qubit; "
                    f"Stage 2 must run with border_mode='permute' so every border "
                    f"node keeps a perimeter touch."
                )
            picks.append((u, best[0], best[1]))
        out[cid] = picks
    return out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The grid of patches
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class _Grid:
    """Where every patch sits, and the rows and columns its lanes may use."""
    Gr: int                                   # grid rows of patches
    Gc: int                                   # grid columns of patches
    cell_of: Dict[int, Tuple[int, int]]       # cid -> (grid row, grid col)
    top0: List[int]                           # grid row -> first row of its top band
    bot0: List[int]                           # grid row -> first row of its bottom band
    tb: List[int]                             # grid row -> top band height
    bb: List[int]                             # grid row -> bottom band height
    halves: Dict[Tuple[int, str], Tuple[int, int]]   # (corridor, half) -> (col0, width)
    band_bot: int                             # first row strictly below every patch
    row_width: int
    patch_cols: int                           # sum of patch-column widths
    patch_rows: int                           # sum of patch-row heights
    n_top: int                                # top-side docks, for reporting


def _dock_route(side: int, rc: RC, L: int, i: int, Gr: int, j: int):
    """Where a lane turns, and into which half of which corridor."""
    if side == _LEFT:
        return "side", (j, "R")
    if side == _RIGHT:
        return "side", (j + 1, "L")
    nearer = (j, "R") if rc[1] * 2 < L else (j + 1, "L")
    if side == _TOP:
        return "top", nearer
    if i == Gr - 1:
        return "free", None
    return "bot", nearer


def _place(embeddings, docks) -> Tuple[List[Placement], Dict[int, int], _Grid]:
    """The grid of patches, in order of size, with every corridor and band sized to exactly the number of lanes that pass through it."""
    # In order of size, smallest first, so that every row holds patches of a similar height.
    cids = sorted(embeddings, key=lambda c: (embeddings[c].L, c))
    k = len(cids)
    L_of = {c: embeddings[c].L for c in cids}

    # A near-square grid, wider rather than taller on a tie, as the height is always the larger side.
    Gc = math.isqrt(k)
    if Gc * Gc < k:
        Gc += 1
    Gr = -(-k // Gc)                     # ceil; the last grid row may be short

    cell_of: Dict[int, Tuple[int, int]] = {}
    grid: Dict[Tuple[int, int], int] = {}
    for idx, cid in enumerate(cids):
        i, j = divmod(idx, Gc)
        cell_of[cid] = (i, j)
        grid[(i, j)] = cid

    # Each column as wide as its widest patch, each row as tall as its tallest.
    Wp = [max(L_of[grid[(i, j)]] for i in range(Gr) if (i, j) in grid) for j in range(Gc)]
    Hp = [max(L_of[grid[(i, j)]] for j in range(Gc) if (i, j) in grid) for i in range(Gr)]

    demand: Dict[Tuple[int, str], int] = {}
    tb = [0] * Gr
    bb = [0] * Gr
    n_top = 0
    for cid in cids:
        i, j = cell_of[cid]
        L = L_of[cid]
        # A band is as deep as the busiest patch of its grid row needs.
        per = {"top": {"L": 0, "R": 0}, "bot": {"L": 0, "R": 0}}
        for _u, side, rc in docks[cid]:
            kind, key = _dock_route(side, rc, L, i, Gr, j)
            if kind == "free":
                continue
            demand[key] = demand.get(key, 0) + 1
            if kind == "top":
                n_top += 1
            if kind in ("top", "bot"):
                per[kind][key[1]] += 1
        tb[i] = max(tb[i], per["top"]["L"], per["top"]["R"])
        bb[i] = max(bb[i], per["bot"]["L"], per["bot"]["R"])

    top0 = [0] * Gr
    bot0 = [0] * Gr
    rowbase = [0] * Gr
    row = 0
    for i in range(Gr):
        top0[i] = row
        row += tb[i]
        rowbase[i] = row
        row += Hp[i]
        bot0[i] = row
        row += bb[i]
    band_bot = row                       # first row strictly below every patch

    halves: Dict[Tuple[int, str], Tuple[int, int]] = {}
    colbase = [0] * Gc
    col = 0
    for j in range(Gc + 1):
        lh = demand.get((j, "L"), 0)
        rh = demand.get((j, "R"), 0)
        width = max(lh + rh, 1)          # never let two patch columns touch
        halves[(j, "L")] = (col, lh)
        halves[(j, "R")] = (col + lh, rh)
        col += width
        if j < Gc:
            colbase[j] = col
            col += Wp[j]
    row_width = col

    placements = [Placement(cid=cid, L=L_of[cid],
                            row0=rowbase[cell_of[cid][0]],
                            col0=colbase[cell_of[cid][1]])
                  for cid in cids]

    return placements, {c: 0 for c in cids}, _Grid(
        Gr=Gr, Gc=Gc, cell_of=cell_of, top0=top0, bot0=bot0, tb=tb, bb=bb,
        halves=halves, band_bot=band_bot, row_width=row_width,
        patch_cols=sum(Wp), patch_rows=sum(Hp), n_top=n_top,
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The lanes, ordered so that no two ever meet
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _assign_lanes(cids, docks, placements, grid: _Grid,
                  with_cells: bool = True) -> Dict[Node, _Lane]:
    """The turn row and turn column of every lane, ordered so that no two lanes ever meet."""
    patch_of = {p.cid: p for p in placements}
    lanes: Dict[Node, _Lane] = {}
    buckets: Dict[Tuple[int, str], List[_Lane]] = {}

    for cid in cids:
        p = patch_of[cid]
        L = p.L
        i, j = grid.cell_of[cid]
        band: Dict[Tuple[str, str], List[Tuple[Node, int, RC, Tuple[int, str]]]] = {}

        for u, side, rc in docks[cid]:
            kind, key = _dock_route(side, rc, L, i, grid.Gr, j)
            if kind == "free":
                lanes[u] = _Lane(node=u, cid=cid, side=_BOTTOM,
                                 dock=(p.row0 + rc[0], p.col0 + rc[1]),
                                 turn_row=p.row0 + rc[0], turn_col=p.col0 + rc[1])
            elif kind == "side":
                lanes[u] = _Lane(node=u, cid=cid, side=side,
                                 dock=(p.row0 + rc[0],
                                       p.col0 + (L - 1 if side == _RIGHT else 0)),
                                 turn_row=p.row0 + rc[0], gap_key=key)
            else:
                band.setdefault((kind, key[1]), []).append((u, side, rc, key))

        for (kind, half), group in band.items():
            # The lane furthest from its corridor first.
            def farthest_first(t, half=half):
                return -t[2][1] if half == "R" else t[2][1]
            group.sort(key=farthest_first if kind == "top"
                       else (lambda t, f=farthest_first: -f(t)))
            base = grid.top0[i] if kind == "top" else grid.bot0[i]
            for m, (u, side, rc, key) in enumerate(group):
                dock = (p.row0, p.col0 + rc[1]) if kind == "top" \
                    else (p.row0 + rc[0], p.col0 + rc[1])
                lanes[u] = _Lane(node=u, cid=cid, side=side, dock=dock,
                                 turn_row=base + m, gap_key=key)

        for u, _side, _rc in docks[cid]:
            ln = lanes[u]
            if ln.gap_key is not None:
                buckets.setdefault(ln.gap_key, []).append(ln)

    for key, group in buckets.items():
        col0, width = grid.halves[key]
        group.sort(key=lambda ln: ln.turn_row)
        if len(group) > width:
            raise AssertionError(f"corridor half {key} sized {width} for {len(group)} lanes")
        for m, ln in enumerate(group):
            # Outward is a larger column in an "L" half, a smaller one in an "R" half.
            ln.turn_col = (col0 + width - 1 - m) if key[1] == "L" else (col0 + m)

    if with_cells:
        for ln in lanes.values():
            ln.cells = _lane_cells(ln, grid.band_bot)
    return lanes


def _lane_cells(ln: _Lane, band_bot: int) -> List[RC]:
    """The hardware vertices of one lane, from its patch down to the bottom of the grid."""
    r, c = ln.dock
    free_bottom = ln.side == _BOTTOM and ln.gap_key is None
    cells: List[RC] = []
    if ln.side == _TOP:
        cells += [(rr, c) for rr in range(r - 1, ln.turn_row - 1, -1)]
        cells += _run(ln.turn_row, c, ln.turn_col)
    elif ln.side == _LEFT:
        cells += [(r, cc) for cc in range(c - 1, ln.turn_col - 1, -1)]
    elif ln.side == _RIGHT:
        cells += [(r, cc) for cc in range(c + 1, ln.turn_col + 1)]
    elif not free_bottom:
        cells += [(rr, c) for rr in range(r + 1, ln.turn_row + 1)]
        cells += _run(ln.turn_row, c, ln.turn_col)
    start = r + 1 if free_bottom else ln.turn_row + 1
    cells += [(rr, ln.turn_col) for rr in range(start, band_bot)]
    return cells


def _run(row: int, c_from: int, c_to: int) -> List[RC]:
    """The horizontal part of a lane, from just outside c_from to c_to."""
    step = -1 if c_to < c_from else 1
    return [(row, cc) for cc in range(c_from + step, c_to + step, step)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The funnel and the routing structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _plaza_stripes(S: int) -> List[List[RC]]:
    """The routing structure, the square pattern of Okuyama et al. (2016) with its first super vertex dropped and transposed."""
    gp = build_guiding_pattern(None, S)
    return [[(c, r) for (r, c) in (oneD2twoD(w, S) for w in gp[i])]
            for i in range(2, S + 2)]


def _contacts(stripes: Sequence[Sequence[RC]], pairs: Sequence[Tuple[int, int]]):
    """Where the chains of each cut edge first meet, and so how much of each chain to keep."""
    owner: Dict[RC, Tuple[int, int]] = {}
    for j, ch in enumerate(stripes):
        for idx, cell in enumerate(ch):
            owner[cell] = (j, idx)

    best: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    wanted = {(min(a, b), max(a, b)) for a, b in pairs}
    for j, ch in enumerate(stripes):
        for idx, (r, c) in enumerate(ch):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    o = owner.get((r + dr, c + dc))
                    if o is None or o[0] == j:
                        continue
                    j2, idx2 = o
                    key = (min(j, j2), max(j, j2))
                    if key not in wanted:
                        continue
                    lo, hi = (idx, idx2) if j < j2 else (idx2, idx)
                    cand = (max(idx, idx2), lo, hi)
                    if key not in best or cand[0] < best[key][0]:
                        best[key] = cand

    missing = wanted - set(best)
    if missing:
        raise AssertionError(
            f"{len(missing)} demanded stripe pair(s) never touch, e.g. "
            f"{sorted(missing)[:3]} -- the guiding pattern's completeness "
            f"guarantee does not hold at this size."
        )
    return best


def _plaza(lanes, cut_edges, band_bot, row_width):
    """The funnel below the grid, and every lane's own chain of the routing structure."""
    ordered = sorted(lanes.values(), key=lambda ln: ln.turn_col)
    T = len(ordered)
    a = [ln.turn_col for ln in ordered]
    if len(set(a)) != T:
        raise AssertionError("escape columns are not distinct")

    S = max(T, 3)
    # The chains centred under the grid, so that no lane drifts further than it has to.
    shifts = sorted(a[i] - i for i in range(T))
    X0 = shifts[T // 2]
    X0 = max(0, min(X0, max(row_width - S, 0)))
    b = [X0 + i for i in range(T)]
    F = max(1, max(abs(a[i] - b[i]) for i in range(T)))
    rail = band_bot - 1 + F

    for i, ln in enumerate(ordered):
        d = a[i] - b[i]
        sgn = (d > 0) - (d < 0)
        for t in range(1, F + 1):
            ln.cells.append((band_bot - 1 + t, b[i] + sgn * max(abs(d) - t, 0)))
        ln.stripe = i

    stripes = _plaza_stripes(S)
    demand = []
    for u, v in cut_edges:
        demand.append((lanes[u].stripe, lanes[v].stripe))
    best = _contacts(stripes, demand)

    reach = [0] * S
    contacts: Dict[Edge, Tuple[int, int]] = {}
    for (u, v) in cut_edges:
        ju, jv = lanes[u].stripe, lanes[v].stripe
        _, lo, hi = best[(min(ju, jv), max(ju, jv))]
        iu, iv = (lo, hi) if ju < jv else (hi, lo)
        contacts[(u, v)] = (iu, iv)
        reach[ju] = max(reach[ju], iu)
        reach[jv] = max(reach[jv], iv)

    for ln in ordered:
        j = ln.stripe
        ln.trunk = [(rail + r, X0 + c) for (r, c) in stripes[j][:reach[j] + 1]]

    # depth = the grid and its bands + the funnel + the routing structure.
    plaza_depth = max(reach) + 1
    depth = rail + 1 + plaza_depth
    width = max(row_width, X0 + S)
    return contacts, max(width, depth), {
        "T": T, "S": S, "funnel": F, "X0": X0, "plaza_depth": plaza_depth,
        "width": width, "height": depth,
        "max_lane": max(len(l.trunk) for l in ordered),
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Public entry points
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def guaranteed_plan(embeddings, partition) -> GuaranteedPlan:
    """The whole construction, from the partition and the patches."""
    cids = sorted(embeddings)
    if not cids:
        return GuaranteedPlan([], {}, 1, {}, {}, {})

    docks = _pick_docks(embeddings, partition)
    placements, orient, grid = _place(embeddings, docks)
    lanes = _assign_lanes(cids, docks, placements, grid)

    cut_edges = [(u, v) for u, v, _, _ in partition.cross_edges]
    contacts, L, stats = _plaza(lanes, cut_edges, grid.band_bot, grid.row_width)
    stats.update({
        "Gr": grid.Gr, "Gc": grid.Gc, "n_top": grid.n_top,
        "row_width": grid.row_width, "band_bot": grid.band_bot,
        "patch_cols": grid.patch_cols, "corridor_cols": grid.row_width - grid.patch_cols,
        "patch_rows": grid.patch_rows, "band_rows": sum(grid.tb) + sum(grid.bb),
    })
    return GuaranteedPlan(placements, orient, L, lanes, contacts, stats)


def guaranteed_floorplan(embeddings, partition, plan: Optional[GuaranteedPlan] = None):
    """The placement of the patches: (placements, orient, L)."""
    plan = plan or guaranteed_plan(embeddings, partition)
    return plan.placements, plan.orient, plan.L


def guaranteed_route(H, phi, cut_edges, plan: GuaranteedPlan):
    """Every lane and chain, checked to be disjoint: (routes, unrouted, phi_out), with unrouted always empty."""
    L = plan.L
    work = {kk: list(vv) for kk, vv in phi.items()}

    claimed: Dict[NodeH, Node] = {}
    for v, chain in work.items():
        for w in chain:
            claimed[w] = v

    trunk_ids: Dict[Node, List[NodeH]] = {}
    for u, ln in plan.lanes.items():
        ids = [twoD2oneD(r, c, L) for (r, c) in ln.cells + ln.trunk]
        for w in ids:
            prev = claimed.get(w)
            if prev is not None:
                raise AssertionError(
                    f"lane for {u!r} collides at hw {w} (held by {prev!r}) -- "
                    f"the construction's disjointness invariant is broken"
                )
            claimed[w] = u
        work[u].extend(ids)
        trunk_ids[u] = ids[len(ln.cells):]

    for u, ln in plan.lanes.items():
        if not nx.is_connected(H.subgraph(work[u])):
            raise AssertionError(f"lane for {u!r} left its chain disconnected")

    routes: Dict[Edge, List[NodeH]] = {}
    for (u, v) in cut_edges:
        if any(H.has_edge(x, y) for x in phi[u] for y in phi[v]):
            continue
        iu, iv = plan.contacts[(u, v)]
        tu, tv = trunk_ids[u], trunk_ids[v]
        if not H.has_edge(tu[iu], tv[iv]):
            raise AssertionError(f"lanes for cut edge {(u, v)!r} do not touch")
        routes[(u, v)] = tu[:iu + 1] + tv[:iv + 1][::-1]

    return routes, [], work
