"""
The construction of the fallback method (construction.py) with the chains of the
routing structure left unbuilt, so that it gives the size of the embedding for very
large graphs. Each chain has one hardware vertex per row and a closed form, so it is
generated only when asked for, and the layout is the same, cell for cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kings_lattice import KingsLattice, Occupancy, rc_of
from construction import (
    _Lane,
    GuaranteedPlan,
    _assign_lanes,
    _lane_cells,
    _pick_docks,
    _place,
)

Node = Any
NodeH = int
Edge = Tuple[Node, Node]
RC = Tuple[int, int]


@dataclass
class FastPlan(GuaranteedPlan):
    """A GuaranteedPlan, plus what it takes to generate any chain of the routing structure on demand."""
    rail: int = 0  # row of chain index 0
    X0: int = 0  # column where the chains start
    S: int = 3  # number of chains
    F: int = 1                    # funnel depth
    band_bot: int = 0             # first row strictly below every patch
    reach: List[int] = field(default_factory=list)  # per chain, last index kept


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The routing structure, in closed form
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def stripe_cols(j, k, S: int):
    """The column of chain j at index k - 1, in closed form."""
    j = np.asarray(j, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    even = (j & 1) == 0
    x = np.where(even, j + k - 1, j - k + 1)
    x = np.where(even & (x >= S), 2 * S - x - 1, x)
    x = np.where(~even & (x < 0), -x - 1, x)
    return x


def stripe_cells(j: int, upto: int, S: int) -> np.ndarray:
    """Cells 0..upto of chain j, as (row, col)."""
    idx = np.arange(upto + 1, dtype=np.int64)
    return np.stack([idx + 1, stripe_cols(j, idx + 1, S)], axis=1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Where the chains meet, only for the pairs with a cut edge
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def contact_lower_bound(A, B, S: int) -> np.ndarray:
    """For chain pairs (A, B), the smallest index at which they could meet."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    P = 2 * S
    never = np.iinfo(np.int64).max
    sa = np.where(A % 2 == 0, 1, -1)
    sb = np.where(B % 2 == 0, 1, -1)
    out = np.full(A.shape, never, dtype=np.int64)
    for c1, step in ((A - B, sa - sb), (A + B - (P - 1), sa + sb)):
        r = c1 % P
        t = np.where((r <= 2) | (r >= P - 2), 0,
                     np.where(step > 0, (P - 1 - r) // 2,
                              np.where(step < 0, (r - 1) // 2, never)))
        out = np.minimum(out, t)
    return out


def contacts_fast(S: int, pairs: Sequence[Tuple[int, int]],
                  window: int = 8) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """construction._contacts in closed form, {(j_lo, j_hi): (max_idx, idx_on_lo, idx_on_hi)}."""
    wanted = sorted({(min(a, b), max(a, b)) for a, b in pairs})
    if not wanted:
        return {}

    A = np.array([p[0] for p in wanted], dtype=np.int64)
    B = np.array([p[1] for p in wanted], dtype=np.int64)
    best: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    m_top = S - 1                       # valid idx range is 0 .. S-2
    start = np.maximum(contact_lower_bound(A, B, S) - 1, 0)
    live = np.nonzero(start < m_top)[0]
    lost = [np.nonzero(start >= m_top)[0]]
    W = np.arange(window, dtype=np.int64)
    while live.size:
        Ms = start[live][:, None] + W[None, :]                         # (nl,W)
        a = A[live][:, None]                                           # (nl,1)
        b = B[live][:, None]
        ok = Ms < m_top
        prev = ok & (Ms >= 1)

        xa_m = stripe_cols(a, Ms + 1, S)        # idx = M
        xb_m = stripe_cols(b, Ms + 1, S)
        xa_p = stripe_cols(a, Ms, S)            # idx = M-1 (masked where M = 0)
        xb_p = stripe_cols(b, Ms, S)
        hit = np.stack([prev & (np.abs(xa_p - xb_m) <= 1),
                        prev & (np.abs(xa_m - xb_p) <= 1),
                        ok & (np.abs(xa_m - xb_m) <= 1)], axis=2)

        any_hit = hit.any(axis=2)
        got = any_hit.any(axis=1)
        first = np.argmax(any_hit, axis=1)
        for t in np.nonzero(got)[0]:
            jm = int(first[t])
            M = int(Ms[t, jm])
            kind = int(np.argmax(hit[t, jm]))
            lo, hi = ((M - 1, M), (M, M - 1), (M, M))[kind]
            best[wanted[int(live[t])]] = (M, lo, hi)
        live = live[~got]
        start[live] += window
        lost.append(live[start[live] >= m_top])
        live = live[start[live] < m_top]
    live = np.concatenate(lost)

    if live.size:
        missing = sorted(wanted[int(i)] for i in live)
        raise AssertionError(
            f"{len(missing)} demanded stripe pair(s) never touch, e.g. "
            f"{missing[:3]} -- the guiding pattern's completeness "
            f"guarantee does not hold at this size."
        )
    return best


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The routing structure, unbuilt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _plaza_fast(lanes, cut_edges, band_bot, row_width):
    """construction._plaza with the chains left unbuilt."""
    ordered = sorted(lanes.values(), key=lambda ln: ln.turn_col)
    T = len(ordered)
    a = [ln.turn_col for ln in ordered]
    if len(set(a)) != T:
        raise AssertionError("escape columns are not distinct")

    S = max(T, 3)
    shifts = sorted(a[i] - i for i in range(T))
    X0 = shifts[T // 2]
    X0 = max(0, min(X0, max(row_width - S, 0)))
    b = [X0 + i for i in range(T)]
    F = max(1, max(abs(a[i] - b[i]) for i in range(T)))
    rail = band_bot - 1 + F

    for i, ln in enumerate(ordered):
        ln.stripe = i

    demand = [(lanes[u].stripe, lanes[v].stripe) for u, v in cut_edges]
    best = contacts_fast(S, demand)

    reach = [0] * S
    contacts: Dict[Edge, Tuple[int, int]] = {}
    for (u, v) in cut_edges:
        ju, jv = lanes[u].stripe, lanes[v].stripe
        _, lo, hi = best[(min(ju, jv), max(ju, jv))]
        iu, iv = (lo, hi) if ju < jv else (hi, lo)
        contacts[(u, v)] = (iu, iv)
        reach[ju] = max(reach[ju], iu)
        reach[jv] = max(reach[jv], iv)

    plaza_depth = max(reach) + 1
    depth = rail + 1 + plaza_depth
    width = max(row_width, X0 + S)
    stats = {
        "T": T, "S": S, "funnel": F, "X0": X0, "plaza_depth": plaza_depth,
        "width": width, "height": depth,
        "max_lane": max(reach[ln.stripe] for ln in ordered) + 1,
    }
    return contacts, max(width, depth), stats, reach, rail, X0, S, F


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One lane, on demand
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def funnel_cells(turn_col: int, stripe: int, plan: FastPlan) -> np.ndarray:
    """The funnel part of a lane, one column of drift per row."""
    F, band_bot = plan.F, plan.band_bot
    b = plan.X0 + stripe
    d = turn_col - b
    sgn = (d > 0) - (d < 0)
    t = np.arange(1, F + 1, dtype=np.int64)
    rows = band_bot - 1 + t
    cols = b + sgn * np.maximum(abs(d) - t, 0)
    return np.stack([rows, cols], axis=1)


def trunk_ids(ln: _Lane, plan: FastPlan, L: int) -> np.ndarray:
    """The hardware vertices of a lane's chain, cut at reach."""
    j = ln.stripe
    upto = plan.reach[j]
    idx = np.arange(upto + 1, dtype=np.int64)
    rows = plan.rail + idx + 1
    cols = plan.X0 + stripe_cols(j, idx + 1, plan.S)
    return rows * L + cols + 1


def lane_ids(ln: _Lane, plan: FastPlan, L: int) -> Tuple[np.ndarray, int]:
    """Every hardware vertex a lane adds to its border node's super vertex: (ids, n_before_trunk)."""
    cells = ln.cells or _lane_cells(ln, plan.band_bot)     # built lazily by the plan
    if cells:
        esc = np.asarray(cells, dtype=np.int64)
        esc_ids = esc[:, 0] * L + esc[:, 1] + 1
    else:
        esc_ids = np.empty(0, dtype=np.int64)
    fun = funnel_cells(ln.turn_col, ln.stripe, plan)
    fun_ids = fun[:, 0] * L + fun[:, 1] + 1
    tr_ids = trunk_ids(ln, plan, L)
    return (np.concatenate([esc_ids, fun_ids, tr_ids]),
            esc_ids.size + fun_ids.size)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Entry points, as in construction.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def guaranteed_plan_fast(embeddings, partition) -> FastPlan:
    """The whole construction, without building the chains, which already settles L."""
    cids = sorted(embeddings)
    if not cids:
        return FastPlan([], {}, 1, {}, {}, {})

    docks = _pick_docks(embeddings, partition)
    placements, orient, grid = _place(embeddings, docks)
    lanes = _assign_lanes(cids, docks, placements, grid, with_cells=False)

    cut_edges = [(u, v) for u, v, _, _ in partition.cross_edges]
    contacts, L, stats, reach, rail, X0, S, F = _plaza_fast(
        lanes, cut_edges, grid.band_bot, grid.row_width)
    stats.update({
        "Gr": grid.Gr, "Gc": grid.Gc, "n_top": grid.n_top,
        "row_width": grid.row_width, "band_bot": grid.band_bot,
        "patch_cols": grid.patch_cols, "corridor_cols": grid.row_width - grid.patch_cols,
        "patch_rows": grid.patch_rows, "band_rows": sum(grid.tb) + sum(grid.bb),
    })
    return FastPlan(placements, orient, L, lanes, contacts, stats,
                    rail=rail, X0=X0, S=S, F=F, band_bot=grid.band_bot,
                    reach=reach)


def guaranteed_floorplan_fast(embeddings, partition,
                              plan: Optional[FastPlan] = None):
    """The placement of the patches: (placements, orient, L)."""
    plan = plan or guaranteed_plan_fast(embeddings, partition)
    return plan.placements, plan.orient, plan.L


def guaranteed_route_fast(H: KingsLattice, phi, cut_edges, plan: FastPlan):
    """construction.guaranteed_route as array operations: (routes, unrouted, phi_out), with unrouted always empty."""
    L = plan.L
    work: Dict[Node, np.ndarray] = {
        k: np.asarray(v, dtype=np.int64).ravel() for k, v in phi.items()
    }

    # lanes are appended to their border node's super vertex and read back from it
    trunk_of: Dict[Node, np.ndarray] = {}
    lane_start: Dict[Node, int] = {}
    for u, ln in plan.lanes.items():
        ids, split = lane_ids(ln, plan, L)
        base = work[u].size
        lane_start[u] = base
        work[u] = np.concatenate([work[u], ids])
        trunk_of[u] = work[u][base + split:]
        del ids

    # M2 for the whole lattice at once
    if Occupancy(L, list(work.values())).duplicates:
        _report_collision(work, lane_start, plan)      # raises with detail

    for u in plan.lanes:
        if not _lane_keeps_chain_connected(work[u][:lane_start[u]],
                                           work[u][lane_start[u]:], L):
            raise AssertionError(f"lane for {u!r} left its chain disconnected")

    routes: Dict[Edge, List[NodeH]] = {}
    for (u, v) in cut_edges:
        if _chains_touch(phi[u], phi[v], L):
            continue
        iu, iv = plan.contacts[(u, v)]
        tu, tv = trunk_of[u], trunk_of[v]
        if not H.has_edge(int(tu[iu]), int(tv[iv])):
            raise AssertionError(f"lanes for cut edge {(u, v)!r} do not touch")
        routes[(u, v)] = np.concatenate([tu[:iu + 1], tv[:iv + 1][::-1]])

    return routes, [], work


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _chains_touch(ca, cb, L: int) -> bool:
    """Whether two super vertices are adjacent on the hardware."""
    a = np.asarray(ca, dtype=np.int64).ravel()
    b = np.asarray(cb, dtype=np.int64).ravel()
    ra, cca = rc_of(a, L)
    rb, ccb = rc_of(b, L)
    return bool((
        (np.abs(ra[:, None] - rb[None, :]) <= 1)
        & (np.abs(cca[:, None] - ccb[None, :]) <= 1)
    ).any())


def _lane_keeps_chain_connected(patch, lane: np.ndarray, L: int) -> bool:
    """Whether a lane keeps its border node's super vertex connected."""
    if lane.size == 0:
        return True
    rows, cols = rc_of(lane, L)
    if lane.size > 1:
        if not ((np.abs(np.diff(rows)) <= 1) & (np.abs(np.diff(cols)) <= 1)).all():
            return False
    p = np.asarray(patch, dtype=np.int64).ravel()
    if p.size == 0:
        return True
    pr, pc = rc_of(p, L)
    return bool(((np.abs(pr - rows[0]) <= 1) & (np.abs(pc - cols[0]) <= 1)).any())


def _report_collision(work, lane_start, plan) -> None:
    """Name the lane that hit another super vertex."""
    claimed: Dict[int, Node] = {}
    for v, chain in work.items():
        stop = lane_start.get(v, chain.size)          # patch cells only
        for w in chain[:stop].tolist():
            claimed[w] = v
    for u in plan.lanes:
        for w in work[u][lane_start[u]:].tolist():
            prev = claimed.get(w)
            if prev is not None:
                raise AssertionError(
                    f"lane for {u!r} collides at hw {w} (held by {prev!r}) -- "
                    f"the construction's disjointness invariant is broken"
                )
            claimed[w] = u
    raise AssertionError(
        "disjointness check failed in aggregate but no single colliding cell "
        "was found -- this should be impossible; suspect duplicate ids inside "
        "one chain"
    )
