"""
The King's graph KG(L) as arithmetic, never built, for the large embeddings of the
fallback method, where u and v are adjacent if their (row, col) differ by at most 1,
and the check that an embedding is valid, (M1)-(M3), in one pass over the occupied cells.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

Node = Any
NodeH = int

# The eight directions of the King's graph.
KINGS_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# The four that list every edge once, in build_hardware_graph's order.
FORWARD_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, -1), (1, 0), (1, 1),
)

# Above this many cells the owners are looked up by binary search, not a dense array.
_DENSE_CELL_BUDGET = 512_000_000


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The lattice
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class KingsLattice:
    """KG(L) that is never built, with the parts of nx.Graph the construction uses."""

    __slots__ = ("L", "graph")

    def __init__(self, L: int):
        if L < 1:
            raise ValueError(f"L must be >= 1, got {L}")
        self.L = int(L)
        self.graph = {"family": "kings_graph", "L": self.L}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # basic graph protocol
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __len__(self) -> int:
        return self.L * self.L

    def number_of_nodes(self) -> int:
        return self.L * self.L

    def number_of_edges(self) -> int:
        """2L(L-1) orthogonal + 2(L-1)^2 diagonal edges."""
        L = self.L
        return 2 * L * (L - 1) + 2 * (L - 1) * (L - 1)

    def nodes(self):
        """A range, never a list."""
        return range(1, self.L * self.L + 1)

    def __iter__(self) -> Iterator[int]:
        return iter(self.nodes())

    def __contains__(self, u: object) -> bool:
        return isinstance(u, (int, np.integer)) and 1 <= int(u) <= self.L * self.L

    def has_node(self, u: object) -> bool:
        return u in self

    def edges(self):
        raise NotImplementedError(
            f"KingsLattice({self.L}) refuses to enumerate its "
            f"{self.number_of_edges()} edges -- that is never the cheap way to "
            f"do anything here. Use has_edge/neighbors, or the vectorised "
            f"helpers in kings_lattice (adjacent_owner_pairs, validate_embedding)."
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # adjacency
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def has_edge(self, u: int, v: int) -> bool:
        L = self.L
        if u == v:
            return False
        n = L * L
        if not (1 <= u <= n and 1 <= v <= n):
            return False
        r1, c1 = divmod(u - 1, L)
        r2, c2 = divmod(v - 1, L)
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1

    def neighbors(self, u: int) -> Iterator[int]:
        L = self.L
        r, c = divmod(u - 1, L)
        for dr, dc in KINGS_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < L and 0 <= cc < L:
                yield rr * L + cc + 1

    # networkx spells it both ways depending on the call site
    def __getitem__(self, u: int) -> Dict[int, dict]:
        return {v: {} for v in self.neighbors(u)}

    def degree(self, u: int) -> int:
        L = self.L
        r, c = divmod(u - 1, L)
        rows = 3 - (r == 0) - (r == L - 1)
        cols = 3 - (c == 0) - (c == L - 1)
        return rows * cols - 1

    def subgraph(self, nodes: Sequence[int]) -> "nx.Graph":
        """A real nx.Graph on the given nodes only."""
        L = self.L
        ids = np.asarray(list(nodes), dtype=np.int64)
        S = nx.Graph()
        S.add_nodes_from(int(x) for x in ids)
        if ids.size == 0:
            return S
        present = set(int(x) for x in ids)
        rows, cols = rc_of(ids, L)
        for dr, dc in FORWARD_OFFSETS:
            rr, cc = rows + dr, cols + dc
            ok = (rr >= 0) & (rr < L) & (cc >= 0) & (cc < L)
            if not ok.any():
                continue
            nbr = rr[ok] * L + cc[ok] + 1
            src = ids[ok]
            for a, b in zip(src.tolist(), nbr.tolist()):
                if b in present:
                    S.add_edge(a, b)
        return S

    def __repr__(self) -> str:
        return f"KingsLattice(L={self.L}, {self.number_of_nodes()} cells, implicit)"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Vectorised coordinate helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rc_of(ids: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hardware vertex ids -> (rows, cols)."""
    ids = np.asarray(ids, dtype=np.int64)
    z = ids - 1
    return z // L, z % L


def id_of(rows: np.ndarray, cols: np.ndarray, L: int) -> np.ndarray:
    """(rows, cols) -> hardware vertex ids."""
    return np.asarray(rows, dtype=np.int64) * L + np.asarray(cols, dtype=np.int64) + 1


def chain_arrays(phi: Dict[Node, Any]) -> Tuple[List[Node], List[np.ndarray]]:
    """phi as a node order and its id arrays."""
    keys = list(phi.keys())
    arrs = [np.asarray(phi[k], dtype=np.int64).ravel() for k in keys]
    return keys, arrs


def to_lists(phi: Dict[Node, Any]) -> Dict[Node, List[int]]:
    """phi with plain lists."""
    return {k: [int(x) for x in np.asarray(v).ravel()] for k, v in phi.items()}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Occupancy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Occupancy:
    """Which super vertex owns which hardware vertex, as an array lookup."""

    __slots__ = ("L", "n_chains", "dense", "duplicates", "_img", "_sorted",
                 "_owner_sorted")

    def __init__(self, L: int, arrs: Sequence[np.ndarray], dense: Optional[bool] = None):
        self.L = int(L)
        self.n_chains = len(arrs)
        cells = self.L * self.L
        self.dense = (cells <= _DENSE_CELL_BUDGET) if dense is None else bool(dense)
        total = sum(int(a.size) for a in arrs)

        if self.dense:
            self._img = np.full(cells, -1, dtype=np.int32)
            self._sorted = None
            self._owner_sorted = None
            for i, ids in enumerate(arrs):
                if ids.size:
                    self._img[ids - 1] = i
            # M2 from the scatter: a cell claimed twice leaves fewer cells set than were written
            self.duplicates = total - int(np.count_nonzero(self._img >= 0))
        else:
            self._img = None
            flat = np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)
            owner = np.concatenate([
                np.full(a.size, i, dtype=np.int32) for i, a in enumerate(arrs)
            ]) if arrs else np.empty(0, dtype=np.int32)
            order = np.argsort(flat, kind="stable")
            self._sorted = flat[order]
            self._owner_sorted = owner[order]
            self.duplicates = int(flat.size) - int(
                np.count_nonzero(np.diff(self._sorted)) + (1 if flat.size else 0))

    @staticmethod
    def collisions(arrs: Sequence[np.ndarray]) -> int:
        """How many hardware vertices more than one super vertex claims (M2)."""
        if not arrs:
            return 0
        flat = np.concatenate(arrs)
        return int(flat.size - np.unique(flat).size)

    def owner_of(self, ids: np.ndarray) -> np.ndarray:
        """The owning super vertex per id, -1 where free."""
        ids = np.asarray(ids, dtype=np.int64)
        if self.dense:
            return self._img[ids - 1]
        pos = np.searchsorted(self._sorted, ids)
        pos = np.clip(pos, 0, self._sorted.size - 1)
        hit = self._sorted[pos] == ids
        out = np.where(hit, self._owner_sorted[pos], np.int32(-1))
        return out.astype(np.int32, copy=False)


def covered_edges(occ: Occupancy, arrs: Sequence[np.ndarray],
                  partners: Sequence[np.ndarray],
                  edge_ids: Sequence[np.ndarray],
                  n_edges: int) -> np.ndarray:
    """Which edges of I are realized by a hardware adjacency, as a mask."""
    found = np.zeros(n_edges, dtype=bool)
    if n_edges == 0:
        return found
    L = occ.L
    for i, ids in enumerate(arrs):
        want = partners[i]
        if ids.size == 0 or want.size == 0:
            continue
        eids = edge_ids[i]
        todo = ~found[eids]
        if not todo.any():
            continue                      # every edge at this chain already seen
        rows, cols = rc_of(ids, L)
        for dr, dc in FORWARD_OFFSETS:
            rr = rows + dr
            cc = cols + dc
            ok = (rr >= 0) & (rr < L) & (cc >= 0) & (cc < L)
            if not ok.any():
                continue
            theirs = occ.owner_of(rr[ok] * L + cc[ok] + 1)
            for t in range(want.size):
                if todo[t] and (theirs == want[t]).any():
                    found[eids[t]] = True
                    todo[t] = False
            if not todo.any():
                break
    return found


def graph_adjacency(I: "nx.Graph", index_of: Dict[Node, int], n_chains: int):
    """I as per-super-vertex neighbour and edge-id arrays."""
    nbr: List[List[int]] = [[] for _ in range(n_chains)]
    eid: List[List[int]] = [[] for _ in range(n_chains)]
    n_edges = 0
    for u, v in I.edges():
        a, b = index_of.get(u), index_of.get(v)
        if a is None or b is None or a == b:
            continue
        nbr[a].append(b); eid[a].append(n_edges)
        nbr[b].append(a); eid[b].append(n_edges)
        n_edges += 1
    partners = [np.asarray(x, dtype=np.int32) for x in nbr]
    edge_ids = [np.asarray(x, dtype=np.int64) for x in eid]
    return partners, edge_ids, n_edges


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Validity, (M1)-(M3) in one pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _chain_is_connected(ids: np.ndarray, L: int) -> bool:
    """(M1) for one super vertex, with its trailing path contracted."""
    n = int(ids.size)
    if n <= 1:
        return n == 1

    rows, cols = rc_of(ids, L)
    step_ok = (np.abs(np.diff(rows)) <= 1) & (np.abs(np.diff(cols)) <= 1)
    if step_ok.all():
        return True                     # a path in H, start to finish

    t = int(np.nonzero(~step_ok)[0][-1]) + 1     # ids[t:] is that trailing path
    head = ids[:t]
    tail = np.sort(ids[t:])                      # non-empty: t <= n-1

    order = np.argsort(head, kind="stable")
    head_sorted = head[order]

    # Node t stands for the whole contracted tail.
    pairs: List[np.ndarray] = []
    for dr, dc in KINGS_OFFSETS:
        r2, c2 = rows[:t] + dr, cols[:t] + dc
        ok = (r2 >= 0) & (r2 < L) & (c2 >= 0) & (c2 < L)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        nbr = r2[idx] * L + c2[idx] + 1

        pos = np.clip(np.searchsorted(head_sorted, nbr), 0, head_sorted.size - 1)
        hit = head_sorted[pos] == nbr
        if hit.any():
            pairs.append(np.stack([idx[hit], order[pos[hit]]], axis=1))

        pos = np.clip(np.searchsorted(tail, nbr), 0, tail.size - 1)
        hit = tail[pos] == nbr
        if hit.any():
            pairs.append(np.stack([idx[hit], np.full(int(hit.sum()), t)], axis=1))

    if not pairs:
        return False                    # head cells exist but touch nothing
    E = np.concatenate(pairs)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(E.shape[0], dtype=np.int8), (E[:, 0], E[:, 1])),
                   shape=(t + 1, t + 1))
    ncomp, _ = connected_components(A, directed=False)
    return ncomp == 1


def validate_embedding(phi: Dict[Node, Any], I: "nx.Graph",
                       lattice: KingsLattice,
                       occ: Optional[Occupancy] = None) -> Tuple[bool, int]:
    """(valid, eemb), from one pass over the occupied cells."""
    L = lattice.L
    keys, arrs = chain_arrays(phi)
    index_of = {k: i for i, k in enumerate(keys)}

    # M1a: no empty chain
    if any(a.size == 0 for a in arrs):
        return False, 0

    if occ is None:
        occ = Occupancy(L, arrs)

    # M2: disjointness -- a by-product of building the index
    disjoint = occ.duplicates == 0

    # M3 + eemb: which edges of I are realized by a hardware adjacency
    partners, edge_ids, n_edges = graph_adjacency(I, index_of, len(arrs))
    if n_edges:
        e_total = int(covered_edges(occ, arrs, partners, edge_ids, n_edges).sum())
        covered = e_total == n_edges
    else:
        e_total, covered = 0, True

    if not (disjoint and covered):
        return False, e_total

    # M1b: connectivity, chain by chain (cheap: mostly two numpy diffs each)
    for ids in arrs:
        if not _chain_is_connected(ids, L):
            return False, e_total

    return True, e_total
