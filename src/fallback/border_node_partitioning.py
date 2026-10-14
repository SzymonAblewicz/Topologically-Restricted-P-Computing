"""
Graph partitioning to minimize border node count (Section 4.3 of the report), the
partitioning stage of the fallback method, where the objective is the number of
border nodes T = #{v : out(v) > 0} and not the cut. Two partitioners, Mt-KaHyPar
on a hypergraph whose cut-net objective equals T exactly (mtkahypar_partitioner.py),
and GGP + T-FM, whose refinement minimizes T directly (ggp_tfm.py).
"""

import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

import ggp_tfm  # GGP + T-FM on CSR arrays

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _HAVE_MPL = True
except ImportError:  # pragma: no cover - plotting is optional
    _HAVE_MPL = False

try:
    # the same colours as the embedding drawings
    from draw_embedding import get_color as _pssa_get_color
    _HAVE_PSSA_VIZ = True
except ImportError:  # pragma: no cover - viz module not on path
    _HAVE_PSSA_VIZ = False


Node = Any
Labels = Dict[Node, int]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Result container
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class PartitionResult:
    k: int
    labels: Labels
    communities: Dict[int, Set[Node]]
    subgraphs: Dict[int, "nx.Graph"]
    border_nodes: Dict[int, Set[Node]]
    cross_edges: List[Tuple[Node, Node, int, int]]
    method: str

    @property
    def sizes(self) -> Dict[int, int]:
        return {c: len(nodes) for c, nodes in self.communities.items()}

    @property
    def cut_size(self) -> int:
        """The number of cut edges."""
        return len(self.cross_edges)

    @property
    def border_count(self) -> int:
        """T, the number of border nodes, ie. vertices with a neighbour outside their own cluster."""
        return sum(len(b) for b in self.border_nodes.values())

    @property
    def plaza_area(self) -> int:
        """The hardware vertices of the routing structure, T chains of T - 1 vertices each."""
        T = self.border_count
        return T * (T - 1) if T > 1 else 0

    def cut_fraction(self, G: "nx.Graph") -> float:
        m = G.number_of_edges()
        return self.cut_size / m if m else 0.0

    def border_fraction_total(self) -> float:
        """T/n."""
        n = len(self.labels)
        return self.border_count / n if n else 0.0

    def border_fraction(self, cid: int) -> float:
        n = len(self.communities.get(cid, ()))
        return len(self.border_nodes.get(cid, ())) / n if n else 0.0

    def avg_cut_degree(self) -> float:
        """Cut edges per border node."""
        T = self.border_count
        return (2 * self.cut_size / T) if T else 0.0

    def modularity(self, G: "nx.Graph") -> float:
        parts = [self.communities[c] for c in sorted(self.communities)]
        return nx.algorithms.community.modularity(G, parts)

    def neighbor_clusters(self, cid: int) -> Set[int]:
        """The clusters that cid shares a cut edge with."""
        out: Set[int] = set()
        for _, _, cu, cv in self.cross_edges:
            if cu == cid:
                out.add(cv)
            elif cv == cid:
                out.add(cu)
        return out

    def stitch_pairs(self, cid_a: int, cid_b: int) -> List[Tuple[Node, Node]]:
        """The pairs of border nodes of clusters a and b joined by a cut edge."""
        out: List[Tuple[Node, Node]] = []
        for u, v, cu, cv in self.cross_edges:
            if cu == cid_a and cv == cid_b:
                out.append((u, v))
            elif cu == cid_b and cv == cid_a:
                out.append((v, u))
        return out

    def summary(self) -> str:
        lines = [
            f"PartitionResult(method={self.method!r}): {self.k} clusters, "
            f"{len(self.labels)} nodes",
            f"  OBJECTIVE  T = {self.border_count} border vertices "
            f"(T/n = {self.border_fraction_total():.3f}), "
            f"plaza {self.border_count}x{self.border_count - 1} "
            f"= {self.plaza_area} qubits",
            f"  diagnostic cut = {self.cut_size} edges "
            f"({self.avg_cut_degree():.2f} cut edges per border vertex)",
        ]
        for cid in sorted(self.communities):
            nodes = self.communities[cid]
            sub = self.subgraphs[cid]
            bnodes = self.border_nodes.get(cid, set())
            lines.append(
                f"  cluster {cid}: {len(nodes):4d} nodes, "
                f"{sub.number_of_edges():4d} internal edges, "
                f"{len(bnodes):4d} border nodes ({self.border_fraction(cid):.0%}), "
                f"-> neighbors {sorted(self.neighbor_clusters(cid))}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()
def fm_refine(
    G: "nx.Graph", labels: Labels, k: int,
    balance_tol: float = 0.15, max_passes: int = 50,
    seed: Optional[int] = None,
    max_size: Optional[int] = None, min_size: Optional[int] = None,
) -> Labels:
    """
    T-FM, the Fiduccia-Mattheyses refinement with its objective rewritten to be T. A vertex
    moves to a neighbouring cluster whenever the move lowers T, read off the vertex and its
    neighbours alone, and between moves that change T equally the one that lowers the cut
    more is chosen.
    """
    labels = dict(labels)
    n = G.number_of_nodes()
    if k <= 1 or n == 0:
        return labels

    target = n / k
    lo = max(1, int(math.floor(target * (1 - balance_tol))))
    hi = max(lo, int(math.ceil(target * (1 + balance_tol))))
    if max_size is not None:
        hi = min(hi, max_size)
    if min_size is not None:
        lo = max(lo, min_size)
    lo = min(lo, hi)  # keep the window non-empty if bounds get tight

    # keep at least one free slot above the mean cluster size, or no move is legal
    if hi <= target:
        hi = int(math.ceil(target)) + 1
        if max_size is not None:
            hi = min(hi, max_size)
        lo = min(lo, hi)
    sizes = Counter(labels.values())
    rng = random.Random(seed)
    nodes = list(G.nodes())

    # out[v] = neighbours of v outside its cluster; v is a border node iff out[v] > 0
    out = {v: sum(1 for w in G.neighbors(v) if labels[w] != labels[v])
           for v in nodes}

    for _ in range(max_passes):
        rng.shuffle(nodes)
        moved = 0
        for u in nodes:
            cu = labels[u]
            # a self-loop never leaves its cluster
            nbrs = [(w, labels[w]) for w in G.neighbors(u) if w != u]
            if not nbrs:
                continue                      # isolated: interior wherever it sits
            nbr_counts = Counter(cw for _, cw in nbrs)
            internal = nbr_counts.get(cu, 0)
            deg_u = len(nbrs)

            # (T gain, cut gain): a move must lower T, or keep T and lower the cut
            best_c, best_gain = cu, (0, 0)
            for c, cnt in nbr_counts.items():
                if c == cu or sizes[c] + 1 > hi or sizes[cu] - 1 < lo:
                    continue
                # u keeps cnt neighbours inside its new cluster
                dT = ((deg_u - cnt) > 0) - (out[u] > 0)
                # only u's neighbours in cu or in c can change border status
                for w, cw in nbrs:
                    if cw == cu:
                        dT += (out[w] + 1 > 0) - (out[w] > 0)
                    elif cw == c:
                        dT += (out[w] - 1 > 0) - (out[w] > 0)
                gain = (-dT, cnt - internal)
                if gain > best_gain:
                    best_gain, best_c = gain, c

            if best_c != cu:
                for w, cw in nbrs:
                    if cw == cu:
                        out[w] += 1
                    elif cw == best_c:
                        out[w] -= 1
                out[u] = deg_u - nbr_counts[best_c]
                labels[u] = best_c
                sizes[cu] -= 1
                sizes[best_c] += 1
                moved += 1
        if moved == 0:
            break
    return labels


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The maximum cluster size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def enforce_size_cap(
    G: "nx.Graph", labels: Labels, max_size: int, seed: Optional[int] = None,
) -> Labels:
    """Bisect every cluster larger than max_size until all fit (fm_refine then reoptimizes T)."""
    labels = dict(labels)
    rng = random.Random(seed)
    next_id = (max(labels.values()) + 1) if labels else 0

    communities: Dict[int, Set[Node]] = defaultdict(set)
    for n, c in labels.items():
        communities[c].add(n)

    stack: List[Tuple[int, Set[Node]]] = list(communities.items())
    final: Dict[int, Set[Node]] = {}

    def _fallback_split(nodes: Set[Node]) -> Tuple[Set[Node], Set[Node]]:
        ordered = sorted(nodes, key=str)
        mid = len(ordered) // 2
        return set(ordered[:mid]), set(ordered[mid:])

    while stack:
        cid, nodes = stack.pop()
        if len(nodes) <= max_size:
            final[cid] = nodes
            continue

        sub = G.subgraph(nodes)
        if sub.number_of_edges() == 0:
            a, b = _fallback_split(nodes)
        else:
            try:
                a, b = nx.algorithms.community.kernighan_lin_bisection(
                    sub, seed=rng.randint(0, 2 ** 31 - 1),
                )
            except Exception:
                a, b = _fallback_split(nodes)
            if not a or not b:
                a, b = _fallback_split(nodes)

        stack.append((cid, set(a)))
        stack.append((next_id, set(b)))
        next_id += 1

    out: Labels = {}
    for cid, nodes in final.items():
        for n in nodes:
            out[n] = cid
    return out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The minimum cluster size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def enforce_min_size(
    G: "nx.Graph", labels: Labels, min_size: int,
    max_size: Optional[int] = None, seed: Optional[int] = None,
) -> Tuple[Labels, bool]:
    """Merge every cluster smaller than min_size into the neighbour that lowers T the most: (labels, ok)."""
    labels = dict(labels)
    while True:
        communities: Dict[int, Set[Node]] = defaultdict(set)
        for n, c in labels.items():
            communities[c].add(n)
        if len(communities) <= 1:
            return labels, True

        sizes = {c: len(ns) for c, ns in communities.items()}
        under = sorted((c for c, s in sizes.items() if s < min_size),
                        key=lambda c: sizes[c])
        if not under:
            return labels, True

        cid = under[0]
        nodes = communities[cid]

        # merging cid into c makes v interior when every cluster v touches is cid or c
        nbr_edge_count: Counter = Counter()
        t_gain: Counter = Counter()
        region = set(nodes)
        for u in nodes:
            for v in G.neighbors(u):
                cv = labels[v]
                if cv != cid:
                    nbr_edge_count[cv] += 1
                    region.add(v)
        for v in region:
            touched = {labels[v]} | {labels[w] for w in G.neighbors(v)}
            if len(touched) == 2 and cid in touched:
                t_gain[(touched - {cid}).pop()] += 1

        candidates = sorted(nbr_edge_count.keys(),
                            key=lambda c: (-t_gain[c], -nbr_edge_count[c]))
        if not candidates:
            candidates = sorted((c for c in sizes if c != cid), key=lambda c: sizes[c])

        merged = False
        for target in candidates:
            if max_size is not None and sizes[cid] + sizes[target] > max_size:
                continue
            for u in nodes:
                labels[u] = target
            merged = True
            break

        if not merged:
            return labels, False


def enforce_bounds(
    G: "nx.Graph", labels: Labels,
    min_size: Optional[int], max_size: Optional[int],
    seed: Optional[int] = None, max_rounds: int = 6,
) -> Tuple[Labels, bool]:
    """Alternate enforce_size_cap and enforce_min_size until both bounds hold: (labels, ok)."""
    labels = dict(labels)
    for _ in range(max_rounds):
        sizes = Counter(labels.values())
        if max_size is not None and any(s > max_size for s in sizes.values()):
            labels = enforce_size_cap(G, labels, max_size, seed=seed)

        sizes = Counter(labels.values())
        if min_size is not None and any(s < min_size for s in sizes.values()):
            labels, ok = enforce_min_size(G, labels, min_size, max_size=max_size, seed=seed)
            if not ok:
                return labels, False

        sizes = Counter(labels.values())
        max_ok = max_size is None or all(s <= max_size for s in sizes.values())
        min_ok = min_size is None or all(s >= min_size for s in sizes.values())
        if max_ok and min_ok:
            return labels, True

    sizes = Counter(labels.values())
    max_ok = max_size is None or all(s <= max_size for s in sizes.values())
    min_ok = min_size is None or all(s >= min_size for s in sizes.values())
    return labels, (max_ok and min_ok)


def feasible_k_range(
    n_total: int, min_cluster_size: Optional[int] = None,
    max_cluster_size: Optional[int] = None,
) -> Tuple[int, int]:
    """The numbers of clusters k that the two size bounds allow."""
    k_min, k_max = 1, n_total
    if max_cluster_size is not None:
        k_min = max(k_min, math.ceil(n_total / max_cluster_size))
    if min_cluster_size is not None:
        k_max = min(k_max, max(1, n_total // min_cluster_size))
    if k_min > k_max:
        raise ValueError(
            f"No k satisfies both bounds on n_total={n_total} nodes: "
            f"max_cluster_size={max_cluster_size} requires k >= {k_min}, "
            f"min_cluster_size={min_cluster_size} requires k <= {k_max}. "
            f"Loosen one or both bounds."
        )
    return k_min, k_max
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Result assembly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _build_result(G: "nx.Graph", labels: Labels, method: str) -> PartitionResult:
    uniq = sorted(set(labels.values()), key=str)
    remap = {old: i for i, old in enumerate(uniq)}
    labels = {n: remap[c] for n, c in labels.items()}
    k = len(uniq)

    communities: Dict[int, Set[Node]] = defaultdict(set)
    for n, c in labels.items():
        communities[c].add(n)
    communities = dict(communities)

    subgraphs = {c: G.subgraph(nodes).copy() for c, nodes in communities.items()}

    border_nodes: Dict[int, Set[Node]] = defaultdict(set)
    cross_edges: List[Tuple[Node, Node, int, int]] = []
    for u, v in G.edges():
        cu, cv = labels[u], labels[v]
        if cu != cv:
            border_nodes[cu].add(u)
            border_nodes[cv].add(v)
            cross_edges.append((u, v, cu, cv))
    border_nodes = {c: border_nodes.get(c, set()) for c in communities}

    return PartitionResult(
        k=k, labels=labels, communities=communities, subgraphs=subgraphs,
        border_nodes=border_nodes, cross_edges=cross_edges, method=method,
    )


def partition_from_labels(
    G: "nx.Graph", labels: Labels, method: str = "ground_truth",
    max_cluster_size: Optional[int] = None, seed: Optional[int] = None,
) -> PartitionResult:
    """A PartitionResult from known labels, with clusters over max_cluster_size bisected."""
    if set(labels) != set(G.nodes()):
        raise ValueError("labels must cover exactly G's nodes.")
    if max_cluster_size is not None:
        sizes = Counter(labels.values())
        if any(s > max_cluster_size for s in sizes.values()):
            labels = enforce_size_cap(G, labels, max_cluster_size, seed=seed)
    return _build_result(G, labels, method)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Large n: the CSR arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _LazySubgraphs(Mapping):
    """{cid: G.subgraph(nodes)}, each view built only when asked, as there are tens of thousands of clusters at a million vertices."""

    __slots__ = ("_G", "_communities", "_cache")

    def __init__(self, G: "nx.Graph", communities: Dict[int, Set[Node]]):
        self._G = G
        self._communities = communities
        self._cache: Dict[int, "nx.Graph"] = {}

    def __getitem__(self, cid: int) -> "nx.Graph":
        sub = self._cache.get(cid)
        if sub is None:
            sub = self._G.subgraph(self._communities[cid])
            self._cache[cid] = sub
        return sub

    def __iter__(self):
        return iter(self._communities)

    def __len__(self) -> int:
        return len(self._communities)


def _csr_from_graph(G: "nx.Graph") -> Tuple[List[Node], np.ndarray, np.ndarray]:
    """G in compressed sparse row form, (nodes, indptr, indices)."""
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.int8)
    A.setdiag(0)
    A.eliminate_zeros()
    n = len(nodes)
    idx_dtype = np.int32 if n < 2 ** 31 else np.int64
    return nodes, A.indptr.astype(np.int64), A.indices.astype(idx_dtype)


def _build_result_csr(
    G: "nx.Graph", nodes: List[Node], indptr: np.ndarray, dst: np.ndarray,
    lab: np.ndarray, method: str,
) -> PartitionResult:
    """_build_result from the CSR arrays, for large n."""
    n = len(nodes)
    _, lab = np.unique(lab, return_inverse=True)   # compact to 0..k-1
    lab = lab.astype(np.int32, copy=False)
    k = int(lab.max()) + 1 if n else 0

    labels: Labels = {nodes[i]: int(lab[i]) for i in range(n)}

    # group node indices by label in one sort instead of k passes
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(k + 1))
    communities = {
        c: {nodes[i] for i in order[bounds[c]:bounds[c + 1]]}
        for c in range(k)
    }

    _, borders = ggp_tfm.T_exact(indptr, dst, lab, n)
    border_nodes: Dict[int, Set[Node]] = {c: set() for c in range(k)}
    for i in np.flatnonzero(borders):
        border_nodes[int(lab[i])].add(nodes[i])

    # each undirected edge appears twice in CSR; `src < dst` keeps one copy
    src = np.repeat(np.arange(n, dtype=dst.dtype), np.diff(indptr))
    keep = (lab[src] != lab[dst]) & (src < dst)
    cross_edges = [
        (nodes[int(u)], nodes[int(v)], int(lab[u]), int(lab[v]))
        for u, v in zip(src[keep], dst[keep])
    ]

    return PartitionResult(
        k=k, labels=labels, communities=communities,
        subgraphs=_LazySubgraphs(G, communities),
        border_nodes=border_nodes, cross_edges=cross_edges, method=method,
    )


def partition_graph_csr(
    G: "nx.Graph",
    max_cluster_size: int,
    target_cluster_size: Optional[int] = None,
    headroom: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    refine_passes: int = 40,
    seed: Optional[int] = 0,
    verbose: bool = False,
) -> PartitionResult:
    """GGP + T-FM on the CSR arrays, linear time per pass, with the hard cap max_cluster_size."""
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot partition an empty graph.")
    if max_cluster_size < 1:
        raise ValueError(f"max_cluster_size must be >= 1, got {max_cluster_size}")

    s_target = target_cluster_size
    if s_target is None:
        hr = headroom if headroom is not None else (1 if max_cluster_size <= 20 else 2)
        s_target = max(1, max_cluster_size - hr)
    s_target = max(1, min(s_target, max_cluster_size))

    t0 = time.perf_counter()
    nodes, indptr, dst = _csr_from_graph(G)
    n = len(nodes)
    t_csr = time.perf_counter() - t0

    t0 = time.perf_counter()
    lab, k = ggp_tfm.bfs_ball(indptr, dst, n, s_target)
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    if k > 1:
        lab, passes, moves = ggp_tfm.fm_refine_csr(
            indptr, dst, lab, n, k, max_cluster_size,
            max_passes=refine_passes, seed=(seed if seed is not None else 0),
        )
    else:
        passes = moves = 0
    t_refine = time.perf_counter() - t0

    sizes = np.bincount(lab)
    if sizes.max() > max_cluster_size:
        raise AssertionError(
            f"CSR path produced a cluster of {sizes.max()} > "
            f"max_cluster_size={max_cluster_size}"
        )

    result = _build_result_csr(G, nodes, indptr, dst, lab, "csr_bfs_ball+T-FM")

    if min_cluster_size is not None and min_cluster_size > 1:
        labels, ok = enforce_min_size(G, result.labels, min_cluster_size,
                                      max_size=max_cluster_size, seed=seed)
        if not ok:
            raise RuntimeError(
                f"Could not satisfy min_cluster_size={min_cluster_size} together "
                f"with max_cluster_size={max_cluster_size} on {n} nodes."
            )
        result = _build_result(G, labels, "csr_bfs_ball+T-FM")

    if verbose:
        print(f"  csr {t_csr:.2f}s  init {t_init:.2f}s  refine {t_refine:.2f}s "
              f"({passes} passes, {moves} moves) | k={result.k} "
              f"mean size {n / result.k:.1f} | T={result.border_count} "
              f"(T/n={result.border_fraction_total():.4f})")
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main entry point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def partition_graph(
    G: "nx.Graph",
    max_cluster_size: int,
    target_cluster_size: Optional[int] = None,
    headroom: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    method: str = "auto",
    eps: float = 0.03,
    threads: Optional[int] = None,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> PartitionResult:
    """
    Partition G into clusters of at most max_cluster_size vertices, minimizing T, with
    method "mtkahypar", "csr" (GGP + T-FM) or "auto" (Mt-KaHyPar where installed).
    """
    if method not in ("auto", "mtkahypar", "csr"):
        raise ValueError(
            f"method must be 'auto', 'mtkahypar' or 'csr', got {method!r}")

    use_kahypar = False
    if method in ("auto", "mtkahypar"):
        try:
            import mtkahypar_partitioner
            use_kahypar = mtkahypar_partitioner.backend_status()["usable"]
        except ImportError:
            use_kahypar = False
        if method == "mtkahypar" and not use_kahypar:
            import mtkahypar_partitioner
            raise RuntimeError(
                "method='mtkahypar' was requested but no backend is "
                "available, and falling back silently would cost 5-19% of "
                "T/n. Either install it, or pass method='csr' to accept the "
                "cheaper partitioner deliberately.\n\n" + mtkahypar_partitioner.INSTALL
            )

    if use_kahypar:
        import mtkahypar_partitioner
        result = mtkahypar_partitioner.partition_graph_kahypar(
            G, max_cluster_size=max_cluster_size,
            target_cluster_size=target_cluster_size, headroom=headroom,
            eps=eps, threads=threads,
        )
    else:
        result = partition_graph_csr(
            G, max_cluster_size=max_cluster_size,
            target_cluster_size=target_cluster_size, headroom=headroom,
            min_cluster_size=min_cluster_size, seed=seed,
        )

    if verbose:
        n = G.number_of_nodes()
        print(f"  Stage 1 [{result.method}]: k={result.k}, "
              f"mean size {n / result.k:.1f}, T={result.border_count} "
              f"(T/n={result.border_fraction_total():.4f})")
        # warn only when the weaker partitioner was chosen for the caller
        if not use_kahypar and method == "auto":
            print("           note: Mt-KaHyPar unavailable, so this is the "
                  "fallback partitioner (~5% worse T/n on random-regular "
                  "graphs, ~19% on scale-free). See mtkahypar_partitioner.INSTALL.")
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _cluster_color(cid: int, k: int):
    if _HAVE_PSSA_VIZ:
        return _pssa_get_color(cid + 1, k)
    return cm.tab20(cid / max(k - 1, 1)) if k > 1 else cm.tab20(0.0)


def draw_partition(
    G: "nx.Graph", result: PartitionResult, ax=None, seed: int = 42,
    show: bool = True, highlight_border: bool = True, R: float = 10.0,
    title: Optional[str] = None,
):
    """Every cluster in its own spring layout around a ring, with the border nodes ringed."""
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib is required for draw_partition().")

    should_show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        should_show = True

    k = max(result.k, 1)
    pos: Dict[Node, Tuple[float, float]] = {}
    for cid, nodes in result.communities.items():
        sub = G.subgraph(nodes)
        if sub.number_of_edges() > 0:
            local_pos = nx.spring_layout(sub, seed=seed, k=0.4, scale=1.6)
        else:
            local_pos = {n: (0.0, 0.0) for n in nodes}
        angle = 2 * math.pi * cid / k
        cx, cy = R * math.cos(angle), R * math.sin(angle)
        for n, (x, y) in local_pos.items():
            pos[n] = (x + cx, y + cy)

    cross = [(u, v) for u, v, _, _ in result.cross_edges]
    intra = [(u, v) for u, v in G.edges() if result.labels[u] == result.labels[v]]

    nx.draw_networkx_edges(G, pos, edgelist=cross, alpha=0.35, width=0.8,
                            edge_color="#999999", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=intra, alpha=0.6, width=0.6,
                            edge_color="#666666", ax=ax)

    node_colors = [_cluster_color(result.labels[n], k) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=45,
                            edgecolors="white", linewidths=0.5, ax=ax)

    if highlight_border:
        border_all = set()
        for s in result.border_nodes.values():
            border_all |= s
        if border_all:
            nx.draw_networkx_nodes(
                G, pos, nodelist=list(border_all), node_size=90,
                node_color="none", edgecolors="black", linewidths=1.6, ax=ax,
            )

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(
        title or f"Partitioned into {k} clusters "
                 f"(T={result.border_count} border vertices, "
                 f"T/n={result.border_fraction_total():.3f}, "
                 f"cut={result.cut_size} edges, Q={result.modularity(G):.3f})",
        fontsize=12,
    )
    if show and should_show:
        plt.tight_layout()
        plt.show()
    return ax

