"""
Min-cut partitioning for Divide and Conquer (Section 3.1 of the report). For the
smallest number of clusters k that keeps every cluster within the maximum cluster
size, and the next few above it, spectral clustering (Ng et al. 2002) and recursive
Kernighan-Lin bisection run from several seeds, every candidate is refined with
Fiduccia-Mattheyses, and the partition with the lowest cut is kept.
"""

import math
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if __name__ == "__main__":  # run directly: the other folders of src
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)
                    for d in ("pssa/pssa_wbc", "benchmark_graphs")]

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
        return len(self.cross_edges)

    def cut_fraction(self, G: "nx.Graph") -> float:
        m = G.number_of_edges()
        return self.cut_size / m if m else 0.0

    def border_fraction(self, cid: int) -> float:
        n = len(self.communities.get(cid, ()))
        return len(self.border_nodes.get(cid, ())) / n if n else 0.0

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
            f"{len(self.labels)} nodes, {self.cut_size} cross-cluster edges cut"
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# k-means
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[rng.integers(n)]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for i in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers[i] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, ((X - centers[i]) ** 2).sum(1))
    return centers


def _kmeans(
    X: np.ndarray, k: int, seed: Optional[int] = None,
    n_init: int = 10, max_iter: int = 300,
) -> np.ndarray:
    """k-means with k-means++ seeding, the best of n_init runs."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = _kmeanspp_init(X, k, rng)
        labels = None
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = d2.argmin(1)
            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centers[c] = X[mask].mean(0)
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d2.argmin(1)
        inertia = d2[np.arange(n), labels].sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Spectral clustering (Ng et al. 2002)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def spectral_kway_labels(
    G: "nx.Graph", k: int, seed: Optional[int] = None,
    sparse_threshold: int = 1500,
) -> Labels:
    """Spectral clustering of Ng et al. (2002), k-means on the top k eigenvectors of the normalized adjacency matrix."""
    nodes = list(G.nodes())
    n = len(nodes)
    if k >= n:
        return {v: i for i, v in enumerate(nodes)}

    rng = np.random.default_rng(seed)

    if n <= sparse_threshold:
        U = _spectral_embedding_dense(G, nodes, k)
    else:
        try:
            U = _spectral_embedding_sparse(G, nodes, k, rng)
        except Exception:
            U = _spectral_embedding_dense(G, nodes, k)

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    U = U / norms

    label_arr = _kmeans(U, k, seed=seed)
    return {nodes[i]: int(label_arr[i]) for i in range(n)}


def _spectral_embedding_dense(G: "nx.Graph", nodes: List[Node], k: int) -> np.ndarray:
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = A.sum(1)
    deg_safe = np.where(deg == 0, 1.0, deg)
    Dm12 = 1.0 / np.sqrt(deg_safe)
    A_sym = (A * Dm12[:, None]) * Dm12[None, :]
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:k]]


def _spectral_embedding_sparse(
    G: "nx.Graph", nodes: List[Node], k: int, rng: np.random.Generator,
) -> np.ndarray:
    n = len(nodes)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)
    deg = np.asarray(A.sum(1)).ravel()
    deg_safe = np.where(deg == 0, 1.0, deg)
    Dm12 = 1.0 / np.sqrt(deg_safe)
    Dm12_mat = sp.diags(Dm12)
    A_sym = Dm12_mat @ A @ Dm12_mat
    A_sym = (A_sym + A_sym.T) * 0.5  # guard against float asymmetry

    v0 = rng.standard_normal(n)
    ncv = min(n - 1, max(2 * k + 1, 3 * k))
    eigvals, U = spla.eigsh(A_sym, k=k, which="LA", v0=v0, ncv=ncv)
    order = np.argsort(eigvals)[::-1]
    return U[:, order]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Recursive Kernighan-Lin bisection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def recursive_bisection_labels(
    G: "nx.Graph", k: int, seed: Optional[int] = None, max_iter: int = 50,
) -> Labels:
    """Split the largest piece in two with Kernighan-Lin bisection until there are k pieces."""
    if k <= 1:
        return {n: 0 for n in G.nodes()}

    rng = random.Random(seed)
    clusters: List[Set[Node]] = [set(G.nodes())]

    def _fallback_split(nodes: Set[Node]) -> Tuple[Set[Node], Set[Node]]:
        ordered = sorted(nodes, key=str)
        mid = len(ordered) // 2
        return set(ordered[:mid]), set(ordered[mid:])

    while len(clusters) < k:
        if all(len(c) <= 1 for c in clusters):
            break  # can't usefully split any further
        idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        nodes = clusters.pop(idx)
        if len(nodes) <= 1:
            clusters.append(nodes)
            continue

        sub = G.subgraph(nodes)
        a: Set[Node]
        b: Set[Node]
        if sub.number_of_edges() == 0:
            a, b = _fallback_split(nodes)
        else:
            try:
                a, b = nx.algorithms.community.kernighan_lin_bisection(
                    sub, max_iter=max_iter, seed=rng.randint(0, 2 ** 31 - 1),
                )
            except Exception:
                a, b = _fallback_split(nodes)
            if not a or not b:
                a, b = _fallback_split(nodes)

        clusters.append(set(a))
        clusters.append(set(b))

    labels: Labels = {}
    for cid, nodes in enumerate(clusters):
        for n in nodes:
            labels[n] = cid
    return labels


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fiduccia-Mattheyses refinement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fm_refine(
    G: "nx.Graph", labels: Labels, k: int,
    balance_tol: float = 0.15, max_passes: int = 50,
    seed: Optional[int] = None,
    max_size: Optional[int] = None, min_size: Optional[int] = None,
) -> Labels:
    """Fiduccia-Mattheyses, moving single vertices to a neighbouring cluster whenever it reduces the cut, in rounds until no move helps, within the size limits."""
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
    sizes = Counter(labels.values())
    rng = random.Random(seed)
    nodes = list(G.nodes())

    for _ in range(max_passes):
        rng.shuffle(nodes)
        moved = 0
        for u in nodes:
            cu = labels[u]
            nbr_counts = Counter(labels[v] for v in G.neighbors(u))
            internal = nbr_counts.get(cu, 0)
            best_c, best_gain = cu, 0
            for c, cnt in nbr_counts.items():
                if c == cu:
                    continue
                gain = cnt - internal
                if gain > best_gain and sizes[c] + 1 <= hi and sizes[cu] - 1 >= lo:
                    best_gain, best_c = gain, c
            if best_c != cu:
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
    """Bisect every cluster larger than max_size until all fit."""
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
    """Merge every cluster smaller than min_size into the neighbour it shares the most cut edges with: (labels, ok)."""
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

        nbr_edge_count: Counter = Counter()
        for u in nodes:
            for v in G.neighbors(u):
                cv = labels[v]
                if cv != cid:
                    nbr_edge_count[cv] += 1

        # the neighbour sharing the most cut edges first
        candidates = sorted(nbr_edge_count.keys(), key=lambda c: -nbr_edge_count[c])
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
# Disconnected graphs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _allocate_k_across_components(sizes: List[int], k: int) -> List[int]:
    """Share k clusters among the connected components in proportion to their size, at least one each."""
    n_total = sum(sizes)
    raw = [k * s / n_total for s in sizes]
    alloc = [max(1, int(math.floor(r))) for r in raw]
    diff = k - sum(alloc)

    order = sorted(range(len(sizes)), key=lambda i: (raw[i] - math.floor(raw[i])), reverse=True)
    i = 0
    while diff > 0 and order:
        alloc[order[i % len(order)]] += 1
        diff -= 1
        i += 1
    i = 0
    by_alloc_desc = sorted(range(len(sizes)), key=lambda i: -alloc[i])
    while diff < 0 and any(a > 1 for a in alloc):
        idx = by_alloc_desc[i % len(by_alloc_desc)]
        if alloc[idx] > 1:
            alloc[idx] -= 1
            diff += 1
        i += 1
    return alloc


def _partition_disconnected(
    G: "nx.Graph", k: int, method: str, balance_tol: float,
    refine: bool, refine_passes: int, seed: Optional[int],
) -> "PartitionResult":
    comps = [set(c) for c in nx.connected_components(G)]
    alloc = _allocate_k_across_components([len(c) for c in comps], k)

    labels: Labels = {}
    next_id = 0
    for comp, kc in zip(comps, alloc):
        sub = G.subgraph(comp)
        if kc <= 1 or sub.number_of_nodes() <= 1:
            for n in comp:
                labels[n] = next_id
            next_id += 1
            continue

        if method == "spectral":
            sub_labels = spectral_kway_labels(sub, kc, seed=seed)
        elif method == "recursive_bisection":
            sub_labels = recursive_bisection_labels(sub, kc, seed=seed)
        else:
            raise ValueError(f"Unknown method {method!r}")

        if refine:
            sub_labels = fm_refine(
                sub, sub_labels, kc, balance_tol=balance_tol,
                max_passes=refine_passes, seed=seed,
            )

        remap = {old: next_id + i for i, old in
                 enumerate(sorted(set(sub_labels.values()), key=str))}
        for n, c in sub_labels.items():
            labels[n] = remap[c]
        next_id += len(remap)

    return _build_result(G, labels, method)


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
# Main entry point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def partition_graph(
    G: "nx.Graph",
    k: Optional[int] = None,
    target_size: Optional[int] = None,
    method: str = "spectral",
    balance_tol: float = 0.15,
    refine: bool = True,
    refine_passes: int = 50,
    seed: Optional[int] = None,
    max_cluster_size: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
) -> PartitionResult:
    """Partition G into k clusters with the lowest cut, within the size bounds."""
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot partition an empty graph.")
    if method not in ("spectral", "recursive_bisection"):
        raise ValueError(f"method must be 'spectral' or 'recursive_bisection', got {method!r}")

    n_total = G.number_of_nodes()
    if k is None:
        if target_size is None:
            raise ValueError("Pass either k (number of clusters) or target_size (nodes per cluster).")
        k = max(1, round(n_total / target_size))

    if max_cluster_size is not None:
        k_min_required = math.ceil(n_total / max_cluster_size)
        if k is not None and target_size is None and k < k_min_required:
            # k was explicitly requested and it's arithmetically impossible
            raise ValueError(
                f"k={k} cannot satisfy max_cluster_size={max_cluster_size} on "
                f"{n_total} nodes; need k >= {k_min_required}."
            )
        k = max(k, k_min_required)

    k = max(1, min(k, n_total))

    if not nx.is_connected(G):
        result = _partition_disconnected(G, k, method, balance_tol, refine, refine_passes, seed)
        labels = dict(result.labels)
    else:
        if k <= 1:
            labels: Labels = {n: 0 for n in G.nodes()}
        elif method == "spectral":
            labels = spectral_kway_labels(G, k, seed=seed)
        else:
            labels = recursive_bisection_labels(G, k, seed=seed)

        if refine and k > 1:
            labels = fm_refine(G, labels, k, balance_tol=balance_tol,
                                max_passes=refine_passes, seed=seed,
                                max_size=max_cluster_size, min_size=min_cluster_size)

    if max_cluster_size is not None:
        sizes = Counter(labels.values())
        if any(s > max_cluster_size for s in sizes.values()):
            labels = enforce_size_cap(G, labels, max_cluster_size, seed=seed)
            # polish the new split boundaries, still respecting the cap
            k_after = len(set(labels.values()))
            labels = fm_refine(G, labels, k_after, balance_tol=balance_tol,
                                max_passes=refine_passes, seed=seed,
                                max_size=max_cluster_size, min_size=min_cluster_size)

    return _build_result(G, labels, method)


def _stable_seed(*parts) -> int:
    """A seed from its parts that is the same in every process."""
    s = "|".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The min-cut search
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def partition_graph_auto(
    G: "nx.Graph",
    min_cluster_size: int,
    max_cluster_size: int,
    methods: Tuple[str, ...] = ("spectral", "recursive_bisection"),
    k_span: int = 3,
    seeds_per_config: int = 8,
    time_budget_s: Optional[float] = None,
    balance_tol: float = 0.15,
    refine_passes: int = 30,
    sparse_threshold: int = 1500,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> PartitionResult:
    """The min-cut search, over several k, both methods and several seeds, keeping the lowest cut within both size bounds."""
    n_total = G.number_of_nodes()
    if not nx.is_connected(G):
        raise NotImplementedError(
            "partition_graph_auto currently expects a connected graph; "
            "run partition_graph(..., method=...) with an explicit k on "
            "each component, or connect the graph first."
        )

    k_min, k_max = feasible_k_range(n_total, min_cluster_size, max_cluster_size)
    k_candidates = list(range(k_min, min(k_max, k_min + k_span - 1) + 1))

    # seed=None draws fresh OS entropy every call; an int fixes every trial's seed
    root_seed = seed if seed is not None else random.SystemRandom().randrange(2**31)

    # seed outer, then k, then method, so a time budget samples every (method, k)
    combos = [
        (k, method, s)
        for s in range(seeds_per_config)
        for k in k_candidates
        for method in methods
    ]

    best: Optional[PartitionResult] = None
    best_meta = None
    n_tried = 0
    n_valid = 0
    t_start = time.time()

    for k, method, s in combos:
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            break

        trial_seed = _stable_seed(root_seed, k, method, s)
        if method == "spectral":
            labels = spectral_kway_labels(G, k, seed=trial_seed, sparse_threshold=sparse_threshold)
        elif method == "recursive_bisection":
            labels = recursive_bisection_labels(G, k, seed=trial_seed)
        else:
            raise ValueError(f"Unknown method {method!r}")

        labels = fm_refine(G, labels, k, balance_tol=balance_tol,
                            max_passes=refine_passes, seed=trial_seed,
                            max_size=max_cluster_size, min_size=min_cluster_size)
        labels, ok = enforce_bounds(G, labels, min_cluster_size, max_cluster_size, seed=trial_seed)
        n_tried += 1

        if not ok:
            if verbose:
                print(f"  [skip]  method={method:20s} k={k:4d} trial_seed={trial_seed}  "
                      f"could not satisfy both bounds")
            continue

        result = _build_result(G, labels, method)
        n_valid += 1
        if verbose:
            note = "" if result.k == k else f" (enforce_size_cap raised k {k}->{result.k})"
            print(f"  [ok]    method={method:20s} target_k={k:4d} actual_k={result.k:4d}{note} "
                  f"trial_seed={trial_seed}  cut={result.cut_size}  sizes={sorted(result.sizes.values())}")

        if best is None or result.cut_size < best.cut_size:
            best, best_meta = result, dict(target_k=k, actual_k=result.k, method=method, trial_seed=trial_seed)

    if best is None:
        raise RuntimeError(
            f"No candidate satisfied both min_cluster_size={min_cluster_size} and "
            f"max_cluster_size={max_cluster_size} after {n_tried} attempts across "
            f"k in {k_candidates} and methods {methods}. Try increasing k_span / "
            f"seeds_per_config, or loosening the bounds."
        )

    if verbose:
        print(f"\nBest: method={best_meta['method']} target_k={best_meta['target_k']} "
              f"actual_k={best_meta['actual_k']} trial_seed={best_meta['trial_seed']}  cut={best.cut_size}  "
              f"({n_valid}/{n_tried} candidates satisfied both bounds)")

    return best


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
                 f"(cut={result.cut_size} edges, Q={result.modularity(G):.3f})",
        fontsize=12,
    )
    if show and should_show:
        plt.tight_layout()
        plt.show()
    return ax


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Demo
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from graphs import build_default_demo_graph, build_demo_graph_1000, build_demo_graph_1000_v2, build_demo_graph_5000

    #G, k_true, gt_labels = build_default_demo_graph(seed=42)
    #G, k_true, gt_labels = build_demo_graph_1000(seed=42)
    #G, k_true, gt_labels = build_demo_graph_1000_v2(seed=42)
    G, k_true, gt_labels = build_demo_graph_5000(seed=42)

    print(f"Benchmark graph (testGraph.py-style, {k_true} communities): "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"avg degree {2 * G.number_of_edges() / G.number_of_nodes():.2f}, "
          f"connected={nx.is_connected(G)}")

    # search settings

    MIN_CLUSTER_SIZE = 1
    MAX_CLUSTER_SIZE = 50
    METHODS = ["spectral"]
    SEEDS_PER_CONFIG = 40
    K_SPAN = 5
    BALANCE_TOL = 0.15
    REFINE_PASSES = 30
    SEARCH_SEED = 7
    TIME_BUDGET_S = None #wall-clock time in seconds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig, axes = plt.subplots(1, len(METHODS), figsize=(10 * len(METHODS), 10), squeeze=False)
    axes = axes.ravel()

    for ax, method in zip(axes, METHODS):
        result = partition_graph_auto(
            G, min_cluster_size=MIN_CLUSTER_SIZE, max_cluster_size=MAX_CLUSTER_SIZE,
            methods=(method,), seeds_per_config=SEEDS_PER_CONFIG, k_span=K_SPAN,
            balance_tol=BALANCE_TOL, refine_passes=REFINE_PASSES, seed=SEARCH_SEED,
            time_budget_s=TIME_BUDGET_S,
        )
        print(f"\n--- method={method} ---")
        print(result.summary())
        print(f"cut fraction: {result.cut_fraction(G):.1%}   "
              f"modularity: {result.modularity(G):.3f}")
        assert MIN_CLUSTER_SIZE <= min(result.sizes.values())
        assert max(result.sizes.values()) <= MAX_CLUSTER_SIZE
        draw_partition(G, result, ax=ax, show=False,
                        title=f"{method}: {result.k} clusters, "
                              f"cut={result.cut_size}, Q={result.modularity(G):.3f}")

    plt.tight_layout()
    fig.savefig("min_cut_partitioning_demo.png", dpi=150, bbox_inches="tight")
    print("\nSaved demo figure to min_cut_partitioning_demo.png")
    plt.show()

    # one cluster of the partition, embedded by PSSA wBC
    try:
        from pssa_wbc import ImprovedPSSA, _perimeter_nodes

        result = partition_graph_auto(
            G, min_cluster_size=MIN_CLUSTER_SIZE, max_cluster_size=MAX_CLUSTER_SIZE,
            methods=tuple(METHODS), seeds_per_config=SEEDS_PER_CONFIG, k_span=K_SPAN,
            balance_tol=BALANCE_TOL, refine_passes=REFINE_PASSES, seed=SEARCH_SEED,
            time_budget_s=TIME_BUDGET_S,
        )
        # a mid-sized cluster
        cid = sorted(result.communities, key=lambda c: len(result.communities[c]))[len(result.communities) // 2]
        cluster_I = result.subgraphs[cid]
        cluster_border = result.border_nodes[cid]
        print(f"\n--- feeding cluster {cid} ({cluster_I.number_of_nodes()} nodes, "
              f"{cluster_I.number_of_edges()} internal edges, "
              f"{len(cluster_border)} border nodes) into ImprovedPSSA ---")

        L = 7
        embedder = ImprovedPSSA(
            L=L, tmax=400_000, gibbs=True, seed=-1, verbose=False,
            border_nodes=cluster_border, border_mode="permute",
        )
        pssa_result = embedder.run(cluster_I)
        print(pssa_result)

        perimeter = _perimeter_nodes(L)
        ok = all(
            any(u in perimeter for u in pssa_result.phi[i])
            for i in cluster_border
        )
        print("Border perimeter-touch constraint:", "OK" if ok else "VIOLATED")
    except ImportError:
        print("\n(pssa_wbc / compiled cheap_pssa_core not found -- "
              "skipping the ImprovedPSSA integration check.)")