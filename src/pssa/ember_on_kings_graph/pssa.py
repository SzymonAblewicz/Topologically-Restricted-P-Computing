"""
The Ember reconstruction of PSSA (MIT licence, third_party/ember) reverted to the King's
graph (Section 2.2 of the report), the baseline of the improved PSSA. The hardware graph,
the guiding pattern, the initial placement and the annealing schedule are put back to
those of Sugie et al., and Ember's scaling of tmax with the size of the hardware is kept.

    success, phi = embed(I, L=17, seed=42)
"""

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Type aliases
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NodeI = int
NodeH = int
Phi = Dict[NodeI, List[NodeH]]
InvPhi = Dict[NodeH, NodeI]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The King's graph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def twoD2oneD(row: int, col: int, L: int) -> int:
    """(row, col) -> hardware vertex number, from 1."""
    return int(row * L + col + 1)


def oneD2twoD(idx: int, L: int) -> Tuple[int, int]:
    """Hardware vertex number -> (row, col)."""
    row = int((idx - 1) / L)
    col = int((idx - 1) % L)
    return row, col


def build_hardware_graph(L: int) -> nx.Graph:
    """The King's graph KG(L), a square grid with both diagonals."""
    H = nx.Graph()
    H.add_nodes_from(range(1, L * L + 1))
    for row in range(L):
        for col in range(L):
            u = twoD2oneD(row, col, L)
            # the four forward directions list every edge once
            for row_n, col_n in [(row, col + 1), (row + 1, col - 1),
                                  (row + 1, col), (row + 1, col + 1)]:
                if 0 <= row_n < L and 0 <= col_n < L:
                    v = twoD2oneD(row_n, col_n, L)
                    H.add_edge(u, v)
    H.graph["family"] = "kings_graph"
    H.graph["L"] = L
    return H


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The guiding pattern
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_guiding_pattern(H: nx.Graph, L: int) -> Dict[int, List[NodeH]]:
    """The guiding pattern, the complete graph embedding of K_{L+1} on KG(L) of Okuyama et al. (2016)."""
    N_complete = L + 1
    KG_complete = [[0] * L for _ in range(L)]
    phi_complete: Dict[int, List[NodeH]] = {}

    # super vertex 1 fills the first column
    phi_1 = [twoD2oneD(row, 0, L) for row in range(L)]
    for row in range(L):
        KG_complete[row][0] = 1
    phi_complete[1] = phi_1

    # super vertices 2, 4, 6, ... descend
    for i in range(2, N_complete + 1, 2):
        phi_i = []
        for k in range(1, L):
            col = k
            row = i - 2 + k - 1
            if row >= L:
                row = L - (row - L) - 1
            KG_complete[row][col] = i
            phi_i.append(twoD2oneD(row, col, L))
        phi_complete[i] = phi_i

    # super vertices 3, 5, 7, ... ascend
    for i in range(3, N_complete + 1, 2):
        phi_i = []
        for k in range(1, L):
            col = k
            row = i - 2 - k + 1
            if row < 0:
                row = -row - 1
            KG_complete[row][col] = i
            phi_i.append(twoD2oneD(row, col, L))
        phi_complete[i] = phi_i

    return phi_complete


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The initial placement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def initial_placement(I: nx.Graph, gp: Dict[int, List[NodeH]], L: int) -> Phi:
    """The initial placement, the guiding pattern split into |V(I)| connected pieces."""
    nodes_I = sorted(I.nodes())
    N = len(nodes_I)
    N_complete = L + 1

    if N > N_complete * L:
        raise ValueError(
            f"Input graph has {N} vertices; King's graph of width {L} "
            f"cannot host more than {N_complete * L} super vertices via "
            f"the guiding-pattern split (increase L)."
        )

    quo = N // N_complete
    rem = N % N_complete

    # the N pieces, in guiding pattern order
    split_chains: List[List[NodeH]] = []
    for i in range(1, N_complete + 1):
        i_split_into = quo + (1 if i <= rem else 0)
        if i_split_into == 0:
            continue
        chain = gp[i]
        i_len = len(chain)
        i_quo = i_len // i_split_into
        i_rem = i_len % i_split_into
        last_ele_idx = -1
        for j in range(1, i_split_into + 1):
            num_ele = i_quo + (1 if j > (i_split_into - i_rem) else 0)
            piece = chain[last_ele_idx + 1: last_ele_idx + 1 + num_ele]
            last_ele_idx += num_ele
            split_chains.append(piece)

    if len(split_chains) < N:
        # never reached, given the check above
        used: Set[NodeH] = {u for c in split_chains for u in c}
        for u in range(1, L * L + 1):
            if len(split_chains) >= N:
                break
            if u not in used:
                split_chains.append([u])
                used.add(u)

    phi: Phi = {}
    for src, chain in zip(nodes_I, split_chains[:N]):
        phi[src] = list(chain)
    return phi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def invert(phi: Phi) -> InvPhi:
    inv: InvPhi = {}
    for i, path in phi.items():
        for u in path:
            inv[u] = i
    return inv


def eemb(phi: Phi, I: nx.Graph, H: nx.Graph, inv: InvPhi) -> int:
    """The number of embedded edges, Eemb(phi)."""
    covered: Set[Tuple[NodeI, NodeI]] = set()
    for u, v in H.edges():
        i = inv.get(u)
        j = inv.get(v)
        if i is None or j is None or i == j:
            continue
        edge = (min(i, j), max(i, j))
        if edge not in covered and I.has_edge(i, j):
            covered.add(edge)
    return len(covered)


def _leaves(chain: List[NodeH], H_adj: Dict[NodeH, Set[NodeH]]) -> List[NodeH]:
    """The leaves of a super vertex, the hardware vertices it can lose and stay connected."""
    if len(chain) <= 1:
        return list(chain)
    chain_set = set(chain)
    deg_in_chain: Dict[NodeH, int] = {}
    for u in chain:
        d = sum(1 for w in H_adj.get(u, ()) if w in chain_set)
        deg_in_chain[u] = d
    leaves = [u for u, d in deg_in_chain.items() if d <= 1]
    if leaves:
        return leaves
    out: List[NodeH] = []
    for u in chain:
        remaining = chain_set - {u}
        if not remaining:
            continue
        start = next(iter(remaining))
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for w in H_adj.get(x, ()):
                if w in remaining and w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(seen) == len(remaining):
            out.append(u)
    return out


def _remove_leaf(chain: List[NodeH], u: NodeH) -> List[NodeH]:
    return [w for w in chain if w != u]


def _attach_leaf(chain: List[NodeH], u: NodeH, v: NodeH) -> List[NodeH]:  # noqa: ARG001
    return chain + [u]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The linear two-phase annealing schedule of Sugie et al.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class KingsGraphSchedule:
    """The linear two-phase annealing schedule of Sugie et al. (2018), with tmax scaled with L^2 from 7e7 at L = 320."""

    PAPER_L2 = 320 * 320       # 102,400 nodes, the paper's benchmark size
    PAPER_TMAX = 7 * 10 ** 7

    def __init__(
        self,
        tmax: Optional[int] = None,
        L: int = 320,
        T0: float = 6.0315,
        Thalf: float = 3.3435,
        ps0: float = 1.0,
        ps_end: float = 0.0,
        pa0: float = 0.095,
        pa_end: float = 0.487,
    ):
        self.L = L
        self.tmax = tmax if tmax is not None else max(
            200_000, int(self.PAPER_TMAX * (L * L) / self.PAPER_L2)
        )
        self.T0 = T0
        self.Thalf = Thalf
        self.ps0 = ps0
        self.ps_end = ps_end
        self.pa0 = pa0
        self.pa_end = pa_end
        self._half = self.tmax // 2

    def temperature(self, t: int) -> float:
        if t < self._half:
            return self.T0 * (1 - 2 * t / self.tmax)
        return self.Thalf * (2 - 2 * t / self.tmax)

    def ps(self, t: int) -> float:
        return self.ps0 + (self.ps_end - self.ps0) * t / self.tmax

    def pa(self, t: int) -> float:
        return self.pa0 + (self.pa_end - self.pa0) * t / self.tmax

    def summary(self) -> str:
        return (
            f"KingsGraphSchedule(L={self.L}): "
            f"T0={self.T0}, Thalf={self.Thalf}, tmax={self.tmax:,}, "
            f"pa0={self.pa0}, pa_end={self.pa_end}"
        )


def _shift_direction_prob(i, j, phi, deg, weighted):
    if not weighted:
        return 0.5
    dri = len(phi[i]) / max(deg[i], 1)
    drj = len(phi[j]) / max(deg[j], 1)
    denom = dri + drj
    return 0.5 if denom == 0 else dri / denom


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Algorithm 1: PSSA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pssa(
    I: nx.Graph,
    H: nx.Graph,
    gp: Dict[int, List[NodeH]],
    schedule: KingsGraphSchedule,
    weighted: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    deadline: Optional[float] = None,
) -> Tuple[Phi, int]:
    """Algorithm 1 of Sugie et al. (2018), the annealing search."""
    if seed is not None:
        random.seed(seed)

    L = schedule.L
    nodes_I = sorted(I.nodes())
    edges_I = list(I.edges())
    m_I = len(edges_I)
    deg = dict(I.degree())
    H_adj: Dict[NodeH, Set[NodeH]] = {u: set(H.neighbors(u)) for u in H.nodes()}

    gp_lookup: Dict[NodeH, int] = {}
    for gk, path in gp.items():
        for hw in path:
            gp_lookup[hw] = gk

    phi = initial_placement(I, gp, L)
    inv = invert(phi)
    cur_e = eemb(phi, I, H, inv)

    phi_best = {i: list(p) for i, p in phi.items()}
    eemb_best = cur_e

    if eemb_best == m_I:
        return phi_best, eemb_best

    tmax = schedule.tmax
    log_every = max(1, tmax // 10)

    for t in range(tmax):
        if deadline is not None and t % 1000 == 0 and time.time() > deadline:
            break
        T = schedule.temperature(t)
        ps = schedule.ps(t)
        pa = schedule.pa(t)

        if verbose and t % log_every == 0:
            print(f"  t={t:>9,} ({100*t//tmax:3d}%)  T={T:6.3f}  Eemb={cur_e}/{m_I}")

        if random.random() < ps:
            # shift
            candidates = [i for i in nodes_I if len(phi[i]) > 1]
            if not candidates:
                continue
            i = random.choice(candidates)
            leaves_i = _leaves(phi[i], H_adj)
            if not leaves_i:
                continue
            u = random.choice(leaves_i)
            allow_any = random.random() < pa

            candidate_jv: List[Tuple[NodeI, NodeH]] = []
            for v in H_adj[u]:
                j = inv.get(v)
                if j is None or j == i:
                    continue
                if allow_any or gp_lookup.get(u) == gp_lookup.get(v):
                    candidate_jv.append((j, v))

            if not candidate_jv:
                continue

            j, v = random.choice(candidate_jv)
            p_ij = _shift_direction_prob(i, j, phi, deg, weighted)

            if random.random() < p_ij:
                new_phi_i = _remove_leaf(phi[i], u)
                new_phi_j = _attach_leaf(phi[j], u, v)
                hw_changed, new_owner = u, j
            else:
                if len(phi[j]) < 2:
                    continue
                if v not in _leaves(phi[j], H_adj):
                    continue
                new_phi_j = _remove_leaf(phi[j], v)
                new_phi_i = _attach_leaf(phi[i], v, u)
                hw_changed, new_owner = v, i

            phi_prop = {k: phi[k] for k in phi}
            phi_prop[i] = new_phi_i
            phi_prop[j] = new_phi_j
            inv_prop = dict(inv)
            inv_prop[hw_changed] = new_owner

        else:
            # swap
            if not edges_I:
                continue
            ie, k = random.choice(edges_I)
            adjacent_j: Set[NodeI] = set()
            for hw_k in phi[k]:
                for hw_nbr in H_adj[hw_k]:
                    j_cand = inv.get(hw_nbr)
                    if j_cand is not None and j_cand != k and j_cand != ie:
                        adjacent_j.add(j_cand)
            if not adjacent_j:
                continue
            j = random.choice(list(adjacent_j))

            phi_prop = {k2: phi[k2] for k2 in phi}
            phi_prop[ie] = list(phi[j])
            phi_prop[j] = list(phi[ie])
            inv_prop = dict(inv)
            for hw in phi[ie]:
                inv_prop[hw] = j
            for hw in phi[j]:
                inv_prop[hw] = ie

        # the Metropolis rule
        prop_e = eemb(phi_prop, I, H, inv_prop)
        delta = prop_e - cur_e

        if T > 0:
            accept = math.exp(min(delta / T, 0)) > random.random() if delta < 0 else True
        else:
            accept = delta >= 0

        if accept:
            phi = phi_prop
            inv = inv_prop
            cur_e = prop_e
            if cur_e > eemb_best:
                phi_best = {k: list(p) for k, p in phi.items()}
                eemb_best = cur_e
            if eemb_best == m_I:
                if verbose:
                    print(f"  Embedding found at t={t:,}")
                return phi_best, eemb_best

    return phi_best, eemb_best


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Algorithm 2: terminal search
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _repr_counts(phi, I, H, inv):
    counts = defaultdict(int)
    for u, v in H.edges():
        i = inv.get(u)
        j = inv.get(v)
        if i is None or j is None or i == j:
            continue
        edge = (min(i, j), max(i, j))
        if I.has_edge(i, j):
            counts[edge] += 1
    return counts


def _is_deletable(u, i, phi, I, H, inv, repr_counts):
    sv = phi[i]
    if len(sv) <= 1:
        return False
    remaining = [w for w in sv if w != u]
    if not nx.is_connected(H.subgraph(remaining)):
        return False
    for v in H.neighbors(u):
        j = inv.get(v)
        if j is None or j == i or not I.has_edge(i, j):
            continue
        edge = (min(i, j), max(i, j))
        via_u = sum(1 for w in H.neighbors(u) if inv.get(w) == j)
        if repr_counts.get(edge, 0) - via_u <= 0:
            return False
    return True


def _bfs_path(i, j, phi, H, free):
    source = set(phi[i])
    target = set(phi[j])
    queue = deque(source)
    parent: Dict[NodeH, Optional[NodeH]] = {u: None for u in source}
    visited = set(source)
    while queue:
        u = queue.popleft()
        for v in H.neighbors(u):
            if v in visited:
                continue
            if v not in free and v not in target:
                continue
            if v in target:
                path: List[NodeH] = []
                cur = u
                while cur not in source:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            visited.add(v)
            parent[v] = u
            queue.append(v)
    return None


def terminal_search(phi: Phi, I: nx.Graph, H: nx.Graph) -> Phi:
    """Algorithm 2 of Sugie et al. (2021), the terminal search, unchanged from Ember."""
    phi = {i: list(p) for i, p in phi.items()}
    inv = invert(phi)
    rc = _repr_counts(phi, I, H, inv)

    hw_nodes = list(H.nodes())
    n_hw = len(hw_nodes)
    free: Set[NodeH] = set()
    no_del = 0
    idx = 0

    while no_del < n_hw:
        u = hw_nodes[idx % n_hw]
        idx += 1
        i = inv.get(u)
        if i is None or u in free:
            no_del += 1
            continue
        if _is_deletable(u, i, phi, I, H, inv, rc):
            phi[i] = [w for w in phi[i] if w != u]
            del inv[u]
            free.add(u)
            for v in H.neighbors(u):
                j = inv.get(v)
                if j is None or j == i or not I.has_edge(i, j):
                    continue
                edge = (min(i, j), max(i, j))
                via_u = sum(1 for w in H.neighbors(u) if inv.get(w) == j)
                rc[edge] = max(0, rc.get(edge, 0) - via_u)
            no_del = 0
        else:
            no_del += 1

    for i in sorted(I.nodes()):
        for j in I.neighbors(i):
            if j <= i:
                continue
            linked = any(inv.get(v) == j for u in phi[i] for v in H.neighbors(u))
            if linked:
                continue
            path = _bfs_path(i, j, phi, H, free)
            if path is not None:
                for v in path:
                    phi[i].append(v)
                    inv[v] = i
                    free.discard(v)
                edge = (min(i, j), max(i, j))
                rc[edge] = rc.get(edge, 0) + 1

    return phi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Validity check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def is_valid_embedding(phi: Phi, I: nx.Graph, H: nx.Graph) -> bool:
    """Whether phi is a minor-embedding, with (M1) connected, (M2) disjoint super vertices and (M3) every edge embedded."""
    for i, sv in phi.items():
        if not sv:
            return False
        if len(sv) > 1 and not nx.is_connected(H.subgraph(sv)):
            return False
    all_hw = [u for sv in phi.values() for u in sv]
    if len(all_hw) != len(set(all_hw)):
        return False
    for i, j in I.edges():
        if not any(H.has_edge(u, v) for u in phi[i] for v in phi[j]):
            return False
    return True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Result container
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class PSSAResult:
    success: bool
    phi: Phi
    eemb: int
    m_I: int
    wall_time: float
    L: int
    n_hw: int

    @property
    def coverage(self) -> float:
        return self.eemb / self.m_I if self.m_I > 0 else 1.0

    def __str__(self):
        return (
            f"PSSAResult(KingsGraph(L={self.L}), "
            f"{'ok' if self.success else 'fail'} "
            f"{self.eemb}/{self.m_I} edges, {self.wall_time:.2f}s)"
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PSSA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImprovedPSSA:
    """PSSA on KG(L), the annealing and then the terminal search."""

    def __init__(
        self,
        L: int = 8,
        tmax: Optional[int] = None,
        weighted: bool = False,
        T0: float = 6.0315,
        Thalf: float = 3.3435,
        pa_end: float = 0.487,
        seed: Optional[int] = None,
        verbose: bool = False,
        hardware_graph: Optional[nx.Graph] = None,
    ):
        self.L = L
        self.weighted = weighted
        self.seed = seed
        self.verbose = verbose

        self.H = hardware_graph if hardware_graph is not None else build_hardware_graph(L)
        n_hw = self.H.number_of_nodes()

        self.schedule = KingsGraphSchedule(
            tmax=tmax, L=L, T0=T0, Thalf=Thalf, pa_end=pa_end,
        )
        self.gp = build_guiding_pattern(self.H, L)

        if verbose:
            print(f"Hardware: King's graph L={L}, {n_hw} nodes, {self.H.number_of_edges()} edges")
            print(f"Guiding pattern: {len(self.gp)} super vertices (K_{L+1})")
            print(self.schedule.summary())

    def run(self, I: nx.Graph, deadline: Optional[float] = None) -> PSSAResult:
        t0 = time.perf_counter()
        m_I = I.number_of_edges()

        phi_pssa, _ = pssa(
            I, self.H, self.gp, self.schedule,
            weighted=self.weighted, seed=self.seed, verbose=self.verbose,
            deadline=deadline,
        )
        phi_final = terminal_search(phi_pssa, I, self.H)
        inv_final = invert(phi_final)
        e_final = eemb(phi_final, I, self.H, inv_final)
        success = (e_final == m_I) and is_valid_embedding(phi_final, I, self.H)

        return PSSAResult(
            success=success, phi=phi_final, eemb=e_final, m_I=m_I,
            wall_time=time.perf_counter() - t0, L=self.L,
            n_hw=self.H.number_of_nodes(),
        )


def embed(
    I: nx.Graph,
    L: int = 8,
    tmax: Optional[int] = None,
    weighted: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    hardware_graph: Optional[nx.Graph] = None,
) -> Tuple[bool, Phi]:
    """One-line embed interface."""
    algo = ImprovedPSSA(L=L, tmax=tmax, weighted=weighted, seed=seed,
                         verbose=verbose, hardware_graph=hardware_graph)
    result = algo.run(I)
    return result.success, result.phi
