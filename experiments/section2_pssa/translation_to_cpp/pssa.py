"""
The improved PSSA as it was for the speed comparison (Section 2.3.4 of the report),
with the annealing loop both in pure Python (pssa) and in C++ (pssa_core.cpp), before
Delta E was computed locally, so both recount Eemb in full at every step.
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
# The geometric annealing schedule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class KingsGraphSchedule:
    """The geometric annealing schedule, T(t) = T0 alpha^t with T(tmax) = T_end, and the linear p_s(t) and p_a(t) of Sugie et al."""

    PAPER_L2 = 320 * 320       # 102,400 nodes, the paper's benchmark size
    PAPER_TMAX = 7 * 10 ** 7

    def __init__(
        self,
        tmax: Optional[int] = None,
        L: int = 320,
        T0: float = 6.0315,
        Thalf: float = 3.3435,
        T_end: float = 0.01,
        alpha: Optional[float] = None,
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
        self.T_end = T_end
        # alpha so that T(tmax) = T_end, unless given
        self.alpha = alpha if alpha is not None else (T_end / T0) ** (1.0 / self.tmax)
        self.ps0 = ps0
        self.ps_end = ps_end
        self.pa0 = pa0
        self.pa_end = pa_end
        self._half = self.tmax // 2

    def temperature(self, t: int) -> float:
        return self.T0 * (self.alpha ** t)

    def ps(self, t: int) -> float:
        return self.ps0 + (self.ps_end - self.ps0) * t / self.tmax

    def pa(self, t: int) -> float:
        return self.pa0 + (self.pa_end - self.pa0) * t / self.tmax

    def summary(self) -> str:
        return (
            f"KingsGraphSchedule(L={self.L}): "
            f"T0={self.T0}, T_end={self.T_end}, alpha={self.alpha:.6f}, "
            f"tmax={self.tmax:,}, pa0={self.pa0}, pa_end={self.pa_end}"
        )


def _shift_direction_prob(i, j, phi, deg, weighted):
    if not weighted:
        return 0.5
    dri = len(phi[i]) / max(deg[i], 1)
    drj = len(phi[j]) / max(deg[j], 1)
    denom = dri + drj
    return 0.5 if denom == 0 else dri / denom


def _chain_linked(chain_a: List[NodeH], chain_b: List[NodeH], H_adj: Dict[NodeH, Set[NodeH]]) -> bool:
    """Whether two super vertices are adjacent on the hardware."""
    if len(chain_a) <= len(chain_b):
        small, big = chain_a, set(chain_b)
    else:
        small, big = chain_b, set(chain_a)
    for u in small:
        if H_adj[u] & big:
            return True
    return False


def _local_swap_delta(
    phi: Phi, I: nx.Graph, H_adj: Dict[NodeH, Set[NodeH]], ie: NodeI, j: NodeI
) -> int:
    """The change in Eemb that swapping ie and j would cause, computed locally."""
    chain_ie = phi[ie]
    chain_j = phi[j]
    delta = 0

    for m in I.neighbors(ie):
        if m == j:
            continue
        chain_m = phi[m]
        before = _chain_linked(chain_ie, chain_m, H_adj)
        after = _chain_linked(chain_j, chain_m, H_adj)
        delta += int(after) - int(before)

    for m in I.neighbors(j):
        if m == ie:
            continue
        chain_m = phi[m]
        before = _chain_linked(chain_j, chain_m, H_adj)
        after = _chain_linked(chain_ie, chain_m, H_adj)
        delta += int(after) - int(before)

    return delta


def _gibbs_choose_swap_partner(
    phi: Phi,
    I: nx.Graph,
    H_adj: Dict[NodeH, Set[NodeH]],
    ie: NodeI,
    candidates: List[NodeI],
    T: float,
) -> NodeI:
    """The swap partner, drawn from the Gibbs distribution of the local deltas at T(t)."""
    if T <= 0 or len(candidates) == 1:
        # at T = 0, or with one candidate, the weights reduce to the greedy choice
        deltas = [_local_swap_delta(phi, I, H_adj, ie, j) for j in candidates]
        best = max(range(len(candidates)), key=lambda idx: deltas[idx])
        return candidates[best]

    deltas = [_local_swap_delta(phi, I, H_adj, ie, j) for j in candidates]
    # subtract the max before exp, for numerical stability
    scaled = [d / T for d in deltas]
    m = max(scaled)
    weights = [math.exp(s - m) for s in scaled]
    return random.choices(candidates, weights=weights, k=1)[0]


def _local_shift_delta(
    phi: Phi, I: nx.Graph, H_adj: Dict[NodeH, Set[NodeH]], i: NodeI, j: NodeI, u: NodeH
) -> int:
    """The change in Eemb that shifting leaf u from i to j would cause, computed locally."""
    new_phi_i = _remove_leaf(phi[i], u)
    new_phi_j = phi[j] + [u]
    delta = 0

    for m in I.neighbors(i):
        if m == j:
            continue
        chain_m = phi[m]
        before = _chain_linked(phi[i], chain_m, H_adj)
        after = _chain_linked(new_phi_i, chain_m, H_adj)
        delta += int(after) - int(before)

    for m in I.neighbors(j):
        if m == i:
            continue
        chain_m = phi[m]
        before = _chain_linked(phi[j], chain_m, H_adj)
        after = _chain_linked(new_phi_j, chain_m, H_adj)
        delta += int(after) - int(before)

    if I.has_edge(i, j):
        before = _chain_linked(phi[i], phi[j], H_adj)
        after = _chain_linked(new_phi_i, new_phi_j, H_adj)
        delta += int(after) - int(before)

    return delta


def _gibbs_choose_shift_target(
    phi: Phi,
    I: nx.Graph,
    H_adj: Dict[NodeH, Set[NodeH]],
    i: NodeI,
    u: NodeH,
    candidate_jv: List[Tuple[NodeI, NodeH]],
    T: float,
) -> Tuple[NodeI, NodeH]:
    """The super vertex receiving the shift, drawn from the Gibbs distribution of the local deltas at T(t)."""
    unique_js = list({j for j, v in candidate_jv})

    if T <= 0 or len(unique_js) == 1:
        deltas = [_local_shift_delta(phi, I, H_adj, i, j, u) for j in unique_js]
        best = max(range(len(unique_js)), key=lambda idx: deltas[idx])
        chosen_j = unique_js[best]
    else:
        deltas = [_local_shift_delta(phi, I, H_adj, i, j, u) for j in unique_js]
        scaled = [d / T for d in deltas]
        m = max(scaled)
        weights = [math.exp(s - m) for s in scaled]
        chosen_j = random.choices(unique_js, weights=weights, k=1)[0]

    matches = [(j, v) for j, v in candidate_jv if j == chosen_j]
    return random.choice(matches)


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
    gibbs: bool = False,
) -> Tuple[Phi, int]:
    """Algorithm 1 of Sugie et al. (2018), the annealing search, in pure Python."""
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

            if gibbs:
                j, v = _gibbs_choose_shift_target(phi, I, H_adj, i, u, candidate_jv, T)
            else:
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

            candidates_j = list(adjacent_j)
            if gibbs:
                j = _gibbs_choose_swap_partner(phi, I, H_adj, ie, candidates_j, T)
            else:
                # Sugie's uniform choice
                j = random.choice(candidates_j)

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
# The annealing loop in C++ (pssa_core), with the set-up in Python
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    import pssa_core as _pssa_core_ext
    _HAVE_PSSA_CORE = True
except ImportError:  # pragma: no cover - extension not built yet
    _pssa_core_ext = None
    _HAVE_PSSA_CORE = False


def _pssa_cpp(
    I: nx.Graph,
    H: nx.Graph,
    gp: Dict[int, List[NodeH]],
    schedule: "KingsGraphSchedule",
    weighted: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    deadline: Optional[float] = None,
    gibbs: bool = False,
) -> Tuple[Phi, int]:
    """Algorithm 1 with the annealing loop in C++ (pssa_core). Returns (phi, eemb)."""
    if not _HAVE_PSSA_CORE:
        raise RuntimeError(
            "pssa_core extension is not built. Build it with build_pssa_core.sh "
            "(see that script for the exact g++ command), or fall back to the "
            "pure-Python pssa() function."
        )

    L = schedule.L
    nodes_I = sorted(I.nodes())
    edges_I = list(I.edges())
    m_I = len(edges_I)

    H_adj = {u: list(H.neighbors(u)) for u in H.nodes()}
    gp_lookup: Dict[int, int] = {}
    for gk, path in gp.items():
        for hw in path:
            gp_lookup[hw] = gk

    phi0 = initial_placement(I, gp, L)
    inv0 = invert(phi0)
    e0 = eemb(phi0, I, H, inv0)
    if e0 == m_I:
        return phi0, e0

    tmax = schedule.tmax
    log_every = max(1, tmax // 10)

    phi_best, eemb_best = _pssa_core_ext.run_pssa_core(
        edges_I, nodes_I, H_adj, gp_lookup, phi0,
        tmax, schedule.T0, schedule.alpha,
        schedule.ps0, schedule.ps_end, schedule.pa0, schedule.pa_end,
        weighted, gibbs,
        -1 if seed is None else seed,
        -1.0 if deadline is None else deadline,
        verbose, log_every,
    )
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
    """Algorithm 2 of Sugie et al. (2021), the terminal search."""
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
# The improved PSSA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImprovedPSSA:
    """The improved PSSA on KG(L), the annealing and then the terminal search."""

    def __init__(
        self,
        L: int = 8,
        tmax: Optional[int] = None,
        weighted: bool = False,
        gibbs: bool = False,
        T0: float = 6.0315,
        Thalf: float = 3.3435,
        pa_end: float = 0.487,
        seed: Optional[int] = None,
        verbose: bool = False,
        hardware_graph: Optional[nx.Graph] = None,
    ):
        self.L = L
        self.weighted = weighted
        self.gibbs = gibbs
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

        phi_pssa, _ = _pssa_cpp(
            I, self.H, self.gp, self.schedule,
            weighted=self.weighted, seed=self.seed, verbose=self.verbose,
            deadline=deadline, gibbs=self.gibbs,
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
    gibbs: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    hardware_graph: Optional[nx.Graph] = None,
) -> Tuple[bool, Phi]:
    """One-line embed interface."""
    algo = ImprovedPSSA(L=L, tmax=tmax, weighted=weighted, gibbs=gibbs,
                         seed=seed, verbose=verbose, hardware_graph=hardware_graph)
    result = algo.run(I)
    return result.success, result.phi
