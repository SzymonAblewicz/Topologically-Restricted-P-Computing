"""
The graphs of the report, as (G, k, labels). Random cubic graphs, the staple benchmark
graphs and the main family of the report, bipartite stochastic block models, random
bipartite graphs made of communities that are densely connected inside and only
sparsely connected to each other, and the other families by name.
"""

import random
import networkx as nx


def build_test_graph(
    community_sizes,   # list of (a_size, b_size, degree) tuples, one per community
    noise=0.0006,       # cross-community edge probability
    redundancy=2,  # repair edges for components
    seed=None,
):
    """A bipartite stochastic block model, community_sizes listing (a, b, mean degree) per community, connected and bipartite as a whole."""
    rng = random.Random(seed)
    G = nx.Graph()
    labels = {}
    node_class = {}  # 0 = side a, 1 = side b, so every edge joins a to b
    community_nodes = []    # community_nodes[cid] = set of all nodes in that community
    theta = {}  # per-vertex propensity
    next_id = 0

    for cid, (na, nb, degree) in enumerate(community_sizes):
        a_nodes = list(range(next_id, next_id + na))
        b_nodes = list(range(next_id + na, next_id + na + nb))
        next_id += na + nb

        G.add_nodes_from(a_nodes)
        G.add_nodes_from(b_nodes)
        for n in a_nodes:
            node_class[n] = 0
            labels[n] = cid
        for n in b_nodes:
            node_class[n] = 1
            labels[n] = cid
        for n in a_nodes + b_nodes:
            theta[n] = rng.uniform(0.7, 1.3)

        # weighted sampling without replacement (Efraimidis-Spirakis)
        size = max(na, nb)
        target_edges = min(round(degree * size), na * nb)
        pairs = [(u, v) for u in a_nodes for v in b_nodes]
        keyed = [(u, v, rng.random() ** (1.0 / (theta[u] * theta[v]))) for u, v in pairs]
        keyed.sort(key=lambda x: -x[2])
        G.add_edges_from((u, v) for u, v, _ in keyed[:target_edges])

        # an isolated vertex joins its community's best-connected vertex on the other side
        for side_nodes, other_side in ((a_nodes, b_nodes), (b_nodes, a_nodes)):
            for n in side_nodes:
                if G.degree(n) == 0:
                    best = max(other_side, key=lambda m: G.degree(m))
                    G.add_edge(n, best)

        community_nodes.append(set(a_nodes) | set(b_nodes))

    # a few cross-community edges, a to b only
    for i in range(len(community_nodes)):
        for j in range(len(community_nodes)):
            if i == j:
                continue
            a_i = [n for n in community_nodes[i] if node_class[n] == 0]
            b_j = [n for n in community_nodes[j] if node_class[n] == 1]
            for u in a_i:
                for v in b_j:
                    if rng.random() < noise:
                        G.add_edge(u, v)

    # repair fragmented communities and connect the whole graph
    def connect_component(comp, target_nodes, redundancy=redundancy):
        """Join comp to the rest of the graph by redundancy edges, so no single edge splits off a large part."""
        k = redundancy if len(comp) > 3 else 1
        comp_list = list(comp)
        for _ in range(k):
            u = rng.choice(comp_list)
            v = rng.choice([n for n in target_nodes if node_class[n] != node_class[u]])
            G.add_edge(u, v)

    for nodes_i in community_nodes:
        comps = list(nx.connected_components(G.subgraph(nodes_i)))
        if len(comps) > 1:
            main = max(comps, key=len)
            for comp in comps:
                if comp is main:
                    continue
                connect_component(comp, list(main))

    components = list(nx.connected_components(G))
    if len(components) > 1:
        merged = set(components[0])
        for comp in components[1:]:
            connect_component(comp, list(merged))
            merged |= comp

    # every community gets at least 2 edges to the rest of the graph
    for nodes_i in community_nodes:
        external_edges = [(u, v) for u, v in G.edges()
                           if (u in nodes_i) != (v in nodes_i)]
        if len(external_edges) < 2:
            outside_nodes = [n for n in G.nodes() if n not in nodes_i]
            u = rng.choice(list(nodes_i))
            v = rng.choice([n for n in outside_nodes if node_class[n] != node_class[u]])
            G.add_edge(u, v)

    # remove any bridge that would split off more than 6 vertices
    for _ in range(50):
        bad = None
        for u, v in nx.bridges(G):
            H = G.copy(); H.remove_edge(u, v)
            comps = list(nx.connected_components(H))
            small_side = min(comps, key=len)
            if len(small_side) > 6:
                bad = (u, v, small_side)
                break
        if bad is None:
            break
        u, v, small_side = bad
        other_side = [n for n in G.nodes() if n not in small_side]
        x = rng.choice(list(small_side))
        y = rng.choice([n for n in other_side if node_class[n] != node_class[x]])
        G.add_edge(x, y)

    # join any vertex still isolated
    for n in list(G.nodes()):
        if G.degree(n) == 0:
            peers = [m for m in community_nodes[labels[n]]
                     if m != n and node_class[m] != node_class[n]]
            if peers:
                G.add_edge(n, rng.choice(peers))

    assert sum(1 for _, d in G.degree() if d == 0) == 0
    assert nx.is_bipartite(G)
    assert nx.is_connected(G)

    return G, len(community_sizes), labels


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Bipartite stochastic block models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_default_demo_graph(seed: int = 42):
    """The 338-vertex block model, 12 communities in three degree tiers."""
    community_sizes = [
        (8, 8, 2.0), (8, 8, 2.0), (9, 9, 2.0), (10, 10, 2.0),
        (14, 14, 1.5), (14, 14, 1.5), (15, 15, 1.5), (16, 16, 1.5),
        (18, 18, 1.3), (18, 18, 1.3), (19, 19, 1.3), (20, 20, 1.3),
    ]
    return build_test_graph(community_sizes, seed=seed)


def _stratified_unit(d1, d2, d3):
    """The 12 communities of the 338-vertex graph, repeated for the larger ones."""
    return [
        (8, 8, d1), (8, 8, d1), (9, 9, d1), (10, 10, d1),
        (14, 14, d2), (14, 14, d2), (15, 15, d2), (16, 16, d2),
        (18, 18, d3), (18, 18, d3), (19, 19, d3), (20, 20, d3),
    ]


def build_demo_graph_338_v2(seed: int = 42):
    """338 vertices, 12 communities, denser."""
    community_sizes = _stratified_unit(3.0, 2.2, 1.8)
    return build_test_graph(community_sizes, noise=0.0002, seed=seed)


def build_demo_graph_1000(seed: int = 42):
    """1,014 vertices, 36 communities, three blocks of the 338-vertex graph."""
    community_sizes = _stratified_unit(3.0, 2.2, 1.8) * 3
    return build_test_graph(community_sizes, noise=0.0006, seed=seed)


def build_demo_graph_1000_v2(seed: int = 42):
    """1,014 vertices, 36 communities, denser and with fewer cross edges."""
    community_sizes = _stratified_unit(4.5, 3.2, 2.4) * 3
    return build_test_graph(community_sizes, noise=0.0001, seed=seed)


def build_demo_graph_5000(seed: int = 42):
    """5,070 vertices, 180 communities."""
    community_sizes = _stratified_unit(4.5, 3.2, 2.4) * 15
    return build_test_graph(community_sizes, noise=0.00005, seed=seed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Graph families without planted communities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _finalize(G: "nx.Graph"):
    """Integer labels, a connectivity check, and (G, None, None)."""
    G = nx.convert_node_labels_to_integers(G)
    if not nx.is_connected(G):
        raise RuntimeError(
            f"generated graph is disconnected ({nx.number_connected_components(G)} "
            f"components); Mt-KaHyPar requires a connected graph "
            f"(partition_graph(..., method='csr') handles disconnected input)."
        )
    return G, None, None


def build_random_regular(n: int = 1000, d: int = 3, seed: int = 42,
                          max_tries: int = 20):
    """A random d-regular graph on exactly n vertices, the next seed tried until it is connected."""
    if d >= n:
        raise ValueError(f"need d < n, got d={d}, n={n}.")
    if (n * d) % 2 != 0:
        raise ValueError(
            f"n*d must be even for a d-regular graph, got n={n}, d={d} "
            f"(n*d={n*d}). Bump n or d by one."
        )
    for t in range(max_tries):
        G = nx.random_regular_graph(d, n, seed=None if seed is None else seed + t)
        if nx.is_connected(G):
            return _finalize(G)
    raise RuntimeError(
        f"could not draw a connected {d}-regular graph on {n} nodes in "
        f"{max_tries} tries (seeds {seed}..{seed + max_tries - 1})."
    )


def build_random_cubic(n: int = 1000, seed: int = 42):
    """A random cubic graph on n vertices, n even."""
    return build_random_regular(n=n, d=3, seed=seed)


def build_barabasi_albert(n: int = 1000, m: int = 2, seed: int = 42):
    """A Barabasi-Albert graph, each new vertex bringing m edges."""
    if not 1 <= m < n:
        raise ValueError(f"need 1 <= m < n, got m={m}, n={n}.")
    return _finalize(nx.barabasi_albert_graph(n, m, seed=seed))


def build_erdos_renyi(n: int = 1000, avg_degree: float = 3.0, seed: int = 42):
    """An Erdos-Renyi G(n, m) graph with mean degree avg_degree, reduced to its giant component."""
    m = round(avg_degree * n / 2)
    G = nx.gnm_random_graph(n, m, seed=seed)
    giant = max(nx.connected_components(G), key=len)
    return _finalize(G.subgraph(giant).copy())


def build_grid_2d(rows: int = 32, cols: int = None):
    """The rows x cols square lattice."""
    return _finalize(nx.grid_2d_graph(rows, cols if cols is not None else rows))


def build_grid_3d(side: int = 10):
    """The side x side x side cubic lattice."""
    return _finalize(nx.grid_graph(dim=[side, side, side]))


def build_hex_lattice(m: int = 25, n: int = 25):
    """The honeycomb lattice of m x n hexagons."""
    return _finalize(nx.hexagonal_lattice_graph(m, n))


# by name, for a sweep over families
BENCHMARK_BUILDERS = {
    "random_cubic":     build_random_cubic,      # n, seed
    "random_regular":   build_random_regular,    # n, d, seed
    "barabasi_albert":  build_barabasi_albert,   # n, m, seed
    "erdos_renyi":      build_erdos_renyi,       # n, avg_degree, seed
    "grid_2d":          build_grid_2d,           # rows, cols
    "grid_3d":          build_grid_3d,           # side
    "hex_lattice":      build_hex_lattice,       # m, n
}


def build_benchmark_graph(family: str, **kwargs):
    """One of BENCHMARK_BUILDERS by name, eg. build_benchmark_graph("random_cubic", n=2000, seed=1)."""
    if family not in BENCHMARK_BUILDERS:
        raise ValueError(
            f"unknown family {family!r}; choose from {sorted(BENCHMARK_BUILDERS)}."
        )
    return BENCHMARK_BUILDERS[family](**kwargs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Drawing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_test_graph(
    G: "nx.Graph", seed: int = 42, ax=None, show: bool = True,
    title: str = None,
):
    """Draw a graph as built."""
    import matplotlib.pyplot as plt

    should_show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        should_show = True

    pos = nx.spring_layout(G, seed=seed, k=0.01)

    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.6,
                            edge_color="#999999", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color="#4C72B0", node_size=40,
                            edgecolors="white", linewidths=0.4, ax=ax)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title or f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                 fontsize=12)

    if show and should_show:
        plt.tight_layout()
        plt.savefig("test_graph.png", dpi=300)
        plt.show()
    return ax


def draw_circular_graph(
    G: "nx.Graph", ax=None, show: bool = True, title: str = None,
    randomize_order: bool = True, seed: int = 42,
):
    """Draw a graph with its vertices round a circle and its edges as chords."""
    import math
    import random as _random
    import matplotlib.pyplot as plt

    should_show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        should_show = True

    order = list(G.nodes())
    if randomize_order:
        _random.Random(seed).shuffle(order)
    n = len(order)
    pos = {node: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
           for i, node in enumerate(order)}

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.5,
                            edge_color="#999999", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=order, node_color="#4C72B0",
                            node_size=25, edgecolors="white", linewidths=0.3, ax=ax)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title or f"Circular layout -- {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
                 fontsize=12)

    if show and should_show:
        plt.tight_layout()
        plt.show()
    return ax


if __name__ == "__main__":
    G, k_true, labels = build_demo_graph_1000(seed=42)
    degs = [d for _, d in G.degree()]
    bridges = list(nx.bridges(G))
    big_cuts = 0
    for u, v in bridges:
        H = G.copy(); H.remove_edge(u, v)
        if min(len(c) for c in nx.connected_components(H)) > 6:
            big_cuts += 1

    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{k_true} communities, connected={nx.is_connected(G)}, "
          f"min degree={min(degs)}")
    print(f"Degree range: {min(degs)}-{max(degs)}, mean: {sum(degs)/len(degs):.2f}")
    print(f"Bridges: {len(bridges)}, of which large-split (>6 nodes): {big_cuts}")

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    draw_test_graph(G, ax=ax1, show=False, title="Test graph: spring layout")
    draw_circular_graph(G, ax=ax2, show=False, title="Test graph: circular layout")
    plt.tight_layout()
    plt.savefig("test_graph.png", dpi=300)
    plt.show()
