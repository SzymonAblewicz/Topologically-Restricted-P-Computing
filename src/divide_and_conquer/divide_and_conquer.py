"""
The Divide and Conquer method (Section 3 of the report), in three steps. Min-cut
partitioning, the embedding of every cluster with PSSA wBC, and the placing and
routing of the clusters into a full minor-embedding of the network, with the
corridors widened until every cut edge routes.

    python divide_and_conquer.py --graph cubic1000
    python divide_and_conquer.py --graph cubic1000 --sa-seed 2 --rejoin
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "section3_divide_and_conquer")
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "divide_and_conquer", "fallback", "benchmark_graphs")]

import cluster_embedding as PP                                     # noqa: E402
from min_cut_partitioning import partition_from_labels, partition_graph_auto  # noqa: E402

S = 30                  # largest cluster size

# The settings of placing and routing the clusters
JOIN = dict(
    pad=True,  # a margin of the corridor width round the layout
    n_moves=60_000,  # floorplanner moves per restart
    n_restarts=4,  # floorplanner restarts
    n_reserve=0,  # free hardware vertices each border node reserves next to its super vertex
    max_rounds=60,  # PathFinder rounds
)
MORE_WIDTH = 16  # corridor widths tried, from corridor_width() up to this much more, 2 apart

FIELDS = ["graph", "seed", "n", "m", "s", "k", "cut", "T", "t_partition", "clusters_pssa",
          "clusters_fallback", "t_clusters", "sa_seed", "tries", "gap", "max_gap", "width",
          "t_join", "L", "valid", "eemb", "unrouted", "qubits", "max_chain", "L_complete",
          "area_vs_complete"]


def build_graph(name: str, seed: int) -> nx.Graph:
    """The graphs of the report, relabelled 0..n-1."""
    from graphs import (build_default_demo_graph, build_demo_graph_1000,
                                   build_random_cubic, build_test_graph, _stratified_unit)
    if name == "demo338":
        G = build_default_demo_graph(seed=42)[0]
    elif name.startswith("cubic"):
        G = build_random_cubic(n=int(name[5:]), seed=seed)[0]
    elif name == "tiled3":                          # the 1014-vertex graph, exactly
        G = build_demo_graph_1000(seed=42)[0]
    elif name.startswith("tiled"):
        # the test-graph unit u times, with the noise scaled so every vertex
        # keeps the same expected number of cross-community edges
        u = int(name[5:])
        G = build_test_graph(_stratified_unit(3.0, 2.2, 1.8) * u, noise=0.0006 * 3 / u, seed=42)[0]
    else:
        raise ValueError(f"unknown graph {name!r}")
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def corridor_width(part) -> int:
    """The first corridor width to try, about cut / sqrt(k) - 14 and at least 3, then up by 2 until every cut edge routes."""
    return max(3, round(part.cut_size / math.sqrt(part.k)) - 14)


def run(graph: str, s: int = S, seed: int = 1, sa_seed: int = 0, tries: int = 1,
        gap: int = None, max_gap: int = None, rejoin: bool = False,
        out: str = RES, dpi: int = 300):
    tag = f"{graph}_s{s}_seed{seed}"
    folder = os.path.join(out, tag)
    os.makedirs(folder, exist_ok=True)

    def say(msg):
        print(f"[{tag}] {msg}", flush=True)

    graph_txt = os.path.join(folder, "graph.txt")
    if rejoin:
        G = nx.read_edgelist(graph_txt, nodetype=int)
    else:
        G = build_graph(graph, seed)
        nx.write_edgelist(G, graph_txt, data=False)
    n, m = G.number_of_nodes(), G.number_of_edges()
    row = dict(graph=graph, seed=seed, n=n, m=m, s=s, L_complete=n - 1)
    say(f"graph n={n} m={m} connected={nx.is_connected(G)}")

    # 1. partition (read back from graph.txt, as the node order matters)
    labels_json = os.path.join(folder, "labels.json")
    if not rejoin:
        t = time.perf_counter()
        p = partition_graph_auto(nx.read_edgelist(graph_txt, nodetype=int),
                                 min_cluster_size=1, max_cluster_size=s, seed=seed)
        with open(labels_json, "w") as fh:
            json.dump({"labels": {str(v): int(c) for v, c in p.labels.items()},
                       "method": str(p.method), "seconds": round(time.perf_counter() - t, 1)}, fh)
    with open(labels_json) as fh:
        info = json.load(fh)
    part = partition_from_labels(G, {int(v): c for v, c in info["labels"].items()})
    T = sum(len(b) for b in part.border_nodes.values())
    row.update(k=part.k, cut=part.cut_size, T=T, t_partition=info["seconds"])
    say(f"partition: k={part.k} cut={part.cut_size} T={T} ({info['method']}, {info['seconds']} s)")

    # 2. every cluster by PSSA wBC, in parallel
    clusters_pkl = os.path.join(folder, "clusters.pkl")
    if rejoin:
        with open(clusters_pkl, "rb") as fh:
            embs, times = pickle.load(fh)
    else:
        t = time.perf_counter()
        embs = PP.embed_clusters(part, verbose=False)
        times = {"t_clusters_wall": round(time.perf_counter() - t, 1)}
        with open(clusters_pkl, "wb") as fh:
            pickle.dump((embs, times), fh)
    fb = sum(1 for e in embs.values() if e.seed_used == -1)
    row.update(clusters_pssa=len(embs) - fb, clusters_fallback=fb, t_clusters=times.get("t_clusters_wall"))
    say(f"clusters: {len(embs) - fb} by PSSA wBC, {fb} by the complete graph embedding, "
        f"{times.get('t_clusters_wall')} s")

    # 3. placing and routing the clusters, with wider corridors until every cut edge routes
    gap = corridor_width(part) if gap is None else gap
    max_gap = gap + MORE_WIDTH if max_gap is None else max_gap
    row.update(sa_seed=sa_seed, tries=tries, gap=gap, max_gap=max_gap)
    t0 = time.perf_counter()
    for width in range(gap, max_gap + 1, 2):
        t = time.perf_counter()
        res = PP.compact_embedding(G, part, embs, gap=width, auto_gap=False, n_sa_tries=tries,
                                   sa_seed=sa_seed, verbose=False, **JOIN)
        say(f"  corridor width {width}: L={res.L}, {len(res.unrouted)} unrouted, "
            f"{time.perf_counter() - t:.0f} s")
        if not res.unrouted:
            break
    chains = [len(c) for c in res.phi.values()]
    row.update(width=width, t_join=round(time.perf_counter() - t0, 1), L=res.L, valid=res.valid,
               eemb=res.eemb, unrouted=len(res.unrouted), qubits=sum(chains), max_chain=max(chains),
               area_vs_complete=round((res.L / (n - 1)) ** 2, 4))
    say(f"joined: L={res.L} valid={res.valid} {res.eemb}/{m} edges, {len(res.unrouted)} unrouted, "
        f"{row['t_join']} s")

    save(G, part, embs, res, row, os.path.join(folder, f"sa{sa_seed}_tries{tries}_gap{gap}-{max_gap}"), out, dpi)
    return res


def save(G, part, embs, res, row, folder, out, dpi):
    """The result (run.pkl), a row in runs.csv, and the two drawings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "run.pkl"), "wb") as fh:
        pickle.dump({"labels": part.labels, "embeddings": embs,
                     "result": {k: getattr(res, k) for k in ("phi", "L", "placements", "cluster_of",
                                                             "routes", "unrouted", "valid", "eemb",
                                                             "m_total", "orientations")}}, fh)

    path = os.path.join(out, "runs.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

    fig, axes = plt.subplots(1, 2, figsize=(26, 13), constrained_layout=True)
    PP.draw_exploded_clusters(part, embs, ax=axes[0], show=False)
    PP.draw_compact_embedding(G, part, res, ax=axes[1], show=False)
    fig.savefig(os.path.join(folder, "stages.png"), dpi=dpi // 2, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 15))
    PP.draw_compact_embedding(G, part, res, ax=ax, show=False)
    fig.savefig(os.path.join(folder, "embedding.png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="ImprovedDnC on one graph (see the module docstring).")
    ap.add_argument("--graph", required=True, help="cubic<n>, demo338 or tiled<u>")
    ap.add_argument("--s", type=int, default=S, help="largest cluster size")
    ap.add_argument("--seed", type=int, default=1, help="seed of the graph and the partition search")
    ap.add_argument("--sa-seed", type=int, default=0, help="seed of the floorplan annealer")
    ap.add_argument("--tries", type=int, default=1, help="annealer seeds per corridor width (the best is kept)")
    ap.add_argument("--gap", type=int, default=None, help="first corridor width (default: from the cut)")
    ap.add_argument("--max-gap", type=int, default=None, help=f"last corridor width (default: first + {MORE_WIDTH})")
    ap.add_argument("--rejoin", action="store_true", help="reuse the saved graph, partition and clusters")
    ap.add_argument("--out", default=RES, help="where the runs are saved")
    ap.add_argument("--dpi", type=int, default=300, help="of embedding.png (stages.png gets half)")
    a = ap.parse_args()
    run(a.graph, s=a.s, seed=a.seed, sa_seed=a.sa_seed, tries=a.tries, gap=a.gap,
        max_gap=a.max_gap, rejoin=a.rejoin, out=a.out, dpi=a.dpi)


if __name__ == "__main__":
    main()
