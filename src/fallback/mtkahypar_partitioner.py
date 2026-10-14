"""
Mt-KaHyPar for the fallback method (Section 4.3.1 of the report). The graph G becomes
a hypergraph H with one net per vertex v, its closed neighbourhood N[v], so a net is
cut exactly when v is a border node and cut-net(H) = T, the row-net model of
Catalyurek and Aykanat (1999). The objective is cut-net, not km1, which counts a
border node touching three clusters twice. On Windows Mt-KaHyPar runs inside WSL.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

Node = Any
Labels = Dict[Node, int]


INSTALL = """\
Mt-KaHyPar cannot be built on Windows/MSVC (upstream refuses the compiler).
To enable this module:

    1. In an ADMINISTRATOR PowerShell:      wsl --install -d Ubuntu
       (this needs a reboot; it is the only step that does)
    2. Then, inside the Ubuntu shell:
           sudo apt update
           sudo apt install -y python3-pip build-essential cmake libtbb-dev libboost-all-dev
           pip3 install --break-system-packages mtkahypar
    3. Check it from Windows:
           python src/fallback/mtkahypar_partitioner.py

Without it, the partitioning falls back to GGP + T-FM (method="auto" or "csr").
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The net model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def neighbourhood_nets(G: "nx.Graph") -> Tuple[List[Node], List[List[int]]]:
    """(nodes, nets), one net per vertex, its closed neighbourhood."""
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    nets = [
        sorted({index[v]} | {index[w] for w in G.neighbors(v) if w != v})
        for v in nodes
    ]
    return nodes, nets


def edge_nets(G: "nx.Graph") -> Tuple[List[Node], List[List[int]]]:
    """(nodes, nets), one 2-pin net per edge, so that cut-net is the cut (for the cut against T comparison)."""
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    nets = [[index[u], index[v]] for u, v in G.edges() if u != v]
    return nodes, nets


def cut_net(nets: Sequence[Sequence[int]], lab: Sequence[int]) -> int:
    """The number of nets that span more than one cluster."""
    total = 0
    for pins in nets:
        first = lab[pins[0]]
        for p in pins:
            if lab[p] != first:
                total += 1
                break
    return total


def km1(nets: Sequence[Sequence[int]], lab: Sequence[int]) -> int:
    """km1, for every net the clusters it touches minus one."""
    return sum(len({lab[p] for p in pins}) - 1 for pins in nets)


def border_count(G: "nx.Graph", labels: Labels) -> int:
    """T, counted on G."""
    return sum(1 for v in G if any(labels[w] != labels[v] for w in G.neighbors(v)))


def verify_cutnet_equals_T(G: "nx.Graph", labels: Labels) -> Tuple[bool, int, int]:
    """(ok, cut_net(H), T), the check that cut-net equals T."""
    nodes, nets = neighbourhood_nets(G)
    index = {v: i for i, v in enumerate(nodes)}
    lab = [0] * len(nodes)
    for v, c in labels.items():
        lab[index[v]] = c
    cn = cut_net(nets, lab)
    t = border_count(G, labels)
    return cn == t, cn, t


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hMETIS format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def to_hmetis(G: "nx.Graph", net_model: str = "closed_neighbourhood") -> Tuple[List[Node], str]:
    """(nodes, text), the hypergraph in hMETIS format."""
    if net_model == "edge":
        nodes, nets = edge_nets(G)
    elif net_model == "closed_neighbourhood":
        nodes, nets = neighbourhood_nets(G)
    else:
        raise ValueError(f"unknown net_model {net_model!r}")
    out = [f"{len(nets)} {len(nodes)}"]
    out.extend(" ".join(str(p + 1) for p in pins) for pins in nets)
    return nodes, "\n".join(out) + "\n"


def read_partition_file(path: str, n: int) -> List[int]:
    """One cluster id per line, in vertex order."""
    with open(path) as fh:
        lab = [int(line) for line in fh if line.strip() != ""]
    if len(lab) != n:
        raise ValueError(f"partition file has {len(lab)} entries, expected {n}")
    return lab


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Backends
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_WSL_DRIVER = r'''
import sys
hgr, k, eps, threads, out, preset = (
    sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
    int(sys.argv[4]), sys.argv[5], sys.argv[6])
import mtkahypar

# initialize() returns the Initializer that builds contexts and hypergraphs.
init = mtkahypar.initialize(threads, False)
ctx = init.context_from_preset(getattr(mtkahypar.PresetType, preset))
# cut-net, not km1: the objective that equals T
ctx.set_partitioning_parameters(k, eps, mtkahypar.Objective.CUT)
ctx.logging = False
hg = init.hypergraph_from_file(hgr, ctx, mtkahypar.FileFormat.HMETIS)
phg = hg.partition(ctx)
with open(out, "w") as fh:
    fh.write("\n".join(str(b) for b in phg.get_partition()) + "\n")
sys.stderr.write(
    f"cutnet={phg.cut()} km1={phg.km1()} imbalance={phg.imbalance(ctx):.4f}\n")
'''


def _win_to_wsl(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _have_wsl() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        r = subprocess.run(["wsl", "-e", "python3", "-c", "import mtkahypar"],
                           capture_output=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


def _have_native() -> bool:
    try:
        import mtkahypar  # noqa: F401
        return True
    except Exception:
        return False


_STATUS_CACHE: Optional[Dict[str, Any]] = None


def backend_status(recheck: bool = False) -> Dict[str, Any]:
    """Which Mt-KaHyPar is available, native or in WSL."""
    global _STATUS_CACHE
    if _STATUS_CACHE is not None and not recheck:
        return _STATUS_CACHE
    native = _have_native()
    wsl = False if native else _have_wsl()  # only probe WSL if the native one fails
    _STATUS_CACHE = {
        "native_mtkahypar": native,
        "wsl_mtkahypar": wsl,
        "usable": native or wsl,
        "hint": None if (native or wsl) else INSTALL,
    }
    return _STATUS_CACHE


def partition_hypergraph(
    G: "nx.Graph",
    k: int,
    eps: float = 0.03,
    threads: Optional[int] = None,
    verify: bool = True,
    timeout_s: float = 3600.0,
    preset: str = "QUALITY",
    net_model: str = "closed_neighbourhood",
) -> Labels:
    """Partition G into k clusters with Mt-KaHyPar, minimizing cut-net on H, ie. T."""
    if threads is None:
        threads = os.cpu_count() or 1

    st = backend_status()
    if not st["usable"]:
        raise RuntimeError("No Mt-KaHyPar backend available.\n\n" + INSTALL)

    nodes, text = to_hmetis(G, net_model=net_model)
    n = len(nodes)

    tmp = tempfile.mkdtemp(prefix="kahypar_")
    try:
        hgr = os.path.join(tmp, "graph.hgr")
        outp = os.path.join(tmp, "graph.part")
        drv = os.path.join(tmp, "driver.py")
        with open(hgr, "w") as fh:
            fh.write(text)
        with open(drv, "w") as fh:
            fh.write(_WSL_DRIVER)

        if st["native_mtkahypar"]:
            import runpy
            import sys
            old = sys.argv
            sys.argv = [drv, hgr, str(k), str(eps), str(threads), outp, preset]
            try:
                runpy.run_path(drv, run_name="__main__")
            finally:
                sys.argv = old
        else:
            cmd = ["wsl", "-e", "python3", _win_to_wsl(drv), _win_to_wsl(hgr),
                   str(k), str(eps), str(threads), _win_to_wsl(outp), preset]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if r.returncode != 0:
                raise RuntimeError(
                    f"Mt-KaHyPar failed (exit {r.returncode}).\n"
                    f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
                )
            last_stats = r.stderr.strip()

        lab = read_partition_file(outp, n)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    labels: Labels = {nodes[i]: lab[i] for i in range(n)}

    # cut-net equals T only for the closed-neighbourhood nets
    if verify and net_model == "closed_neighbourhood":
        ok, cn, t = verify_cutnet_equals_T(G, labels)
        if not ok:
            raise AssertionError(
                f"cut-net({cn}) != T({t}). The partitioner is not optimising T "
                f"-- almost certainly the objective came back as km1 rather "
                f"than cut. See the module docstring."
            )
    return labels


TWO_LEVEL_FROM = 100_000   # from this many vertices, Mt-KaHyPar runs in two levels
PART_SIZE = 2_500          # vertices per top-level part


def _two_level(G, K1, target, eps, threads):
    """Mt-KaHyPar in two levels, K1 parts first, then each part into clusters of about target vertices."""
    top = partition_hypergraph(G, K1, eps=0.03, threads=threads)
    parts = {}
    for v, c in top.items():
        parts.setdefault(c, []).append(v)
    labels, nxt = {}, 0
    for nodes in parts.values():
        k2 = max(1, -(-len(nodes) // target))
        sub = partition_hypergraph(G.subgraph(nodes), k2, eps=eps, threads=threads) if k2 > 1 else {v: 0 for v in nodes}
        for v, b in sub.items():
            labels[v] = nxt + b
        nxt += k2
    return labels


def partition_graph_kahypar(
    G: "nx.Graph",
    max_cluster_size: int,
    target_cluster_size: Optional[int] = None,
    headroom: Optional[int] = None,
    eps: float = 0.03,
    threads: Optional[int] = None,
    levels: Optional[int] = None,
):
    """
    A PartitionResult from Mt-KaHyPar QUALITY, in one call below TWO_LEVEL_FROM vertices and
    in two levels from it (levels forces 1 or 2), with the hard cap enforced and then T-FM.
    """
    import border_node_partitioning as gp

    n = G.number_of_nodes()
    s_target = target_cluster_size
    if s_target is None:
        hr = headroom if headroom is not None else (1 if max_cluster_size <= 20 else 2)
        s_target = max(1, max_cluster_size - hr)
    if levels is None:
        levels = 1 if n < TWO_LEVEL_FROM else 2
    if levels == 1:
        labels = partition_hypergraph(G, max(1, -(-n // s_target)), eps=eps, threads=threads)
    else:
        labels = _two_level(G, max(2, round(n / PART_SIZE)), s_target, eps, threads)

    labels, ok = gp.enforce_bounds(G, labels, None, max_cluster_size, seed=0)
    if not ok:
        raise RuntimeError(
            f"could not bring every block under max_cluster_size={max_cluster_size}"
        )
    k_after = len(set(labels.values()))
    if k_after > 1:
        labels = gp.fm_refine(G, labels, k_after, max_passes=40, seed=0,
                              max_size=max_cluster_size)
    return gp._build_result(G, labels, "mtkahypar[cut-net]")


if __name__ == "__main__":
    st = backend_status()
    print("backend status:", {k: v for k, v in st.items() if k != "hint"})
    if not st["usable"]:
        print("\n" + INSTALL)
