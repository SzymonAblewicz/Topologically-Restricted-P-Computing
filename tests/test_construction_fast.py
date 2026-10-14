"""
Tests that the fast construction (construction_fast.py, kings_lattice.py) gives exactly
what the full construction (construction.py) and the full King's graph give.

    python test_construction_fast.py               # a few seconds
    python test_construction_fast.py --oracle 1000 # also the whole embedding at n = 1000, both ways
"""

from __future__ import annotations

import os
import random
import sys
import time

import networkx as nx
import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "divide_and_conquer", "fallback", "benchmark_graphs")]

from kings_lattice import (
    KingsLattice, Occupancy, rc_of, id_of, validate_embedding,
)
from pssa_wbc import (
    build_hardware_graph, eemb, invert, is_valid_embedding,
    oneD2twoD, twoD2oneD,
)
from construction import _contacts, _plaza_stripes
from construction_fast import contacts_fast, stripe_cells, stripe_cols

_PASS: list = []


def check(name, cond, detail=""):
    _PASS.append((name, bool(cond)))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not cond:
        raise AssertionError(name + " " + detail)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. The lattice as arithmetic is the King's graph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_lattice():
    print("KingsLattice vs build_hardware_graph")
    for L in (1, 2, 3, 5, 12, 40):
        H = build_hardware_graph(L)
        K = KingsLattice(L)
        n = L * L
        same_nodes = K.number_of_nodes() == H.number_of_nodes()
        same_edges = K.number_of_edges() == H.number_of_edges()
        same_adj = all(sorted(K.neighbors(u)) == sorted(H.neighbors(u))
                       for u in range(1, n + 1))
        same_deg = all(K.degree(u) == H.degree(u) for u in range(1, n + 1))
        same_has = all(K.has_edge(u, v) == H.has_edge(u, v)
                       for u in range(1, n + 1) for v in range(1, n + 1))
        check(f"L={L:<3d} identical graph",
              same_nodes and same_edges and same_adj and same_deg and same_has,
              f"({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)")

    # subgraph must stay usable as an argument to nx.is_connected
    rng = np.random.default_rng(0)
    L = 12
    H, K = build_hardware_graph(L), KingsLattice(L)
    ok = True
    for _ in range(200):
        cells = rng.choice(np.arange(1, L * L + 1),
                           size=int(rng.integers(1, 25)), replace=False)
        ok &= (nx.is_connected(H.subgraph(list(cells)))
               == nx.is_connected(K.subgraph(list(cells))))
    check("subgraph/is_connected agree", ok, "(200 random cell sets)")

    # enumerating edges is a mistake, and should say so
    raised = False
    try:
        KingsLattice(5).edges()
    except NotImplementedError:
        raised = True
    check("edges() refuses to enumerate", raised)

    # the coordinate helpers match the numbering from 1
    L = 17
    ids = np.arange(1, L * L + 1)
    r, c = rc_of(ids, L)
    ref = np.array([oneD2twoD(int(u), L) for u in ids])
    check("rc_of == oneD2twoD", (r == ref[:, 0]).all() and (c == ref[:, 1]).all())
    check("id_of == twoD2oneD", (id_of(r, c, L) == ids).all())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. The guiding pattern in closed form
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_stripes():
    print("stripe closed form vs _plaza_stripes")
    for S in (3, 6, 7, 12, 41, 60, 77, 120):
        ref = _plaza_stripes(S)
        ok = True
        for j, ch in enumerate(ref):
            got = stripe_cells(j, len(ch) - 1, S)
            ok &= got.shape[0] == len(ch)
            ok &= all((int(got[i, 0]), int(got[i, 1])) == ch[i] for i in range(len(ch)))
        check(f"S={S:<4d} cell-for-cell", ok, f"({len(ref)}x{len(ref[0])} cells)")

    # one cell per row, so two chains can only meet between neighbouring rows
    S = 60
    rows = np.array([[int(x) for x in stripe_cells(j, S - 2, S)[:, 0]]
                     for j in range(S)])
    check("one cell per row per stripe",
          all((rows[j] == np.arange(1, S)).all() for j in range(S)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Where the chains meet, the same as construction.py, ties included
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_contacts():
    print("contacts_fast vs _contacts")
    rng = random.Random(5)
    for S in (8, 13, 30, 41, 77, 120):
        stripes = _plaza_stripes(S)
        same = True
        for _ in range(4):
            pairs = [(rng.randrange(S), rng.randrange(S)) for _ in range(200)]
            pairs = [(a, b) for a, b in pairs if a != b]
            same &= _contacts(stripes, pairs) == contacts_fast(S, pairs)
        check(f"S={S:<4d} random demand sets", same, "(4 x 200 pairs)")

    for S in (8, 13, 30, 41):
        stripes = _plaza_stripes(S)
        allp = [(a, b) for a in range(S) for b in range(a + 1, S)]
        check(f"S={S:<4d} EXHAUSTIVE all-pairs",
              _contacts(stripes, allp) == contacts_fast(S, allp),
              f"({len(allp)} pairs)")

    check("empty demand", contacts_fast(30, []) == {})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Validity, the same verdict as is_valid_embedding + eemb
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_validate():
    print("validate_embedding vs is_valid_embedding + eemb")
    rng = random.Random(11)
    mismatches = 0
    n_valid = 0
    n_total = 0
    for trial in range(600):
        L = rng.randint(4, 10)
        H, K = build_hardware_graph(L), KingsLattice(L)
        free = set(range(1, L * L + 1))
        phi = {}
        for i in range(rng.randint(2, 7)):
            if not free:
                break
            start = rng.choice(sorted(free))
            chain = [start]
            free.discard(start)
            for _ in range(rng.randint(0, 8)):
                cand = sorted({w for u in chain for w in H.neighbors(u) if w in free})
                if not cand:
                    break
                w = rng.choice(cand)
                chain.append(w)
                free.discard(w)
            mode = rng.random()
            if mode < 0.30:                       # fully scrambled order
                rng.shuffle(chain)
            elif mode < 0.60 and len(chain) > 3:  # scrambled head, path tail
                h = len(chain) // 2
                head = chain[:h]
                rng.shuffle(head)
                chain = head + chain[h:]
            elif mode < 0.70 and len(chain) > 2:  # punch a hole -> may disconnect
                chain = chain[:1] + chain[2:]
            phi[i] = chain
        if len(phi) < 2:
            continue
        I = nx.Graph()
        I.add_nodes_from(phi)
        for a in phi:
            for b in phi:
                if a < b and rng.random() < 0.85 and any(
                        H.has_edge(x, y) for x in phi[a] for y in phi[b]):
                    I.add_edge(a, b)
        if rng.random() < 0.3:
            u, v = rng.sample(sorted(phi), 2)
            I.add_edge(u, v)                      # may be uncoverable

        ref = (is_valid_embedding(phi, I, H), eemb(phi, I, H, invert(phi)))
        got = validate_embedding(phi, I, K)
        n_total += 1
        n_valid += ref[0]
        mismatches += (ref != got)
    check("random embeddings agree", mismatches == 0,
          f"({n_total} cases, {n_valid} valid, {mismatches} mismatches)")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. The relabelling of the trim
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_trim_remap():
    """The arithmetic relabelling of the trim to the occupied box."""
    print("trim relabelling")
    rng = np.random.default_rng(2)
    for L, r_min, c_min, L_new in ((40, 3, 5, 30), (17, 0, 0, 12), (64, 11, 2, 50)):
        ids = rng.choice(np.arange(1, L * L + 1), size=min(400, L * L),
                         replace=False)
        r, c = rc_of(ids, L)
        keep = (r >= r_min) & (r < r_min + L_new) & (c >= c_min) & (c < c_min + L_new)
        ids = ids[keep]
        r, c = r[keep], c[keep]
        fast = (r - r_min) * L_new + (c - c_min) + 1
        ref = np.array([twoD2oneD(*(lambda rc: (rc[0] - r_min, rc[1] - c_min))(
            oneD2twoD(int(u), L)), L_new) for u in ids])
        check(f"L={L}->{L_new} offset ({r_min},{c_min})",
              (fast == ref).all(), f"({ids.size} cells)")

        # adjacency survives the relabelling
        K, Kn = KingsLattice(L), KingsLattice(L_new)
        ok = all(K.has_edge(int(ids[i]), int(ids[j]))
                 == Kn.has_edge(int(fast[i]), int(fast[j]))
                 for i in range(0, min(60, ids.size))
                 for j in range(0, min(60, ids.size)))
        check(f"L={L}->{L_new} adjacency preserved", ok)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. Occupancy: both backends, and M2 detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_occupancy():
    print("Occupancy backends")
    rng = np.random.default_rng(7)
    L = 30
    arrs = [rng.choice(np.arange(1, L * L + 1), size=20, replace=False)
            for _ in range(10)]
    seen = set()
    arrs = [np.array([x for x in a if not (x in seen or seen.add(x))]) for a in arrs]
    probe = np.arange(1, L * L + 1)
    dense = Occupancy(L, arrs, dense=True).owner_of(probe)
    sparse = Occupancy(L, arrs, dense=False).owner_of(probe)
    check("dense and sparse backends agree", (dense == sparse).all())

    check("no collisions on disjoint chains", Occupancy.collisions(arrs) == 0)
    clash = list(arrs) + [np.array([arrs[0][0]])]
    check("collision detected", Occupancy.collisions(clash) == 1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The whole embedding, both ways (opt-in, as it embeds every cluster)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_oracle(n: int):
    from graphs import build_random_cubic
    from border_node_partitioning import partition_graph
    from cluster_embedding import embed_clusters
    from full_embedding import compact_embedding_guaranteed
    from construction import guaranteed_plan
    from construction_fast import guaranteed_plan_fast

    print(f"oracle diff at n={n} (fast vs reference Stage 3)")
    G, _, _ = build_random_cubic(n=n)
    part = partition_graph(G, max_cluster_size=15, method="auto", seed=7,
                           verbose=False)
    emb = embed_clusters(part, verbose=False)
    print(f"  k={part.k} T={part.border_count} cut={part.cut_size}", flush=True)

    t = time.perf_counter(); ps = guaranteed_plan(emb, part); ts = time.perf_counter() - t
    t = time.perf_counter(); pf = guaranteed_plan_fast(emb, part); tf = time.perf_counter() - t
    check("plan L", ps.L == pf.L, f"(L={ps.L})")
    check("plan stats", ps.stats == pf.stats)
    check("plan contacts", ps.contacts == pf.contacts, f"({len(ps.contacts)} pairs)")
    check("plan placements",
          [(p.cid, p.L, p.row0, p.col0) for p in ps.placements]
          == [(p.cid, p.L, p.row0, p.col0) for p in pf.placements])
    print(f"    plan {ts:.2f}s -> {tf:.2f}s  [{ts / max(tf, 1e-9):.0f}x]", flush=True)

    t = time.perf_counter()
    slow = compact_embedding_guaranteed(G, part, emb, verbose=False, fast=False)
    ts = time.perf_counter() - t
    t = time.perf_counter()
    fast = compact_embedding_guaranteed(G, part, emb, verbose=False, fast=True)
    tf = time.perf_counter() - t

    check("final L", slow.L == fast.L, f"(L={slow.L})")
    check("phi keys", set(slow.phi) == set(fast.phi))
    check("phi cells", all(
        np.array_equal(np.asarray(slow.phi[v], dtype=np.int64),
                       np.asarray(fast.phi[v], dtype=np.int64))
        for v in slow.phi), f"({slow.qubits} qubits)")
    check("route keys", set(slow.routes) == set(fast.routes), f"({len(slow.routes)})")
    check("route cells", all(
        np.array_equal(np.asarray(slow.routes[e], dtype=np.int64),
                       np.asarray(fast.routes[e], dtype=np.int64))
        for e in slow.routes))
    check("valid", slow.valid == fast.valid, f"(valid={slow.valid})")
    check("eemb", slow.eemb == fast.eemb, f"({slow.eemb}/{slow.m_total})")
    check("summary", slow.summary() == fast.summary())
    print(f"    stage 3 {ts:.1f}s -> {tf:.1f}s  [{ts / max(tf, 1e-9):.0f}x]", flush=True)


def main(argv):
    t0 = time.perf_counter()
    test_lattice()
    test_stripes()
    test_contacts()
    test_validate()
    test_trim_remap()
    test_occupancy()
    if "--oracle" in argv:
        i = argv.index("--oracle")
        test_oracle(int(argv[i + 1]) if len(argv) > i + 1 else 1000)
    print(f"\n{len(_PASS)} checks passed in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
