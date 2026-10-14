"""
Export the whole embedding of a finished fallback run in compact form, for stream_validate,
which checks embeddings far too large to build (the routing structure alone holds about
T^2/2 hardware vertices, 1.1 x 10^11 at a million vertices). The patches are listed cell by
cell, the lanes as straight runs, and the chains of the routing structure by their index.

    python -B export_embedding.py <family> <n_request> <seed> <s> [--check] [--sample K] [--full]

--check compares every lane (or K of them) with the construction's own cells, and --full
compares every super vertex with the whole construction (up to about 30,000 vertices).
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "fallback", "benchmark_graphs")]

import run_fallback as rf                                              # noqa: E402
from kings_lattice import rc_of                                        # noqa: E402
from construction import _BOTTOM, _LEFT, _RIGHT, _TOP, _lane_cells     # noqa: E402
from construction_fast import (funnel_cells, guaranteed_plan_fast,     # noqa: E402
                               stripe_cols, trunk_ids)


def _range_len(a, b, step):
    """len(range(a, b, step)) for step = +-1."""
    return max(0, b - a) if step > 0 else max(0, a - b)


def lane_escape_segments(ln, band_bot):
    """_lane_cells(ln, band_bot) as straight runs, in the same cell order."""
    r, c = ln.dock
    free_bottom = ln.side == _BOTTOM and ln.gap_key is None
    segs = []

    def add(r0, c0, dr, dc, n):
        if n > 0:
            segs.append((r0, c0, dr, dc, n))

    def run(row, c_from, c_to):                       # construction._run
        step = -1 if c_to < c_from else 1
        add(row, c_from + step, 0, step, _range_len(c_from + step, c_to + step, step))

    if ln.side == _TOP:
        add(r - 1, c, -1, 0, _range_len(r - 1, ln.turn_row - 1, -1))
        run(ln.turn_row, c, ln.turn_col)
    elif ln.side == _LEFT:
        add(r, c - 1, 0, -1, _range_len(c - 1, ln.turn_col - 1, -1))
    elif ln.side == _RIGHT:
        add(r, c + 1, 0, 1, _range_len(c + 1, ln.turn_col + 1, 1))
    elif not free_bottom:
        add(r + 1, c, 1, 0, _range_len(r + 1, ln.turn_row + 1, 1))
        run(ln.turn_row, c, ln.turn_col)
    start = r + 1 if free_bottom else ln.turn_row + 1
    add(start, ln.turn_col, 1, 0, _range_len(start, band_bot, 1))
    return segs


def lane_funnel_segments(ln, plan):
    """funnel_cells(ln.turn_col, ln.stripe, plan) as straight runs, in the same order."""
    F, band_bot = plan.F, plan.band_bot
    b = plan.X0 + ln.stripe
    d = ln.turn_col - b
    sgn = (d > 0) - (d < 0)
    ad = abs(d)
    assert ad <= F, (ad, F)                      # F is the largest drift of any lane
    segs = []
    if ad > 0:                                   # one column of drift per row
        segs.append((band_bot, b + sgn * (ad - 1), 1, -sgn, ad))
    if F - ad > 0:                               # then straight down onto the chain
        segs.append((band_bot + ad, b, 1, 0, F - ad))
    return segs


def expand(segs):
    """The cells of straight runs, as (row, col)."""
    out = []
    for r0, c0, dr, dc, n in segs:
        t = np.arange(n, dtype=np.int64)
        out.append(np.stack([r0 + dr * t, c0 + dc * t], axis=1))
    return np.concatenate(out) if out else np.empty((0, 2), dtype=np.int64)


def main():
    family, n_req, seed, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    check = "--check" in sys.argv
    full = "--full" in sys.argv
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    tag = f"{family}_{n_req}_{seed}_s{s}"
    ck = os.path.join(rf.RES, "ckpt", tag)
    out = os.path.join(rf.RES, "export", tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()

    G = rf.load_graph(family, n_req, seed)
    part, _ = pickle.load(open(ck + "_part.pkl", "rb"))
    done = rf.load_done(ck + "_embs.pkl")
    embs = {cid: best for cid, (best, _, _) in done.items()}
    assert len(embs) == part.k, f"{len(embs)} of {part.k} clusters in the checkpoint"
    print(f"[{tag}] loaded n={G.number_of_nodes()} m={G.number_of_edges()} k={part.k} "
          f"in {time.perf_counter() - t0:.0f} s", flush=True)

    plan = guaranteed_plan_fast(embs, part)
    L = plan.L
    print(f"[{tag}] plan L={L} rail={plan.rail} X0={plan.X0} S={plan.S} F={plan.F} "
          f"band_bot={plan.band_bot} lanes={len(plan.lanes)} in {time.perf_counter() - t0:.0f} s", flush=True)

    # border node u becomes its chain's index 0 .. T-1, so the validator reads the graph in
    # the order it sweeps the chains; the other vertices follow
    index = {u: ln.stripe for u, ln in plan.lanes.items()}
    assert sorted(index.values()) == list(range(len(index))), "chains are not 0 .. T-1"
    for v in sorted(G.nodes()):
        if v not in index:
            index[v] = len(index)
    n = len(index)
    assert n == G.number_of_nodes()

    # the graph, as CSR
    deg = np.zeros(n + 1, dtype=np.int64)
    eu = np.fromiter((index[u] for u, v in G.edges()), dtype=np.int64, count=G.number_of_edges())
    ev = np.fromiter((index[v] for u, v in G.edges()), dtype=np.int64, count=G.number_of_edges())
    assert not np.any(eu == ev), "self-loop in the graph"
    src = np.concatenate([eu, ev])
    dst = np.concatenate([ev, eu])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    np.add.at(deg, src + 1, 1)
    off = np.cumsum(deg)
    off.astype(np.int64).tofile(os.path.join(out, "csr_off.bin"))
    dst.astype(np.int32).tofile(os.path.join(out, "csr_nbr.bin"))

    # the patches, cell by cell, placed as the construction places them
    pr, pc, po = [], [], []
    for p in plan.placements:
        emb = embs[p.cid]
        for v, chain in emb.phi.items():
            r, c = rc_of(np.asarray(chain, dtype=np.int64), emb.L)
            pr.append(r + p.row0)
            pc.append(c + p.col0)
            po.append(np.full(r.size, index[v], dtype=np.int64))
    pr, pc, po = np.concatenate(pr), np.concatenate(pc), np.concatenate(po)
    o = np.lexsort((pc, pr))
    np.stack([pr[o], pc[o], po[o]], axis=1).astype(np.int32).tofile(os.path.join(out, "patch.bin"))

    # the lanes as straight runs, and the chains of the routing structure as (owner, index, reach)
    segs, stripes = [], []
    for u, ln in plan.lanes.items():
        own = index[u]
        for r0, c0, dr, dc, k in lane_escape_segments(ln, plan.band_bot) + lane_funnel_segments(ln, plan):
            segs.append((r0, c0, dr, dc, k, own))
        stripes.append((own, ln.stripe, plan.reach[ln.stripe]))
    seg = np.array(segs, dtype=np.int64).reshape(-1, 6)
    stripe = np.array(stripes, dtype=np.int64).reshape(-1, 3)
    seg.astype(np.int32).tofile(os.path.join(out, "seg.bin"))
    stripe.astype(np.int32).tofile(os.path.join(out, "stripe.bin"))

    meta = dict(tag=tag, n=n, m=G.number_of_edges(), L_plan=int(L), rail=int(plan.rail), X0=int(plan.X0),
                S=int(plan.S), F=int(plan.F), band_bot=int(plan.band_bot), n_patch=int(pr.size),
                n_seg=int(seg.shape[0]), n_stripe=int(stripe.shape[0]),
                trunk_cells=int(sum(r + 1 for _, _, r in stripes)),
                seg_cells=int(seg[:, 4].sum()) if seg.size else 0, T=int(part.border_count), k=int(part.k))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[{tag}] exported {meta} in {time.perf_counter() - t0:.0f} s", flush=True)

    if check:
        lanes = list(plan.lanes.items())
        if sample and sample < len(lanes):
            rng = np.random.default_rng(0)
            lanes = [lanes[i] for i in rng.choice(len(lanes), sample, replace=False)]
        bad = 0
        for u, ln in lanes:
            ref = np.asarray(_lane_cells(ln, plan.band_bot), dtype=np.int64).reshape(-1, 2)
            got = expand(lane_escape_segments(ln, plan.band_bot))
            fref = funnel_cells(ln.turn_col, ln.stripe, plan)
            fgot = expand(lane_funnel_segments(ln, plan))
            tref = trunk_ids(ln, plan, L)
            j = ln.stripe
            idx = np.arange(plan.reach[j] + 1, dtype=np.int64)
            tgot = (plan.rail + idx + 1) * L + plan.X0 + stripe_cols(j, idx + 1, plan.S) + 1
            if not (np.array_equal(ref, got) and np.array_equal(fref, fgot) and np.array_equal(tref, tgot)):
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH lane {u!r}: escape {np.array_equal(ref, got)} funnel "
                          f"{np.array_equal(fref, fgot)} trunk {np.array_equal(tref, tgot)}", flush=True)
        print(f"[{tag}] check: {len(lanes)} lanes compared with _lane_cells/funnel_cells/trunk_ids, "
              f"{bad} mismatches", flush=True)
        meta["check_lanes"], meta["check_mismatch"] = len(lanes), bad

    if full:
        # the whole construction, as validate_checkpoint.py builds it, compared super vertex by super vertex
        from kings_lattice import KingsLattice
        from construction_fast import guaranteed_route_fast
        H = KingsLattice(L)
        phi = {}
        for p in plan.placements:
            emb = embs[p.cid]
            for v, chain in emb.phi.items():
                r, c = rc_of(np.asarray(chain, dtype=np.int64), emb.L)
                phi[v] = (r + p.row0) * L + (c + p.col0) + 1
        cut_edges = [(u, v) for u, v, _, _ in part.cross_edges]
        _, _, work = guaranteed_route_fast(H, phi, cut_edges, plan)
        ids_patch = pr * L + pc + 1
        from collections import defaultdict
        acc = defaultdict(list)
        o2 = np.argsort(po, kind="stable")
        bounds = np.searchsorted(po[o2], np.arange(n + 1))
        for i in range(n):
            acc[i].append(ids_patch[o2[bounds[i]:bounds[i + 1]]])
        for r0, c0, dr, dc, k, own in seg.tolist():
            t = np.arange(k, dtype=np.int64)
            acc[own].append((r0 + dr * t) * L + (c0 + dc * t) + 1)
        for own, j, rch in stripe.tolist():
            idx = np.arange(rch + 1, dtype=np.int64)
            acc[own].append((plan.rail + idx + 1) * L + plan.X0 + stripe_cols(j, idx + 1, plan.S) + 1)
        diff = 0
        for v, arr in work.items():
            a = np.sort(np.asarray(arr, dtype=np.int64))
            b = np.sort(np.concatenate(acc[index[v]]))
            if not np.array_equal(a, b):
                diff += 1
        print(f"[{tag}] full: {len(work)} super vertices of the construction compared, {diff} differ", flush=True)
        meta["full_chains"], meta["full_differ"] = len(work), diff

    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    ok = meta.get("check_mismatch", 0) == 0 and meta.get("full_differ", 0) == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
