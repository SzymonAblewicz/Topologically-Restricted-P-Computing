"""
GGP + T-FM on compressed sparse row arrays (Section 4.3.2 of the report). Graph
growing partitioning grows the clusters one at a time by a breadth-first search, and
the Fiduccia-Mattheyses refinement, with its objective rewritten to be T, moves single
vertices between clusters whenever the move lowers T. Each pass takes linear time.

    python ggp_tfm.py 20000 100000
"""
import numpy as np
import time, sys


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# graph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def random_cubic_csr(n, seed=0, d=3):
    """A random d-regular graph by the configuration model, as CSR, with self-loops and parallel edges removed."""
    assert (n * d) % 2 == 0
    rng = np.random.default_rng(seed)
    stubs = np.repeat(np.arange(n, dtype=np.int64), d)
    rng.shuffle(stubs)
    u = stubs[0::2]
    v = stubs[1::2]
    keep = u != v
    u, v = u[keep], v[keep]
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    key = a * np.int64(n) + b
    key = np.unique(key)
    a = key // n
    b = key % n
    del key, u, v, stubs
    # symmetrise
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    del a, b
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order].astype(np.int32 if n < 2**31 else np.int64)
    del order
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    del src
    return indptr, dst


def grid2d_csr(side):
    n = side * side
    idx = np.arange(n, dtype=np.int64)
    r, c = idx // side, idx % side
    es = []
    m = c < side - 1
    es.append((idx[m], idx[m] + 1))
    m = r < side - 1
    es.append((idx[m], idx[m] + side))
    a = np.concatenate([x for x, _ in es])
    b = np.concatenate([y for _, y in es])
    src = np.concatenate([a, b]); dst = np.concatenate([b, a])
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o].astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GGP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bfs_ball(indptr, dst, n, target):
    """GGP, growing each cluster by BFS up to target, reseeding into the same cluster when its frontier runs dry."""
    ip = indptr.tolist()
    ds = dst.tolist()
    lab = [-1] * n
    cid = 0
    scan = 0
    frontier = [0] * (n if n < 1 << 22 else 1 << 22)
    for start in range(n):
        if lab[start] != -1:
            continue
        head = tail = 0
        frontier[tail] = start; tail += 1
        lab[start] = cid
        members = [start]
        mi = 0
        cnt = 1
        while cnt < target:
            if head < tail:
                u = frontier[head]; head += 1
                for j in range(ip[u], ip[u + 1]):
                    w = ds[j]
                    if lab[w] == -1:
                        lab[w] = cid
                        members.append(w)
                        if tail >= len(frontier):
                            frontier.extend([0] * len(frontier))
                        frontier[tail] = w; tail += 1
                        cnt += 1
                        if cnt >= target:
                            break
                continue
            # frontier dry -> reseed into the same cluster
            found = -1
            while mi < len(members):
                x = members[mi]
                for j in range(ip[x], ip[x + 1]):
                    if lab[ds[j]] == -1:
                        found = ds[j]; break
                if found != -1:
                    break
                mi += 1
            if found == -1:
                while scan < n and lab[scan] != -1:
                    scan += 1
                if scan >= n:
                    break
                found = scan
            lab[found] = cid
            members.append(found)
            if tail >= len(frontier):
                frontier.extend([0] * len(frontier))
            frontier[tail] = found; tail += 1
            cnt += 1
        cid += 1
    return np.asarray(lab, dtype=np.int32), cid


def T_exact(indptr, dst, lab, n):
    """T = #{v : out(v) > 0}, and the mask of border nodes."""
    src_lab = np.repeat(lab, np.diff(indptr))
    diff = (src_lab != lab[dst])
    borders = np.zeros(n, dtype=bool)
    deg = np.diff(indptr)
    nz = deg > 0
    if nz.any():
        red = np.add.reduceat(diff.astype(np.int64), indptr[:-1][nz])
        borders[nz] = red > 0
    return int(borders.sum()), borders


def fm_refine_csr(indptr, dst, lab, n, k, cap, max_passes=40, seed=0, verbose=False):
    """T-FM, single vertex moves that lower T, with the cut as the tie-break."""
    rng = np.random.default_rng(seed)
    sizes = np.bincount(lab, minlength=k).astype(np.int64)
    _, borders = T_exact(indptr, dst, lab, n)
    out = np.zeros(n, dtype=np.int32)
    src_lab = np.repeat(lab, np.diff(indptr))
    diff = (src_lab != lab[dst]).astype(np.int32)
    deg = np.diff(indptr)
    nz = deg > 0
    out[nz] = np.add.reduceat(diff, indptr[:-1][nz])
    del src_lab, diff

    target = n / k
    lo = max(1, int(np.floor(target * 0.85)))
    hi = min(int(np.ceil(target * 1.15)), cap)
    # keep one free slot above the mean cluster size, under the hard cap, or no move is legal
    if hi <= target:
        hi = min(int(np.ceil(target)) + 1, cap)
    lo = min(lo, hi)

    # Python lists in the hot loop, faster to index than numpy
    ip = indptr.tolist()
    ds = dst.tolist()
    lab_l = lab.tolist()
    ou = out.tolist()
    sz = sizes.tolist()

    order = np.arange(n, dtype=np.int64)
    total_moves = 0
    # so that max_passes=0 returns 0 passes
    p = -1
    for p in range(max_passes):
        rng.shuffle(order)
        moved = 0
        for u in order.tolist():
            cu = lab_l[u]
            s, e = ip[u], ip[u + 1]
            if s == e:
                continue
            nb = ds[s:e]
            cnts = {}
            for w in nb:
                if w == u:
                    continue
                cw = lab_l[w]
                cnts[cw] = cnts.get(cw, 0) + 1
            if not cnts:
                continue
            deg_u = e - s
            internal = cnts.get(cu, 0)
            best_c, best_gain = cu, (0, 0)
            ou_u_pos = 1 if ou[u] > 0 else 0
            if sz[cu] - 1 >= lo:
                for c, cnt in cnts.items():
                    if c == cu or sz[c] + 1 > hi:
                        continue
                    dT = (1 if (deg_u - cnt) > 0 else 0) - ou_u_pos
                    for w in nb:
                        if w == u:
                            continue
                        cw = lab_l[w]
                        if cw == cu:
                            if ou[w] == 0:
                                dT += 1
                        elif cw == c:
                            if ou[w] == 1:
                                dT -= 1
                    gain = (-dT, cnt - internal)
                    if gain > best_gain:
                        best_gain, best_c = gain, c
            if best_c != cu:
                for w in nb:
                    if w == u:
                        continue
                    cw = lab_l[w]
                    if cw == cu:
                        ou[w] += 1
                    elif cw == best_c:
                        ou[w] -= 1
                ou[u] = deg_u - cnts[best_c]
                lab_l[u] = best_c
                sz[cu] -= 1
                sz[best_c] += 1
                moved += 1
        total_moves += moved
        if verbose:
            print(f"      pass {p}: {moved} moves", flush=True)
        if moved == 0:
            break
    return np.asarray(lab_l, dtype=np.int32), p + 1, total_moves


def bytes_report(n, indptr, dst):
    return (indptr.nbytes + dst.nbytes) / 2**20


def run(n, s=50, seed=0, passes=40, family="cubic", verbose=False):
    t = {}
    t0 = time.perf_counter()
    if family == "cubic":
        indptr, dst = random_cubic_csr(n, seed=seed)
    else:
        side = int(round(n ** 0.5)); n = side * side
        indptr, dst = grid2d_csr(side)
    t["gen"] = time.perf_counter() - t0
    m = len(dst) // 2
    mem = bytes_report(n, indptr, dst)

    t0 = time.perf_counter()
    lab, k = bfs_ball(indptr, dst, n, int(s * 0.9))
    t["init"] = time.perf_counter() - t0
    T0, _ = T_exact(indptr, dst, lab, n)

    t0 = time.perf_counter()
    lab, np_used, moves = fm_refine_csr(indptr, dst, lab, n, k, s,
                                        max_passes=passes, seed=seed, verbose=verbose)
    t["refine"] = time.perf_counter() - t0
    T1, _ = T_exact(indptr, dst, lab, n)
    sizes = np.bincount(lab)
    assert sizes.max() <= s, f"cap violated: {sizes.max()} > {s}"

    print(f"n={n:>12,d} m={m:>13,d} k={k:>10,d} | graph {mem:8.1f} MB | "
          f"gen {t['gen']:7.2f}s  init {t['init']:7.2f}s  refine {t['refine']:8.2f}s "
          f"({np_used} passes) | T/n {T0/n:.4f} -> {T1/n:.4f} | "
          f"total {sum(t.values()):8.2f}s", flush=True)
    return dict(n=n, m=m, k=k, mem_mb=mem, T0=T0 / n, T1=T1 / n, passes=np_used, **t)


if __name__ == "__main__":
    ns = [int(float(x)) for x in sys.argv[1:]] or [20000]
    print("CSR Stage 1: BFS-ball(target 0.9s) + T-FM, cap s=50, cubic")
    print("=" * 132)
    for n in ns:
        run(n, verbose=(n >= 2_000_000))
