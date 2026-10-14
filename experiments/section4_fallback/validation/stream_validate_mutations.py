"""
A test that stream_validate catches broken embeddings. Each case copies a valid export,
breaks it in one specific way, and requires the validator to call it invalid, for the right
reason. Takes seconds on a small export (eg. random cubic, 1,000 vertices).

    python -B stream_validate_mutations.py <export dir>
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "stream_validate" + (".exe" if os.name == "nt" else ""))


def run(d):
    p = subprocess.run([EXE, d], capture_output=True, text=True)
    return json.loads(p.stdout), p.returncode


def main():
    src = sys.argv[1]
    meta = json.load(open(os.path.join(src, "meta.json")))
    L = meta["L_plan"]
    patch = np.fromfile(os.path.join(src, "patch.bin"), dtype=np.int32).reshape(-1, 3)
    seg = np.fromfile(os.path.join(src, "seg.bin"), dtype=np.int32).reshape(-1, 6)
    stripe = np.fromfile(os.path.join(src, "stripe.bin"), dtype=np.int32).reshape(-1, 3)
    off = np.fromfile(os.path.join(src, "csr_off.bin"), dtype=np.int64)
    nbr = np.fromfile(os.path.join(src, "csr_nbr.bin"), dtype=np.int32)
    n = meta["n"]
    rng = np.random.default_rng(1)

    base, rc = run(src)
    assert base["valid"] and rc == 0, base
    tests = []

    def case(name, expect, **files):
        d = src + "_mut"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
        for fname, arr in files.items():
            arr.tofile(os.path.join(d, fname + ".bin"))
        out, code = run(d)
        ok = (not out["valid"]) and code == 1 and out.get(expect, 0) > 0
        tests.append((name, ok, expect, out.get(expect), out["first_error"]))
        shutil.rmtree(d, ignore_errors=True)

    # 1. a lane's descent loses a middle cell: a super vertex split in two (M1)
    vert = np.nonzero((seg[:, 2] != 0) & (seg[:, 4] >= 5))[0]
    i = int(vert[rng.integers(len(vert))])
    r0, c0, dr, dc, k, o = seg[i].tolist()
    a = (r0, c0, dr, dc, 2, o)
    b = (r0 + 3 * dr, c0 + 3 * dc, dr, dc, k - 3, o)
    seg1 = np.concatenate([np.delete(seg, i, axis=0), np.array([a, b], dtype=np.int32)])
    case("gap in a lane's descent", "split_chains", seg=seg1)

    # 2. a hardware vertex claimed by two super vertices (M2)
    j = int(rng.integers(len(patch)))
    k2 = int(np.nonzero(patch[:, 2] != patch[j, 2])[0][0])
    p2 = patch.copy()
    p2 = np.concatenate([p2, np.array([[patch[k2, 0], patch[k2, 1], patch[j, 2]]], dtype=np.int32)])
    p2 = p2[np.lexsort((p2[:, 1], p2[:, 0]))]
    case("a cell claimed by two chains", "conflicts", patch=p2)

    # 3. a hardware vertex outside KG(L)
    p3 = np.concatenate([patch, np.array([[L - 1, L, patch[0, 2]]], dtype=np.int32)])
    p3 = p3[np.lexsort((p3[:, 1], p3[:, 0]))]
    case("a cell outside KG(L)", "out_of_bounds", patch=p3)

    # 4. a new edge between two interior vertices in different patches (M3)
    lane_owners = set(stripe[:, 0].tolist())
    interior = [int(o) for o in patch[:, 2] if int(o) not in lane_owners]
    u, v = interior[0], interior[-1]
    lists = [list(nbr[off[x]:off[x + 1]]) for x in range(n)]
    if v not in lists[u]:
        lists[u].append(v); lists[v].append(u)
    for x in range(n):
        lists[x].sort()
    off4 = np.zeros(n + 1, dtype=np.int64)
    off4[1:] = np.cumsum([len(t) for t in lists])
    nbr4 = np.array([y for t in lists for y in t], dtype=np.int32)
    case("an edge whose chains never touch", "edges_missing", csr_off=off4, csr_nbr=nbr4)

    # 5. a chain of the routing structure cut short, before its contacts (M3)
    t = int(np.argmax(stripe[:, 2]))
    s5 = stripe.copy()
    s5[t, 2] = 0
    case("a trunk cut short", "edges_missing", stripe=s5)

    # 6. a super vertex with no hardware vertices (M1)
    has_lane = set(stripe[:, 0].tolist())
    w = next(x for x in range(n) if x not in has_lane)
    p6 = patch[patch[:, 2] != w]
    case("an empty chain", "empty_chains", patch=p6)

    # 7. one hardware vertex of a super vertex moved to a free corner (M1)
    occupied = set(map(tuple, patch[:, :2].tolist()))
    chain_sizes = np.bincount(patch[:, 2], minlength=n)
    x = next(i for i in range(len(patch)) if chain_sizes[patch[i, 2]] >= 3 and patch[i, 2] not in has_lane)
    p7 = patch.copy()
    p7[x, 0], p7[x, 1] = L - 1, L - 1
    assert (L - 1, L - 1) not in occupied
    p7 = p7[np.lexsort((p7[:, 1], p7[:, 0]))]
    case("a patch cell moved away from its chain", "split_chains", patch=p7)

    # 8. a hardware vertex listed twice in one super vertex
    p8 = np.concatenate([patch, patch[:1]])
    p8 = p8[np.lexsort((p8[:, 1], p8[:, 0]))]
    case("a cell listed twice in one chain", "self_dup", patch=p8)

    for name, ok, expect, val, err in tests:
        print(f"{'CAUGHT' if ok else 'MISSED'}  {name:40s} {expect}={val}  ({err})")
    print(f"{sum(ok for _, ok, *_ in tests)}/{len(tests)} broken embeddings caught; the valid one passed")
    sys.exit(0 if all(ok for _, ok, *_ in tests) else 1)


if __name__ == "__main__":
    main()
