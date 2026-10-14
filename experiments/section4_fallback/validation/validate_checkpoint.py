"""
End-to-end validation of a finished fallback run from its checkpoint, with no PSSA rerun.
The whole embedding is built by the construction and checked as a minor-embedding of the
whole graph, (M1)-(M3), as run_fallback.py --validate does. Feasible up to about 30,000
vertices; beyond that, validate_stream.py.

    python -B validate_checkpoint.py <family> <n_request> <seed> <s>
"""
import csv
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "fallback", "benchmark_graphs")]

import run_fallback as rf                                  # noqa: E402
from full_embedding import compact_embedding_guaranteed    # noqa: E402
from construction_fast import guaranteed_plan_fast         # noqa: E402

FIELDS = ["family", "n_request", "seed", "s", "n", "m", "k", "T", "L_plan", "valid", "L_final",
          "eemb", "m_total", "t_build_validate", "peak_rss_gb", "error"]


def main():
    family, n_req, seed, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    tag = f"{family}_{n_req}_{seed}_s{s}"
    ck = os.path.join(rf.RES, "ckpt", tag)
    G = rf.load_graph(family, n_req, seed)
    part, _ = pickle.load(open(ck + "_part.pkl", "rb"))
    done = rf.load_done(ck + "_embs.pkl")
    embs = {cid: best for cid, (best, _, _) in done.items()}
    row = dict(family=family, n_request=n_req, seed=seed, s=s, n=G.number_of_nodes(), m=G.number_of_edges(),
               k=part.k, T=part.border_count, error="")
    print(f"[{tag}] n={row['n']} m={row['m']} k={part.k}, {len(embs)} cluster embeddings", flush=True)
    try:
        assert len(embs) == part.k, f"{len(embs)} of {part.k} clusters in the checkpoint"
        row["L_plan"] = guaranteed_plan_fast(embs, part).L
        t0 = time.perf_counter()
        res = compact_embedding_guaranteed(G, part, embs, verbose=False, fast=True)
        row.update(valid=str(res.valid), L_final=res.L, eemb=res.eemb, m_total=res.m_total,
                   t_build_validate=round(time.perf_counter() - t0, 1))
    except Exception as exc:                               # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        import psutil
        row["peak_rss_gb"] = round(psutil.Process().memory_info().peak_wset / 2**30, 2)   # Windows only
    except Exception:                                      # noqa: BLE001
        pass
    path = os.path.join(rf.RES, "validate_checkpoint.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{tag}] {row}", flush=True)
    sys.exit(0 if row.get("valid") == "True" else 1)


if __name__ == "__main__":
    main()
