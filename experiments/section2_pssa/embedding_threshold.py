"""
Figure 4 of the report: the embedding threshold Vbar(L) of Sugie et al. (2021), the
smallest input size at which fewer than 19 of 20 random cubic graphs embed, for the
improved PSSA with uniform moves, with Gibbs-weighted moves, and with uniform moves
given the same annealing time as the Gibbs version. Resumable.

    python -B embedding_threshold.py --quick
    python -B embedding_threshold.py
"""
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pssa", "improved_pssa"))
import networkx as nx            # noqa: E402
import improved_pssa as P     # noqa: E402

OUT = os.path.join(ROOT, "results", "section2_pssa")
TMAX = 300_000  # as in Table 3
P_EMB = 0.95
GRAPHS = 20                      # random cubic graphs per (L, n): seeds 1..20
LS = [6, 7, 8, 9, 10, 11, 12]
WORKERS = 8
RUNS = "embedding_threshold.csv"      # main() switches to embedding_threshold_quick.csv for --quick
FIELDS = ["mode", "L", "n", "graph", "tmax", "success", "e_final", "m", "t_anneal"]


def one_based(G):
    return nx.relabel_nodes(G, {v: v + 1 for v in G.nodes()})


def run_once(args):
    mode, L, n, g, tmax = args
    I = one_based(nx.random_regular_graph(3, n, seed=g))
    H = P.build_hardware_graph(L)
    gp = P.build_guiding_pattern(H, L)
    sched = P.KingsGraphSchedule(tmax=tmax, L=L)
    t0 = time.perf_counter()
    phi, _ = P._pssa_cpp(I, H, gp, sched, seed=g, gibbs=(mode == "gibbs"))
    t_anneal = time.perf_counter() - t0
    phi_f = P.terminal_search(phi, I, H)
    e_final = P.eemb(phi_f, I, H, P.invert(phi_f))
    m = I.number_of_edges()
    ok = int(e_final == m and P.is_valid_embedding(phi_f, I, H))
    return dict(mode=mode, L=L, n=n, graph=g, tmax=tmax, success=ok,
                e_final=e_final, m=m, t_anneal=round(t_anneal, 4))


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "embedding_threshold_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def load_done():
    path = os.path.join(OUT, RUNS)
    if not os.path.exists(path):
        return []
    return [dict(r, L=int(r["L"]), n=int(r["n"]), graph=int(r["graph"]), tmax=int(r["tmax"]),
                 success=int(r["success"]), t_anneal=float(r["t_anneal"]))
            for r in csv.DictReader(open(path, newline=""))]


def save(rows):
    with open(os.path.join(OUT, RUNS), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)


def point(ex, rows, mode, L, n, tmax, graphs):
    """The number of successes at (mode, L, n), running only what is not already in rows."""
    have = [r for r in rows if r["mode"] == mode and r["L"] == L and r["n"] == n]
    todo = [(mode, L, n, g, tmax) for g in graphs if g not in {r["graph"] for r in have}]
    if todo:
        new = list(ex.map(run_once, todo))
        rows.extend(new)
        save(rows)
        have += new
    return sum(r["success"] for r in have), have


def threshold(ex, rows, mode, L, tmax, graphs, n_start):
    """Vbar(L), the smallest even n >= n_start whose success rate falls below P_EMB."""
    need = P_EMB * len(graphs)
    n = n_start
    while True:
        k, _ = point(ex, rows, mode, L, n, tmax, graphs)
        log(f"  {mode:12s} L={L:2d} n={n:3d}: {k}/{len(graphs)}")
        if k < need:
            return n
        n += 2


def main(quick=False):
    global RUNS
    if quick:
        RUNS = "embedding_threshold_quick.csv"
    ls = [5, 6] if quick else LS
    graphs = list(range(1, 4)) if quick else list(range(1, GRAPHS + 1))
    tmax = 20_000 if quick else TMAX
    rows = [] if quick else load_done()
    summary = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for L in ls:
            n0 = max(4, 2 * ((L + 1) // 2))           # start at about L, even, as Vbar >= L + 1
            vb_u = threshold(ex, rows, "uniform", L, tmax, graphs, n0)
            vb_g = threshold(ex, rows, "gibbs", L, tmax, graphs, n0)
            # the same annealing time: uniform tmax scaled by the Gibbs/uniform time ratio at this L
            t_g = statistics.mean(r["t_anneal"] for r in rows if r["mode"] == "gibbs" and r["L"] == L)
            t_u = statistics.mean(r["t_anneal"] for r in rows if r["mode"] == "uniform" and r["L"] == L)
            tmax_eq = int(tmax * t_g / t_u)
            vb_t = threshold(ex, rows, "uniform_time", L, tmax_eq, graphs, n0)
            summary[L] = dict(uniform=vb_u, gibbs=vb_g, uniform_time=vb_t, tmax_uniform_time=tmax_eq)
            log(f"L={L}: Vbar uniform {vb_u}, gibbs {vb_g}, uniform at matched time {vb_t} (tmax {tmax_eq})")
    if not quick:
        json.dump(summary, open(os.path.join(OUT, "embedding_threshold_summary.json"), "w"), indent=2)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
