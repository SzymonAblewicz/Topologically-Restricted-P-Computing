"""The improved PSSA, without border nodes, on random cubic graphs of 30 to 100 vertices (Section 2.5 of the report)."""
import csv, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pssa", "pssa_wbc"))
import networkx as nx
from pssa_wbc import ImprovedPSSA

NS = [30, 50, 70, 100]
SEEDS = [1, 2, 3]
TMAX = 1_500_000

def job(a):
    n, L, seed = a
    I = nx.random_regular_graph(3, n, seed=seed)
    t0 = time.perf_counter()
    try:
        r = ImprovedPSSA(L=L, tmax=TMAX, gibbs=True, seed=seed, verbose=False).run(I, deadline=time.time() + 300)
        return n, L, seed, int(r.success), r.eemb, r.m_I, round(time.perf_counter() - t0, 1)
    except ValueError as e:
        return n, L, seed, 0, "", "", round(time.perf_counter() - t0, 1)

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, L, s) for n, Ls in ((50, (20, 25)), (70, (22, 28, 35)), (100, (31, 40, 50))) for L in Ls for s in SEEDS]
    with ProcessPoolExecutor(max_workers=14) as ex, open(os.path.join(ROOT, "results", "section2_pssa", "pssa_limit_random_cubic_2.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["n", "L", "seed", "success", "eemb", "m", "secs"])
        for row in ex.map(job, jobs):
            w.writerow(row); fh.flush(); print(row, flush=True)
