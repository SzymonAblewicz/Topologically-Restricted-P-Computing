"""
The fallback method end to end on one graph (Section 4 of the report). Partitioning
to minimize the border node count, every cluster by PSSA wBC in parallel, and then
the construction, which gives L without writing out every chain. With --validate the
whole embedding is also built and checked against the whole graph (feasible for small
n), and with --checkpoint a stopped run resumes where it was.

    python run_fallback.py --n 100000 --checkpoint
    python run_fallback.py --n 1000 --validate
    python run_fallback.py --family erdos_renyi --n 10630 --seed 1
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "section4_fallback")
sys.path[:0] = [os.path.join(ROOT, "src", d) for d in ("pssa/pssa_wbc", "cluster_embedding", "divide_and_conquer", "fallback", "benchmark_graphs")]

import networkx as nx                                              # noqa: E402
import cluster_embedding as PP                                     # noqa: E402
from cluster_embedding import ClusterEmbedding                     # noqa: E402
from graphs import build_benchmark_graph                           # noqa: E402

FIELDS = ["family", "n_request", "seed", "n", "m", "mean_degree", "s", "k", "T", "T_over_n",
          "cut", "t_stage1", "stage2_clusters", "stage2_ok", "stage2_missing_edges",
          "stage2_hard_fail", "t_stage2_wall", "t_stage2_cpu", "workers", "t_stage3_plan",
          "L_plan", "area_plan", "floor", "validated", "valid", "L_final", "area_final",
          "eemb", "m_total", "error"]


def worker(args):
    cid = args[0]
    t0 = time.perf_counter()
    try:
        _, best = PP._embed_single_worker(args)
        return cid, best, "", time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001  (any error, so one bad cluster is a hard fail and cannot stall the run)
        return cid, None, f"{type(exc).__name__}: {exc}"[:120], time.perf_counter() - t0


def load_graph(family, n_req, seed):
    return build_benchmark_graph(family, n=n_req, seed=seed)[0]


def load_done(path):
    """The finished clusters of a checkpointed run, {cid: (embedding, cpu s, wall s)}, dropping a last frame cut short."""
    done, good = {}, 0
    if os.path.exists(path):
        with open(path, "rb") as f:
            while True:
                try:
                    cid, best, secs, wall = pickle.load(f)
                except EOFError:
                    break
                except Exception:                     # noqa: BLE001  (a frame cut short)
                    break
                done[cid], good = (best, secs, wall), f.tell()
        with open(path, "r+b") as f:
            f.truncate(good)
    return done


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from border_node_partitioning import partition_graph
    from construction_fast import guaranteed_plan_fast
    from full_embedding import compact_embedding_guaranteed

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="random_cubic")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--s", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default="runs.csv")
    ap.add_argument("--method", default="mtkahypar", choices=["auto", "mtkahypar", "csr"],
                    help="the partitioner: Mt-KaHyPar, GGP + T-FM (csr), or auto")
    ap.add_argument("--checkpoint", action="store_true", help="resumable, see the docstring")
    ap.add_argument("--stage1-only", action="store_true", help="stop after the partition")
    a = ap.parse_args()
    tag = f"{a.family}_{a.n}_{a.seed}_s{a.s}"
    os.makedirs(os.path.join(RES, "phi"), exist_ok=True)
    row = dict(family=a.family, n_request=a.n, seed=a.seed, s=a.s, workers=a.workers,
               validated=int(a.validate), error="")

    G = load_graph(a.family, a.n, a.seed)
    n, m = G.number_of_nodes(), G.number_of_edges()
    row.update(n=n, m=m, mean_degree=round(2 * m / n, 3))
    print(f"[{tag}] graph n={n} m={m}", flush=True)

    ck = os.path.join(RES, "ckpt", tag) if a.checkpoint else None
    if ck and os.path.exists(ck + "_part.pkl"):
        part, t_stage1 = pickle.load(open(ck + "_part.pkl", "rb"))
        print(f"[{tag}] resuming from the checkpoint", flush=True)
    else:
        t0 = time.perf_counter()
        part = partition_graph(G, max_cluster_size=a.s, method=a.method, seed=7, verbose=False)
        t_stage1 = time.perf_counter() - t0
        if ck:
            os.makedirs(os.path.dirname(ck), exist_ok=True)
            with open(ck + "_part.tmp", "wb") as fh:
                pickle.dump((part, t_stage1), fh)
            os.replace(ck + "_part.tmp", ck + "_part.pkl")
    row.update(t_stage1=round(t_stage1, 1), k=part.k, T=part.border_count,
               T_over_n=round(part.border_count / n, 4), cut=part.cut_size)
    json.dump({str(v): c for v, c in part.labels.items()},
              open(os.path.join(RES, f"labels_{tag}.json"), "w"))
    print(f"[{tag}] stage 1: k={part.k} T/n={row['T_over_n']} in {row['t_stage1']} s "
          f"(method {part.method})", flush=True)
    if a.stage1_only:
        return

    d = PP.embed_clusters.__defaults__            # (L_slack, tmax, tmax_hard, seeds, gibbs, verbose)
    tasks = []
    for cid in part.communities:
        I = nx.Graph(part.subgraphs[cid])
        border = part.border_nodes[cid]
        L0 = PP._min_feasible_L(I.number_of_nodes(), len(border))
        tasks.append((cid, I, border, L0, d[0], d[1], d[2], d[3], d[4]))
    tasks.sort(key=lambda t: -(t[1].number_of_nodes() * 100 + len(t[2])))

    embs, fails, cpu, missing = {}, 0, 0.0, 0
    done = load_done(ck + "_embs.pkl") if ck else {}

    def phi_line(cid, best):
        return json.dumps({"cid": cid, "L": best.L, "success": int(best.success), "eemb": best.eemb,
                           "phi": {str(v): [int(u) for u in ch] for v, ch in best.phi.items()}}) + "\n"

    prior_wall = max((w for _, _, w in done.values()), default=0.0)
    todo = [t for t in tasks if t[0] not in done]
    if done:
        print(f"[{tag}] stage 2: {len(done)} clusters from the checkpoint, {len(todo)} to go", flush=True)
    ckf = open(ck + "_embs.pkl", "ab") if ck else None
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, \
            open(os.path.join(RES, "phi", f"{tag}.jsonl"), "w") as fp:
        for cid, (best, secs, _) in done.items():      # the clusters from the checkpoint first
            embs[cid] = best
            cpu += secs
            missing += best.m_I - best.eemb if best.m_I is not None else 0
            fp.write(phi_line(cid, best))
        futs = [ex.submit(worker, t) for t in todo]
        try:
            for i, f in enumerate(as_completed(futs), len(done) + 1):
                cid, best, err, secs = f.result()
                cpu += secs
                if best is None:
                    fails += 1
                    print(f"[{tag}] cluster {cid} failed: {err}", flush=True)
                else:
                    embs[cid] = best
                    if ckf:
                        pickle.dump((cid, best, secs, prior_wall + time.perf_counter() - t0), ckf)
                        ckf.flush()
                    missing += best.m_I - best.eemb if best.m_I is not None else 0
                    fp.write(phi_line(cid, best))
                if i % 100 == 0 or i == len(tasks):
                    print(f"[{tag}] stage 2: {i}/{len(tasks)} done, {time.perf_counter() - t0:.0f} s, "
                          f"{fails} hard fails", flush=True)
        except BaseException:  # a worker died: stop at once, as embed_clusters does
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    if ckf:
        ckf.close()
    ok = sum(1 for e in embs.values() if e.success)
    row.update(stage2_clusters=len(tasks), stage2_ok=ok, stage2_hard_fail=fails,
               stage2_missing_edges=missing, t_stage2_wall=round(prior_wall + time.perf_counter() - t0, 1),
               t_stage2_cpu=round(cpu, 1))

    if fails == 0:
        t0 = time.perf_counter()
        try:
            plan = guaranteed_plan_fast(embs, part)
            row.update(t_stage3_plan=round(time.perf_counter() - t0, 2), L_plan=plan.L,
                       area_plan=round((plan.L / (n - 1)) ** 2, 4),
                       floor=round((part.border_count / n) ** 2, 4))
            print(f"[{tag}] stage 3 plan: L={plan.L} area={row['area_plan']} "
                  f"in {row['t_stage3_plan']} s", flush=True)
        except Exception as exc:                  # noqa: BLE001
            row["error"] = f"stage 3 plan: {type(exc).__name__}: {exc}"[:150]
        if a.validate and not row["error"]:
            res = compact_embedding_guaranteed(G, part, embs, verbose=False, fast=True)
            row.update(valid=str(res.valid), L_final=res.L, area_final=round((res.L / (n - 1)) ** 2, 4),
                       eemb=res.eemb, m_total=res.m_total)
            print(f"[{tag}] validated: valid={res.valid} {res.eemb}/{res.m_total} L={res.L}", flush=True)
    else:
        row["error"] = f"{fails} clusters with no result at any L"

    path = os.path.join(RES, a.out)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{tag}] done: {row}", flush=True)


if __name__ == "__main__":
    main()
