"""
The Divide and Conquer runs of Section 3.3 of the report, on many random cubic graphs
at once. For each graph, one run makes the partition and the clusters, and then the
placing and routing is rerun with a new random seed until it succeeds, up to
--max-seeds tries. Resumable, as finished tries are read back from their logs.

    python run_until_success.py --n 700 800 900 --graph-seeds 1 --workers 8 --max-seeds 40
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DNC = os.path.join(ROOT, "src", "divide_and_conquer")
RES = os.path.join(ROOT, "results", "section3_divide_and_conquer")


class Graph:
    """One graph and its tries."""

    def __init__(self, name, seed, s, script="divide_and_conquer.py", extra=()):
        self.name, self.seed, self.s = name, seed, s
        self.script, self.extra = script, list(extra)
        self.folder = os.path.join(RES, f"{name}_s{s}_seed{seed}")
        self.prep = None  # (process, log file) making the partition and the clusters
        self.running = {}  # seed -> (process, log file)
        self.valid = {}  # seed -> complete and valid?
        self.crashes = {}  # seed -> times it crashed (eg. out of memory)
        if os.path.isdir(self.folder):
            for f in os.listdir(self.folder):
                if f.startswith("log_sa"):
                    text = open(os.path.join(self.folder, f)).read()
                    if "joined:" in text:
                        self.valid[int(f[6:-4])] = "valid=True" in text

    def log(self, k):
        return os.path.join(self.folder, f"log_sa{k}.txt")

    def prepared(self):
        return self.prep is None and os.path.exists(os.path.join(self.folder, "clusters.pkl"))

    def success(self):
        return min((k for k, ok in self.valid.items() if ok), default=None)

    def next_seed(self):
        k = 1
        while k in self.valid or k in self.running:
            k += 1
        return k

    def busy(self):
        return len(self.running) + (self.prep is not None)

    def can_start(self, max_seeds, preparing=0):
        if not self.prepared():                 # one preparation at a time: each starts a pool of workers
            return self.prep is None and preparing == 0
        k, s = self.next_seed(), self.success()
        return k <= max_seeds and (s is None or k < s)

    def done(self, max_seeds):
        s = self.success()
        if s is not None:
            return all(k in self.valid for k in range(1, s)) and not any(k < s for k in self.running)
        return self.prepared() and self.next_seed() > max_seeds and not self.running

    def start(self):
        os.makedirs(self.folder, exist_ok=True)
        cmd = [sys.executable, "-B", self.script, "--graph", self.name, "--s", str(self.s),
               "--seed", str(self.seed), "--dpi", "150", "--out", RES] + self.extra
        if not self.prepared():
            fh = open(os.path.join(self.folder, "log_prep.txt"), "w")
            self.prep = (subprocess.Popen(cmd + ["--sa-seed", "0"], cwd=DNC, stdout=fh,
                                          stderr=subprocess.STDOUT), fh)
            return
        k = self.next_seed()
        fh = open(self.log(k), "w")
        self.running[k] = (subprocess.Popen(cmd + ["--sa-seed", str(k), "--rejoin"], cwd=DNC, stdout=fh,
                                            stderr=subprocess.STDOUT), fh)

    def poll(self):
        if self.prep is not None:
            p, fh = self.prep
            text = open(os.path.join(self.folder, "log_prep.txt")).read()
            if "clusters:" in text or p.poll() is not None:
                if p.poll() is None:  # partition and clusters saved, the rest of this run is not needed
                    p.kill()
                    p.wait()
                fh.close()
                self.prep = None
                if "clusters:" not in text:  # crashed before the clusters were saved, made again
                    print(f"  {self.name} seed {self.seed}: preparation CRASHED\n{text[-300:]}", flush=True)
                    if os.path.exists(os.path.join(self.folder, "clusters.pkl")):
                        os.remove(os.path.join(self.folder, "clusters.pkl"))
                else:
                    print(f"  {self.name} seed {self.seed}: {[l for l in text.splitlines() if 'partition:' in l][0].split('] ')[1]}",
                          flush=True)
        for k, (p, fh) in list(self.running.items()):
            if p.poll() is not None:
                del self.running[k]
                fh.close()
                text = open(self.log(k)).read()
                if "joined:" not in text:  # crashed (eg. out of memory), not counted and tried again
                    self.crashes[k] = self.crashes.get(k, 0) + 1
                    print(f"  {self.name} seed {self.seed}, annealer seed {k}: crashed "
                          f"({self.crashes[k]}), tried again", flush=True)
                    continue
                self.valid[k] = "valid=True" in text
                last = [line for line in text.splitlines() if "joined:" in line]
                print(f"  {self.name} seed {self.seed}, annealer seed {k}: {last[-1].split('] ')[1] if last else 'no result'}",
                      flush=True)

    def stop_later_seeds(self):
        s = self.success()
        for k, (p, fh) in list(self.running.items()):
            if s is not None and k > s:
                p.kill()
                p.wait()
                fh.close()
                del self.running[k]
                os.remove(self.log(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[], help="random cubic graphs on n vertices")
    ap.add_argument("--gap", type=int, default=None, help="first corridor width (default: from the cut)")
    ap.add_argument("--max-gap", type=int, default=None, help="last corridor width (default: first + 16)")
    ap.add_argument("--graph-seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--s", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-seeds", type=int, default=40)
    a = ap.parse_args()
    # import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keeps a Windows laptop awake

    extra = (["--gap", str(a.gap)] if a.gap is not None else []) + \
            (["--max-gap", str(a.max_gap)] if a.max_gap is not None else [])
    graphs = [Graph(f"cubic{n}", gs, a.s, extra=extra) for gs in a.graph_seeds for n in a.n]
    t0 = time.time()
    while True:
        for g in graphs:
            g.poll()
            g.stop_later_seeds()
        busy = sum(g.busy() for g in graphs)
        launched = True
        while busy < a.workers and launched:        # one new attempt per graph per pass
            launched = False
            for g in graphs:
                preparing = sum(h.prep is not None for h in graphs)
                if busy < a.workers and not g.done(a.max_seeds) and g.can_start(a.max_seeds, preparing):
                    g.start()
                    busy += 1
                    launched = True
        if all(g.done(a.max_seeds) for g in graphs):
            break
        time.sleep(5)

    print(f"\nall done in {(time.time() - t0) / 3600:.1f} h", flush=True)
    for g in graphs:
        s = g.success()
        print(f"{g.name} seed {g.seed}: " + (f"first success at annealer seed {s}" if s
                                             else f"no success in {len(g.valid)} attempts"), flush=True)


if __name__ == "__main__":
    main()
