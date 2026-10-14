"""
The fallback method on the Erdos-Renyi graphs of Section 4.4 of the report, generated with
mean degree 3 and reduced to the giant component, of 10,000, 30,000, 100,000 and 1,000,000
vertices at s = 50, each by run_fallback.py with --checkpoint.

    python -B run_erdos_renyi.py --workers 8
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = os.path.join(ROOT, "src", "fallback", "run_fallback.py")
LOG = os.path.join(ROOT, "results", "section4_fallback", "log_erdos_renyi.txt")
GRAPHS = [(10630, 1), (31950, 1), (106301, 4), (1063440, 50)]   # (n requested, seed): giant components of exactly 10k, 30k, 100k, 1M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for n_req, seed in GRAPHS:
        with open(LOG, "a", encoding="utf-8") as fh:
            subprocess.call([sys.executable, "-B", "-u", RUN, "--family", "erdos_renyi", "--n", str(n_req),
                             "--seed", str(seed), "--workers", str(a.workers), "--checkpoint",
                             "--out", "runs_erdos_renyi.csv"], stdout=fh, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
