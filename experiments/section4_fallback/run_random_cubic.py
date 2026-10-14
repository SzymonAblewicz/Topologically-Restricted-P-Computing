"""
The fallback method on the random cubic graphs (seed 1) of Figure 17 of the report, from
1,000 to 1,000,000 vertices at s = 50, each by run_fallback.py with --checkpoint, so a
stopped run resumes where it was.

    python -B run_random_cubic.py 1000 3000 10000 30000 100000 300000 1000000 --workers 8
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = os.path.join(ROOT, "src", "fallback", "run_fallback.py")
LOG = os.path.join(ROOT, "results", "section4_fallback", "log_random_cubic.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    # import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)  # keeps a Windows laptop awake
    for n in a.n:
        with open(LOG, "a", encoding="utf-8") as fh:
            subprocess.call([sys.executable, "-B", "-u", RUN, "--n", str(n), "--workers", str(a.workers),
                             "--checkpoint", "--out", "runs_random_cubic.csv"], stdout=fh, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
