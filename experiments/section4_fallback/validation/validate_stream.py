"""
End-to-end validation of a finished fallback run at any size. The embedding is exported
(export_embedding.py, checked against the construction itself), then swept by stream_validate
over every hardware vertex, for (M1)-(M3). One row per run in validate_stream.csv.

    python -B validate_stream.py <family> <n_request> <seed> <s> [lanes to check, default all]
                                 [--export-only | --validate-only]

--export-only stops after the export, and --validate-only waits (up to 8 h) for it, so big
exports can run one at a time while their validations run side by side.
"""
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, "results", "section4_fallback")
EXE = os.path.join(HERE, "stream_validate" + (".exe" if os.name == "nt" else ""))   # python build.py makes it
FIELDS = ["family", "n_request", "seed", "s", "n", "m", "T", "L_plan", "L_final", "valid", "cells",
          "conflicts", "self_dup", "out_of_bounds", "empty_chains", "split_chains", "edges_missing",
          "lanes_checked", "lane_mismatches", "full_chains_compared", "full_chains_differ",
          "t_export", "t_validate", "first_error", "error"]


def main():
    family, n_req, seed, s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sample = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else None
    tag = f"{family}_{n_req}_{seed}_s{s}"
    out = os.path.join(RES, "export", tag)
    row = dict(family=family, n_request=n_req, seed=seed, s=s, error="")
    t0 = time.perf_counter()
    done_flag = os.path.join(out, "EXPORT_DONE")
    if "--validate-only" in sys.argv:
        while not os.path.exists(done_flag) and time.perf_counter() - t0 < 8 * 3600:
            print(time.strftime("%H:%M:%S ") + f"waiting for the export of {tag}", flush=True)
            time.sleep(120)
        rc = 0 if os.path.exists(done_flag) else 1
        row["t_export"] = open(done_flag).read().strip() if rc == 0 else ""
    else:
        if os.path.exists(done_flag):
            os.remove(done_flag)
        embs = os.path.join(RES, "ckpt", tag + "_embs.pkl")
        while os.path.exists(embs) and time.time() - os.path.getmtime(embs) < 300:
            print(time.strftime("%H:%M:%S ") + f"{tag}: run still writing its checkpoint, waiting", flush=True)
            time.sleep(60)
        cmd = [sys.executable, "-B", "export_embedding.py", family, n_req, seed, s, "--check"]
        if sample:
            cmd += ["--sample", sample]
        if int(n_req) <= 11000:
            cmd += ["--full"]                     # the whole construction too (small runs only)
        rc = subprocess.call(cmd, cwd=HERE)
        row["t_export"] = round(time.perf_counter() - t0, 1)
        if rc == 0:
            with open(done_flag, "w") as fh:
                fh.write(str(row["t_export"]) + "\n")
        if "--export-only" in sys.argv:
            print(f"{tag}: export exit {rc} in {row['t_export']} s", flush=True)
            sys.exit(rc)
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    row.update(n=meta.get("n"), m=meta.get("m"), T=meta.get("T"), lanes_checked=meta.get("check_lanes"),
               lane_mismatches=meta.get("check_mismatch"), full_chains_compared=meta.get("full_chains"),
               full_chains_differ=meta.get("full_differ"))
    if rc != 0:
        row["error"] = f"export exit {rc}"
    else:
        t1 = time.perf_counter()
        p = subprocess.run([EXE, out], capture_output=True, text=True)
        row["t_validate"] = round(time.perf_counter() - t1, 1)
        try:
            v = json.loads(p.stdout)
            for k in ("L_plan", "L_final", "cells", "conflicts", "self_dup", "out_of_bounds", "empty_chains",
                      "split_chains", "edges_missing", "first_error"):
                row[k] = v[k]
            row["valid"] = str(v["valid"])
        except ValueError:
            row["error"] = f"validator exit {p.returncode}: {p.stderr[-150:]}"
        print(p.stdout.strip(), flush=True)
    path = os.path.join(RES, "validate_stream.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)
    sys.exit(0 if row.get("valid") == "True" and not row["error"] else 1)


if __name__ == "__main__":
    main()
