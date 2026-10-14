"""
The comparison of Divide and Conquer and the fallback method on the same random cubic
graphs (Section 5.1 of the report), with the complete graph embedding on KG(n-1) as the
reference. A graph's Divide and Conquer entry is its first seed whose embedding is
complete and valid, read from the logs, and the area is (L / (n - 1))^2.

    python dnc_vs_fallback.py     # prints the tables, writes results/section3_divide_and_conquer/table.md
"""
import csv
import glob
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "section5_discussion")
DNC = os.path.join(ROOT, "results", "section3_divide_and_conquer")


def attempts(folder):
    """{annealer seed: (valid, L, [(width, L, unrouted, seconds), ...])} from the logs."""
    out = {}
    for path in glob.glob(os.path.join(folder, "log_sa*.txt")):
        text = open(path).read()
        joined = re.search(r"joined: L=(\d+) valid=(\w+)", text)
        if not joined:
            continue                                     # stopped or crashed
        widths = [tuple(map(int, m)) for m in re.findall(r"corridor width (\d+): L=(\d+), (\d+) unrouted, (\d+) s", text)]
        out[int(path[path.rindex("_sa") + 3:-4])] = (joined.group(2) == "True", int(joined.group(1)), widths)
    return out


def partition_seconds(folder):
    for path in [os.path.join(folder, "log_prep.txt")] + glob.glob(os.path.join(folder, "log_sa*.txt")):
        if os.path.exists(path):
            m = re.search(r"partition: k=\d+ cut=\d+ T=\d+ \(\w+, ([\d.]+) s\)", open(path).read())
            if m:
                return float(m.group(1))
    return None


def dnc_entries():
    """{(n, graph seed): entry} for every graph tried."""
    entries = {}
    for folder in glob.glob(os.path.join(DNC, "cubic*_s30_seed*")):
        n, seed = map(int, re.search(r"cubic(\d+)_s30_seed(\d+)", folder).groups())
        att = attempts(folder)
        first = next((k for k in sorted(att) if att[k][0]), None)
        if first is not None and not all(k in att for k in range(1, first)):
            first = None                                 # an earlier seed has no result
        entries[(n, seed)] = dict(
            attempts=len(att), success=first, L=att[first][1] if first else None,
            best_short=min((w[2] for k in att for w in att[k][2]), default=None),
            sec_per_width=[w[3] for k in att for w in att[k][2]],
            t_partition=partition_seconds(folder))
    return entries


def fallback_entries(path, s):
    out = {}
    for r in csv.DictReader(open(os.path.join(RES, path))):
        L = r["L_final"] or r["L_plan"]
        if L and r["family"] == "random_cubic" and r["s"] == str(s):
            out[(int(r["n"]), int(r["seed"]))] = dict(
                L=int(L), T=int(r["T"]), validated=r["valid"] == "True",
                seconds=sum(float(r[f] or 0) for f in ("t_stage1", "t_stage2_wall", "t_stage3_plan")))
    return out


def area(L, n):
    return (L / (n - 1)) ** 2


def spread(values, fmt="{:.0f}"):
    """median (min-max), or the single value."""
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.median(values))} ({fmt.format(min(values))}-{fmt.format(max(values))})"


def main():
    dnc, fb = dnc_entries(), fallback_entries("fallback_s30.csv", 30)
    fb50 = fallback_entries("fallback_s50_s40.csv", 50)
    out = []

    out += ["## Divide and Conquer against the fallback (random cubic graphs, s = 30)", "",
            "Median over the graphs (range in brackets); Graphs gives how many.",
            "Area is (L / (n - 1))^2 against the complete graph embedding.", "",
            "| n | Graphs | D&C embedded | D&C side L | D&C seeds needed | Fallback side L | Complete graph | D&C area | Fallback area |",
            "|---|---|---|---|---|---|---|---|---|"]
    ns = sorted({n for n, _ in dnc} | {n for n, _ in fb})
    for n in ns:
        ds = [e for (m, _), e in dnc.items() if m == n]
        fs = [e for (m, _), e in fb.items() if m == n]
        ok = [e for e in ds if e["success"]]
        g = max(len(ds), len(fs))
        emb = f"{len(ok)}/{len(ds)}" if ds else "-"
        dL = spread([e["L"] for e in ok]) if ok else ("none" if ds else "-")
        seeds = spread([e["success"] for e in ok]) if ok else "-"
        if ds and len(ok) < len(ds):
            fails = [e for e in ds if not e["success"]]
            seeds += f"; failed: {', '.join(str(e['attempts']) for e in fails)} seeds"
        fL = spread([e["L"] for e in fs]) if fs else "-"
        da = spread([area(e["L"], n) for e in ok], "{:.2f}") if ok else "-"
        fa = spread([area(e["L"], n) for e in fs], "{:.2f}") if fs else "-"
        out.append(f"| {n:,} | {g} | {emb} | {dL} | {seeds} | {fL} | {n - 1} | {da} | {fa} |")

    out += ["", "## The fallback at the largest n (one graph each)", "",
            "| n | s | T/n | Side L | Floor (T/(n-1))^2 | Area | Times smaller than complete | Above the floor |",
            "|---|---|---|---|---|---|---|---|"]
    for (n, _), e in sorted(list((k, v) for k, v in fb.items() if k[0] >= 10000 and k[1] == 1)):
        out.append(_big(n, 30, e))
    for (n, _), e in sorted((k, v) for k, v in fb50.items() if k[0] >= 10000 and k[1] == 1):
        out.append(_big(n, 50, e))

    out += ["", "## Time (ImprovedDnC_2; the fallback on the same graphs)", "",
            "| n | D&C partition (s) | D&C one joining width (s) | Fallback, all stages (s) |",
            "|---|---|---|---|"]
    for n in ns:
        ds = [e for (m, _), e in dnc.items() if m == n]
        fs = [e for (m, _), e in fb.items() if m == n]
        tp = [e["t_partition"] for e in ds if e["t_partition"] is not None]
        tw = [t for e in ds for t in e["sec_per_width"]]
        out.append(f"| {n:,} | {spread(tp) if tp else '-'} | {statistics.median(tw):.0f} | "
                   f"{spread([e['seconds'] for e in fs]) if fs else '-'} |" if tw else
                   f"| {n:,} | - | - | {spread([e['seconds'] for e in fs]) if fs else '-'} |")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(DNC, "table.md"), "w") as fh:
        fh.write(text + "\n")


def _big(n, s, e):
    floor, a = (e["T"] / (n - 1)) ** 2, area(e["L"], n)
    return (f"| {n:,} | {s} | {e['T'] / n:.3f} | {e['L']:,} | {floor:.3f} | {a:.3f} | "
            f"{1 / a:.2f}x | {a / floor - 1:.1%} |")


if __name__ == "__main__":
    main()
