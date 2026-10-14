"""
Which auxiliary bits make the AND gate's truth-table rows orthogonal: every
choice of 5 auxiliary columns (16^5) is added to the 4 x 3 bipolar truth
table, and the 4 x 8 matrices whose rows are mutually orthogonal are counted.
"""
from itertools import product

# the AND truth table, bipolar (0 -> -1)
BASE = [
    [-1, -1, -1],   # 00
    [-1,  1, -1],   # 01
    [ 1, -1, -1],   # 10
    [ 1,  1,  1],   # 11
]

ROWS = range(4)

PAIRS = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]

# All possible bipolar columns
ALL_COLS = [
    tuple(v)
    for v in product([-1, 1], repeat=4)
]

def pair_contrib(col):
    """One column's contribution to the dot products of every pair of rows."""
    return tuple(
        col[i] * col[j]
        for i, j in PAIRS
    )

def base_pair_sums():
    """The dot products of every pair of rows of BASE."""
    sums = []

    for i, j in PAIRS:
        s = sum(
            BASE[i][k] * BASE[j][k]
            for k in range(len(BASE[0]))
        )
        sums.append(s)

    return tuple(sums)

BASE_SUMS = base_pair_sums()

def search_5_aux_columns(limit=None):
    """Every choice of 5 auxiliary columns (16^5), keeping the 4 x 8 matrices whose rows are mutually orthogonal."""

    solutions = []
    total_count = 0

    for cols in product(ALL_COLS, repeat=5):

        totals = list(BASE_SUMS)

        for c in cols:

            pc = pair_contrib(c)

            for k in range(6):
                totals[k] += pc[k]

        if all(t == 0 for t in totals):

            total_count += 1

            M = [
                [c[r] for c in cols] + BASE[r]
                for r in ROWS
            ]

            if limit is None or len(solutions) < limit:
                solutions.append(M)

    return solutions, total_count

def print_matrix(M):
    labels = ["u1", "u2", "u3", "u4"]
    header = "#    [ D   E   F   G   Z   A   B   C]"
    print(header)
    for label, row in zip(labels, M):
        row_str = ", ".join(f"{x:2d}" for x in row)
        print(f"{label} = [{row_str}]")

if __name__ == "__main__":

    sols, total = search_5_aux_columns(limit=5)

    print(f"\nTOTAL SOLUTIONS = {total}")
    print(f"Showing first {len(sols)} solutions.\n")

    for idx, M in enumerate(sols, start=1):

        print(f"Solution {idx}:")

        print_matrix(M)

        print()