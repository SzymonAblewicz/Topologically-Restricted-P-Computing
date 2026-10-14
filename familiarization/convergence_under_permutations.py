"""
Does the order of the columns of U change how well the AND gate works?
For 100 random column permutations, J is rebuilt and the gate run forwards
by Gibbs sampling (A and B clamped, C read), and the permutations are ranked
by how often C comes out right.
"""
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Energy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def hamiltonian(m, J, h):
    m = np.asarray(m, dtype=float)
    return -0.5 * m @ J @ m - h @ m


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gibbs Sampling Function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gibbs_sample(J, h, num_sweeps, rng, m0=None, clamp=None):
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)

    if m0 is None:
        m = rng.choice([-1, 1], size=n)
    else:
        m = np.array(m0, dtype=int)

    if clamp is None:
        clamp = {}

    samples = []
    energies = []

    for _ in range(num_sweeps):
        for i in rng.permutation(n):
            if i in clamp:
                m[i] = clamp[i]
                continue

            I_i = J[i] @ m + h[i]
            p_up = 1.0 / (1.0 + np.exp(-2.0 * I_i))
            m[i] = 1 if rng.random() < p_up else -1

        samples.append(m.copy())
        energies.append(hamiltonian(m, J, h))

    return np.array(samples), np.array(energies)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# J Derived From U
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_J_from_truth_rows(U):
    U = np.asarray(U, dtype=float)
    L = np.dot(U[0], U[0])
    Jeq = 4 * (U.T @ U) / L
    np.fill_diagonal(Jeq, 0.0)
    return Jeq


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Original AND gate truth-table rows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ D  E  F  G  Z  A  B  C ]
u1 = [-1, -1, -1,  1,  1, -1, -1, -1]
u2 = [-1,  1,  1, -1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [-1, -1, -1, -1,  1,  1,  1,  1]

U_base = np.array([u1, u2, u3, u4], dtype=float)
labels_base = np.array(["D", "E", "F", "G", "Z", "A", "B", "C"])
h_base = np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=float)

input_cases = [
    (-1, -1),
    (-1,  1),
    ( 1, -1),
    ( 1,  1),
]

# AND truth table in bipolar form for C
truth_C = {
    (-1, -1): -1,
    (-1,  1): -1,
    ( 1, -1): -1,
    ( 1,  1):  1,
}

num_sweeps = 1000
master_rng = np.random.default_rng()


def evaluate_one_permutation(U_base, labels_base, h_base, perm, num_sweeps, seed):
    rng = np.random.default_rng(seed)

    # Permute columns of U, and permute labels/biases the same way
    U_perm = U_base[:, perm]
    labels_perm = labels_base[perm]
    h_perm = h_base[perm]

    J_perm = build_J_from_truth_rows(U_perm)
    idx = {name: i for i, name in enumerate(labels_perm)}

    results = {}

    for a_val, b_val in input_cases:
        clamp = {
            idx["A"]: a_val,
            idx["B"]: b_val,
        }

        samples, energies = gibbs_sample(
            J_perm, h_perm, num_sweeps, rng=rng, clamp=clamp
        )

        C_vals = samples[:, idx["C"]]
        target_C = truth_C[(a_val, b_val)]

        p_C_plus = (C_vals == 1).mean()
        p_C_minus = (C_vals == -1).mean()
        p_correct = (C_vals == target_C).mean()

        results[(a_val, b_val)] = {
            "target_C": target_C,
            "mean_C": C_vals.mean(),
            "p_C_plus": p_C_plus,
            "p_C_minus": p_C_minus,
            "p_correct": p_correct,
        }

    avg_convergence = np.mean([results[case]["p_correct"] for case in input_cases])

    return {
        "perm_indices": tuple(int(x) for x in perm),
        "labels_perm": labels_perm.tolist(),
        "idx": idx,
        "results": results,
        "avg_convergence": avg_convergence,
    }


from itertools import permutations

all_perms = np.array(list(permutations(range(U_base.shape[1]))), dtype=int)
chosen_idx = master_rng.choice(len(all_perms), size=100, replace=False)
perms = [all_perms[i] for i in chosen_idx]

# a seed per permutation
seeds = master_rng.integers(0, 2**32 - 1, size=100)

all_results = []
for perm, seed in zip(perms, seeds):
    record = evaluate_one_permutation(
        U_base, labels_base, h_base, perm, num_sweeps, seed
    )
    all_results.append(record)

# best first
all_results = sorted(all_results, key=lambda r: r["avg_convergence"], reverse=True)

print("\n" + "=" * 110)
print("100 RANDOM COLUMN PERMUTATIONS OF U, RANKED BY AVERAGE CONVERGENCE (HIGHEST TO LOWEST)")
print("=" * 110)

ranks_to_show = [1, 2, 3, 4, 5, 99, 100]

for rank in ranks_to_show:
    record = all_results[rank - 1]

    print(f"\nRank {rank:02d}")
    print(f"Permutation indices : {record['perm_indices']}")
    print(f"Column order        : {record['labels_perm']}")
    print(
        f"Bit positions       : "
        f"A@{record['idx']['A']}  B@{record['idx']['B']}  C@{record['idx']['C']}  Z@{record['idx']['Z']}"
    )
    print(f"Average convergence : {100 * record['avg_convergence']:.2f}%")

    for a_val, b_val in input_cases:
        stats = record["results"][(a_val, b_val)]
        print(
            f"  Input (A={a_val:+d}, B={b_val:+d}) | "
            f"target C={stats['target_C']:+d} | "
            f"convergence={100 * stats['p_correct']:.2f}% | "
            f"P(C=+1)={100 * stats['p_C_plus']:.2f}% | "
            f"P(C=-1)={100 * stats['p_C_minus']:.2f}% | "
            f"mean C={stats['mean_C']:+.4f}"
        )

print("\n" + "=" * 110)