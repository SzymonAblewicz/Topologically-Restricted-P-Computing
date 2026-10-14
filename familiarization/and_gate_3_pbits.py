"""
The AND gate on 3 p-bits [A, B, C], from a hand-built J and h. All 2^3 states are
enumerated, to check that the four truth-table states have the lowest energy, and by how much.
"""
import itertools
import numpy as np

def hamiltonian(m, J, h):
    m = np.asarray(m, dtype=float)
    return -0.5 * m @ J @ m - h @ m

def analyze_energy_landscape(J, h, desired_states, top_k=50):
    """Every state's energy, the top_k lowest printed, and the gap between the truth-table states and the rest."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)

    # make sure J is correct
    if J.shape != (n, n):
        raise ValueError("J must be NxN with N = len(h).")
    if not np.allclose(J, J.T):
        raise ValueError("J must be symmetric.")

    desired_set = {tuple(s) for s in desired_states}

    all_results = []
    for state in itertools.product([-1, 1], repeat=n):
        state = tuple(state)
        E = hamiltonian(state, J, h)
        label = "DESIRED" if state in desired_set else "undesired"
        all_results.append((E, state, label))

    all_results.sort(key=lambda x: x[0])

    desired_results = [r for r in all_results if r[2] == "DESIRED"]
    undesired_results = [r for r in all_results if r[2] == "undesired"]

    print("\nDesired states:")
    for E, state, _ in desired_results:
        print(f"E = {E: .6f}, m = {state}")

    print("\nLowest undesired states:")
    for E, state, _ in undesired_results[:top_k]:
        print(f"E = {E: .6f}, m = {state}")

    print(f"\nLowest {top_k} states overall:")
    for rank, (E, state, label) in enumerate(all_results[:top_k], start=1):
        print(f"{rank:2d}. E = {E: .6f}, {label:9s}, m = {state}")

    if desired_results and undesired_results:
        best_desired = min(E for E, _, _ in desired_results)
        worst_desired = max(E for E, _, _ in desired_results)
        best_undesired = min(E for E, _, _ in undesired_results)

        print("\nEnergy summary:")
        print(f"Best desired energy:    {best_desired: .6f}")
        print(f"Worst desired energy:   {worst_desired: .6f}")
        print(f"Best undesired energy:  {best_undesired: .6f}")
        print(f"Energy gap:             {best_undesired - worst_desired: .6f}")

    return all_results


# The AND gate, [A, B, C]
J = np.array([
    [0, -1,  2],
    [-1, 0,  2],
    [2,  2,  0]
])

h = np.array([1, 1, -2])

desired_truth_table = [
    (-1, -1, -1),  # 0 AND 0 = 0
    (-1,  1, -1),  # 0 AND 1 = 0
    ( 1, -1, -1),  # 1 AND 0 = 0
    ( 1,  1,  1),  # 1 AND 1 = 1
]

analyze_energy_landscape(J, h, desired_truth_table, top_k=8)