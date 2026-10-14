"""
An earlier version of and_gate_8_pbits.py: the same 8 p-bit AND gate and the
energies of all 2^8 states, with the plot by state number only.
"""
import itertools
import numpy as np

# m: length-N vector of spins (bipolar entries), J: NxN, h: length-N bias vector.
def hamiltonian(m, J, h):
    m = np.asarray(m, dtype=float)
    return -0.5 * m @ J @ m - h @ m

def analyze_energy_landscape(J, h, desired_states, num_lowest_energy_states=50): #(prints 50 lowest energy states)
    # every state's energy, and where the truth-table states sit among the rest

    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)

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
    for E, state, _ in undesired_results[:num_lowest_energy_states]:
        print(f"E = {E: .6f}, m = {state}")

    print(f"\nLowest {num_lowest_energy_states} states overall:")
    for rank, (E, state, label) in enumerate(all_results[:num_lowest_energy_states], start=1):
        # (1 = absolute minimum energy and state)
        print(f"{rank:2d}. E = {E: .6f}, {label:9s}, m = {state}")

    # the gap, if there are states of both kinds
    if desired_results and undesired_results:
        lowest_desired = min(E for E, _, _ in desired_results)
        highest_desired = max(E for E, _, _ in desired_results)
        lowest_undesired = min(E for E, _, _ in undesired_results)
        print("\nEnergy summary:")
        print(f"Lowest desired energy:    {lowest_desired: .6f}")
        print(f"Highest desired energy:   {highest_desired: .6f}")
        print(f"Lowest undesired energy:  {lowest_undesired: .6f}")
        print(f"Energy gap:             {lowest_undesired - highest_desired: .6f}") #(>0 good, <=0 bad)

    return all_results


# J from the truth-table rows, as Camsari does (trivial S)
def build_J_from_truth_rows(U):
    # U: one bipolar truth-table row u_k per row, mutually orthogonal (trivial S)
    U = np.asarray(U, dtype=float)
    n_rows, n_bits = U.shape
    # J_raw = (1/L) * sum_k u_k u_k^T = (1/L) * U^T U, with norm L = ||u_k||^2
    L = np.dot(U[0], U[0])
    Jeq = 4 * (U.T @ U) / L
    np.fill_diagonal(Jeq, 0.0) # Zero diagonal
    return Jeq

# AND gate
#    [ D   E   F   G   Z   A   B   C]
u1 = [-1,  1,  1,  1,  1, -1, -1, -1]
u2 = [ 1, -1, -1,  1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]
'''
#    [ D   E   F   G   Z   A   B   C] Camsari truth table
u1 = [-1,  1,  1,  1,  1, -1, -1, -1]
u2 = [ 1, -1, -1,  1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]
'''
# U, one truth-table row per row
U = np.array([u1, u2, u3, u4], dtype=float)
J = build_J_from_truth_rows(U)
np.set_printoptions(precision=3, suppress=True)  # 3 decimal places and no scientific notation
print("coupling matrix J:")
print(J)

h = np.array([0, 0, 0, 0, 2, 0, 0, 0])  # biases for D, E, F, G, Z, A, B, C

desired_truth_table = [tuple(row) for row in U] #(the desired truth table states are exactly the four rows of U.)

analyze_energy_landscape(J, h, desired_truth_table, num_lowest_energy_states=8)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The energy against the state number, from all -1 to all +1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.pyplot as plt
import numpy as np
import itertools
def state_to_number(state):
    # Maps (-1, -1, ..., -1) -> 0 and (1, 1, ..., 1) -> 2^n - 1
    # First spin is the most significant bit.
    bits = [(1 if s == 1 else 0) for s in state]
    idx = 0
    n = len(bits)
    for i, b in enumerate(bits):
        idx += b << (n - 1 - i)
    return idx
def plot_energy_landscape_by_state_number(J, h, desired_states=None):
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)
    desired_set = {tuple(s) for s in desired_states} if desired_states is not None else set()
    xs = []
    ys = []
    desired_mask = []
    for state in itertools.product([-1, 1], repeat=n):
        state = tuple(state)
        idx = state_to_number(state)
        E = hamiltonian(state, J, h)

        xs.append(idx)
        ys.append(E)
        desired_mask.append(state in desired_set)
    xs = np.array(xs)
    ys = np.array(ys)
    desired_mask = np.array(desired_mask, dtype=bool)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    desired_mask = desired_mask[order]
    #local minima in the 1D ordering
    minima_idx = []
    for i in range(1, len(ys) - 1):
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            minima_idx.append(i)
    plt.figure(figsize=(14, 5))
    plt.plot(xs, ys, linewidth=1.5, color="0.35", zorder=1)
    plt.scatter(xs[~desired_mask], ys[~desired_mask], s=18, color="0.55", label="undesired", zorder=2)
    plt.scatter(xs[desired_mask], ys[desired_mask], s=45, color="C3", label="desired", zorder=3)
    plt.scatter(xs[minima_idx], ys[minima_idx], s=80, facecolors="none", edgecolors="C0",
                linewidths=2, label="local minima", zorder=4)
    plt.xlabel("state number")
    plt.ylabel("energy")
    plt.title("Energy landscape ordered from all -1 spins to all +1 spins")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return xs, ys, minima_idx
plot_energy_landscape_by_state_number(J, h, desired_truth_table)