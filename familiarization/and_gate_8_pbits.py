"""
Camsari's AND gate on 8 p-bits [D, E, F, G, Z, A, B, C], with J built from the
truth-table rows as J = 4 U^T U / L and a zero diagonal. The energy landscape over all
2^8 states is drawn on the hypercube, with the local minima under single spin flips and
the basin each state falls into.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt


# m: length-N vector of spins (bipolar entries), J: NxN, h: length-N bias vector.
def hamiltonian(m, J, h):
    m = np.asarray(m, dtype=float)
    return -0.5 * m @ J @ m - h @ m


def analyze_energy_landscape(J, h, desired_states, num_lowest_energy_states=50):  # prints lowest-energy states
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
        print(f"Energy gap:               {lowest_undesired - highest_desired: .6f}")  # (>0 good, <=0 bad)

    return all_results

# J from the truth-table rows, as Camsari does (trivial S)
def build_J_from_truth_rows(U):
    # U: one bipolar truth-table row u_k per row, mutually orthogonal (trivial S)
    U = np.asarray(U, dtype=float)
    n_rows, n_bits = U.shape
    # J_raw = (1/L) * sum_k u_k u_k^T = (1/L) * U^T U, with norm L = ||u_k||^2
    L = np.dot(U[0], U[0])
    Jeq = 4 * (U.T @ U) / L
    np.fill_diagonal(Jeq, 0.0)  # Zero diagonal
    return Jeq


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basins on the hypercube
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def iter_states(n):
    for state in itertools.product([-1, 1], repeat=n):
        yield tuple(state)


def local_minima_and_basins(J, h):
    """Every state's energy, the local minima under single spin flips, and the minimum each state reaches by steepest descent."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)

    energy_cache = {}
    for s in iter_states(n):
        energy_cache[s] = hamiltonian(s, J, h)

    # local minima under single spin flips
    minima = []
    for s, E in energy_cache.items():
        is_min = True
        for i in range(n):
            s_flipped = list(s)
            s_flipped[i] *= -1
            s_flipped = tuple(s_flipped)
            if energy_cache[s_flipped] < E - 1e-9:
                is_min = False
                break
        if is_min:
            minima.append(s)

    # basins by steepest descent
    basin_of = {}
    for s in energy_cache.keys():
        path = []
        current = s

        while current not in basin_of:
            path.append(current)

            if current in minima:
                for p in path:
                    basin_of[p] = current
                break

            E_curr = energy_cache[current]
            best_E = E_curr
            best_neighbor = current

            for i in range(n):
                s_flipped = list(current)
                s_flipped[i] *= -1
                s_flipped = tuple(s_flipped)
                E_nb = energy_cache[s_flipped]
                if E_nb < best_E - 1e-9:
                    best_E = E_nb
                    best_neighbor = s_flipped

            # already a minimum
            if best_neighbor == current:
                for p in path:
                    basin_of[p] = current
                break

            current = best_neighbor

        if current in basin_of and path:
            final_minimum = basin_of[current]
            for p in path:
                basin_of[p] = final_minimum

    return energy_cache, minima, basin_of


def hamming_weight(state):
    # Number of +1 spins
    return sum(1 for s in state if s == 1)


def compute_positions(energy_cache):
    """A 2D layout of the states, the Hamming weight across and the energy up."""
    states = list(energy_cache.keys())

    layers = {}
    for s in states:
        w = hamming_weight(s)
        layers.setdefault(w, []).append(s)

    pos = {}
    for w, layer_states in layers.items():
        k = len(layer_states)
        for j, s in enumerate(sorted(layer_states)):
            x = w + 0.8 * ((j + 0.5) / k - 0.5)
            y = energy_cache[s]
            pos[s] = (x, y)

    return pos


def print_basin_summary(energy_cache, minima, basin_of, desired_states=None):
    desired_set = {tuple(s) for s in desired_states} if desired_states is not None else set()

    basin_sizes = {m: 0 for m in minima}
    basin_desired_counts = {m: 0 for m in minima}

    for s, m in basin_of.items():
        basin_sizes[m] += 1
        if s in desired_set:
            basin_desired_counts[m] += 1

    print("\nLocal minima and basin sizes:")
    rows = []
    for m in minima:
        rows.append((energy_cache[m], basin_sizes[m], basin_desired_counts[m], m))
    rows.sort(key=lambda x: (x[0], -x[1]))

    for rank, (E, size, n_desired, m) in enumerate(rows, start=1):
        label = "DESIRED" if m in desired_set else "undesired"
        print(f"{rank:2d}. E = {E: .6f}, basin size = {size:3d}, desired in basin = {n_desired:2d}, {label:9s}, minimum = {m}")


def plot_energy_landscape_with_basins(J, h, desired_states=None):
    """The states on the hypercube by energy, coloured by basin, with the truth-table states highlighted and the local minima circled."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    desired_set = {tuple(s) for s in desired_states} if desired_states is not None else set()

    energy_cache, minima, basin_of = local_minima_and_basins(J, h)
    pos = compute_positions(energy_cache)

    minima_list = list(minima)
    basin_index = {m: i for i, m in enumerate(minima_list)}

    xs = []
    ys = []
    colors = []
    sizes = []
    desired_mask = []

    for s, E in energy_cache.items():
        x, y = pos[s]
        xs.append(x)
        ys.append(y)
        colors.append(basin_index[basin_of[s]])
        sizes.append(44 if s in minima else 22)
        desired_mask.append(s in desired_set)

    xs = np.array(xs)
    ys = np.array(ys)
    colors = np.array(colors)
    sizes = np.array(sizes)
    desired_mask = np.array(desired_mask, dtype=bool)

    # hypercube edges, between states one spin apart
    states = list(energy_cache.keys())
    state_index = {s: i for i, s in enumerate(states)}
    edges_x = []
    edges_y = []

    for s in states:
        for i in range(len(s)):
            s_flipped = list(s)
            s_flipped[i] *= -1
            s_flipped = tuple(s_flipped)
            if state_index[s] < state_index[s_flipped]:
                x1, y1 = pos[s]
                x2, y2 = pos[s_flipped]
                edges_x.extend([x1, x2, np.nan])
                edges_y.extend([y1, y2, np.nan])

    plt.figure(figsize=(14, 7))

    plt.plot(edges_x, edges_y, color="0.82", linewidth=0.55, zorder=1)

    # the other states
    plt.scatter(
        xs[~desired_mask], ys[~desired_mask],
        c=colors[~desired_mask], cmap="tab20",
        s=sizes[~desired_mask], alpha=0.90,
        label="undesired", zorder=2
    )

    # the truth-table states
    plt.scatter(
        xs[desired_mask], ys[desired_mask],
        c=colors[desired_mask], cmap="tab20",
        s=sizes[desired_mask] * 1.35,
        edgecolors="k", linewidths=0.8,
        label="desired", zorder=3
    )

    # the local minima
    min_x = [pos[m][0] for m in minima]
    min_y = [pos[m][1] for m in minima]
    plt.scatter(
        min_x, min_y,
        facecolors="none", edgecolors="k",
        s=150, linewidths=1.6,
        label="local minima", zorder=4
    )

    plt.xlabel("Hamming weight (number of +1 spins)")
    plt.ylabel("Energy")
    plt.title("Accurate energy landscape and basin structure on the hypercube")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return energy_cache, minima, basin_of


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The energy against the state number, for comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # local minima in the 1D ordering only
    minima_idx = []
    for i in range(1, len(ys) - 1):
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            minima_idx.append(i)

    plt.figure(figsize=(14, 5))
    plt.plot(xs, ys, linewidth=1.5, color="0.35", zorder=1)
    plt.scatter(xs[~desired_mask], ys[~desired_mask], s=18, color="0.55", label="undesired", zorder=2)
    plt.scatter(xs[desired_mask], ys[desired_mask], s=45, color="C3", label="desired", zorder=3)
    plt.scatter(xs[minima_idx], ys[minima_idx], s=80, facecolors="none", edgecolors="C0",
                linewidths=2, label="local minima in 1D ordering", zorder=4)
    plt.xlabel("state number")
    plt.ylabel("energy")
    plt.title("Energy landscape ordered from all -1 spins to all +1 spins")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return xs, ys, minima_idx


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The AND gate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     [ D   E   F   G   Z   A   B   C]
u1 = [-1,  1,  1,  1,  1, -1, -1, -1]
u2 = [ 1, -1, -1,  1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]

# construct U where each row is a truth-table state
U = np.array([u1, u2, u3, u4], dtype=float)
J = build_J_from_truth_rows(U)

np.set_printoptions(precision=3, suppress=True)
print("coupling matrix J:")
print(J)

h = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)  # biases for D, E, F, G, Z, A, B, C

desired_truth_table = [tuple(row) for row in U]

# the energies, as text
analyze_energy_landscape(J, h, desired_truth_table, num_lowest_energy_states=8)

# the plot by state number: a rough scan, not the basins
plot_energy_landscape_by_state_number(J, h, desired_truth_table)

# the basins under single-spin-flip steepest descent, on the hypercube
energy_cache, minima, basin_of = plot_energy_landscape_with_basins(J, h, desired_truth_table)

# the basins, as text
print_basin_summary(energy_cache, minima, basin_of, desired_truth_table)