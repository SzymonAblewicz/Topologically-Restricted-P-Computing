"""
The 8 p-bit AND gate run forwards by Gibbs sampling: A and B are clamped to
each of the four inputs, and the probability of C = +1 and C = -1 is
averaged over 100 runs of 10^6 sweeps, then drawn as a bar chart. Also
prints the lowest-energy states, the local minima and the sampling
statistics.
"""
import numpy as np

# m: length-N vector of spins (bipolar entries), J: NxN, h: length-N bias vector.
def hamiltonian(m, J, h):
    m = np.asarray(m, dtype=float)
    return -0.5 * m @ J @ m - h @ m

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gibbs Sampling Function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gibbs_sample(J, h, num_sweeps, m0=None, clamp=None):
    """Gibbs sampling at beta = 1, num_sweeps sweeps over all spins from m0 (or a random state), with the bits in clamp held fixed."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)
    if m0 is None:
        m = np.random.choice([-1, 1], size=n) # (nothing clamped)
    else:
        m = np.array(m0, dtype=int)
    if clamp is None:
        clamp = {}

    samples = []
    energies = []
    for sweep in range(num_sweeps):
        # random sequential updates
        for i in np.random.permutation(n):
            if i in clamp:
                m[i] = clamp[i]
                continue

            # the input to spin i, I_i = J_i . m + h_i
            b = 1
            I_i = b * (J[i] @ m + h[i])

            # Gibbs p-bit update: P(m_i=+1) = (1/2)(1 + tanh(I_i)) = 1/(1 + exp(-2*I_i)) by CDF
            p_up = 1.0 / (1.0 + np.exp(-2.0 * I_i))

            if np.random.rand() < p_up:
                m[i] = 1
            else:
                m[i] = -1

        samples.append(m.copy())
        energies.append(hamiltonian(m, J, h))

    return np.array(samples), np.array(energies)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# J Derived From U
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

desired_truth_table = [tuple(row) for row in U]  # the desired truth table states are exactly the four rows of U.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The gate forwards: clamp A and B, read C
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

num_sweeps = 1_000_000

idx_D, idx_E, idx_F, idx_G, idx_Z, idx_A, idx_B, idx_C = range(8) # indices for [D, E, F, G, Z, A, B, C]
# Full AND gate truth table: (A, B) = (+1,+1), (+1,-1), (-1,+1), (-1,-1)
input_cases = [
    (-1, -1),
    (-1,  1),
    ( 1, -1),
    ( 1,  1),
]

num_runs = 100

results = {}
for a_val, b_val in input_cases:
    clamp = {idx_A: a_val, idx_B: b_val}

    p_plus_runs = []
    p_minus_runs = []
    mean_C_runs = []

    for _ in range(num_runs):
        samples, energies = gibbs_sample(J, h, num_sweeps, clamp=clamp)
        C_vals = samples[:, idx_C]
        p_plus_runs.append((C_vals == 1).mean())
        p_minus_runs.append((C_vals == -1).mean())
        mean_C_runs.append(C_vals.mean())

    results[(a_val, b_val)] = {
        "mean_C":    np.mean(mean_C_runs),
        "p_C_plus":  np.mean(p_plus_runs),
        "p_C_minus": np.mean(p_minus_runs),
    }

print("\n" + "=" * 72)
print("AND GATE OUTPUT STATISTICS")
print("=" * 72)

for (a_val, b_val), stats in results.items():
    print(f"\nInput clamp: A = {a_val:+d}, B = {b_val:+d}")
    print(f"Mean C           : {stats['mean_C']:+.4f}")
    print(f"P(C = +1)        : {stats['p_C_plus']:.4f}")
    print(f"P(C = -1)        : {stats['p_C_minus']:.4f}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import itertools

def E(s, J, h):
    s = np.asarray(s, dtype=float)
    return -0.5 * s @ J @ s - h @ s

def state_str(s):
    return "[" + " ".join(f"{int(x):2d}" for x in s) + "]"

def all_states(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=int)

def is_local_min(s, J, h):
    e = E(s, J, h)
    for i in range(len(s)):
        t = s.copy()
        t[i] *= -1
        if E(t, J, h) < e - 1e-12:
            return False
    return True

def analyze(J, h, samples, energies, desired_states, idx_C=None, top_k=10):
    desired = {tuple(map(int, s)) for s in desired_states}
    states = all_states(len(h))
    vals = np.array([E(s, J, h) for s in states])

    print("\n" + "=" * 72)
    print("LOWEST-ENERGY STATES")
    print("=" * 72)
    for r, i in enumerate(np.argsort(vals)[:top_k], 1):
        s = tuple(states[i])
        tag = "target" if s in desired else "spurious"
        print(f"{r:2d}. E = {vals[i]:7.3f}  {state_str(s)}  [{tag}]")

    print("\n" + "=" * 72)
    print("TRUTH-TABLE ENERGIES")
    print("=" * 72)
    desired_list = sorted(((s, E(s, J, h)) for s in desired), key=lambda x: x[1])
    for r, (s, e) in enumerate(desired_list, 1):
        print(f"{r:2d}. E = {e:7.3f}  {state_str(s)}")
    if desired_list:
        print(f"\nBest desired state: E = {desired_list[0][1]:.3f}  {state_str(desired_list[0][0])}")

    minima = [(tuple(s), e) for s, e in zip(states, vals) if is_local_min(s, J, h)]
    minima.sort(key=lambda x: x[1])

    print("\n" + "=" * 72)
    print("LOCAL MINIMA")
    print("=" * 72)
    for r, (s, e) in enumerate(minima[:top_k], 1):
        tag = "target" if s in desired else "spurious"
        print(f"{r:2d}. E = {e:7.3f}  {state_str(s)}  [{tag}]")

    spurious = [(s, e) for s, e in minima if s not in desired]
    print("\n" + "=" * 72)
    print("SPURIOUS MINIMA SUMMARY")
    print("=" * 72)
    print(f"Total local minima   : {len(minima)}")
    print(f"Desired local minima : {len(minima) - len(spurious)}")
    print(f"Spurious minima      : {len(spurious)}")
    for r, (s, e) in enumerate(spurious[:top_k], 1):
        print(f"{r:2d}. E = {e:7.3f}  {state_str(s)}")

    samples = np.asarray(samples, dtype=int)
    energies = np.asarray(energies, dtype=float)

    print("\n" + "=" * 72)
    print("SAMPLING STATISTICS")
    print("=" * 72)
    print(f"Samples   : {len(samples)}")
    print(f"Mean E    : {energies.mean():.3f}")
    print(f"Min E     : {energies.min():.3f}")
    print(f"Max E     : {energies.max():.3f}")
    print(f"Std(E)    : {energies.std():.3f}")

    uniq, counts = np.unique(samples, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    print("\nMost frequent sampled states:")
    for r, i in enumerate(order[:10], 1):
        print(f"{r:2d}. count = {counts[i]:5d}  p = {counts[i]/len(samples):.4f}  {state_str(uniq[i])}")

    print("\nPer-spin magnetization:")
    for i, m in enumerate(samples.mean(axis=0)):
        print(f"spin {i}: {m:+.3f}")

desired_states = [tuple(map(int, row)) for row in U]
analyze(J, h, samples, energies, desired_states, idx_C=idx_C)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convergence Probability Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.pyplot as plt

# colours
color_c_plus  = "#FF6361"   # pink  → C = +1
color_c_minus = "#3A4F8A"   # blue-purple → C = -1

x_labels = [f"A={a:+d}, B={b:+d}" for (a, b) in input_cases]
p_plus  = np.array([100 * results[(a, b)]["p_C_plus"]  for (a, b) in input_cases])
p_minus = np.array([100 * results[(a, b)]["p_C_minus"] for (a, b) in input_cases])
x = np.arange(len(input_cases))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

bars_plus  = ax.bar(x - width / 2, p_plus,  width, label="C = +1", color=color_c_plus,  alpha=0.85)
bars_minus = ax.bar(x + width / 2, p_minus, width, label="C = -1", color=color_c_minus, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylim(0, 100)
ax.set_ylabel("Probability (%)")
ax.set_xlabel("Clamped inputs")
ax.set_title("Conditional output probabilities for AND gate (A & B clamped)")
ax.legend(title="C spin")
ax.grid(axis="y", alpha=0.3)

for i in range(len(input_cases)):
    for bar, val in [(bars_plus[i], p_plus[i]), (bars_minus[i], p_minus[i])]:
        if val > 15:  # Only label bars with significant probability
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 0.85,
                f"{val:.1f}%",
                ha="center", va="center",
                fontsize=16, fontweight="normal", color="white",
            )

plt.tight_layout()
plt.show()