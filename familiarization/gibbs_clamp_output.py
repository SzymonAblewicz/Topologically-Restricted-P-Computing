"""
The 8 p-bit AND gate run backwards by Gibbs sampling: C is clamped to -1 and
+1, and the probability of each input pair (A, B) is averaged over 100 runs
of 10^6 sweeps.
"""
import numpy as np

def hamiltonian(m, J, h):
    m = np.asarray(m, dtype=float)
    return -0.5 * m @ J @ m - h @ m

def gibbs_sample(J, h, num_sweeps, m0=None, clamp=None):
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)
    if m0 is None:
        m = np.random.choice([-1, 1], size=n)
    else:
        m = np.array(m0, dtype=int)
    if clamp is None:
        clamp = {}

    samples = []
    energies = []
    for sweep in range(num_sweeps):
        for i in np.random.permutation(n):
            if i in clamp:
                m[i] = clamp[i]
                continue
            b = 1
            I_i = b * (J[i] @ m + h[i])
            p_up = 1.0 / (1.0 + np.exp(-2.0 * I_i))
            if np.random.rand() < p_up:
                m[i] = 1
            else:
                m[i] = -1
        samples.append(m.copy())
        energies.append(hamiltonian(m, J, h))

    return np.array(samples), np.array(energies)

def build_J_from_truth_rows(U):
    U = np.asarray(U, dtype=float)
    n_rows, n_bits = U.shape
    L = np.dot(U[0], U[0])
    Jeq = 4 * (U.T @ U) / L
    np.fill_diagonal(Jeq, 0.0)
    return Jeq

#    [ D   E   F   G   Z   A   B   C]
u1 = [-1,  1,  1,  1,  1, -1, -1, -1]
u2 = [ 1, -1, -1,  1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]

U = np.array([u1, u2, u3, u4], dtype=float)
J = build_J_from_truth_rows(U)
np.set_printoptions(precision=3, suppress=True)
print("coupling matrix J:")
print(J)

h = np.array([0, 0, 0, 0, 2, 0, 0, 0])

desired_truth_table = [tuple(row) for row in U]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The gate backwards: clamp C, read A and B
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

num_sweeps = 1_000_000

idx_D, idx_E, idx_F, idx_G, idx_Z, idx_A, idx_B, idx_C = range(8)

input_cases = [-1, 1]  # C clamped values

ab_pairs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

num_runs = 100

results = {}
for c_val in input_cases:
    clamp = {idx_C: c_val}

    pair_probs_runs = {pair: [] for pair in ab_pairs}

    for _ in range(num_runs):
        samples, energies = gibbs_sample(J, h, num_sweeps, clamp=clamp)
        A_vals = samples[:, idx_A]
        B_vals = samples[:, idx_B]
        for pair in ab_pairs:
            a, b = pair
            pair_probs_runs[pair].append(((A_vals == a) & (B_vals == b)).mean())

    results[c_val] = {pair: np.mean(pair_probs_runs[pair]) for pair in ab_pairs}

print("\n" + "=" * 72)
print("AND GATE OUTPUT STATISTICS")
print("=" * 72)

for c_val in input_cases:
    print(f"\nInput clamp: C = {c_val:+d}")
    for pair, prob in results[c_val].items():
        a, b = pair
        print(f"  P(A={a:+d}, B={b:+d}) : {prob:.4f}")

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

x_labels = [f"C = {c_val:+d}" for c_val in input_cases]
x = np.arange(len(input_cases))
n_pairs = len(ab_pairs)
width = 0.18
pair_colors = ["#003F5C", "#2F4B7C", "#665191", "#FF6361"]

fig, ax = plt.subplots(figsize=(10, 5))

for j, pair in enumerate(ab_pairs):
    a, b = pair
    probs = np.array([100 * results[c_val][pair] for c_val in input_cases])
    offset = (j - (n_pairs - 1) / 2) * width
    bars = ax.bar(x + offset, probs, width,
                  label=f"A={a:+d}, B={b:+d}", color=pair_colors[j], alpha=0.80)
    for bar, val in zip(bars, probs):
        if val > 15:  # Only label bars with significant probability
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 0.85,
                f"{val:.1f}%",
                ha="center", va="center",
                fontsize=16, color="white", fontweight="normal",
            )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=13)
ax.set_ylim(0, 100)
ax.set_ylabel("Probability (%)")
ax.set_xlabel("Clamped inputs")
ax.set_title("Conditional output probabilities for AND gate (C clamped)")
ax.legend(title="(A, B) spin pair")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()