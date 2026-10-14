"""
The AND gate's coupling matrix built from its truth-table rows,
J = 4 U^T U / L with a zero diagonal, and the check J @ U^T against U.
"""
import numpy as np

def build_J_from_truth_rows(U):
    # U: one bipolar truth-table row u_k per row, mutually orthogonal (trivial S)
    U = np.asarray(U, dtype=float)
    n_rows, n_bits = U.shape
    # J_raw = (1/L) * sum_k u_k u_k^T = (1/L) * U^T U, with norm L = ||u_k||^2
    L = np.dot(U[0], U[0])
    J = 4 * (U.T @ U) / L
    np.fill_diagonal(J, 0.0) # Zero diagonal
    return J

# AND gate
#    [ D   E   F   G   Z   A   B   C]
u1 = [-1,  1,  1,  1,  1, -1, -1, -1]
u2 = [ 1, -1, -1,  1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]

''' # 2 blocks coupling matrix
#    [ D   E   F   G   Z   A   B   C]
u1 = [ 1, -1,  1, -1, -1,  1, -1,  1]
u2 = [ 1,  1, -1, -1, -1,  1,  1, -1]
u3 = [-1,  1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]
'''
'''
#    [ D   E   F   G   Z   A   B   C] Camsari table
u1 = [-1,  1,  1,  1,  1, -1, -1, -1]
u2 = [ 1, -1, -1,  1,  1, -1,  1, -1]
u3 = [ 1, -1,  1, -1,  1,  1, -1, -1]
u4 = [ 1,  1,  1,  1,  1,  1,  1,  1]
'''

# U, one truth-table row per row
U = np.array([u1, u2, u3, u4], dtype=float)
J_AND = build_J_from_truth_rows(U)

np.set_printoptions(precision=3, suppress=True)  # 3 decimal places and no scientific notation
print("coupling matrix J:")
print(J_AND)

A = J_AND @ U.T
print("Camsari truth table")
print(U)
print("Camsari J @ U^T")
print(A.T)