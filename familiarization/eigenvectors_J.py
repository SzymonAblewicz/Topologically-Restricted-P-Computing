"""
The eigenvectors of the 8 p-bit AND gate's J (and of J/4 + I/2), written in
a +-1 basis where one exists, to recover the truth-table rows from J.
"""
from sympy import Matrix, Rational, eye # type: ignore
import itertools

J = Matrix([
    [ 0, -1,  0,  0,  1,  1,  1,  0],
    [-1,  0,  1,  1,  0,  0,  0,  1],
    [ 0,  1,  0,  0,  1,  1, -1,  0],
    [ 0,  1,  0,  0,  1, -1,  1,  0],
    [ 1,  0,  1,  1,  0,  0,  0, -1],
    [ 1,  0,  1, -1,  0,  0,  0,  1],
    [ 1,  0, -1,  1,  0,  0,  0,  1],
    [ 0,  1,  0,  0, -1,  1,  1,  0] ])
Jraw = J * Rational(1, 4) + eye(8) * Rational(1, 2)

def pm1_basis_for_eigenspace(eig_vecs):
    """A +-1 basis of an eigenspace, from SymPy's eigenvectors, to recover the truth-table rows (SymPy's own vectors if there is none)."""
    n = len(eig_vecs)
    pm1_vecs = []
    for signs in itertools.product([-1, 1], repeat=n):
        candidate = eig_vecs[0] * signs[0]
        for s, v in zip(signs[1:], eig_vecs[1:]):
            candidate = candidate + s * v
        entries = list(candidate)
        if all(abs(e) == 1 for e in entries):
            if candidate not in pm1_vecs:
                pm1_vecs.append(candidate)
    return pm1_vecs

def is_positive_canonical(v):
    for e in list(v):
        if e != 0:
            return int(e) > 0  # int() converts Sympy Integer to Python int first
    return True

def print_eigenvects_pm1(label, matrix):
    print(f"{label} eigenvectors (±1 basis):")
    for val, mult, vecs in matrix.eigenvects():
        print(f"λ = {val}  (algebraic multiplicity {mult})")
        all_vecs = pm1_basis_for_eigenspace(vecs)

        positives = [v for v in all_vecs if is_positive_canonical(v)]
        negatives = [v for v in all_vecs if not is_positive_canonical(v)]

        print("  canonical: (first entry positive)")
        for i, v in enumerate(positives, 1):
            print(f"    u{i} = {list(v)}")
        print("  negations: (first entry negative)")
        for i, v in enumerate(negatives, 1):
            print(f"   -u{i} = {list(v)}")
        print()


print_eigenvects_pm1("J", J)
print_eigenvects_pm1("Jraw", Jraw)