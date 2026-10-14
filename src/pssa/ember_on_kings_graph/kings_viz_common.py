"""The fixed L x L canvas shared by the three drawings, so every figure has the same scale."""
def _set_fixed_kings_extent(ax, L: int, pad: float = 0.5):
    """Fix the axes to the whole L x L lattice, row 0 at the top."""
    ax.set_xlim(-pad, (L - 1) + pad)
    ax.set_ylim(-((L - 1) + pad), pad)
    ax.set_aspect('equal')
