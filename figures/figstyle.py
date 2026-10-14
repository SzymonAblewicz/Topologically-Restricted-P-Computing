"""
The shared style of the report's data figures: one palette with the colours by role, the
report's font (Times, TeX Gyre Termes, from the TeX installation) at the captions' size,
and every figure drawn at its printed size in millimetres, so LaTeX never rescales it.
"""
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import matplotlib.ticker as mt                                    # noqa: E402
from matplotlib import font_manager as fm                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# sizes (mm)
MM = 1 / 25.4
COL, PAGE = 87.5, 180.0           # one column, two columns

# colours by role, the same in every figure
BLUE = "#1C5AA6"                  # the method, or the adopted variant
ORANGE = "#E07B39"                # the alternative it is compared with
GREY = "#8C8C8C"                  # baselines and controls
PURPLE = "#7A5BA8"                # a third method (Sugie's PSSA)
INK = "#1A1A1A"                   # bounds, such as the complete graph embedding
FAINT = "#EBEBEB"                 # faint horizontal gridlines
SEQ = LinearSegmentedColormap.from_list(
    "report_blues", ["#FFFFFF", "#D6E3F3", "#7FA6D6", BLUE, "#0E2F5A"])

# the report's font, found with kpsewhich
try:
    _f = subprocess.run(["kpsewhich", "texgyretermes-regular.otf"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    _f = ""
if _f:
    for _s in ("regular", "bold", "italic", "bolditalic"):
        fm.fontManager.addfont(os.path.join(os.path.dirname(_f), f"texgyretermes-{_s}.otf"))
FONT = "TeX Gyre Termes"
LABEL = 9                         # axis labels and titles: the captions' size
SMALL = 8                         # ticks, legends, annotations, heatmap cells
LETTER = 10                       # bold panel letters

plt.rcParams.update({
    "font.family": FONT, "font.sans-serif": [FONT], "font.serif": [FONT],
    "font.size": LABEL, "axes.labelsize": LABEL, "axes.titlesize": LABEL,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
    "legend.title_fontsize": SMALL,
    "mathtext.fontset": "custom", "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "axes.labelpad": 3, "axes.titlepad": 3,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.grid": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
    "pdf.fonttype": 3, "savefig.dpi": 300, "savefig.bbox": None, "axes.unicode_minus": False,
    "axes.titleweight": "normal",
})


def figure(w_mm, h_mm):
    """A figure of exactly w x h millimetres."""
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def axes(fig, x, y, w, h):
    """Axes placed in millimetres, from the figure's top-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def letter(fig, s, x, y):
    """A bold panel letter at (x, y) mm from the figure's top-left corner."""
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, 1 - y / H, s, fontsize=LETTER, fontweight="bold", va="top", ha="left", color=INK)


def hgrid(ax):
    """Faint horizontal gridlines, for a plot whose message is a rise, a fall or a level."""
    ax.yaxis.grid(True, color=FAINT, lw=0.5)
    ax.set_axisbelow(True)


def dot(ax, x, y, color, ls="-", hollow=False, lw=1.2, ms=3.5, **kw):
    """Measured data: points joined by a line, with a white edge (or hollow)."""
    return ax.plot(x, y, ls=ls, marker="o", color=color, lw=lw, ms=ms,
                   mfc="white" if hollow else color, mec=color if hollow else "white",
                   mew=0.8 if hollow else 0.5, **kw)


def ink_on(cmap, v):
    """White text on dark cells, ink on light ones."""
    r, g, b = to_rgb(cmap(v))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def heatmap_frame(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def percent(axis):
    axis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))


def save(fig, name):
    """Save the figure beside this file, under its name in the report."""
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    plt.close(fig)
    print("saved", name + ".pdf")
