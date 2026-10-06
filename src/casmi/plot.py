"""Shared matplotlib style for EDA figures (static PNGs for the markdown report)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import FIG  # noqa: E402

# Validated categorical order (light surface); never cycle past 8.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OTHER = "#c3c2b7"  # de-emphasised context series

# Fixed colour per key population so a library keeps its colour across figures.
KEY = {"test": SERIES[1], "enveda-np-examples": SERIES[2], "enveda-180": SERIES[0]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
    "lines.linewidth": 2,
})


def save(fig, name: str) -> str:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{name}.png"


def hbar(ax, labels, values, color=SERIES[0], fmt="{:,.0f}", highlight=None):
    """Horizontal bar chart, largest at top, value labels at bar end."""
    y = range(len(labels))[::-1]
    colors = [highlight.get(l, color) for l in labels] if highlight else color
    ax.barh(list(y), values, color=colors, height=0.72, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, vmax * 1.22)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: compact(v)))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))


def compact(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"
