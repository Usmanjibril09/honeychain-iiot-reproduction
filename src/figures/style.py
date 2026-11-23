"""Shared plotting style for HoneyChain-IIoT figures.

All figures follow IEEE IoT Journal conventions:
- 2x2 subplots labeled (a)-(d) in top-left of each panel.
- Times New Roman 14-15 pt.
- Bar width <= 0.4, no pie charts.
- Rich shading: CI bands, error bars, distribution overlays.
- PNG (300 dpi) + PDF outputs with the same base name.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent qualitative palette (colorblind-friendly).
PALETTE = {
    "primary":   "#1f77b4",  # blue
    "secondary": "#d62728",  # red
    "success":   "#2ca02c",  # green
    "warning":   "#ff7f0e",  # orange
    "purple":    "#9467bd",
    "brown":     "#8c564b",
    "pink":      "#e377c2",
    "gray":      "#7f7f7f",
    "olive":     "#bcbd22",
    "cyan":      "#17becf",
}
PALETTE_LIST = list(PALETTE.values())

BAR_WIDTH = 0.35  # thin bars as requested


def apply_style() -> None:
    """Apply global matplotlib style. Call once at start of each figure script."""
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size":   15,
        "axes.titlesize":  15,
        "axes.labelsize":  15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.frameon":  True,
        "legend.framealpha": 0.9,
        "axes.grid":       True,
        "grid.alpha":      0.25,
        "grid.linewidth":  0.5,
        "axes.axisbelow":  True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":       120,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":     42,  # embed as truetype for editability
        "ps.fonttype":      42,
    })


def panel_label(ax, label: str, dx: float = -0.15, dy: float = 1.05) -> None:
    """Place a bold (a)/(b)/(c)/(d) label at top-left of a subplot."""
    ax.text(dx, dy, f"({label})", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top", ha="left")


def save_figure(fig, base_name: str) -> None:
    """Save figure as both PNG (300 dpi) and PDF."""
    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"[save] {png_path} + {pdf_path}")
