# -*- coding: utf-8 -*-
"""figure_style.py — shared visual style for all figures (publication quality, 300 dpi)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FOLDER = Path(__file__).resolve().parents[1] / "figures"

#: Sober palette, legible in grey scale.
COLORS = {
    "observed": "#1f1f1f",     # observation
    "model": "#c0392b",        # selected model
    "persistence": "#7f8c8d",  # naive baseline
    "mean": "#2e86c1",         # moving average
    "train": "#2e86c1", "val": "#f39c12", "test": "#c0392b",
}
CYCLE = ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad", "#f39c12", "#16a085", "#7f8c8d"]


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def save(fig, name: str):
    """Save the figure as a 300 dpi PNG in figures/ and display it."""
    FOLDER.mkdir(exist_ok=True)
    fig.savefig(FOLDER / f"{name}.png")
    plt.show()
