# -*- coding: utf-8 -*-
"""figures_style.py — charte graphique commune (qualité publication, 300 dpi)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

DOSSIER = Path(__file__).resolve().parents[1] / "figures"

#: Palette sobre et lisible en niveaux de gris.
COULEURS = {
    "observe": "#1f1f1f",     # observation
    "modele": "#c0392b",      # modèle retenu
    "persistance": "#7f8c8d", # référence naïve
    "moyenne": "#2e86c1",     # moyenne glissante
    "train": "#2e86c1", "val": "#f39c12", "test": "#c0392b",
}
CYCLE = ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad", "#f39c12", "#16a085", "#7f8c8d"]


def appliquer():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def sauver(fig, nom: str):
    """Enregistre la figure en PNG 300 dpi dans figures/ et l'affiche."""
    DOSSIER.mkdir(exist_ok=True)
    fig.savefig(DOSSIER / f"{nom}.png")
    plt.show()
