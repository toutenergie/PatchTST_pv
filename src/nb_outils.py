# -*- coding: utf-8 -*-
"""nb_outils.py — construction et exécution des notebooks à partir de listes de cellules.

Chaque notebook est décrit comme une liste de tuples ("md" | "code", texte).
Il est ensuite exécuté de bout en bout avec nbclient : les sorties et les
figures sont intégrées au fichier .ipynb livré.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RACINE = Path(__file__).resolve().parents[1]

ENTETE_CODE = """# --- Environnement commun à tous les notebooks -------------------------------
import sys, warnings
from pathlib import Path
RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RACINE / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patchtst_pv as m          # module commun de l'étude (entièrement commenté)
import figures_style as fs       # charte graphique des figures de l'article
fs.appliquer()
pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 4)
"""


def construire(nom: str, cellules: list[tuple[str, str]], executer: bool = True,
               delai: int = 7200) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for type_, texte in cellules:
        texte = texte.strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(texte) if type_ == "md"
                        else nbformat.v4.new_code_cell(texte))
    chemin = RACINE / "notebooks" / f"{nom}.ipynb"
    chemin.parent.mkdir(exist_ok=True)
    if executer:
        NotebookClient(nb, timeout=delai, kernel_name="python3",
                       resources={"metadata": {"path": str(chemin.parent)}}).execute()
    nbformat.write(nb, chemin)
    return chemin
