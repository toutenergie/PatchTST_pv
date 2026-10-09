# -*- coding: utf-8 -*-
"""organigramme.py — organigramme de la démarche pour chaque configuration.

Le facteur étudié par la configuration est mis en évidence (bloc rouge).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figures_style as fs

ETAPES = [
    ("donnees", "Journal horaire SENELEC\nenergie.xlsx (2019–2023)"),
    ("cible", "Cible PV_total = Σ 11 centrales\n(écrêtage, imputation causale)"),
    ("cov", "Covariables : réseau (3) · cycliques (6)\n· géométrie solaire (2)"),
    ("split", "Découpage temporel strict\nTrain 2019–21 · Val 2022 · Test 10/22–05/23"),
    ("norm", "Normalisation ajustée sur Train\n+ RevIN par fenêtre"),
    ("fen", "Fenêtrage : passé L h → futur H h"),
    ("patch", "Découpage en patchs\nlongueur P, pas P/2"),
    ("canaux", "Encodeur Transformer partagé\n+ traitement des canaux (CI / CA)"),
    ("tete", "Tête de prévision directe (H pas)\n± conditionnement futur X"),
    ("train", "Entraînement × 5 graines\nAdam · arrêt précoce sur Val"),
    ("eval", "Évaluation sur Test : précision · bruit 5 %\n· rolling 30 j (dérive) · efficience"),
    ("score", "Score de décision\n0,4/nRMSE + 0,3/σ + 0,3/dérive + bonus"),
    ("choix", "Configuration retenue →\nétape suivante"),
]

FACTEUR = {1: ("patch", "P ∈ {12, 24, 48}"), 2: ("fen", "L ∈ {96, 168, 336}"),
           3: ("canaux", "CI · CI-X · CA · CA-X · CA-Xg"), 4: ("tete", "H ∈ {1, 6, 12, 24, 168}")}


def tracer(num_config: int, titre: str, nom_fichier: str):
    cle, valeurs = FACTEUR[num_config]
    fig, ax = plt.subplots(figsize=(6.2, 9.2))
    ax.set_xlim(-1.0, 10); ax.set_ylim(0, len(ETAPES) * 1.0 + 0.5); ax.axis("off")
    y0 = len(ETAPES) * 1.0
    for i, (k, texte) in enumerate(ETAPES):
        y = y0 - i * 1.0
        mis_en_avant = k == cle
        couleur = "#fdecea" if mis_en_avant else ("#eaf2fb" if i < 5 else "#f4f6f6")
        bord = fs.COULEURS["modele"] if mis_en_avant else "#5d6d7e"
        ax.add_patch(FancyBboxPatch((1.2, y - 0.36), 6.2, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=couleur, ec=bord, lw=1.8 if mis_en_avant else 0.9))
        ax.text(4.3, y, texte, ha="center", va="center", fontsize=7.6,
                fontweight="bold" if mis_en_avant else "normal")
        if mis_en_avant:
            ax.text(7.6, y, "facteur étudié :\n" + valeurs, ha="left", va="center", fontsize=7.4,
                    color=fs.COULEURS["modele"], fontweight="bold")
        if i < len(ETAPES) - 1:
            ax.add_patch(FancyArrowPatch((4.3, y - 0.37), (4.3, y - 0.63), arrowstyle="-|>",
                                         mutation_scale=9, color="#5d6d7e", lw=0.9))
    # boucle de sélection : score → retour sur le facteur
    y_cle = y0 - [k for k, _ in ETAPES].index(cle) * 1.0
    y_score = y0 - [k for k, _ in ETAPES].index("score") * 1.0
    ax.add_patch(FancyArrowPatch((1.2, y_score), (1.2, y_cle), connectionstyle="arc3,rad=-0.45",
                                 arrowstyle="-|>", mutation_scale=9, color=fs.COULEURS["modele"],
                                 lw=1.0, ls="--"))
    ax.text(-0.95, (y_score + y_cle) / 2, "une\nvaleur\npar\nessai", fontsize=7, color=fs.COULEURS["modele"],
            va="center")
    ax.set_title(titre, fontsize=10, fontweight="bold")
    fs.sauver(fig, nom_fichier)
