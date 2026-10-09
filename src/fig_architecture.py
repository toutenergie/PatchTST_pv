# -*- coding: utf-8 -*-
"""Schéma de l'architecture PatchTST-CA-X proposée (figure de l'article)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figures_style as fs


def boite(ax, x, y, l, h, texte, fc="#eaf2fb", ec="#5d6d7e", gras=False, taille=7.5):
    ax.add_patch(FancyBboxPatch((x, y), l, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + l / 2, y + h / 2, texte, ha="center", va="center", fontsize=taille,
            fontweight="bold" if gras else "normal")


def fleche(ax, a, b, couleur="#5d6d7e", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=couleur, lw=1, ls=ls))


def tracer():
    fs.appliquer()
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 22); ax.set_ylim(0, 10.4); ax.axis("off")
    R = fs.COULEURS["modele"]
    # --- entrées
    boite(ax, 0.2, 7.6, 3.4, 2.2, "Passé (L heures)\n\nPV_total (cible)\nThermique, Éolien,\nImport Manantali\n+ 8 covariables\ndéterministes", fc="#fdf2e9")
    boite(ax, 0.2, 0.4, 3.4, 2.0, "Futur connu (H heures)\n\nsin/cos : heure,\njour de l'année,\njour de semaine\ncos θz · GHI ciel clair", fc="#fdf2e9")
    # --- RevIN sélective
    boite(ax, 4.3, 7.9, 3.0, 1.6, "RevIN sélective\n(cible + réseau) ;\ndéterministes :\nnorm. globale")
    fleche(ax, (3.6, 8.7), (4.3, 8.7))
    # --- patchs
    boite(ax, 8.0, 7.9, 3.0, 1.6, "Patchs de longueur P,\npas P/2 → N jetons ;\nprojection P→D\n+ position + canal")
    fleche(ax, (7.3, 8.7), (8.0, 8.7))
    # --- encodeur temporel partagé
    boite(ax, 11.7, 7.9, 3.3, 1.6, "Encodeur Transformer\npartagé entre canaux\n(attention temporelle,\n2 couches, D = 64)", fc="#eaf2fb", gras=True)
    fleche(ax, (11.0, 8.7), (11.7, 8.7))
    # --- branche CI
    boite(ax, 15.8, 8.3, 2.7, 1.2, "CI : canal cible seul\n(exogènes ignorés)", fc="#f4f6f6", taille=7)
    fleche(ax, (15.0, 9.0), (15.8, 8.9))
    # --- branche CA
    boite(ax, 15.8, 5.6, 2.7, 1.9, "CA : attention\nINTER-CANAUX\npar position de patch\nrequête = cible", fc="#fdecea", ec=R, gras=True)
    fleche(ax, (15.0, 8.2), (15.8, 6.8), couleur=R)
    # --- tête
    boite(ax, 19.1, 6.9, 2.7, 1.6, "Tête linéaire\nN·D → H\n(prévision directe)")
    fleche(ax, (18.5, 8.9), (19.1, 8.0)); fleche(ax, (18.5, 6.6), (19.1, 7.3), couleur=R)
    # --- conditionnement futur X
    boite(ax, 4.6, 0.5, 9.0, 1.8, "Conditionnement futur (X) : pour chaque pas h,\nMLP[contexte(z) ; covariables futures(h) ; ŷ_base(h)]\n→ correction additive Δŷ(h)", fc="#fdecea", ec=R, gras=True)
    fleche(ax, (3.6, 1.4), (4.6, 1.4), couleur=R)
    fleche(ax, (20.4, 6.9), (13.6, 2.0), couleur="#5d6d7e", ls="--")
    ax.text(17.4, 4.1, "contexte z, ŷ_base", fontsize=7, color="#5d6d7e", rotation=26)
    # --- sortie
    boite(ax, 15.2, 0.5, 3.1, 1.8, "Dé-normalisation RevIN\n(échelle de la fenêtre\ndu canal cible)")
    fleche(ax, (13.6, 1.4), (15.2, 1.4), couleur=R)
    boite(ax, 19.0, 0.5, 2.8, 1.8, "Porte physique (g)\nŷ = 0 si cos θz = 0\n→ ŷ(t+1 … t+H) MW", fc="#f4f6f6")
    fleche(ax, (18.3, 1.4), (19.0, 1.4))
    # légende
    ax.text(0.2, 4.9, "Variantes étudiées :", fontsize=8, fontweight="bold")
    ax.text(0.2, 3.3, "CI     : PatchTST standard (indépendance des canaux)\n"
                      "CA     : + attention inter-canaux\n"
                      "X      : + conditionnement par le futur connu\n"
                      "g      : + porte physique nocturne\n"
                      "en rouge : contributions de cette étude", fontsize=7.4, family="monospace")
    ax.set_title("Architecture PatchTST-CA-X pour la prévision de la production PV totale", fontsize=10, fontweight="bold")
    fs.sauver(fig, "fig_architecture")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    tracer()
