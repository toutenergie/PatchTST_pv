# -*- coding: utf-8 -*-
"""
run_experiences.py — Exécution séquentielle des 4 configurations (un facteur à la fois).

Chaque étape reprend la meilleure valeur de l'étape précédente (score de
décision normalisé, configurations rejetées au test de bruit exclues).
Les résultats sont mis en cache par patchtst_pv.evaluer_configuration :
le script peut être interrompu et relancé sans perte.

    Config 1 : P ∈ {12, 24, 48}             (L=168, CA-X, H=24)
    Config 2 : L ∈ {96, 168, 336}           (P*)
    Config 3 : canaux ∈ {CI, CI-X, CA, CA-X, CA-Xg}   (P*, L*)
    Config 4 : H ∈ {1, 6, 12, 24, 168}      (P*, L*, canaux*)
"""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patchtst_pv as m  # noqa: E402

jeu = m.construire_jeu()
journal = m.DOSSIER_CACHE / "selection.json"
selection = json.loads(journal.read_text()) if journal.exists() else {}


def etape(nom, configs):
    """Évalue une liste de configurations et retourne la meilleure."""
    print(f"\n=== {nom} ===", flush=True)
    res = [m.evaluer_configuration(c, jeu) for c in configs]
    tab = m.tableau_comparatif(res)
    tab.to_csv(m.DOSSIER_TABLEAUX / f"{nom}.csv", index=False)
    admissibles = tab[~tab["rejete_bruit"]] if (~tab["rejete_bruit"]).any() else tab
    meilleur = admissibles.sort_values("score_normalise", ascending=False).iloc[0]["configuration"]
    print(tab[["configuration", "RMSE", "RMSE_std", "nRMSE", "degradation_bruit", "drift_7j",
               "score", "score_normalise"]].round(4).to_string(), flush=True)
    print(f"--> retenu : {meilleur}", flush=True)
    selection[nom] = meilleur
    journal.write_text(json.dumps(selection, indent=1))
    return configs[[c.nom() for c in configs].index(meilleur)]


base = m.ConfigPatchTST(P=24, L=168, H=24, mode_canaux="CA", futur_connu=True)

c1 = etape("config1_patch", [dataclasses.replace(base, P=p) for p in (12, 24, 48)])
c2 = etape("config2_historique", [dataclasses.replace(c1, L=l) for l in (96, 168, 336)])
c3 = etape("config3_canaux", [
    dataclasses.replace(c2, mode_canaux="CI", futur_connu=False, porte_physique=False),
    dataclasses.replace(c2, mode_canaux="CI", futur_connu=True, porte_physique=False),
    dataclasses.replace(c2, mode_canaux="CA", futur_connu=False, porte_physique=False),
    dataclasses.replace(c2, mode_canaux="CA", futur_connu=True, porte_physique=False),
    dataclasses.replace(c2, mode_canaux="CA", futur_connu=True, porte_physique=True),
])
c4 = etape("config4_horizon", [dataclasses.replace(c3, H=h) for h in (1, 6, 12, 24, 168)])
print("\nFIN — sélection :", json.dumps(selection, indent=1), flush=True)
