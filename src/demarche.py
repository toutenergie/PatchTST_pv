# -*- coding: utf-8 -*-
"""
demarche.py — Document Word « Démarche étape par étape » (item 6 de projet.md).

Explique, dans l'ordre d'exécution, ce qui a été fait, pourquoi, avec quel
code, et comment le reproduire. Les chiffres proviennent du cache de résultats.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patchtst_pv as m  # noqa: E402

RACINE = m.RACINE


def fr(x, d=2):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def fig(nom, legende, largeur=16):
    w, h = Image.open(RACINE / "figures" / f"{nom}.png").size
    return {"type": "figure", "chemin": str(RACINE / "figures" / f"{nom}.png"), "ratio": h / w,
            "largeur_cm": largeur, "legende": legende}


def blocs():
    jeu = m.construire_jeu()
    sel = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())
    B = []
    B += [{"type": "h1", "texte": "Objet du document"},
          {"type": "p", "texte": (
              "Ce document décrit pas à pas la démarche suivie pour construire, évaluer et sélectionner un modèle "
              "PatchTST de prévision de la production photovoltaïque totale du réseau SENELEC, selon le cahier des "
              "charges `projet.md`. Pour chaque étape, il précise **ce qui est fait**, **pourquoi**, **où se trouve le "
              "code** et **comment le vérifier**. Le document s'adresse autant au lecteur qui veut comprendre la méthode "
              "qu'à celui qui veut la reproduire.")},
          {"type": "encadre", "texte": (
              "**Écart assumé avec `projet.md`.** Le cahier des charges décrit un problème univarié ; la demande a "
              "ensuite évolué vers une prévision **multivariée sans les productions PV individuelles comme variables "
              "d'entrée**, avec des variables cycliques. Faute de données météorologiques dans le journal, le test de "
              "bruit de 5 % « sur la météo » porte sur l'ensemble des covariables exogènes.")},
          {"type": "h1", "texte": "Étape 0 — Organisation du projet"},
          {"type": "tableau", "legende": "**Tableau A.** Arborescence des livrables.", "largeurs": [3.2, 6.8], "gauche": [1],
           "entetes": ["Élément", "Contenu"],
           "lignes": [["data/", "energie.xlsx (journal horaire), projet.md (cahier des charges)"],
                      ["src/patchtst_pv.py", "module commun commenté : données, covariables, PatchTST, entraînement, métriques, protocole, score"],
                      ["src/analyse.py", "références (persistance, moyenne, Ridge), tests DM, figures, validation croisée, XAI"],
                      ["src/run_experiences.py", "enchaîne les configurations 1 → 4 (un facteur à la fois)"],
                      ["notebooks/00 … 05", "EDA ; une configuration par notebook ; modèle final"],
                      ["figures/", "toutes les figures (PNG 300 dpi), dont les organigrammes org_config1…4"],
                      ["tableaux/", "tableaux comparatifs (CSV) et classeur Excel récapitulatif"],
                      ["models/best_patchtst.pt", "poids du modèle final + configuration + paramètres de normalisation"],
                      ["cache/", "résultats par configuration et par graine (prévisions, poids, historiques)"]]},
          {"type": "p", "texte": (
              "**Reproduire l'étude.** (1) `python src/run_experiences.py` entraîne et évalue toutes les configurations "
              "(plusieurs heures sur processeur) ; (2) les notebooks 00 à 05 peuvent ensuite être exécutés dans l'ordre : "
              "ils relisent le cache. Supprimer un dossier `cache/<configuration>/` relance son entraînement.")},
          ]

    # --- Étape 1 : données
    d = jeu.diagnostic_cible
    B += [{"type": "h1", "texte": "Étape 1 — Chargement et contrôle des données (notebook 00)"},
          {"type": "p", "texte": (
              f"Le journal couvre {len(jeu.tableau):,} heures, du {jeu.index[0]:%d/%m/%Y} au {jeu.index[-1]:%d/%m/%Y}. "
              "La grille horaire est complétée pour les 288 heures absentes (huit sauts d'un à trois jours) : "
              "le fenêtrage exige une grille régulière. **Pourquoi** : supprimer ces heures décalerait toutes les "
              "fenêtres qui les contiennent.").replace(",", " ", 1)},
          {"type": "h1", "texte": "Étape 2 — Construction de la cible PV_total"},
          {"type": "p", "texte": (
              "La cible est la somme des onze centrales PV. Pour chaque centrale : (a) écrêtage au-delà de 1,2 × le "
              "quantile 99,9 % **de la période d'entraînement** ; (b) valeur nulle hors période d'existence ; (c) "
              "imputation causale des lacunes. Code : `construire_cible`, `imputer_serie`.")},
          {"type": "tableau", "legende": "**Tableau B.** Traçabilité de la construction de la cible, par centrale.",
           "largeurs": [2.4, 2.2, 2.2, 1.2, 1.3, 1.3, 1.2], "taille": 15,
           "entetes": ["Centrale", "Mise en service", "Dernier relevé", "Seuil (MW)", "Écrêtées", "Lacunes imputées", "Moyenne (MW)"],
           "lignes": [[r.centrale, f"{r.mise_en_service:%d/%m/%Y}", f"{r.dernier_releve:%d/%m/%Y}", fr(r.seuil_MW, 1),
                       str(r.valeurs_ecretees), str(r.lacunes_imputees), fr(r.moyenne_MW)] for r in d.itertuples()]},
          {"type": "p", "texte": (
              "**Vérification** : le notebook 00 affirme par des assertions que la normalisation est calculée sur la "
              "seule période d'entraînement et qu'aucune colonne PV individuelle, ni aucun total qui la contient, "
              "n'entre dans le modèle.")},
          fig("fig01_serie_pv_total", "**Figure A.** Série cible, enveloppe de capacité et découpage temporel."),
          {"type": "h1", "texte": "Étape 3 — Analyse exploratoire"},
          {"type": "puces", "items": [
              "Saisonnalité : cycle de 24 h dominant (ACF(24) ≈ 0,97 sur l'entraînement), harmonique à 12 h, modulation annuelle.",
              "Non-stationnarité : la production moyenne croît de 47 % entre 2019 et 2022 (mises en service).",
              "Corrélations : forte avec la géométrie solaire (≈ 0,94), faible avec le contexte réseau une fois le cycle moyen retiré.",
              "Conséquence : la persistance J-1 sera une référence difficile à battre."]},
          fig("fig03_saisonnalite", "**Figure B.** Profil heure × mois, autocorrélation et périodogramme."),
          {"type": "h1", "texte": "Étape 4 — Covariables"},
          {"type": "p", "texte": (
              "Douze canaux : la cible ; trois canaux réseau (thermique agrégé, éolien Taiba, import Manantali) ; six "
              "encodages cycliques (sin/cos de l'heure, du jour de l'année, du jour de semaine) ; deux variables de "
              "géométrie solaire (cos θz, rayonnement de ciel clair de Haurwitz, pvlib). Le décalage entre horodatage "
              f"et temps solaire est estimé sur l'entraînement : {fr(jeu.decalage_solaire_h)} h. Code : "
              "`covariables_cycliques`, `covariables_solaires`, `estimer_decalage`, `covariables_reseau`.")},
          fig("fig04_covariables", "**Figure C.** Covariables cycliques, géométrie solaire et contexte réseau."),
          {"type": "h1", "texte": "Étape 5 — Découpage temporel et normalisation"},
          {"type": "p", "texte": (
              f"Train : 01/01/2019 → 31/12/2021 ({jeu.i_val} h) ; Validation : 01/01/2022 → 30/09/2022 "
              f"({jeu.i_test - jeu.i_val} h) ; Test : 01/10/2022 → 09/05/2023 ({len(jeu.valeurs) - jeu.i_test} h). "
              "Aucun mélange. La normalisation globale est ajustée sur Train ; à l'intérieur du modèle, une normalisation "
              "réversible (RevIN) est appliquée fenêtre par fenêtre aux **seuls canaux stochastiques**.")},
          {"type": "encadre", "texte": (
              "**Correction faite en cours d'étude.** Une première version appliquait RevIN à tous les canaux. Sur une "
              "fenêtre de 7 jours, le jour de l'année varie à peine : le centrer-réduire effaçait la saison et créait une "
              "rampe artificielle. Les gradients intégrés l'ont révélé (24 % de l'attribution sur `doy_sin` passé). "
              "Tous les résultats présentés ont été recalculés après correction.")},
          {"type": "h1", "texte": "Étape 6 — Modèle PatchTST et variantes"},
          fig("fig_architecture", "**Figure D.** Architecture PatchTST-CA-X et variantes CI / CA / X / g.", 16.5),
          {"type": "p", "texte": (
              "Classe `PatchTST` du module. Les patchs de longueur P (pas P/2) sont projetés en dimension 64 ; un "
              "encodeur Transformer de 2 couches est partagé entre canaux. **CI** : seule la cible alimente la tête ; "
              "**CA** : attention inter-canaux à chaque position de patch ; **X** : correction pas à pas par les "
              "covariables déterministes futures ; **g** : porte physique nocturne.")},
          {"type": "h1", "texte": "Étape 7 — Entraînement et protocole d'évaluation"},
          {"type": "numeros", "instance": 3, "items": [
              "Entraînement sur Train, arrêt précoce sur Val (patience 4, au plus 25 époques), 20 % des origines horaires tirées à chaque époque, 5 graines.",
              "Précision sur toutes les origines horaires du Test : RMSE, MAE, nRMSE = RMSE/moyenne, R², MBE.",
              "Robustesse : écart-type sur 5 graines ; bruit gaussien 5 % sur les exogènes (rejet si le RMSE augmente de plus de 15 %).",
              "Stabilité : rolling forecast de 30 jours sans ré-entraînement ; dérive = RMSE(J24–30)/RMSE(J1–7) (objectif < 1,2), ratio brut J30/J1 rapporté.",
              "Efficience : temps d'inférence d'un échantillon sur un cœur (< 100 ms) ; taille des poids (< 50 Mo).",
              "Score = 0,4/nRMSE + 0,3/σ + 0,3/dérive + bonus ; sélection sur le score normalisé (critères ramenés à [0, 1]).",
              "Tests de Diebold-Mariano contre la persistance et la moyenne 7 jours.",
          ]},
    ]
    # --- Étapes 8 à 11 : configurations
    noms_etapes = {"config1_patch": "Configuration 1 — longueur de patch P", "config2_historique": "Configuration 2 — historique L",
                   "config3_canaux": "Configuration 3 — traitement des canaux", "config4_horizon": "Configuration 4 — horizon H"}
    for k, (etape, titre) in enumerate(noms_etapes.items(), start=1):
        f = m.DOSSIER_TABLEAUX / f"tableau_config{k}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        B += [{"type": "h1", "texte": f"Étape {7 + k} — {titre} (notebook 0{k})"},
              fig(f"org_config{k}", f"**Figure {chr(68 + k)}.** Organigramme de la configuration {k}.", 9.5),
              {"type": "tableau", "legende": f"**Tableau {chr(66 + k)}.** Résultats de la configuration {k} (moyenne sur 5 graines).",
               "largeurs": [3.4, 1.3, 1.1, 1.2, 1.3, 1.2, 1.2, 1.3], "taille": 14, "gauche": [0],
               "surligner": [i for i, c in enumerate(t.configuration) if c == sel.get(etape)],
               "entetes": ["Configuration", "RMSE (MW)", "σ (MW)", "nRMSE", "Bruit 5 %", "Dérive", "Inférence (ms)", "Score norm."],
               "lignes": [[str(r.configuration), fr(r.RMSE), fr(r.RMSE_std, 3) if r.RMSE_std == r.RMSE_std else "–",
                           fr(r.nRMSE, 3), (fr(100 * r.degradation_bruit, 1) + " %") if r.degradation_bruit == r.degradation_bruit else "–",
                           fr(r.drift_7j) if r.drift_7j == r.drift_7j else "–",
                           fr(r.inference_ms, 1) if r.inference_ms == r.inference_ms else "–",
                           fr(r.score_normalise, 3) if r.score_normalise == r.score_normalise else "–"]
                          for r in t.itertuples()],
               "note": (f"Ligne surlignée : configuration retenue ({sel.get(etape, '–')}). Persistance et moyenne 7 jours : mêmes origines de test."
                        if k < 4 else f"Ligne surlignée : meilleur score ({sel.get(etape, '–')}) ; l'horizon final reste J+1 (24 h), fixé par l'usage.")},
              fig(f"fig_config{k}_chapelet", f"**Figure {chr(68 + k)}bis.** Chapelet des graines et dérive — configuration {k}."),
              fig(f"fig_config{k}_observe_predit", f"**Figure {chr(68 + k)}ter.** Observé vs prédit — meilleur modèle de la configuration {k}.")]
    # --- Étape finale
    B += [{"type": "h1", "texte": "Étape 12 — Modèle final (notebook 05)"},
          {"type": "p", "texte": (
              "L'architecture retenue aux configurations 1 à 3 est évaluée à l'horizon opérationnel J+1 (24 h). "
              "L'horizon n'est pas choisi par le score, parce que comparer des horizons revient à comparer des tâches "
              "différentes. Le notebook 05 ajoute : comparaison à Ridge et au PatchTST univarié, courbe de robustesse au "
              "bruit, RMSE mensuel, validation croisée TimeSeriesSplit à 5 plis, ablation « indice de capacité », "
              "explicabilité, et sauvegarde dans `models/best_patchtst.pt`. La graine sauvegardée est choisie sur la "
              "perte de validation, jamais sur le test.")},
          {"type": "tableau", "legende": "**Tableau G.** Modèle final vs références (horizon J+1, test complet).",
           "largeurs": [3.6, 1.3, 1.3, 1.2, 1.2, 1.2], "taille": 15,
           "entetes": ["Modèle", "RMSE (MW)", "MAE (MW)", "nRMSE", "R²", "MBE (MW)"],
           "lignes": [[str(r.modele), fr(r.RMSE), fr(r.MAE), fr(r.nRMSE, 3), fr(r.R2, 3), fr(r.MBE)]
                      for r in pd.read_csv(m.DOSSIER_TABLEAUX / "comparaison_references.csv").itertuples()],
           "note": "Diebold-Mariano : modèle final meilleur que la persistance (p = 0,003) et que la moyenne 7 jours ; écart avec Ridge non significatif (p ≈ 0,42)."},
          {"type": "tableau", "legende": "**Tableau H.** Validation croisée TimeSeriesSplit (5 plis).",
           "largeurs": [0.7, 2.4, 3.0, 1.1, 1.4, 1.2], "taille": 15,
           "entetes": ["Pli", "Entraînement", "Test", "RMSE", "RMSE pers.", "Gain"],
           "lignes": [[str(r.pli), r.train, r.test, fr(r.RMSE), fr(r.RMSE_persistance), fr(100 * r.gain_vs_persistance, 1) + " %"]
                      for r in pd.read_csv(m.DOSSIER_TABLEAUX / "validation_croisee.csv").itertuples()]},
          fig("fig_final_chapelet", "**Figure I.** Chapelet des 5 graines du modèle final et du PatchTST univarié, avec les références."),
          fig("fig_final_stabilite", "**Figure J.** Rolling 30 jours sans ré-entraînement et RMSE mensuel sur le test."),
          fig("fig_final_residus", "**Figure K.** Résidus du modèle final."),
          fig("fig_final_xai", "**Figure L.** Explicabilité : occultation, gradients intégrés, attention inter-canaux."),
          {"type": "h1", "texte": "Étape 13 — Contrôles de qualité"},
          {"type": "puces", "items": [
              "Exécution de bout en bout de chaque notebook (`nbclient`), sorties et figures intégrées.",
              "Assertions anti-fuite : ordre des segments, normalisation sur Train, colonnes interdites absentes.",
              "Rechargement de `best_patchtst.pt` et contrôle de prévisions identiques.",
              "Tableaux de l'article régénérés à partir des prévisions en cache (aucune saisie manuelle de chiffres)."]},
          ]
    return B


def construire():
    doc = {"titre": "Démarche étape par étape",
           "sous_titre": "Prévision multivariée de la production PV totale du réseau SENELEC par PatchTST",
           "auteurs": "Abdoulaye FAYE — UADB / EDTSS", "affiliation": "Document méthodologique accompagnant les notebooks",
           "entete": "Démarche — PatchTST PV total SENELEC", "blocs": blocs()}
    sortie_json = RACINE / "cache" / "demarche.json"
    sortie_json.write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")
    subprocess.run(["node", str(RACINE / "src" / "docx_rendu.js"), str(sortie_json),
                    str(RACINE / "Demarche_etape_par_etape.docx")], check=True, cwd=RACINE)


if __name__ == "__main__":
    construire()
