# -*- coding: utf-8 -*-
"""
Construit et exécute les notebooks 01 à 04 (une configuration par notebook).

Usage : python nb_config.py 1   (ou 2, 3, 4)

Les conclusions en Markdown sont rédigées à partir des résultats réels lus
dans le cache : aucun chiffre n'est saisi à la main.
"""
import dataclasses
import json
import re
import sys

import numpy as np

sys.path.insert(0, ".")
import analyse as an  # noqa: E402
import patchtst_pv as m  # noqa: E402
from nb_outils import ENTETE_CODE, construire  # noqa: E402

DEFS = {
    1: dict(etape="config1_patch", fichier="01_Config1_Patch", facteur="P",
            titre="Configuration 1 — Influence de la granularité temporelle (longueur de patch P)",
            question="Quelle taille de patch représente le mieux la dynamique horaire de la production PV ?",
            hypothese=("Un patch de 24 h correspond au cycle solaire et donne des jetons « journaliers » ; "
                       "un patch de 12 h sépare matin et après-midi ; un patch de 48 h réduit le nombre de jetons "
                       "(coût) mais agrège deux journées. Le pas entre patchs vaut P/2 (recouvrement 50 %)."),
            fixes="L = 168 h, traitement des canaux CA-X (attention inter-canaux + futur connu), H = 24 h"),
    2: dict(etape="config2_historique", fichier="02_Config2_Historique", facteur="L",
            titre="Configuration 2 — Influence de l'historique (fenêtre d'entrée L)",
            question="Combien d'heures de passé le modèle doit-il voir ?",
            hypothese=("L = 96 h (4 jours) capte la persistance nuageuse récente ; L = 168 h (1 semaine) ajoute un "
                       "cycle hebdomadaire complet ; L = 336 h (2 semaines) donne plus de contexte mais expose "
                       "davantage aux changements de régime et double le coût d'attention."),
            fixes="P retenu à la configuration 1, canaux CA-X, H = 24 h"),
    3: dict(etape="config3_canaux", fichier="03_Config3_Canaux_CI_CA", facteur="canaux",
            titre="Configuration 3 — Indépendance des canaux : PatchTST-CI vs PatchTST-CA",
            question="Les covariables (réseau, cycliques, géométrie solaire) améliorent-elles la prévision, et comment les injecter ?",
            hypothese=("**CI** : encodeur partagé, seul le canal cible alimente la tête : les exogènes n'ont aucune "
                       "influence (équivalent univarié). **CA** : attention inter-canaux à chaque patch : la cible "
                       "« interroge » les exogènes passés. **X** : les covariables déterministes FUTURES (heure, saison, "
                       "géométrie solaire de l'instant prévu) corrigent la prévision pas à pas. **g** : porte physique : "
                       "la prévision est forcée à 0 quand le soleil est couché."),
            fixes="P et L retenus aux configurations 1 et 2, H = 24 h"),
    4: dict(etape="config4_horizon", fichier="04_Config4_Horizon", facteur="H",
            titre="Configuration 4 — Horizon de prédiction H = 1, 6, 12, 24, 168 h",
            question="Comment la qualité et la stabilité évoluent-elles avec l'horizon ?",
            hypothese=("L'erreur croît avec l'horizon jusqu'au cycle journalier ; au-delà (168 h), l'information "
                       "nuageuse récente est perdue et la prévision tend vers une climatologie conditionnelle. "
                       "La comparaison à la persistance est faite au même horizon."),
            fixes="P, L et traitement des canaux retenus aux configurations 1 à 3"),
}


def configs_de(num: int) -> list:
    """Reconstruit exactement la liste des configurations de run_experiences.py."""
    sel = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())

    def parse(nom):
        # ex. "PatchTST-CAX_P24_L168_H24" → mode "CAX", P=24, L=168, H=24
        mode = nom.split("-")[1].split("_")[0]
        parties = nom.split("_")
        return m.ConfigPatchTST(P=int(parties[1][1:]), L=int(parties[2][1:]), H=int(parties[3][1:]),
                                mode_canaux=mode[:2], futur_connu="X" in mode[2:], porte_physique="g" in mode[2:])
    base = m.ConfigPatchTST(P=24, L=168, H=24, mode_canaux="CA", futur_connu=True)
    if num == 1:
        return [dataclasses.replace(base, P=p) for p in (12, 24, 48)]
    if num == 2:
        c1 = parse(sel["config1_patch"])
        return [dataclasses.replace(c1, L=l) for l in (96, 168, 336)]
    if num == 3:
        c2 = parse(sel["config2_historique"])
        return [dataclasses.replace(c2, mode_canaux=a, futur_connu=b, porte_physique=c)
                for a, b, c in [("CI", False, False), ("CI", True, False), ("CA", False, False),
                                ("CA", True, False), ("CA", True, True)]]
    c3 = parse(sel["config3_canaux"])
    return [dataclasses.replace(c3, H=h) for h in (1, 6, 12, 24, 168)]


def conclusion(num: int) -> str:
    """Rédige la conclusion à partir des résultats en cache."""
    jeu = m.construire_jeu()
    cfgs = configs_de(num)
    res = [m.evaluer_configuration(c, jeu) for c in cfgs]
    tab = m.tableau_comparatif(res).set_index("configuration")
    retenu = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())[DEFS[num]["etape"]]
    r = tab.loc[retenu]
    lignes = [f"### Conclusion de la configuration {num}", "",
              f"**Configuration retenue : `{retenu}`** (score de décision normalisé = {r.score_normalise:.3f} ; "
              f"score brut = {r.score:.2f}).", ""]
    if num == 4:
        lignes[2] = (f"**Meilleur score : `{retenu}`.** Le score compare ici des *tâches* différentes : l'erreur à 1 h est "
                     "mécaniquement plus faible qu'à 24 h. L'horizon n'est donc pas choisi par le score. Le modèle final "
                     "(notebook 05) est évalué à l'horizon opérationnel J+1 (24 h), et cette configuration décrit comment "
                     "la performance et la stabilité évoluent avec l'horizon ; la comparaison pertinente est le gain "
                     "par rapport à la persistance **au même horizon**.")
    # position vs persistance au même horizon
    for c, rr in zip(cfgs, res):
        _, vrai, orig = an.charger_predictions(rr["nom"])
        ref = an.lignes_references(jeu, orig, vrai, c.H).set_index("configuration")
        tab.loc[rr["nom"], "RMSE_persistance"] = ref.loc["Persistance J-1", "RMSE"]
    tab["gain_vs_persistance"] = 1 - tab["RMSE"] / tab["RMSE_persistance"]
    for nom, l in tab.iterrows():
        lignes.append(f"- `{nom}` : RMSE = {l.RMSE:.2f} ± {l.RMSE_std:.2f} MW, nRMSE = {l.nRMSE:.3f}, "
                      f"gain vs persistance = {100*l.gain_vs_persistance:+.1f} %, dégradation sous bruit = "
                      f"{100*l.degradation_bruit:+.1f} %, dérive 7 j = {l.drift_7j:.2f}, "
                      f"inférence = {l.inference_ms:.1f} ms, {l.n_parametres:,} paramètres.")
    lignes += ["", f"- Écart de RMSE entre la meilleure et la moins bonne variante : "
               f"{tab.RMSE.max() - tab.RMSE.min():.2f} MW ({100*(tab.RMSE.max()/tab.RMSE.min()-1):.1f} %).",
               f"- Toutes les variantes respectent le critère de robustesse (dégradation < 15 %) : "
               f"{'oui' if (tab.degradation_bruit < 0.15).all() else 'non'} ; "
               f"toutes respectent l'objectif de dérive < 1,2 : {'oui' if (tab.drift_7j < 1.2).all() else 'non'} ; "
               f"toutes respectent l'efficience (< 100 ms, < 50 Mo) : "
               f"{'oui' if ((tab.inference_ms < 100) & (tab.taille_Mo < 50)).all() else 'non'}."]
    texte = "\n".join(lignes)
    # typographie française : séparateur de milliers (espace fine) puis virgule décimale
    texte = re.sub(r"(\d),(\d{3})", "\\1\u202f\\2", texte)
    return re.sub(r"(\d)\.(\d)", r"\1,\2", texte)


def cellules(num: int) -> list:
    d = DEFS[num]
    return [
("md", f"""
# {d['fichier'][:2]} — {d['titre']}

**Question.** {d['question']}

**Hypothèses.** {d['hypothese']}

**Paramètres fixés.** {d['fixes']}.

**Protocole** (identique pour toutes les configurations — module `patchtst_pv`, section 8) :
1. entraînement sur Train (2019–2021), arrêt précoce sur Val (01–09/2022), **5 graines** ;
2. précision sur **toutes les origines horaires du Test** (10/2022–05/2023) : RMSE, MAE, nRMSE = RMSE / moyenne, R², MBE ;
3. **robustesse** : écart-type sur les graines et bruit gaussien 5 % sur les exogènes (rejet si ΔRMSE > 15 %) ;
4. **stabilité** : rolling forecast de 30 jours sans ré-entraînement, dérive = RMSE(J24–30) / RMSE(J1–7) (objectif < 1,2) ;
5. **efficience** : temps d'inférence d'un échantillon (< 100 ms) et taille des poids (< 50 Mo) ;
6. **score de décision** de `projet.md` (brut et normalisé) et tests de Diebold-Mariano contre la persistance et la moyenne.

> Les entraînements sont mis en cache (`cache/<configuration>/`) : supprimer le dossier correspondant relance
> l'entraînement complet depuis ce notebook.
"""),
("code", ENTETE_CODE + "import analyse as an, organigramme as org, json, dataclasses\nfrom nb_config import configs_de"),
("md", "## 1. Organigramme de la configuration"),
("code", f'org.tracer({num}, "Organigramme — {d["titre"].split(" — ")[0]}", "org_config{num}")'),
("md", "## 2. Configurations évaluées"),
("code", f"""
jeu = m.construire_jeu()
configs = configs_de({num})
pd.DataFrame([dict(nom=c.nom(), P=c.P, L=c.L, H=c.H, canaux=c.mode_canaux, futur_connu=c.futur_connu,
                   porte_physique=c.porte_physique, n_patchs=c.n_patchs) for c in configs])
"""),
("md", "## 3. Entraînement et évaluation (5 graines chacune)"),
("code", """
resultats = [m.evaluer_configuration(c, jeu) for c in configs]   # relit le cache s'il existe
detail = pd.DataFrame([dict(configuration=r["nom"], **{k: g[k] for k in
         ["graine", "epoques", "duree_s", "RMSE", "MAE", "nRMSE", "R2", "degradation_bruit", "drift_7j", "drift_J30_J1"]})
         for r in resultats for g in r["par_graine"]])
detail
"""),
("md", "## 4. Tableau comparatif des métriques (moyenne sur 5 graines) et références naïves"),
("code", """
tableau = m.tableau_comparatif(resultats)
_, vrai0, orig0 = an.charger_predictions(resultats[0]["nom"])
refs = an.lignes_references(jeu, orig0, vrai0, configs[0].H) if len({c.H for c in configs}) == 1 else None
colonnes = ["configuration", "RMSE", "RMSE_std", "MAE", "nRMSE", "nRMSE_jour", "R2", "MBE", "degradation_bruit",
            "drift_J30_J1", "drift_7j", "inference_ms", "taille_Mo", "n_parametres", "score", "score_normalise", "rejete_bruit"]
affiche = pd.concat([tableau[colonnes], refs], ignore_index=True) if refs is not None else tableau[colonnes]
affiche.to_csv(m.DOSSIER_TABLEAUX / "tableau_configNUM.csv", index=False)
affiche.round(4)
""".replace("NUM", str(num))),
("md", """
*Lecture des colonnes.* `RMSE_std` : écart-type sur les 5 graines (robustesse). `nRMSE_jour` : nRMSE restreint aux heures
diurnes. `degradation_bruit` : hausse relative du RMSE sous bruit de 5 % sur les exogènes. `drift_J30_J1` : ratio brut de
`projet.md` (sensible à la nébulosité de deux journées isolées), `drift_7j` : ratio lissé retenu pour le score.
`score` : formule brute de `projet.md` ; `score_normalise` : chaque critère ramené à [0, 1] entre les variantes comparées.
"""),
("code", """
dm = an.tests_dm(jeu, resultats)
dm.round(4)
"""),
("md", "## 5. Chapelet des graines et stabilité sur 30 jours"),
("code", f'an.fig_chapelet(resultats, jeu, "fig_config{num}_chapelet", "{d["titre"].split(" — ")[0]}")'),
("code", f'an.fig_apprentissage_et_derive(resultats, "fig_config{num}_apprentissage_derive")'),
("md", "## 6. Série observée vs prédite — meilleur modèle de la configuration"),
("code", f"""
retenu = an.selection_precedente("{d['etape']}")
print("Configuration retenue :", retenu)
res_retenu = [r for r in resultats if r["nom"] == retenu][0]
an.fig_observe_predit(res_retenu, jeu, "fig_config{num}_observe_predit")
an.fig_residus(res_retenu, jeu, "fig_config{num}_residus")
"""),
("md", conclusion(num)),
]


if __name__ == "__main__":
    num = int(sys.argv[1])
    print(construire(DEFS[num]["fichier"], cellules(num)))
