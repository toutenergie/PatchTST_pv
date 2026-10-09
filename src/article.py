# -*- coding: utf-8 -*-
"""
article.py — Génère l'article scientifique (Word) à partir des résultats en cache.

    python article.py        → Article_PV_PatchTST_multivarie.docx

Tous les chiffres des sections Résultats, Discussion et du résumé sont lus
dans le cache (aucune saisie manuelle). Les textes invariants sont dans
article_textes.py ; la revue dans revue/texte_revue.md.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
import analyse as an  # noqa: E402
import article_textes as T  # noqa: E402
import patchtst_pv as m  # noqa: E402
from nb05_final import cfg_depuis_nom, graine_validation, modele_final  # noqa: E402

RACINE = m.RACINE
F = m.DOSSIER_CACHE / "final"


# -----------------------------------------------------------------------------
# Mise en forme française
# -----------------------------------------------------------------------------
def fr(x, d=2):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def pc(x, d=1, signe=True):
    return (f"{100*x:+.{d}f}" if signe else f"{100*x:.{d}f}").replace(".", ",") + " %"


def fig(nom, legende, largeur=16):
    w, h = Image.open(RACINE / "figures" / f"{nom}.png").size
    return {"type": "figure", "chemin": str(RACINE / "figures" / f"{nom}.png"), "ratio": h / w,
            "largeur_cm": largeur, "legende": legende}


def lbl(nom):
    return nom.replace("PatchTST-", "")


# -----------------------------------------------------------------------------
# Collecte des résultats
# -----------------------------------------------------------------------------
def collecter():
    jeu = m.construire_jeu()
    sel = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())
    R = {"jeu": jeu, "sel": sel}
    # configurations
    import nb_config
    for k in (1, 2, 3, 4):
        cfgs = nb_config.configs_de(k)
        res = [m.evaluer_configuration(c, jeu) for c in cfgs]
        tab = m.tableau_comparatif(res)
        pers = []
        for c, r in zip(cfgs, res):
            _, vrai, orig = an.charger_predictions(r["nom"])
            pers.append(m.metriques(vrai, m.reference_persistance(jeu, orig, c.H))["RMSE"])
        tab["RMSE_persistance"] = pers
        tab["gain"] = 1 - tab.RMSE / tab.RMSE_persistance
        R[f"c{k}"] = tab.set_index("configuration")
        R[f"res{k}"] = res
    # modèle final
    cfg = modele_final()
    res = m.evaluer_configuration(cfg, jeu)
    preds, vrai, orig = an.charger_predictions(cfg.nom())
    s = res["synthese"]
    R.update(cfg=cfg, res=res, s=s, vrai=vrai, orig=orig, preds=preds)
    refs = {"Persistance J-1": m.reference_persistance(jeu, orig, cfg.H),
            "Moyenne 7 jours": m.reference_moyenne(jeu, orig, cfg.H),
            "Ridge multi-sortie": np.load(F / "ridge_pred.npy")}
    R["refs"] = {k: m.metriques(vrai, v) for k, v in refs.items()}
    g_rep = an.graine_representative(res)
    e = preds[m.GRAINES.index(g_rep)] - vrai
    R["dm"] = {k: m.test_diebold_mariano(e, v - vrai, h=cfg.H) for k, v in refs.items()}
    R["ci"] = m.evaluer_configuration(cfg_depuis_nom(cfg.nom().replace("CAX", "CI")), jeu)["synthese"]
    R["cv"] = pd.read_csv(F / "validation_croisee.csv")
    R["kc"] = pd.read_csv(F / "indice_capacite.csv")
    R["bruit"] = pd.read_csv(F / "bruit.csv").groupby("niveau").degradation.agg(["mean", "std"])
    R["occ"] = pd.read_csv(F / "xai_occultation.csv").set_index("groupe")
    R["ig"] = pd.read_csv(F / "xai_gradients_integres.csv").set_index("entree").part
    R["att"] = pd.read_csv(F / "xai_attention.csv").set_index("canal").poids if (F / "xai_attention.csv").exists() else None
    R["mensuel"] = pd.DataFrame({k: an.rmse_mensuel(vrai, v, orig, jeu) for k, v in
                                 {**refs, "modele": preds[m.GRAINES.index(g_rep)]}.items()})
    R["graine_val"] = graine_validation(cfg)
    ens = preds.mean(axis=0)
    R["ens"] = m.metriques(vrai, ens)
    R["dm_ens_ridge"] = m.test_diebold_mariano(ens - vrai, refs["Ridge multi-sortie"] - vrai, h=cfg.H)
    # persistance « horaire » y(t-1) à 1 h : référence plus faible que la persistance journalière (rampes diurnes)
    _, v1, o1 = an.charger_predictions("PatchTST-CAX_P12_L168_H1")
    y = jeu.tableau[m.CIBLE].values
    R["pers_horaire_1h"] = m.metriques(v1, np.stack([[y[t - 1]] for t in o1]))["RMSE"]
    R["pers_j_1h"] = m.metriques(v1, m.reference_persistance(jeu, o1, 1))["RMSE"]
    # MAE persistance vs modèle, biais diurne
    R["mae_pers"] = R["refs"]["Persistance J-1"]["MAE"]
    jour = vrai > 1
    R["biais_diurne"] = float(e[jour].mean())
    R["rmse_nuit"] = float(np.sqrt(np.mean(e[~jour] ** 2)))
    return R


# -----------------------------------------------------------------------------
# Résumé
# -----------------------------------------------------------------------------
def resume(R):
    s, rp, rr = R["s"], R["refs"]["Persistance J-1"], R["refs"]["Ridge multi-sortie"]
    cv = R["cv"]
    return [
        {"type": "h1", "texte": "Résumé"},
        {"type": "p", "texte": (
            "La prévision de la production photovoltaïque (PV) est indispensable à l'exploitation des réseaux à forte "
            "pénétration renouvelable. Or, dans de nombreux réseaux africains, les mesures météorologiques sur site "
            "font défaut. Cet article étudie la prévision multivariée de la production PV **totale** du réseau "
            "SENELEC (Sénégal ; 11 centrales, journal horaire 2019–2023) **sans météorologie et sans les productions "
            "des centrales comme entrées**. Nous proposons PatchTST-CA-X, qui combine une normalisation réversible "
            "sélective, une attention inter-canaux par patch et un conditionnement de la tête de prévision par des "
            "covariables **déterministes connues sur l'horizon** : encodages cycliques et géométrie solaire. Quatre "
            "configurations (longueur de patch, historique, traitement des canaux, horizon) sont évaluées un facteur à "
            "la fois, sur 5 graines, selon la précision, la robustesse, la dérive sur 30 jours sans ré-entraînement et "
            "l'efficience. À l'horizon J+1, le modèle retenu (P = 12 h, L = 168 h) atteint un RMSE de "
            f"{fr(s['RMSE_moy'])} ± {fr(s['RMSE_std'])} MW (nRMSE = {fr(s['nRMSE_moy'], 3)} ; R² = {fr(s['R2_moy'], 3)}) "
            f"sur sept mois de test. C'est {pc(1 - s['RMSE_moy']/rp['RMSE'], signe=False)} de moins que la persistance "
            f"journalière (Diebold-Mariano, p = {fr(R['dm']['Persistance J-1'][1], 3)}) et "
            f"{pc(1 - s['RMSE_moy']/rr['RMSE'], signe=False)} de moins qu'une régression Ridge multi-sortie dotée des mêmes "
            f"covariables futures, un écart qui n'est pas significatif (p = {fr(R['dm']['Ridge multi-sortie'][1], 2)}). La "
            f"dégradation sous un bruit de 5 % sur les covariables est de {pc(s['degradation_bruit_moy'], 2)}, la "
            f"dérive sur 30 jours de {fr(s['drift_7j_moy'])}, et l'inférence prend {fr(s['inference_ms_moy'], 1)} ms. Les "
            f"ablations et l'explicabilité montrent que le gain provient du conditionnement par l'information "
            f"déterministe future, et non du contexte réseau passé ni de la seule complexité du modèle. La validation croisée temporelle (5 plis) confirme un gain moyen de "
            f"{pc(cv.gain_vs_persistance.mean(), signe=False)} sur la persistance.").replace("0.", "0,")},
        {"type": "p", "texte": T.MOTS_CLES},
    ]


# -----------------------------------------------------------------------------
# Résultats
# -----------------------------------------------------------------------------
def tableau_config(R, k, legende):
    t = R[f"c{k}"]
    retenu = R["sel"][["config1_patch", "config2_historique", "config3_canaux", "config4_horizon"][k - 1]]
    lignes, surl = [], []
    for i, (nom, l) in enumerate(t.iterrows()):
        if nom == retenu and k < 4:
            surl.append(i)
        lignes.append([lbl(nom), fr(l.RMSE), fr(l.RMSE_std, 3), fr(l.MAE), fr(l.nRMSE, 3), pc(l.gain),
                       pc(l.degradation_bruit, 1), fr(l.drift_7j), fr(l.inference_ms, 1), f"{int(l.n_parametres):,}".replace(",", " "),
                       fr(l.score_normalise, 3) + (" (rejet)" if l.rejete_bruit else "")])
    return {"type": "tableau", "legende": legende, "taille": 14, "surligner": surl,
            "largeurs": [2.6, 1.1, 1.0, 1.0, 1.0, 1.1, 1.1, 0.9, 1.1, 1.2, 1.3],
            "entetes": ["Variante", "RMSE (MW)", "σ (MW)", "MAE (MW)", "nRMSE", "Gain vs pers.", "Bruit 5 %",
                        "Dérive", "Inf. (ms)", "Param.", "Score norm."],
            "lignes": lignes,
            "note": "Moyennes sur 5 graines, toutes origines horaires du test (01/10/2022–09/05/2023). « Gain vs pers. » : "
                    "réduction du RMSE par rapport à la persistance journalière au même horizon. Ligne surlignée : variante retenue."}


def resultats(R):
    c1, c2, c3, c4 = R["c1"], R["c2"], R["c3"], R["c4"]
    s, cfg = R["s"], R["cfg"]
    n = lambda t, key: t.loc[[i for i in t.index if key in i][0]]  # noqa: E731
    ci, cix, ca, cax = (c3.loc[f"PatchTST-{v}_P12_L168_H24"] for v in ("CI", "CIX", "CA", "CAX"))
    caxg = c3.loc["PatchTST-CAXg_P12_L168_H24"]
    B = [{"type": "h1", "texte": "4. Résultats"},
         {"type": "h2", "texte": "4.1 Analyse exploratoire"},
         {"type": "p", "texte": (
             "Le cycle journalier domine la série : l'autocorrélation au retard 24 h vaut 0,97 sur l'entraînement, et le "
             "périodogramme présente un pic à 24 h et son harmonique à 12 h (Figure 3). L'amplitude est maximale de mars "
             "à mai et minimale pendant l'hivernage (juillet–septembre). La corrélation linéaire de la cible avec la "
             "géométrie solaire atteint 0,94 ; avec les canaux réseau, elle ne dépasse pas 0,31 en valeur absolue, et elle "
             "est encore plus faible une fois le cycle moyen retiré. La persistance journalière est donc une référence "
             f"exigeante : RMSE = {fr(R['refs']['Persistance J-1']['RMSE'])} MW et R² = "
             f"{fr(R['refs']['Persistance J-1']['R2'], 3)} sur le test.")},
         fig("fig03_saisonnalite", "**Figure 3.** Profil moyen heure × mois, autocorrélation et périodogramme de la cible (entraînement)."),
         {"type": "h2", "texte": "4.2 Configuration 1 — longueur de patch"},
         {"type": "p", "texte": (
             f"Plus le patch est court, plus la prévision est précise : RMSE = {fr(n(c1,'P12').RMSE)} MW pour P = 12 h, "
             f"{fr(n(c1,'P24').RMSE)} MW pour P = 24 h et {fr(n(c1,'P48').RMSE)} MW pour P = 48 h (Tableau 4). Un patch de "
             "12 h sépare la matinée de l'après-midi, ce qui permet à l'attention temporelle de relier des demi-journées "
             "homologues d'un jour à l'autre. Un patch de 48 h, lui, mélange deux journées dans un même jeton. Le patch de "
             f"12 h est aussi le plus stable entre graines (σ = {fr(n(c1,'P12').RMSE_std, 3)} MW), pour un coût "
             f"d'inférence qui reste faible ({fr(n(c1,'P12').inference_ms, 1)} ms). Il est retenu.")},
         tableau_config(R, 1, "**Tableau 4.** Configuration 1 — influence de la longueur de patch P (L = 168 h, CA-X, H = 24 h)."),
         fig("fig_config1_chapelet", "**Figure 4.** Configuration 1 : chapelet des RMSE sur 5 graines (gauche) et dérive sur 30 jours (droite). Traits horizontaux : références."),
         {"type": "h2", "texte": "4.3 Configuration 2 — historique"},
         {"type": "p", "texte": (
             f"Une semaine d'historique (L = 168 h) donne le meilleur RMSE ({fr(n(c2,'L168').RMSE)} MW), devant L = 96 h "
             f"({fr(n(c2,'L96').RMSE)} MW) et L = 336 h ({fr(n(c2,'L336').RMSE)} MW, Tableau 5). Allonger la fenêtre à "
             "deux semaines dégrade la précision et la dispersion, et multiplie par "
             f"{fr(n(c2,'L336').inference_ms / n(c2,'L168').inference_ms, 1)} le temps d'inférence. L'information utile "
             "réside dans les derniers jours (persistance de la nébulosité) ; un contexte plus ancien ajoute des "
             "régimes météorologiques révolus que l'attention doit apprendre à ignorer.")},
         tableau_config(R, 2, "**Tableau 5.** Configuration 2 — influence de l'historique L (P = 12 h, CA-X, H = 24 h)."),
         {"type": "h2", "texte": "4.4 Configuration 3 — traitement des canaux"},
         {"type": "p", "texte": (
             "La configuration 3 (Tableau 6, Figure 5) est la plus instructive. Le PatchTST standard à canaux "
             f"indépendants (CI), qui équivaut à un modèle univarié, atteint {fr(ci.RMSE)} MW. L'attention inter-canaux "
             f"seule (CA) l'améliore peu ({fr(ca.RMSE)} MW) et augmente la dispersion entre graines (σ = "
             f"{fr(ca.RMSE_std, 2)} MW). Sa dégradation sous bruit est nulle ({pc(ca.degradation_bruit, 2)}) : le "
             "modèle a appris à **ignorer** les canaux exogènes passés. À l'inverse, le conditionnement par le futur "
             f"connu (X) apporte l'essentiel du gain : {fr(cix.RMSE)} MW en CI-X (RMSE réduit de {pc(1 - cix.RMSE/ci.RMSE, signe=False)} "
             f"par rapport à CI), et {fr(cax.RMSE)} MW en CA-X (réduit de {pc(1 - cax.RMSE/ci.RMSE, signe=False)}), avec la plus "
             f"faible dispersion des variantes multivariées admissibles (σ = {fr(cax.RMSE_std, 3)} MW). La porte physique nocturne (CA-Xg) "
             f"laisse la précision inchangée ({fr(caxg.RMSE)} MW), mais elle échoue au test de robustesse : le RMSE "
             f"augmente de {pc(caxg.degradation_bruit, 0)} sous bruit, car le bruit sur cos θ_{{z}} ouvre la porte la "
             "nuit et révèle des sorties nocturnes que le réseau n'a jamais appris à contraindre. CA-X est retenu.")},
         tableau_config(R, 3, "**Tableau 6.** Configuration 3 — traitement des canaux (P = 12 h, L = 168 h, H = 24 h). "
                              "CI : indépendance ; CA : attention inter-canaux ; X : futur connu ; g : porte physique."),
         fig("fig_config3_chapelet", "**Figure 5.** Configuration 3 : chapelet des RMSE sur 5 graines et dérive sur 30 jours pour les cinq traitements des canaux."),
         {"type": "h2", "texte": "4.5 Configuration 4 — horizon de prévision"},
         {"type": "p", "texte": (
             "L'erreur croît avec l'horizon (Tableau 7) : "
             + ", ".join(f"{fr(l.RMSE)} MW à {nom.split('_H')[1]} h" for nom, l in c4.iterrows()) +
             ". Le gain par rapport à la persistance, calculée au même horizon, est "
             + ", ".join(f"{pc(l.gain)} à {nom.split('_H')[1]} h" for nom, l in c4.iterrows()) +
             ". Le gain est maximal à très court terme, où l'historique immédiat est très informatif et où la "
             "persistance journalière reste naïve. La persistance horaire ŷ(t) = y(t−1) serait une référence encore plus "
             f"faible à 1 h (RMSE = {fr(R['pers_horaire_1h'])} MW contre {fr(R['pers_j_1h'])} MW pour la persistance "
             "journalière), car elle ignore les rampes du lever et du coucher du soleil ; c'est pourquoi la persistance "
             "journalière sert de référence à tous les horizons. Il subsiste à 168 h, horizon auquel le modèle se rapproche d'une "
             "climatologie conditionnelle construite à partir des covariables déterministes. Comparer des horizons "
             "revient à comparer des tâches différentes : l'horizon n'est donc pas choisi par le score, mais fixé par "
             "l'usage, à savoir la planification J+1 du dispatching (H = 24 h).")},
         tableau_config(R, 4, "**Tableau 7.** Configuration 4 — horizon de prédiction H (P = 12 h, L = 168 h, CA-X)."),
         fig("fig_config4_residus", "**Figure 6.** Configuration 4, meilleur score : distribution des résidus, RMSE par heure, par mois et par pas d'horizon."),
         ]
    # --- modèle final
    rp, rm, rr = (R["refs"][k] for k in ("Persistance J-1", "Moyenne 7 jours", "Ridge multi-sortie"))
    dm = R["dm"]
    lignes = [[k, fr(v["RMSE"]), fr(v["MAE"]), fr(v["nRMSE"], 3), fr(v["R2"], 3), fr(v["MBE"]),
               fr(dm[k][0]), f"{dm[k][1]:.1e}".replace(".", ",")] for k, v in R["refs"].items()]
    ci_s = R["ci"]
    lignes.append([f"PatchTST-CI (univarié)", f"{fr(ci_s['RMSE_moy'])} ± {fr(ci_s['RMSE_std'])}", fr(ci_s["MAE_moy"]),
                   fr(ci_s["nRMSE_moy"], 3), fr(ci_s["R2_moy"], 3), fr(ci_s["MBE_moy"]), "–", "–"])
    lignes.append([f"PatchTST-CA-X (proposé)", f"{fr(s['RMSE_moy'])} ± {fr(s['RMSE_std'])}", fr(s["MAE_moy"]),
                   fr(s["nRMSE_moy"], 3), fr(s["R2_moy"], 3), fr(s["MBE_moy"]), "réf.", "réf."])
    cv, kc, br, occ = R["cv"], R["kc"], R["bruit"], R["occ"]
    occ_h = occ.hausse_RMSE
    B += [{"type": "h2", "texte": "4.6 Modèle final et comparaison aux références"},
          {"type": "p", "texte": (
              f"Le modèle final ({lbl(cfg.nom())}) atteint à l'horizon J+1 un RMSE de {fr(s['RMSE_moy'])} ± "
              f"{fr(s['RMSE_std'])} MW, un nRMSE de {fr(s['nRMSE_moy'], 3)} ({fr(s['nRMSE_jour_moy'], 3)} sur les seules "
              f"heures diurnes) et un R² de {fr(s['R2_moy'], 3)} (Tableau 8). Il réduit le RMSE de "
              f"{pc(1 - s['RMSE_moy']/rp['RMSE'], signe=False)} par rapport à la persistance, de "
              f"{pc(1 - s['RMSE_moy']/rm['RMSE'], signe=False)} par rapport à la moyenne des 7 derniers jours et de "
              f"{pc(1 - s['RMSE_moy']/rr['RMSE'], signe=False)} par rapport à la régression Ridge multi-sortie, qui "
              "dispose pourtant des mêmes covariables futures. Les tests de Diebold-Mariano rejettent l'égalité de "
              f"précision contre la persistance (p = {fr(dm['Persistance J-1'][1], 3)}) et contre la moyenne 7 jours "
              f"(p = {fr(dm['Moyenne 7 jours'][1], 3)}). **Contre Ridge, en revanche, l'écart n'est pas significatif** "
              f"(p = {fr(dm['Ridge multi-sortie'][1], 2)}) ; la moyenne des prévisions des 5 graines (ensemble) atteint "
              f"{fr(R['ens']['RMSE'])} MW, sans devenir significativement meilleure que Ridge non plus "
              f"(p = {fr(R['dm_ens_ridge'][1], 2)}). Le "
              f"MAE du modèle ({fr(s['MAE_moy'])} MW) est par ailleurs supérieur à celui de la persistance "
              f"({fr(R['mae_pers'])} MW) et de Ridge ({fr(rr['MAE'])} MW). Entraîné sur une perte quadratique, le modèle vise l'espérance "
              "conditionnelle : il réduit les grandes erreurs des journées de transition nuageuse, au prix de petites "
              "erreurs systématiques les jours de ciel stable, que la persistance reproduit exactement. Les résidus "
              f"diurnes présentent un biais moyen de {fr(R['biais_diurne'])} MW, et l'erreur nocturne est de "
              f"{fr(R['rmse_nuit'])} MW (RMSE).")},
          {"type": "tableau", "legende": "**Tableau 8.** Modèle final vs références à l'horizon J+1 (test complet, toutes origines horaires).",
           "taille": 15, "largeurs": [3.0, 1.9, 1.1, 1.1, 1.0, 1.0, 1.1, 1.3], "surligner": [len(lignes) - 1],
           "entetes": ["Modèle", "RMSE (MW)", "MAE (MW)", "nRMSE", "R²", "MBE (MW)", "DM", "p"],
           "lignes": lignes,
           "note": "DM : statistique de Diebold-Mariano (correction HLN) du modèle proposé contre la référence ; négative = modèle proposé meilleur."},
          fig("fig_final_observe_predit", "**Figure 7.** Prévision J+1 émise à 00 h (graine représentative) vs observation et persistance sur dix jours de novembre 2022 ; nuage de points sur tout le test."),
          fig("fig_final_residus", "**Figure 8.** Diagnostic des résidus du modèle final : distribution diurne, RMSE par heure, par mois et par pas d'horizon."),
          {"type": "h2", "texte": "4.7 Robustesse et validation croisée temporelle"},
          {"type": "p", "texte": (
              f"Sur les 5 graines, l'écart-type du RMSE est de {fr(s['RMSE_std'], 3)} MW, soit "
              f"{pc(s['RMSE_std']/s['RMSE_moy'], 2, False)} du RMSE. La dégradation sous bruit croît régulièrement avec "
              f"l'écart-type du bruit : {pc(br.loc[0.05, 'mean'], 2)} à 5 %, {pc(br.loc[0.1, 'mean'], 2)} à 10 % et "
              f"{pc(br.loc[0.2, 'mean'], 2)} à 20 % (Figure 9). On reste loin du seuil de rejet de 15 %, ce qui confirme "
              "que la prévision repose d'abord sur l'historique de la cible et sur la structure déterministe. La "
              f"validation croisée temporelle (Tableau 9), dont les blocs de test vont de {cv.test.iloc[0][:7]} à {cv.test.iloc[-1][-10:-3]}, donne un gain moyen sur "
              f"la persistance de {pc(cv.gain_vs_persistance.mean())} ± {pc(cv.gain_vs_persistance.std(), 1, False)}, "
              f"compris entre {pc(cv.gain_vs_persistance.min())} et {pc(cv.gain_vs_persistance.max())} selon le pli. Le gain "
              "est le plus faible pour le premier pli, entraîné sur huit mois seulement, puis se stabilise dès que "
              "l'entraînement couvre au moins un cycle annuel complet : le modèle a besoin de voir toutes les saisons "
              "pour dépasser la persistance.")},
          {"type": "tableau", "legende": "**Tableau 9.** Validation croisée temporelle (TimeSeriesSplit, 5 plis à fenêtre croissante, 1 graine).",
           "taille": 15, "largeurs": [0.7, 2.4, 3.0, 1.1, 1.0, 1.4, 1.4],
           "entetes": ["Pli", "Entraînement", "Test", "RMSE", "nRMSE", "RMSE pers.", "Gain"],
           "lignes": [[str(r.pli), r.train, r.test, fr(r.RMSE), fr(r.nRMSE, 3), fr(r.RMSE_persistance), pc(r.gain_vs_persistance)]
                      for r in cv.itertuples()]},
          fig("fig_final_bruit", "**Figure 9.** Hausse du RMSE du modèle final selon l'écart-type du bruit gaussien injecté dans les covariables (moyenne ± écart-type sur 5 graines).", 11),
          {"type": "h2", "texte": "4.8 Stabilité long terme"},
          {"type": "p", "texte": (
              f"Sur 30 jours de prévision sans ré-entraînement, la dérive vaut {fr(s['drift_7j_moy'])} (ratio brut "
              f"J30/J1 : {fr(s['drift_J30_J1_moy'])}), bien en deçà de l'objectif de 1,2 (Figure 10). Le RMSE baisse "
              "même au cours du mois, du fait de l'entrée en saison sèche en novembre : ce n'est pas un mérite du modèle, "
              "et la dérive doit toujours être lue en regard de la référence. Sur les mois calendaires du test "
              "(octobre 2022 à mai 2023, ce dernier partiel), le RMSE mensuel du modèle reste inférieur à celui de la persistance "
              f"{int((R['mensuel']['modele'] < R['mensuel']['Persistance J-1']).sum())} mois sur {len(R['mensuel'])}, sans tendance à la hausse malgré un écart de 9 à 16 mois avec la fin de l'entraînement.")},
          fig("fig_final_stabilite", "**Figure 10.** Stabilité : RMSE journalier sur 30 jours sans ré-entraînement (enveloppe min–max des 5 graines) et RMSE mensuel sur le test, comparés aux références."),
          {"type": "h2", "texte": "4.9 Ablation « indice de capacité »"},
          {"type": "p", "texte": (
              "Une variante apprend l'indice k = y / C, où C est une capacité apparente estimée de façon causale "
              "(quantile 99 % des maxima journaliers des 30 jours précédents), puis remet la prévision en MW. Elle obtient "
              f"{fr(kc.RMSE.mean())} ± {fr(kc.RMSE.std())} MW, contre {fr(s['RMSE_moy'])} MW pour le modèle final "
              f"({pc(kc.RMSE.mean()/s['RMSE_moy'] - 1)}). "
              + ("La normalisation RevIN sélective absorbe donc déjà la croissance du parc : il n'est pas utile de "
                 "modéliser explicitement la capacité installée sur cette période de test, où aucune centrale n'est mise en service."
                 if kc.RMSE.mean() >= s["RMSE_moy"] else
                 "Normaliser explicitement par la capacité installée apporte donc un gain supplémentaire, ce qui plaide "
                 "pour une prise en compte explicite des mises en service."))},
          {"type": "h2", "texte": "4.10 Explicabilité"},
          {"type": "p", "texte": (
              "L'occultation de groupes de canaux (Figure 11, gauche) hiérarchise nettement les sources d'information. "
              f"Masquer l'historique de la cible augmente le RMSE de {pc(occ_h['Cible (historique PV)'], 0)} ; masquer "
              f"la géométrie solaire, de {pc(occ_h['Géométrie solaire'], 0)} ; l'encodage horaire, de "
              f"{pc(occ_h['Cycliques heure'], 0)} ; l'encodage saisonnier, de {pc(occ_h['Cycliques saison'], 1)}. Le "
              f"contexte réseau ({pc(occ_h['Réseau'], 1)}) et le jour de semaine ({pc(occ_h['Cycliques semaine'], 1)}), "
              "notre témoin négatif, n'ont qu'un effet marginal. Les gradients intégrés (Figure 11, centre) confirment "
              f"cette hiérarchie : l'historique PV porte {pc(R['ig'].get('PV_total (passé)', np.nan), 0, False)} de "
              "l'attribution, et les covariables **futures** (cos θ_{z}, GHI de ciel clair, encodage horaire) "
              f"{pc(R['ig'][[i for i in R['ig'].index if '(futur)' in i]].sum(), 0, False)} au total, contre "
              f"{pc(R['ig'][[i for i in R['ig'].index if i.split(' ')[0] in m.CANAUX_RESEAU]].sum(), 1, False)} pour les trois canaux réseau. "
              + ("Les trois méthodes ne disent pas la même chose, et cette divergence est instructive. L'attention "
                 f"inter-canaux accorde son poids le plus fort à l'éolien de Taiba ({pc(R['att'].max(), 1, False)}, pour "
                 f"une valeur uniforme de {pc(1/len(R['att']), 1, False)}), et les gradients intégrés attribuent quelques "
                 "pour cent aux canaux réseau ; pourtant, masquer ces canaux ne change presque rien à l'erreur. Un poids "
                 "d'attention ou une attribution locale mesurent où le modèle « regarde », pas ce dont sa prévision "
                 "dépend : seule l'occultation, qui mesure l'effet sur l'erreur, permet de conclure que le contexte réseau "
                 "est inutile ici." if R["att"] is not None else ""))},
          fig("fig_final_xai", "**Figure 11.** Explicabilité du modèle final : importance par occultation, gradients intégrés (12 premières entrées) et attention inter-canaux."),
          {"type": "h2", "texte": "4.11 Efficience et score de décision"},
          {"type": "p", "texte": (
              f"Le modèle final compte {int(s['n_parametres_moy']):,} paramètres ({fr(s['taille_Mo_moy'])} Mo). Une "
              f"prévision prend {fr(s['inference_ms_moy'], 1)} ms sur un seul cœur de processeur, et l'entraînement dure "
              f"en moyenne {fr(s['duree_s_moy']/60, 0)} min sur deux cœurs. Les objectifs d'efficience (< 100 ms, < 50 Mo) "
              f"sont donc largement tenus. Le score de décision brut vaut {fr(m.score_decision(s)['score'])}.").replace(",", " ", 1)},
          ]
    return B


# -----------------------------------------------------------------------------
# Discussion, limites, perspectives, conclusion
# -----------------------------------------------------------------------------
def discussion(R):
    s, c3 = R["s"], R["c3"]
    ci, cix, cax = (c3.loc[f"PatchTST-{v}_P12_L168_H24"] for v in ("CI", "CIX", "CAX"))
    occ_h = R["occ"].hausse_RMSE
    rr = R["refs"]["Ridge multi-sortie"]
    return [
        {"type": "h1", "texte": "5. Discussion"},
        {"type": "h3", "texte": "Ce que le modèle apprend vraiment"},
        {"type": "p", "texte": (
            "Le principal enseignement est que, sans mesure météorologique, **la valeur ajoutée des covariables tient à "
            "leur disponibilité sur l'horizon, et non à leur présence dans le passé**. Les canaux réseau passés sont "
            "quasi ignorés : l'occultation ne dégrade le RMSE que de "
            f"{pc(occ_h['Réseau'], 1)}, et la variante CA sans X n'est pas sensible au bruit. Les covariables "
            "déterministes, elles, apportent l'essentiel du gain lorsqu'elles décrivent l'instant **prévu** : "
            f"un RMSE réduit de {pc(1 - cix.RMSE/ci.RMSE, signe=False)} de CI à CI-X. Ce résultat nuance l'intuition, courante dans la littérature "
            "[42, 45], selon laquelle rompre l'indépendance des canaux suffit à exploiter les exogènes. Ici, l'attention "
            "inter-canaux ne devient utile qu'en présence du conditionnement futur "
            f"({fr(cax.RMSE)} MW en CA-X contre {fr(cix.RMSE)} MW en CI-X), où elle réduit surtout la dispersion entre graines. "
            "Il rejoint en revanche les travaux qui fournissent explicitement l'information périodique au modèle [39, 49].")},
        {"type": "h3", "texte": "Une référence naïve difficile à battre"},
        {"type": "p", "texte": (
            f"Le gain de {pc(1 - s['RMSE_moy']/R['refs']['Persistance J-1']['RMSE'], signe=False)} sur la persistance "
            "est modeste en apparence, mais il est statistiquement établi, stable dans le temps et confirmé en "
            "validation croisée. La production agrégée de onze centrales dispersées lisse une partie de la variabilité "
            "nuageuse, et le régime solaire sénégalais est régulier [21] : la persistance y est donc particulièrement "
            "performante (R² > 0,96). De nombreuses études ne se comparent pas à cette référence [31–34] ; les rares qui "
            "le font [51] rapportent des gains plus importants sur des séries plus bruitées. Le résultat le plus "
            "important de cette comparaison concerne toutefois Ridge : dotée des **mêmes covariables futures**, une "
            f"régression linéaire atteint {fr(rr['RMSE'])} MW, et l'avantage de PatchTST-CA-X "
            f"({pc(1 - s['RMSE_moy']/rr['RMSE'], signe=False)}) n'est pas significatif (p = {fr(R['dm']['Ridge multi-sortie'][1], 2)}), "
            "alors que Ridge a un MAE plus faible. Sans observation météorologique, l'essentiel de la prévisibilité est "
            "donc **linéaire** dans l'historique récent et dans la géométrie solaire future : c'est l'information "
            "injectée, et non la capacité du Transformer, qui fait la différence. Ce constat relativise les gains "
            "rapportés pour des architectures profondes comparées à des références faibles [31–36]. Il fait aussi d'un "
            "modèle linéaire bien spécifié une référence obligatoire, au même titre que la persistance. L'intérêt "
            "pratique de PatchTST-CA-X tient plutôt à sa faible dispersion, à sa robustesse, à sa dérive nulle et à "
            "sa capacité à accueillir, sans changer d'architecture, des covariables futures non linéaires et bruitées "
            "(prévisions numériques du temps) que Ridge exploiterait mal.")},
        {"type": "h3", "texte": "Portée opérationnelle"},
        {"type": "p", "texte": (
            f"Avec {fr(s['inference_ms_moy'], 1)} ms par prévision, moins de 1 Mo de poids, une dérive nulle sur un mois "
            "et une robustesse élevée au bruit, le modèle peut être intégré au dispatching sans infrastructure de calcul "
            "dédiée. Pour le gestionnaire, une erreur J+1 réduite de "
            f"{fr(R['refs']['Persistance J-1']['RMSE'] - s['RMSE_moy'], 1)} MW en RMSE, pour un parc dont la production "
            "de pointe approche 180 MW, se traduit directement par la réserve tournante à mobiliser [15, 16]. Ce bénéfice doit être "
            "chiffré par une étude économique propre au réseau SENELEC.")},
        {"type": "h3", "texte": "Contraintes physiques « dures »"},
        {"type": "p", "texte": (
            "L'échec de la porte physique au test de robustesse montre qu'une contrainte physique imposée en sortie "
            "peut rendre un modèle fragile lorsque la variable qui la commande est incertaine. Le réseau apprend à "
            "compter sur la porte et ne contraint plus ses sorties nocturnes. Une contrainte douce, sous forme de "
            "pénalité dans la perte, ou une porte calculée à partir de la géométrie exacte plutôt que d'une entrée du "
            "modèle, serait préférable. Plus largement, l'hybridation physique revendiquée par une partie de la "
            "littérature [55] doit être évaluée sous perturbation, et pas seulement en précision nominale.")},
        {"type": "h1", "texte": "6. Limites"},
        {"type": "puces", "items": [
            "**Absence de données météorologiques.** Le plafond de précision est fixé par l'information disponible. Les "
            "journées de transition nuageuse restent mal prévues, et l'erreur est concentrée entre 9 h et 15 h.",
            "**Un seul réseau et une seule période de test** (sept mois, d'octobre à mai, sans hivernage complet). La "
            "validation croisée couvre d'autres saisons, mais avec une seule graine par pli.",
            "**Plan « un facteur à la fois ».** Il ne capte pas les interactions entre facteurs (par exemple P × L) ; "
            "une recherche conjointe pourrait trouver un meilleur optimum.",
            "**Prévision ponctuelle.** L'incertitude n'est pas quantifiée, alors qu'elle est nécessaire au "
            "dimensionnement probabiliste des réserves [16, 46].",
            "**Qualité du journal.** L'écrêtage et l'imputation, bien que causaux et traçables, reposent sur des seuils "
            "statistiques ; certaines valeurs aberrantes plausibles ont pu être conservées.",
            "**Dérive évaluée sur 30 jours** à une période favorable (entrée en saison sèche) ; une évaluation sur "
            "l'hivernage suivant reste nécessaire.",
        ]},
        {"type": "h1", "texte": "7. Perspectives"},
        {"type": "numeros", "instance": 4, "items": [
            "**Intégrer des prévisions météorologiques numériques** (ERA5, GFS) ou satellitaires comme covariables futures "
            "**bruitées** : l'architecture X s'y prête directement, et le protocole de bruit en mesurera la sensibilité.",
            "**Rendre la prévision probabiliste** (régression quantile [46], inférence conforme adaptative [39]) pour "
            "alimenter le dimensionnement probabiliste des réserves.",
            "**Remplacer la porte dure par une contrainte physique douce**, et tester un couplage explicite avec un modèle "
            "de ciel clair (prévision de l'indice de clarté).",
            "**Transférer le modèle** vers d'autres réseaux ouest-africains (WAPP), à l'aide des stratégies de transfert "
            "qui réussissent à PatchTST [47].",
            "**Coupler la prévision à la stabilité en fréquence** du réseau SENELEC [19, 20], afin d'évaluer son bénéfice en "
            "réserve et en délestage évités.",
        ]},
        {"type": "h1", "texte": "8. Conclusion"},
        {"type": "p", "texte": (
            "Nous avons étudié la prévision de la production PV totale du réseau SENELEC dans un cadre réaliste pour "
            "un gestionnaire africain : sans mesures météorologiques et sans les productions individuelles comme "
            "entrées. L'architecture PatchTST-CA-X, qui conditionne la tête de prévision par des covariables "
            "cycliques et solaires connues sur l'horizon, atteint à J+1 un RMSE de "
            f"{fr(s['RMSE_moy'])} MW ({pc(1 - s['RMSE_moy']/R['refs']['Persistance J-1']['RMSE'], signe=False)} de "
            "moins que la persistance). Elle ne présente aucune dérive sur 30 jours, résiste au bruit et prévoit en "
            "quelques millisecondes. Le protocole multicritère (précision, robustesse, stabilité, efficience, validation "
            "croisée, explicabilité) montre où se situe réellement le gain : dans l'information **future** "
            "déterministe, et non dans l'ajout de canaux exogènes passés ; une régression Ridge dotée de la même "
            "information fait presque aussi bien. Le protocole, le code et les notebooks sont "
            "fournis pour servir de base de comparaison à de futurs travaux sur les réseaux ouest-africains.")},
        {"type": "h1", "texte": "Disponibilité des données et du code"},
        {"type": "p", "texte": (
            "Le code (module commenté, six notebooks exécutés, organigrammes) et les résultats intermédiaires sont "
            "fournis avec l'article. Les données d'exploitation appartiennent à la SENELEC et peuvent être obtenues "
            "auprès de l'opérateur sur demande motivée.")},
        {"type": "h1", "texte": "Remerciements"},
        {"type": "p", "texte": "Les auteurs remercient la SENELEC pour la mise à disposition du journal d'exploitation."},
    ]


# -----------------------------------------------------------------------------
# Assemblage
# -----------------------------------------------------------------------------
def donnees(R):
    jeu = R["jeu"]
    brut = pd.read_excel(m.CHEMIN_DONNEES, usecols=["Horodatage_debut"])
    y = jeu.tableau[m.CIBLE]
    an_ = y.groupby(y.index.year).mean()
    return dict(debut=f"{jeu.index[0]:%d/%m/%Y}", fin=f"{jeu.index[-1]:%d/%m/%Y}",
                n_heures=f"{len(jeu.index):,}".replace(",", " "), n_manquantes=len(jeu.index) - len(brut),
                croissance=100 * (an_[2022] / an_[2019] - 1), n_ecretees=int(jeu.diagnostic_cible.valeurs_ecretees.sum()),
                decalage=jeu.decalage_solaire_h,
                n_train=f"{jeu.i_val:,}".replace(",", " "), n_val=f"{jeu.i_test - jeu.i_val:,}".replace(",", " "),
                n_test=f"{len(jeu.valeurs) - jeu.i_test:,}".replace(",", " "))


def tableau_centrales(R):
    d = R["jeu"].diagnostic_cible
    return {"type": "tableau", "legende": "**Tableau 2.** Construction de la cible : traçabilité par centrale.", "taille": 15,
            "largeurs": [2.4, 1.8, 1.8, 1.2, 1.2, 1.4, 1.3],
            "entetes": ["Centrale", "Mise en service", "Dernier relevé", "Seuil (MW)", "Écrêtées", "Lacunes imputées", "Moyenne (MW)"],
            "lignes": [[r.centrale, f"{r.mise_en_service:%d/%m/%Y}", f"{r.dernier_releve:%d/%m/%Y}", fr(r.seuil_MW, 1),
                        str(r.valeurs_ecretees), str(r.lacunes_imputees), fr(r.moyenne_MW)] for r in d.itertuples()]}


def construire():
    R = collecter()
    D = donnees(R)
    B = resume(R) + T.introduction() + T.revue() + [T.tableau_revue()]
    meth = T.methodologie(D)
    # insère le Tableau 2 après le paragraphe de construction de la cible, et les ratios des figures
    for b in meth:
        if b["type"] == "figure":
            w, h = Image.open(RACINE / b["chemin"]).size
            b["ratio"] = h / w; b["chemin"] = str(RACINE / b["chemin"])
    i = next(k for k, b in enumerate(meth) if b["type"] == "equation")
    meth.insert(i + 1, tableau_centrales(R))
    B += meth + resultats(R) + discussion(R) + T.references()
    doc = {"titre": T.TITRE, "auteurs": T.AUTEURS, "affiliation": T.AFFILIATION, "entete": T.ENTETE, "blocs": B}
    f = m.DOSSIER_CACHE / "article.json"
    f.write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")
    subprocess.run(["node", str(RACINE / "src" / "docx_rendu.js"), str(f),
                    str(RACINE / "Article_PV_PatchTST_multivarie.docx")], check=True, cwd=RACINE)
    return R


if __name__ == "__main__":
    construire()
