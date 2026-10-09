# -*- coding: utf-8 -*-
"""
Construit et exécute notebooks/05_Modele_Final.ipynb.

Étapes lourdes mises en cache (cache/final/) par `precalculer()` : Ridge,
courbe de bruit, validation croisée TimeSeriesSplit, ablation « indice de
capacité », XAI. Les conclusions Markdown sont rédigées à partir des
résultats réels.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import analyse as an  # noqa: E402
import patchtst_pv as m  # noqa: E402
from nb_config import configs_de  # noqa: E402
from nb_outils import ENTETE_CODE, construire  # noqa: E402

ETAPES = ["config1_patch", "config2_historique", "config3_canaux", "config4_horizon"]


def cfg_depuis_nom(nom):
    mode = nom.split("-")[1].split("_")[0]
    p = nom.split("_")
    return m.ConfigPatchTST(P=int(p[1][1:]), L=int(p[2][1:]), H=int(p[3][1:]), mode_canaux=mode[:2],
                            futur_connu="X" in mode[2:], porte_physique="g" in mode[2:])


def modele_final():
    """Architecture retenue aux configurations 1 à 3, évaluée à l'horizon
    opérationnel J+1 (24 h) — l'horizon n'est pas « sélectionné » par le score
    (comparer des horizons revient à comparer des tâches différentes)."""
    sel = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())
    return cfg_depuis_nom(sel["config3_canaux"])


def charger_modele(cfg, graine):
    mod = m.PatchTST(cfg)
    mod.load_state_dict(torch.load(m.DOSSIER_CACHE / cfg.nom() / f"poids_graine{graine}.pt"))
    return mod.eval()


def graine_validation(cfg):
    """Graine retenue pour le modèle livré : plus faible perte de VALIDATION
    (et non de test, pour ne pas sélectionner sur le Test)."""
    pertes = {g: pd.read_csv(m.DOSSIER_CACHE / cfg.nom() / f"historique_graine{g}.csv").perte_val.min()
              for g in m.GRAINES}
    return min(pertes, key=pertes.get)


def precalculer():
    jeu = m.construire_jeu()
    cfg = modele_final()
    res = m.evaluer_configuration(cfg, jeu)
    g = graine_validation(cfg)
    mod = charger_modele(cfg, g)

    def ridge():
        yp, yv, o, a = an.ridge_multisortie(jeu, cfg.L, cfg.H)
        np.save(m.DOSSIER_CACHE / "final" / "ridge_pred.npy", yp.astype(np.float32))
        return pd.DataFrame([dict(alpha=a, **m.metriques(yv, yp))])
    an.en_cache("ridge.csv", ridge)
    an.en_cache("bruit.csv", lambda: an.courbe_bruit(cfg, jeu, cfg.nom()))
    an.en_cache("validation_croisee.csv", lambda: an.validation_croisee(cfg, jeu))
    an.en_cache("indice_capacite.csv", lambda: an.evaluer_indice_capacite(cfg, jeu))
    an.en_cache("xai_occultation.csv", lambda: an.importance_occultation(mod, jeu)[0])
    an.en_cache("xai_gradients_integres.csv",
                lambda: an.gradients_integres(mod, jeu).rename("part").rename_axis("entree").reset_index())
    att = an.poids_attention_canaux(mod, jeu)
    if att is not None:
        an.en_cache("xai_attention.csv", lambda: att.rename("poids").rename_axis("canal").reset_index())
    return jeu, cfg, res, g


def conclusion():
    jeu, cfg, res, g = precalculer()
    s = res["synthese"]
    preds, vrai, orig = an.charger_predictions(cfg.nom())
    rp = m.metriques(vrai, m.reference_persistance(jeu, orig, cfg.H))["RMSE"]
    rmoy = m.metriques(vrai, m.reference_moyenne(jeu, orig, cfg.H))["RMSE"]
    rr = pd.read_csv(m.DOSSIER_CACHE / "final" / "ridge.csv").iloc[0]
    cv = pd.read_csv(m.DOSSIER_CACHE / "final" / "validation_croisee.csv")
    kc = pd.read_csv(m.DOSSIER_CACHE / "final" / "indice_capacite.csv")
    br = pd.read_csv(m.DOSSIER_CACHE / "final" / "bruit.csv").groupby("niveau").degradation.mean()
    occ = pd.read_csv(m.DOSSIER_CACHE / "final" / "xai_occultation.csv").set_index("groupe").hausse_RMSE
    sc = m.score_decision(s)
    g_rep = an.graine_representative(res)
    y_ridge = np.load(m.DOSSIER_CACHE / "final" / "ridge_pred.npy")
    p_ridge = m.test_diebold_mariano(preds[m.GRAINES.index(g_rep)] - vrai, y_ridge - vrai, h=cfg.H)[1]
    texte = f"""
## Conclusion — modèle final `{cfg.nom()}`

**Précision (40 %).** RMSE = {s['RMSE_moy']:.2f} ± {s['RMSE_std']:.2f} MW, nRMSE = {s['nRMSE_moy']:.3f},
R² = {s['R2_moy']:.3f} sur les 7 mois de Test. Références au même horizon : persistance J-1 {rp:.2f} MW
({100*(1-s['RMSE_moy']/rp):+.1f} %), moyenne 7 jours {rmoy:.2f} MW ({100*(1-s['RMSE_moy']/rmoy):+.1f} %),
Ridge multi-sortie {rr.RMSE:.2f} MW ({100*(1-s['RMSE_moy']/rr.RMSE):+.1f} %). **L'écart avec Ridge n'est pas significatif**
(Diebold-Mariano, p = {p_ridge:.2f}) : sans météorologie, la prévisibilité est surtout linéaire dans l'historique récent et la
géométrie solaire future, informations dont Ridge dispose aussi.

**Robustesse (30 %).** Écart-type inter-graines {s['RMSE_std']:.3f} MW ; dégradation sous bruit de 5 % sur les exogènes
{100*br.get(0.05, np.nan):+.2f} % (20 % de bruit : {100*br.get(0.2, np.nan):+.2f} %), très en deçà du seuil de rejet de 15 %.
Validation croisée TimeSeriesSplit (5 plis) : gain moyen vs persistance {100*cv.gain_vs_persistance.mean():+.1f} %
(min {100*cv.gain_vs_persistance.min():+.1f} %, max {100*cv.gain_vs_persistance.max():+.1f} %).

**Stabilité long terme (30 %).** Dérive lissée sur 30 jours sans ré-entraînement = {s['drift_7j_moy']:.2f}
(objectif < 1,2) ; ratio brut J30/J1 = {s['drift_J30_J1_moy']:.2f}.

**Efficience.** {s['inference_ms_moy']:.1f} ms par prévision, {s['taille_Mo_moy']:.2f} Mo, {int(s['n_parametres_moy']):,} paramètres.

**Score de décision** = {sc['score']:.2f} (bonus d'efficience {sc['bonus']:.1f}).

**Ablation « indice de capacité ».** RMSE = {kc.RMSE.mean():.2f} ± {kc.RMSE.std():.2f} MW contre {s['RMSE_moy']:.2f} MW
pour la normalisation RevIN seule, soit {100*(kc.RMSE.mean()/s['RMSE_moy']-1):+.1f} %.

**Explicabilité.** Masquer l'historique de la cible augmente le RMSE de {100*occ.get('Cible (historique PV)', np.nan):+.0f} %,
la géométrie solaire de {100*occ.get('Géométrie solaire', np.nan):+.0f} %, l'encodage horaire de {100*occ.get('Cycliques heure', np.nan):+.0f} %,
le contexte réseau de {100*occ.get('Réseau', np.nan):+.1f} % et le jour de semaine (témoin négatif) de {100*occ.get('Cycliques semaine', np.nan):+.1f} %.

Le modèle livré (`models/best_patchtst.pt`) correspond à la graine {g}, choisie sur la perte de **validation**.
"""
    import re
    texte = re.sub(r"(\d),(\d{3})", "\\1\u202f\\2", texte)
    return re.sub(r"(\d)\.(\d)", r"\1,\2", texte)


C_DEBUT = [
("md", """
# 05 — Modèle final : synthèse des configurations et évaluation approfondie

Ce notebook regroupe :
1. les **meilleurs modèles de chaque configuration** (P, L, canaux, horizon) ;
2. le **modèle final** (architecture retenue aux configurations 1 à 3, horizon opérationnel J+1 = 24 h) ;
3. sa comparaison aux références (persistance, moyenne 7 jours, **Ridge multi-sortie**, PatchTST univarié) avec tests de Diebold-Mariano ;
4. les graphes réel/prédit, les résidus et le **chapelet** des graines ;
5. la robustesse (courbe de bruit), la stabilité (rolling 30 jours, RMSE mensuel), la **validation croisée TimeSeriesSplit** ;
6. l'ablation « indice de capacité » ;
7. l'explicabilité (occultation, gradients intégrés, attention inter-canaux) ;
8. l'efficience, le score final et la sauvegarde `models/best_patchtst.pt`.

> **Pourquoi l'horizon final n'est-il pas choisi par le score ?** Comparer des horizons revient à comparer des tâches
> différentes : l'erreur à 1 h est mécaniquement plus faible qu'à 24 h. L'horizon est imposé par l'usage :
> la planification J+1 du dispatching. La configuration 4 décrit comment la performance évolue avec l'horizon.
"""),
("code", ENTETE_CODE + "import analyse as an, json, torch\nfrom nb05_final import cfg_depuis_nom, modele_final, graine_validation, charger_modele, ETAPES"),
("md", "## 1. Meilleur modèle de chaque configuration"),
("code", """
jeu = m.construire_jeu()
selection = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())
lignes = []
for etape in ETAPES:
    r = m.evaluer_configuration(cfg_depuis_nom(selection[etape]), jeu)
    t = m.tableau_comparatif([r]).iloc[0]
    lignes.append(dict(etape=etape, retenu=selection[etape], RMSE=t.RMSE, RMSE_std=t.RMSE_std, nRMSE=t.nRMSE,
                       R2=t.R2, degradation_bruit=t.degradation_bruit, drift_7j=t.drift_7j, inference_ms=t.inference_ms,
                       score=t.score))
meilleurs = pd.DataFrame(lignes); meilleurs.to_csv(m.DOSSIER_TABLEAUX / "meilleurs_par_configuration.csv", index=False)
meilleurs.round(4)
"""),
("md", "## 2. Modèle final et comparaison aux références"),
("code", """
cfg = modele_final()
res = m.evaluer_configuration(cfg, jeu)
preds, vrai, orig = an.charger_predictions(cfg.nom())
print("Modèle final :", cfg.nom(), "|", cfg)
ridge = pd.read_csv(m.DOSSIER_CACHE / "final" / "ridge.csv")
y_ridge = np.load(m.DOSSIER_CACHE / "final" / "ridge_pred.npy")
res_ci = m.evaluer_configuration(dataclasses.replace(cfg, mode_canaux="CI", futur_connu=False, porte_physique=False), jeu)
refs = {"Persistance J-1": m.reference_persistance(jeu, orig, cfg.H), "Moyenne 7 jours": m.reference_moyenne(jeu, orig, cfg.H),
        "Ridge multi-sortie": y_ridge}
lignes = [dict(modele=k, **m.metriques(vrai, v)) for k, v in refs.items()]
for r in [res_ci, res]:
    s = r["synthese"]
    lignes.append(dict(modele=r["nom"] + " (moy. 5 graines)", RMSE=s["RMSE_moy"], MAE=s["MAE_moy"], nRMSE=s["nRMSE_moy"],
                       R2=s["R2_moy"], MBE=s["MBE_moy"]))
comparaison = pd.DataFrame(lignes); comparaison.to_csv(m.DOSSIER_TABLEAUX / "comparaison_references.csv", index=False)
comparaison.round(4)
""".replace("res_ci = m.evaluer", "import dataclasses\nres_ci = m.evaluer")),
("code", """
# Tests de Diebold-Mariano (graine représentative du modèle final contre chaque référence)
g_rep = an.graine_representative(res)
e_mod = preds[m.GRAINES.index(g_rep)] - vrai
dm = []
for k, v in refs.items():
    stat, p = m.test_diebold_mariano(e_mod, v - vrai, h=cfg.H)
    dm.append(dict(reference=k, DM=stat, p_valeur=p, skill_score=1 - np.sqrt((e_mod**2).mean()) / np.sqrt(((v - vrai)**2).mean())))
pd.DataFrame(dm).round(4)
"""),
("md", "## 3. Chapelet des graines, série observée vs prédite et résidus"),
("code", """
fig, ax = plt.subplots(figsize=(7.5, 3.4))
groupes = {an.etiquette(r["nom"]): [g["RMSE"] for g in r["par_graine"]] for r in [res_ci, res]}
ax.boxplot(list(groupes.values()), widths=0.4, showfliers=False, medianprops=dict(color=fs.COULEURS["modele"]))
for i, d in enumerate(groupes.values()):
    ax.scatter(np.full(len(d), i + 1) + np.linspace(-0.07, 0.07, len(d)), d, s=16, color=fs.COULEURS["observe"], zorder=3)
styles = {"Persistance J-1": ("--", fs.COULEURS["persistance"]), "Moyenne 7 jours": (":", fs.COULEURS["moyenne"]),
          "Ridge multi-sortie": ("-.", "#27ae60")}
for k, v in refs.items():
    r = m.metriques(vrai, v)["RMSE"]; ax.axhline(r, ls=styles[k][0], color=styles[k][1], lw=1.1, label=f"{k} ({r:.2f})")
ax.set_xticks(range(1, len(groupes) + 1), list(groupes.keys())); ax.set_ylabel("RMSE test (MW)")
ax.legend(fontsize=7, loc="upper right"); ax.set_title("Chapelet des 5 graines — modèle final vs univarié et références")
fs.sauver(fig, "fig_final_chapelet")
an.fig_observe_predit(res, jeu, "fig_final_observe_predit")
an.fig_observe_predit(res, jeu, "fig_final_observe_predit_saison_seche", jours=10, debut_jour=150)
an.fig_residus(res, jeu, "fig_final_residus")
"""),
("md", "## 4. Robustesse : dégradation en fonction du niveau de bruit sur les exogènes"),
("code", """
bruit = pd.read_csv(m.DOSSIER_CACHE / "final" / "bruit.csv")
b = bruit.groupby("niveau").degradation.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.errorbar(100 * b.index, 100 * b["mean"], yerr=100 * b["std"], marker="o", color=fs.COULEURS["modele"], capsize=3)
ax.axhline(15, color="k", ls="--", lw=0.9, label="seuil de rejet (15 %)")
ax.axvline(5, color=fs.COULEURS["persistance"], ls=":", lw=0.9, label="test de projet.md (5 %)")
ax.set_xlabel("écart-type du bruit (% de l'écart-type Train)"); ax.set_ylabel("hausse du RMSE (%)"); ax.legend(fontsize=7)
ax.set_title("Robustesse au bruit sur les covariables")
fs.sauver(fig, "fig_final_bruit")
(100 * b).round(3)
"""),
("md", "## 5. Stabilité long terme : rolling 30 jours et RMSE mensuel sur tout le Test"),
("code", """
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
rj = np.array([g["rmse_journaliers"] for g in res["par_graine"]])
k30 = orig < jeu.i_test + 24 * 30
rj_p = m.rmse_journaliers(vrai[k30], refs["Persistance J-1"][k30], orig[k30], jeu)
jours = np.arange(1, rj.shape[1] + 1)
ax[0].fill_between(jours, rj.min(0), rj.max(0), color=fs.COULEURS["modele"], alpha=0.2, label="min–max 5 graines")
ax[0].plot(jours, rj.mean(0), color=fs.COULEURS["modele"], marker=".", label=an.etiquette(cfg.nom()))
ax[0].plot(jours, rj_p.values, color=fs.COULEURS["persistance"], ls="--", label="persistance J-1")
ax[0].set_xlabel("jour (sans ré-entraînement)"); ax[0].set_ylabel("RMSE journalier (MW)"); ax[0].legend(fontsize=7)
ax[0].set_title("Rolling forecast 30 jours")
mm = {k: an.rmse_mensuel(vrai, v, orig, jeu) for k, v in refs.items()}
mm[an.etiquette(cfg.nom())] = an.rmse_mensuel(vrai, preds[m.GRAINES.index(g_rep)], orig, jeu)
couleurs = {"Persistance J-1": fs.COULEURS["persistance"], "Moyenne 7 jours": fs.COULEURS["moyenne"],
            "Ridge multi-sortie": "#27ae60", an.etiquette(cfg.nom()): fs.COULEURS["modele"]}
for k, s in mm.items():
    ax[1].plot(range(len(s)), s.values, marker="o", label=k, color=couleurs[k],
               ls="--" if k == "Persistance J-1" else "-")
ax[1].set_xticks(range(len(s)), s.index, rotation=30); ax[1].set_ylabel("RMSE mensuel (MW)"); ax[1].legend(fontsize=7)
ax[1].set_title("Stabilité sur les 7 mois de Test")
fs.sauver(fig, "fig_final_stabilite")
pd.DataFrame(mm).round(3)
"""),
("md", """
## 6. Validation croisée temporelle (TimeSeriesSplit, 5 plis à fenêtre croissante)

Pour chaque pli, la normalisation est **ré-ajustée** sur la partie Train du pli (R2) ; 15 % de la fenêtre
d'apprentissage servent à l'arrêt précoce ; le bloc suivant sert de test. Les plis couvrent 2019–2023 :
on vérifie ainsi que les performances ne dépendent pas d'une période particulière.
"""),
("code", """
cv = pd.read_csv(m.DOSSIER_CACHE / "final" / "validation_croisee.csv")
cv.to_csv(m.DOSSIER_TABLEAUX / "validation_croisee.csv", index=False)
display(cv.round(4))
print(f"Gain moyen vs persistance : {100*cv.gain_vs_persistance.mean():+.1f} % ± {100*cv.gain_vs_persistance.std():.1f} %")
"""),
("md", """
## 7. Ablation : « indice de capacité » causal

Variante : le modèle apprend k = PV_total / C(t), où C(t) est le quantile 99 % des maxima journaliers des 30 jours
**précédents** (capacité apparente, mise à jour causalement après chaque mise en service). La prévision est remise
en MW en multipliant par C(t). Même configuration, 5 graines.
"""),
("code", """
kc = pd.read_csv(m.DOSSIER_CACHE / "final" / "indice_capacite.csv")
s = res["synthese"]
pd.DataFrame([dict(variante="RevIN seul (modèle final)", RMSE=s["RMSE_moy"], RMSE_std=s["RMSE_std"], drift_7j=s["drift_7j_moy"]),
              dict(variante="Indice de capacité + RevIN", RMSE=kc.RMSE.mean(), RMSE_std=kc.RMSE.std(), drift_7j=kc.drift_7j.mean())]).round(4)
"""),
("md", "## 8. Explicabilité"),
("code", """
occ = pd.read_csv(m.DOSSIER_CACHE / "final" / "xai_occultation.csv")
ig = pd.read_csv(m.DOSSIER_CACHE / "final" / "xai_gradients_integres.csv")
f_att = m.DOSSIER_CACHE / "final" / "xai_attention.csv"
att = pd.read_csv(f_att) if f_att.exists() else None
fig, ax = plt.subplots(1, 3 if att is not None else 2, figsize=(14, 4.2), gridspec_kw={"wspace": 0.75})
o = occ.sort_values("hausse_RMSE")
ax[0].barh(o.groupe, 100 * o.hausse_RMSE, color=[fs.COULEURS["modele"] if v > 0.05 else fs.COULEURS["persistance"] for v in o.hausse_RMSE])
ax[0].set_xlabel("hausse du RMSE si le groupe est masqué (%)"); ax[0].set_title("Importance par occultation")
i = ig.sort_values("part").tail(12)
ax[1].barh(i.entree, 100 * i.part, color=fs.COULEURS["moyenne"]); ax[1].set_xlabel("part de l'attribution (%)")
ax[1].set_title("Gradients intégrés (12 premières entrées)")
if att is not None:
    ax[2].bar(range(len(att)), att.poids, color=fs.COULEURS["observe"]); ax[2].axhline(1 / len(att), ls="--", color=fs.COULEURS["modele"], lw=0.9, label="uniforme")
    ax[2].set_xticks(range(len(att)), att.canal, rotation=60, fontsize=7); ax[2].legend(fontsize=7)
    ax[2].set_title("Attention inter-canaux (cible → canaux)")
fs.sauver(fig, "fig_final_xai")
display(occ.round(4)); display(ig.sort_values("part", ascending=False).round(4))
"""),
("md", "## 9. Efficience, score final et sauvegarde du modèle"),
("code", """
s = res["synthese"]
effi = pd.DataFrame([dict(inference_ms=s["inference_ms_moy"], taille_Mo=s["taille_Mo_moy"], n_parametres=int(s["n_parametres_moy"]),
                          duree_entrainement_s=s["duree_s_moy"], epoques=s["epoques_moy"], **m.score_decision(s))])
display(effi.round(3))
g = graine_validation(cfg)
mod = charger_modele(cfg, g)
(RACINE / "models").mkdir(exist_ok=True)
torch.save({"etat": mod.state_dict(), "config": dataclasses.asdict(cfg), "graine": g, "canaux": m.CANAUX,
            "moyenne_train": jeu.moyenne, "ecart_type_train": jeu.ecart_type,
            "decalage_solaire_h": jeu.decalage_solaire_h}, RACINE / "models" / "best_patchtst.pt")
taille = (RACINE / "models" / "best_patchtst.pt").stat().st_size / 2**20
print(f"Modèle sauvegardé : models/best_patchtst.pt (graine {g}, choisie sur la perte de validation) — {taille:.2f} Mo")
# contrôle : rechargement et prévision identique
verif = torch.load(RACINE / "models" / "best_patchtst.pt", weights_only=False)
mod2 = m.PatchTST(m.ConfigPatchTST(**verif["config"])); mod2.load_state_dict(verif["etat"]); mod2.eval()
o_ = orig[:50]
assert np.allclose(m.predire(mod, jeu, o_)[0], m.predire(mod2, jeu, o_)[0])
print("Rechargement vérifié : prévisions identiques.")
"""),
]


def cellules():
    return C_DEBUT + [("md", conclusion())]


if __name__ == "__main__":
    print(construire("05_Modele_Final", cellules()))
