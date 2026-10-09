# -*- coding: utf-8 -*-
"""
analyse.py — Fonctions d'analyse et de figures partagées par les notebooks 01 à 05.

    A. Chargement des résultats mis en cache
    B. Références naïves et tests de Diebold-Mariano
    C. Figures : chapelet des graines, courbes d'apprentissage, dérive 30 j,
       observé vs prédit, résidus
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures_style as fs
import patchtst_pv as m


# =============================================================================
# A. Chargement des résultats
# =============================================================================

def dossier(nom):
    return m.DOSSIER_CACHE / nom


def charger_predictions(nom: str):
    """Prévisions des 5 graines [g, n, H], observations [n, H], origines [n]."""
    d = dossier(nom)
    preds = np.stack([np.load(d / f"pred_graine{g}.npy") for g in m.GRAINES])
    return preds, np.load(d / "vrai.npy"), np.load(d / "origines.npy")


def graine_representative(res: dict) -> int:
    """Graine dont le RMSE est la médiane des 5 (utilisée pour les figures)."""
    rm = [(r["RMSE"], r["graine"]) for r in res["par_graine"]]
    return sorted(rm)[len(rm) // 2][1]


def selection_precedente(etape: str):
    """Configuration retenue à une étape (journal écrit par run_experiences.py)."""
    sel = json.loads((m.DOSSIER_CACHE / "selection.json").read_text())
    return sel.get(etape)


# =============================================================================
# B. Références naïves et tests statistiques
# =============================================================================

def lignes_references(jeu, orig, y_vrai, H) -> pd.DataFrame:
    """Persistance journalière et moyenne 7 jours sur les mêmes origines."""
    lignes = []
    for nom, f in [("Persistance J-1", m.reference_persistance), ("Moyenne 7 jours", m.reference_moyenne)]:
        yp = f(jeu, orig, H)
        mt = m.metriques(y_vrai, yp)
        lignes.append(dict(configuration=nom, RMSE=mt["RMSE"], RMSE_std=0.0, MAE=mt["MAE"],
                           nRMSE=mt["nRMSE"], R2=mt["R2"], MBE=mt["MBE"]))
    return pd.DataFrame(lignes)


def tests_dm(jeu, resultats: list[dict]) -> pd.DataFrame:
    """Diebold-Mariano de chaque configuration (graine représentative) contre
    la persistance J-1 et la moyenne 7 jours. Stat < 0 : le modèle est meilleur."""
    lignes = []
    for r in resultats:
        preds, vrai, orig = charger_predictions(r["nom"])
        H = vrai.shape[1]
        g = graine_representative(r)
        e = preds[m.GRAINES.index(g)] - vrai
        for nom_ref, f in [("persistance", m.reference_persistance), ("moyenne 7 j", m.reference_moyenne)]:
            stat, p = m.test_diebold_mariano(e, f(jeu, orig, H) - vrai, h=H)
            ss = 1 - np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean((f(jeu, orig, H) - vrai) ** 2))
            lignes.append(dict(configuration=r["nom"], reference=nom_ref, skill_score=ss,
                               DM=stat, p_valeur=p))
    return pd.DataFrame(lignes)


# =============================================================================
# C. Figures
# =============================================================================

def etiquette(nom: str) -> str:
    return nom.replace("PatchTST-", "")


def fig_chapelet(resultats, jeu, nom_fichier, titre):
    """Chapelet (boîte + points) du RMSE sur les 5 graines, avec les références."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    noms = [etiquette(r["nom"]) for r in resultats]
    for k, crit in enumerate(["RMSE", "drift_7j"]):
        donnees = [[g[crit] for g in r["par_graine"]] for r in resultats]
        ax[k].boxplot(donnees, widths=0.45, showfliers=False,
                      medianprops=dict(color=fs.COULEURS["modele"]))
        for i, d in enumerate(donnees):
            ax[k].scatter(np.full(len(d), i + 1) + np.linspace(-0.08, 0.08, len(d)), d, s=14,
                          color=fs.COULEURS["observe"], zorder=3)
        ax[k].set_xticks(range(1, len(noms) + 1), noms, rotation=20, ha="right")
    _, vrai, orig = charger_predictions(resultats[0]["nom"])
    H = vrai.shape[1]
    ref = lignes_references(jeu, orig, vrai, H)
    for _, l in ref.iterrows():
        ax[0].axhline(l.RMSE, ls="--" if "Pers" in l.configuration else ":", lw=1,
                      color=fs.COULEURS["persistance" if "Pers" in l.configuration else "moyenne"],
                      label=f"{l.configuration} ({l.RMSE:.2f})")
    ax[0].set_ylabel("RMSE test (MW)"); ax[0].legend(fontsize=7); ax[0].set_title("Précision et dispersion (5 graines)")
    ax[1].axhline(1.2, color=fs.COULEURS["modele"], ls="--", lw=1, label="seuil 1,2")
    ax[1].axhline(1.0, color="k", lw=0.6)
    ax[1].set_ylabel("dérive RMSE(J24–30)/RMSE(J1–7)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Stabilité : dérive sur 30 jours")
    fig.suptitle(titre, fontsize=10)
    fs.sauver(fig, nom_fichier)


def fig_apprentissage_et_derive(resultats, nom_fichier):
    """Courbes d'apprentissage (graine représentative) et RMSE journalier sur 30 jours."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for r in resultats:
        g = graine_representative(r)
        h = pd.read_csv(dossier(r["nom"]) / f"historique_graine{g}.csv")
        l, = ax[0].plot(h.epoque, h.perte_val, marker="o", ms=3, label=etiquette(r["nom"]))
        ax[0].plot(h.epoque, h.perte_train, ls=":", color=l.get_color())
        rj = np.mean([pg["rmse_journaliers"] for pg in r["par_graine"]], axis=0)
        ax[1].plot(np.arange(1, len(rj) + 1), rj, marker=".", label=etiquette(r["nom"]))
    ax[0].set_xlabel("époque"); ax[0].set_ylabel("MSE (unités normalisées)")
    ax[0].set_title("Apprentissage (trait plein : Val, pointillé : Train)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("jour du rolling forecast (sans ré-entraînement)"); ax[1].set_ylabel("RMSE journalier (MW)")
    ax[1].set_title("Rolling 30 jours — moyenne des 5 graines"); ax[1].legend(fontsize=7)
    fs.sauver(fig, nom_fichier)


def serie_continue(pred, vrai, orig, jeu, H):
    """Assemble des prévisions non chevauchantes émises toutes les H heures
    (à partir d'une origine à 00 h) en une série continue observé/prévu."""
    heures = jeu.index[orig].hour
    debut = int(np.argmax(heures == 0))
    sel = np.arange(debut, len(orig), H)
    idx = np.concatenate([np.arange(orig[i], orig[i] + H) for i in sel])
    return (pd.Series(np.concatenate([vrai[i] for i in sel]), index=jeu.index[idx]),
            pd.Series(np.concatenate([pred[i] for i in sel]), index=jeu.index[idx]))


def fig_observe_predit(res, jeu, nom_fichier, jours=10, debut_jour=35):
    """Observé vs prédit (graine représentative) + nuage de points + persistance."""
    preds, vrai, orig = charger_predictions(res["nom"])
    H = vrai.shape[1]
    g = graine_representative(res)
    yv, yp = serie_continue(preds[m.GRAINES.index(g)], vrai, orig, jeu, H)
    ypers = pd.Series(jeu.tableau[m.CIBLE].shift(24).reindex(yv.index).values, index=yv.index)
    t0 = yv.index[0].normalize() + pd.Timedelta(days=debut_jour)
    f = (yv.index >= t0) & (yv.index < t0 + pd.Timedelta(days=jours))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.6, 1]})
    ax[0].plot(yv[f].index, yv[f], color=fs.COULEURS["observe"], lw=1.3, label="observé")
    ax[0].plot(yp[f].index, yp[f], color=fs.COULEURS["modele"], lw=1.2, label=f"{etiquette(res['nom'])} (graine {g})")
    ax[0].plot(ypers[f].index, ypers[f], color=fs.COULEURS["persistance"], lw=0.9, ls="--", label="persistance J-1")
    ax[0].set_ylabel("PV_total (MW)"); ax[0].legend(fontsize=7, ncol=3, loc="upper left")
    ax[0].set_title(f"Observé vs prédit — prévisions émises toutes les {H} h ({jours} jours du Test)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].hexbin(yv, yp, gridsize=40, mincnt=1, cmap="Greys", bins="log")
    lim = [0, max(yv.max(), yp.max()) * 1.02]
    ax[1].plot(lim, lim, color=fs.COULEURS["modele"], lw=1)
    mt = m.metriques(yv, yp)
    ax[1].set_title(f"Test, prévisions émises toutes les {H} h\nR² = {mt['R2']:.3f}, RMSE = {mt['RMSE']:.2f} MW")
    ax[1].set_xlabel("observé (MW)"); ax[1].set_ylabel("prédit (MW)")
    fs.sauver(fig, nom_fichier)


def fig_residus(res, jeu, nom_fichier):
    """Diagnostic des résidus : distribution, par heure, par mois, par pas d'horizon."""
    preds, vrai, orig = charger_predictions(res["nom"])
    H = vrai.shape[1]
    g = graine_representative(res)
    e = preds[m.GRAINES.index(g)] - vrai
    heures = (jeu.index[orig].hour.values[:, None] + np.arange(H)[None]) % 24
    mois = jeu.index[orig].to_period("M").astype(str)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    jour = (vrai > 1)
    ax[0].hist(e[jour], bins=80, color=fs.COULEURS["moyenne"], density=True)
    ax[0].set_title(f"Résidus diurnes (moy. {e[jour].mean():+.2f} MW)"); ax[0].set_xlabel("prédit − observé (MW)")
    rmse_h = [np.sqrt(np.mean(e[heures == h] ** 2)) for h in range(24)]
    ax[1].bar(range(24), rmse_h, color=fs.COULEURS["modele"]); ax[1].set_title("RMSE par heure"); ax[1].set_xlabel("heure")
    em = pd.Series((e ** 2).mean(axis=1), index=mois).groupby(level=0).mean().pipe(np.sqrt)
    ax[2].plot(range(len(em)), em.values, marker="o", color=fs.COULEURS["observe"])
    ax[2].set_xticks(range(len(em)), em.index, rotation=45, fontsize=7); ax[2].set_title("RMSE mensuel (Test)")
    ax[3].plot(np.arange(1, H + 1), np.sqrt((e ** 2).mean(axis=0)), color=fs.COULEURS["modele"], marker="." if H < 50 else None)
    ax[3].set_title("RMSE par pas d'horizon"); ax[3].set_xlabel("pas h")
    fs.sauver(fig, nom_fichier)


# =============================================================================
# D. Référence linéaire : Ridge multi-sortie
# =============================================================================

def ridge_multisortie(jeu, L: int, H: int, alphas=(0.1, 1, 10, 100, 1000)):
    """Régression Ridge directe multi-horizon (référence linéaire forte).

    Entrées : L dernières valeurs de la cible + covariables déterministes des
    H pas futurs. Alpha choisi sur Val ; réajusté sur Train seul (R2).
    Retourne les prévisions Test (MW) sur toutes les origines horaires.
    """
    from sklearn.linear_model import Ridge
    v = jeu.valeurs
    i_det = [m.CANAUX.index(c) for c in m.CANAUX_DETERMINISTES]

    def X_y(orig):
        X = np.stack([np.concatenate([v[t - L:t, 0], v[t:t + H][:, i_det].ravel()]) for t in orig])
        y = np.stack([v[t:t + H, 0] for t in orig])
        return X, y
    o_tr = m.origines(0, jeu.i_val, L, H, 2)
    o_va = m.origines(jeu.i_val, jeu.i_test, L, H, 3)
    o_te = m.origines(jeu.i_test, len(v), L, H, 1)
    Xtr, ytr = X_y(o_tr); Xva, yva = X_y(o_va)
    scores = {a: np.mean((Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - yva) ** 2) for a in alphas}
    a = min(scores, key=scores.get)
    mod = Ridge(alpha=a).fit(Xtr, ytr)
    Xte, _ = X_y(o_te)
    yp = np.clip(jeu.vers_mw(mod.predict(Xte)), 0, None)
    yv = np.stack([jeu.tableau[m.CIBLE].values[t:t + H] for t in o_te])
    return yp, yv, o_te, a


# =============================================================================
# E. Validation croisée temporelle (TimeSeriesSplit) de la configuration finale
# =============================================================================

def jeu_pour_pli(jeu, i_fin_train: int, i_fin_val: int, i_fin_test: int):
    """Copie du jeu de données pour un pli : normalisation ré-ajustée sur la
    seule partie Train du pli (aucune information postérieure, R2)."""
    import copy as _copy
    brut = jeu.tableau.values.astype(np.float64)[:i_fin_test]
    mu, sd = brut[:i_fin_train].mean(0), brut[:i_fin_train].std(0)
    sd[sd < 1e-8] = 1.0
    j = _copy.copy(jeu)
    j.tableau = jeu.tableau.iloc[:i_fin_test]
    j.valeurs = ((brut - mu) / sd).astype(np.float32)
    j.moyenne, j.ecart_type = mu, sd
    j.i_val, j.i_test = i_fin_train, i_fin_val
    return j


def validation_croisee(cfg, jeu, n_plis: int = 5, graine: int = 0):
    """TimeSeriesSplit (fenêtre croissante) sur l'ensemble de la série.

    Pour chaque pli : Train = début de série → 85 % du segment d'apprentissage,
    Val = 15 % restants (arrêt précoce), Test = bloc suivant. Blocs alignés sur
    des journées entières. Chaque pli est comparé à la persistance J-1.
    """
    from sklearn.model_selection import TimeSeriesSplit
    jours = np.arange(len(jeu.valeurs) // 24)
    lignes = []
    for k, (app, te) in enumerate(TimeSeriesSplit(n_splits=n_plis).split(jours)):
        fin_app, fin_te = (app[-1] + 1) * 24, (te[-1] + 1) * 24
        fin_tr = int(fin_app * 0.85) // 24 * 24
        j = jeu_pour_pli(jeu, fin_tr, fin_app, fin_te)
        mod, hist, duree = m.entrainer(cfg, j, graine)
        o = m.origines(fin_app, fin_te, cfg.L, cfg.H, 1)
        yp, yv = m.predire(mod, j, o)
        mt = m.metriques(yv, yp)
        rp = m.metriques(yv, m.reference_persistance(j, o, cfg.H))["RMSE"]
        lignes.append(dict(pli=k + 1, train=f"{j.index[0]:%Y-%m} → {j.index[fin_tr-1]:%Y-%m}",
                           test=f"{j.index[fin_app]:%Y-%m-%d} → {j.index[fin_te-1]:%Y-%m-%d}",
                           RMSE=mt["RMSE"], nRMSE=mt["nRMSE"], R2=mt["R2"], RMSE_persistance=rp,
                           gain_vs_persistance=1 - mt["RMSE"] / rp, epoques=len(hist)))
    return pd.DataFrame(lignes)


# =============================================================================
# F. Robustesse : courbe de dégradation en fonction du niveau de bruit
# =============================================================================

def courbe_bruit(cfg, jeu, nom, niveaux=(0, 0.01, 0.02, 0.05, 0.10, 0.20)):
    """RMSE (moyenne des 5 graines) lorsque les exogènes sont bruités à σ = niveau."""
    import torch
    o = m.origines(jeu.i_test, len(jeu.valeurs), cfg.L, cfg.H, 1)
    lignes = []
    for g in m.GRAINES:
        mod = m.PatchTST(cfg)
        mod.load_state_dict(torch.load(dossier(nom) / f"poids_graine{g}.pt"))
        mod.eval()
        for n in niveaux:
            v = m.bruiter_exogenes(jeu, n) if n > 0 else jeu.valeurs
            yp, yv = m.predire(mod, jeu, o, v)
            lignes.append(dict(graine=g, niveau=n, RMSE=m.metriques(yv, yp)["RMSE"]))
    df = pd.DataFrame(lignes)
    base = df[df.niveau == 0].set_index("graine").RMSE
    df["degradation"] = df.apply(lambda r: r.RMSE / base[r.graine] - 1, axis=1)
    return df


# =============================================================================
# G. Explicabilité (XAI)
# =============================================================================

GROUPES = {"Cible (historique PV)": [m.CIBLE], "Réseau": m.CANAUX_RESEAU,
           "Cycliques heure": ["h_sin", "h_cos"], "Cycliques saison": ["doy_sin", "doy_cos"],
           "Cycliques semaine": ["dow_sin", "dow_cos"], "Géométrie solaire": m.CANAUX_SOLAIRES}


def importance_occultation(mod, jeu, n_ech: int = 1500, graine: int = 0):
    """Importance par occultation de groupes de canaux : le groupe est remplacé
    par sa moyenne d'entraînement (0 en unités normalisées) dans le passé ET,
    pour les déterministes, dans le futur. Hausse du RMSE = importance."""
    cfg = mod.cfg
    rng = np.random.default_rng(graine)
    o = np.sort(rng.choice(m.origines(jeu.i_test, len(jeu.valeurs), cfg.L, cfg.H, 1), n_ech, replace=False))
    yp, yv = m.predire(mod, jeu, o)
    base = m.metriques(yv, yp)["RMSE"]
    lignes = []
    for nom_g, cols in GROUPES.items():
        v = jeu.valeurs.copy()
        v[:, [m.CANAUX.index(c) for c in cols]] = 0.0
        if nom_g.startswith("Cible"):
            # la cible doit rester observable pour le calcul de l'erreur : on
            # n'occulte que la partie « passé » via un jeu dédié
            yp2 = _predire_cible_occultee(mod, jeu, o)
        else:
            yp2, _ = m.predire(mod, jeu, o, v)
        lignes.append(dict(groupe=nom_g, RMSE=m.metriques(yv, yp2)["RMSE"],
                           hausse_RMSE=m.metriques(yv, yp2)["RMSE"] / base - 1))
    return pd.DataFrame(lignes), base


def _predire_cible_occultee(mod, jeu, o):
    """Prévision lorsque l'historique de la cible est remplacé par son profil
    moyen journalier d'entraînement (on garde la forme diurne, on retire
    l'information nuageuse récente)."""
    v = jeu.valeurs.copy()
    y = pd.Series(v[: jeu.i_val, 0], index=jeu.index[: jeu.i_val])
    profil = y.groupby(y.index.hour).mean()
    v[:, 0] = profil.reindex(jeu.index.hour).values
    yp, _ = m.predire(mod, jeu, o, v)
    return yp


def gradients_integres(mod, jeu, n_ech: int = 300, pas: int = 32, graine: int = 0):
    """Gradients intégrés (Sundararajan et al., 2017) sur l'entrée passée [L, C]
    et, pour la variante X, sur les covariables futures [H, C_det]. Référence :
    entrée nulle (moyenne Train). Cible expliquée : énergie prévue sur l'horizon.
    Retourne l'attribution absolue moyenne par canal (passé) et par canal futur."""
    import torch
    cfg = mod.cfg
    rng = np.random.default_rng(graine)
    o = np.sort(rng.choice(m.origines(jeu.i_test, len(jeu.valeurs), cfg.L, cfg.H, 1), n_ech, replace=False))
    ds = m.FenetresDataset(jeu.valeurs, o, cfg.L, cfg.H)
    passe = torch.stack([ds[i][0] for i in range(len(o))])
    futur = torch.stack([ds[i][1] for i in range(len(o))])
    cz = m._cz_brut(jeu, futur)
    att_p = torch.zeros_like(passe); att_f = torch.zeros_like(futur)
    mod.eval()
    for a in torch.linspace(1 / pas, 1, pas):
        p = (a * passe).requires_grad_(True); f = (a * futur).requires_grad_(True)
        sortie = mod(p, f, cz).sum()
        gp, gf = torch.autograd.grad(sortie, [p, f], allow_unused=True)
        if gp is not None: att_p += gp / pas
        if gf is not None: att_f += gf / pas
    att_p = (att_p * passe).abs().mean(dim=(0, 1)).numpy()
    att_f = (att_f * futur).abs().mean(dim=(0, 1)).numpy()
    ser_p = pd.Series(att_p, index=[f"{c} (passé)" for c in m.CANAUX])
    ser_f = pd.Series(att_f, index=[f"{c} (futur)" for c in m.CANAUX_DETERMINISTES])
    tout = pd.concat([ser_p, ser_f])
    return tout / tout.sum()


def poids_attention_canaux(mod, jeu, n_ech: int = 1000, graine: int = 0):
    """Poids moyens de l'attention inter-canaux (variantes CA) : part de
    l'attention que le canal cible accorde à chaque canal."""
    import torch
    cfg = mod.cfg
    if cfg.mode_canaux != "CA":
        return None
    rng = np.random.default_rng(graine)
    o = np.sort(rng.choice(m.origines(jeu.i_test, len(jeu.valeurs), cfg.L, cfg.H, 1), n_ech, replace=False))
    ds = m.FenetresDataset(jeu.valeurs, o, cfg.L, cfg.H)
    passe = torch.stack([ds[i][0] for i in range(len(o))]); futur = torch.stack([ds[i][1] for i in range(len(o))])
    with torch.no_grad():
        mod(passe, futur, m._cz_brut(jeu, futur))
    w = mod.derniers_poids_canaux.mean(dim=(0, 1)).numpy()
    return pd.Series(w, index=m.CANAUX)


# =============================================================================
# H. Variante « indice de capacité » (normalisation causale par le parc installé)
# =============================================================================

def capacite_causale(jeu, fenetre_j: int = 30) -> pd.Series:
    """Capacité apparente causale : quantile 99 % des maxima journaliers des
    jours J-30…J-1 (n'utilise que le passé), ramenée au pas horaire."""
    jmax = jeu.tableau[m.CIBLE].resample("D").max()
    cap = jmax.shift(1).rolling(fenetre_j, min_periods=7).quantile(0.99).bfill()
    return cap.reindex(jeu.index, method="ffill")


def jeu_indice_capacite(jeu):
    """Jeu de données dont la cible est k = PV_total / capacité causale.
    La normalisation est ré-ajustée sur Train (R2)."""
    import copy as _copy
    cap = capacite_causale(jeu).values
    j = _copy.copy(jeu)
    tab = jeu.tableau.copy()
    tab[m.CIBLE] = tab[m.CIBLE].values / cap
    brut = tab.values.astype(np.float64)
    mu, sd = brut[: jeu.i_val].mean(0), brut[: jeu.i_val].std(0)
    sd[sd < 1e-8] = 1.0
    j.tableau, j.valeurs, j.moyenne, j.ecart_type = tab, ((brut - mu) / sd).astype(np.float32), mu, sd
    return j, cap


# =============================================================================
# I. Mise en cache des analyses lourdes du modèle final
# =============================================================================

def en_cache(nom_fichier: str, calcul):
    """Exécute `calcul()` une seule fois et met le DataFrame résultat en cache (CSV)."""
    f = m.DOSSIER_CACHE / "final" / nom_fichier
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        return pd.read_csv(f)
    df = calcul()
    df.to_csv(f, index=False)
    return df


def evaluer_indice_capacite(cfg, jeu):
    """Ablation : la même configuration entraînée sur k = PV / capacité causale
    (5 graines), prévision remise en MW en multipliant par la capacité de
    l'instant prévu (connue causalement : calculée sur les jours précédents)."""
    jk, cap = jeu_indice_capacite(jeu)
    o = m.origines(jeu.i_test, len(jeu.valeurs), cfg.L, cfg.H, 1)
    cap_fut = np.stack([cap[t:t + cfg.H] for t in o])
    lignes = []
    for g in m.GRAINES:
        mod, hist, duree = m.entrainer(cfg, jk, g)
        k_pred, _ = m.predire(mod, jk, o)
        yp = k_pred * cap_fut
        yv = np.stack([jeu.tableau[m.CIBLE].values[t:t + cfg.H] for t in o])
        mt = m.metriques(yv, yp)
        rj = m.rmse_journaliers(yv[o < jeu.i_test + 24 * 30], yp[o < jeu.i_test + 24 * 30],
                                o[o < jeu.i_test + 24 * 30], jeu)
        lignes.append(dict(graine=g, RMSE=mt["RMSE"], nRMSE=mt["nRMSE"], R2=mt["R2"], MBE=mt["MBE"],
                           drift_7j=float(rj.iloc[-7:].mean() / rj.iloc[:7].mean()), epoques=len(hist)))
    return pd.DataFrame(lignes)


def rmse_mensuel(yv, yp, orig, jeu):
    """RMSE par mois de l'origine (stabilité sur les 7 mois de Test)."""
    mois = jeu.index[orig].to_period("M").astype(str)
    return pd.Series(((yp - yv) ** 2).mean(axis=1), index=mois).groupby(level=0).mean().pipe(np.sqrt)
