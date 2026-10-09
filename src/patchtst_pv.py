# -*- coding: utf-8 -*-
"""
===============================================================================
 patchtst_pv.py — Module commun de l'étude
 « Prévision multivariée de la production photovoltaïque totale du réseau
   SENELEC par PatchTST, variables cycliques et géométrie solaire »
===============================================================================

Auteur      : Abdoulaye FAYE — UADB / EDTSS, Équipe Efficacité et Systèmes
              Énergétiques
Données     : energie.xlsx (journal horaire SENELEC, 01/01/2019 → 09/05/2023)

Organisation du module (chaque bloc est indépendant et documenté) :

    0. CONFIGURATION GLOBALE ........ chemins, dates de découpage, listes de colonnes
    1. CHARGEMENT & NETTOYAGE ....... journal horaire, écrêtage, imputation, cible
    2. COVARIABLES .................. cycliques, géométrie solaire, contexte réseau
    3. JEU DE DONNÉES ............... assemblage, découpage temporel, normalisation
    4. FENÊTRAGE .................... échantillons (passé L, futur H) sans fuite
    5. MODÈLE PatchTST .............. RevIN, patchs, encodeur, CI / CA / X
    6. ENTRAÎNEMENT & PRÉDICTION .... boucle, arrêt précoce, inférence
    7. MÉTRIQUES & RÉFÉRENCES ....... RMSE/MAE/nRMSE, persistance, moyenne, DM
    8. PROTOCOLE D'ÉVALUATION ....... 5 graines, bruit 5 %, rolling 30 j, efficience
    9. SCORE DE DÉCISION ............ score composite de projet.md

Règles d'or (projet.md), vérifiées par des assertions dans le code :
    R1. Jamais de mélange aléatoire : Train < Val < Test.
    R2. Aucune fuite : écrêtage, imputation et normalisation ajustés sur Train.
    R3. Comparaison systématique à la persistance et à la moyenne.
    R4. Les productions PV individuelles ne sont JAMAIS des variables d'entrée :
        elles servent uniquement à construire la cible PV_total.
===============================================================================
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# 0. CONFIGURATION GLOBALE
# =============================================================================

#: Racine du projet (dossier « outputs »), déduite de l'emplacement de ce fichier.
RACINE = Path(__file__).resolve().parents[1]
CHEMIN_DONNEES = RACINE / "data" / "energie.xlsx"
DOSSIER_CACHE = RACINE / "cache"
DOSSIER_FIGURES = RACINE / "figures"
DOSSIER_TABLEAUX = RACINE / "tableaux"
DOSSIER_MODELES = RACINE / "models"

#: Découpage temporel strict (R1). Les bornes sont inclusives à gauche.
DEBUT_VAL = pd.Timestamp("2022-01-01 00:00")
DEBUT_TEST = pd.Timestamp("2022-10-01 00:00")

#: Graines utilisées pour mesurer la dispersion (critère de robustesse).
GRAINES = [0, 1, 2, 3, 4]

#: Colonnes de production PV individuelle : UNIQUEMENT pour construire la cible.
COLONNES_PV = [
    "PV-Mbour", "PV-CICAD", "PV Kahone", "PV ERS Kahone", "PV Scaling Kahone",
    "PV Kael Touba", "PV-Bokhol", "PV Mékhé Mérina", "PV S Mekhe Santhiou",
    "PV DIASS", "PV Sakal",
]

#: Unités thermiques (et assimilées) agrégées en un seul canal « Thermique ».
COLONNES_THERMIQUES = [
    "106", "301", "303", "TAG2", "TAG4", "401", "402", "403", "404", "405",
    "Agg", "Agg1", "Agg2", "Sendou", "Koun", "M Power", "TP", "PSHIP", "APR",
    "CG", "Boutoute", "Kah1", "C7", "C6", "ICS", "Dang",
]

#: Colonnes exclues car elles CONTIENNENT la production PV (fuite indirecte).
COLONNES_EXCLUES = ["Total_unites", "RI", "RGI_1", "RGI_2", "RGI_3", "RGI_4", "RGI_5"]

#: Site de référence : barycentre approximatif des centrales PV du réseau.
LATITUDE, LONGITUDE, ALTITUDE = 14.9, -16.4, 30.0

#: Nom des canaux (ordre fixe ; le canal 0 est TOUJOURS la cible).
CIBLE = "PV_total"
CANAUX_RESEAU = ["Thermique", "Eolien_Taiba", "Import_Manantali"]
CANAUX_CYCLIQUES = ["h_sin", "h_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
CANAUX_SOLAIRES = ["cos_zenith", "ghi_ciel_clair"]
#: Covariables déterministes : connues aussi dans le futur (conditionnement « X »).
CANAUX_DETERMINISTES = CANAUX_CYCLIQUES + CANAUX_SOLAIRES
CANAUX = [CIBLE] + CANAUX_RESEAU + CANAUX_DETERMINISTES


# =============================================================================
# 1. CHARGEMENT & NETTOYAGE
# =============================================================================

def charger_journal(chemin: Path | str = CHEMIN_DONNEES) -> pd.DataFrame:
    """Lit le journal horaire et le ré-indexe sur une grille horaire complète.

    Les 288 heures absentes du fichier apparaissent en NaN : elles seront
    imputées plus loin, jamais supprimées (la grille doit rester régulière
    pour le fenêtrage).
    """
    df = pd.read_excel(chemin)
    df = df.set_index("Horodatage_debut").sort_index()
    grille = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(grille)
    df.index.name = "horodatage"
    return df


def periode_existence(s: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Première et dernière heure de production > 0 d'une centrale.

    Avant la mise en service (ou après le retrait), la production vaut 0 par
    définition ; un NaN pendant la période d'existence est une vraie lacune.
    """
    actif = s[s > 0]
    return actif.index.min(), actif.index.max()


def imputer_serie(s: pd.Series, lacune_max: int = 3) -> pd.Series:
    """Imputation causale en deux temps.

    1) Interpolation linéaire des lacunes courtes (<= `lacune_max` heures).
    2) Lacunes longues : moyenne de la même heure sur les 7 jours PRÉCÉDENTS
       (profil horaire glissant — n'utilise que le passé).
    3) Reliquat (début de série) : 0.
    """
    s = s.copy()
    # 1) lacunes courtes : on n'interpole que les « trous » encadrés et courts
    masque_nan = s.isna()
    id_bloc = (masque_nan != masque_nan.shift()).cumsum()
    taille_bloc = masque_nan.groupby(id_bloc).transform("sum")
    courtes = masque_nan & (taille_bloc <= lacune_max)
    interp = s.interpolate(method="linear", limit_area="inside")
    s[courtes] = interp[courtes]
    # 2) lacunes longues : profil horaire des 7 jours précédents (causal)
    if s.isna().any():
        tableau = s.to_frame("v")
        tableau["heure"] = tableau.index.hour
        profil = (tableau.groupby("heure")["v"]
                  .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()))
        s = s.fillna(profil)
    # 3) reliquat
    return s.fillna(0.0)


def construire_cible(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Construit PV_total (MW) à partir des 11 centrales, avec traçabilité.

    Pour chaque centrale :
      a) seuil d'écrêtage = 1,2 × quantile 99,9 % calculé sur TRAIN seulement (R2)
         → une valeur au-dessus (ex. 958 MW à Mbour) est une erreur de saisie : NaN ;
      b) hors période d'existence : 0 ;
      c) lacunes pendant l'existence : imputation causale.
    Retourne la cible et un tableau de diagnostic par centrale.
    """
    train = df.index < DEBUT_VAL
    diagnostic, series = [], {}
    for col in COLONNES_PV:
        s = df[col].astype(float).clip(lower=0)
        debut, fin = periode_existence(s)
        seuil = 1.2 * s[train].quantile(0.999) if s[train].notna().sum() > 100 \
            else 1.2 * s.quantile(0.999)
        n_aberrants = int((s > seuil).sum())
        s[s > seuil] = np.nan
        existe = (s.index >= debut) & (s.index <= fin)
        n_lacunes = int(s[existe].isna().sum())
        s[~existe] = 0.0
        s[existe] = imputer_serie(s[existe])
        series[col] = s
        diagnostic.append(dict(centrale=col, mise_en_service=debut, dernier_releve=fin,
                               seuil_MW=round(seuil, 2), valeurs_ecretees=n_aberrants,
                               lacunes_imputees=n_lacunes,
                               moyenne_MW=round(s.mean(), 2), max_MW=round(s.max(), 2)))
    cible = pd.DataFrame(series).sum(axis=1).rename(CIBLE)
    return cible, pd.DataFrame(diagnostic)


# =============================================================================
# 2. COVARIABLES
# =============================================================================

def covariables_cycliques(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encodages sinus/cosinus : continuité 23 h → 0 h, 31/12 → 01/01, dim. → lun.

    Heure (période 24), jour de l'année (période 365,25) et jour de la semaine
    (période 7). Le jour de semaine capte un éventuel effet d'exploitation
    (délestages, maintenance) — il n'a pas d'effet physique sur l'irradiance,
    son importance mesurée servira de témoin négatif dans l'analyse XAI.
    """
    h = index.hour.values + 0.5            # milieu de l'heure [t, t+1)
    doy = index.dayofyear.values - 1 + h / 24
    dow = index.dayofweek.values
    return pd.DataFrame({
        "h_sin": np.sin(2 * np.pi * h / 24), "h_cos": np.cos(2 * np.pi * h / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=index)


def covariables_solaires(index: pd.DatetimeIndex, decalage_h: float = 0.0) -> pd.DataFrame:
    """Géométrie solaire et rayonnement de ciel clair (pvlib), calculés au
    milieu de chaque heure, décalés de `decalage_h` heures pour aligner
    l'horodatage du journal sur le temps solaire (décalage estimé sur Train).

    - cos_zenith     : cosinus de l'angle zénithal apparent, borné à 0 la nuit ;
    - ghi_ciel_clair : rayonnement global horizontal de ciel clair (Haurwitz, W/m²).
    Ces variables sont DÉTERMINISTES : disponibles sans erreur pour tout horizon.
    """
    import pvlib
    t = (index + pd.Timedelta(minutes=30) + pd.Timedelta(hours=decalage_h)).tz_localize("UTC")
    pos = pvlib.solarposition.get_solarposition(t, LATITUDE, LONGITUDE, ALTITUDE)
    ghi = pvlib.clearsky.haurwitz(pos["apparent_zenith"])["ghi"].values
    cz = np.clip(np.cos(np.deg2rad(pos["apparent_zenith"].values)), 0, None)
    return pd.DataFrame({"cos_zenith": cz, "ghi_ciel_clair": ghi}, index=index)


def estimer_decalage(cible: pd.Series, candidats=np.arange(-2, 2.01, 0.25)) -> float:
    """Décalage horaire (h) maximisant la corrélation entre le profil moyen
    journalier de PV_total et celui du rayonnement de ciel clair — estimé sur
    TRAIN uniquement. Il absorbe la convention d'horodatage du journal
    (début/fin d'heure) et l'écart heure légale / temps solaire."""
    train = cible[cible.index < DEBUT_VAL]
    profil_pv = train.groupby(train.index.hour).mean().values
    meilleur, corr_max = 0.0, -1
    jours = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    for d in candidats:
        cs = covariables_solaires(jours, d)["ghi_ciel_clair"]
        profil_cs = cs.groupby(cs.index.hour).mean().values
        c = np.corrcoef(profil_pv, profil_cs)[0, 1]
        if c > corr_max:
            meilleur, corr_max = float(d), c
    return meilleur


def covariables_reseau(df: pd.DataFrame) -> pd.DataFrame:
    """Contexte d'exploitation du réseau, utilisé comme « proxy météo ».

    - Thermique        : somme des unités thermiques (répond à la demande nette,
                         donc indirectement à la nébulosité passée) ;
    - Eolien_Taiba     : production éolienne (régime de vent, advection) ;
    - Import_Manantali : import hydraulique OMVS.
    Ces canaux ne sont utilisés que sur la fenêtre PASSÉE (inconnus au futur).
    """
    thermique = df[COLONNES_THERMIQUES].sum(axis=1, min_count=1)
    eolien = df["Eolien Taiba"].copy()
    debut_eolien, _ = periode_existence(eolien)
    eolien[eolien.index < debut_eolien] = 0.0
    sortie = pd.DataFrame({
        "Thermique": thermique,
        "Eolien_Taiba": eolien.clip(lower=0),
        "Import_Manantali": df["Import_Manantali_Felou_Somelec"].clip(lower=0),
    }, index=df.index)
    # écrêtage des erreurs de saisie (ex. Thermique = 17 316 MW) : seuil
    # 1,2 × q99,9 calculé sur Train (R2), puis imputation causale
    train = sortie.index < DEBUT_VAL
    for c in sortie:
        seuil = 1.2 * sortie.loc[train, c].quantile(0.999)
        sortie.loc[sortie[c] > seuil, c] = np.nan
        sortie[c] = imputer_serie(sortie[c])
    return sortie


# =============================================================================
# 3. JEU DE DONNÉES : assemblage, découpage, normalisation
# =============================================================================

@dataclass
class JeuDonnees:
    """Conteneur du jeu de données prêt à fenêtrer."""
    tableau: pd.DataFrame          # valeurs physiques (MW, sans unité)
    valeurs: np.ndarray            # valeurs normalisées float32 [T, C]
    moyenne: np.ndarray            # paramètres de normalisation (Train)
    ecart_type: np.ndarray
    i_val: int                     # premier indice de Val
    i_test: int                    # premier indice de Test
    decalage_solaire_h: float
    diagnostic_cible: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.tableau.index

    def vers_mw(self, y_norm: np.ndarray) -> np.ndarray:
        """Dé-normalise des valeurs du canal cible vers des MW."""
        return y_norm * self.ecart_type[0] + self.moyenne[0]


def construire_jeu(chemin: Path | str = CHEMIN_DONNEES, cache: bool = True) -> JeuDonnees:
    """Pipeline complet de préparation (mis en cache sur disque)."""
    fichier_cache = DOSSIER_CACHE / "jeu_donnees.pkl"
    if cache and fichier_cache.exists():
        return pd.read_pickle(fichier_cache)

    df = charger_journal(chemin)
    cible, diagnostic = construire_cible(df)
    decalage = estimer_decalage(cible)
    tableau = pd.concat([cible, covariables_reseau(df),
                         covariables_cycliques(df.index),
                         covariables_solaires(df.index, decalage)], axis=1)[CANAUX]

    # --- R4 : aucune colonne PV individuelle ni colonne « fuyante » en entrée
    assert not set(COLONNES_PV + COLONNES_EXCLUES) & set(tableau.columns)
    assert tableau.notna().all().all(), "des NaN subsistent après imputation"

    i_val = int(np.searchsorted(tableau.index, DEBUT_VAL))
    i_test = int(np.searchsorted(tableau.index, DEBUT_TEST))
    # --- R1 : ordre strict des segments
    assert 0 < i_val < i_test < len(tableau)
    assert tableau.index[i_val - 1] < DEBUT_VAL <= tableau.index[i_val]

    # --- R2 : normalisation ajustée sur Train seulement
    brut = tableau.values.astype(np.float64)
    moyenne = brut[:i_val].mean(axis=0)
    ecart = brut[:i_val].std(axis=0)
    ecart[ecart < 1e-8] = 1.0
    valeurs = ((brut - moyenne) / ecart).astype(np.float32)

    jeu = JeuDonnees(tableau, valeurs, moyenne, ecart, i_val, i_test, decalage, diagnostic)
    if cache:
        DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(jeu, fichier_cache)
    return jeu


# =============================================================================
# 4. FENÊTRAGE
# =============================================================================
#
# Convention : une « origine » t signifie que la dernière heure observée est
# t-1. Le modèle reçoit X[t-L : t] (tous les canaux, passé) et, pour la
# variante X, les covariables déterministes F[t : t+H] (futur connu). Il prédit
# la cible Y[t : t+H]. Aucune valeur de la cible ou du réseau postérieure à t-1
# n'entre dans le modèle.

def origines(debut: int, fin: int, L: int, H: int, pas: int = 1) -> np.ndarray:
    """Origines admissibles dont passé et futur tiennent dans [debut-L, fin)."""
    premier = max(debut, L)
    dernier = fin - H          # inclus
    return np.arange(premier, dernier + 1, pas, dtype=np.int64)


class FenetresDataset(torch.utils.data.Dataset):
    """Échantillons (passé, futur déterministe, cible) pour un ensemble d'origines."""

    def __init__(self, valeurs: np.ndarray, orig: np.ndarray, L: int, H: int):
        self.v = torch.from_numpy(valeurs)
        self.orig, self.L, self.H = orig, L, H
        self.i_det = [CANAUX.index(c) for c in CANAUX_DETERMINISTES]

    def __len__(self):
        return len(self.orig)

    def __getitem__(self, k):
        t = int(self.orig[k])
        passe = self.v[t - self.L:t]                       # [L, C]
        futur = self.v[t:t + self.H][:, self.i_det]        # [H, C_det]
        cible = self.v[t:t + self.H, 0]                    # [H]
        return passe, futur, cible


# =============================================================================
# 5. MODÈLE PatchTST
# =============================================================================

class RevIN(nn.Module):
    """Normalisation réversible par instance (Kim et al., ICLR 2022).

    Chaque fenêtre est centrée-réduite canal par canal ; la prévision est
    ramenée à l'échelle de la fenêtre d'entrée du canal cible. Atténue la
    non-stationnarité (croissance de la capacité installée).

    RevIN n'est appliqué qu'aux canaux STOCHASTIQUES (cible et réseau). Les
    covariables déterministes gardent leur normalisation globale : sur une
    fenêtre de quelques jours, le jour de l'année varie à peine, et le
    centrer-réduire effacerait l'information utile (la position dans l'année)
    en la remplaçant par une rampe artificielle.
    """

    def __init__(self, n_canaux: int, n_stochastiques: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.k = n_stochastiques            # canaux 0..k-1 normalisés par fenêtre
        self.gamma = nn.Parameter(torch.ones(n_stochastiques))
        self.beta = nn.Parameter(torch.zeros(n_stochastiques))

    def normaliser(self, x):                # x : [B, L, C]
        xs = x[:, :, :self.k]
        self.mu = xs.mean(dim=1, keepdim=True).detach()
        self.sigma = torch.sqrt(xs.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        xs = (xs - self.mu) / self.sigma * self.gamma + self.beta
        return torch.cat([xs, x[:, :, self.k:]], dim=-1)

    def denormaliser_cible(self, y):        # y : [B, H] (canal 0)
        y = (y - self.beta[0]) / (self.gamma[0] + self.eps)
        return y * self.sigma[:, 0, 0:1] + self.mu[:, 0, 0:1]


@dataclass
class ConfigPatchTST:
    """Hyper-paramètres d'une configuration (les 4 facteurs de projet.md + options)."""
    P: int = 24                  # longueur de patch (granularité temporelle)
    L: int = 168                 # historique (lookback)
    H: int = 24                  # horizon de prévision
    mode_canaux: str = "CA"      # "CI" (indépendance) ou "CA" (attention inter-canaux)
    futur_connu: bool = False    # conditionnement par covariables déterministes futures
    porte_physique: bool = False # sortie mise à zéro quand le soleil est couché
    d_model: int = 64
    n_tetes: int = 4
    n_couches: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    # entraînement
    lr: float = 1e-3
    taille_lot: int = 128
    epoques_max: int = 25
    patience: int = 4
    pas_train: int = 1           # toutes les origines horaires sont candidates…
    fraction_epoque: float = 0.2 # …mais 20 % tirées au hasard à chaque époque (coût CPU)
    pas_val: int = 6

    @property
    def stride(self) -> int:
        """Pas entre patchs : moitié du patch (recouvrement 50 %, Nie et al. 2023)."""
        return max(1, self.P // 2)

    @property
    def n_patchs(self) -> int:
        return (self.L - self.P) // self.stride + 1

    def nom(self) -> str:
        suffixe = ("X" if self.futur_connu else "") + ("g" if self.porte_physique else "")
        return f"PatchTST-{self.mode_canaux}{suffixe}_P{self.P}_L{self.L}_H{self.H}"


class PatchTST(nn.Module):
    """PatchTST multivarié, trois variantes de traitement des canaux.

    Chaîne de calcul (B = lot, C = canaux, N = patchs, D = d_model) :

      x [B, L, C] ──RevIN──► découpage en patchs [B, C, N, P]
         ──projection linéaire P→D + position──► [B·C, N, D]
         ──encodeur Transformer PARTAGÉ entre canaux (attention temporelle)──►
      ┌─ CI : on ne garde que le canal cible → [B, N·D] → tête linéaire → H
      │       (les exogènes n'influencent donc PAS la prévision : c'est
      │        l'équivalent d'un PatchTST univarié à poids partagés)
      └─ CA : attention INTER-CANAUX à chaque position de patch [B·N, C, D]
              (le canal cible « interroge » les exogènes), puis canal cible
              → [B, N·D] → tête linéaire → H
      Option X : correction pas à pas à partir des covariables déterministes
              FUTURES (heure, saison, géométrie solaire de l'instant prévu).
      Option g : porte physique — prévision forcée à 0 quand cos_zenith futur = 0.
    """

    def __init__(self, cfg: ConfigPatchTST, n_canaux: int = len(CANAUX),
                 n_det: int = len(CANAUX_DETERMINISTES)):
        super().__init__()
        self.cfg = cfg
        self.C = n_canaux if cfg.mode_canaux == "CA" else 1
        D, N = cfg.d_model, cfg.n_patchs
        self.revin = RevIN(self.C, 1 + len(CANAUX_RESEAU) if cfg.mode_canaux == "CA" else 1)
        # --- plongement des patchs (partagé entre canaux) + position apprise
        self.projection = nn.Linear(cfg.P, D)
        self.position = nn.Parameter(torch.randn(1, N, D) * 0.02)
        # --- plongement d'identité de canal (utile uniquement en CA)
        self.id_canal = nn.Parameter(torch.randn(1, self.C, 1, D) * 0.02)
        self.dropout = nn.Dropout(cfg.dropout)
        couche = nn.TransformerEncoderLayer(D, cfg.n_tetes, cfg.d_ff, cfg.dropout,
                                            batch_first=True, norm_first=True,
                                            activation="gelu")
        self.encodeur_temporel = nn.TransformerEncoder(couche, cfg.n_couches,
                                                       enable_nested_tensor=False)
        if cfg.mode_canaux == "CA":
            self.attention_canaux = nn.MultiheadAttention(D, cfg.n_tetes, dropout=cfg.dropout,
                                                          batch_first=True)
            self.norme_canaux = nn.LayerNorm(D)
        # --- tête de prévision directe multi-horizon
        self.tete = nn.Sequential(nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(N * D, cfg.H))
        # --- option X : correction conditionnée par le futur déterministe
        if cfg.futur_connu:
            self.contexte = nn.Linear(N * D, 32)
            self.correction = nn.Sequential(nn.Linear(32 + n_det + 1, 64), nn.GELU(),
                                            nn.Linear(64, 1))
        self.i_cz = CANAUX_DETERMINISTES.index("cos_zenith")
        self.derniers_poids_canaux = None   # pour l'analyse XAI

    def forward(self, passe, futur=None, cz_futur_brut=None):
        cfg = self.cfg
        x = passe if cfg.mode_canaux == "CA" else passe[:, :, :1]      # [B, L, C]
        x = self.revin.normaliser(x)
        B = x.shape[0]
        # découpage en patchs : [B, C, N, P]
        patchs = x.permute(0, 2, 1).unfold(-1, cfg.P, cfg.stride)
        z = self.projection(patchs) + self.position.unsqueeze(1)        # [B, C, N, D]
        z = z + self.id_canal
        z = self.dropout(z)
        N, D = z.shape[2], z.shape[3]
        z = self.encodeur_temporel(z.reshape(B * self.C, N, D)).reshape(B, self.C, N, D)
        if cfg.mode_canaux == "CA":
            # attention inter-canaux à chaque position de patch
            zc = z.permute(0, 2, 1, 3).reshape(B * N, self.C, D)         # [B·N, C, D]
            requete = zc[:, :1]                                          # canal cible
            sortie, poids = self.attention_canaux(requete, zc, zc, need_weights=True)
            self.derniers_poids_canaux = poids.detach().reshape(B, N, self.C)
            zt = self.norme_canaux(requete + sortie).reshape(B, N, D)
        else:
            zt = z[:, 0]                                                 # [B, N, D]
        y = self.tete(zt)                                                # [B, H]
        if cfg.futur_connu:
            ctx = self.contexte(zt.flatten(1))                           # [B, 32]
            ctx = ctx.unsqueeze(1).expand(-1, cfg.H, -1)
            base = y.unsqueeze(-1)                                       # prévision de base
            y = y + self.correction(torch.cat([ctx, futur, base], dim=-1)).squeeze(-1)
        y = self.revin.denormaliser_cible(y)
        if cfg.porte_physique and cz_futur_brut is not None:
            y = y * (cz_futur_brut > 0).float()
        return y


# =============================================================================
# 6. ENTRAÎNEMENT & PRÉDICTION
# =============================================================================

def fixer_graine(graine: int):
    """Reproductibilité (numpy, torch, algorithmes déterministes)."""
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _cz_brut(jeu: JeuDonnees, futur_norm: torch.Tensor) -> torch.Tensor:
    """cos_zenith futur remis à l'échelle physique (pour la porte physique)."""
    i = CANAUX.index("cos_zenith")
    j = CANAUX_DETERMINISTES.index("cos_zenith")
    return futur_norm[..., j] * jeu.ecart_type[i] + jeu.moyenne[i]


def perte_mse(jeu, y_pred, y_vrai):
    """Erreur quadratique moyenne en unités normalisées."""
    return torch.mean((y_pred - y_vrai) ** 2)


def entrainer(cfg: ConfigPatchTST, jeu: JeuDonnees, graine: int, verbeux: bool = False):
    """Entraîne un modèle sur Train, arrêt précoce sur Val (Test jamais vu).

    Optimiseur Adam, réduction du taux d'apprentissage sur plateau,
    écrêtage du gradient, restauration des meilleurs poids.
    """
    fixer_graine(graine)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    o_train = origines(0, jeu.i_val, cfg.L, cfg.H, cfg.pas_train)
    o_val = origines(jeu.i_val, jeu.i_test, cfg.L, cfg.H, cfg.pas_val)
    # R1 : aucune cible d'entraînement ne déborde sur Val, ni de Val sur Test
    assert o_train.max() + cfg.H <= jeu.i_val and o_val.max() + cfg.H <= jeu.i_test
    g = torch.Generator().manual_seed(graine)
    # Sous-échantillonnage aléatoire des origines à chaque époque : toutes les
    # phases horaires (0 h … 23 h) restent représentées, contrairement à un pas
    # fixe (un pas de 3 h ne montrerait au modèle que 8 phases sur 24).
    echantillonneur = torch.utils.data.RandomSampler(
        range(len(o_train)), replacement=False,
        num_samples=int(cfg.fraction_epoque * len(o_train)), generator=g)
    dl_train = torch.utils.data.DataLoader(FenetresDataset(jeu.valeurs, o_train, cfg.L, cfg.H),
                                           batch_size=cfg.taille_lot, sampler=echantillonneur)
    # NB : le tirage aléatoire ne porte que sur les fenêtres d'ENTRAÎNEMENT au
    # sein d'une époque (descente de gradient stochastique) ; le découpage
    # temporel Train < Val < Test reste strict (R1).
    dl_val = torch.utils.data.DataLoader(FenetresDataset(jeu.valeurs, o_val, cfg.L, cfg.H),
                                         batch_size=512, shuffle=False)
    modele = PatchTST(cfg)
    opt = torch.optim.Adam(modele.parameters(), lr=cfg.lr)
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    meilleur, meilleurs_poids, attente = math.inf, None, 0
    historique = []
    t0 = time.time()
    for epoque in range(cfg.epoques_max):
        modele.train()
        pertes = []
        for passe, futur, cible in dl_train:
            opt.zero_grad()
            pred = modele(passe, futur, _cz_brut(jeu, futur))
            perte = perte_mse(jeu, pred, cible)
            perte.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            opt.step()
            pertes.append(perte.item())
        modele.eval()
        with torch.no_grad():
            pv = [perte_mse(jeu, modele(p, f, _cz_brut(jeu, f)), c).item() * len(c)
                  for p, f, c in dl_val]
        perte_val = sum(pv) / len(o_val)
        plan.step(perte_val)
        historique.append(dict(epoque=epoque + 1, perte_train=float(np.mean(pertes)),
                               perte_val=perte_val))
        if verbeux:
            print(f"  époque {epoque+1:2d}  train={np.mean(pertes):.4f}  val={perte_val:.4f}")
        if perte_val < meilleur - 1e-5:
            meilleur, meilleurs_poids, attente = perte_val, copy.deepcopy(modele.state_dict()), 0
        else:
            attente += 1
            if attente >= cfg.patience:
                break
    modele.load_state_dict(meilleurs_poids)
    modele.eval()
    return modele, pd.DataFrame(historique), time.time() - t0


@torch.no_grad()
def predire(modele: PatchTST, jeu: JeuDonnees, orig: np.ndarray,
            valeurs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Prévisions et observations (MW) pour une liste d'origines : [n, H]."""
    cfg = modele.cfg
    v = jeu.valeurs if valeurs is None else valeurs
    dl = torch.utils.data.DataLoader(FenetresDataset(v, orig, cfg.L, cfg.H),
                                     batch_size=512, shuffle=False)
    preds, vrais = [], []
    for passe, futur, cible in dl:
        preds.append(modele(passe, futur, _cz_brut(jeu, futur)).numpy())
    y_pred = jeu.vers_mw(np.concatenate(preds))
    # les observations sont lues sur les données ORIGINALES (non bruitées)
    y_vrai = np.stack([jeu.tableau[CIBLE].values[t:t + cfg.H] for t in orig])
    return np.clip(y_pred, 0, None), y_vrai


# =============================================================================
# 7. MÉTRIQUES & RÉFÉRENCES
# =============================================================================

def metriques(y_vrai: np.ndarray, y_pred: np.ndarray, moyenne_ref: float | None = None,
              masque: np.ndarray | None = None) -> dict:
    """RMSE, MAE, nRMSE = RMSE / moyenne(production), R², MBE (en MW).

    `masque` permet de restreindre le calcul (ex. heures diurnes).
    """
    yv, yp = np.asarray(y_vrai, float).ravel(), np.asarray(y_pred, float).ravel()
    if masque is not None:
        m = np.asarray(masque).ravel()
        yv, yp = yv[m], yp[m]
    e = yp - yv
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ref = float(np.mean(yv)) if moyenne_ref is None else moyenne_ref
    return dict(RMSE=rmse, MAE=float(np.mean(np.abs(e))), nRMSE=rmse / ref,
                R2=float(1 - np.sum(e ** 2) / np.sum((yv - yv.mean()) ** 2)),
                MBE=float(np.mean(e)))


def reference_persistance(jeu: JeuDonnees, orig: np.ndarray, H: int) -> np.ndarray:
    """Persistance journalière : ŷ(t+h) = y(même heure, dernier jour observé).

    Pour h ≤ 24 : y(t+h-24) ; au-delà, on répète le dernier jour observé.
    """
    y = jeu.tableau[CIBLE].values
    h = np.arange(H)
    return np.stack([y[t - 24 + (h % 24)] for t in orig])


def reference_moyenne(jeu: JeuDonnees, orig: np.ndarray, H: int, jours: int = 7) -> np.ndarray:
    """Moyenne glissante : ŷ(t+h) = moyenne des `jours` derniers jours à la même heure."""
    y = jeu.tableau[CIBLE].values
    h = np.arange(H)
    return np.stack([np.mean([y[t - 24 * k + (h % 24)] for k in range(1, jours + 1)], axis=0)
                     for t in orig])


def test_diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Test de Diebold-Mariano (perte quadratique) avec correction de
    Harvey-Leybourne-Newbold. H0 : précisions égales. Stat < 0 ⇒ modèle 1 meilleur.

    e1, e2 : erreurs [n, H] ; on agrège par origine (moyenne des carrés sur
    l'horizon) pour obtenir une série de pertes, puis variance de long terme
    de Newey-West avec h-1 retards.
    """
    from scipy import stats
    d = (np.asarray(e1) ** 2).mean(axis=-1) - (np.asarray(e2) ** 2).mean(axis=-1)
    n = len(d)
    d_bar = d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[:n - k] - d_bar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = d_bar / math.sqrt(max(var, 1e-12))
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)      # correction HLN
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


# =============================================================================
# 8. PROTOCOLE D'ÉVALUATION D'UNE CONFIGURATION
# =============================================================================

def bruiter_exogenes(jeu: JeuDonnees, niveau: float = 0.05, graine: int = 123) -> np.ndarray:
    """Test de robustesse : bruit gaussien de niveau 5 % (σ = 5 % de l'écart-type
    Train de chaque canal) ajouté à TOUS les canaux exogènes (réseau passé et
    covariables déterministes) — la cible n'est pas bruitée. En l'absence de
    météo dans le journal, ce sont les covariables qui jouent le rôle de la
    « météo » de projet.md."""
    rng = np.random.default_rng(graine)
    v = jeu.valeurs.copy()
    v[:, 1:] += rng.normal(0, niveau, size=v[:, 1:].shape).astype(np.float32)
    return v


def rmse_journaliers(y_vrai, y_pred, orig, jeu: JeuDonnees) -> pd.Series:
    """RMSE par jour calendaire de l'origine de prévision."""
    jours = jeu.index[orig].normalize()
    e2 = ((y_pred - y_vrai) ** 2).mean(axis=1)
    return pd.Series(e2, index=jours).groupby(level=0).mean().pipe(np.sqrt)


def mesurer_efficience(modele: PatchTST, jeu: JeuDonnees, repetitions: int = 50) -> dict:
    """Temps d'inférence d'UN échantillon (ms, médiane) et taille du modèle (Mo)."""
    cfg = modele.cfg
    ds = FenetresDataset(jeu.valeurs, np.array([jeu.i_test]), cfg.L, cfg.H)
    passe, futur, _ = ds[0]
    passe, futur = passe[None], futur[None]
    cz = _cz_brut(jeu, futur)
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(5):
            modele(passe, futur, cz)
        temps = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            modele(passe, futur, cz)
            temps.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tampon = io.BytesIO()
    torch.save(modele.state_dict(), tampon)
    return dict(inference_ms=float(np.median(temps)), taille_Mo=tampon.tell() / 2 ** 20,
                n_parametres=int(sum(p.numel() for p in modele.parameters())))


def evaluer_configuration(cfg: ConfigPatchTST, jeu: JeuDonnees, graines=GRAINES,
                          jours_rolling: int = 30, forcer: bool = False,
                          verbeux: bool = True) -> dict:
    """Évalue une configuration selon les 4 critères de projet.md.

    Pour chaque graine : entraînement, prévision sur tout le Test (toutes les
    origines horaires), test de bruit 5 %, rolling 30 jours sans
    ré-entraînement. Les résultats (et les poids) sont mis en cache :
    supprimer le dossier cache/<nom> pour ré-entraîner.
    """
    dossier = DOSSIER_CACHE / cfg.nom()
    fichier = dossier / "resultats.json"
    if fichier.exists() and not forcer:
        return json.loads(fichier.read_text())
    dossier.mkdir(parents=True, exist_ok=True)

    o_test = origines(jeu.i_test, len(jeu.valeurs), cfg.L, cfg.H, 1)
    fin_rolling = jeu.i_test + 24 * jours_rolling
    o_roll = o_test[o_test < fin_rolling]
    v_bruit = bruiter_exogenes(jeu)
    par_graine = []
    for graine in graines:
        modele, hist, duree = entrainer(cfg, jeu, graine)
        y_pred, y_vrai = predire(modele, jeu, o_test)
        y_bruit, _ = predire(modele, jeu, o_test, v_bruit)
        jour = (y_vrai > 0) | (y_pred > 0)
        m = metriques(y_vrai, y_pred)
        m_jour = metriques(y_vrai, y_pred, moyenne_ref=float(y_vrai.mean()), masque=jour)
        rmse_bruit = metriques(y_vrai, y_bruit)["RMSE"]
        # rolling 30 jours : prévisions dont l'origine est dans les 30 premiers jours
        k = np.isin(o_test, o_roll)
        rj = rmse_journaliers(y_vrai[k], y_pred[k], o_roll, jeu)
        pente = float(np.polyfit(np.arange(len(rj)), rj.values, 1)[0])
        eff = mesurer_efficience(modele, jeu)
        par_graine.append(dict(
            graine=graine, duree_s=duree, epoques=len(hist), **m,
            RMSE_jour=m_jour["RMSE"], nRMSE_jour=m_jour["nRMSE"],
            RMSE_bruit=rmse_bruit, degradation_bruit=rmse_bruit / m["RMSE"] - 1,
            drift_J30_J1=float(rj.iloc[-1] / rj.iloc[0]),
            drift_7j=float(rj.iloc[-7:].mean() / rj.iloc[:7].mean()),
            pente_MW_par_jour=pente, rmse_journaliers=rj.round(4).tolist(), **eff))
        np.save(dossier / f"pred_graine{graine}.npy", y_pred.astype(np.float32))
        torch.save(modele.state_dict(), dossier / f"poids_graine{graine}.pt")
        hist.to_csv(dossier / f"historique_graine{graine}.csv", index=False)
        if verbeux:
            print(f"  {cfg.nom()}  graine {graine}: RMSE={m['RMSE']:.3f} MW  "
                  f"nRMSE={m['nRMSE']:.3f}  bruit={par_graine[-1]['degradation_bruit']:+.1%}  "
                  f"drift7j={par_graine[-1]['drift_7j']:.2f}  ({duree:.0f}s, {len(hist)} ép.)")
    np.save(dossier / "vrai.npy", y_vrai.astype(np.float32))
    np.save(dossier / "origines.npy", o_test)
    res = dict(config=asdict(cfg), nom=cfg.nom(), par_graine=par_graine,
               synthese=synthese_graines(par_graine))
    fichier.write_text(json.dumps(res, indent=1, default=str))
    return res


def synthese_graines(par_graine: list[dict]) -> dict:
    """Moyennes et écarts-types (ddof=1) sur les graines."""
    df = pd.DataFrame(par_graine)
    cles = ["RMSE", "MAE", "nRMSE", "R2", "MBE", "RMSE_jour", "nRMSE_jour", "degradation_bruit",
            "drift_J30_J1", "drift_7j", "pente_MW_par_jour", "inference_ms", "taille_Mo",
            "n_parametres", "duree_s", "epoques"]
    s = {f"{k}_moy": float(df[k].mean()) for k in cles}
    s.update({f"{k}_std": float(df[k].std(ddof=1)) for k in ["RMSE", "MAE", "nRMSE", "R2"]})
    return s


# =============================================================================
# 9. SCORE DE DÉCISION (projet.md)
# =============================================================================

def score_decision(synthese: dict) -> dict:
    """Score = 0,4·(1/nRMSE) + 0,3·(1/std) + 0,3·(1/drift) + bonus efficience.

    - nRMSE : moyenne sur 5 graines (toutes heures) ;
    - std   : écart-type du RMSE (MW) sur 5 graines ;
    - drift : ratio lissé RMSE(J24–J30)/RMSE(J1–J7) — le ratio brut J30/J1
              dépend de la nébulosité de deux journées isolées et est rapporté
              à titre indicatif ;
    - bonus : +0,1 si inférence < 100 ms, +0,1 si poids < 50 Mo ;
    - rejet si la dégradation sous bruit 5 % dépasse 15 %.
    """
    std = max(synthese["RMSE_std"], 1e-3)
    bonus = 0.1 * (synthese["inference_ms_moy"] < 100) + 0.1 * (synthese["taille_Mo_moy"] < 50)
    score = (0.4 / synthese["nRMSE_moy"] + 0.3 / std + 0.3 / synthese["drift_7j_moy"] + bonus)
    return dict(score=float(score), bonus=float(bonus),
                rejete_bruit=bool(synthese["degradation_bruit_moy"] > 0.15))


def tableau_comparatif(resultats: list[dict]) -> pd.DataFrame:
    """Tableau comparatif (une ligne par configuration) avec scores brut et normalisé.

    Le score normalisé ramène chaque critère dans [0, 1] (min-max entre les
    configurations comparées) avant pondération 40/30/30, pour éviter qu'un
    terme d'échelle plus grande (1/std) ne domine. Pour la dérive, seule une
    DÉGRADATION est pénalisée (dérive > 1) : une baisse du RMSE au fil du mois
    reflète la météo (entrée en saison sèche), pas un mérite du modèle.
    """
    lignes = []
    for r in resultats:
        s = r["synthese"]
        sc = score_decision(s)
        lignes.append(dict(configuration=r["nom"], RMSE=s["RMSE_moy"], RMSE_std=s["RMSE_std"],
                           MAE=s["MAE_moy"], nRMSE=s["nRMSE_moy"], nRMSE_jour=s["nRMSE_jour_moy"],
                           R2=s["R2_moy"], MBE=s["MBE_moy"],
                           degradation_bruit=s["degradation_bruit_moy"],
                           drift_J30_J1=s["drift_J30_J1_moy"], drift_7j=s["drift_7j_moy"],
                           inference_ms=s["inference_ms_moy"], taille_Mo=s["taille_Mo_moy"],
                           n_parametres=int(s["n_parametres_moy"]), **sc))
    df = pd.DataFrame(lignes)

    def mm(x, inverse=False):
        x = x.astype(float)
        r = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0 + 1
        return 1 - r if inverse else r
    df["score_normalise"] = (0.4 * mm(df["nRMSE"], True) + 0.3 * mm(df["RMSE_std"], True)
                             + 0.3 * mm(df["drift_7j"].clip(lower=1.0), True) + df["bonus"] / 2)
    return df
