# -*- coding: utf-8 -*-
"""
article_textes.py — Textes de l'article qui ne dépendent pas des résultats :
titre, introduction, revue (lue depuis revue/texte_revue.md), données et
méthodologie. Chaque section est une liste de blocs pour docx_rendu.js.
"""
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]

TITRE = ("Prévision multivariée de la production photovoltaïque totale d'un réseau national sans données "
         "météorologiques : un PatchTST à attention inter-canaux conditionné par des covariables cycliques et "
         "la géométrie solaire — application au réseau SENELEC (Sénégal)")
AUTEURS = "Abdoulaye FAYE^1,*^, Alphousseyni NDIAYE^1^"
AFFILIATION = ("^1^ Équipe de Recherche en Efficacité et Systèmes Énergétiques, École Doctorale des Sciences et Techniques "
               "et des Sciences de la Société (EDTSS), Université Alioune Diop de Bambey (UADB), Sénégal — "
               "^*^ Auteur correspondant : abdoulaye7.faye@uadb.edu.sn")
ENTETE = "Faye & Ndiaye — Prévision PatchTST de la production PV totale, réseau SENELEC"

MOTS_CLES = ("**Mots-clés :** prévision photovoltaïque ; PatchTST ; Transformer ; variables cycliques ; géométrie solaire ; "
             "attention inter-canaux ; stabilité long terme ; réseau SENELEC ; Afrique de l'Ouest")


def introduction():
    return [
        {"type": "h1", "texte": "1. Introduction"},
        {"type": "p", "texte": (
            "L'intégration massive des centrales photovoltaïques (PV) transforme l'exploitation des réseaux électriques. "
            "Parce qu'elle est raccordée par onduleur et dépend de la nébulosité, la production PV réduit l'inertie "
            "synchrone et introduit une variabilité rapide que le gestionnaire doit compenser par des réserves [1–5]. "
            "Des travaux récents montrent que, au-delà d'un certain taux d'intégration, la résilience d'un réseau peut "
            "se dégrader brutalement en l'absence de stockage et d'anticipation [6, 7]. La prévision de la production "
            "est donc l'un des premiers leviers d'une intégration sûre et économique [10, 11].")},
        {"type": "p", "texte": (
            "Le réseau de la SENELEC (Sénégal) illustre ces enjeux. Une dizaine de centrales PV de quelques dizaines "
            "de MWc chacune y ont été raccordées au réseau de transport depuis le milieu des années 2010. Leur intermittence a été associée à une gestion de la "
            "fréquence plus difficile et à des délestages plus fréquents [12]. Les taux de pénétration admissibles "
            "ont été estimés entre 16 et 24 % selon le niveau de tension et la présence de stockage [13, 14], et la "
            "production PV figure parmi les premiers déterminants statistiques de la fréquence du réseau [19, 20]. Or "
            "le dispatching ne dispose pas de mesures météorologiques synchrones sur les sites ; la seule information "
            "systématiquement disponible est le journal horaire d'exploitation.")},
        {"type": "p", "texte": (
            "La littérature de la prévision PV par apprentissage profond est abondante [23–30], mais trois limites "
            "restreignent sa transposition à ce contexte. D'abord, elle suppose presque toujours des variables "
            "météorologiques en entrée [28, 38, 45]. Ensuite, elle porte sur des centrales individuelles, non sur la "
            "production agrégée d'un réseau national, dont la capacité installée évolue à chaque mise en service. Enfin, "
            "les protocoles d'évaluation sont hétérogènes : la comparaison à une référence naïve, la dispersion entre "
            "initialisations, la robustesse, la stabilité des performances dans le temps et le coût d'inférence sont "
            "rarement mesurés ensemble [24, 41, 51]. Les Transformers à patchs, et PatchTST en particulier [37], ont "
            "renouvelé la prévision de séries temporelles et commencent à être appliqués au photovoltaïque [38–40]. "
            "Leur hypothèse d'indépendance des canaux est toutefois discutée dès que des covariables informatives sont "
            "disponibles [42, 45].")},
        {"type": "p", "texte": "Cet article apporte quatre contributions :"},
        {"type": "numeros", "instance": 1, "items": [
            "**Une formulation multivariée sans météorologie** de la prévision de la production PV totale d'un réseau "
            "national. Les productions des centrales individuelles sont exclues des entrées (elles ne servent qu'à "
            "construire la cible). Le modèle combine l'historique de la cible, le contexte d'exploitation du réseau "
            "(thermique, éolien, import) et des covariables **déterministes** : encodages cycliques et géométrie solaire "
            "calculée pour le barycentre du parc.",
            "**Une architecture PatchTST-CA-X** qui combine (i) une normalisation réversible **sélective**, appliquée aux "
            "seuls canaux stochastiques ; (ii) une attention **inter-canaux** par position de patch, où la cible interroge "
            "les covariables ; (iii) un **conditionnement de la tête de prévision par le futur connu**, puisque les "
            "covariables déterministes sont disponibles sans erreur à tout horizon.",
            "**Un protocole d'évaluation orienté décision.** Quatre configurations sont étudiées un facteur à la fois "
            "(longueur de patch, historique, traitement des canaux, horizon), chacune sur 5 graines. Le protocole mesure "
            "la précision, la robustesse au bruit, la dérive sur 30 jours de prévision sans ré-entraînement, l'efficience, "
            "une validation croisée temporelle et des tests de Diebold-Mariano. Le tout est agrégé dans un score de décision.",
            "**Une analyse d'explicabilité** (occultation de groupes de canaux, gradients intégrés, poids d'attention) qui "
            "quantifie ce que chaque famille de covariables apporte réellement. Un canal sans lien physique avec "
            "l'irradiance (le jour de la semaine) sert de témoin négatif.",
        ]},
        {"type": "p", "texte": (
            "La section 2 présente la revue de la littérature. La section 3 décrit les données, l'architecture et le "
            "protocole. La section 4 rapporte les résultats, que la section 5 discute. Les sections 6 et 7 exposent les "
            "limites et les perspectives, et la section 8 conclut.")},
    ]


def revue():
    """Lit revue/texte_revue.md et le convertit en blocs."""
    texte = (RACINE / "revue" / "texte_revue.md").read_text(encoding="utf-8")
    blocs, liste = [], []

    def vider():
        nonlocal liste
        if liste:
            blocs.append({"type": "numeros", "instance": 2, "items": liste}); liste = []
    for ligne in texte.split("\n"):
        l = ligne.strip()
        if not l:
            continue
        if l.startswith("### "):
            vider(); blocs.append({"type": "h2", "texte": l[4:]})
        elif l.startswith("## "):
            vider(); blocs.append({"type": "h1", "texte": l[3:]})
        elif re.match(r"^\d+\. ", l):
            liste.append(re.sub(r"^\d+\. ", "", l))
        elif l.startswith("(i)") or l.startswith("(ii)") or l.startswith("(iii)"):
            blocs[-1]["texte"] += " " + l
        else:
            vider(); blocs.append({"type": "p", "texte": l})
    vider()
    return blocs


def tableau_revue():
    import sys
    sys.path.insert(0, str(RACINE / "revue"))
    from tableau_synthese_56 import AXES, TABLEAU
    return {"type": "tableau", "taille": 14, "largeurs": [0.5, 1.9, 1.2, 3.2, 1.9, 3.0], "gauche": [1, 3, 5],
            "legende": "**Tableau 1.** Synthèse des 56 travaux de la revue (axes : A stabilité et intégration PV, "
                       "B Sénégal/Afrique, C revues de prévision, D apprentissage profond PV/ENR, E PatchTST et Transformers).",
            "entetes": ["N°", "Référence", "Axe", "Méthode / objet", "Données / validation", "Apport pour cette étude"],
            "lignes": [[f"[{n}]", ref, f"{ax}", meth, don, app] for n, ref, ax, meth, don, app in TABLEAU],
            "note": "Rédigé à partir des résumés et des fiches du corpus ; « s.o. » : non documenté dans la source."}


def methodologie(D):
    """D : dictionnaire de valeurs issues des données (période, effectifs, décalage…)."""
    return [
        {"type": "h1", "texte": "3. Données et méthodologie"},
        {"type": "h2", "texte": "3.1 Données"},
        {"type": "p", "texte": (
            f"Les données proviennent du journal horaire d'exploitation de la SENELEC, du {D['debut']} au {D['fin']} "
            f"({D['n_heures']} heures ; {D['n_manquantes']} heures absentes, réparties en huit sauts d'un à trois jours). "
            "Le journal contient la production des unités thermiques, de la centrale éolienne de Taiba Ndiaye, des "
            "importations (Manantali, Félou, SOMELEC) et de onze centrales photovoltaïques. La cible est la production "
            "PV totale du réseau (Figure 1). Son niveau moyen augmente de "
            f"{D['croissance']:.0f} % entre 2019 et 2022, au rythme des mises en service : Diass (juin 2019), remplacement "
            "de Kahone par ERS et Scaling Kahone (janvier 2021), Kael Touba (mars 2021).").replace(",", " ")},
        {"type": "figure", "chemin": "figures/fig01_serie_pv_total.png", "legende":
            "**Figure 1.** Production PV totale horaire du réseau SENELEC (2019–2023), enveloppe de capacité apparente "
            "et énergie journalière. Fonds orange et rouge : périodes de validation et de test ; pointillés : mises en service."},
        {"type": "h2", "texte": "3.2 Construction de la cible et prétraitement"},
        {"type": "p", "texte": (
            "La cible est la somme des onze centrales PV, construite en trois étapes pour chaque centrale c. (i) Les "
            "valeurs supérieures à 1,2 fois le quantile 99,9 % calculé sur la période d'entraînement sont considérées "
            f"comme des erreurs de saisie ({D['n_ecretees']} valeurs, jusqu'à 958 MW pour une centrale de 20 MW). (ii) En "
            "dehors de la période d'existence de la centrale, la production vaut zéro. (iii) Les lacunes sont imputées "
            "de manière **causale** : interpolation linéaire jusqu'à 3 h, puis moyenne de la même heure sur les sept "
            "jours précédents. Le Tableau 2 détaille ces opérations. Les productions individuelles ne sont **jamais** "
            "utilisées comme entrées du modèle.")},
        {"type": "equation", "texte": "y(t) = Σ_{c=1..11} p̃_{c}(t)", "numero": 1},
        {"type": "h2", "texte": "3.3 Covariables"},
        {"type": "p", "texte": (
            "Trois familles de covariables sont construites (douze canaux au total, cible comprise). Les **encodages "
            "cycliques** assurent la continuité aux frontières des cycles (23 h → 0 h, 31 décembre → 1^er^ janvier). Ils "
            "sont calculés au milieu de chaque heure, pour l'heure h (T = 24), le jour de l'année (T = 365,25) et le jour "
            "de la semaine (T = 7) :")},
        {"type": "equation", "texte": "x_{sin}(t) = sin(2π·u(t)/T),   x_{cos}(t) = cos(2π·u(t)/T)", "numero": 2},
        {"type": "p", "texte": (
            "La **géométrie solaire** est calculée avec pvlib [62] au barycentre approximatif des centrales (14,9° N ; "
            "16,4° O). Elle comprend le cosinus de l'angle zénithal apparent θ_{z} (borné à 0 la nuit) et le rayonnement "
            "global de ciel clair selon le modèle de Haurwitz [61] :")},
        {"type": "equation", "texte": "GHI_{cs}(t) = 1098 · cos θ_{z}(t) · exp(−0,057 / cos θ_{z}(t))", "numero": 3},
        {"type": "p", "texte": (
            "Un décalage horaire δ entre l'horodatage du journal et le temps solaire est estimé sur la période "
            "d'entraînement en maximisant la corrélation entre le profil journalier moyen de la cible et celui de "
            f"GHI_{{cs}} (δ = {D['decalage']:+.2f} h). Ces huit covariables sont **déterministes** : elles sont connues "
            "sans erreur à tout horizon. Le **contexte réseau** (production thermique agrégée, éolien de Taiba, "
            "importation de Manantali) sert de proxy de l'état atmosphérique : la demande nette et le régime de vent "
            "reflètent la nébulosité et l'advection. Ces trois canaux ne sont utilisés que sur la fenêtre passée. Les "
            "colonnes agrégées du journal qui incluent la production PV (total des unités, réseau interconnecté) sont "
            "exclues, afin d'éviter toute fuite indirecte.")},
        {"type": "h2", "texte": "3.4 Architecture PatchTST-CA-X"},
        {"type": "p", "texte": (
            "PatchTST [37] découpe chaque canal de la fenêtre d'entrée de longueur L en N patchs de longueur P, avec un "
            "pas S = P/2 :")},
        {"type": "equation", "texte": "N = ⌊(L − P) / S⌋ + 1", "numero": 4},
        {"type": "p", "texte": (
            "Chaque patch est projeté dans un espace de dimension D = 64 et additionné d'un plongement de position et "
            "d'un plongement d'identité de canal. Un encodeur Transformer [58] de deux couches (quatre têtes, "
            "pré-normalisation), dont les poids sont **partagés** entre canaux, modélise ensuite les dépendances "
            "temporelles. L'architecture proposée (Figure 2) ajoute trois éléments :")},
        {"type": "puces", "items": [
            "**RevIN sélective.** La normalisation réversible par instance [57] est appliquée à la cible et aux canaux "
            "réseau, dont le niveau dérive avec la capacité installée. Les covariables déterministes conservent leur "
            "normalisation globale. En effet, sur une fenêtre de quelques jours, le jour de l'année varie à peine : le "
            "centrer-réduire effacerait la position dans l'année au profit d'une rampe artificielle. Nous avons observé "
            "ce défaut dans une première version, où il se traduisait par une attribution anormale de 24 % au jour de "
            "l'année passé.",
            "**Attention inter-canaux (CA).** À chaque position de patch n, le jeton du canal cible sert de requête et "
            "les C jetons des canaux servent de clés et de valeurs : z'_{n} = LN(z_{n,0} + MHA(z_{n,0}, Z_{n}, Z_{n})). En "
            "indépendance stricte des canaux (CI), la tête ne reçoit que le jeton de la cible : les covariables n'ont alors "
            "**aucune** influence sur la prévision, et le modèle équivaut à un PatchTST univarié à poids partagés.",
            "**Conditionnement par le futur connu (X).** La tête linéaire produit une prévision de base ŷ_{0} ∈ R^H^. "
            "Pour chaque pas h, un perceptron à une couche cachée reçoit un contexte c = W·vec(z') ∈ R^32^, les huit "
            "covariables déterministes de l'instant t+h et ŷ_{0}(h), et produit une correction additive : "
            "ŷ(h) = ŷ_{0}(h) + g([c ; f(t+h) ; ŷ_{0}(h)]). Une variante (g) ajoute une porte physique qui annule la "
            "prévision lorsque cos θ_{z}(t+h) = 0.",
        ]},
        {"type": "figure", "chemin": "figures/fig_architecture.png", "largeur_cm": 16.5, "legende":
            "**Figure 2.** Architecture PatchTST-CA-X. En rouge : éléments proposés (attention inter-canaux, "
            "conditionnement par les covariables déterministes futures). La variante CI (PatchTST standard) n'utilise "
            "que le jeton de la cible."},
        {"type": "h2", "texte": "3.5 Protocole expérimental"},
        {"type": "p", "texte": (
            "**Découpage temporel strict**, sans mélange : entraînement du 01/01/2019 au 31/12/2021 "
            f"({D['n_train']} h), validation du 01/01/2022 au 30/09/2022 ({D['n_val']} h), test du 01/10/2022 au "
            f"09/05/2023 ({D['n_test']} h). Toutes les opérations ajustées sur les données (seuils d'écrêtage, "
            "normalisation globale, pénalité de la référence Ridge) le sont sur la seule période d'entraînement. Une "
            "origine de prévision t signifie que la dernière heure observée est t − 1 ; le modèle prévoit directement "
            "y(t), …, y(t+H−1).").replace(",", " ")},
        {"type": "p", "texte": (
            "**Entraînement.** Perte quadratique ; Adam (taux initial 10^−3^, réduit de moitié après deux époques sans "
            "amélioration) ; lots de 128 ; écrêtage du gradient à 1 ; au plus 25 époques ; arrêt précoce (patience 4) "
            "sur la perte de validation, avec restauration des meilleurs poids. À chaque époque, 20 % des origines "
            "horaires d'entraînement sont tirées au hasard, ce qui garantit que toutes les phases horaires sont vues. Un "
            "pas fixe de 3 h, par exemple, n'en montrerait que 8 sur 24. Chaque configuration est entraînée avec "
            "5 graines. Le modèle est implémenté en PyTorch [64].")},
        {"type": "p", "texte": (
            "**Plan d'expériences « un facteur à la fois »** (Tableau 3). La valeur retenue à une étape est reportée à "
            "l'étape suivante. Le choix se fait sur le score de décision normalisé, parmi les variantes qui passent le "
            "test de bruit.")},
        {"type": "tableau", "legende": "**Tableau 3.** Plan d'expériences.", "largeurs": [1.4, 2.2, 3.4, 3.0],
         "gauche": [3],
         "entetes": ["Configuration", "Facteur", "Valeurs", "Paramètres fixés"],
         "lignes": [["1", "Longueur de patch P", "12, 24, 48 h", "L = 168 h, CA-X, H = 24 h"],
                    ["2", "Historique L", "96, 168, 336 h", "P*, CA-X, H = 24 h"],
                    ["3", "Traitement des canaux", "CI, CI-X, CA, CA-X, CA-Xg", "P*, L*, H = 24 h"],
                    ["4", "Horizon H", "1, 6, 12, 24, 168 h", "P*, L*, canaux*"]],
         "note": "* : valeur retenue à l'étape précédente."},
        {"type": "p", "texte": (
            "**Références.** (i) Persistance journalière : ŷ(t+h) = y(t+h−24), avec répétition du dernier jour observé "
            "au-delà de 24 h. (ii) Moyenne des sept derniers jours à la même heure. (iii) Régression Ridge directe "
            "multi-sortie, dont les entrées sont les L dernières valeurs de la cible et les covariables déterministes "
            "des H pas futurs ; la pénalité est choisie sur la validation. (iv) PatchTST-CI, équivalent univarié.")},
        {"type": "h2", "texte": "3.6 Critères d'évaluation et score de décision"},
        {"type": "p", "texte": (
            "Les métriques sont calculées sur **toutes les origines horaires** du test, tous pas d'horizon confondus :")},
        {"type": "equation", "texte": "RMSE = √( (1/n) Σ (ŷ_{i} − y_{i})² ),   nRMSE = RMSE / ȳ", "numero": 5},
        {"type": "equation", "texte": "MAE = (1/n) Σ |ŷ_{i} − y_{i}|", "numero": 6},
        {"type": "p", "texte": (
            "complétées par le R², le biais moyen (MBE) et le nRMSE restreint aux heures diurnes. La **robustesse** "
            "combine l'écart-type du RMSE sur les 5 graines, σ, et la dégradation relative du RMSE lorsque les canaux "
            "exogènes (passés et futurs) sont perturbés par un bruit gaussien d'écart-type égal à 5 % de leur écart-type "
            "d'entraînement. Une dégradation supérieure à 15 % entraîne le rejet de la configuration. La **stabilité "
            "long terme** est évaluée par une prévision glissante de 30 jours sans ré-entraînement, au début du test. Le "
            "ratio RMSE(J30)/RMSE(J1) dépend de la nébulosité de deux journées isolées ; nous retenons donc un ratio "
            "lissé :")},
        {"type": "equation", "texte": "Dérive = RMSE(J24…J30) / RMSE(J1…J7)   (objectif < 1,2)", "numero": 6},
        {"type": "p", "texte": (
            "L'**efficience** est mesurée par le temps d'inférence d'un échantillon sur un cœur de processeur (objectif "
            "< 100 ms) et par la taille des poids (< 50 Mo). Le **score de décision** agrège ces critères :")},
        {"type": "equation", "texte": "S = 0,4 / nRMSE + 0,3 / σ + 0,3 / Dérive + bonus", "numero": 8},
        {"type": "p", "texte": (
            "où le bonus vaut 0,1 par critère d'efficience satisfait. Les trois termes ont des échelles différentes : "
            "1/σ, en particulier, peut dominer lorsque σ est très faible. Nous rapportons donc aussi un score normalisé, "
            "dans lequel chaque critère est ramené à [0, 1] entre les variantes comparées ; c'est ce score qui guide la "
            "sélection. L'égalité de précision avec les références est testée par le test de Diebold-Mariano [59], avec "
            "la correction de Harvey et al. [60] et une variance de long terme à H − 1 retards. Enfin, la configuration "
            "finale est soumise à une **validation croisée temporelle** (TimeSeriesSplit, 5 plis à fenêtre croissante "
            "couvrant 2019–2023, normalisation ré-ajustée à chaque pli), à une ablation par **indice de capacité** "
            "causal, et à une analyse d'**explicabilité** : occultation de groupes de canaux, gradients intégrés [63] "
            "et poids de l'attention inter-canaux.")},
    ]


REFERENCES_METHODO = [
    "[57] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo, Reversible instance normalization for accurate time-series forecasting against distribution shift, in: Proc. 10th Int. Conf. Learn. Represent. (ICLR), 2022.",
    "[58] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Adv. Neural Inf. Process. Syst. 30 (NeurIPS), 2017.",
    "[59] F.X. Diebold, R.S. Mariano, Comparing predictive accuracy, Journal of Business & Economic Statistics 13 (1995) 253–263. https://doi.org/10.1080/07350015.1995.10524599",
    "[60] D. Harvey, S. Leybourne, P. Newbold, Testing the equality of prediction mean squared errors, International Journal of Forecasting 13 (1997) 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4",
    "[61] B. Haurwitz, Insolation in relation to cloudiness and cloud density, Journal of Meteorology 2 (1945) 154–166. https://doi.org/10.1175/1520-0469(1945)002<0154:IIRTCA>2.0.CO;2",
    "[62] W.F. Holmgren, C.W. Hansen, M.A. Mikofski, pvlib python: a python package for modeling solar energy systems, Journal of Open Source Software 3 (2018) 884. https://doi.org/10.21105/joss.00884",
    "[63] M. Sundararajan, A. Taly, Q. Yan, Axiomatic attribution for deep networks, in: Proc. 34th Int. Conf. Mach. Learn. (ICML), PMLR 70, 2017, pp. 3319–3328.",
    "[64] A. Paszke et al., PyTorch: An imperative style, high-performance deep learning library, in: Adv. Neural Inf. Process. Syst. 32 (NeurIPS), 2019.",
]


def references():
    lignes = (RACINE / "revue" / "liste_56_utilisateur.txt").read_text(encoding="utf-8").strip().split("\n")
    return [{"type": "h1", "texte": "Références"},
            {"type": "references", "items": [l.strip() for l in lignes] + REFERENCES_METHODO}]
