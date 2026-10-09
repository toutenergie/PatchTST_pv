# -*- coding: utf-8 -*-
"""Tableau 1 de l'article : synthèse des 56 travaux de la revue.

Colonnes : numéro, référence courte, axe, méthode / objet, données ou
validation, apport pour la présente étude. Rédigé à partir des fiches du
corpus 303 (résumés) et des titres ; « s.o. » quand la fiche ne documente pas
l'élément.
"""
AXES = {
    "A": "Stabilité et intégration PV",
    "B": "Sénégal / Afrique",
    "C": "Revues prévision ENR",
    "D": "DL pour prévision PV/ENR",
    "E": "PatchTST et Transformers",
}

# (num, référence courte, axe, méthode/objet, données/validation, apport)
TABLEAU = [
    (1, "Shah et al., 2015", "A", "Revue des défis de stabilité liés au PV à grande échelle", "Revue", "Motivation : la variabilité PV menace la stabilité"),
    (2, "Kenyon et al., 2020", "A", "Revue stabilité/commande à forte part d'onduleurs", "Revue", "Dynamiques rapides : besoin d'anticipation"),
    (3, "Seneviratne & Ozansoy, 2016", "A", "Réponse en fréquence à la perte d'un groupe", "Revue", "Lien pénétration PV/éolien – réserve"),
    (4, "Cheng et al., 2020", "A", "Réglage de fréquence en faible inertie", "Revue", "La prévision alimente le réglage"),
    (5, "Johnson et al., 2019", "A", "Inertie de rotation et fiabilité", "Simulation", "Enjeu d'exploitation à forte part variable"),
    (6, "Smith et al., 2022", "A", "Stabilité et résilience vs décentralisation ENR", "Données réelles", "Oscillations journalières de la résilience"),
    (7, "Xu et al., 2025", "A", "Modèle couplé climat-énergie, pannes en cascade", "Données réelles (Porto Rico)", "Seuil ≈ 45 % de solaire sans stockage"),
    (8, "Beck & Hesse, 2007", "A", "Machine synchrone virtuelle", "Concept", "Réponse côté commande (complémentaire)"),
    (9, "Bevrani et al., 2014", "A", "Générateurs synchrones virtuels", "Revue", "Réponse côté commande (complémentaire)"),
    (10, "Che et al., 2025", "A", "Stratégies d'atténuation de la variabilité ENR", "Revue, 10 pays", "Prévision = levier d'intégration"),
    (11, "Yuan et al., 2017", "A", "Bénéfice de la prévision et du stockage (réseau isolé)", "Données réelles", "Valeur économique de la prévision"),
    (12, "Abdoulaye et al., 2020", "B", "Impact de l'intermittence PV sur la fréquence", "3 ans de mesures SENELEC", "Délestages liés au PV au Sénégal"),
    (13, "Sarr et al., 2018", "B", "Taux de pénétration PV maximal (PowerFactory)", "Simulation SENELEC", "17,3 % HTA / 16,4 % HTB (2020)"),
    (14, "Sarr et al., 2020", "B", "Optimisation du taux de pénétration PV", "Simulation SENELEC", "≈ 24 % avec stockage"),
    (15, "Dieng et al., 2025", "B", "Stabilité tension/fréquence à 57–67 % d'onduleurs", "Simulation DIgSILENT", "Contrainte : perte < 5,2 % de la puissance"),
    (16, "Fall et al., 2024", "B", "Réserves dimensionnées de façon probabiliste", "Données réelles SENELEC", "Réserve ↔ incertitude de prévision"),
    (17, "Ba et al., 2025", "B", "Congestion du transport (OPF, N-1) 2025–2030", "Simulation", "Pression croissante des ENR sur le réseau"),
    (18, "Ndiaye M. et al., 2025", "B", "Effet de 5 centrales PV sur tension, flux, pertes", "Simulation DIgSILENT", "Mêmes centrales que la cible étudiée"),
    (19, "Faye et al., 2025a", "B", "Tests de causalité et ACP sur la stabilité", "Données SENELEC", "Le PV est un déterminant de la fréquence"),
    (20, "Faye et al., 2025b", "B", "ACP sur la stabilité en fréquence PV-éolien", "Données SENELEC", "Le PV total pèse sur la fréquence"),
    (21, "Bloomfield et al., 2022", "B", "Variabilité et facteurs météo solaire/éolien", "Réanalyses, Sénégal/Kenya", "Régime solaire sénégalais stable"),
    (22, "Ndiaye F.A. et al., 2025", "B", "Performance réelle et prévision saisonnière de Diass", "Données réelles (23 MWc)", "Centrale incluse dans la cible"),
    (23, "Wang et al., 2019", "C", "Revue du deep learning pour la prévision ENR", "Revue", "Cadre de référence des architectures"),
    (24, "Benti et al., 2023", "C", "Revue ML/DL pour la prévision ENR", "Revue", "Verrous : données, interprétabilité"),
    (25, "Sharifzadeh et al., 2019", "C", "Comparaison ANN, SVR, GPR", "Comparative", "Importance des références simples"),
    (26, "Aslam et al., 2021", "C", "Revue DL charge et ENR (micro-réseaux)", "Revue", "Panorama des architectures"),
    (27, "Klaiber & Van Dinther, 2023", "C", "Revue systématique DL pour ENR variables", "Revue systématique", "10 approches, 3 domaines"),
    (28, "Khouili et al., 2025", "C", "Revue systématique DL pour la prévision PV", "26 articles", "Entrées surtout météo ; LSTM/CNN dominants"),
    (29, "Alazemi et al., 2024", "C", "Revue ML pour l'intégration ENR", "Revue systématique", "LSTM et ensembles recommandés"),
    (30, "Devaraj et al., 2021", "C", "Revue big data et DL pour la prévision énergétique", "Revue", "MAPE solaire moyen ≈ 10 %"),
    (31, "Alsafrani et al., 2025", "D", "CNN dilaté + LSTM résiduel bidirectionnel", "Jeu de référence", "Hybrides convolutifs-récurrents"),
    (32, "Gangwar et al., 2024", "D", "RNN-GRU amélioré (uni/multivarié)", "Simulation", "Prévision conjointe charge/PV"),
    (33, "Khan et al., 2022", "D", "ESN-CNN léger à connexions résiduelles", "Jeux de référence", "Souci d'efficience de calcul"),
    (34, "Sankarananth et al., 2023", "D", "LSTM-RL, CNN-PSO (métaheuristiques)", "Simulation", "Hybridation algorithmique"),
    (35, "Cheng et al., 2025", "D", "CNN-LSTM spatio-temporel sous contrainte de stabilité", "Données réelles", "−14,1 % de RMSE vs LSTM"),
    (36, "Abdelsattar et al., 2025", "D", "Comparaison de 8 architectures DL", "4 200 enregistrements réels", "Transformer standard : R² = 0,07"),
    (37, "Nie et al., 2023", "E", "PatchTST : patchs + indépendance des canaux", "Jeux de référence", "Architecture de base de l'étude"),
    (38, "Lv et al., 2026", "E", "PatchTST + détection d'anomalies + GAN", "6 centrales PV", "PatchTST appliqué au PV court terme"),
    (39, "Suresh, 2025", "E", "Banc de 5 Transformers + inférence conforme", "Toiture 5 kW, 5 ans", "Encodages cycliques ; PatchTST en tête"),
    (40, "El-kenawy et al., 2026", "E", "PatchTST + sélection de variables (SMOA)", "Données réelles", "Choix des entrées de PatchTST"),
    (41, "Guo et al., 2025", "D", "Modèle PV incrémental, oubli catastrophique", "s.o.", "Dérive dans le temps (rare dans le corpus)"),
    (42, "Lu et al., 2025", "E", "CT-PatchTST : dépendances canal-temps", "Données réelles (Danemark)", "Remise en cause de l'indépendance des canaux"),
    (43, "Gao et al., 2024", "E", "ZS-DT-PatchTST (normalisation adaptative)", "Données expérimentales", "Non-stationnarité → normalisation"),
    (44, "Zhao et al., 2024", "E", "CausalPatchTST + sélection causale", "2 fermes éoliennes", "−13,3 % de RMSE vs PatchTST"),
    (45, "Xu et al., 2025", "E", "PatchTST-GRU seq2seq + raffinement NWP", "Ferme éolienne", "Apport des prévisions météo (absentes ici)"),
    (46, "Zhang et al., 2024", "E", "QR-PatchTST probabiliste", "Campus ASU", "Extension probabiliste (perspective)"),
    (47, "Spencer et al., 2025", "E", "Apprentissage par transfert, 3 Transformers", "16 bâtiments", "PatchTST se transfère le mieux"),
    (48, "Yang et al., 2025", "E", "DiffTST : attention différentielle", "Jeux de référence", "Amélioration générique de PatchTST"),
    (49, "Lin et al., 2024", "E", "CycleNet : cycles récurrents apprenables", "Jeux de référence", "Intérêt d'une information cyclique explicite"),
    (50, "Li et al., 2025", "D", "Équations différentielles neuronales contrôlées", "4 parcs éoliens", "Correction du décalage de prévision"),
    (51, "Mamyrbayev et al., 2025", "D", "RF, XGBoost, LSTM vs persistance", "30 sites, 6 mois", "Comparaison explicite à la persistance"),
    (52, "Liao et al., 2024", "C", "Modélisation probabiliste et optimisation stochastique", "Revue", "Incertitude pour l'exploitation"),
    (53, "Anis et al., 2026", "E", "Prédiction de stabilité explicable (LIME)", "Jeu simulé", "Interprétabilité pour les opérateurs"),
    (54, "Babakhani et al., 2026", "E", "Mélange d'experts Informer explicable", "3 jeux de charge thermique", "Transformers explicables"),
    (55, "Lin et al., 2025", "C", "Revue : modèles guidés par données / mécanistes / hybrides", "Revue", "Limite « boîte noire » des modèles"),
    (56, "Coulibaly et al., 2025", "B", "MLP pour la prévision PV-éolien SENELEC", "Données SENELEC", "Seul précédent prédictif sénégalais"),
]
assert len(TABLEAU) == 56 and [t[0] for t in TABLEAU] == list(range(1, 57))
