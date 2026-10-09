# Projet: Prédiction Production Énergies Renouvelables - PatchTST Decision Tool
## Objectif
Développer un modèle PatchTST efficace (performant) et efficient (léger, rapide) qui sert d'outil d'aide à la décision robuste et stable long terme pour la production renouvelable.
## Type de problème
Time Series Forecasting unitivarié - Solaire 
Target: production_kW
## Règles d'or - À RESPECTER STRICTEMENT
1.  JAMAIS de shuffle. Split temporel uniquement: Train < Val < Test. Utiliser TimeSeriesSplit.
2.  Pas de data leakage: normalisation fit uniquement sur Train.
3.  Toujours comparer à baseline persistance et moyenne.
4.  Si incertitude > 10%, poser question avant de coder.
5. Ne jamais creer si tu comprend pas pose une question je repond
## Architecture PatchTST 
Il faut plusieurs configurations du modele et chaque configuration tu évalue et valide celui le plus performant en terme de précision, robustesse, stabilité
Configurations à tester:
Configurations 1 : Influence de la granularité temporelle P=12, 24, 48
Configurations 2 : Influence de l'historique L=96, 168, 336
Configurations 3 : Indépendance des canaux PatchTST−CI, PatchTST−CA
Configurations 4 : Horizon de prédiction 1h, 6h, 12h, 24h, 168h
## Critères d'évaluation - Score Final
1.  Précision (40%): RMSE, MAE, nRMSE = RMSE / mean(prod)
2.  Robustesse (30%): Std sur 5 seeds, Test robustesse: bruit gaussien 5% sur météo. Si dégradation RMSE >15% = rejeté
3.  Stabilité long terme (30%) - LE PLUS IMPORTANT:
    - Rolling forecast sans ré-entrainement sur 30 jours
    - Drift = RMSE_Jour30 / RMSE_Jour1. Objectif < 1.2
4.  Efficience: Temps inférence <100ms, poids modèle <50Mo
Score Final = 0.4 * (1/nRMSE) + 0.3 * (1/std) + 0.3 * (1/drift) + bonus efficience
Sauvegarder meilleur modèle dans /models/best_patchtst.pt
## Workflow attendu
1.  EDA: saisonnalité, corrélation, valeurs manquantes
2.  Preprocessing: resample horaire, imputation, scaling
3.  Tableau comparatif des métrique de performances pour chaque configuration 
4. graphe de comparaison série originale et celle prédite du meilleur modele de la configuration
5. Notebook python pour toute la démarche pour chaque configuration
6. organigramme pour chaque configuration 
6. document Word explicatif de votre démarche étape par étape
7. notebook du Modele finale qui regroupe les meilleurs modelés pour chaque configuration. et la meilleurs configuration finale tu mets les métriques, les graphes de résiduels, graphe réel prédite, chapelet
