# IDXIA — Module model

Pipeline d'entraînement du modèle de détection : téléchargement du dataset, préparation des données, entraînement comparatif, évaluation et explicabilité (SHAP). Les artefacts produits (modèle, encodeurs, liste des colonnes) sont consommés par l'API du dossier `backend/`.

## Structure

```
model/
├── data/
│   ├── import_data.py        Téléchargement du dataset Kaggle via kagglehub
│   └── raw/                  CSV téléchargé (non versionné)
├── src/
│   ├── preprocessing.py      Nettoyage, encodage, découpage train/test
│   ├── train_model.py        Entraînement Random Forest + XGBoost, évaluation, sauvegarde
│   └── explain.py            Explicabilité SHAP (importance globale et par session)
├── models/                   Modèle et encodeurs sauvegardés (non versionnés)
├── reports/                  Métriques JSON et graphiques SHAP
└── requirements.txt
```

## Installation

```bash
cd model
python3 -m venv venv
source venv/bin/activate        # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Pour quitter l'environnement virtuel : `deactivate`.

## Étape 1 — Télécharger le dataset

```bash
python3 data/import_data.py
```

Le script télécharge le dataset "Cybersecurity Intrusion Detection Dataset" (Kaggle, licence MIT) et le copie dans `data/raw/cybersecurity_intrusion_data.csv`. Si `kagglehub` réclame une authentification sur votre machine, suivre la procédure indiquée dans sa documentation (token à placer dans `~/.kaggle/kaggle.json`).

## Étape 2 — Entraîner le modèle

```bash
cd src
python3 train_model.py
```

Le script :
- charge et nettoie les données (doublons, valeurs manquantes) ;
- encode les variables catégorielles (protocole, chiffrement, navigateur) ;
- sépare les données en 80 % d'entraînement et 20 % de test (stratifié, graine fixe à 42) ;
- entraîne un Random Forest et un XGBoost ;
- affiche pour chacun la precision, le rappel, le F1-score, le ROC AUC et la matrice de confusion ;
- retient le modèle au meilleur F1-score et le sauvegarde.

Fichiers produits :

| Fichier | Contenu |
|---|---|
| `models/idxia_model_<nom>.pkl` | Modèle retenu (random_forest ou xgboost) |
| `models/idxia_encoders.pkl` | Encodeurs des variables catégorielles |
| `models/idxia_feature_cols.pkl` | Ordre exact des colonnes utilisé à l'entraînement |
| `reports/metrics_<horodatage>.json` | Métriques de cet entraînement (historique) |
| `reports/metrics_latest.json` | Métriques du dernier entraînement |

### Résultats observés

Sur le jeu de test (1 515 sessions) :

| Modèle | F1-score | ROC AUC | Rappel (attaque) | Précision (attaque) |
|---|---|---|---|---|
| Random Forest | 0,8627 | 0,8755 | 0,76 | 1,00 |
| XGBoost | 0,8559 | 0,8767 | 0,76 | 0,98 |

Le Random Forest est retenu. Le rappel de 0,76 sur la classe attaque signifie qu'environ un quart des attaques du jeu de test n'est pas détecté (faux négatifs). Ce point est traité dans le registre des risques.

## Étape 3 — Explicabilité (SHAP)

```bash
cd src
python3 explain.py
```

Le script charge le modèle le plus récent, génère le graphique d'importance globale des variables (`reports/shap_summary.png`) et affiche un exemple d'explication pour une session du jeu de test. La logique d'explication par session est reprise par l'API.

Le code gère les deux formats de sortie de la bibliothèque `shap` (liste de tableaux par classe pour les anciennes versions, tableau 3D pour les versions récentes).

## Limites

- Le dataset est synthétique : les performances mesurées ne se transposent pas telles quelles à du trafic réel.
- Le modèle travaille sur des variables de session liées à l'authentification (tentatives, échecs, réputation IP, horaire, navigateur) et non sur le contenu du trafic réseau.
- Le dossier `cic-validation/` du projet (validation sur CIC-IDS2017) est une piste d'approfondissement en pause, indépendante de ce pipeline.