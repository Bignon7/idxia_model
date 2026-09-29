# 1. Installez le paquet requis (si ce n'est pas déjà fait)
sudo apt update && sudo apt install python3-full -y

# 2. Créez l'environnement virtuel nommé idxia
python3 -m venv idxia

# 3. Activez l'environnement virtuel
source idxia/bin/activate

# 4. Installez enfin votre package sans erreur
pip install kagglehub


















# IDXIA — Système de détection d'intrusion explicable

## Installation

```bash
# 1. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate      # sous Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt
```

## Configuration Kaggle (une seule fois)

`kagglehub` a besoin d'un token API Kaggle :
1. Aller sur https://www.kaggle.com/settings
2. Section "API" → "Create New Token" → télécharge `kaggle.json`
3. Placer ce fichier dans `~/.kaggle/kaggle.json` (Linux/Mac) ou
   `C:\Users\<toi>\.kaggle\kaggle.json` (Windows)

## Étape 1 — Télécharger le dataset

```bash
python data/import_data.py
```

Le CSV atterrit dans `data/raw/cybersecurity_intrusion_data.csv`.

## Étape 2 — Entraîner le modèle

```bash
cd src
python train_model.py
```

Ce script :
- charge et nettoie les données
- encode les variables catégorielles (protocole, chiffrement, navigateur)
- entraîne un Random Forest **et** un XGBoost
- affiche les métriques de chacun (precision, recall, F1, ROC AUC, matrice de confusion)
- sauvegarde le meilleur des deux dans `models/`

## Prochaines étapes (à venir)

- `explain.py` : explicabilité SHAP
- API FastAPI de prédiction
- Dashboard web de supervision en temps réel
- Bot Telegram pour les alertes