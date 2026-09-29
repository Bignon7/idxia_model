"""
preprocessing.py

Chargement et preparation des donnees pour IDXIA.
Separe volontairement de train_model.py pour pouvoir reutiliser
cette logique plus tard dans l'API de prediction (meme encodage
au moment de l'entrainement et au moment de la prediction en live).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["protocol_type", "encryption_used", "browser_type"]
TARGET_COL = "attack_detected"
ID_COL = "session_id"


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_features(df: pd.DataFrame):
    """Encode les colonnes categorielles en entiers et renvoie les encodeurs
    utilises, pour pouvoir re-appliquer exactement le meme encodage plus
    tard (API de prediction) sur de nouvelles donnees."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def prepare_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Pipeline complet : charge, nettoie, encode et separe train/test."""
    df = load_data(csv_path)
    df = clean_data(df)
    df, encoders = encode_features(df)

    feature_cols = [c for c in df.columns if c not in (TARGET_COL, ID_COL)]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoders, feature_cols