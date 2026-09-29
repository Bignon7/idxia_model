"""
train_model.py

Entraine deux modeles (Random Forest et XGBoost) sur le dataset IDXIA,
les compare sur des metriques standards de classification, puis
sauvegarde le meilleur modele ainsi que les encodeurs utilises
(necessaires plus tard pour l'API de prediction).

Usage :
    python train_model.py
"""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

from preprocessing import prepare_dataset

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "cybersecurity_intrusion_data.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate_model(name: str, model, X_test, y_test) -> tuple[float, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"ROC AUC : {auc:.4f}")
    print(f"F1-score : {f1:.4f}")
    return f1, auc


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, encoders, feature_cols = prepare_dataset(
        DATA_PATH
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_f1, rf_auc = evaluate_model("Random Forest", rf, X_test, y_test)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_f1, xgb_auc = evaluate_model("XGBoost", xgb, X_test, y_test)

    # --- Selection du meilleur modele sur le F1-score ---
    if xgb_f1 >= rf_f1:
        best_model, best_name = xgb, "xgboost"
    else:
        best_model, best_name = rf, "random_forest"

    print(f"\nModele retenu : {best_name}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, f"idxia_model_{best_name}.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "idxia_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "idxia_feature_cols.pkl"))

    print("Modele et encodeurs sauvegardes dans", MODEL_DIR)


if __name__ == "__main__":
    main()