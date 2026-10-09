"""
explain.py

Genere l'explicabilite du modele retenu avec SHAP :
- importance globale des features (quelles colonnes comptent le plus)
- graphique "summary plot" sauvegarde en image
- fonction reutilisable pour expliquer UNE prediction individuelle
  (utile plus tard dans l'API : "pourquoi cette session est signalee ?")

Usage :
    python3 explain.py
"""

import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from preprocessing import prepare_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "cybersecurity_intrusion_data.csv"
)


def load_latest_model():
    """Charge le modele sauvegarde (peu importe si c'est random_forest ou
    xgboost, le nom de fichier varie selon celui retenu a l'entrainement)."""
    candidates = glob.glob(os.path.join(MODEL_DIR, "idxia_model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            "Aucun modele trouve dans models/. Lance train_model.py d'abord."
        )
    model_path = max(candidates, key=os.path.getmtime)
    print("Modele charge :", model_path)
    return joblib.load(model_path)


def explain_global(model, X_test, feature_cols):
    """Calcule et affiche/sauvegarde l'importance globale des features."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Ancienne API : liste [array_classe_0, array_classe_1]
    # Nouvelle API : array numpy (n_echantillons, n_features, n_classes)
    if isinstance(shap_values, list):
        values_for_plot = shap_values[1]
    else:
        values_arr = shap_values if hasattr(shap_values, "ndim") else None
        if values_arr is not None and values_arr.ndim == 3:
            values_for_plot = values_arr[:, :, 1]
        else:
            values_for_plot = shap_values

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(values_for_plot, X_test, feature_names=feature_cols, show=False)
    plot_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print("Graphique SHAP sauvegarde dans", plot_path)

    return explainer


def explain_single_session(explainer, session_row, feature_cols, top_n: int = 3):
    """Renvoie les `top_n` facteurs qui ont le plus pousse la prediction
    vers 'attaque' pour UNE session donnee. C'est cette fonction que l'API
    reutilisera pour repondre 'pourquoi cette alerte ?'."""
    shap_values = explainer.shap_values(session_row)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values_arr = np.asarray(shap_values)
        values = values_arr[0, :, 1] if values_arr.ndim == 3 else values_arr[0]

    contributions = list(zip(feature_cols, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": name, "impact": float(impact)}
        for name, impact in contributions[:top_n]
    ]


if __name__ == "__main__":
    model = load_latest_model()
    X_train, X_test, y_train, y_test, encoders, feature_cols = prepare_dataset(
        DATA_PATH
    )

    explainer = explain_global(model, X_test, feature_cols)

    # Exemple : explication de la toute premiere session du jeu de test
    example = X_test.iloc[[0]]
    top_factors = explain_single_session(explainer, example, feature_cols)
    print("\nExemple d'explication pour une session :")
    for factor in top_factors:
        print(f"  - {factor['feature']} : impact {factor['impact']:.4f}")