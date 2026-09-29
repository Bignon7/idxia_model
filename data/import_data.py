"""
import_data.py

Telecharge le dataset Kaggle "Cybersecurity Intrusion Detection Dataset"
et le copie dans data/raw/ sous un nom fixe, pour que le reste du pipeline
(preprocessing.py, train_model.py) puisse toujours le retrouver au meme
endroit, quelle que soit la version telechargee par kagglehub.

Prerequis : avoir un compte Kaggle et un token API configure localement.
Voir : https://github.com/Kaggle/kagglehub#authentication
(en resume : creer un token sur kaggle.com/settings puis le placer dans
~/.kaggle/kaggle.json, ou suivre le prompt interactif de kagglehub)
"""

import os
import shutil

import kagglehub

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
TARGET_CSV_NAME = "cybersecurity_intrusion_data.csv"


def download_dataset() -> str:
    """Telecharge la derniere version du dataset et renvoie le chemin local."""
    path = kagglehub.dataset_download(
        "dnkumars/cybersecurity-intrusion-detection-dataset"
    )
    print("Dataset telecharge dans :", path)
    return path


def copy_to_raw(source_dir: str) -> str:
    """Copie le fichier CSV trouve dans source_dir vers data/raw/ avec un nom fixe."""
    os.makedirs(RAW_DIR, exist_ok=True)

    csv_found = None
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_found = os.path.join(root, f)
                break
        if csv_found:
            break

    if not csv_found:
        raise FileNotFoundError(
            "Aucun fichier CSV trouve dans le dataset telecharge par kagglehub."
        )

    target_path = os.path.join(RAW_DIR, TARGET_CSV_NAME)
    shutil.copy(csv_found, target_path)
    print("Fichier copie vers :", target_path)
    return target_path


if __name__ == "__main__":
    dataset_path = download_dataset()
    final_path = copy_to_raw(dataset_path)
    print("\nDataset pret a l'emploi :", final_path)