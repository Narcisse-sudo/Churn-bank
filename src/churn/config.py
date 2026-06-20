"""Configuration centrale du projet : chemins, colonnes et constantes.

Un seul endroit pour décrire le schéma des données et la localisation des artefacts,
afin que le pipeline, l'API et l'UI partagent exactement la même définition.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent  # src/churn
PROJECT_ROOT = PACKAGE_DIR.parents[1]  # racine du dépôt

DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.getenv("CHURN_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

RAW_TRAIN_PATH = RAW_DIR / "train.csv"
RAW_TEST_PATH = RAW_DIR / "test.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"

# --- Schéma des données ------------------------------------------------------
TARGET = "Exited"

# Identifiants : sans valeur prédictive, retirés avant la modélisation.
ID_COLUMNS = ["ID", "CustomerId", "Surname"]

# Variables d'origine du dataset (hors identifiants et cible).
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]

# Colonnes minimales attendues en entrée d'une prédiction (après nettoyage des IDs).
INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Découpage des données (stratifié sur la cible).
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # part du train restant après extraction du test
