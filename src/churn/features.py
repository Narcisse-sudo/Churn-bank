"""Feature engineering.

Fonction *pure* et picklable : elle reçoit un DataFrame brut (éventuellement avec les
colonnes d'identifiant) et renvoie les variables prêtes pour le préprocessing. Comme
elle est intégrée au pipeline scikit-learn, exactement les mêmes transformations sont
appliquées à l'entraînement, à l'évaluation et en production — pas de divergence.

Les variables créées s'appuient sur les constats de l'EDA :
- ``Balance_void``       : un solde nul est un signal fort (comptes inactifs).
- ``BalanceSalaryRatio`` : capacité d'épargne relative au revenu.
- ``MultiProduct``       : détenir 3+ produits est très corrélé au départ.
- ``TenureByAge``        : ancienneté rapportée à l'âge (fidélité relative).
"""

from __future__ import annotations

import pandas as pd

from churn import config

ENGINEERED_NUMERIC = ["BalanceSalaryRatio", "TenureByAge"]
ENGINEERED_BINARY = ["Balance_void", "MultiProduct"]

# Groupes de colonnes consommés par le préprocesseur.
CATEGORICAL_COLUMNS = config.CATEGORICAL_FEATURES
SCALED_COLUMNS = config.NUMERIC_FEATURES + ENGINEERED_NUMERIC
PASSTHROUGH_COLUMNS = config.BINARY_FEATURES + ENGINEERED_BINARY

ALL_MODEL_COLUMNS = CATEGORICAL_COLUMNS + SCALED_COLUMNS + PASSTHROUGH_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables dérivées et ne garde que les colonnes du modèle.

    Recalcule systématiquement les variables sur le DataFrame reçu : c'est ce qui
    corrige le bug de l'ancien notebook où ``Balance_void`` du *train* était réinjecté
    tel quel dans le *test* (désalignement d'index → valeurs erronées).
    """
    df = df.drop(columns=[c for c in config.ID_COLUMNS if c in df.columns], errors="ignore")
    df = df.copy()

    df["Balance_void"] = (df["Balance"] == 0).astype(int)
    df["MultiProduct"] = (df["NumOfProducts"] >= 3).astype(int)
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
    df["TenureByAge"] = df["Tenure"] / df["Age"]

    return df[ALL_MODEL_COLUMNS]
