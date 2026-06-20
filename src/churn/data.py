"""Chargement et découpage des données.

Le fichier ``train.csv`` est le seul jeu étiqueté (la cible ``Exited`` est présente).
On en extrait des ensembles d'entraînement / validation / test *stratifiés* pour
évaluer le modèle de façon honnête. ``test.csv`` n'est pas étiqueté : il sert
uniquement à générer une soumission (voir ``predict.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


@dataclass
class DataSplits:
    """Conteneur des différents jeux après découpage stratifié."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_labeled(path=None) -> pd.DataFrame:
    """Charge le jeu étiqueté et vérifie la présence de la cible."""
    path = path or config.RAW_TRAIN_PATH
    df = pd.read_csv(path)
    if config.TARGET not in df.columns:
        raise ValueError(f"Colonne cible '{config.TARGET}' absente de {path}")
    return df


def load_unlabeled(path=None) -> pd.DataFrame:
    """Charge le jeu de soumission (sans cible)."""
    path = path or config.RAW_TEST_PATH
    return pd.read_csv(path)


def split_data(df: pd.DataFrame) -> DataSplits:
    """Découpe en train/val/test stratifiés sur ``Exited`` (anti-fuite, reproductible).

    L'ancienne version utilisait ``test_size=0.8`` (80 % en test, 20 % en train) et
    n'était pas stratifiée : deux défauts corrigés ici.
    """
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=config.VAL_SIZE,
        stratify=y_trainval,
        random_state=config.RANDOM_STATE,
    )
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
