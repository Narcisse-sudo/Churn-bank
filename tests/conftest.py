"""Fixtures partagées : petit jeu synthétique et garde-fou sur le modèle entraîné."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config


@pytest.fixture
def sample_raw() -> pd.DataFrame:
    """Mini jeu de données conforme au schéma (avec colonnes d'identifiant et cible)."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "ID": range(n),
            "CustomerId": rng.integers(1e7, 2e7, n),
            "Surname": ["Doe"] * n,
            "CreditScore": rng.integers(350, 850, n),
            "Geography": rng.choice(["France", "Germany", "Spain"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 92, n).astype(float),
            "Tenure": rng.integers(0, 10, n),
            "Balance": rng.choice([0.0, 120000.0, 95000.0], n),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n).astype(float),
            "IsActiveMember": rng.integers(0, 2, n).astype(float),
            "EstimatedSalary": rng.uniform(10, 200000, n),
            "Exited": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def customer_payload() -> dict:
    """Profil client unique valide pour l'API."""
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 5,
        "Balance": 120000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 100000.0,
    }


@pytest.fixture
def require_model():
    """Saute le test si le modèle n'a pas encore été entraîné."""
    if not config.MODEL_PATH.exists():
        pytest.skip("Modèle absent — lancer `python -m churn.train` d'abord.")
