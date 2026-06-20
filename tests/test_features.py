"""Tests du feature engineering, dont la régression sur le bug ``Balance_void``."""

from __future__ import annotations

import pandas as pd

from churn import config, features


def test_drops_id_columns_and_target_kept_out(sample_raw):
    out = features.engineer_features(sample_raw.drop(columns=[config.TARGET]))
    assert not set(config.ID_COLUMNS) & set(out.columns)
    assert set(out.columns) == set(features.ALL_MODEL_COLUMNS)


def test_balance_void_recomputed_per_frame():
    # Régression : Balance_void doit refléter le DataFrame reçu, pas un autre.
    df = pd.DataFrame(
        {
            "CreditScore": [600, 700],
            "Geography": ["France", "Spain"],
            "Gender": ["Male", "Female"],
            "Age": [30.0, 50.0],
            "Tenure": [2, 8],
            "Balance": [0.0, 120000.0],
            "NumOfProducts": [1, 3],
            "HasCrCard": [1.0, 0.0],
            "IsActiveMember": [1.0, 0.0],
            "EstimatedSalary": [50000.0, 90000.0],
        }
    )
    out = features.engineer_features(df)
    assert out["Balance_void"].tolist() == [1, 0]
    assert out["MultiProduct"].tolist() == [0, 1]


def test_no_nan_introduced(sample_raw):
    out = features.engineer_features(sample_raw)
    assert not out.isna().any().any()
