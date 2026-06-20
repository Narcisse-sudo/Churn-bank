"""Tests d'inférence (nécessitent le modèle entraîné)."""

from __future__ import annotations

import pandas as pd

from churn import predict as predict_mod


def test_predict_proba_in_range(require_model, customer_payload):
    proba = predict_mod.predict_proba(pd.DataFrame([customer_payload]))
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0


def test_predict_returns_decision(require_model, customer_payload):
    out = predict_mod.predict(pd.DataFrame([customer_payload]))
    assert set(out.columns) == {"probability", "churn"}
    assert out["churn"].iloc[0] in (0, 1)


def test_threshold_loaded():
    t = predict_mod.load_threshold()
    assert 0.0 < t < 1.0
