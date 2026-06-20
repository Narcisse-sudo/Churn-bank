"""Tests de l'API FastAPI via TestClient."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(require_model, customer_payload):
    r = client.post("/predict", json=customer_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["churn"], bool)


def test_predict_rejects_unknown_field(customer_payload):
    r = client.post("/predict", json={**customer_payload, "Foo": 1})
    assert r.status_code == 422  # extra="forbid"


def test_predict_validates_range(customer_payload):
    r = client.post("/predict", json={**customer_payload, "Age": 5})
    assert r.status_code == 422  # Age >= 18


def test_model_info(require_model):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_batch_json(require_model, customer_payload):
    r = client.post("/predict_batch", json=[customer_payload, customer_payload])
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_predict_batch_csv(require_model, tmp_path, customer_payload):
    csv = tmp_path / "batch.csv"
    pd.DataFrame([customer_payload]).to_csv(csv, index=False)
    with open(csv, "rb") as fh:
        r = client.post("/predict_batch_csv", files={"file": ("batch.csv", fh, "text/csv")})
    assert r.status_code == 200
    assert r.json()["count"] == 1
