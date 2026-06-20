"""API FastAPI de scoring du churn.

Endpoints :
- GET  /            : index des routes
- GET  /health      : statut + modèle chargé
- GET  /model-info  : métriques (test), seuil, top features
- POST /predict     : score d'un client
- POST /predict_batch : score d'un lot (JSON) ou d'un CSV uploadé

Le modèle est un pipeline bout-en-bout : il reçoit les variables brutes et applique
lui-même le feature engineering, le préprocessing puis XGBoost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

# Rendre le package `churn` importable (src/ layout) sans installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from churn import config  # noqa: E402
from churn import predict as predict_mod  # noqa: E402

app = FastAPI(
    title="Churn Bank API",
    version="1.0.0",
    description="Prédiction de l'attrition client bancaire (XGBoost).",
)


class Customer(BaseModel):
    """Profil client attendu en entrée (variables brutes, avant feature engineering)."""

    model_config = ConfigDict(extra="forbid")

    CreditScore: int = Field(..., ge=300, le=900, examples=[650])
    Geography: str = Field(..., examples=["France"])
    Gender: str = Field(..., examples=["Female"])
    Age: float = Field(..., ge=18, le=120, examples=[42])
    Tenure: int = Field(..., ge=0, le=10, examples=[5])
    Balance: float = Field(..., ge=0, examples=[120000.0])
    NumOfProducts: int = Field(..., ge=1, le=4, examples=[1])
    HasCrCard: int = Field(..., ge=0, le=1, examples=[1])
    IsActiveMember: int = Field(..., ge=0, le=1, examples=[0])
    EstimatedSalary: float = Field(..., ge=0, examples=[100000.0])


class PredictionResponse(BaseModel):
    probability: float
    churn: bool
    threshold: float


def _score(df: pd.DataFrame) -> list[float]:
    missing = [c for c in config.INPUT_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Colonnes manquantes : {missing}")
    try:
        return predict_mod.predict_proba(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def root() -> dict:
    return {
        "message": "Churn Bank API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
        "predict_batch": "/predict_batch",
        "predict_batch_csv": "/predict_batch_csv",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.get("/model-info")
def model_info() -> dict:
    metrics = json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}
    payload = {
        "model_loaded": config.MODEL_PATH.exists(),
        "threshold": predict_mod.load_threshold(),
        "metrics": metrics,
    }
    if config.FEATURE_IMPORTANCE_PATH.exists():
        fi = json.loads(config.FEATURE_IMPORTANCE_PATH.read_text())
        payload["top_features"] = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    return payload


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer) -> PredictionResponse:
    proba = _score(pd.DataFrame([customer.model_dump()]))[0]
    threshold = predict_mod.load_threshold()
    return PredictionResponse(probability=proba, churn=proba >= threshold, threshold=threshold)


def _batch_response(df: pd.DataFrame) -> dict:
    proba = _score(df)
    threshold = predict_mod.load_threshold()
    results = [{"probability": p, "churn": p >= threshold} for p in proba]
    return {"count": len(results), "threshold": threshold, "results": results}


@app.post("/predict_batch")
def predict_batch(customers: list[Customer]) -> dict:
    """Score un lot de clients fournis en JSON."""
    return _batch_response(pd.DataFrame([c.model_dump() for c in customers]))


@app.post("/predict_batch_csv")
def predict_batch_csv(file: UploadFile = File(...)) -> dict:
    """Score un lot de clients à partir d'un CSV uploadé."""
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001 - renvoyer l'erreur de parsing au client
        raise HTTPException(status_code=400, detail=f"CSV invalide : {exc}") from exc
    return _batch_response(df)
