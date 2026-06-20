"""Modèles candidats et métriques.

Trois familles comparées, toutes avec gestion du déséquilibre de classes (≈ 21 % de
churn) : régression logistique (baseline interprétable), forêt aléatoire et XGBoost.
On évalue avec PR-AUC / ROC-AUC / F1 plutôt que l'accuracy, peu informative quand les
classes sont déséquilibrées.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.preprocess import build_preprocessor


def candidate_models(y_train) -> dict[str, object]:
    """Instancie les classifieurs candidats, pondérés contre le déséquilibre."""
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1.0)

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    """Assemble préprocesseur + classifieur en un pipeline bout-en-bout (df brut -> proba)."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def compute_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    """Métriques de classification au seuil donné (PR-AUC/ROC-AUC indépendants du seuil)."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def best_threshold(y_true, proba) -> float:
    """Cherche le seuil qui maximise le F1 sur un balayage fin de [0.05, 0.95]."""
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (np.asarray(proba) >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])
