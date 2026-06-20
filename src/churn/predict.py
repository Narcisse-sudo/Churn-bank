"""Inférence : charge les artefacts et prédit le churn sur de nouvelles données.

Réutilisé par l'API et l'UI. Expose aussi un point d'entrée pour régénérer la
soumission Kaggle à partir de ``data/raw/test.csv``.

Usage :
    python -m churn.predict            # écrit artifacts/submission.csv
"""

from __future__ import annotations

import functools
import json

import joblib
import pandas as pd

from churn import config, data


@functools.lru_cache(maxsize=1)
def load_model():
    """Charge le pipeline entraîné (mis en cache). Lève une erreur claire si absent."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable ({config.MODEL_PATH}). Lance d'abord `python -m churn.train`."
        )
    return joblib.load(config.MODEL_PATH)


def load_threshold() -> float:
    """Seuil de décision optimisé à l'entraînement (0.5 par défaut)."""
    if config.THRESHOLD_PATH.exists():
        return float(json.loads(config.THRESHOLD_PATH.read_text()).get("threshold", 0.5))
    return 0.5


def predict_proba(df: pd.DataFrame) -> list[float]:
    """Probabilité de churn pour chaque ligne du DataFrame."""
    model = load_model()
    proba = model.predict_proba(df)[:, 1]
    return [float(p) for p in proba]


def predict(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Renvoie un DataFrame avec ``probability`` et ``churn`` (0/1) au seuil donné."""
    threshold = load_threshold() if threshold is None else threshold
    proba = predict_proba(df)
    return pd.DataFrame({"probability": proba, "churn": [int(p >= threshold) for p in proba]})


def make_submission(output_path=None) -> None:
    """Régénère la soumission Kaggle (ID, Exited) à partir de test.csv."""
    test = data.load_unlabeled()
    preds = predict(test)
    output_path = output_path or (config.ARTIFACTS_DIR / "submission.csv")
    pd.DataFrame({"ID": test["ID"], "Exited": preds["churn"]}).to_csv(output_path, index=False)
    print(f"Soumission écrite : {output_path} ({len(preds)} lignes)")


if __name__ == "__main__":
    make_submission()
