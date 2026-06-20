"""Entraînement bout-en-bout.

Étapes : chargement -> split stratifié train/val/test -> comparaison des modèles sur la
validation -> sélection du meilleur (PR-AUC) -> réglage du seuil de décision (F1) ->
ré-entraînement du gagnant sur train+val -> évaluation honnête sur le test held-out ->
sauvegarde des artefacts (modèle, métriques, seuil, importances).

Usage :
    python -m churn.train
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from churn import config, data, model


def _feature_importance(pipeline) -> dict[str, float]:
    """Extrait l'importance des variables après préprocessing, si le modèle l'expose."""
    try:
        names = pipeline.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
    except Exception:
        return {}
    clf = pipeline.named_steps["model"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = abs(clf.coef_[0])
    else:
        return {}
    return {str(n): float(v) for n, v in zip(names, values, strict=False)}


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = data.load_labeled()
    splits = data.split_data(df)
    print(
        f"Données : train={len(splits.X_train)} | val={len(splits.X_val)} | "
        f"test={len(splits.X_test)} (churn={df[config.TARGET].mean():.1%})"
    )

    # 1) Comparaison des candidats sur la validation.
    results = {}
    fitted = {}
    for name, estimator in model.candidate_models(splits.y_train).items():
        pipe = model.build_pipeline(estimator)
        pipe.fit(splits.X_train, splits.y_train)
        proba_val = pipe.predict_proba(splits.X_val)[:, 1]
        metrics = model.compute_metrics(splits.y_val, proba_val, threshold=0.5)
        results[name] = metrics
        fitted[name] = pipe
        print(f"  {name:20s} PR-AUC={metrics['pr_auc']:.3f} F1@0.5={metrics['f1']:.3f}")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    print(f"Meilleur modèle : {best_name}")

    # 2) Seuil optimal (F1) sur la validation.
    proba_val = fitted[best_name].predict_proba(splits.X_val)[:, 1]
    threshold = model.best_threshold(splits.y_val, proba_val)

    # 3) Ré-entraînement du gagnant sur train+val pour plus de données.
    X_trainval = pd.concat([splits.X_train, splits.X_val])
    y_trainval = pd.concat([splits.y_train, splits.y_val])
    final_pipe = model.build_pipeline(model.candidate_models(y_trainval)[best_name])
    final_pipe.fit(X_trainval, y_trainval)

    # 4) Évaluation honnête sur le test held-out.
    proba_test = final_pipe.predict_proba(splits.X_test)[:, 1]
    test_metrics = model.compute_metrics(splits.y_test, proba_test, threshold)
    print(
        f"Test held-out @seuil={threshold:.3f} : "
        + " ".join(f"{k}={v:.3f}" for k, v in test_metrics.items())
    )

    # 5) Sauvegarde des artefacts.
    joblib.dump(final_pipe, config.MODEL_PATH)
    config.THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))
    config.METRICS_PATH.write_text(
        json.dumps(
            {
                "model": best_name,
                "validation": results,
                "test": test_metrics,
                "n_train": len(splits.X_train),
                "n_val": len(splits.X_val),
                "n_test": len(splits.X_test),
                "churn_rate": float(df[config.TARGET].mean()),
            },
            indent=2,
        )
    )
    importances = _feature_importance(final_pipe)
    if importances:
        config.FEATURE_IMPORTANCE_PATH.write_text(json.dumps(importances, indent=2))

    print(f"Artefacts écrits dans {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
