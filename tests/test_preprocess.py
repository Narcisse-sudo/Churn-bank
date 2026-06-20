"""Tests du préprocesseur : fit sur train uniquement, sortie numérique sans NaN."""

from __future__ import annotations

import numpy as np

from churn import config
from churn.preprocess import build_preprocessor


def test_fit_transform_shapes(sample_raw):
    X = sample_raw.drop(columns=[config.TARGET])
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.shape[0] == len(X)
    # Toutes les sorties doivent être numériques (One-Hot + scale + passthrough).
    assert np.issubdtype(np.asarray(out).dtype, np.number)
    assert not np.isnan(np.asarray(out)).any()


def test_handles_unknown_category(sample_raw):
    X = sample_raw.drop(columns=[config.TARGET])
    pre = build_preprocessor()
    pre.fit(X)
    X2 = X.copy()
    X2.loc[X2.index[0], "Geography"] = "Italy"  # catégorie inédite
    out = pre.transform(X2)  # ne doit pas lever (handle_unknown="ignore")
    assert out.shape[0] == len(X2)
