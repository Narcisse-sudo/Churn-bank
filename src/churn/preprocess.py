"""Construction du préprocesseur scikit-learn.

Pipeline = feature engineering -> ColumnTransformer :
- One-Hot sur les variables catégorielles (Geography, Gender) ;
- StandardScaler sur les variables numériques (originales + dérivées) ;
- passthrough sur les variables déjà binaires (0/1).

Le préprocesseur est *fit* uniquement sur le train (aucune fuite). Aucune sélection
de variables (l'ancien ``SelectKBest(k=6)`` retirait arbitrairement des features utiles
que les modèles à base d'arbres exploitent très bien).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import features


def build_preprocessor() -> Pipeline:
    """Retourne un pipeline non entraîné : feature engineering + transformation colonnes."""
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                features.CATEGORICAL_COLUMNS,
            ),
            ("scale", StandardScaler(), features.SCALED_COLUMNS),
            ("passthrough", "passthrough", features.PASSTHROUGH_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "engineer",
                FunctionTransformer(features.engineer_features, validate=False),
            ),
            ("columns", column_transformer),
        ]
    )
