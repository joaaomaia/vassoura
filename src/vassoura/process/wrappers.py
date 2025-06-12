from __future__ import annotations

from typing import Callable, Optional  # noqa: F401

import numpy as np
import pandas as pd

from .basic import basic_importance
from .medium import medium_importance

advanced_importance: Optional[Callable]
try:
    from .advanced import advanced_importance as _advanced_importance

    advanced_importance = _advanced_importance
except Exception:  # optional dependency "shap"
    advanced_importance = None


class BasicHeuristic:
    """Wrapper around :func:`basic_importance`."""

    def __init__(self, model=None):
        self.model = model

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        random_state: int | None = 42,
    ) -> pd.Series:
        high_card_limit = 50
        cat_cols = X.select_dtypes("object").columns
        safe_cols = [
            c for c in cat_cols if X[c].nunique(dropna=False) <= high_card_limit
        ]

        from sklearn.preprocessing import OrdinalEncoder

        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_ordinal = pd.DataFrame(
            oe.fit_transform(X[safe_cols]),
            columns=[f"ord_{c}" for c in safe_cols],
            index=X.index,
        )
        X_num = X.drop(columns=cat_cols)
        X_enc = pd.concat([X_num, X_ordinal], axis=1)

        return basic_importance(
            X_enc,
            y,
            model="logistic",
            method="coef",
            sample_weight=sample_weight,
            random_state=random_state,
        )


class MediumHeuristic:
    """Wrapper around :func:`medium_importance`."""

    def __init__(self, model=None):
        self.model = model

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        random_state: int | None = 42,
    ) -> pd.Series:
        return medium_importance(
            X,
            y,
            sample_weight=sample_weight,
            random_state=random_state,
        )


class AdvancedHeuristic:
    """Wrapper around :func:`advanced_importance`."""

    def __init__(self, model=None):
        self.model = model

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        random_state: int | None = 42,
    ) -> pd.Series:
        if advanced_importance is None:
            raise ImportError("advanced_importance requires 'shap' installed")
        return advanced_importance(
            X,
            y,
            sample_weight=sample_weight,
            random_state=random_state,
        )


def import_heuristic(name: str):
    mapping = {
        "basic": BasicHeuristic,
        "medium": MediumHeuristic,
        "advanced": AdvancedHeuristic,
    }
    if name not in mapping:
        raise ValueError(f"Unknown heuristic '{name}'")
    return mapping[name]
