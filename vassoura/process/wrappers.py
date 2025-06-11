from __future__ import annotations

import pandas as pd
import numpy as np

from .basic import basic_importance
from .medium import medium_importance
from .advanced import advanced_importance


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
        X_enc = pd.get_dummies(X, drop_first=False)
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
