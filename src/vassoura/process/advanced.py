from __future__ import annotations

import numpy as np
import pandas as pd

from .heuristic_boruta_multi_shap import boruta_multi_shap


def advanced_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 3,
    top_k: int | None = None,
    sample_weight: np.ndarray | None = None,
    random_state: int | None = 42,
) -> pd.Series:
    """Heavy-duty multi-run SHAP importance.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    n_trials : int, default 3
        Number of repetitions.
    top_k : int | None, optional
        Return only top k features.
    sample_weight : numpy.ndarray | None, optional
        Sample weights.
    random_state : int | None, optional
        Random seed.
    """
    return boruta_multi_shap(
        X,
        y,
        n_trials=n_trials,
        top_k=top_k,
        sample_weight=sample_weight,
        random_state=random_state,
    )
