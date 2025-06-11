from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from vassoura.logs import get_logger
from vassoura.models.utils import supports_sample_weight
from vassoura.utils.weights import make_balanced_sample_weights

logger = get_logger(__name__)


def boruta_multi_shap(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 3,
    top_k: int | None = None,
    sample_weight: np.ndarray | None = None,
    random_state: int | None = None,
) -> pd.Series:
    """Compute repeated SHAP importances using RandomForest.

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
    if sample_weight is None:
        sample_weight = make_balanced_sample_weights(y)

    rng = np.random.default_rng(random_state)
    Xw = X.copy()
    Xw["__noise_uniform__"] = rng.uniform(0, 1, size=len(Xw))

    importances = pd.Series(0.0, index=Xw.columns)

    for i in range(n_trials):
        est = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=rng.integers(0, 1_000_000),
            n_jobs=-1,
        )
        if supports_sample_weight(est):
            est.fit(Xw, y, sample_weight=sample_weight)
        else:
            est.fit(Xw, y)

        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(Xw)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        if sv.ndim == 3:
            sv = sv.sum(axis=-1)
        vals = np.abs(sv).mean(axis=0)
        importances = importances.add(pd.Series(vals, index=Xw.columns), fill_value=0)
        logger.info(
            "[Advanced] trial %d/%d finished – kept=%d features", i + 1, n_trials, len(Xw.columns)
        )

    importances /= n_trials
    importances = importances.abs().sort_values(ascending=False)
    if top_k is not None:
        importances = importances.iloc[:top_k]
    return importances
