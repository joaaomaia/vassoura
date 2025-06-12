from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from vassoura.models.utils import supports_sample_weight
from vassoura.utils.weights import make_balanced_sample_weights


def medium_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_splits: int = 5,
    n_repeats: int = 5,
    top_k: int | None = None,
    sample_weight: np.ndarray | None = None,
    random_state: int | None = 42,
) -> pd.Series:
    """Permutation importance with cross-validation.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    cv_splits : int, default 5
        Number of folds.
    n_repeats : int, default 5
        Permutation repeats.
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

    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=random_state)
    base_model = LogisticRegression(max_iter=200, n_jobs=-1, random_state=random_state)

    scores = pd.Series(0.0, index=Xw.columns)

    for train_idx, test_idx in cv.split(Xw, y):
        Xtr, ytr = Xw.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = Xw.iloc[test_idx], y.iloc[test_idx]
        sw_train = sample_weight[train_idx]
        sw_test = sample_weight[test_idx]
        model = clone(base_model)
        if supports_sample_weight(model):
            model.fit(Xtr, ytr, sample_weight=sw_train)
        else:
            if hasattr(model, "get_params") and "class_weight" in model.get_params():
                model.set_params(class_weight="balanced")
            model.fit(Xtr, ytr)

        scorer = make_scorer(roc_auc_score, needs_proba=True, sample_weight=sw_test)
        result = permutation_importance(
            model,
            Xte,
            yte,
            scoring=scorer,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        scores = scores.add(pd.Series(result.importances_mean, index=Xw.columns), fill_value=0)

    scores /= cv_splits
    scores = scores.abs().sort_values(ascending=False)
    if top_k is not None:
        scores = scores.iloc[:top_k]
    return scores
