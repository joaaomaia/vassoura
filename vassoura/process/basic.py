from __future__ import annotations

import numpy as np
import pandas as pd


from vassoura.logs import get_logger
from vassoura.models.utils import supports_sample_weight
from vassoura.utils.weights import make_balanced_sample_weights

logger = get_logger(__name__)


def _make_model(name: str, random_state: int | None):
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=200, n_jobs=-1, random_state=random_state)
    if name == "lgb":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgb":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model '{name}'")


def basic_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model: str = "logistic",
    method: str = "coef",
    top_k: int | None = None,
    sample_weight: np.ndarray | None = None,
    random_state: int | None = 42,
) -> pd.Series:
    """Quick feature importance using a lightweight model.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    model : {{"logistic", "lgb", "xgb"}}, default "logistic"
        Base model to fit.
    method : {{"coef", "gain", "shap"}}, default "coef"
        Importance type.
    top_k : int | None, optional
        Return only the top k features.
    sample_weight : numpy.ndarray | None, optional
        Sample weights array.
    random_state : int | None, optional
        Random seed.

    Returns
    -------
    pandas.Series
        Importance ordered by absolute value.
    """
    if sample_weight is None:
        sample_weight = make_balanced_sample_weights(y)

    if method == "coef" and model in {"lgb", "xgb"}:
        method = "gain"

    rng = np.random.default_rng(random_state)
    Xw = X.copy()
    Xw["__noise_uniform__"] = rng.uniform(0, 1, size=len(Xw))

    est = _make_model(model, random_state)
    if supports_sample_weight(est):
        est.fit(Xw, y, sample_weight=sample_weight)
    else:
        if hasattr(est, "get_params") and "class_weight" in est.get_params():
            est.set_params(class_weight="balanced")
        est.fit(Xw, y)

    if method in {"coef", "gain"}:
        if model == "logistic":
            coef = getattr(est, "coef_", np.zeros(Xw.shape[1]))
            imp = pd.Series(coef.ravel(), index=Xw.columns)
        elif model == "lgb":
            booster = est.booster_
            vals = booster.feature_importance(importance_type="gain")
            imp = pd.Series(vals, index=est.feature_name_)
        else:
            booster = est.get_booster()
            score = booster.get_score(importance_type="gain")
            imp = pd.Series({k: v for k, v in score.items()})
            imp = imp.reindex(Xw.columns, fill_value=0)
    elif method == "shap":
        import shap

        if model == "logistic":
            masker = shap.maskers.Independent(Xw)
            expl = shap.LinearExplainer(est, masker=masker)
        else:
            expl = shap.TreeExplainer(est)
        sv = expl.shap_values(Xw)
        if isinstance(sv, list):
            sv = np.stack(sv).sum(axis=0)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=Xw.columns)
    else:
        raise ValueError("method must be 'coef', 'gain' or 'shap'")

    imp = imp.reindex(Xw.columns).fillna(0)
    imp = imp.reindex(Xw.columns)
    imp = imp.abs().sort_values(ascending=False)
    if top_k is not None:
        imp = imp.iloc[:top_k]
    return imp
