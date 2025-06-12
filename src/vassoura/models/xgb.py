from __future__ import annotations

import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

from vassoura.logs import get_logger

from .base import WrapperBase
from .utils import make_sample_weights, supports_sample_weight


class XGBoostWrapper(WrapperBase, BaseEstimator, ClassifierMixin):
    """XGBoost classifier wrapper with balanced weights."""

    name = "xgboost_balanced"
    _estimator_type = "classifier"

    def __init__(self, **params) -> None:
        defaults = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.05,
            n_estimators=400,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
        defaults.update(params)
        self.model = xgb.XGBClassifier(**defaults)
        self.logger = get_logger("ModelWrapper")

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = make_sample_weights(y)
        sw_used = supports_sample_weight(self.model)
        if sw_used:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            if "class_weight" in self.model.get_params():
                self.model.set_params(class_weight="balanced")
            self.model.fit(X, y)
        self.logger.debug(
            "[ModelWrapper] model='%s' | params=%d | sample_weight=%s",
            self.name,
            len(self.get_params()),
            "used" if sw_used else "fallback",
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)
