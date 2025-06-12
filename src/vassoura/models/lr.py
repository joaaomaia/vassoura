from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from vassoura.logs import get_logger

from .base import WrapperBase
from .utils import make_sample_weights, supports_sample_weight


class LogisticRegressionWrapper(WrapperBase, BaseEstimator, ClassifierMixin):
    """Logistic Regression with balanced sampling."""

    name = "logistic_balanced"
    _estimator_type = "classifier"

    def __init__(self, **params) -> None:
        defaults = dict(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=400,
            n_jobs=-1,
        )
        defaults.update(params)
        self.model = LogisticRegression(**defaults)
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

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict(self, X):
        return self.model.predict(X)
