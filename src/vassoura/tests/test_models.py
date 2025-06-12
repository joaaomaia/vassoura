from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from vassoura.models import get, list_models


@pytest.mark.parametrize("name", list_models())
def test_wrapper_fit_predict(name):
    Model = get(name)
    if "xgboost" in name:
        pytest.importorskip("xgboost")
    if "lightgbm" in name:
        pytest.importorskip("lightgbm")
    X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = Model()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray) or isinstance(preds, pd.Series)
    assert len(preds) == len(X)


def test_sample_weight_priority():
    Model = get("logistic_balanced")
    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 0, 0, 1])
    called = {}
    orig_fit = LogisticRegression.fit

    def spy(self, X, y, sample_weight=None):
        called["sw"] = sample_weight
        return orig_fit(self, X, y, sample_weight=sample_weight)

    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setattr(LogisticRegression, "fit", spy, raising=False)
        m = Model()
        m.fit(X, y, sample_weight=None)

    assert called["sw"] is not None
    assert not np.allclose(called["sw"], 1.0)


def test_class_weight_fallback(monkeypatch):
    class Dummy:
        def __init__(self, **kw):
            self.params = kw

        def get_params(self, deep=True):
            return self.params

        def set_params(self, **kw):
            self.params.update(kw)

        def fit(self, X, y):
            self.fitted = True

    from vassoura.models.lr import LogisticRegressionWrapper

    wrapper = LogisticRegressionWrapper()
    dummy = Dummy(class_weight=None)
    wrapper.model = dummy
    wrapper.fit(pd.DataFrame({"a": [0, 1]}), pd.Series([0, 1]))
    assert dummy.params["class_weight"] == "balanced"


def test_registry_lookup():
    Model = get("logistic_balanced")
    from vassoura.models.lr import LogisticRegressionWrapper

    assert Model is LogisticRegressionWrapper


def test_wrappers_classifier_flag():
    for name in list_models():
        Model = get(name)
        if "xgboost" in name:
            pytest.importorskip("xgboost")
        if "lightgbm" in name:
            pytest.importorskip("lightgbm")
        est = Model()
        assert getattr(est, "_estimator_type", None) == "classifier"
