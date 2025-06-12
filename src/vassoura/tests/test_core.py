from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vassoura import Vassoura


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": range(10),
            "cat": ["a", "b"] * 5,
            "target": [0, 1] * 5,
        }
    )


def test_fit_runs_end_to_end_small_df():
    df = _make_df()
    v = Vassoura(target_col="target", random_state=0, verbose=0)
    v.fit(df)
    assert hasattr(v, "model_")


def test_ranking_not_empty():
    df = _make_df()
    v = Vassoura(target_col="target", random_state=0, verbose=0)
    v.fit(df)
    ranking = v.get_feature_ranking()
    assert len(ranking) > 0


def test_metrics_contains_auc():
    df = _make_df()
    v = Vassoura(target_col="target", random_state=0, verbose=0)
    v.fit(df)
    assert any("auc" in k for k in v.metrics_.keys())


def test_predict_after_fit():
    df = _make_df()
    v = Vassoura(target_col="target", random_state=0, verbose=0)
    v.fit(df)
    preds = v.predict(df.drop(columns=["target"]))
    assert len(preds) == len(df)


def test_sample_weight_fallback(monkeypatch, caplog):
    class DummyEstimator:
        name = "dummy"

        def __init__(self, **kw):
            pass

        def fit(self, X, y):
            self.fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.vstack([1 - self.predict(X), self.predict(X)]).T

        def get_params(self, deep=True):
            return {}

    from vassoura.models import registry

    monkeypatch.setitem(registry._registry, "dummy", DummyEstimator)

    df = _make_df()
    v = Vassoura(target_col="target", model_name="dummy", random_state=0, verbose=0)
    with caplog.at_level("INFO"):
        v.fit(df)
    msgs = [r.message for r in caplog.records]
    assert any("falling back" in m for m in msgs)


def test_datetime_column_handled():
    df = _make_df()
    df["date"] = pd.date_range("2021-01-01", periods=len(df))
    v = Vassoura(
        target_col="target",
        date_cols=["date"],
        random_state=0,
        verbose=0,
    )
    v.fit(df)
    assert hasattr(v, "model_")


def test_keep_cols_in_ranking():
    df = _make_df()
    v = Vassoura(target_col="target", keep_cols=["cat"], random_state=0, verbose=0)
    v.fit(df)
    ranking = v.get_feature_ranking()
    assert "cat" in ranking.index
