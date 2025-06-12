from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vassoura.process import basic_importance, advanced_importance


def test_noise_uniform_present():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    imp = basic_importance(X, y)
    assert "__noise_uniform__" in imp.index


def test_method_switch_gain():
    pytest.importorskip("xgboost")
    X = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
    y = pd.Series([0, 1] * 10)
    imp_gain = basic_importance(X, y, model="xgb", method="gain", random_state=0)
    imp_coef = basic_importance(X, y, model="xgb", method="coef", random_state=0)
    pd.testing.assert_series_equal(imp_gain, imp_coef)


def test_weight_effect():
    X = pd.DataFrame(np.random.randn(30, 2), columns=["a", "b"])
    y = pd.Series([0] * 15 + [1] * 15)
    w1 = np.where(y == 1, 1, 2)
    w2 = np.where(y == 1, 2, 1)
    imp1 = basic_importance(X, y, sample_weight=w1, random_state=0)
    imp2 = basic_importance(X, y, sample_weight=w2, random_state=0)
    assert not imp1.equals(imp2)


def test_top_k():
    X = pd.DataFrame(np.random.randn(50, 5), columns=list("abcde"))
    y = pd.Series(np.random.randint(0, 2, size=50))
    imp = basic_importance(X, y, top_k=3)
    assert len(imp) == 3


def test_advanced_runs_subset():
    pytest.importorskip("shap")
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, size=100))
    imp = advanced_importance(X, y, n_trials=2, top_k=5, random_state=0)
    assert len(imp) == 5


def test_weights_consistency():
    pytest.importorskip("shap")
    X = pd.DataFrame(np.random.randn(100, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.random.randint(0, 2, size=100))
    imp1 = advanced_importance(X, y, n_trials=1, random_state=0)
    w = np.where(y == 1, 2, 1)
    imp2 = advanced_importance(X, y, n_trials=1, sample_weight=w, random_state=0)
    assert not imp1.equals(imp2)


def test_advanced_performance():
    pytest.importorskip("shap")
    X = pd.DataFrame(np.random.randn(1000, 100), columns=[f"f{i}" for i in range(100)])
    y = pd.Series(np.random.randint(0, 2, size=1000))
    advanced_importance(X, y, n_trials=1, top_k=10, random_state=0)
