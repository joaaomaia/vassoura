from __future__ import annotations

import pandas as pd

from vassoura.validation import get_stratified_cv, get_time_series_cv
from vassoura.validation.cv import train_test_split_by_date


def test_stratified_cv_preserves_ratio():
    X = pd.DataFrame({"a": range(200)})
    y = pd.Series([0] * 150 + [1] * 50)
    cv = get_stratified_cv(n_splits=4, shuffle=True, random_state=0)
    overall = y.mean()
    for train_idx, test_idx in cv.split(X, y):
        train_ratio = y.iloc[train_idx].mean()
        test_ratio = y.iloc[test_idx].mean()
        assert abs(train_ratio - overall) <= 0.01 + 1e-8
        assert abs(test_ratio - overall) <= 0.01 + 1e-8


def test_time_series_gap_enforced():
    X = pd.DataFrame({"val": range(20)})
    cv = get_time_series_cv(n_splits=4, gap=2)
    for train_idx, test_idx in cv.split(X):
        assert train_idx.max() + 2 < test_idx.min()


def test_split_by_date():
    dates = pd.date_range("2022-01-01", periods=10)
    df = pd.DataFrame({"ts": dates, "val": range(10)})
    train, test = train_test_split_by_date(df, "ts", "2022-01-05")
    assert train["ts"].max() <= pd.Timestamp("2022-01-05")
    assert test["ts"].min() > pd.Timestamp("2022-01-05")
