from __future__ import annotations

import pandas as pd

from vassoura.preprocessing import SampleManager


def test_no_sampling_small_df():
    df = pd.DataFrame({'a': range(10)})
    sm = SampleManager(limit_mb=1, frac=0.5)
    Xt = sm.fit_transform(df)
    assert Xt.equals(df)
    assert sm.mask_.all()


def test_stratified_ratio():
    X = pd.DataFrame({'feat': range(1000)})
    y = pd.Series([0] * 700 + [1] * 300)
    sm = SampleManager(strategy='stratified', frac=0.3, random_state=0)
    Xs, ys = sm.fit_resample(X, y)
    ratio_orig = y.mean()
    ratio_new = ys.mean()
    assert abs(ratio_orig - ratio_new) <= 0.01
    assert len(Xs) == len(ys)


def test_time_series_order():
    dates = pd.date_range('2021-01-01', periods=100)
    X = pd.DataFrame({'ts': dates, 'val': range(100)})
    sm = SampleManager(strategy='time_series', frac=0.4, time_col='ts')
    Xs = sm.fit_transform(X)
    assert Xs['ts'].is_monotonic_increasing


def test_get_support_mask_shape():
    X = pd.DataFrame({'a': range(100)})
    sm = SampleManager(strategy='none')
    sm.fit(X)
    mask = sm.get_support_mask()
    assert len(mask) == len(X)


def test_errors_missing_y():
    X = pd.DataFrame({'a': range(10)})
    sm = SampleManager(strategy='stratified')
    try:
        sm.fit(X)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError for missing y'


def test_errors_missing_timecol():
    X = pd.DataFrame({'a': range(10)})
    sm = SampleManager(strategy='time_series')
    try:
        sm.fit(X)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError for missing time_col'
