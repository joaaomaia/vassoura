from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from vassoura.preprocessing import DynamicScaler


@given(st.lists(st.floats(-10, 10), min_size=5, max_size=10))
def test_inverse_roundtrip(vals):
    df = pd.DataFrame(
        {
            "a": vals,
            "b": vals,
        }
    )
    sc = DynamicScaler(strategy="numeric")
    out = sc.fit_transform(df)
    inv = sc.inverse_transform(out)
    pd.testing.assert_frame_equal(inv[df.columns], df, check_dtype=False)


def test_auto_none_for_flat():
    df = pd.DataFrame({"flat": [1] * 10})
    sc = DynamicScaler(strategy="auto")
    sc.fit(df)
    assert sc.scalers_["flat"] is None


def test_auto_select_quantile():
    df = pd.DataFrame({"x": [1, 1, 1, 100, 100, 100]})
    sc = DynamicScaler(strategy="auto", preferred="quantile", random_state=0)
    sc.fit(df)
    assert isinstance(sc.scalers_["x"], QuantileTransformer)


def test_numeric_minmax():
    df = pd.DataFrame({"a": range(5)})
    sc = DynamicScaler(strategy="numeric", preferred="minmax")
    sc.fit(df)
    assert isinstance(sc.scalers_["a"], MinMaxScaler)


def test_auto_standard_for_normal():
    df = pd.DataFrame({"x": [0, 2, -2, 4, -4]})
    sc = DynamicScaler(strategy="auto", random_state=0)
    sc.fit(df)
    assert isinstance(sc.scalers_["x"], StandardScaler)


def test_auto_minmax_branch():
    df = pd.DataFrame({"x": [0, 0, 0, 1, 2, 4]})
    sc = DynamicScaler(strategy="auto", preferred="minmax")
    sc.fit(df)
    assert isinstance(sc.scalers_["x"], MinMaxScaler)


def test_get_scaler_none_strategy():
    sc = DynamicScaler(strategy="none")
    assert sc.get_scaler(pd.Series([1, 2, 3])) is None


def test_transform_missing_columns_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    sc = DynamicScaler(strategy="numeric")
    sc.fit(df)
    with pytest.raises(ValueError):
        sc.transform(pd.DataFrame({"b": [1, 2, 3]}))
