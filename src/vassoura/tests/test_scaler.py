from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from vassoura.preprocessing import DynamicScaler


def _make_df() -> pd.DataFrame:
    data = {
        "normal": pd.Series([1, 2, 3, 4, 5], dtype=float),
        "skewed": pd.Series([1, 1, 1, 1, 100], dtype=float),
        "category": pd.Series(["a", "b", "a", "b", "c"]),
        "bin_col": pd.Series([0, 1, 0, 1, 1], dtype=int),
        "woe_feat": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float),
    }
    return pd.DataFrame(data)


def test_pass_through_none():
    df = _make_df()
    sc = DynamicScaler(strategy="none")
    out = sc.fit_transform(df)
    pd.testing.assert_frame_equal(out, df)


def test_numeric_only():
    df = _make_df()
    sc = DynamicScaler(strategy="numeric")
    out = sc.fit_transform(df)
    assert out["category"].equals(df["category"])


def test_auto_selects_correct_scalers():
    df = _make_df()
    sc = DynamicScaler(strategy="auto", random_state=0)
    sc.fit(df)
    assert isinstance(sc.scalers_["skewed"], QuantileTransformer)
    norm_scaler = sc.scalers_["normal"]
    assert norm_scaler is None or isinstance(norm_scaler, StandardScaler)


def test_inverse_transform_roundtrip():
    df = _make_df()
    sc = DynamicScaler(strategy="numeric")
    out = sc.fit_transform(df)
    inv = sc.inverse_transform(out)
    pd.testing.assert_frame_equal(inv[df.columns], df, check_dtype=False)


def test_exclude_cols():
    df = _make_df()
    sc = DynamicScaler(strategy="numeric", exclude_cols=["skewed"])
    sc.fit(df)
    assert sc.scalers_["skewed"] is None


def test_woe_skip():
    df = _make_df()
    sc = DynamicScaler(strategy="numeric")
    sc.fit(df)
    assert sc.scalers_["woe_feat"] is None


def test_in_pipeline():
    df = _make_df()
    pipe = Pipeline([("sc", DynamicScaler(strategy="numeric"))])
    out = pipe.fit_transform(df)
    assert isinstance(out, pd.DataFrame)


def test_logging_counts(caplog):
    df = _make_df()
    sc = DynamicScaler(strategy="numeric", verbose=1)
    with caplog.at_level("INFO"):
        sc.fit(df)
    assert any("scaled=" in rec.message for rec in caplog.records)
