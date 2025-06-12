from __future__ import annotations

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from vassoura.preprocessing.encoders import OrdinalSafe, WOEGuard


@given(
    st.lists(st.integers(0, 1), min_size=20, max_size=50).filter(
        lambda vals: 0 in vals and 1 in vals
    )
)
def test_woe_preserves_binary_target(y_list):
    X = pd.DataFrame({"cat": ["a"] * len(y_list)})
    y = pd.Series(y_list)
    enc = WOEGuard(min_samples=1)
    enc.fit(X, y)
    out = enc.transform(X)
    assert out.shape[0] == len(X)
    assert "woe_cat" in out.columns


def test_handle_missing_modes():
    X = pd.DataFrame({"cat": ["a", None, "b", None]})
    y = pd.Series([0, 1, 0, 1])
    enc = WOEGuard(handle_missing="most_frequent", min_samples=1)
    enc.fit(X, y)
    assert enc.fill_map_["cat"] in {"a", "b"}
    out = enc.transform(X)
    assert not out.isna().any().any()


def test_ordinal_safe_roundtrip():
    df = pd.DataFrame({"c": ["x", "y", "x", "z"]})
    enc = OrdinalSafe()
    enc.fit(df)
    out = enc.transform(df)
    assert set(out["c"].unique()).issubset({0, 1, 2})


def test_woe_drop_and_clip():
    X = pd.DataFrame({"cat": ["a", "b", None, "b", "c", "c", "c"]})
    y = pd.Series([0, 1, 0, 1, 0, 0, 1])
    enc = WOEGuard(
        handle_missing="drop", apply_smoothing=False, clip_values=False, min_samples=1
    )
    enc.fit(X, y)
    out = enc.transform(X)
    assert out.shape[0] == len(X)
