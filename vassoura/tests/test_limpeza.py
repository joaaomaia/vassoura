import numpy as np
import pandas as pd

from vassoura.limpeza import clean


def _make_limpeza_data():
    np.random.seed(1)
    x1 = np.random.normal(size=150)
    x2 = x1 * 0.9 + np.random.normal(
        scale=0.1, size=150
    )  # altamente correlacionada com x1
    x3 = np.random.normal(size=150)
    target = (x1 + x3 + np.random.normal(scale=0.3, size=150)) > 0
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target.astype(int)})


def test_clean_removes_correlated_variable():
    df = _make_limpeza_data()
    df_clean, dropped, _, _ = clean(
        df,
        target_col="target",
        keep_cols=["x1"],
        corr_threshold=0.85,
        vif_threshold=5,
        verbose="none",
    )
    assert "x2" in dropped
    assert "x1" in df_clean.columns
    assert "x2" not in df_clean.columns


def test_clean_keeps_unrelated_variable():
    df = _make_limpeza_data()
    df_clean, dropped, _, _ = clean(
        df,
        target_col="target",
        keep_cols=["x3"],
        corr_threshold=0.85,
        vif_threshold=5,
        verbose="none",
    )
    assert "x3" in df_clean.columns


def test_clean_handles_negative_correlation():
    df = pd.DataFrame(
        {
            "a": np.arange(50, dtype=float),
            "b": -np.arange(50, dtype=float),
            "c": np.random.randn(50),
            "target": np.random.randint(0, 2, size=50),
        }
    )
    df_clean, dropped, _, _ = clean(
        df,
        target_col="target",
        corr_threshold=0.8,
        vif_threshold=None,
        verbose="none",
    )
    assert ("a" in dropped) or ("b" in dropped)
