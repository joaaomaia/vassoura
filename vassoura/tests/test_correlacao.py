import pandas as pd
import numpy as np
from vassoura.core import compute_corr_matrix

def _make_correlacao_data():
    np.random.seed(123)
    x1 = np.random.normal(size=100)
    x2 = x1 * 0.95 + np.random.normal(scale=0.05, size=100)
    x3 = np.random.uniform(size=100)
    cat = np.random.choice(['A', 'B', 'C'], size=100)
    target = (x1 + x3 + np.random.normal(scale=0.2, size=100)) > 0
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "target": target.astype(int)})

def test_corr_matrix_pearson():
    df = _make_correlacao_data()
    corr = compute_corr_matrix(df, method="pearson", target_col="target", verbose=False)
    assert isinstance(corr, pd.DataFrame)
    assert "x1" in corr.columns and "x2" in corr.columns
    assert corr.loc["x1", "x2"] > 0.9

def test_corr_matrix_spearman():
    df = _make_correlacao_data()
    corr = compute_corr_matrix(df, method="spearman", target_col="target", verbose=False)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]

def test_corr_matrix_auto_detect():
    df = _make_correlacao_data()
    corr = compute_corr_matrix(df, method="auto", target_col="target", verbose=False)
    assert isinstance(corr, pd.DataFrame)
    assert "x3" in corr.columns
