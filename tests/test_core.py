from __future__ import annotations
"""Testes básicos de integração do pacote Vassoura.

Execute com:

    pytest -q

Requisitos: ``pytest`` instalado (incluído em ``pip install vassoura[dev]``).
"""
import numpy as np
import pandas as pd

import vassoura as vs  # type: ignore

# ---------------------------------------------------------------------------
# Dataset sintético
# ---------------------------------------------------------------------------

def _make_dummy_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = x1 * 0.95 + rng.normal(scale=0.05, size=n)  # altamente correlacionada com x1
    x3 = rng.normal(size=n)
    cat = rng.choice(list("ABC"), size=n)
    target = (x1 + x3 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat, "target": target})

# ---------------------------------------------------------------------------
# search_dtypes
# ---------------------------------------------------------------------------

def test_search_dtypes():
    df = _make_dummy_df()
    num, cat = vs.search_dtypes(df, target_col="target", verbose=False)
    assert set(num) == {"x1", "x2", "x3"}
    assert set(cat) == {"cat"}

# ---------------------------------------------------------------------------
# compute_corr_matrix
# ---------------------------------------------------------------------------

def test_compute_corr_matrix():
    df = _make_dummy_df()
    corr = vs.compute_corr_matrix(df, method="pearson", target_col="target", verbose=False)
    assert corr.shape[0] == corr.shape[1]  # quadrada
    assert "x1" in corr.columns and "x2" in corr.columns
    # x1 e x2 devem ter alta correlação
    assert corr.loc["x1", "x2"] > 0.9

# ---------------------------------------------------------------------------
# compute_vif
# ---------------------------------------------------------------------------

def test_compute_vif():
    df = _make_dummy_df()
    vif = vs.compute_vif(df, target_col="target", verbose=False)
    # Deve existir uma linha por variável numérica
    assert set(vif["variable"]) == {"x1", "x2", "x3"}
    # x1 ou x2 devem ter VIF alto devido à correlação
    assert vif["vif"].max() > 5

# ---------------------------------------------------------------------------
# clean_multicollinearity
# ---------------------------------------------------------------------------

def test_clean_multicollinearity():
    df = _make_dummy_df()
    df_clean, dropped, _, _ = vs.clean_multicollinearity(
        df,
        target_col="target",
        keep_cols=["x1"],
        corr_threshold=0.9,
        vif_threshold=5,
        verbose=False,
    )
    # 'x2' deve ser removida (alta correlação com x1) mas x1 deve permanecer
    assert "x2" in dropped
    assert "x1" in df_clean.columns
