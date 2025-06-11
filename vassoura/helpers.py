from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["search_dtypes", "calculate_ks", "calculate_psi"]


def search_dtypes(
    df: pd.DataFrame,
    target_col: str = "target",
    limite_categorico: int = 50,
) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical columns.

    Columns with integer type and fewer than ``limite_categorico`` unique values
    are treated as categorical. The ``target_col`` is ignored.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols: list[str] = [c for c in df.columns if c not in num_cols]
    for col in num_cols.copy():
        if col == target_col:
            num_cols.remove(col)
            continue
        if (
            pd.api.types.is_integer_dtype(df[col])
            and df[col].nunique() <= limite_categorico
        ):
            num_cols.remove(col)
            cat_cols.append(col)
    cat_cols = [c for c in cat_cols if c != target_col]
    return num_cols, cat_cols


def _prepare_series(dist: dict[str, int]) -> pd.Series:
    s = pd.Series(dist, dtype=float)
    s = s.reindex(sorted(s.index))
    s = s.fillna(0)
    if s.sum() == 0:
        return s
    return s / s.sum()


def calculate_ks(dist1: dict[str, int], dist2: dict[str, int]) -> float:
    """Compute KS statistic from two distributions represented as dictionaries."""
    s1 = _prepare_series(dist1)
    s2 = _prepare_series(dist2)
    all_bins = s1.index.union(s2.index)
    s1 = s1.reindex(all_bins, fill_value=0)
    s2 = s2.reindex(all_bins, fill_value=0)
    cdf1 = s1.cumsum()
    cdf2 = s2.cumsum()
    return float((cdf1 - cdf2).abs().max())


def calculate_psi(dist1: dict[str, int], dist2: dict[str, int]) -> float:
    """Compute Population Stability Index between two distributions."""
    s1 = _prepare_series(dist1)
    s2 = _prepare_series(dist2)
    all_bins = s1.index.union(s2.index)
    s1 = s1.reindex(all_bins, fill_value=1e-6)
    s2 = s2.reindex(all_bins, fill_value=1e-6)
    psi = ((s1 - s2) * np.log(s1 / s2)).sum()
    return float(psi)
