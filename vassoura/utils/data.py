from __future__ import annotations

import pandas as pd

from .weights import make_balanced_sample_weights

__all__ = ["split_dtypes", "calculate_sample_weights"]


def split_dtypes(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return lists of numeric and categorical columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def calculate_sample_weights(y) -> pd.Series:
    """Wrapper for balanced sample weight calculation."""
    return make_balanced_sample_weights(y)
