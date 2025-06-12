from __future__ import annotations

from typing import Any, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from .scaler import DynamicScaler
from .encoders import WOEGuard, OneHotLite, OrdinalSafe


def make_default_pipeline(
    num_cols: List[str] | None = None,
    cat_cols: List[str] | None = None,
    *,
    scaler_strategy: str = "auto",
    encoder: str = "woe",
    sampler: BaseEstimator | None = None,
) -> Pipeline:
    """Return a pre-configured preprocessing pipeline."""

    if num_cols is None:
        num_sel: Any = make_column_selector(dtype_include=np.number)
    else:
        num_sel = num_cols

    if cat_cols is None:
        cat_sel: Any = make_column_selector(dtype_exclude=np.number)
    else:
        cat_sel = cat_cols

    num_pipe = Pipeline([
        ("scaler", DynamicScaler(strategy=scaler_strategy)),
    ])

    if encoder == "woe":
        cat_encoder: Any = WOEGuard()
    elif encoder == "onehot":
        cat_encoder = OneHotLite()
    elif encoder == "ordinal":
        cat_encoder = OrdinalSafe()
    elif encoder == "none":
        cat_encoder = "drop"
    else:
        raise ValueError(f"Unknown encoder '{encoder}'")

    transformers = [
        ("num", num_pipe, num_sel),
        ("cat", cat_encoder, cat_sel),
    ]

    ct = ColumnTransformer(transformers, remainder="drop")

    steps: list[tuple[str, Any]] = []
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("ct", ct))

    return Pipeline(steps)
