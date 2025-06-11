from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.utils.multiclass import type_of_target

from vassoura.logs import get_logger


logger = get_logger("validation")


class _StratifiedWrapper:
    """Choose StratifiedKFold or KFold based on target type."""

    def __init__(
        self, n_splits: int, shuffle: bool, random_state: int | None
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self._kf = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self.name = "StratifiedKFold(%d)" % n_splits

    def split(self, X, y=None, groups=None):
        if y is not None:
            try:
                target_type = type_of_target(y)
            except Exception:
                target_type = "unknown"
            if target_type in {"binary", "multiclass"}:
                return self._skf.split(X, y, groups)
        self.name = "KFold(%d)" % self.n_splits
        return self._kf.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def get_stratified_cv(
    n_splits: int = 5, shuffle: bool = True, random_state: int | None = 42
):
    """Return a CV object that stratifies when appropriate."""
    cv = _StratifiedWrapper(n_splits, shuffle, random_state)
    logger.info(
        "[CV] Using %s(n_splits=%d, shuffle=%s)",
        "StratifiedKFold",
        n_splits,
        shuffle,
    )
    return cv


class _TimeSeriesWrapper(TimeSeriesSplit):
    def split(self, X, y=None, groups=None):  # type: ignore[override]
        for train_idx, test_idx in super().split(X, y, groups):
            if train_idx.size and test_idx.size:
                if train_idx.max() + self.gap >= test_idx.min():
                    raise ValueError("Gap leads to data leakage")
            yield train_idx, test_idx


def get_time_series_cv(
    n_splits: int = 5, test_size: int | None = None, gap: int = 0
):
    if gap < 0:
        raise ValueError("gap must be >= 0")
    cv = _TimeSeriesWrapper(n_splits=n_splits, test_size=test_size, gap=gap)
    cv.name = f"TimeSeriesSplit({n_splits})"
    logger.info(
        "[CV] Using TimeSeriesSplit(n_splits=%d, gap=%d)",
        n_splits,
        gap,
    )
    return cv


def train_test_split_by_date(
    df: pd.DataFrame, time_col: str, split_date: str | pd.Timestamp
):
    if time_col not in df.columns:
        raise ValueError(f"{time_col} not in DataFrame")
    ts = pd.to_datetime(split_date)
    train = df[df[time_col] <= ts]
    test = df[df[time_col] > ts]
    return train, test
