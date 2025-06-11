from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit

from vassoura.logs import get_logger


class SampleManager(BaseEstimator, TransformerMixin):
    """Down-sample large datasets while keeping target integrity."""

    def __init__(
        self,
        strategy: str = "auto",
        limit_mb: int = 500,
        frac: float = 0.2,
        stratify: bool = True,
        time_col: str | None = None,
        random_state: int | None = 42,
        verbose: int = 1,
    ) -> None:
        self.strategy = strategy
        self.limit_mb = limit_mb
        self.frac = frac
        self.stratify = stratify
        self.time_col = time_col
        self.random_state = random_state
        self.verbose = verbose
        self.logger = get_logger("SampleManager")
        if verbose >= 2:
            self.logger.setLevel("DEBUG")
        self._do_sample = False
        self.sample_size_ = 0
        self.mask_ = np.array([], dtype=bool)

    def _should_sample(self, X: pd.DataFrame) -> bool:
        if self.strategy == "none":
            return False
        if self.strategy in {"time_series", "stratified"}:
            return True
        # auto strategy
        mem_mb = X.memory_usage(deep=True).sum() / 2 ** 20
        row_cap = self.limit_mb * 25
        return mem_mb > self.limit_mb or len(X) > row_cap

    def fit(
        self, X: pd.DataFrame, y: Iterable | None = None
    ) -> "SampleManager":
        if self.strategy == "stratified" and y is None:
            raise ValueError("y is required for stratified sampling")
        if self.strategy == "time_series":
            if self.time_col is None or self.time_col not in X.columns:
                raise ValueError(
                    "time_col must be provided for time_series strategy"
                )
        if self.strategy not in {"auto", "stratified", "time_series", "none"}:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        self._do_sample = self._should_sample(X)
        if not self._do_sample:
            self.sample_size_ = len(X)
            self.mask_ = np.ones(len(X), dtype=bool)
        else:
            self.sample_size_ = int(len(X) * self.frac)
            self.mask_ = np.zeros(len(X), dtype=bool)
        original_rows = len(X)
        sampled_rows = self.sample_size_ if self._do_sample else original_rows
        if self.verbose >= 2:
            mem_before = X.memory_usage(deep=True).sum() / 2 ** 20
            mem_after = mem_before * (self.frac if self._do_sample else 1)
            self.logger.debug(
                "memory before=%.2fMB after=%.2fMB", mem_before, mem_after
            )
        self.logger.info(
            "[SampleManager] strategy='%s', triggered=%s, "
            "original_rows=%d, sampled_rows=%d, frac=%s",
            self.strategy,
            self._do_sample,
            original_rows,
            sampled_rows,
            self.frac,
        )
        return self

    def _apply_sampling(
        self, X: pd.DataFrame, y: Iterable | None = None
    ) -> tuple[pd.DataFrame, Iterable | None]:
        if not self._do_sample:
            return X, y

        if self.strategy == "time_series":
            total = int(np.ceil(len(X) * self.frac))
            first = total // 2
            last = total - first
            idx_first = X.index[:first]
            idx_last = X.index[-last:]
            mask = X.index.isin(idx_first.union(idx_last))
        elif self.strategy == "stratified" or (
            self.strategy == "auto" and self.stratify and y is not None
        ):
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - self.frac,
                random_state=self.random_state,
            )
            train_idx, _ = next(splitter.split(X, y))
            mask = np.zeros(len(X), dtype=bool)
            mask[train_idx] = True
        else:
            idx = X.sample(
                frac=self.frac, random_state=self.random_state
            ).index
            mask = X.index.isin(idx)

        self.mask_ = mask
        self.sample_size_ = int(mask.sum())
        X_res = X.loc[mask]
        if y is not None:
            if isinstance(y, pd.Series):
                y_res = y.loc[mask]
            else:
                y_res = np.asarray(y)[mask]
            return X_res, y_res
        return X_res, None

    def transform(
        self, X: pd.DataFrame, y: Iterable | None = None
    ) -> pd.DataFrame | tuple[pd.DataFrame, Iterable]:
        X_res, y_res = self._apply_sampling(X, y)
        if y is not None:
            return X_res, y_res
        return X_res

    def fit_resample(
        self, X: pd.DataFrame, y: Iterable
    ) -> tuple[pd.DataFrame, Iterable]:
        return self.fit(X, y).transform(X, y)

    def get_support_mask(self) -> np.ndarray:
        if self.mask_.size == 0:
            raise AttributeError("SampleManager has not been fitted yet")
        return self.mask_
