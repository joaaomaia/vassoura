# -*- coding: utf-8 -*-
"""Dynamic feature scaling utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
)

from vassoura.logs import get_logger


class DynamicScaler(BaseEstimator, TransformerMixin):
    """Adaptive per-column feature scaling."""

    def __init__(
        self,
        strategy: str = "auto",
        preferred: str = "standard",
        exclude_cols: list[str] | None = None,
        n_quantiles: int = 1000,
        output_distribution: str = "normal",
        verbose: int = 1,
        random_state: int | None = 42,
    ) -> None:
        self.strategy = strategy
        self.preferred = preferred
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.verbose = verbose
        self.random_state = random_state

        self.logger = get_logger("DynamicScaler")
        if verbose >= 2:
            self.logger.setLevel("DEBUG")
        else:
            self.logger.setLevel("INFO")

        self.scalers_: dict[str, BaseEstimator | None] = {}
        self.columns_: list[str] = []
        self.stats_: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _make_scaler(self) -> BaseEstimator:
        if self.preferred == "standard":
            return StandardScaler()
        if self.preferred == "minmax":
            return MinMaxScaler()
        if self.preferred == "quantile":
            return QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution=self.output_distribution,
                random_state=self.random_state,
            )
        # default
        return StandardScaler()

    # ------------------------------------------------------------------
    def _choose_auto(self, x: pd.Series) -> BaseEstimator | None:
        sample = x.dropna().astype(float)
        if sample.empty:
            return None
        if sample.std(ddof=0) == 0:
            return None
        var = sample.var()
        sk = skew(sample, nan_policy="omit")
        kt = kurtosis(sample, nan_policy="omit")
        uniq = sample.nunique()
        self._cur_stats = {
            "var": var,
            "skew": sk,
            "kurtosis": kt,
            "unique": uniq,
            "mean": sample.mean(),
            "std": sample.std(),
        }
        # gating condition
        if not (var > 1.5 or abs(sk) > 1 or kt > 3 or uniq > 30):
            return None
        if (sample.min() >= 0 and sample.max() <= 1) or var < 0.05:
            return None
        if abs(sk) < 0.5 and abs(kt) < 1:
            return StandardScaler()
        if abs(sk) >= 1 or kt > 3:
            return QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution=self.output_distribution,
                random_state=self.random_state,
            )
        if self.preferred == "minmax":
            return MinMaxScaler()
        if self.preferred == "quantile":
            return QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution=self.output_distribution,
                random_state=self.random_state,
            )
        return StandardScaler()

    # ------------------------------------------------------------------
    def get_scaler(self, col: pd.Series) -> BaseEstimator | None:
        if self.strategy == "none":
            return None
        if self.strategy == "auto":
            return self._choose_auto(col)
        # numeric/all
        return self._make_scaler()

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any = None) -> "DynamicScaler":
        df = pd.DataFrame(X).copy()
        self.columns_ = df.columns.tolist()
        self.scalers_ = {}
        self.stats_ = {}

        count_std = count_qt = count_mm = 0
        scaled = skipped = 0

        for col in self.columns_:
            series = df[col]
            reason = ""
            scaler = None
            if col in self.exclude_cols:
                reason = "excluded"
            elif str(col).startswith("woe_"):
                reason = "woe"
            elif series.isna().all():
                reason = "all_nan"
            elif not pd.api.types.is_numeric_dtype(series):
                reason = "non_numeric"
            else:
                vals = series.dropna().unique()
                if len(vals) <= 2 and set(vals).issubset({0, 1}):
                    reason = "binary"
                else:
                    scaler = self.get_scaler(series)
                    stats = getattr(self, "_cur_stats", {})
                    self.stats_[col] = stats
                    if scaler is not None:
                        scaled += 1
                        if isinstance(scaler, StandardScaler):
                            count_std += 1
                        elif isinstance(scaler, QuantileTransformer):
                            count_qt += 1
                        elif isinstance(scaler, MinMaxScaler):
                            count_mm += 1
                        scaler.fit(series.to_frame())
                    else:
                        reason = (
                            "auto_skip" if self.strategy == "auto" else "none"
                        )
            if col not in self.stats_:
                self.stats_[col] = {}
            self.scalers_[col] = scaler
            if self.verbose >= 2:
                name = scaler.__class__.__name__ if scaler else "None"
                self.logger.debug(
                    "%s -> %s %s", col, name, reason
                )
            if scaler is None:
                skipped += 1

        if self.verbose >= 1:
            msg = (
                "[DynamicScaler] strategy='%s' | examined=%d cols | scaled=%d "
                "(Std:%d, QT:%d, MinMax:%d) | skipped=%d"
            )
            self.logger.info(
                msg,
                self.strategy,
                len(self.columns_),
                scaled,
                count_std,
                count_qt,
                count_mm,
                skipped,
            )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        missing = set(self.columns_) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for transform: {missing}")

        for col, scaler in self.scalers_.items():
            if scaler is not None:
                df[col] = scaler.transform(df[[col]]).ravel()
        return df

    # ------------------------------------------------------------------
    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        for col, scaler in self.scalers_.items():
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                df[col] = scaler.inverse_transform(df[[col]]).ravel()
        return df
