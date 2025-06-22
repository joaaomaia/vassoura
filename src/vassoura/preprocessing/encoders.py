# -*- coding: utf-8 -*-
"""Encoding utilities for categorical features."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from vassoura.logs import get_logger

class WOEGuard(BaseEstimator, TransformerMixin):
    """Weight-of-Evidence encoder with safeguards for rare categories."""

    def __init__(
        self,
        min_samples: int = 50,
        apply_smoothing: bool = True,
        smoothing_alpha: float = 0.5,
        handle_missing: str = "separate",
        clip_values: bool = True,
        verbose: int = 1,
    ) -> None:
        self.min_samples = min_samples
        self.apply_smoothing = apply_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.handle_missing = handle_missing
        self.clip_values = clip_values
        self.verbose = verbose
        self.logger = get_logger("WOEGuard")
        if verbose >= 2:
            self.logger.setLevel("DEBUG")
        else:
            self.logger.setLevel("INFO")
        self.woe_dict_: dict[str, dict[Any, float]] = {}
        self.iv_: pd.Series = pd.Series(dtype=float)
        self.cat_counts_: pd.Series = pd.Series(dtype=int)
        self.fill_map_: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _prepare_col(self, s: pd.Series, col: str) -> pd.Series:
        if self.handle_missing == "separate":
            return s.fillna("__missing__")
        if self.handle_missing == "most_frequent":
            mode = s.dropna().mode()
            fill_val = mode.iloc[0] if not mode.empty else "__missing__"
            self.fill_map_[col] = fill_val
            return s.fillna(fill_val)
        if self.handle_missing == "drop":
            return s.dropna()
        raise ValueError(
            "handle_missing must be 'separate', 'most_frequent' or 'drop'"
        )

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Iterable) -> "WOEGuard":
        df = pd.DataFrame(X).copy()
        y_ser = pd.Series(y)
        uniq = sorted(pd.unique(y_ser.dropna()))
        if len(uniq) != 2 or set(uniq) != {0, 1}:
            raise ValueError(
                "WOEGuard supports binary target with labels {0,1}"
            )
        total_good = int((y_ser == 0).sum())
        total_bad = int((y_ser == 1).sum())

        iv_dict: dict[str, float] = {}
        cat_count: dict[str, int] = {}
        merged_total = 0

        for col in df.columns:
            s = df[col]
            y_col = y_ser
            if self.handle_missing == "drop":
                mask = s.notna()
                s = s[mask]
                y_col = y_ser[mask]
            else:
                s = self._prepare_col(s, col)

            counts = s.value_counts()
            rare = counts[counts < self.min_samples].index
            merged_total += int(counts[counts < self.min_samples].sum())
            if len(rare) > 0:
                s = s.where(~s.isin(rare), "__other__")
            temp = pd.DataFrame({"cat": s, "target": y_col})
            ctab = pd.crosstab(temp["cat"], temp["target"])
            if 0 not in ctab.columns:
                ctab[0] = 0
            if 1 not in ctab.columns:
                ctab[1] = 0
            good = ctab[0].astype(float)
            bad = ctab[1].astype(float)
            if self.apply_smoothing:
                good += self.smoothing_alpha
                bad += self.smoothing_alpha
                N_good = total_good + self.smoothing_alpha * len(ctab)
                N_bad = total_bad + self.smoothing_alpha * len(ctab)
            else:
                N_good = total_good
                N_bad = total_bad
            if self.clip_values:
                good = good.clip(lower=1e-6)
                bad = bad.clip(lower=1e-6)
            woe = np.log((good / N_good) / (bad / N_bad))
            iv_vals = ((good / N_good) - (bad / N_bad)) * woe
            iv = float(iv_vals.sum())
            mapping = woe.to_dict()
            self.woe_dict_[col] = mapping
            iv_dict[col] = iv
            cat_count[col] = len(mapping)
            if self.verbose >= 2:
                self.logger.debug("%s IV=%.4f", col, iv)

        self.iv_ = pd.Series(iv_dict)
        self.cat_counts_ = pd.Series(cat_count)
        self.logger.info(
            "[WOEGuard] fitted %d categorical cols | IV>0.02: %d "
            "| merged rare cats: %d",
            len(df.columns),
            int((self.iv_ > 0.02).sum()),
            merged_total,
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        out: dict[str, Any] = {}
        for col in df.columns:
            s = df[col]
            if self.handle_missing == "separate":
                s = s.fillna("__missing__")
            elif self.handle_missing == "most_frequent":
                fill_val = self.fill_map_.get(col, "__missing__")
                s = s.fillna(fill_val)
            elif self.handle_missing == "drop":
                pass  # keep NaNs
            mapping = self.woe_dict_.get(col, {})
            default = mapping.get("__other__", 0.0)
            out[f"woe_{col}"] = (
                s.map(mapping).fillna(default).astype(np.float32)
            )
        return pd.DataFrame(out, index=df.index)


class OneHotLite(OneHotEncoder):
    """Wrapper around :class:`sklearn.preprocessing.OneHotEncoder`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=0.01,
            **kwargs,
        )


class OrdinalSafe(BaseEstimator, TransformerMixin):
    """Frequency-based ordinal encoder for tree models."""

    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self.logger = get_logger("OrdinalSafe")
        if verbose >= 2:
            self.logger.setLevel("DEBUG")
        else:
            self.logger.setLevel("INFO")
        self.mapping_: dict[str, dict[Any, int]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OrdinalSafe":
        df = pd.DataFrame(X)
        for col in df.columns:
            counts = df[col].value_counts()
            mapping = {cat: i for i, cat in enumerate(counts.index)}
            self.mapping_[col] = mapping
            if self.verbose >= 2:
                self.logger.debug("%s categories=%d", col, len(mapping))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X)
        out: dict[str, Any] = {}
        for col in df.columns:
            mapping = self.mapping_.get(col, {})
            default = len(mapping)
            out[col] = df[col].map(mapping).fillna(default).astype(np.int32)
        return pd.DataFrame(out, index=df.index)


__all__ = ["WOEGuard", "OneHotLite", "OrdinalSafe"]
