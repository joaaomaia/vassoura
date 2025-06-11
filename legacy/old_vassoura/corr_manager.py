from __future__ import annotations

"""Correlation manager selecting association metrics per variable type."""

from collections import Counter
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import normaltest, pearsonr, spearmanr, pointbiserialr, chi2_contingency
from sklearn.metrics import mutual_info_score

from .utils import parse_verbose, search_dtypes, woe_encode
from .correlacao import _cramers_v, _cramers_v_matrix


LOGGER = logging.getLogger(__name__)

__all__ = ["CorrelationManager"]


class CorrelationManager:
    """Compute pairwise associations choosing methods automatically."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        include_target: bool = False,
        limite_categorico: int = 50,
        force_categorical: Optional[List[str]] = None,
        remove_ids: bool = False,
        id_patterns: Optional[List[str]] = None,
        date_col: Optional[List[str]] = None,
        ordinal_cols: Optional[List[str]] = None,
        method: str = "auto",
        verbose: str | bool = "basic",
        engine: str = "pandas",
        cramer: bool = False,
    ) -> None:
        verbose, verbose_types = parse_verbose(verbose)
        self.verbose = verbose
        self.method = method
        self.engine = engine
        self.cramer = cramer
        self.target = None
        if target_col and target_col in df.columns:
            self.target = df[target_col]
        df_work = df.copy()
        if not cramer and self.target is not None and self.target.dropna().nunique() == 2:
            t = self.target
            if set(t.dropna().unique()) != {0, 1}:
                mapping = {val: i for i, val in enumerate(sorted(t.dropna().unique()))}
                t = t.map(mapping)
            try:
                tmp = df_work.drop(columns=[target_col], errors="ignore")
                tmp = woe_encode(tmp, t)
                for col in tmp.columns:
                    df_work[col] = tmp[col]
            except Exception:
                cat_cols_tmp = df_work.select_dtypes(include=["object", "category"]).columns
                for c in cat_cols_tmp:
                    df_work[c] = pd.factorize(df_work[c])[0]
        elif not cramer:
            cat_cols_tmp = [c for c in df_work.select_dtypes(include=["object", "category"]).columns if c != target_col]
            for c in cat_cols_tmp:
                df_work[c] = pd.factorize(df_work[c])[0]
        self.df = (
            df_work.drop(columns=[target_col], errors="ignore")
            if target_col and not include_target
            else df_work.copy()
        )
        num_cols, cat_cols = search_dtypes(
            self.df,
            target_col=None,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            date_col=date_col,
            verbose=verbose,
            verbose_types=verbose_types,
        )
        self.ordinal_cols = set(ordinal_cols or [])
        self.num_cols = num_cols
        self.cat_cols = [c for c in cat_cols if c not in self.ordinal_cols]
        self.binary_cols = [
            c for c in self.df.columns if self.df[c].dropna().nunique() == 2
        ]
        self.method_counts: Counter[str] = Counter()
        self.matrix: pd.DataFrame | None = None
        self.methods_matrix: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def _col_type(self, col: str) -> str:
        if col in self.ordinal_cols:
            return "ordinal"
        if col in self.binary_cols:
            return "binary"
        if col in self.num_cols:
            return "numeric"
        return "nominal"

    def _calc_eta(self, x: pd.Series, y: pd.Series) -> float:
        if y.dtype.kind not in "biufc":
            categories = list(y.dropna().unique())
            groups = [x[y == cat] for cat in categories]
        else:
            categories = list(x.dropna().unique())
            groups = [y[x == cat] for cat in categories]
            x, y = y, x
        means = np.array([g.mean() for g in groups])
        n = np.array([g.count() for g in groups])
        grand_mean = np.average(x.dropna(), weights=None)
        ss_between = np.sum(n * (means - grand_mean) ** 2)
        ss_total = np.nansum((x - grand_mean) ** 2)
        if ss_total == 0:
            return np.nan
        return float(np.sqrt(ss_between / ss_total))

    def _choose_numeric_method(self, s1: pd.Series, s2: pd.Series) -> str:
        try:
            p1 = normaltest(s1.dropna()).pvalue
            p2 = normaltest(s2.dropna()).pvalue
            if p1 > 0.05 and p2 > 0.05:
                return "pearson"
        except Exception:
            pass
        return "spearman"

    def _compute_pair(self, s1: pd.Series, s2: pd.Series, m: str) -> float:
        try:
            if m == "pearson":
                return float(pearsonr(s1, s2)[0])
            if m == "spearman":
                return float(spearmanr(s1, s2)[0])
            if m == "pointbiserial":
                if s1.nunique() == 2:
                    return float(pointbiserialr(s1, s2)[0])
                return float(pointbiserialr(s2, s1)[0])
            if m == "cramer":
                return float(_cramers_v(s1, s2))
            if m == "chi2":
                tab = pd.crosstab(s1, s2)
                if tab.empty:
                    return np.nan
                chi2, _, _, _ = chi2_contingency(tab, correction=False)
                n = tab.to_numpy().sum()
                return float(np.sqrt(chi2 / n))
            if m == "mi":
                return float(mutual_info_score(s1, s2))
            if m == "eta":
                return self._calc_eta(s1, s2)
        except Exception as exc:
            LOGGER.warning("Falha ao correlacionar %s e %s: %s", s1.name, s2.name, exc)
        return np.nan

    def _auto_method(self, t1: str, t2: str, s1: pd.Series, s2: pd.Series) -> str:
        pair = {t1, t2}
        if pair == {"numeric"}:
            return self._choose_numeric_method(s1, s2)
        if pair == {"ordinal"} or pair == {"ordinal", "numeric"}:
            return "spearman"
        if pair == {"binary", "numeric"}:
            return "pointbiserial"
        if pair == {"nominal"}:
            return "cramer"
        if pair == {"binary", "nominal"}:
            return "chi2"
        if pair == {"binary"}:
            return "cramer"
        if ("nominal" in pair and "numeric" in pair) or ("nominal" in pair and "ordinal" in pair):
            return "eta"
        return "spearman"

    # ------------------------------------------------------------------
    def compute(self) -> pd.DataFrame:
        cols = list(self.df.columns)
        n = len(cols)
        mat = np.eye(n)
        methods = np.full((n, n), "")
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = cols[i], cols[j]
                s1, s2 = self.df[c1], self.df[c2]
                if self.method == "auto":
                    t1, t2 = self._col_type(c1), self._col_type(c2)
                    m = self._auto_method(t1, t2, s1, s2)
                else:
                    m = self.method
                val = self._compute_pair(s1, s2, m)
                mat[i, j] = mat[j, i] = val
                methods[i, j] = methods[j, i] = m
                self.method_counts[m] += 1
                if self.verbose == "full":
                    LOGGER.info("%s × %s -> %s = %.3f", c1, c2, m, val)
        self.matrix = pd.DataFrame(mat, index=cols, columns=cols)
        self.methods_matrix = pd.DataFrame(methods, index=cols, columns=cols)
        return self.matrix

    # Convenience -------------------------------------------------------
    def dominant_numeric_method(self) -> str:
        num_pairs_methods = []
        if self.methods_matrix is None:
            return "pearson"
        cols = [c for c in self.df.columns if self._col_type(c) in {"numeric", "binary", "ordinal"}]
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                m = self.methods_matrix.loc[c1, c2]
                if m:
                    num_pairs_methods.append(m)
        cnt = Counter(num_pairs_methods)
        if cnt.get("spearman", 0) > cnt.get("pearson", 0):
            return "spearman"
        return "pearson"

    def numeric_matrix(self, method: str | None = None) -> pd.DataFrame:
        cols = [c for c in self.df.columns if self._col_type(c) in {"numeric", "binary", "ordinal"}]
        if not cols:
            return pd.DataFrame()
        method = method or self.dominant_numeric_method()
        return self.df[cols].corr(method=method)

    def cat_matrix(self) -> pd.DataFrame:
        cols = [c for c in self.df.columns if self._col_type(c) in {"nominal", "binary"}]
        if len(cols) < 2:
            return pd.DataFrame()
        return _cramers_v_matrix(self.df[cols])

