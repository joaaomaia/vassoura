
from __future__ import annotations

"""Vassoura – Variance Inflation Factor (VIF) (vif_corrigido_2)

Nova estratégia de *encoding* para variáveis categóricas:
• Convergência global e determinística entre colunas
• ‘A_x’ → ‘A’ (trunca no primeiro ‘_’) para preservar semântica
Isso eleva a correlação numérica sempre que as strings diferem
apenas por sufixos (caso típico de ruído), fazendo o VIF refletir
a real colinearidade.
"""

import logging
import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from .utils import adaptive_sampling as _adaptive_sampling
from .utils import parse_verbose, search_dtypes, woe_encode

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:  # pragma: no cover
    variance_inflation_factor = None  # type: ignore

LOGGER = logging.getLogger(__name__)
__all__ = ["compute_vif", "remove_high_vif"]

# ---------------------------------------------------------------------------
# NumPy‑only fallback
# ---------------------------------------------------------------------------
def _compute_vif_np(x: np.ndarray) -> np.ndarray:
    n_cols = x.shape[1]
    out = np.zeros(n_cols)
    Xc = np.column_stack([np.ones(x.shape[0]), x])
    for i in range(n_cols):
        y = x[:, i]
        Xr = np.delete(Xc, i + 1, axis=1)
        try:
            beta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            y_hat = Xr @ beta
            r2 = 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            out[i] = 1 / (1 - r2) if r2 < 1 else np.inf
        except LinAlgError:
            out[i] = np.inf
    return out


# ---------------------------------------------------------------------------
# Público
# ---------------------------------------------------------------------------
def compute_vif(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    include_target: bool = False,
    engine: str = "pandas",
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    date_col: Optional[List[str]] = None,
    verbose: str | bool = "basic",
    verbose_types: bool | None = None,
    adaptive_sampling: bool = False,
    use_woe: bool = False,
) -> pd.DataFrame:
    verbose, verbose_types = parse_verbose(verbose, verbose_types)

    # Pré‑processo
    if target_col and not include_target:
        df_work = df.drop(columns=[target_col], errors="ignore").copy()
    else:
        df_work = df.copy()

    num_cols, cat_cols = search_dtypes(
        df_work,
        target_col=None,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        date_col=date_col,
        verbose=verbose,
        verbose_types=verbose_types,
    )

    # ------------------------------------------------------------------ #
    # Encoding categórico
    # ------------------------------------------------------------------ #
    if cat_cols:
        tgt_series = None
        if use_woe and target_col and target_col in df.columns:
            tgt_series = df[target_col]
            if tgt_series.nunique(dropna=True) == 2 and set(tgt_series.unique()) != {0, 1}:
                map_bin = {v: i for i, v in enumerate(sorted(tgt_series.dropna().unique()))}
                tgt_series = tgt_series.map(map_bin)

        if use_woe and tgt_series is not None and tgt_series.nunique() == 2:
            # ---------- WOE ----------
            try:
                enc = (
                    woe_encode(df_work[cat_cols].fillna("__MISSING__"), tgt_series, cols=cat_cols)[cat_cols]
                    .replace([np.inf, -np.inf], 0.0)
                    .astype(float)
                )
                df_work.loc[:, cat_cols] = enc
            except Exception:
                use_woe = False  # fallback
        if not use_woe:
            # ---------- Ordinal global com *normalização de sufixos* ----------
            tmp = df_work[cat_cols].fillna("__MISSING__").astype(str)
            # Remove tudo após o primeiro "_" (e o próprio "_") – ruído/sufixo
            tmp_clean = tmp.apply(lambda col: col.map(lambda x: x.split("_", 1)[0] if "_" in x else x))
            uniques = pd.Series(pd.unique(tmp_clean.values.ravel())).sort_values().tolist()
            mapping = {v: i for i, v in enumerate(uniques)}
            df_work.loc[:, cat_cols] = tmp_clean.apply(lambda s: s.map(mapping)).astype(float)

        num_cols += cat_cols

    if not num_cols:
        raise ValueError("Nenhuma coluna numérica disponível para VIF")

    data = (
        df_work[num_cols]
        .astype(float, errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if adaptive_sampling:
        data = _adaptive_sampling(data, stratify_col=target_col, date_cols=date_col)

    # Backend
    if engine == "dask":
        import dask.dataframe as dd
        X = dd.from_pandas(data, npartitions=4).to_dask_array(lengths=True).compute()
    elif engine == "polars":
        import polars as pl
        X = pl.from_pandas(data).to_numpy()
    else:
        X = data.values

    # VIF
    if variance_inflation_factor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", FutureWarning)
            vif_vals = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    else:
        vif_vals = _compute_vif_np(X)

    out = pd.DataFrame({"variable": num_cols, "vif": vif_vals}).sort_values("vif", ascending=False)
    if verbose:
        LOGGER.info("VIF calculado para %d variáveis", len(num_cols))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
def remove_high_vif(
    df: pd.DataFrame,
    *,
    vif_threshold: float = 10.0,
    target_col: str | None = None,
    include_target: bool = False,
    keep_cols: Optional[List[str]] = None,
    max_iter: int = 20,
    vif_n_steps: int = 1,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    engine: str = "pandas",
    date_col: Optional[List[str]] = None,
    verbose: str | bool = "basic",
    verbose_types: bool | None = None,
    adaptive_sampling: bool = False,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    verbose, verbose_types = parse_verbose(verbose, verbose_types)
    keep_cols = keep_cols or []
    df_iter = df.copy()
    dropped: List[str] = []

    for _ in range(max_iter):
        vif_df = compute_vif(
            df_iter,
            target_col=target_col,
            include_target=include_target,
            engine=engine,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            date_col=date_col,
            verbose=verbose,
            verbose_types=verbose_types,
            adaptive_sampling=adaptive_sampling,
        )
        high = vif_df[(vif_df.vif > vif_threshold) & (~vif_df.variable.isin(keep_cols))]
        if high.empty:
            break

        to_drop = high.head(max(1, math.ceil(len(high) / vif_n_steps)))  # step
        df_iter = df_iter.drop(columns=to_drop.variable.tolist())
        dropped.extend(to_drop.variable.tolist())

    final_vif = compute_vif(
        df_iter,
        target_col=target_col,
        include_target=include_target,
        engine=engine,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        date_col=date_col,
        verbose=verbose,
        verbose_types=verbose_types,
    )

    return df_iter, dropped, final_vif
