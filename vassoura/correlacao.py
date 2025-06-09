from __future__ import annotations

"""Vassoura – Módulo de correlação
================================

Funções para cálculo e visualização de matrizes de correlação, com
suporte a Pearson, Spearman e Cramér‑V. Inclui *heat‑maps* em Seaborn com
dimensionamento automático de figura.

Principais funções públicas
---------------------------
- ``compute_corr_matrix``  → retorna *DataFrame* de correlação
- ``plot_corr_heatmap``    → plota *heat‑map* estilizado usando Seaborn

Todas as funções registram *logs* detalhados através do *logger*
``vassoura``, configurado em ``vassoura.utils``.
"""

import itertools
import logging
import math
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt

try:  # mplcursors é opcional
    import mplcursors
except Exception:  # pragma: no cover - ambiente pode não ter mplcursors
    mplcursors = None
import numpy as np


def _pick_text_color(rgb: tuple[float, float, float]) -> str:
    """Escolhe cor de texto de acordo com luminância."""
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000" if lum > 0.6 else "#fff"


import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from .utils import (
    adaptive_sampling,
    figsize_from_matrix,
    parse_verbose,
    search_dtypes,
    suggest_corr_method,
    woe_encode,
)

__all__ = [
    "compute_corr_matrix",
    "plot_corr_heatmap",
]

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calcula o Cramér‑V para duas variáveis categóricas."""
    confusion = pd.crosstab(x, y)
    if confusion.empty:
        return np.nan
    chi2, _, _, _ = chi2_contingency(confusion, correction=False)
    n = confusion.values.sum()
    if n == 0:
        return np.nan
    r, k = confusion.shape
    phi2 = chi2 / n
    denom = min(k - 1, r - 1)
    if denom == 0:
        return np.nan
    return math.sqrt(phi2 / denom)


def _cramers_v_matrix(df_cat: pd.DataFrame) -> pd.DataFrame:
    """Retorna matriz de Cramér‑V para *DataFrame* apenas com categóricas."""
    cols = df_cat.columns
    n = len(cols)
    mat = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        v = _cramers_v(df_cat.iloc[:, i], df_cat.iloc[:, j])
        mat[i, j] = mat[j, i] = v
    return pd.DataFrame(mat, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def compute_corr_matrix(
    df: pd.DataFrame,
    *,
    method: str = "auto",
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
    cramer: bool = False,
) -> pd.DataFrame:
    """Calcula matriz de correlação.

    Parameters
    ----------
    df : pandas.DataFrame
        *DataFrame* completo.
    method : {"auto", "pearson", "spearman", "cramer"}
        Método de correlação. ``auto`` escolhe com base nas colunas.
    target_col : str | None
        Nome da coluna *target*. Será removida a não ser que
        ``include_target`` seja ``True``.
    include_target : bool, default False
        Inclui a coluna *target* na matriz de correlação.
    limite_categorico, force_categorical, remove_ids, id_patterns, date_col,
    verbose, verbose_types
        Encaminhados para ``search_dtypes``.
    adaptive_sampling : bool
        Se ``True``, usa amostra de até 50k linhas para acelerar o cálculo
        em datasets grandes.
    engine : {"pandas", "dask", "polars"}
        Backend a ser utilizado quando possível.
    cramer : bool, default False
        Quando ``True``, calcula também a matriz de Cramér‑V para variáveis
        categóricas e a incorpora ao resultado.

    Returns
    -------
    pandas.DataFrame
        Matriz de correlação.
    """
    # Configura verbosidade
    verbose, verbose_types = parse_verbose(verbose, verbose_types)

    # Filtra colunas segundo parâmetros
    drop_target = target_col and not include_target
    df_work = (
        df.drop(columns=[target_col], errors="ignore") if drop_target else df.copy()
    )

    # Identifica tipos
    num_cols, cat_cols = search_dtypes(
        df_work,
        target_col=None,  # já removido se for o caso
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        date_col=date_col,
        verbose=verbose,
        verbose_types=verbose_types,
    )

    # ------------------------------------------------------------------
    # WoE fallback for categorical features when Cramér is disabled
    # ------------------------------------------------------------------
    if not cramer and cat_cols:
        target_series = None
        if target_col and target_col in df.columns:
            target_series = df[target_col]
            if target_series.dropna().nunique() == 2 and set(target_series.dropna().unique()) != {0, 1}:
                mapping = {val: i for i, val in enumerate(sorted(target_series.dropna().unique()))}
                target_series = target_series.map(mapping)
        if target_series is not None and target_series.dropna().nunique() == 2:
            try:
                df_work[cat_cols] = woe_encode(df_work[cat_cols], target_series, cols=cat_cols)[cat_cols]
            except Exception:
                df_work[cat_cols] = df_work[cat_cols].apply(lambda s: pd.factorize(s)[0])
        else:
            df_work[cat_cols] = df_work[cat_cols].apply(lambda s: pd.factorize(s)[0])
        num_cols = num_cols + cat_cols
        cat_cols = []

    # Escolha automática do método para variáveis numéricas
    numeric_method = method
    if method == "auto":
        numeric_method = "pearson" if cat_cols == [] else "spearman"
        if not num_cols and cat_cols:
            numeric_method = "cramer"
        if verbose:
            LOGGER.info("Método de correlação sugerido: %s", numeric_method)

    if numeric_method == "cramer" and not cramer:
        if num_cols:
            numeric_method = "spearman"
        else:
            raise ValueError("Cramér-V desabilitado e não há colunas numéricas")

    cat_corr = None
    if numeric_method in ("pearson", "spearman"):
        if not num_cols:
            raise ValueError(
                "Não há colunas numéricas para calcular correlação %s" % numeric_method
            )
        data = df_work[num_cols].copy()
        if adaptive_sampling:
            data = adaptive_sampling(
                data,
                stratify_col=target_col,
                date_cols=date_col,
            )
        corr_engine = engine
        if engine == "dask":
            try:
                import dask.dataframe as dd
            except ImportError as exc:
                raise ImportError("engine='dask' requer dask instalado") from exc
            data_dd = dd.from_pandas(data, npartitions=4)
            if numeric_method == "pearson":
                num_corr = data_dd.corr(method="pearson").compute()
            else:
                LOGGER.info(
                    "Método %s não suportado por engine 'dask'; utilizando pandas.",
                    numeric_method,
                )
                corr_engine = "pandas"
                num_corr = data.corr(method=numeric_method)
        elif engine == "polars":
            try:
                import polars as pl
            except ImportError as exc:
                raise ImportError("engine='polars' requer polars instalado") from exc
            if numeric_method == "pearson":
                num_corr = pl.from_pandas(data).corr().to_pandas()
            else:
                LOGGER.info(
                    "Método %s não suportado por engine 'polars'; utilizando pandas.",
                    numeric_method,
                )
                corr_engine = "pandas"
                num_corr = data.corr(method=numeric_method)
        else:
            num_corr = data.corr(method=numeric_method)
        if verbose:
            LOGGER.info(
                "Matriz de correlação %s calculada para %d variáveis numéricas (engine=%s)",
                numeric_method,
                len(num_cols),
                corr_engine,
            )
    else:  # numeric_method == "cramer"
        num_corr = pd.DataFrame()
        if cramer and cat_cols:
            numeric_method = "cramer"

    if cramer and cat_cols:
        data_cat = df_work[cat_cols].copy()
        if adaptive_sampling:
            data_cat = adaptive_sampling(
                data_cat,
                stratify_col=target_col,
                date_cols=date_col,
            )
        cat_corr = _cramers_v_matrix(data_cat)
        if verbose:
            LOGGER.info(
                "Matriz de correlação Cramér‑V calculada para %d variáveis categóricas",
                len(cat_cols),
            )

    if not num_cols and cat_corr is not None:
        return cat_corr

    if num_cols:
        cols = num_cols + (cat_cols if cat_corr is not None else [])
        corr = pd.DataFrame(np.nan, index=cols, columns=cols)
        corr.loc[num_cols, num_cols] = num_corr
        if cat_corr is not None:
            corr.loc[cat_cols, cat_cols] = cat_corr
        np.fill_diagonal(corr.values, 1.0)
        return corr

    raise ValueError("Método inválido: %s" % method)


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------


def plot_corr_heatmap(
    corr: pd.DataFrame,
    *,
    title: str | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    cmap: matplotlib.colors.Colormap = sns.diverging_palette(240, 10, as_cmap=True),
    mask_upper: bool = True,
    base_figsize: float = 0.45,
    min_size: int = 6,
    max_size: int = 20,
    highlight_labels: bool = False,
    corr_threshold: float = 0.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plota *heat‑map* da matriz de correlação.

    Parameters
    ----------
    corr : pandas.DataFrame
        Matriz quadrada de correlação.
    title : str | None
        Título do gráfico.
    annot, fmt : parâmetros do ``seaborn.heatmap``.
    cmap : str
        Paleta de cores.
    mask_upper : bool, default True
        Se ``True``, mascara o triângulo superior.
    highlight_labels : bool, default False
        Quando ``True`` adiciona rótulos apenas às células com
        ``|corr| >= corr_threshold``. Útil para matrizes grandes.
    corr_threshold : float
        Limiar utilizado em ``highlight_labels``.
    base_figsize, min_size, max_size : float/int
        Encaminhados para ``figsize_from_matrix``.
    ax : matplotlib.axes.Axes | None
        Axes existente; se ``None`` cria-se figura nova.

    Returns
    -------
    matplotlib.axes.Axes
        Objeto Axes contendo o *heat‑map*.
    """
    corr = corr.fillna(0)
    n_feat = len(corr)
    figsize = figsize_from_matrix(
        n_feat, base=base_figsize, min_size=min_size, max_size=max_size
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    hm = sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        square=True,
        linewidths=0.5,
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={"shrink": 0.8},
    )

    if annot:
        data = corr.to_numpy()
        mesh = ax.collections[0]
        norm = mesh.norm
        cmap_obj = mesh.cmap
        for k, (text, val) in enumerate(zip(ax.texts, data.flatten())):
            i, j = divmod(k, len(corr))
            if i == j:
                text.set_visible(False)
                continue
            rgb = cmap_obj(norm(val))[:3]
            text.set_color(_pick_text_color(rgb))

    if highlight_labels:
        thr = abs(corr_threshold)
        for i, row in enumerate(corr.index):
            for j, col in enumerate(corr.columns):
                if i == j:
                    continue
                val = corr.iloc[i, j]
                if abs(val) >= thr:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        if mplcursors is not None:  # pragma: no cover - apenas visual
            cursor = mplcursors.cursor(ax.collections[0], hover=True)

            @cursor.connect("add")
            def _(sel):
                i, j = map(int, sel.index)
                var1 = corr.index[i]
                var2 = corr.columns[j]
                sel.annotation.set_text(f"{var1} × {var2} = {corr.iloc[i, j]:.2f}")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if title:
        ax.set_title(title)
    return ax
