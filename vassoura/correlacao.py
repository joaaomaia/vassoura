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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from .utils import figsize_from_matrix, search_dtypes, suggest_corr_method

__all__ = [
    "compute_corr_matrix",
    "plot_corr_heatmap",
]

LOGGER = logging.getLogger("vassoura")

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
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    verbose: bool = True,
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
    limite_categorico, force_categorical, remove_ids, id_patterns, verbose
        Encaminhados para ``search_dtypes``.

    Returns
    -------
    pandas.DataFrame
        Matriz de correlação.
    """
    # Filtra colunas segundo parâmetros
    drop_target = (target_col and not include_target)
    df_work = df.drop(columns=[target_col], errors="ignore") if drop_target else df.copy()

    # Identifica tipos
    num_cols, cat_cols = search_dtypes(
        df_work,
        target_col=None,  # já removido se for o caso
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    # Escolha automática se necessário
    if method == "auto":
        method = suggest_corr_method(num_cols, cat_cols)
        if verbose:
            LOGGER.info("Método de correlação sugerido: %s", method)

    if method in ("pearson", "spearman"):
        if not num_cols:
            raise ValueError("Não há colunas numéricas para calcular correlação %s" % method)
        data = df_work[num_cols].copy()
        corr = data.corr(method=method)
        if verbose:
            LOGGER.info("Matriz de correlação %s calculada para %d variáveis numéricas", method, len(num_cols))
        return corr

    if method == "cramer":
        if not cat_cols:
            raise ValueError("Não há colunas categóricas para calcular Cramér‑V")
        data = df_work[cat_cols].copy()
        corr = _cramers_v_matrix(data)
        if verbose:
            LOGGER.info("Matriz de correlação Cramér‑V calculada para %d variáveis categóricas", len(cat_cols))
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
    cmap: str = "coolwarm",
    mask_upper: bool = True,
    base_figsize: float = 0.45,
    min_size: int = 6,
    max_size: int = 20,
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
    base_figsize, min_size, max_size : float/int
        Encaminhados para ``figsize_from_matrix``.
    ax : matplotlib.axes.Axes | None
        Axes existente; se ``None`` cria-se figura nova.

    Returns
    -------
    matplotlib.axes.Axes
        Objeto Axes contendo o *heat‑map*.
    """
    n_feat = len(corr)
    figsize = figsize_from_matrix(n_feat, base=base_figsize, min_size=min_size, max_size=max_size)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        square=True,
        linewidths=0.5,
        mask=mask,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if title:
        ax.set_title(title)
    return ax
