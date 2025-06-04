from __future__ import annotations
"""Vassoura – Variance Inflation Factor (VIF)
=========================================

Ferramentas para cálculo do *Variance Inflation Factor* e remoção
iterativa de variáveis com VIF elevado, respeitando lista de colunas a
preservar e opção de incluir o *target* no *DataFrame* analisado.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from .utils import search_dtypes

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:  # pragma: no cover
    variance_inflation_factor = None  # type: ignore

__all__ = [
    "compute_vif",
    "remove_high_vif",
]

LOGGER = logging.getLogger("vassoura")

# ---------------------------------------------------------------------------
# Funções internas auxiliares
# ---------------------------------------------------------------------------

def _compute_vif_np(x: np.ndarray) -> np.ndarray:
    """Calcula VIF usando operações numpy (fallback se Statsmodels ausente).

    VIF(i) = 1 / (1 - R_i^2), onde R_i^2 é o R² da regressão da coluna i
    contra todas as outras.
    """
    n_cols = x.shape[1]
    vif_vals = np.zeros(n_cols)
    # Adiciona intercepto
    X_const = np.column_stack([np.ones(x.shape[0]), x])
    for i in range(n_cols):
        y = x[:, i]
        X_others = np.delete(X_const, i + 1, axis=1)  # remove coluna i
        try:
            beta, *_ = np.linalg.lstsq(X_others, y, rcond=None)
            y_hat = X_others @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            vif_vals[i] = 1 / (1 - r2) if r2 < 1 else np.inf
        except LinAlgError:
            vif_vals[i] = np.inf
    return vif_vals

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_vif(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    include_target: bool = False,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calcula VIF para todas as colunas numéricas.

    Parameters
    ----------
    df : pandas.DataFrame
        Conjunto de dados completo.
    target_col : str | None
        Nome da coluna *target*.
    include_target : bool, default False
        Considera a coluna *target* no cálculo de VIF.
    limite_categorico, force_categorical, remove_ids, id_patterns
        Encaminhados para ``search_dtypes``.
    verbose : bool
        Exibe *logs*.

    Returns
    -------
    pandas.DataFrame
        DataFrame com colunas ``variable`` e ``vif`` ordenado
decrescentemente.
    """
    # Remove target se necessário
    drop_target = (target_col and not include_target)
    df_work = df.drop(columns=[target_col], errors="ignore") if drop_target else df.copy()

    num_cols, _ = search_dtypes(
        df_work,
        target_col=None,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    if not num_cols:
        raise ValueError("Nenhuma coluna numérica identificada para cálculo de VIF")

    X = df_work[num_cols].astype(float).values

    if variance_inflation_factor is not None:
        vif_vals = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    else:  # fallback numpy puro
        vif_vals = _compute_vif_np(X)

    vif_df = pd.DataFrame({"variable": num_cols, "vif": vif_vals})
    vif_df = vif_df.sort_values("vif", ascending=False).reset_index(drop=True)

    if verbose:
        LOGGER.info("VIF calculado para %d variáveis", len(num_cols))
    return vif_df


def remove_high_vif(
    df: pd.DataFrame,
    *,
    vif_threshold: float = 10.0,
    target_col: str | None = None,
    include_target: bool = False,
    keep_cols: Optional[List[str]] = None,
    max_iter: int = 20,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Remove iterativamente variáveis com VIF acima do limiar.

    Mantém intactas quaisquer colunas listadas em ``keep_cols``.

    Returns
    -------
    df_clean : pandas.DataFrame
        DataFrame após remoção.
    dropped : list[str]
        Colunas removidas.
    final_vif : pandas.DataFrame
        VIF das variáveis remanescentes.
    """
    keep_cols = keep_cols or []
    df_iter = df.copy()
    dropped: List[str] = []

    for iteration in range(max_iter):
        vif_df = compute_vif(
            df_iter,
            target_col=target_col,
            include_target=include_target,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )

        # Encontra maior VIF acima do limiar (ignorando forças)
        vif_high = vif_df[(vif_df["vif"] > vif_threshold) & (~vif_df["variable"].isin(keep_cols))]
        if vif_high.empty:
            if verbose:
                LOGGER.info("Iteração %d: nenhum VIF > %.2f restante", iteration + 1, vif_threshold)
            break
        # Remove variável com maior VIF
        worst_var = vif_high.iloc[0, 0]
        df_iter = df_iter.drop(columns=[worst_var])
        dropped.append(worst_var)
        if verbose:
            LOGGER.info("Iteração %d: removendo '%s' (VIF=%.2f)", iteration + 1, worst_var, vif_high.iloc[0, 1])
    else:
        LOGGER.warning("Número máximo de iterações (%d) atingido", max_iter)

    final_vif = compute_vif(
        df_iter,
        target_col=target_col,
        include_target=include_target,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    return df_iter, dropped, final_vif