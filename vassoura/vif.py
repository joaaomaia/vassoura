from __future__ import annotations

"""Vassoura – Variance Inflation Factor (VIF)
=========================================

Ferramentas para cálculo do *Variance Inflation Factor* e remoção
iterativa de variáveis com VIF elevado, respeitando lista de colunas a
preservar e opção de incluir o *target* no *DataFrame* analisado.
"""

import logging
import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from .utils import adaptive_sampling, parse_verbose, search_dtypes, woe_encode

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:  # pragma: no cover
    variance_inflation_factor = None  # type: ignore

__all__ = [
    "compute_vif",
    "remove_high_vif",
]

LOGGER = logging.getLogger(__name__)

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
    engine: str = "pandas",
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    date_col: Optional[List[str]] = None,
    verbose: str | bool = "basic",
    verbose_types: bool | None = None,
    adaptive_sampling: bool = False,
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
    limite_categorico, force_categorical, remove_ids, id_patterns, date_col,
    verbose, verbose_types
        Encaminhados para ``search_dtypes``.
    engine : {"pandas", "dask", "polars"}
        Backend utilizado para acelerar o cálculo quando disponível.
    verbose : str | bool
        ``"none"``, ``"basic"`` ou ``"full"``.
    adaptive_sampling : bool
        Se ``True``, amostra dados (até 50k linhas) antes do cálculo.

        Returns
        -------
        pandas.DataFrame
            DataFrame com colunas ``variable`` e ``vif`` ordenado
    decrescentemente.
    """
    verbose, verbose_types = parse_verbose(verbose, verbose_types)

    # Remove target se necessário
    drop_target = target_col and not include_target
    df_work = (
        df.drop(columns=[target_col], errors="ignore") if drop_target else df.copy()
    )

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

    if cat_cols:
        target_series = None
        if target_col and target_col in df.columns:
            target_series = df[target_col]
            if target_series.dropna().nunique() == 2 and set(target_series.dropna().unique()) != {0, 1}:
                mapping = {val: i for i, val in enumerate(sorted(target_series.dropna().unique()))}
                target_series = target_series.map(mapping)
        if target_series is not None and target_series.dropna().nunique() == 2:
            try:
                tmp = df_work[cat_cols].fillna("__MISSING__")
                df_work[cat_cols] = woe_encode(tmp, target_series, cols=cat_cols)[cat_cols]
            except Exception:
                df_work[cat_cols] = df_work[cat_cols].apply(lambda s: pd.factorize(s.fillna("__MISSING__"))[0])
        else:
            df_work[cat_cols] = df_work[cat_cols].apply(lambda s: pd.factorize(s.fillna("__MISSING__"))[0])
        num_cols = num_cols + cat_cols

    if not num_cols:
        raise ValueError("Nenhuma coluna numérica identificada para cálculo de VIF")

    data = df_work[num_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    rows_before = len(data)
    data = data.dropna()
    rows_after = len(data)
    if verbose and rows_after < rows_before:
        LOGGER.info(
            "Desconsiderando %d linha(s) com NaN/inf para cálculo de VIF",
            rows_before - rows_after,
        )
    if adaptive_sampling:
        data = adaptive_sampling(
            data,
            stratify_col=target_col,
            date_cols=date_col,
        )

    if engine == "dask":
        try:
            import dask.dataframe as dd
        except ImportError as exc:
            raise ImportError("engine='dask' requer dask instalado") from exc
        X = dd.from_pandas(data, npartitions=4).to_dask_array(lengths=True).compute()
    elif engine == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError("engine='polars' requer polars instalado") from exc
        X = pl.from_pandas(data).to_numpy()
    else:
        X = data.values

    if variance_inflation_factor is not None:
        # statsmodels dispara RuntimeWarning quando existe multicolinearidade
        # perfeita e o denominador do VIF se torna zero. Esses avisos poluem
        # a saída dos testes; portanto suprimimos apenas nessa chamada.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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
    """Remove iterativamente variáveis com VIF acima do limiar.

    Mantém intactas quaisquer colunas listadas em ``keep_cols``.
    A remoção pode ser fracionada em ``vif_n_steps`` etapas, recomputando
    o VIF a cada rodada.

    Returns
    -------
    df_clean : pandas.DataFrame
        DataFrame após remoção.
    dropped : list[str]
        Colunas removidas.
    final_vif : pandas.DataFrame
        VIF das variáveis remanescentes.
    engine : {"pandas", "dask", "polars"}
        Backend a ser utilizado nas chamadas de ``compute_vif``.
    verbose : str | bool
        ``"none"``, ``"basic"`` ou ``"full"``.
    adaptive_sampling : bool
        Ativa amostragem adaptativa em ``compute_vif``.
    date_col : list[str] | None
        Colunas convertidas para ``datetime`` antes da detecção de tipos.
    verbose_types : bool
        Exibe logs detalhados de classificação de tipos.
    """
    verbose, verbose_types = parse_verbose(verbose, verbose_types)
    keep_cols = keep_cols or []
    df_iter = df.copy()
    dropped: List[str] = []

    if vif_n_steps < 1:
        raise ValueError("vif_n_steps deve ser >= 1")

    for iteration in range(max_iter):
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

        vif_high = vif_df[
            (vif_df["vif"] > vif_threshold) & (~vif_df["variable"].isin(keep_cols))
        ]
        if vif_high.empty:
            if verbose:
                LOGGER.info(
                    "Iteração %d: nenhum VIF > %.2f restante",
                    iteration + 1,
                    vif_threshold,
                )
            break

        if verbose and iteration == 0:
            LOGGER.info(
                "%d variáveis acima do limiar inicial de VIF",
                len(vif_high),
            )

        step_limit = math.ceil(len(vif_high) / vif_n_steps)
        removed_this_iter = 0
        for _, row in vif_high.sort_values("vif", ascending=False).iterrows():
            var = row["variable"]
            df_iter = df_iter.drop(columns=[var])
            dropped.append(var)
            removed_this_iter += 1
            if verbose:
                LOGGER.info(
                    "Iteração %d: removendo '%s' (VIF=%.2f)",
                    iteration + 1,
                    var,
                    row["vif"],
                )
            if removed_this_iter >= step_limit:
                break
        if verbose:
            LOGGER.info(
                "Iteração %d: %d variáveis removidas", iteration + 1, removed_this_iter
            )
    else:
        LOGGER.warning("Número máximo de iterações (%d) atingido", max_iter)

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
