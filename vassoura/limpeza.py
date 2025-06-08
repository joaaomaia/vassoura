from __future__ import annotations

"""Vassoura – Limpeza de multicolinearidade
=======================================

Pipeline de alta‑nível que combina:

1. **Filtro por correlação** – remove variáveis altamente correlacionadas
   (|corr| > *corr_threshold*), preservando lista de colunas "intocáveis".
2. **Filtro por VIF** – aplica remoção iterativa baseada em Variance
   Inflation Factor (``vif_threshold``) via ``vassoura.vif``.

O objetivo é devolver um *DataFrame* enxuto e relatórios completos de
colunas descartadas, matriz de correlação final e VIF das variáveis
remanescentes.
"""
import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .correlacao import compute_corr_matrix
from .utils import parse_verbose
from .vif import compute_vif, remove_high_vif

__all__ = [
    "clean",
]

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def _select_var_to_drop(
    corr_df: pd.DataFrame,
    pair: Tuple[str, str],
    keep_cols: List[str],
    *,
    metric: str = "median",  # "mean", "median", "max"
    weight_keep: float = 1.5,  # pondera correlação com keep_cols
) -> str:
    """Decide qual variável descartar em um par altamente correlacionado.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Matriz de correlação (index e colunas = variáveis).
    pair : tuple(str, str)
        Par de variáveis correlacionadas (var1, var2).
    keep_cols : list[str]
        Lista de colunas prioritárias a serem preservadas.
    metric : str, optional
        Estatística usada na decisão: "mean", "median" ou "max".
    weight_keep : float, optional
        Fator multiplicativo aplicado às correlações contra colunas
        prioritárias.

    Returns
    -------
    str
        Nome da variável escolhida para remoção.
        Retorna "" caso ambas pertençam a keep_cols.
    """
    var1, var2 = pair

    # 1) Prioridade absoluta às colunas protegidas
    if var1 in keep_cols and var2 in keep_cols:
        return ""  # não remove nenhuma
    if var1 in keep_cols:
        return var2
    if var2 in keep_cols:
        return var1

    # 2) Define função de pontuação
    def _score(var: str) -> float:
        others = corr_df.columns.difference([var1, var2])
        vals = corr_df.loc[others, var].abs()

        # Pondera correlação com keep_cols
        if keep_cols:
            mask_keep = vals.index.isin(keep_cols)
            vals = vals.where(~mask_keep, vals * weight_keep)

        if metric == "median":
            return vals.median()
        if metric == "max":
            return vals.max()
        return vals.mean()

    return var1 if _score(var1) >= _score(var2) else var2


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def clean(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    include_target: bool = False,
    corr_threshold: float = 0.9,
    corr_method: str = "auto",
    vif_threshold: float = 10.0,
    keep_cols: Optional[List[str]] = None,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    max_vif_iter: int = 20,
    n_steps: int | None = None,
    vif_n_steps: int = 1,
    date_col: Optional[List[str]] = None,
    verbose: str | bool = "basic",
    verbose_types: bool | None = None,
    adaptive_sampling: bool = False,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    """Executa limpeza de multicolinearidade via correlação + VIF.

    Parameters
    ----------
    df : pandas.DataFrame
        Conjunto de dados original.
    target_col : str | None
        Nome da coluna *target*. Por padrão é excluída das análises.
    include_target : bool
        Se ``True``, mantém o *target* nos cálculos; ele nunca é removido.
    corr_threshold : float
        Parâmetro |corr| acima do qual pares entram na *grid* de remoção.
        Se ``None`` ou ``0``, ignora etapa de filtro por correlação.
    corr_method : {"auto", "pearson", "spearman", "cramer"}
        Método a ser enviado a ``compute_corr_matrix``.
    vif_threshold : float
        Limiar VIF. Se ``None`` ou ``np.inf`` ignora passo de VIF.
    keep_cols : list[str] | None
        Colunas que jamais devem ser removidas.
    limite_categorico, force_categorical, remove_ids, id_patterns, date_col,
    verbose, verbose_types
        Encaminhados para ``search_dtypes``.
    max_vif_iter : int
        Máximo de iterações do filtro VIF.
    n_steps : int | None
        Quantidade de etapas fracionadas para remoção por correlação.
        ``None`` mantém o comportamento tradicional (remoção completa).
    vif_n_steps : int
        Número de etapas para remoção por VIF. Por padrão ``1``.
    verbose : str | bool
        ``"none"``, ``"basic"`` ou ``"full"``.
    verbose_types : bool
        Se ``True``, logs detalhados de detecção de tipos.
    adaptive_sampling : bool
        Usa amostragem adaptativa em ``compute_corr_matrix`` e ``compute_vif``.

    Returns
    -------
    df_clean : pandas.DataFrame
    dropped_total : list[str]
    corr_matrix_final : pandas.DataFrame
    vif_final : pandas.DataFrame
    """
    verbose, verbose_types = parse_verbose(verbose, verbose_types)

    keep_cols = set(keep_cols or [])
    if target_col and include_target:
        keep_cols.add(target_col)

    df_work = df.copy()
    dropped_overall: List[str] = []
    dropped_corr: List[str] = []
    dropped_vif: List[str] = []

    # ---------------------------------------------------------------------
    # 1) Remoção por correlação
    # ---------------------------------------------------------------------
    if corr_threshold and corr_threshold > 0:
        iteration = 0
        while True:
            corr_matrix = compute_corr_matrix(
                df_work,
                method=corr_method,
                target_col=target_col,
                include_target=include_target,
                limite_categorico=limite_categorico,
                force_categorical=force_categorical,
                remove_ids=remove_ids,
                id_patterns=id_patterns,
                date_col=date_col,
                verbose=verbose,
                verbose_types=verbose_types,
                adaptive_sampling=adaptive_sampling,
            )
            upper_tri = corr_matrix.where(
                np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            )
            pairs = upper_tri.stack().loc[lambda s: s.abs() > corr_threshold]

            if iteration == 0:
                if verbose:
                    LOGGER.info(
                        "Encontrados %d pares com |corr| > %.2f",
                        len(pairs),
                        corr_threshold,
                    )
            else:
                if verbose:
                    LOGGER.info(
                        "Recontagem: %d pares acima do limiar",
                        len(pairs),
                    )

            if pairs.empty:
                break

            step_limit = (
                len(pairs) if n_steps is None else math.ceil(len(pairs) / n_steps)
            )
            removed_this_iter: List[str] = []
            for (var1, var2), corr_val in pairs.sort_values(
                key=lambda s: s.abs(), ascending=False
            ).items():
                if var1 not in df_work.columns or var2 not in df_work.columns:
                    continue
                drop_var = _select_var_to_drop(
                    corr_matrix, (var1, var2), list(keep_cols)
                )
                if (
                    drop_var
                    and drop_var in df_work.columns
                    and drop_var not in keep_cols
                ):
                    df_work = df_work.drop(columns=[drop_var])
                    dropped_overall.append(drop_var)
                    dropped_corr.append(drop_var)
                    removed_this_iter.append(drop_var)
                    if verbose:
                        LOGGER.info(
                            "Iteração %d: removendo '%s' devido a |corr|=%.3f "
                            "com '%s'",
                            iteration + 1,
                            drop_var,
                            abs(corr_val),
                            var1 if drop_var == var2 else var2,
                        )
                    if len(removed_this_iter) >= step_limit:
                        break
            if verbose:
                LOGGER.info(
                    "Iteração %d: %d variáveis removidas",
                    iteration + 1,
                    len(removed_this_iter),
                )
            iteration += 1

        corr_matrix_final = compute_corr_matrix(
            df_work,
            method=corr_method,
            target_col=target_col,
            include_target=include_target,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            date_col=date_col,
            verbose=verbose,
            verbose_types=verbose_types,
            adaptive_sampling=adaptive_sampling,
        )
    else:
        corr_matrix_final = pd.DataFrame()

    # ---------------------------------------------------------------------
    # 2) Remoção por VIF
    # ---------------------------------------------------------------------
    if vif_threshold and vif_threshold < np.inf:
        df_work, dropped_vif_list, vif_final = remove_high_vif(
            df_work,
            vif_threshold=vif_threshold,
            target_col=target_col,
            include_target=include_target,
            keep_cols=list(keep_cols),
            max_iter=max_vif_iter,
            vif_n_steps=vif_n_steps,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            date_col=date_col,
            verbose=verbose,
            verbose_types=verbose_types,
            adaptive_sampling=adaptive_sampling,
        )
        dropped_vif.extend(dropped_vif_list)
        dropped_overall.extend(dropped_vif_list)
    else:
        vif_final = compute_vif(
            df_work,
            target_col=target_col,
            include_target=include_target,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            date_col=date_col,
            verbose=verbose,
            verbose_types=verbose_types,
            adaptive_sampling=adaptive_sampling,
        )

    if verbose:
        LOGGER.info("Resumo final de remoções:")
        LOGGER.info("Método | Variáveis removidas | Total")
        LOGGER.info(
            "corr   | %s | %d",
            dropped_corr,
            len(dropped_corr),
        )
        LOGGER.info(
            "vif    | %s | %d",
            dropped_vif,
            len(dropped_vif),
        )

    return df_work, dropped_overall, corr_matrix_final, vif_final
