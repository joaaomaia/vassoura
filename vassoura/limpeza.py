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
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .correlacao import compute_corr_matrix
from .utils import search_dtypes, suggest_corr_method
from .vif import remove_high_vif, compute_vif

__all__ = [
    "clean",
]

LOGGER = logging.getLogger("vassoura")

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _select_var_to_drop(
    corr_df: pd.DataFrame,
    pair: Tuple[str, str],
    keep_cols: List[str],
) -> str:
    """Heurística para decidir qual variável descartar de um par."""
    var1, var2 = pair
    # Preserva explicitamente keep_cols
    if var1 in keep_cols and var2 in keep_cols:
        return ""  # não pode remover nenhum
    if var1 in keep_cols:
        return var2
    if var2 in keep_cols:
        return var1
    # Caso comum: remove variável com maior correlação média absoluta
    mean1 = corr_df[var1].abs().mean()
    mean2 = corr_df[var2].abs().mean()
    return var1 if mean1 >= mean2 else var2

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
    verbose: bool = True,
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
    limite_categorico, force_categorical, remove_ids, id_patterns
        Encaminhados para ``search_dtypes``.
    max_vif_iter : int
        Máximo de iterações do filtro VIF.
    verbose : bool
        Controla *logs*.

    Returns
    -------
    df_clean : pandas.DataFrame
    dropped_total : list[str]
    corr_matrix_final : pandas.DataFrame
    vif_final : pandas.DataFrame
    """
    keep_cols = set(keep_cols or [])
    if target_col and include_target:
        keep_cols.add(target_col)

    df_work = df.copy()
    dropped_overall: List[str] = []

    # ---------------------------------------------------------------------
    # 1) Remoção por correlação
    # ---------------------------------------------------------------------
    if corr_threshold and corr_threshold > 0:
        corr_matrix = compute_corr_matrix(
            df_work,
            method=corr_method,
            target_col=target_col,
            include_target=include_target,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )
        # Procurar pares com correlação alta
        upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
        pairs = upper_tri.stack().loc[lambda s: s.abs() > corr_threshold]

        if verbose:
            LOGGER.info("Encontrados %d pares com |corr| > %.2f", len(pairs), corr_threshold)
        for (var1, var2), corr_val in pairs.sort_values(key=lambda s: s.abs(), ascending=False).items():
            if var1 not in df_work.columns or var2 not in df_work.columns:
                continue  # uma delas já foi removida
            drop_var = _select_var_to_drop(corr_matrix, (var1, var2), list(keep_cols))
            if drop_var and drop_var in df_work.columns and drop_var not in keep_cols:
                df_work = df_work.drop(columns=[drop_var])
                dropped_overall.append(drop_var)
                if verbose:
                    LOGGER.info("Removendo '%s' devido a |corr|=%.3f com '%s'", drop_var, abs(corr_val), var1 if drop_var == var2 else var2)
        # Corr matriz pós remoções
        corr_matrix_final = df_work[corr_matrix.columns.intersection(df_work.columns)].corr(method=suggest_corr_method(*search_dtypes(df_work, target_col=None, limite_categorico=limite_categorico, force_categorical=force_categorical, remove_ids=remove_ids, id_patterns=id_patterns, verbose=False)))
    else:
        corr_matrix_final = pd.DataFrame()

    # ---------------------------------------------------------------------
    # 2) Remoção por VIF
    # ---------------------------------------------------------------------
    if vif_threshold and vif_threshold < np.inf:
        df_work, dropped_vif, vif_final = remove_high_vif(
            df_work,
            vif_threshold=vif_threshold,
            target_col=target_col,
            include_target=include_target,
            keep_cols=list(keep_cols),
            max_iter=max_vif_iter,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )
        dropped_overall.extend(dropped_vif)
    else:
        vif_final = compute_vif(
            df_work,
            target_col=target_col,
            include_target=include_target,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )

    return df_work, dropped_overall, corr_matrix_final, vif_final
