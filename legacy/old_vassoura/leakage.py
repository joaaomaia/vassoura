from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

__all__ = ["target_leakage"]

logger = logging.getLogger(__name__)


def target_leakage(
    df: pd.DataFrame,
    target_col: str,
    *,
    threshold: float = 0.80,
    method: str = "spearman",
    keep_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect features strongly associated with the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset completo.
    target_col : str
        Coluna alvo (0/1 ou numérica).
    threshold : float, default 0.80
        Correlação mínima absoluta para sinalizar vazamento.
    method : str, default "spearman"
        Método de correlação usado para colunas numéricas.
    keep_cols : list[str] | None
        Colunas protegidas que jamais serão removidas.
    id_cols, date_cols : list[str] | None
        Colunas de identificação ou data para ignorar.
    """
    keep_cols = set(keep_cols or [])
    ignore = set(id_cols or []) | set(date_cols or []) | {target_col}

    df_work = df.dropna(subset=[target_col])
    target = df_work[target_col]

    corr_vals: Dict[str, float] = {}
    removed: List[str] = []

    for col in df_work.columns:
        if col in ignore:
            continue
        s = df_work[col]
        if s.isna().all() or s.nunique(dropna=False) <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s):
            corr = s.corr(target, method=method)
        else:
            corr = s.astype("category").cat.codes.corr(target, method=method)
        if pd.isna(corr):
            continue
        val = abs(float(corr))
        corr_vals[col] = val
        if val >= threshold and col not in keep_cols:
            removed.append(col)
            logger.warning("Potential target leakage: %s corr=%.3f", col, val)

    flagged = pd.Series({k: v for k, v in corr_vals.items() if v >= threshold})
    flagged = flagged.sort_values(ascending=False)

    return {
        "removed": removed,
        "artefacts": flagged,
        "meta": {"threshold": threshold, "method": method},
    }
