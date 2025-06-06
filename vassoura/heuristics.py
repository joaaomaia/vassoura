"""heuristics.py – Pluggable feature‑selection rules for Vassoura.

Cada heurística deve seguir a *assinatura*:

    def heuristic_name(df: pd.DataFrame, **kwargs) -> dict:
        'Remove columns based on rule X'
        return {
            "removed": list[str],      # colunas dropadas
            "artefacts": Any,          # intermed. p/ relatório (matrizes, figs...)
            "meta": dict[str, Any],    # info resumida p/ audit trail
        }

**NOTA**
-----
O módulo não conhece `Vassoura`. Ele deve ser *stateless* e puro.
A sessão garantirá cacheamento.
"""
from __future__ import annotations

from typing import Any, Dict, List

import warnings

import pandas as pd

# Dependências opcionais (import inside functions)

__all__ = [
    "importance",
    "graph_cut",
    "iv",
]


# --------------------------------------------------------------------- #
# IV: remove variáveis com Information Value abaixo do threshold        #
# --------------------------------------------------------------------- #

def iv(
    df: pd.DataFrame,
    target_col: str,
    *,
    threshold: float = 0.02,
    bins: int = 10,
    keep_cols: List[str] | None = None,
) -> Dict[str, Any]:
    from numpy import log as _ln  # local import p/ velocidade

    keep_cols = set(keep_cols or [])
    target = df[target_col]
    removed, iv_scores = [], {}

    for col in df.columns:
        if col == target_col or col in keep_cols:
            continue
        s = df[col]
        if s.dtype.kind in "bifc":  # numérico
            try:
                binned = pd.qcut(s, q=bins, duplicates="drop")
            except ValueError:
                continue  # série constante
        else:
            binned = s.astype("category")
        tab = pd.crosstab(binned, target)
        if tab.shape[1] != 2:
            continue  # target não binário
        tab = tab.rename(columns={0: "good", 1: "bad"}).replace(0, 0.5)
        dist_good = tab["good"] / tab["good"].sum()
        dist_bad = tab["bad"] / tab["bad"].sum()
        woe = _ln(dist_good / dist_bad)
        iv_val = ((dist_good - dist_bad) * woe).sum()
        iv_scores[col] = iv_val
        if iv_val < threshold:
            removed.append(col)

    return {
        "removed": removed,
        "artefacts": pd.Series(iv_scores, name="iv"),
        "meta": {"threshold": threshold},
    }


# --------------------------------------------------------------------- #
# Importance: XGBoost/LightGBM SHAP gain ranking                        #
# --------------------------------------------------------------------- #

def importance(
    df: pd.DataFrame,
    target_col: str,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    keep_cols: List[str] | None = None,
    drop_lowest: float | int = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Treina XGBoost (ou LightGBM se instalado) rápido e remove features
    de baixa importância.

    *Se* `drop_lowest` < 1 → tratado como quantil (ex.: 0.2 → dropar 20%);
    caso contrário, remover as `drop_lowest` piores variáveis.
    """
    try:
        from xgboost import XGBClassifier
        import shap
    except ImportError:  # pragma: no cover
        warnings.warn("importance heuristic skipped – xgboost/shap not installed")
        return {"removed": [], "artefacts": None, "meta": {}}

    keep_cols = set(keep_cols or [])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        eval_metric="logloss",
        random_state=random_state,
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    gain = pd.Series(shap_values.std(axis=0), index=X.columns, name="shap_gain")

    if drop_lowest < 1:
        cutoff = gain.quantile(drop_lowest)
        removed = gain[gain <= cutoff].index.tolist()
    else:
        removed = gain.sort_values().head(int(drop_lowest)).index.tolist()

    removed = [c for c in removed if c not in keep_cols]

    return {
        "removed": removed,
        "artefacts": gain,
        "meta": {"drop_lowest": drop_lowest},
    }


# --------------------------------------------------------------------- #
# Graph‑cut: mínimo conjunto de vértices em grafo de correlações        #
# --------------------------------------------------------------------- #

def graph_cut(
    df: pd.DataFrame,
    *,
    corr_threshold: float = 0.9,
    keep_cols: List[str] | None = None,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Constrói grafo onde arestas unem pares |corr| > corr_threshold e resolve
    minimum vertex cover para quebrar todas as arestas com o menor nº
    possível de vértices (features).
    """
    try:
        import networkx as nx
        import numpy as np
    except ImportError:
        warnings.warn("graph_cut heuristic skipped – networkx not installed")
        return {"removed": [], "artefacts": None, "meta": {}}

    keep_cols = set(keep_cols or [])
    corr = df.corr(method=method).abs()
    np.fill_diagonal(corr.values, 0)
    edges = [
        (i, j)
        for i in corr.columns
        for j in corr.columns
        if corr.loc[i, j] > corr_threshold and i < j
    ]

    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    G.add_edges_from(edges)

    # ``min_vertex_cover`` was removed in newer NetworkX versions (>=3.0).
    # ``min_weighted_vertex_cover`` is the replacement.  For compatibility with
    # older versions we try ``min_vertex_cover`` first and fall back to the
    # weighted variant if needed.
    approx = nx.algorithms.approximation
    if hasattr(approx, "min_vertex_cover"):
        cover = approx.min_vertex_cover(G)
    else:
        cover = approx.min_weighted_vertex_cover(G)
    removed = [v for v in cover if v not in keep_cols]

    return {
        "removed": removed,
        "artefacts": G.subgraph(cover).copy(),
        "meta": {"corr_threshold": corr_threshold, "method": method},
    }

