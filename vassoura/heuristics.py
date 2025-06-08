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

import os
import warnings

import pandas as pd
import numpy as np

# Dependências opcionais (import inside functions)

__all__ = [
    "importance",
    "graph_cut",
    "iv",
    "variance",
    "psi_stability",
    "ks_separation",
    "perm_importance_lgbm",
    "partial_corr_cluster",
    "drift_vs_target_leakage",
]

os.environ.setdefault("LIGHTGBM_DISABLE_STDERR_REDIRECT", "1")
warnings.filterwarnings("ignore", message="No further splits with positive gain")
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed",
)


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
    sample_weight: pd.Series | None = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    keep_cols: list[str] | None = None,
    drop_lowest: float | int = 0.2,
    random_state: int = 42,
) -> dict[str, any]:
    """
    Treina XGBoost rápido, remove features de baixa importância (shap_gain)
    e trata automaticamente colunas não numéricas com WOE encoding no fallback.

    - Detecta colunas object e PeriodDtype → converte para category.
    - Se XGBClassifier aceitar `enable_categorical`, usa.
    - Caso contrário, tenta usar WOEEncoder para cat_cols; se não disponível,
      faz one-hot encoding.
    - Mantém sample_weight automático via compute_sample_weight (balanced).
    """
    try:
        from xgboost import XGBClassifier
        import shap
    except ImportError:
        warnings.warn("importance skipped – xgboost/shap not installed")
        return {"removed": [], "artefacts": None, "meta": {}}

    try:
        from sklearn.utils.class_weight import compute_sample_weight
    except ImportError:
        compute_sample_weight = None
        warnings.warn("scikit-learn missing – sample_weight automático não disponível")

    # tenta importar WOEEncoder
    try:
        from category_encoders.woe import WOEEncoder
    except ImportError:
        WOEEncoder = None
        warnings.warn(
            "category_encoders not installed – fallback para one-hot encoding em categoricals"
        )

    keep_cols = set(keep_cols or [])
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    # 1) Sample weights automáticos
    if sample_weight is None and compute_sample_weight is not None:
        try:
            sample_weight = compute_sample_weight("balanced", y)
        except Exception as e:
            warnings.warn(f"Não foi possível calcular sample_weight: {e}")
            sample_weight = None

    # 2) Detecta colunas object ou PeriodDtype
    cat_cols = [
        col for col in X.columns
        if X[col].dtype == object or is_period_dtype(X[col].dtype)
    ]
    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype("category")

    # 3) Configura XGBClassifier com fallback para WOE / one-hot
    xgb_kwargs = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        # usa categoricals nativos, se disponível
        model = XGBClassifier(**xgb_kwargs, enable_categorical=True)
    except TypeError:
        # fallback: WOEEncoder ou one-hot
        if cat_cols:
            if WOEEncoder is not None:
                encoder = WOEEncoder(cols=cat_cols)
                X = encoder.fit_transform(X, y)
            else:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        model = XGBClassifier(**xgb_kwargs)

    # 4) Treina e calcula SHAP gain
    model.fit(X, y, sample_weight=sample_weight)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    gain = pd.Series(shap_values.std(axis=0), index=X.columns, name="shap_gain")

    # 5) Seleção de features a remover
    if drop_lowest < 1:
        cutoff = gain.quantile(drop_lowest)
        removed = gain[gain <= cutoff].index.tolist()
    else:
        removed = gain.sort_values().head(int(drop_lowest)).index.tolist()
    removed = [c for c in removed if c not in keep_cols]

    return {
        "removed": removed,
        "artefacts": gain,
        "meta": {
            "drop_lowest": drop_lowest,
            "n_removed": len(removed),
            "n_features": X.shape[1],
            "sample_weight_used": sample_weight is not None,
        },
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


# --------------------------------------------------------------------- #
# Variance: low variance or dominant category removal                    #
# --------------------------------------------------------------------- #

def variance(
    df: pd.DataFrame,
    *,
    var_threshold: float = 1e-4,
    dom_threshold: float = 0.95,
    min_nonnull: int = 30,
    keep_cols: List[str] | None = None,
) -> Dict[str, Any]:
    """Identify and remove columns with very low variance or dominant class."""

    import numpy as np

    keep = set(keep_cols or [])
    metrics = {}
    removed: List[str] = []

    for col in df.columns:
        s = df[col]
        valid = s.dropna()
        if valid.size < min_nonnull:
            metrics[col] = np.nan
            continue

        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(valid, errors="coerce").replace([np.inf, -np.inf], np.nan)
            var = float(num.var())
            metrics[col] = var
            if var < var_threshold and col not in keep:
                removed.append(col)
        else:
            vc = valid.value_counts(normalize=True)
            p_major = float(vc.iloc[0]) if not vc.empty else np.nan
            metrics[col] = p_major
            if (p_major >= dom_threshold or valid.nunique() <= 1) and col not in keep:
                removed.append(col)

    return {
        "removed": removed,
        "artefacts": pd.Series(metrics, name="variance_metric"),
        "meta": {
            "var_threshold": var_threshold,
            "dom_threshold": dom_threshold,
            "min_nonnull": min_nonnull,
        },
    }


# --------------------------------------------------------------------- #
# PSI Stability: population stability index between time windows         #
# --------------------------------------------------------------------- #

def psi_stability(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str | None = None,
    window: tuple[str, str],
    bins: int = 10,
    psi_thr: float = 0.25,
    keep_cols: List[str] | None = None,
) -> Dict[str, Any]:
    """Calcula o *Population Stability Index* por coluna.

    Parameters
    ----------
    df : pandas.DataFrame
        Conjunto completo de dados.
    date_col : str
        Coluna base para o recorte temporal.
    target_col : str, optional
        Coluna alvo a ser ignorada no cálculo.
    window : tuple[str, str]
        Par de datas (referência, out-of-time) em formato ISO ou datetime.
    bins : int
        Número de quantis para binarizar numéricos.
    psi_thr : float
        Limiar acima do qual a coluna é considerada instável.
    keep_cols : list[str] | None
        Colunas protegidas contra remoção.
    """
    keep_cols = set(keep_cols or [])
    if date_col not in df.columns:
        return {"removed": [], "artefacts": None, "meta": {"window": window}}

    ref, oot = window
    sdate = pd.to_datetime(df[date_col].astype(str))
    df_ref = df[sdate.astype(str).str.startswith(str(ref))]
    df_oot = df[sdate.astype(str).str.startswith(str(oot))]

    if df_ref.empty or df_oot.empty:
        warnings.warn("psi_stability skipped – janelas incompletas")
        return {"removed": [], "artefacts": None, "meta": {"window": window}}

    psi_vals = {}
    removed: List[str] = []
    for col in df.columns:
        if col in {date_col, target_col} or col in keep_cols:
            continue
        s1 = df_ref[col]
        s2 = df_oot[col]
        if s1.dtype.kind in "bifc" and s1.nunique() > 1:
            try:
                b1 = pd.qcut(s1, q=bins, duplicates="drop")
                b2 = pd.qcut(s2, q=bins, duplicates="drop")
            except ValueError:
                continue
        else:
            b1 = s1.astype("category")
            b2 = s2.astype("category")
        p1 = b1.value_counts(normalize=True)
        p2 = b2.value_counts(normalize=True)
        all_bins = p1.index.union(p2.index)
        p1 = p1.reindex(all_bins, fill_value=1e-6)
        p2 = p2.reindex(all_bins, fill_value=1e-6)
        psi = ((p1 - p2) * np.log(p1 / p2)).sum()
        psi_vals[col] = psi
        if psi > psi_thr and col not in keep_cols:
            removed.append(col)

    return {
        "removed": removed,
        "artefacts": pd.Series(psi_vals, name="psi"),
        "meta": {"window": window, "psi_thr": psi_thr},
    }


# --------------------------------------------------------------------- #
# KS Separation                                                          #
# --------------------------------------------------------------------- #

def ks_separation(
    df: pd.DataFrame,
    *,
    target_col: str,
    ks_thr: float = 0.05,
    n_bins: int = 10,
    keep_cols: List[str] | None = None,
) -> Dict[str, Any]:
    """Calcula o KS-statistic por coluna."""

    keep_cols = set(keep_cols or [])
    target = df[target_col]
    if target.nunique(dropna=False) != 2:
        warnings.warn("ks_separation skipped – target não binário")
        return {"removed": [], "artefacts": None, "meta": {}}

    ks_vals = {}
    removed: List[str] = []
    for col in df.columns:
        if col == target_col or col in keep_cols:
            continue
        s = df[col]
        if s.dtype.kind in "bifc" and s.nunique() > 1:
            try:
                b = pd.qcut(s, q=n_bins, duplicates="drop")
            except ValueError:
                continue
        else:
            b = s.astype("category")
        tab = pd.crosstab(b, target)
        if tab.shape[1] != 2:
            continue
        cdf_good = tab[0].cumsum() / tab[0].sum()
        cdf_bad = tab[1].cumsum() / tab[1].sum()
        ks = (cdf_good - cdf_bad).abs().max()
        ks_vals[col] = float(ks)
        if ks < ks_thr and col not in keep_cols:
            removed.append(col)

    return {
        "removed": removed,
        "artefacts": pd.Series(ks_vals, name="ks"),
        "meta": {"ks_thr": ks_thr},
    }


# --------------------------------------------------------------------- #
# Permutation importance with LightGBM                                  #
# --------------------------------------------------------------------- #

def perm_importance_lgbm(
    df: pd.DataFrame,
    *,
    target_col: str,
    metric: str = "auc",
    n_estimators: int = 300,
    drop_lowest: float | int = 0.2,
    random_state: int = 42,
    keep_cols: List[str] | None = None,
) -> Dict[str, Any]:
    """Avalia importância de permutação via LightGBM."""

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.inspection import permutation_importance
    except Exception:  # pragma: no cover
        warnings.warn("perm_importance_lgbm skipped – lightgbm não instalado")
        return {"removed": [], "artefacts": None, "meta": {}}

    keep_cols = set(keep_cols or [])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.nunique() == 2:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    model.fit(X, y)

    score = metric
    if metric == "auc":
        score = "roc_auc"
    result = permutation_importance(
        model, X, y, scoring=score, random_state=random_state, n_repeats=5
    )
    imp = pd.Series(result.importances_mean, index=X.columns, name="perm_imp")

    if drop_lowest < 1:
        cutoff = imp.quantile(drop_lowest)
        removed = imp[imp <= cutoff].index.tolist()
    else:
        removed = imp.sort_values().head(int(drop_lowest)).index.tolist()
    removed = [c for c in removed if c not in keep_cols]

    return {
        "removed": removed,
        "artefacts": imp,
        "meta": {"drop_lowest": drop_lowest},
    }


# --------------------------------------------------------------------- #
# Partial correlation cluster                                            #
# --------------------------------------------------------------------- #

def partial_corr_cluster(
    df: pd.DataFrame,
    *,
    corr_thr: float = 0.6,
    keep_cols: List[str] | None = None,
    method: str = "pearson",
) -> Dict[str, Any]:
    """Agrupa colunas correlacionadas via correlação parcial."""

    try:
        import networkx as nx
        import numpy as np
    except ImportError:  # pragma: no cover
        warnings.warn("partial_corr_cluster skipped – networkx não instalado")
        return {"removed": [], "artefacts": None, "meta": {}}

    keep_cols = set(keep_cols or [])
    corr = df.corr(method=method)
    prec = np.linalg.pinv(corr)
    pcorr = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
    np.fill_diagonal(pcorr, 1)
    pc_df = pd.DataFrame(pcorr, index=corr.index, columns=corr.columns)

    edges = [
        (i, j)
        for i in pc_df.columns
        for j in pc_df.columns
        if abs(pc_df.loc[i, j]) > corr_thr and i < j
    ]

    G = nx.Graph()
    G.add_nodes_from(pc_df.columns)
    G.add_edges_from(edges)
    approx = nx.algorithms.approximation
    if hasattr(approx, "min_vertex_cover"):
        cover = approx.min_vertex_cover(G)
    else:
        cover = approx.min_weighted_vertex_cover(G)
    removed = [v for v in cover if v not in keep_cols]

    return {
        "removed": removed,
        "artefacts": G.subgraph(cover).copy(),
        "meta": {"corr_thr": corr_thr, "method": method},
    }


# --------------------------------------------------------------------- #
# Drift vs Target Leakage                                                #
# --------------------------------------------------------------------- #

def drift_vs_target_leakage(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    drift_thr: float = 0.3,
    leak_thr: float = 0.5,
    keep_cols: List[str] | None = None,
) -> Dict[str, Any]:
    """Detecta variáveis com forte drift temporal e alta correlação com o target."""

    keep_cols = set(keep_cols or [])
    if date_col not in df.columns:
        return {"removed": [], "artefacts": None, "meta": {}}

    date_ord = pd.to_datetime(df[date_col]).view("int64")
    target = df[target_col]

    metrics = {}
    removed: List[str] = []

    for col in df.columns:
        if col in {date_col, target_col} or col in keep_cols:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            corr_date = abs(pd.Series(s).corr(pd.Series(date_ord)))
            corr_target = abs(pd.Series(s).corr(pd.Series(target)))
        else:
            # categórica: usa Spearman para data e IV p/ target
            codes = s.astype("category").cat.codes
            corr_date = abs(pd.Series(codes).corr(pd.Series(date_ord), method="spearman"))
            # informação de valor (IV)
            tab = pd.crosstab(s, target)
            if tab.shape[1] != 2:
                corr_target = 0.0
            else:
                dist_good = tab[0] / tab[0].sum()
                dist_bad = tab[1] / tab[1].sum()
                woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
                corr_target = float(((dist_good - dist_bad) * woe).sum())

        metrics[col] = {"corr_date": corr_date, "corr_target": corr_target}
        if corr_date > drift_thr and corr_target > leak_thr and col not in keep_cols:
            removed.append(col)

    artefacts = pd.DataFrame(metrics).T
    return {
        "removed": removed,
        "artefacts": artefacts,
        "meta": {"drift_thr": drift_thr, "leak_thr": leak_thr},
    }


