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

import logging
import os
import warnings
from typing import Any, Dict, List
from itertools import combinations

import numpy as np
import pandas as pd

from .heuristics_boruta_multi_shap import BorutaMultiShap
from .scaler import DynamicScaler

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
    "boruta_multi_shap",
]

os.environ.setdefault("LIGHTGBM_DISABLE_STDERR_REDIRECT", "1")
warnings.filterwarnings("ignore", message="No further splits with positive gain")
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed",
)

logger = logging.getLogger(__name__)


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
    params: dict[str, Any] | None = None,
    scaler_args: dict[str, Any] | None = None,
    sample_weight: pd.Series | None = None,
    woe_cols: list[str] | None = None,
    keep_cols: list[str] | None = None,
) -> Dict[str, Any]:
    """Calcula importâncias de features comparadas a uma uniform noise feature."""

    import inspect

    from sklearn.base import clone
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_sample_weight

    params = params or {}
    scaler_args = scaler_args or {}
    keep_cols = set(keep_cols or [])

    # global settings
    sample_frac = params.pop("sample_frac", 0.7)
    random_state = params.pop("random_state", 42)
    models_cfg = params.pop("models", None)
    cv_folds = params.pop("cv_folds", 5)
    cv_type = params.pop("cv_type", "stratified")
    shuffle = params.pop("shuffle", True)
    approval_ratio_fold = params.pop("approval_ratio_fold", 0.7)

    lr_params = params
    lr_params.setdefault("max_iter", 500)

    rng = np.random.default_rng(random_state)

    X_full = df.drop(columns=[target_col])
    y_full = df[target_col]

    if sample_weight is None:
        try:
            sample_weight = compute_sample_weight("balanced", y_full)
        except Exception:
            sample_weight = None

    default_models: List[Dict[str, Any]] = []

    try:
        from sklearn.linear_model import LogisticRegression

        if y_full.nunique() == 2:
            lr_est = Pipeline(
                [
                    ("scaler", DynamicScaler(**scaler_args)),
                    ("lr", LogisticRegression(**lr_params)),
                ]
            )
            default_models.append({"name": "lr", "estimator": lr_est})
    except Exception as err:  # pragma: no cover
        logger.warning(
            "[Heuristic][importance] LogisticRegression failed: %s, bypassing",
            err,
        )

    try:
        from xgboost import XGBClassifier, XGBRegressor

        tree_cls = XGBClassifier if y_full.nunique() == 2 else XGBRegressor
        default_models.append(
            {
                "name": "xgb",
                "estimator": tree_cls(
                    n_estimators=50,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    min_child_weight=max(1, int(0.05 * len(df))),
                    n_jobs=-1,
                    eval_metric="logloss" if y_full.nunique() == 2 else None,
                    random_state=random_state,
                ),
            }
        )
    except Exception:  # pragma: no cover - optional
        warnings.warn("XGBoost não disponível para importance")

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        lgbm_cls = LGBMClassifier if y_full.nunique() == 2 else LGBMRegressor
        default_models.append(
            {
                "name": "lgbm",
                "estimator": lgbm_cls(
                    n_estimators=50,
                    max_depth=5,
                    min_child_samples=max(2, int(0.05 * len(df))),
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            }
        )
    except Exception:  # pragma: no cover - optional
        warnings.warn("LightGBM não disponível para importance")

    models_cfg = models_cfg or default_models

    # ------------------------------------------------------------------
    # Pré-processamento único para todos os modelos
    # ------------------------------------------------------------------
    idx = rng.choice(len(df), size=int(len(df) * sample_frac), replace=False)
    X = X_full.iloc[idx].copy()
    y = y_full.iloc[idx]
    sw = sample_weight[idx] if sample_weight is not None else None

    X["__noise_uniform__"] = rng.random(len(X))

    cat_cols = (
        woe_cols or X.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    if cat_cols:
        from .utils import woe_encode

        X = woe_encode(X, y, cols=cat_cols)
    X = X.fillna(0)

    scaler = DynamicScaler(**scaler_args)
    scaler.fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    importances_folds: Dict[str, List[pd.Series]] = {}
    noise_values_folds: Dict[str, List[float]] = {}
    models_without_weights: List[str] = []
    conv_info: Dict[str, bool] = {}

    for cfg in models_cfg:
        name = cfg.get("name", str(len(importances_folds)))
        importances_folds[name] = []
        noise_values_folds[name] = []
        conv_info[f"{name}_converged"] = True

    if cv_folds <= 1:
        splits = [(np.arange(len(X_scaled)), np.arange(len(X_scaled)))]
    else:
        if cv_type == "time_series":
            from sklearn.model_selection import TimeSeriesSplit

            splitter = TimeSeriesSplit(n_splits=cv_folds)
            splits = list(splitter.split(X_scaled))
        elif cv_type == "stratified" and y.nunique() <= len(y):
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(
                n_splits=cv_folds, shuffle=shuffle, random_state=random_state
            )
            splits = list(splitter.split(X_scaled, y))
        else:
            from sklearn.model_selection import KFold

            splitter = KFold(
                n_splits=cv_folds, shuffle=shuffle, random_state=random_state
            )
            splits = list(splitter.split(X_scaled))

    for train_idx, _ in splits:
        for cfg in models_cfg:
            name = cfg.get("name", str(len(importances_folds)))
            estimator = clone(cfg.get("estimator"))
            if cfg.get("hyperparams"):
                estimator.set_params(**cfg["hyperparams"])

            if isinstance(estimator, Pipeline):
                X_train = X.iloc[train_idx]
            else:
                X_train = X_scaled.iloc[train_idx]

            fit_params = {}
            sig = inspect.signature(estimator.fit)
            sw_train = sw[train_idx] if sw is not None else None
            if "sample_weight" in sig.parameters and sw_train is not None:
                fit_params["sample_weight"] = sw_train
            elif "class_weight" in sig.parameters and y.nunique() == 2:
                if hasattr(estimator, "class_weight"):
                    estimator.set_params(class_weight="balanced")
                else:
                    fit_params["class_weight"] = "balanced"
                models_without_weights.append(name)
            else:
                if sw_train is not None:
                    models_without_weights.append(name)

            converged = True
            try:
                with warnings.catch_warnings(record=True) as warn:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(X_train, y.iloc[train_idx], **fit_params)
                    conv_warn = any(
                        isinstance(w.message, ConvergenceWarning) for w in warn
                    )
                if (
                    conv_warn
                    and hasattr(estimator, "set_params")
                    and "max_iter" in estimator.get_params()
                ):
                    logger.info(
                        "%s atingiu limite de iterações; tentando novamente com max_iter duplicado",
                        name,
                    )
                    new_iter = estimator.get_params()["max_iter"] * 2
                    estimator.set_params(max_iter=new_iter)
                    with warnings.catch_warnings(record=True) as warn2:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimator.fit(X_train, y.iloc[train_idx], **fit_params)
                        converged = not any(
                            isinstance(w.message, ConvergenceWarning) for w in warn2
                        )
                else:
                    converged = not conv_warn
            except Exception as err:
                if name == "lr":
                    logger.warning(
                        "[Heuristic][importance] LogisticRegression failed: %s, bypassing",
                        err,
                    )
                    continue
                estimator.fit(X_train, y.iloc[train_idx])
                converged = False
                if sw_train is not None and name not in models_without_weights:
                    models_without_weights.append(name)

            est_final = (
                estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
            )
            if hasattr(est_final, "feature_importances_"):
                imp = pd.Series(est_final.feature_importances_, index=X_train.columns)
            elif hasattr(est_final, "coef_"):
                coef = (
                    est_final.coef_[0] if est_final.coef_.ndim > 1 else est_final.coef_
                )
                imp = pd.Series(np.abs(coef), index=X_train.columns)
            else:  # pragma: no cover - fallback
                try:
                    import shap

                    expl = shap.TreeExplainer(est_final)
                    sv = expl.shap_values(X_train)
                    if isinstance(sv, list):
                        sv = sv[0]
                    imp = pd.Series(np.abs(sv).mean(axis=0), index=X_train.columns)
                except Exception:
                    continue

            noise_val = float(imp.get("__noise_uniform__", 0.0))
            if "__noise_uniform__" in imp:
                imp = imp.drop("__noise_uniform__")

            noise_values_folds[name].append(noise_val)
            importances_folds[name].append(imp)
            conv_info[f"{name}_converged"] = (
                conv_info[f"{name}_converged"] and converged
            )
            if not converged:
                logger.info("Modelo %s n\u00e3o convergiu", name)

    has_data = any(len(v) > 0 for v in importances_folds.values())
    if not has_data:
        return {
            "kept": list(X_full.columns),
            "removed": [],
            "importances": None,
            "meta": {},
        }

    imp_med: Dict[str, pd.Series] = {}
    fold_ratio: Dict[str, pd.Series] = {}
    noise_values: Dict[str, float] = {}
    for name, imps in importances_folds.items():
        if not imps:
            continue
        df_imp = pd.concat(imps, axis=1)
        med = df_imp.median(axis=1)
        noise_med = float(np.median(noise_values_folds.get(name, [0.0])))
        ratio = (df_imp.gt(noise_med)).sum(axis=1) / df_imp.shape[1]
        imp_med[name] = med
        fold_ratio[name] = ratio
        noise_values[name] = noise_med

    imp_df = pd.DataFrame(imp_med)

    kept: List[str] = []
    removed: List[str] = []
    for feat in imp_df.index:
        if feat in keep_cols:
            kept.append(feat)
            continue
        keep = False
        for model_name in imp_df.columns:
            ratio = fold_ratio.get(model_name, pd.Series()).get(feat, 0.0)
            if ratio >= approval_ratio_fold:
                keep = True
                break
        if keep:
            kept.append(feat)
        else:
            removed.append(feat)

    meta = {
        "models_used": list(imp_df.columns),
        "sample_frac": sample_frac,
        "random_state": random_state,
        "sample_weight_provided": sample_weight is not None,
        "models_without_weights": models_without_weights,
        "cv_folds": cv_folds,
        "cv_type": cv_type,
        "approval_ratio_fold": approval_ratio_fold,
        **conv_info,
    }

    return {
        "kept": kept,
        "removed": removed,
        "importances": imp_df,
        "meta": meta,
    }


# --------------------------------------------------------------------- #
# Graph‑cut: mínimo conjunto de vértices em grafo de correlações        #
# --------------------------------------------------------------------- #

# def graph_cut(  # noqa: C901
#     df: pd.DataFrame,
#     *,
#     target_col: str | None = None,
#     corr_threshold: float = 0.9,
#     keep_cols: List[str] | None = None,
#     method: str = "pearson",
# ) -> Dict[str, Any]:
#     """Seleciona variáveis quebrando pares |corr| > ``corr_threshold``.

#     * Colunas numéricas → Pearson/Spearman (``method``)
#     * Pares categóricos → **share‑of‑equality**  
#       ``identical_share = mean(col1 == col2)``

#     O subconjunto mínimo de vértices que cobre todas as arestas é
#     aproximado via NetworkX ≥3.0.
#     """
#     try:
#         import networkx as nx
#         import numpy as np
#     except ImportError:  # pragma: no cover
#         warnings.warn("graph_cut heuristic skipped – networkx not installed")
#         return {"removed": [], "artefacts": None, "meta": {}}

#     keep_cols = set(keep_cols or [])
#     df_work = df.drop(columns=[target_col], errors="ignore").copy()
#     target = df[target_col] if target_col and target_col in df.columns else None

#     # ------------------------------------------------------------------
#     # Identify categorical columns early (before numeric encoding)
#     # ------------------------------------------------------------------
#     cat_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
#     if cat_cols:
#         df_work[cat_cols] = df_work[cat_cols].fillna("__MISSING__").astype(str)

#     # ------------------------------------------------------------------
#     # Numeric representation for correlation matrix
#     # ------------------------------------------------------------------
#     if cat_cols:
#         # Global ordinal mapping – same mapping for all categorical cols
#         uniques = pd.Series(pd.unique(df_work[cat_cols].values.ravel())).sort_values().tolist()
#         mapping = {v: i for i, v in enumerate(uniques)}
#         df_work[cat_cols] = df_work[cat_cols].apply(lambda s: s.map(mapping)).astype(float)

#     corr_num = df_work.corr(method=method, numeric_only=True).abs()

#     # ------------------------------------------------------------------
#     # Build graph
#     # ------------------------------------------------------------------
#     G = nx.Graph()
#     G.add_nodes_from(df_work.columns)

#     # -------- Add edges for numeric ↔ numeric (and num/cat mixes) -------
#     for i, j in combinations(corr_num.columns, 2):
#         if corr_num.loc[i, j] > corr_threshold:
#             G.add_edge(i, j)

#     # -------- Add edges for categorical ↔ categorical -------------------
#     if cat_cols:
#         tmp_cat = df.drop(columns=[target_col], errors="ignore")[cat_cols].fillna("__MISSING__").astype(str)
#         for i, j in combinations(cat_cols, 2):
#             mask = tmp_cat[i].notna() & tmp_cat[j].notna()
#             if not mask.any():
#                 continue
#             share = (tmp_cat.loc[mask, i] == tmp_cat.loc[mask, j]).mean()
#             if share > corr_threshold:
#                 G.add_edge(i, j)

#     # ------------------------------------------------------------------
#     # Minimum vertex cover (approx)
#     # ------------------------------------------------------------------
#     approx = nx.algorithms.approximation
#     cover = (
#         approx.min_vertex_cover(G)
#         if hasattr(approx, "min_vertex_cover")
#         else approx.min_weighted_vertex_cover(G)
#     )

#     removed = [v for v in cover if v not in keep_cols]

#     return {
#         "removed": removed,
#         "artefacts": G.subgraph(cover).copy() if cover else None,
#         "meta": {"corr_threshold": corr_threshold, "method": method},
#     }

def graph_cut(  # noqa: C901 – complexidade OK p/ heurística
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    corr_threshold: float = 0.9,
    keep_cols: List[str] | None = None,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Seleciona variáveis quebrando pares |corr| > ``corr_threshold``.

    • Numéricas → correlação ``method``  
    • Categóricas → porcentagem de igualdade entre valores
      (``share = mean(col1 == col2)``). Mantém compatibilidade
      com o mesmo threshold.

    Retorna dict com:
        removed   : list[str]
        artefacts : subgrafo com o vertex-cover
        meta      : parâmetros usados
    """
    try:
        import networkx as nx
        import numpy as np
    except ImportError:  # pragma: no cover
        warnings.warn("graph_cut heuristic skipped – networkx not installed")
        return {"removed": [], "artefacts": None, "meta": {}}

    keep_cols = set(keep_cols or [])

    # --- preparação ---
    df_work = df.drop(columns=[target_col], errors="ignore").copy()
    cat_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()

    # normaliza NaNs em categóricas
    if cat_cols:
        df_work[cat_cols] = df_work[cat_cols].fillna("__MISSING__").astype(str)

        # mapeamento ordinal *global* (mesma chave p/ todas as colunas)
        uniques = pd.Series(pd.unique(df_work[cat_cols].values.ravel())).sort_values()
        mapping = {v: i for i, v in enumerate(uniques)}
        df_work[cat_cols] = df_work[cat_cols].apply(lambda s: s.map(mapping)).astype(float)

    # --- correlações numéricas ---
    corr_num = df_work.corr(method=method, numeric_only=True).abs()

    # --- grafo ---
    import itertools as _it
    G = nx.Graph()
    G.add_nodes_from(df_work.columns)

    # arestas numéricas
    for i, j in _it.combinations(corr_num.columns, 2):
        if corr_num.loc[i, j] > corr_threshold:
            G.add_edge(i, j)

    # arestas categóricas (share-of-equality)
    if cat_cols:
        tmp_cat = df[cat_cols].fillna("__MISSING__").astype(str)
        for i, j in _it.combinations(cat_cols, 2):
            share = (tmp_cat[i] == tmp_cat[j]).mean()
            if share > corr_threshold:
                G.add_edge(i, j)

    # --- mínimo vertex cover (aprox) ---
    approx = nx.algorithms.approximation
    cover = (
        approx.min_vertex_cover(G)
        if hasattr(approx, "min_vertex_cover")
        else approx.min_weighted_vertex_cover(G)
    )
    removed = [v for v in cover if v not in keep_cols]

    return {
        "removed": removed,
        "artefacts": G.subgraph(cover).copy() if cover else None,
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
            num = pd.to_numeric(valid, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
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
            corr_date = abs(
                pd.Series(codes).corr(pd.Series(date_ord), method="spearman")
            )
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


# --------------------------------------------------------------------- #
# Boruta Multi SHAP                                                      #
# --------------------------------------------------------------------- #


def boruta_multi_shap(
    df: pd.DataFrame,
    target_col: str,
    *,
    n_iter: int = 50,
    sample_frac: float = 0.7,
    approval_ratio: float = 0.9,
    random_state: int | None = None,
    models: list[dict[str, object]] | None = None,
    problem: str = "auto",
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Wrapper for :class:`BorutaMultiShap`."""

    selector = BorutaMultiShap(
        n_iter=n_iter,
        sample_frac=sample_frac,
        approval_ratio=approval_ratio,
        random_state=random_state,
        models=models,
    )
    return selector(df, target_col, problem=problem, logger=logger)
