from __future__ import annotations

import warnings
import inspect
from typing import Any, Dict, List
import time

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

# Imports absolutos para evitar falhas quando o módulo é executado
# fora do contexto de pacote.
from vassoura.scaler import DynamicScaler
from vassoura.utils import woe_encode


class BorutaMultiShap:
    """Boruta-inspired feature selector using multiple models and SHAP values."""

    def __init__(
        self,
        n_iter: int = 50,
        sample_frac: float = 0.7,
        approval_ratio: float = 0.9,
        random_state: int | None = None,
        models: list[dict[str, object]] | None = None,
        *,
        model_names: list[str] | None = None,
        shap_rows: int | None = None,
        estimators: int = 50,
        max_depth: int = 5,
        fast_mode: bool = False,
    ) -> None:
        if fast_mode:
            n_iter = 20
            sample_frac = 0.5
            shap_rows = 1000
            estimators = 30
            max_depth = 4
            model_names = ["lgbm"]

        self.n_iter = n_iter
        self.sample_frac = sample_frac
        self.approval_ratio = approval_ratio
        self.random_state = random_state
        self.models = models
        self.model_names = model_names
        self.shap_rows = shap_rows
        self.estimators = estimators
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    def __call__(
        self,
        df: pd.DataFrame,
        target_col: str,
        *,
        problem: str = "auto",
        logger: Any | None = None,
    ) -> Dict[str, Any]:
        import shap

        start_t = time.perf_counter()

        rng = np.random.default_rng(self.random_state)
        X_full = df.drop(columns=[target_col])
        y_full = df[target_col]

        # ------------------------------------------------------------------
        # Determine problem type and sample weights
        # ------------------------------------------------------------------
        if problem == "auto":
            if y_full.nunique() == 2:
                problem = "binary"
            else:
                problem = "reg"

        sample_weight = None
        if problem == "binary":
            try:
                sample_weight = compute_sample_weight("balanced", y_full)
            except Exception:
                sample_weight = None

        # ------------------------------------------------------------------
        # Shadow features
        # ------------------------------------------------------------------
        X = X_full.copy()
        shadow_cols: List[str] = []
        for col in X_full.columns:
            sh_col = f"__shadow__{col}"
            X[sh_col] = rng.permutation(X_full[col].values)
            shadow_cols.append(sh_col)
        real_cols = X_full.columns.tolist()

        # ------------------------------------------------------------------
        # Pre-processing (WoE, fillna, scaling once)
        # ------------------------------------------------------------------
        cat_real = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_shadow = [f"__shadow__{c}" for c in cat_real]
        cat_all = cat_real + cat_shadow
        if cat_all:
            try:
                X = woe_encode(X, y_full, cols=cat_all)
            except Exception:
                pass
        X = X.fillna(0)

        if not df.attrs.get("scaled_by_vassoura", False):
            scaler = DynamicScaler()
            scaler.fit(X)
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
            df.attrs["scaled_by_vassoura"] = True
        # if already scaled, keep X as is

        # ------------------------------------------------------------------
        # Default models
        # ------------------------------------------------------------------
        models_cfg = self.models
        if models_cfg is None:
            models_cfg = []
            try:
                from sklearn.linear_model import LogisticRegression

                if problem == "binary":
                    models_cfg.append(
                        {"name": "lr", "estimator": LogisticRegression(max_iter=200, penalty='l2',random_state=42)}
                    )
            except Exception:
                warnings.warn("LogisticRegression unavailable")

            try:
                from xgboost import XGBClassifier, XGBRegressor

                tree_cls = XGBClassifier if problem == "binary" else XGBRegressor
                models_cfg.append(
                    {
                        "name": "xgb",
                        "estimator": tree_cls(
                            n_estimators=self.estimators,
                            max_depth=self.max_depth,
                            learning_rate=0.1,
                            subsample=0.8,
                            eval_metric="logloss" if problem == "binary" else None,
                            n_jobs=-1,
                            random_state=self.random_state,
                        ),
                    }
                )
            except Exception:
                warnings.warn("XGBoost unavailable")

            try:
                from lightgbm import LGBMClassifier, LGBMRegressor

                lgbm_cls = LGBMClassifier if problem == "binary" else LGBMRegressor
                models_cfg.append(
                    {
                        "name": "lgbm",
                        "estimator": lgbm_cls(
                            n_estimators=self.estimators,
                            max_depth=self.max_depth,
                            learning_rate=0.1,
                            subsample=0.8,
                            random_state=self.random_state,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    }
                )
            except Exception:
                warnings.warn("LightGBM unavailable")

        if self.model_names is not None:
            models_cfg = [m for m in models_cfg if m.get("name") in self.model_names]

        # initialise tracking
        approvals = pd.Series(0, index=real_cols, dtype=float)
        shap_sums: Dict[str, pd.Series] = {
            cfg.get("name", str(i)): pd.Series(0.0, index=real_cols)
            for i, cfg in enumerate(models_cfg)
        }

        from shap.maskers import Independent
        masker = (
            Independent(X_full, max_samples=self.shap_rows)
            if self.shap_rows is not None
            else Independent(X_full)
        )

        # ------------------------------------------------------------------
        # Iterations
        # ------------------------------------------------------------------
        thr_count = self.approval_ratio * self.n_iter
        removed_early: list[str] = []
        for it in range(self.n_iter):
            idx = rng.choice(len(X), size=int(len(X) * self.sample_frac), replace=True)
            Xi = X.iloc[idx]
            yi = y_full.iloc[idx]
            swi = sample_weight[idx] if sample_weight is not None else None

            if self.shap_rows is not None and len(Xi) > self.shap_rows:
                sub_idx = rng.choice(len(Xi), size=self.shap_rows, replace=False)
                Xi_shap = Xi.iloc[sub_idx]
            else:
                Xi_shap = Xi

            for i, cfg in enumerate(models_cfg):
                name = cfg.get("name", str(i))
                estimator = clone(cfg.get("estimator"))

                fit_params = {}
                sig = inspect.signature(estimator.fit)
                if "sample_weight" in sig.parameters and swi is not None:
                    fit_params["sample_weight"] = swi
                elif "class_weight" in sig.parameters and problem == "binary":
                    try:
                        estimator.set_params(class_weight="balanced")
                    except Exception:
                        fit_params["class_weight"] = "balanced"

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "error",
                        category=ConvergenceWarning,
                    )
                    try:
                        estimator.fit(Xi, yi, **fit_params)
                    except ConvergenceWarning:
                        if logger is not None:
                            logger.info(
                                "Convergence warning with %s; switching solver",
                                name,
                            )
                        try:
                            estimator.set_params(solver="liblinear", max_iter=500)
                        except Exception:
                            pass
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        estimator.fit(Xi, yi, **fit_params)

                # SHAP values
                if name == "lr":
                    try:
                        coef = estimator.coef_.ravel()
                        sv = Xi_shap.values * coef
                    except Exception:
                        expl = shap.LinearExplainer(estimator, masker=masker)
                        sv = expl.shap_values(Xi_shap)
                        if isinstance(sv, list):
                            sv = np.stack(sv).sum(axis=0)
                else:
                    expl = shap.TreeExplainer(estimator)
                    sv = expl.shap_values(Xi_shap)
                    if isinstance(sv, list):
                        sv = np.stack(sv).sum(axis=0)

                mean_abs = np.abs(sv).mean(axis=0)
                shap_series = pd.Series(mean_abs, index=Xi_shap.columns)
                shap_sums[name] = shap_sums[name].add(shap_series[real_cols], fill_value=0)
                thr = shap_series[shadow_cols].quantile(0.75)
                winners = shap_series[real_cols] > thr
                approvals[winners.index] += winners.astype(int)

            remaining = self.n_iter - it - 1
            cannot_reach = approvals + remaining < thr_count
            if cannot_reach.any() and len(real_cols) - cannot_reach.sum() > 0:
                drop_cols = cannot_reach[cannot_reach].index.tolist()
                X.drop(columns=drop_cols + [f"__shadow__{c}" for c in drop_cols], inplace=True)
                for name in shap_sums:
                    shap_sums[name].drop(labels=drop_cols, inplace=True)
                approvals.drop(labels=drop_cols, inplace=True)
                real_cols = [c for c in real_cols if c not in drop_cols]
                shadow_cols = [f"__shadow__{c}" for c in real_cols]
                removed_early.extend(drop_cols)
                if not real_cols:
                    break

            done = ((approvals >= thr_count) | (approvals + remaining < thr_count)).sum()
            if it >= max(5, self.n_iter // 5) and done / len(real_cols) >= 0.95:
                break

        approvals_int = approvals.astype(int)
        approval_prop = approvals / self.n_iter
        kept = approval_prop[approval_prop >= self.approval_ratio].index.tolist()
        removed = approval_prop[approval_prop < self.approval_ratio / 2].index.tolist()
        removed += [c for c in removed_early if c not in removed]

        shap_mean_by_model = {
            name: series / self.n_iter for name, series in shap_sums.items()
        }
        shap_df = pd.DataFrame(shap_mean_by_model)

        elapsed = time.perf_counter() - start_t
        meta = {
            "n_iter": self.n_iter,
            "sample_frac": self.sample_frac,
            "approval_ratio": self.approval_ratio,
            "random_state": self.random_state,
            "models_used": list(shap_mean_by_model.keys()),
            "elapsed_sec": elapsed,
        }

        artefacts = {
            "shap_mean_by_model": shap_df,
            "approvals": approvals_int,
            "timings": {"total_sec": elapsed},
        }

        return {"kept": kept, "removed": removed, "artefacts": artefacts, "meta": meta}
