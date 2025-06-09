from __future__ import annotations

import warnings
import inspect
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from .scaler import DynamicScaler
from .utils import woe_encode


class BorutaMultiShap:
    """Boruta-inspired feature selector using multiple models and SHAP values."""

    def __init__(
        self,
        n_iter: int = 50,
        sample_frac: float = 0.7,
        approval_ratio: float = 0.9,
        random_state: int | None = None,
        models: list[dict[str, object]] | None = None,
    ) -> None:
        self.n_iter = n_iter
        self.sample_frac = sample_frac
        self.approval_ratio = approval_ratio
        self.random_state = random_state
        self.models = models

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
                            n_estimators=50,
                            max_depth=5,
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
                            n_estimators=50,
                            max_depth=5,
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

        # initialise tracking
        approvals = pd.Series(0, index=real_cols, dtype=float)
        shap_sums: Dict[str, pd.Series] = {
            cfg.get("name", str(i)): pd.Series(0.0, index=real_cols)
            for i, cfg in enumerate(models_cfg)
        }

        # ------------------------------------------------------------------
        # Iterations
        # ------------------------------------------------------------------
        for _ in range(self.n_iter):
            idx = rng.choice(len(X), size=int(len(X) * self.sample_frac), replace=True)
            Xi = X.iloc[idx]
            yi = y_full.iloc[idx]
            swi = sample_weight[idx] if sample_weight is not None else None

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

                from shap.maskers import Independent

                # SHAP values
                if name == "lr":
                    masker = Independent(Xi)
                    expl = shap.LinearExplainer(estimator, masker=masker)
                    sv = expl.shap_values(Xi)
                    if isinstance(sv, list):
                        sv = np.stack(sv).sum(axis=0)
                else:
                    expl = shap.TreeExplainer(estimator)
                    sv = expl.shap_values(Xi)
                    if isinstance(sv, list):
                        sv = np.stack(sv).sum(axis=0)


                mean_abs = np.abs(sv).mean(axis=0)
                shap_series = pd.Series(mean_abs, index=Xi.columns)
                shap_sums[name] = shap_sums[name].add(shap_series[real_cols], fill_value=0)
                thr = shap_series[shadow_cols].quantile(0.75)
                winners = shap_series[real_cols] > thr
                approvals[winners.index] += winners.astype(int)

        approvals_int = approvals.astype(int)
        approval_prop = approvals / self.n_iter
        kept = approval_prop[approval_prop >= self.approval_ratio].index.tolist()
        removed = approval_prop[approval_prop < self.approval_ratio / 2].index.tolist()

        shap_mean_by_model = {
            name: series / self.n_iter for name, series in shap_sums.items()
        }
        shap_df = pd.DataFrame(shap_mean_by_model)

        meta = {
            "n_iter": self.n_iter,
            "sample_frac": self.sample_frac,
            "approval_ratio": self.approval_ratio,
            "random_state": self.random_state,
            "models_used": list(shap_mean_by_model.keys()),
        }

        artefacts = {
            "shap_mean_by_model": shap_df,
            "approvals": approvals_int,
        }

        return {"kept": kept, "removed": removed, "artefacts": artefacts, "meta": meta}
