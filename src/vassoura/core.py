from __future__ import annotations

import inspect

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


def _supports_sample_weight(estimator) -> bool:
    """Return True if `estimator.fit` has a `sample_weight` param."""
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (AttributeError, ValueError):
        return False


def _is_classifier(est) -> bool:
    return getattr(est, "_estimator_type", None) == "classifier"


from vassoura.audit import AuditTrail
from vassoura.logs import get_logger
from vassoura.models import registry
from vassoura.preprocessing import SampleManager, make_default_pipeline
from vassoura.process.wrappers import import_heuristic
from vassoura.report import SECTION_REGISTRY, ReportManager
from vassoura.utils import SCORERS, calculate_sample_weights, split_dtypes
from vassoura.validation import get_stratified_cv


class Vassoura:
    def __init__(
        self,
        target_col: str,
        *,
        sampler_cfg: dict | None = None,
        pipeline_cfg: dict | None = None,
        model_name: str = "logistic_balanced",
        heuristic: str = "basic",
        cv_cfg: dict | None = None,
        metrics: list[str] | None = None,
        report: bool = False,
        random_state: int | None = 42,
        verbose: int = 1,
        id_cols: list[str] | None = None,
        date_cols: list[str] | None = None,
        ignore_cols: list[str] | None = None,
        keep_cols: list[str] | None = None,
        drop_ignored: bool = True,
    ) -> None:
        self.target_col = target_col
        self.sampler_cfg = sampler_cfg or {}
        self.pipeline_cfg = pipeline_cfg or {}
        self.model_name = model_name
        self.heuristic = heuristic
        self.cv_cfg = cv_cfg or {}
        self.metrics = metrics or ["auc"]
        self.report = report
        self.random_state = random_state
        self.verbose = verbose
        self.id_cols = id_cols or []
        self.date_cols = date_cols or []
        self.ignore_cols = ignore_cols or []
        self.keep_cols = keep_cols or []
        self.drop_ignored = drop_ignored
        self.logger = get_logger("Vassoura")
        if verbose >= 2:
            self.logger.setLevel("DEBUG")

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Vassoura":
        self.logger.info("=== Vassoura Fit Started ===")
        df = df.copy()

        if self.ignore_cols:
            self.ignored_ = df[self.ignore_cols]
            if self.drop_ignored:
                df = df.drop(columns=self.ignore_cols)
        if self.id_cols:
            self.ids_ = df[self.id_cols]
            df = df.drop(columns=self.id_cols)

        for c in self.date_cols:
            if c not in df.columns:
                continue
            if not np.issubdtype(df[c].dtype, np.datetime64):
                df[c] = pd.to_datetime(df[c], errors="coerce")
            df[c] = (df[c].view("int64") // 86_400_000_000_000).astype("float64")

        self.keep_cols_ = [c for c in self.keep_cols if c in df.columns]

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        if self.report:
            self.audit_ = AuditTrail(auto_detect_types=True)
            self.audit_.take_snapshot(df, "raw")
            self.dataset_shape_ = df.shape

        # Sampling
        self.sampler_ = SampleManager(**self.sampler_cfg)
        X_s, y_s = self.sampler_.fit_resample(X, y)

        # Pipeline
        num_cols, cat_cols = split_dtypes(X_s)
        pipe_args = {
            "scaler_strategy": self.pipeline_cfg.get("scaler_strategy", "auto"),
            "encoder": self.pipeline_cfg.get("encoder", "woe"),
        }
        self.pipeline_ = make_default_pipeline(num_cols, cat_cols, **pipe_args)

        # Model
        ModelCls = registry.get(self.model_name)
        self.model_ = ModelCls(random_state=self.random_state)
        sample_weights = calculate_sample_weights(y_s)
        self.logger.debug("Using sample weights – mean: %.3f", sample_weights.mean())

        cv = get_stratified_cv(**self.cv_cfg)
        needs_proba = {"auc", "brier", "logloss"}
        if not _is_classifier(self.model_):
            self.metrics = [m for m in self.metrics if m in {"mse", "mae", "r2"}]
        elif not hasattr(self.model_, "predict_proba"):
            self.metrics = [m for m in self.metrics if m not in {"auc", "brier"}]
        if not self.metrics:
            self.metrics = ["accuracy"] if _is_classifier(self.model_) else ["r2"]
        scoring = {m: SCORERS.get(m, m) for m in self.metrics}

        pipeline = Pipeline([("prep", self.pipeline_), ("clf", self.model_)])
        fit_params = (
            {"clf__sample_weight": sample_weights}
            if _supports_sample_weight(self.model_)
            else {}
        )
        if not fit_params:
            self.logger.info(
                "Sample weights not supported – falling back to class_weight"
            )
            if hasattr(self.model_, "get_params") and "class_weight" in (
                self.model_.get_params()
            ):
                self.model_.set_params(class_weight="balanced")

        cv_kwargs = {
            "cv": cv,
            "scoring": scoring,
            "return_train_score": False,
        }
        if "params" in inspect.signature(cross_validate).parameters:
            if fit_params:
                cv_kwargs["params"] = fit_params
        else:
            if fit_params:
                cv_kwargs["fit_params"] = fit_params

        self.metrics_ = cross_validate(pipeline, X_s, y_s, **cv_kwargs)
        self._used_sample_weight = bool(fit_params)

        # Fit final pipeline and model on full data
        self.pipeline_.fit(X_s, y_s)
        Xt_all = self.pipeline_.transform(X_s)
        if _supports_sample_weight(self.model_):
            self.model_.fit(Xt_all, y_s, sample_weight=sample_weights)
        else:
            if (
                hasattr(self.model_, "get_params")
                and "class_weight" in self.model_.get_params()
            ):
                self.model_.set_params(class_weight="balanced")
            self.model_.fit(Xt_all, y_s)

        # Importance heuristic
        HeurCls = import_heuristic(self.heuristic)
        self.heuristic_ = HeurCls(model=self.model_)
        self.ranking_ = self.heuristic_.run(X_s, y_s, sample_weight=sample_weights)
        self.ranking_ = self.ranking_.reindex(
            self.ranking_.index.union(self.keep_cols_)
        ).fillna(0)

        if self.report:
            df_proc = X_s.copy()
            df_proc[self.target_col] = y_s
            self.audit_.take_snapshot(df_proc, "processed")
            self.audit_.compare_snapshots("raw", "processed")
            self.export_report()

        self.logger.info("=== Vassoura Fit Completed ===")
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline_.transform(df)

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame):
        Xt = self.pipeline_.transform(df)
        return self.model_.predict(Xt)

    # ------------------------------------------------------------------
    def get_feature_ranking(self) -> pd.Series:
        return self.ranking_.sort_values(ascending=False)

    # ------------------------------------------------------------------
    def export_report(self, path: str = "reports/report.html") -> None:
        rm = ReportManager()
        rm.add_section(
            SECTION_REGISTRY["overview"](
                audit=self.audit_,
                snapshot_names=list(self.audit_.snapshots.keys()),
                dataset_shape=self.dataset_shape_,
                id_cols=self.id_cols,
                date_cols=self.date_cols,
                ignore_cols=self.ignore_cols,
                keep_cols=self.keep_cols,
            )
        )
        metrics_df = pd.DataFrame(self.metrics_)
        rm.add_section(SECTION_REGISTRY["performance"](metrics=metrics_df))
        rm.add_section(
            SECTION_REGISTRY["feature_importance"](
                importance=self.get_feature_ranking()
            )
        )
        rm.add_section(
            SECTION_REGISTRY["audit_diff"](
                audit=self.audit_, base="raw", new="processed"
            )
        )
        rm.render(path)

    # ------------------------------------------------------------------
    def save(self, path: str = "models/vassoura.pkl") -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Vassoura":
        return joblib.load(path)
