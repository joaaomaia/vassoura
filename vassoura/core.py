from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from vassoura.logs import get_logger
from vassoura.preprocessing import SampleManager, make_default_pipeline
from vassoura.models import registry
from vassoura.utils import SCORERS, split_dtypes, calculate_sample_weights
from vassoura.validation import get_stratified_cv
from vassoura.process.wrappers import import_heuristic


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
        self.logger = get_logger("Vassoura")
        if verbose >= 2:
            self.logger.setLevel("DEBUG")

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Vassoura":
        self.logger.info("=== Vassoura Fit Started ===")
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Sampling
        self.sampler_ = SampleManager(**self.sampler_cfg)
        X_s, y_s = self.sampler_.fit_resample(X, y)

        # Pipeline
        num_cols, cat_cols = split_dtypes(X_s)
        pipe_args = {
            "scaler_strategy": self.pipeline_cfg.get(
                "scaler_strategy", "auto"
            ),
            "encoder": self.pipeline_cfg.get("encoder", "woe"),
        }
        self.pipeline_ = make_default_pipeline(num_cols, cat_cols, **pipe_args)

        # Model
        ModelCls = registry.get(self.model_name)
        self.model_ = ModelCls(random_state=self.random_state)
        sample_weights = calculate_sample_weights(y_s)
        self.logger.debug(
            "Using sample weights – mean: %.3f", sample_weights.mean()
        )

        cv = get_stratified_cv(**self.cv_cfg)
        scoring = {m: SCORERS[m] for m in self.metrics}

        pipeline = Pipeline([("prep", self.pipeline_), ("clf", self.model_)])
        try:
            self.metrics_ = cross_validate(
                pipeline,
                X_s,
                y_s,
                cv=cv,
                fit_params={"clf__sample_weight": sample_weights},
                scoring=scoring,
                return_train_score=False,
            )
            self._used_sample_weight = True
        except TypeError:
            self.logger.info(
                "Sample weights not supported – falling back to class_weight"
            )
            if hasattr(self.model_, "get_params") and "class_weight" in (
                self.model_.get_params()
            ):
                self.model_.set_params(class_weight="balanced")
            self.metrics_ = cross_validate(
                pipeline,
                X_s,
                y_s,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
            )
            self._used_sample_weight = False

        # Fit final pipeline and model on full data
        self.pipeline_.fit(X_s, y_s)
        Xt_all = self.pipeline_.transform(X_s)
        try:
            self.model_.fit(Xt_all, y_s, sample_weight=sample_weights)
        except TypeError:
            if (
                hasattr(self.model_, "get_params")
                and "class_weight" in self.model_.get_params()
            ):
                self.model_.set_params(class_weight="balanced")
            self.model_.fit(Xt_all, y_s)

        # Importance heuristic
        HeurCls = import_heuristic(self.heuristic)
        self.heuristic_ = HeurCls(model=self.model_)
        self.ranking_ = self.heuristic_.run(
            X_s, y_s, sample_weight=sample_weights
        )

        if self.report:
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
        self.logger.warning("Report generation not implemented")

    # ------------------------------------------------------------------
    def save(self, path: str = "models/vassoura.pkl") -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Vassoura":
        return joblib.load(path)
