from __future__ import annotations

"""Minimal cross-validation helpers."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit


@dataclass
class CrossValidator:
    """Wrapper around a sklearn splitter."""

    cv_type: str = "stratified"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42

    def _make_splitter(self, y=None):
        if self.cv_type == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        if self.cv_type == "time_series":
            return TimeSeriesSplit(n_splits=self.n_splits)
        raise ValueError(f"Unknown cv_type: {self.cv_type}")

    def split(self, X: Sequence, y: Sequence | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        splitter = self._make_splitter(y)
        if isinstance(splitter, StratifiedKFold):
            return splitter.split(X, y)
        return splitter.split(X)

    def get_n_splits(self, X: Sequence | None = None, y: Sequence | None = None) -> int:
        splitter = self._make_splitter(y)
        return splitter.get_n_splits(X, y)


def _take(data: Sequence, idx: Iterable[int]):
    if hasattr(data, "iloc"):
        return data.iloc[list(idx)]
    return data[list(idx)]


def cross_validate(
    estimator: BaseEstimator,
    X: Sequence,
    y: Sequence,
    *,
    cv: CrossValidator | None = None,
    scoring: Dict[str, Callable[[BaseEstimator, Sequence, Sequence], float]] | None = None,
) -> Dict[str, list[float]]:
    """Run cross validation and return metrics."""

    cv = cv or CrossValidator()
    if scoring is None:
        scoring = {"score": lambda est, X_, y_: est.score(X_, y_)}

    results: Dict[str, list[float]] = {}
    for name in scoring:
        results[f"train_{name}"] = []
        results[f"test_{name}"] = []

    for train_idx, test_idx in cv.split(X, y):
        est = clone(estimator)
        X_train, y_train = _take(X, train_idx), _take(y, train_idx)
        X_test, y_test = _take(X, test_idx), _take(y, test_idx)
        est.fit(X_train, y_train)
        for name, func in scoring.items():
            results[f"train_{name}"].append(func(est, X_train, y_train))
            results[f"test_{name}"].append(func(est, X_test, y_test))

    return results
