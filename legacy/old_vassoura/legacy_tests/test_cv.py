import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from vassoura.validation.cv import CrossValidator, cross_validate


def test_cross_validator_stratified():
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    cv = CrossValidator(cv_type="stratified", n_splits=3, random_state=0)
    splits = list(cv.split(X, y))
    assert len(splits) == 3
    base_rate = y.mean()
    for _, test_idx in splits:
        assert abs(y[test_idx].mean() - base_rate) < 0.1


def test_cross_validate_simple():
    X, y = make_classification(n_samples=50, n_features=4, random_state=1)
    est = LogisticRegression(max_iter=200)
    cv = CrossValidator(cv_type="stratified", n_splits=3, random_state=0)

    scoring = {"acc": lambda est, X, y: accuracy_score(y, est.predict(X))}
    res = cross_validate(est, X, y, cv=cv, scoring=scoring)

    assert set(res) == {"train_acc", "test_acc"}
    assert len(res["train_acc"]) == 3
    assert all(isinstance(v, float) for v in res["test_acc"])
