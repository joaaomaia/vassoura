from __future__ import annotations

import pandas as pd

from vassoura.utils.metrics import (
    roc_auc_score_safe,
    brier_score,
    ks_statistic,
    classification_report_df,
    SCORERS,
)


def test_auc_safe_single_class():
    y = pd.Series([1, 1, 1])
    pred = pd.Series([0.2, 0.3, 0.4])
    assert roc_auc_score_safe(y, pred) == 0.5


def test_brier_bounds():
    y = pd.Series([0, 1, 0, 1])
    p = pd.Series([0.1, 0.8, 0.2, 0.7])
    val = brier_score(y, p)
    assert 0 <= val <= 1


def test_ks_statistic_range():
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0.1, 0.2, 0.8, 0.7])
    ks = ks_statistic(y, p)
    assert 0 <= ks <= 1


def test_classification_report_df_shape():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 1, 1])
    df = classification_report_df(y_true, y_pred)
    for col in ["precision", "recall", "f1-score", "support"]:
        assert col in df.columns


def test_scorers_dict_completeness():
    assert set(SCORERS.keys()) == {"auc", "pr_auc", "f1", "mcc", "brier", "ks"}
