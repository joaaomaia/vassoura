from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    classification_report,
    make_scorer,
)
from scipy.stats import ks_2samp


__all__ = [
    "roc_auc_score_safe",
    "pr_auc_score_safe",
    "f1_safe",
    "mcc_safe",
    "brier_score",
    "ks_statistic",
    "classification_report_df",
    "SCORERS",
]


def roc_auc_score_safe(y_true, y_pred) -> float:
    try:
        score = roc_auc_score(y_true, y_pred)
        if pd.isna(score):
            return 0.5
        return score
    except ValueError:
        return 0.5


def pr_auc_score_safe(y_true, y_pred) -> float:
    return average_precision_score(y_true, y_pred)


def f1_safe(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, zero_division=0)


def mcc_safe(y_true, y_pred) -> float:
    return matthews_corrcoef(y_true, y_pred)


def brier_score(y_true, y_prob) -> float:
    return brier_score_loss(y_true, y_prob)


def ks_statistic(y_true, y_prob) -> float:
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    return ks_2samp(pos_scores, neg_scores).statistic


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(rep).T


SCORERS = {
    "auc": make_scorer(roc_auc_score_safe, needs_proba=True),
    "pr_auc": make_scorer(pr_auc_score_safe, needs_proba=True),
    "f1": make_scorer(f1_safe),
    "mcc": make_scorer(mcc_safe),
    "brier": make_scorer(
        brier_score, needs_proba=True, greater_is_better=False
    ),
    "ks": make_scorer(ks_statistic, needs_proba=True),
}
