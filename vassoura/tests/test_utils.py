import pandas as pd
import pytest
from vassoura.utils import (
    adaptive_sampling,
    criar_dataset_pd_behavior,
    figsize_from_matrix,
    suggest_corr_method,
)

def test_suggest_corr_method():
    assert suggest_corr_method(["a", "b"], []) == "pearson"
    assert suggest_corr_method(["a"], ["c"]) == "spearman"
    assert suggest_corr_method([], ["c"]) == "cramer"

def test_figsize_from_matrix_bounds():
    small = figsize_from_matrix(2, base=0.4, min_size=6, max_size=20)
    large = figsize_from_matrix(100, base=0.4, min_size=6, max_size=20)
    assert small[0] >= 6 and small[0] == small[1]
    assert large[0] <= 20 and large[0] == large[1]

def test_criar_dataset_pd_behavior_columns():
    df = criar_dataset_pd_behavior(n_clientes=5, max_anos=1, n_features=3, seed=0)
    expected_cols = {"NroContrato", "AnoMesReferencia", "feature_01", "feature_02", "feature_03", "ever90m12"}
    assert expected_cols <= set(df.columns)
    # AnoMesReferencia deve estar no formato YYYYMM (int)
    assert df["AnoMesReferencia"].dtype == int


def test_adaptive_sampling_stratify_and_order():
    df = pd.DataFrame({
        "target": [0] * 80 + [1] * 20,
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
    })
    sampled = adaptive_sampling(
        df,
        max_cells=40,
        stratify_col="target",
        date_cols=["date"],
        random_state=0,
    )
    assert len(sampled) == 20
    assert sampled["target"].mean() == pytest.approx(df["target"].mean())
    assert sampled["date"].is_monotonic_increasing
