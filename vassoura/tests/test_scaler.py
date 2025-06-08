# tests/test_scaler.py
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # garante execução headless

from vassoura.scaler import DynamicScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator


# ----------------------------------------------------------------------
# -------------- GERADOR DE DADOS --------------------------------------
# ----------------------------------------------------------------------
def make_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "const": np.full(n, 5.0),
        "already_scaled": rng.uniform(0, 1, n),
        "normal": rng.normal(0, 1, n),
        "skewed": rng.exponential(1, n),
        "heavy_tail": rng.standard_t(df=2, size=n),
    })


# ----------------------------------------------------------------------
# -------------- FIXTURES ----------------------------------------------
# ----------------------------------------------------------------------
@pytest.fixture(params=[30, 500, 5000])
def sample_df(request):
    rng = np.random.default_rng(0)
    return make_dataset(request.param, rng)


# ----------------------------------------------------------------------
# -------------- TESTES GERAIS -----------------------------------------
# ----------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["auto", "standard", "robust", "minmax", "quantile", None])
def test_roundtrip_identity(sample_df, strategy):
    sc = DynamicScaler(strategy=strategy, random_state=0)
    sc.fit(sample_df)
    inv = sc.inverse_transform(sc.transform(sample_df, return_df=True), return_df=True)
    np.testing.assert_allclose(sample_df.values, inv.values, rtol=1e-5, atol=1e-8)


def test_auto_strategy_core_checks(sample_df):
    sc = DynamicScaler(strategy="auto", random_state=0, shapiro_p_val=0.01).fit(sample_df)
    assert sc.scalers_["const"] is None
    ascaler = sc.scalers_["already_scaled"]
    assert ascaler is None or isinstance(ascaler, MinMaxScaler)
    # “normal” deve receber um scaler concreto
    assert isinstance(sc.scalers_["normal"], BaseEstimator)


def test_standard_strategy_assigns_standard_scalers(sample_df):
    sc = DynamicScaler(strategy="standard").fit(sample_df)
    assert all(isinstance(s, StandardScaler) for s in sc.scalers_.values())


def test_minmax_range(sample_df):
    sc = DynamicScaler(strategy="minmax").fit(sample_df)
    out = sc.transform(sample_df, return_df=True)
    assert out.min().min() >= -1e-9
    assert out.max().max() <= 1.0 + 1e-9


def test_transform_without_fit_raises(sample_df):
    sc = DynamicScaler(strategy="robust")
    with pytest.warns(UserWarning):
        sc.transform(sample_df)


def test_missing_columns_raises(sample_df):
    sc = DynamicScaler(strategy="standard").fit(sample_df)
    with pytest.raises(ValueError):
        sc.transform(sample_df.drop(columns=["skewed"]))


def test_serialization_consistency(sample_df, tmp_path):
    path = tmp_path / "scaler.pkl"
    sc1 = DynamicScaler(strategy="auto", serialize=True, save_path=path).fit(sample_df)
    out1 = sc1.transform(sample_df)

    sc2 = DynamicScaler().load(path)
    out2 = sc2.transform(sample_df)

    np.testing.assert_array_equal(out1, out2)


def test_plot_histograms_runs(sample_df, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    sc = DynamicScaler(strategy="auto").fit(sample_df)
    sc.plot_histograms(sample_df, sc.transform(sample_df, return_df=True),
                       features=["normal", "skewed"])


def test_get_feature_names_and_report(sample_df):
    sc = DynamicScaler(strategy="robust").fit(sample_df)
    assert list(sc.get_feature_names_out(sample_df.columns)) == list(sample_df.columns)
    rep = sc.report_as_df()
    assert isinstance(rep, pd.DataFrame)
    assert rep.shape[0] == sample_df.shape[1]


def test_nan_handling():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(size=100)})
    df.loc[df.sample(frac=0.1, random_state=0).index, "x"] = np.nan
    sc = DynamicScaler(strategy="standard").fit(df)
    out = sc.transform(df, return_df=True)
    assert out.isna().sum().sum() == df.isna().sum().sum()
