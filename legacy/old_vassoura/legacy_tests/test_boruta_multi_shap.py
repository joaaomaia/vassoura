import pandas as pd
import numpy as np
from vassoura.core import Vassoura

from vassoura.heuristics import boruta_multi_shap


def test_boruta_multi_shap_basic():
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    target = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x_noise": x_noise, "target": target})

    result = boruta_multi_shap(df, "target", n_iter=1, sample_frac=0.8, random_state=0)
    assert set(result["kept"] + result["removed"]) == {"x1", "x_noise"}
    assert "approvals" in result["artefacts"]


def test_boruta_multi_shap_respects_scaled_flag():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=50), "target": rng.integers(0, 2, 50)})
    df.attrs["scaled_by_vassoura"] = True
    result = boruta_multi_shap(df, "target", n_iter=1, sample_frac=0.8, random_state=0)
    assert df.attrs["scaled_by_vassoura"]
    assert isinstance(result["meta"], dict)


def test_boruta_multi_shap_class_imbalance():
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=200)
    target = (rng.random(200) > 0.95).astype(int)
    df = pd.DataFrame({"x1": x1, "target": target})
    result = boruta_multi_shap(df, "target", n_iter=1, sample_frac=0.8, random_state=0)
    assert isinstance(result["meta"], dict)


def test_boruta_multi_shap_vassoura_integration():
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=120)
    x_noise = rng.normal(size=120)
    target = (x1 + rng.normal(scale=0.3, size=120) > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x_noise": x_noise, "target": target})

    vs = Vassoura(
        df,
        target_col="target",
        heuristics=["boruta_multi_shap"],
        params={
            "boruta_multi_shap": {"n_iter": 1, "sample_frac": 0.8, "random_state": 3}
        },
    )
    out = vs.run()
    assert out.shape[1] <= df.shape[1]
    assert any(step["reason"] == "boruta_multi_shap" for step in vs.history)
