import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vassoura.scaler import DynamicScaler


def test_roundtrip_auto():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": rng.normal(size=100),
        "b": rng.lognormal(size=100),
    })
    sc = DynamicScaler()
    sc.fit(df)
    tr = sc.transform(df, return_df=True)
    inv = sc.inverse_transform(tr, return_df=True)
    assert np.allclose(df.values, inv.values, atol=1e-6, rtol=1e-6)


def test_standard_strategy():
    df = pd.DataFrame({"x": np.random.normal(size=50), "y": np.random.normal(size=50)})
    sc = DynamicScaler(strategy="standard")
    sc.fit(df)
    assert all(isinstance(s, StandardScaler) for s in sc.scalers_.values())


def test_save_load(tmp_path):
    df = pd.DataFrame({"x": np.random.normal(size=30), "y": np.random.normal(size=30)})
    sc = DynamicScaler()
    sc.fit(df)
    p = tmp_path / "scaler.pkl"
    sc.save(p)
    sc2 = DynamicScaler().load(p)
    assert np.allclose(sc.transform(df), sc2.transform(df))
