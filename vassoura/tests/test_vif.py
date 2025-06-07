import pandas as pd
import numpy as np
from vassoura.core import compute_vif

def _make_vif_data():
    np.random.seed(0)
    x1 = np.random.normal(size=200)
    x2 = x1 * 0.95 + np.random.normal(scale=0.05, size=200)  # alta correlação
    x3 = np.random.normal(size=200)
    target = (x1 + x3 + np.random.normal(size=200)) > 0
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target.astype(int)})

def test_vif_shape_and_variables():
    df = _make_vif_data()
    result = compute_vif(df, target_col="target", verbose="none")
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"variable", "vif"}
    assert set(result["variable"]) == {"x1", "x2", "x3"}

def test_vif_values_reasonable():
    df = _make_vif_data()
    result = compute_vif(df, target_col="target", verbose="none")
    assert result["vif"].max() > 5, "Esperado VIF alto por causa da correlação entre x1 e x2"
