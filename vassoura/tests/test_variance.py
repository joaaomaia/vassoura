import numpy as np
import pandas as pd
from vassoura.core import Vassoura


def test_variance_removes_constant():
    df = pd.DataFrame({"const": 1, "x": np.random.normal(size=100)})
    vs = Vassoura(df, heuristics=["variance"], thresholds={"variance": 1e-4})
    out = vs.run()
    assert "const" not in out.columns
    assert any("variance" in h["reason"] for h in vs.history)


def test_variance_keeps_variable():
    df = pd.DataFrame({"x": np.random.normal(size=100)})
    vs = Vassoura(df, heuristics=["variance"], thresholds={"variance": 1e-4})
    out = vs.run()
    assert "x" in out.columns


def test_dominant_category_removed():
    cat = ["A"] * 96 + ["B"] * 4
    df = pd.DataFrame({"cat": cat, "x": np.random.normal(size=100)})
    vs = Vassoura(
        df,
        heuristics=["variance"],
        thresholds={"variance": 1e-4, "variance_dom": 0.95},
    )
    out = vs.run()
    assert "cat" not in out.columns


def test_keep_cols_protected():
    df = pd.DataFrame({"const": 1, "x": np.random.normal(size=50)})
    vs = Vassoura(
        df,
        heuristics=["variance"],
        thresholds={"variance": 1e-4},
        keep_cols=["const"],
    )
    out = vs.run()
    assert "const" in out.columns

