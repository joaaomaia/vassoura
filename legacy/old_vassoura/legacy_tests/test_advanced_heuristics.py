import numpy as np
import pandas as pd
from vassoura.core import Vassoura
from vassoura.heuristics import psi_stability, ks_separation


def test_psi_stability_removal():
    df = pd.DataFrame(
        {
            "date": ["2024-01"] * 50 + ["2025-01"] * 50,
            "x": np.r_[np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)],
            "y": np.r_[np.random.normal(0, 1, 50), np.random.normal(0, 1, 50)],
            "z": np.random.normal(size=100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    result = psi_stability(
        df, date_col="date", window=("2024-01", "2025-01"), psi_thr=0.25
    )
    assert "x" in result["removed"]


def test_ks_separation():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, 1000)
    weak = rng.normal(size=1000)
    strong = target + rng.normal(scale=0.1, size=1000)
    df = pd.DataFrame({"target": target, "weak": weak, "strong": strong})
    result = ks_separation(df, target_col="target", ks_thr=0.05)
    assert "weak" in result["removed"] and "strong" not in result["removed"]


def test_perm_importance_keep_cols():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
    df["target"] = (df["a"] + rng.normal(size=100)) > 0
    vs = Vassoura(
        df,
        target_col="target",
        heuristics=["perm_importance"],
        keep_cols=["e"],
        params={"perm_importance": 0.4},
    )
    out = vs.run()
    assert "e" in out.columns
    assert out.shape[1] <= 6


def test_partial_corr_cluster():
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=150)
    x2 = x1 * 0.9 + rng.normal(scale=0.1, size=150)
    x3 = rng.normal(size=150)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    vs = Vassoura(
        df, heuristics=["partial_corr_cluster"], params={"partial_corr_cluster": 0.6}
    )
    out = vs.run()
    assert ("x1" not in out.columns) or ("x2" not in out.columns)


def test_drift_vs_target_leakage():
    rng = np.random.default_rng(4)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    target = rng.integers(0, 2, 100)
    leak = np.arange(100) + target * 80
    safe = rng.normal(size=100)
    df = pd.DataFrame({"date": dates, "leak": leak, "safe": safe, "target": target})
    vs = Vassoura(
        df,
        target_col="target",
        date_cols=["date"],
        heuristics=["drift_leak"],
        params={"drift_leak_drift": 0.4, "drift_leak_leak": 0.5},
    )
    out = vs.run()
    assert "leak" not in out.columns and "safe" in out.columns
