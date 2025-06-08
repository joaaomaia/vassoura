import numpy as np
import pandas as pd

from vassoura.leakage import target_leakage


def test_detect_exact_copy():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 100)
    df = pd.DataFrame({"target": target})
    df["copy"] = target
    result = target_leakage(df, target_col="target", threshold=0.8)
    assert "copy" in result["removed"]
    assert "copy" in result["artefacts"].index


def test_detect_partial_copy():
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, 100)
    df = pd.DataFrame({"target": target})
    df["leak"] = target.copy()
    idx = rng.choice(len(df), size=int(0.1 * len(df)), replace=False)
    df.loc[idx, "leak"] = 1 - df.loc[idx, "leak"]
    result = target_leakage(df, target_col="target", threshold=0.8)
    assert "leak" in result["removed"]


def test_extreme_cases_not_flagged():
    rng = np.random.default_rng(2)
    target = rng.integers(0, 2, 50)
    df = pd.DataFrame({"target": target, "const": 1, "all_na": np.nan})
    result = target_leakage(df, target_col="target", threshold=0.8)
    assert result["removed"] == []
