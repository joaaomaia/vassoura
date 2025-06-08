import pandas as pd
import numpy as np
from vassoura.core import Vassoura


def _make_df():
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(3), 4),
            "dt": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "a": np.random.normal(size=12),
        }
    )
    df["b"] = df["a"] * 0.9 + np.random.normal(scale=0.1, size=12)
    df["ignore"] = df["a"] * -1  # correlacionada mas para ignorar
    df["target"] = np.random.randint(0, 2, size=12)
    return df


def test_id_and_date_cols_preserved_and_sorted():
    df = _make_df()
    vs = Vassoura(
        df,
        target_col="target",
        heuristics=["corr"],
        params={"corr": 0.8},
        id_cols=["id"],
        date_cols=["dt"],
    )
    out = vs.run()
    # ids e datas devem permanecer e vir no inicio
    assert list(out.columns[:3]) == ["id", "dt", "target"]
    # deve estar ordenado por id e dt
    assert out["id"].is_monotonic_increasing
    assert out["dt"].is_monotonic_increasing


def test_ignore_cols_kept_when_drop_false():
    df = _make_df()
    vs = Vassoura(
        df,
        target_col="target",
        heuristics=["corr"],
        params={"corr": 0.8},
        ignore_cols=["ignore"],
        drop_ignored=False,
    )
    out = vs.run()
    assert "ignore" in out.columns


def test_ignore_cols_dropped_when_true():
    df = _make_df()
    vs = Vassoura(
        df,
        target_col="target",
        heuristics=["corr"],
        params={"corr": 0.8},
        ignore_cols=["ignore"],
        drop_ignored=True,
    )
    out = vs.run()
    assert "ignore" not in out.columns
