import time
import pandas as pd

from vassoura.base_heuristic import BaseHeuristic, chunk_iter
from vassoura.core import Vassoura


class SlowHeuristic(BaseHeuristic):
    def run(
        self, df: pd.DataFrame, *, budget_sec: float | None = None, chunk_size: int = 1
    ) -> list[str]:
        start = time.perf_counter()
        drops: list[str] = []
        for cols in chunk_iter(df.columns, chunk_size):
            if cols[0] == "target":
                continue
            time.sleep(0.02)
            if budget_sec is not None and time.perf_counter() - start >= budget_sec:
                break
            drops.extend([c for c in cols if c != "target"])
        return drops


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "target": [0, 1]})


def test_happy_path():
    df = _base_df()
    h = SlowHeuristic()
    removed = h.run(df, budget_sec=1)
    assert set(removed) == {"a", "b", "c"}


def test_timeout_partial():
    df = _base_df()
    h = SlowHeuristic()
    removed = h.run(df, budget_sec=0.05, chunk_size=1)
    assert 0 < len(removed) < 3


def test_pipeline_applies_partial_result():
    df = _base_df()
    h = SlowHeuristic()
    vs = Vassoura(
        df,
        target_col="target",
        heuristics=["dummy"],
        process=["scaler"],
        timeout_map={"dummy": 0.05},
        verbose="none",
    )
    vs._heuristic_funcs["dummy"] = lambda: df.drop(
        columns=h.run(vs.df_current, budget_sec=0.05, chunk_size=1), inplace=True
    )
    result = vs.run()
    assert result.shape[1] <= 4
