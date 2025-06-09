import pandas as pd

from vassoura.heuristics import graph_cut


def test_graph_cut_with_categorical_nan():
    df = pd.DataFrame(
        {
            "num": [0, 0, 1, 1, 1, 1],
            "cat": ["a", "a", None, "b", "b", None],
            "target": [0, 0, 1, 1, 1, 1],
        }
    )
    result = graph_cut(df, target_col="target", corr_threshold=0.9)
    removed = result["removed"]
    assert any(col in removed for col in ["num", "cat"])
