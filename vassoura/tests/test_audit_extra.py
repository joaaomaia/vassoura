import pandas as pd
import pytest

from vassoura.audit import AuditTrail


def test_histograms_collected(tmp_path):
    df = pd.DataFrame({"c": ["a", "b", "a", "c"], "n": [1, 2, 3, 4]})
    at = AuditTrail(track_histograms=True)
    at.take_snapshot(df, "s")
    assert "histograms" in at.snapshots["s"]


def test_log_written(tmp_path):
    df = pd.DataFrame({"c": ["a", "b"], "target": [0, 1]})
    at = AuditTrail(enable_logging=True)
    at.take_snapshot(df, "s")
    at._log("msg")
    assert True


def test_describe_and_compare(capsys):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "target": [0, 1, 0]})
    at = AuditTrail()
    at.take_snapshot(df, "s1")
    df2 = df.copy()
    df2["a"] += 1
    at.take_snapshot(df2, "s2")
    at.describe_snapshot("s1")
    at.compare_snapshots("s1", "s2")
    at.list_snapshots()
    with pytest.raises(ValueError):
        at.take_snapshot(df, "s1")
    captured = capsys.readouterr()
    assert "Descrição" in captured.out
