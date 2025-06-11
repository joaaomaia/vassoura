from __future__ import annotations

import pandas as pd
import pytest

from vassoura.audit import AuditTrail


@pytest.fixture
def df_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 2, 3],
            "b": ["x", "y", "y", "z"],
            "target": [0, 1, 0, 1],
        }
    )


def test_snapshot_roundtrip(df_sample):
    at = AuditTrail()
    at.take_snapshot(df_sample, "s1")
    at.describe_snapshot("s1")


def test_duplicate_detection(df_sample):
    at = AuditTrail(default_keys=["a", "b"])
    at.take_snapshot(df_sample, "s1")
    assert at.snapshots["s1"]["key_dupes"] > 0


def test_auto_detect_types_returns_lists(df_sample):
    at = AuditTrail(auto_detect_types=True)
    at.take_snapshot(df_sample, "s1")
    snap = at.snapshots["s1"]
    assert isinstance(snap["num_cols"], list)
    assert isinstance(snap["cat_cols"], list)


def test_compare_snapshots_shapes(df_sample):
    at = AuditTrail()
    at.take_snapshot(df_sample, "s1")
    df2 = df_sample.copy()
    df2["a"] += 1
    at.take_snapshot(df2, "s2")
    at.compare_snapshots("s1", "s2")


def test_psi_alert_logs(monkeypatch, df_sample, caplog):
    at = AuditTrail(
        track_histograms=True, track_distributions=True, enable_logging=True
    )

    def fake_psi(d1, d2):
        return 0.3

    from vassoura import audit as audit_module

    monkeypatch.setattr(audit_module, "calculate_psi", fake_psi)
    at.take_snapshot(df_sample, "s1")
    at.take_snapshot(df_sample, "s2")
    with caplog.at_level("INFO"):
        at.compare_snapshots("s1", "s2")
    assert any("PSI>0.2" in r.message for r in caplog.records)
