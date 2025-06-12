from __future__ import annotations

import pandas as pd
from pathlib import Path

from vassoura.report import ReportManager, SECTION_REGISTRY
from vassoura.audit import AuditTrail


def make_manager(tmp_path: Path) -> ReportManager:
    at = AuditTrail()
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    at.take_snapshot(df, "raw")
    at.take_snapshot(df, "processed")

    metrics = pd.DataFrame({"test": [0.5]})
    rm = ReportManager()
    rm.add_section(
        SECTION_REGISTRY["overview"](
            audit=at,
            snapshot_names=["raw", "processed"],
            dataset_shape=df.shape,
        )
    )
    rm.add_section(SECTION_REGISTRY["performance"](metrics=metrics))
    rm.add_section(
        SECTION_REGISTRY["feature_importance"](importance=pd.Series({"a": 1.0}))
    )
    rm.add_section(
        SECTION_REGISTRY["audit_diff"](audit=at, base="raw", new="processed")
    )
    return rm


def test_render_creates_file(tmp_path: Path):
    rm = make_manager(tmp_path)
    out = rm.render(tmp_path / "report.html")
    assert out.exists()


def test_missing_optional_section(tmp_path: Path):
    at = AuditTrail()
    df = pd.DataFrame({"a": [1], "target": [0]})
    at.take_snapshot(df, "raw")
    metrics = pd.DataFrame({"test": [0.1]})
    rm = ReportManager()
    rm.add_section(
        SECTION_REGISTRY["overview"](
            audit=at,
            snapshot_names=["raw"],
            dataset_shape=df.shape,
        )
    )
    rm.add_section(SECTION_REGISTRY["performance"](metrics=metrics))
    out = rm.render(tmp_path / "report.html")
    assert out.exists()


def test_async_render_equivalence(tmp_path: Path):
    rm_sync = make_manager(tmp_path)
    rm_async = make_manager(tmp_path)
    rm_async.async_render = True
    out1 = rm_sync.render(tmp_path / "r1.html")
    out2 = rm_async.render(tmp_path / "r2.html")
    content1 = out1.read_text()
    content2 = out2.read_text()
    assert "Feature Importance" in content1
    assert "Feature Importance" in content2
    assert abs(len(content1) - len(content2)) < 500


def test_section_registry_population():
    for name in ["overview", "performance", "feature_importance", "audit_diff"]:
        assert name in SECTION_REGISTRY
